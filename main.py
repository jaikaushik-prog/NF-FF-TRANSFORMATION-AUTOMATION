"""
MetrologyPipeline — Backend Integration Layer
==============================================
Chains the NF-FF antenna measurement modules into a single callable
pipeline with strict shape validation and progress callbacks for the
Streamlit dashboard.

Data flow
---------
    file_obj
      → ingestion.py       (parse raw VNA / CST data)
      → nf_autoencoder.py  (detect + interpolate anomalies)
      → nf_calibration.py  (LightGBM thermal-drift correction)
      → nf_ff_planar.py    (2-D FFT Plane-Wave Spectrum transform)
      → results dict        (far-field arrays + metrics)

Author : jai kaushik
"""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Union

import numpy as np
import pandas as pd
import torch

# ── Resolve import path ─────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src" / "python"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ingestion import VNADataIngester, IngestedData          # noqa: E402
from nf_autoencoder import (                                  # noqa: E402
    NearFieldAutoencoder,
    load_cst_simulation,
    detect_anomalies,
    smart_interpolate,
    generate_raw_sweep_with_anomalies,
    generate_cst_augmented_sweeps,
    train_model,
)
from nf_calibration import (                                  # noqa: E402
    CalibrationPipeline,
    SyntheticDriftDataGenerator,
    calibrate_sweep,
    FEATURE_COLS,
)
from nf_ff_planar import (                                    # noqa: E402
    PlanarNFFFTransformer,
    PlanarFFResult,
    SphericalFFResult,
)
from nf_ff_circular import (                                  # noqa: E402
    CircularNFFFTransformer,
    CircularFFResult,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════
# Pipeline stage enum
# ═══════════════════════════════════════════════════════════════════

class PipelineStage(Enum):
    """Identifiers for each processing stage."""
    INGESTION  = "Ingesting Data"
    ANOMALY    = "Scrubbing Anomalies"
    CALIBRATION = "Calibrating Phase"
    TRANSFORM  = "Computing NF-FF Transform"


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class PipelineConfig:
    """User-tunable parameters for the metrology pipeline."""
    freq_ghz: float = 10.0
    step_mm: float = 10.0
    n_fft: int = 256
    n_sweep_points: int = 360
    r_probe_m: float = 0.5
    autoencoder_weights: str = "models/nf_autoencoder.pth"
    calibration_model: str = "models/lightgbm_calibration_model.txt"
    cst_base_file: str = "data/raw/Simulated_NF_Data.txt"
    device: str = "cpu"
    autoencoder_epochs: int = 500
    autoencoder_batch: int = 64


# ═══════════════════════════════════════════════════════════════════
# Pipeline
# ═══════════════════════════════════════════════════════════════════

class MetrologyPipeline:
    """End-to-end NF-FF antenna measurement pipeline.

    Parameters
    ----------
    config : PipelineConfig
        Tunable parameters.  Defaults are suitable for 10 GHz horn.

    Examples
    --------
    >>> pipe = MetrologyPipeline()
    >>> results = pipe.run_full_sweep("data/raw/Simulated_NF_Data.txt")
    >>> print(results["metrics"])
    """

    def __init__(self, config: Optional[PipelineConfig] = None) -> None:
        self.config = config or PipelineConfig()
        self._ingester = VNADataIngester()

    # ── public API ───────────────────────────────────────────────

    def run_full_sweep(
        self,
        source: Union[str, Path, Any],
        progress_cb: Optional[Callable[[PipelineStage, float, str], None]] = None,
    ) -> Dict[str, Any]:
        """Execute the full four-stage processing pipeline.

        Parameters
        ----------
        source : str, Path, or file-like
            Raw measurement data file.
        progress_cb : callable, optional
            ``progress_cb(stage, fraction, message)`` called to update
            the front-end progress bar.

        Returns
        -------
        dict
            Keys: ``metrics``, ``nearfield``, ``anomaly``, ``calibration``,
            ``farfield``, ``stages_completed``, ``timings``.

        Raises
        ------
        ValueError
            On shape mismatches or missing data between stages.
        """
        results: Dict[str, Any] = {
            "metrics": {},
            "nearfield": None,
            "anomaly": None,
            "calibration": None,
            "farfield": None,
            "stages_completed": [],
            "timings": {},
        }

        def _progress(stage: PipelineStage, frac: float, msg: str = ""):
            if progress_cb:
                progress_cb(stage, frac, msg)

        # ── STAGE 1: INGESTION ──────────────────────────────────
        try:
            _progress(PipelineStage.INGESTION, 0.0, "Parsing file …")
            t0 = time.perf_counter()

            data: IngestedData = self._ingester.parse(source)
            results["nearfield"] = data
            results["metrics"]["input_format"] = data.format_detected
            results["metrics"]["input_rows"] = len(data.df)
            if data.grid_shape:
                results["metrics"]["grid_shape"] = (
                    f"{data.grid_shape[1]}×{data.grid_shape[0]}"
                )

            results["timings"]["ingestion"] = time.perf_counter() - t0
            results["stages_completed"].append("ingestion")
            _progress(PipelineStage.INGESTION, 1.0, "Done")

        except Exception as exc:
            raise ValueError(f"[Ingestion] {exc}") from exc

        # ── STAGE 2: ANOMALY DETECTION ──────────────────────────
        try:
            _progress(PipelineStage.ANOMALY, 0.0, "Loading autoencoder …")
            t0 = time.perf_counter()

            anom = self._run_anomaly_detection(data, _progress)
            results["anomaly"] = anom
            results["metrics"]["anomalies_detected"] = int(anom["n_anomalies"])
            results["metrics"]["anomaly_threshold_dB"] = round(
                float(anom["threshold"]), 3
            )

            results["timings"]["anomaly"] = time.perf_counter() - t0
            results["stages_completed"].append("anomaly")
            _progress(PipelineStage.ANOMALY, 1.0, "Done")

        except Exception as exc:
            raise ValueError(
                f"[Anomaly Detection] {exc}\n"
                f"Input shape: magnitudes={getattr(data, 'magnitudes', 'N/A')}"
            ) from exc

        # ── STAGE 3: PHASE CALIBRATION ──────────────────────────
        try:
            _progress(PipelineStage.CALIBRATION, 0.0, "Loading calibrator …")
            t0 = time.perf_counter()

            cal = self._run_calibration(data, anom, _progress)
            results["calibration"] = cal
            results["metrics"]["phase_drift_corrected_deg"] = round(
                float(cal["mean_drift_deg"]), 3
            )
            results["metrics"]["calibration_mode"] = cal["mode"]

            results["timings"]["calibration"] = time.perf_counter() - t0
            results["stages_completed"].append("calibration")
            _progress(PipelineStage.CALIBRATION, 1.0, "Done")

        except Exception as exc:
            raise ValueError(f"[Calibration] {exc}") from exc

        # ── STAGE 4: NF → FF TRANSFORM ─────────────────────────
        try:
            _progress(PipelineStage.TRANSFORM, 0.0, "Building PWS …")
            t0 = time.perf_counter()

            ff = self._run_transform(data, _progress)
            results["farfield"] = ff
            results["metrics"]["peak_gain_dB"] = round(
                float(ff.get("peak_dB", 0.0)), 2
            )
            results["metrics"]["fft_size"] = self.config.n_fft

            results["timings"]["transform"] = time.perf_counter() - t0
            results["stages_completed"].append("transform")
            _progress(PipelineStage.TRANSFORM, 1.0, "Done")

        except Exception as exc:
            raise ValueError(
                f"[NF-FF Transform] {exc}\n"
                f"Grid shape: {getattr(data, 'grid_shape', 'N/A')}"
            ) from exc

        return results

    # ── STAGE 2 implementation ───────────────────────────────────

    def _run_anomaly_detection(self, data: IngestedData, _progress) -> dict:
        """Load the autoencoder and detect anomalies in the sweep."""
        cfg = self.config

        # Build model
        model = NearFieldAutoencoder()
        weights_path = Path(ROOT_DIR / cfg.autoencoder_weights)

        if weights_path.is_file():
            model.load_state_dict(
                torch.load(str(weights_path), map_location=cfg.device,
                           weights_only=True)
            )
            logger.info("Autoencoder weights loaded <- %s", weights_path)
            trained_now = False
        else:
            # Train on the CST base if weights don't exist
            _progress(PipelineStage.ANOMALY, 0.1, "Training autoencoder …")
            cst_path = Path(ROOT_DIR / cfg.cst_base_file)
            if not cst_path.is_file():
                raise FileNotFoundError(
                    f"No weights and no CST base file at {cst_path}"
                )
            clean_base = load_cst_simulation(str(cst_path), cfg.n_sweep_points)
            train_model(
                model, clean_base,
                epochs=cfg.autoencoder_epochs,
                batch_size=cfg.autoencoder_batch,
                save_path=str(weights_path),
                device=cfg.device,
            )
            trained_now = True

        model.to(cfg.device).eval()

        # Prepare the 1-D magnitude sweep for anomaly detection
        if data.magnitudes is not None and len(data.magnitudes) > 0:
            sweep = data.magnitudes.copy()
        else:
            raise ValueError(
                "No magnitude data available for anomaly detection. "
                f"Ingested format: {data.format_detected}"
            )

        # Validate shape
        if sweep.ndim != 1:
            raise ValueError(
                f"Expected 1-D magnitude sweep, got shape {sweep.shape}"
            )

        # Resample to n_sweep_points if needed
        if len(sweep) != cfg.n_sweep_points:
            orig_angles = np.linspace(0, 359, len(sweep))
            target_angles = np.linspace(0, 359, cfg.n_sweep_points)
            sweep = np.interp(target_angles, orig_angles, sweep).astype(
                np.float32
            )

        _progress(PipelineStage.ANOMALY, 0.5, "Running inference …")

        anom_idx, errors, reconstructed, threshold = detect_anomalies(
            model, sweep, device=cfg.device
        )

        # Smart-interpolate the flagged points
        angles = np.arange(len(sweep), dtype=np.float64)
        phases = (
            data.phases.copy()
            if data.phases is not None
            else np.zeros(len(sweep), dtype=np.float32)
        )
        if len(phases) != len(sweep):
            phases = np.interp(
                np.linspace(0, 359, len(sweep)),
                np.linspace(0, 359, len(phases)),
                phases,
            ).astype(np.float32)

        mag_clean, pha_clean = smart_interpolate(
            angles, sweep, phases, anom_idx
        )

        return {
            "n_anomalies": len(anom_idx),
            "anomaly_indices": anom_idx,
            "errors": errors,
            "threshold": threshold,
            "reconstructed": reconstructed,
            "mag_raw": sweep,
            "mag_clean": mag_clean,
            "phase_clean": pha_clean,
            "angles": angles,
            "trained_now": trained_now,
        }

    # ── STAGE 3 implementation ───────────────────────────────────

    def _run_calibration(self, data: IngestedData, anom: dict, _progress) -> dict:
        """Run LightGBM thermal-drift calibration."""
        import lightgbm as lgb

        cfg = self.config
        model_path = Path(ROOT_DIR / cfg.calibration_model)

        # Check if we have real calibration features
        has_features = data.has_temperature and data.has_elapsed_time

        if has_features and model_path.is_file():
            # Full calibration with real data
            _progress(PipelineStage.CALIBRATION, 0.3, "Predicting drift …")
            pipeline = CalibrationPipeline(model_path=str(model_path))
            pipeline.load(str(model_path))

            cal_df = calibrate_sweep(
                data.df, pipeline,
                output_path="data/processed/calibrated_sweep.csv",
            )
            mean_drift = float(cal_df["predicted_drift_deg"].abs().mean())
            mode = "full"
        else:
            # Synthetic demo calibration
            _progress(PipelineStage.CALIBRATION, 0.2,
                      "No temperature data — running synthetic demo …")

            n_pts = len(anom["angles"])
            gen = SyntheticDriftDataGenerator(n_samples=n_pts, seed=42)
            synth_df = gen.generate()

            _progress(PipelineStage.CALIBRATION, 0.4, "Calibrating …")

            # Use the Booster directly to avoid sklearn wrapper issues
            # with _n_features validation on models loaded from file
            demo_df = synth_df.head(n_pts).copy()
            X_demo = demo_df[FEATURE_COLS].values

            if model_path.is_file():
                booster = lgb.Booster(model_file=str(model_path))
                predicted_drift = booster.predict(X_demo)
            else:
                # Train a fresh model
                _progress(PipelineStage.CALIBRATION, 0.3,
                          "Training calibration model …")
                pipeline = CalibrationPipeline(model_path=str(model_path))
                big_gen = SyntheticDriftDataGenerator(
                    n_samples=10000, seed=42
                )
                pipeline.train(big_gen.generate())
                predicted_drift = pipeline.predict(X_demo)

            mean_drift = float(np.abs(predicted_drift).mean())
            mode = "synthetic_demo"

        return {
            "mean_drift_deg": mean_drift,
            "mode": mode,
            "phase_clean": anom["phase_clean"],
        }

    # ── STAGE 4 implementation ───────────────────────────────────

    def _run_transform(self, data: IngestedData, _progress) -> dict:
        """Run the appropriate NF-FF transform based on data geometry.

        - 2D planar grid → Planar PWS (2D-FFT) transform
        - 1D angular sweep → Circular CME (Hankel) transform
        """
        cfg = self.config
        freq_hz = cfg.freq_ghz * 1.0e9
        is_planar = data.grid_shape is not None and data.Ex_2D is not None

        if is_planar:
            return self._run_planar_transform(data, freq_hz, _progress)
        else:
            return self._run_circular_transform(data, freq_hz, _progress)

    def _run_planar_transform(self, data, freq_hz, _progress) -> dict:
        """2D Planar PWS transform for spatial grid data."""
        cfg = self.config

        transformer = PlanarNFFFTransformer(
            freq_hz=freq_hz,
            dx_mm=cfg.step_mm,
            dy_mm=cfg.step_mm,
            n_fft=cfg.n_fft,
        )

        _progress(PipelineStage.TRANSFORM, 0.2, "Loading near-field grid …")

        if data.Ex_2D is not None and data.Ey_2D is not None:
            transformer.load_arrays(
                data.Ex_2D, data.Ey_2D, data.x_mm, data.y_mm
            )
        else:
            cst_path = Path(ROOT_DIR / cfg.cst_base_file)
            transformer.load_cst_data(str(cst_path))

        _progress(PipelineStage.TRANSFORM, 0.5, "Computing 2-D FFT …")
        result: PlanarFFResult = transformer.transform()

        _progress(PipelineStage.TRANSFORM, 0.8, "Converting to spherical …")
        sph: SphericalFFResult = transformer.to_spherical()

        visible = result.visible_region
        peak_dB = float(np.max(result.farfield_Ey_dB[visible]))

        return {
            "transform_mode": "planar",
            "U_grid": result.U_grid,
            "V_grid": result.V_grid,
            "farfield_Ey_dB": result.farfield_Ey_dB,
            "farfield_Ex_dB": result.farfield_Ex_dB,
            "visible_region": visible,
            "u": result.u,
            "v": result.v,
            "theta": sph.theta,
            "phi": sph.phi,
            "E_theta_dB": sph.E_theta_dB,
            "E_phi_dB": sph.E_phi_dB,
            "peak_dB": peak_dB,
            "n_fft": cfg.n_fft,
        }

    def _run_circular_transform(self, data, freq_hz, _progress) -> dict:
        """1D Circular CME transform for angular sweep data."""
        cfg = self.config

        _progress(PipelineStage.TRANSFORM, 0.2,
                  "Using circular (CME + Hankel) transform …")

        transformer = CircularNFFFTransformer(
            freq_hz=freq_hz,
            r_probe_m=cfg.r_probe_m,
        )

        angles = data.angles if data.angles is not None else np.arange(360.0)
        mags = (
            data.magnitudes if data.magnitudes is not None
            else np.zeros(len(angles), dtype=np.float32)
        )
        phases = (
            data.phases if data.phases is not None
            else np.zeros(len(angles), dtype=np.float32)
        )

        transformer.load_arrays(angles, mags, phases)

        _progress(PipelineStage.TRANSFORM, 0.5,
                  "Computing cylindrical-mode expansion …")
        result: CircularFFResult = transformer.transform()

        peak_dB = float(np.max(result.ff_mag_dB))

        return {
            "transform_mode": "circular",
            "angles_deg": result.angles_deg,
            "nf_mag_dB": result.nf_mag_dB,
            "ff_mag_dB": result.ff_mag_dB,
            "modes_NF": result.modes_NF,
            "modes_FF": result.modes_FF,
            "mode_indices": result.mode_indices,
            "max_mode": result.max_mode,
            "peak_dB": peak_dB,
        }


# ═══════════════════════════════════════════════════════════════════
# Quick self-test
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    pipe = MetrologyPipeline()
    cst_file = ROOT_DIR / "data" / "raw" / "Simulated_NF_Data.txt"

    if cst_file.is_file():
        print("\n  Running full pipeline on CST demo data …\n")
        results = pipe.run_full_sweep(str(cst_file))
        print("\n  ── Pipeline Results ──")
        for k, v in results["metrics"].items():
            print(f"    {k:30s} : {v}")
        print(f"\n  Stages completed: {results['stages_completed']}")
        for k, v in results["timings"].items():
            print(f"    {k:20s} : {v:.3f} s")
    else:
        print(f"  CST file not found: {cst_file}")
