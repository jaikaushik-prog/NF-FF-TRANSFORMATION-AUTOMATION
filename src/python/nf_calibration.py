"""
Near-Field Antenna Measurement -- LightGBM Phase Drift Calibration
===================================================================
Production-ready gradient-boosted regression pipeline for predicting
and correcting thermal phase drift in 10 GHz anechoic chamber scans.

During long measurement sweeps (up to 4 hours), temperature fluctuations
cause slow baseline drift in the VNA and RF cables, corrupting S21 phase.
This module trains a LightGBM regressor on physics-informed synthetic data
to predict the exact phase drift at every measurement point, then subtracts
it to recover the true antenna phase pattern.

Usage
-----
    python nf_calibration.py --mode train                  # synthetic train + demo
    python nf_calibration.py --mode calibrate --input raw.csv  # calibrate a file
    python nf_calibration.py --mode train --samples 20000  # more training data

Author : NF Autoencoder Pipeline
Date   : 2026-06-03
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

import lightgbm as lgb
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

# ---------------------------------------------------------------------------
# Global configuration
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s | %(levelname)-8s | %(message)s"
LOG_DATE = "%Y-%m-%d %H:%M:%S"

logger = logging.getLogger("nf_calibration")


def _setup_logging(level: int = logging.INFO) -> None:
    """Configure root logger with timestamped format."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FMT, datefmt=LOG_DATE))
    logger.setLevel(level)
    logger.addHandler(handler)


# ===================================================================
# 1.  SYNTHETIC DATA GENERATOR
# ===================================================================

@dataclass
class SyntheticDriftDataGenerator:
    """Physics-informed simulator for VNA thermal phase drift.

    Generates realistic training data mimicking a 4-hour anechoic chamber
    scan at 10 GHz.  The drift model is non-linear and couples temperature
    with elapsed time, matching real-world cable-flex and VNA LO drift
    behaviour.

    Parameters
    ----------
    n_samples : int
        Number of measurement points to generate.
    temp_start : float
        Chamber temperature at the beginning of the scan (deg C).
    temp_end : float
        Chamber temperature at the end of the scan (deg C).
    scan_duration_min : float
        Total scan duration in minutes.
    noise_std_phase : float
        Standard deviation of additive white Gaussian phase noise (deg).
    noise_std_mag : float
        Standard deviation of additive magnitude noise (dB).
    seed : int
        Random seed for reproducibility.
    """

    n_samples: int = 10000
    temp_start: float = 22.0
    temp_end: float = 26.0
    scan_duration_min: float = 240.0
    noise_std_phase: float = 1.5
    noise_std_mag: float = 0.4
    seed: int = 42

    # internal state -------------------------------------------------------
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)

    # -- helpers -----------------------------------------------------------

    def _base_phase_profile(self, angles_deg: np.ndarray) -> np.ndarray:
        """Simulate a broadside horn antenna phase profile.

        A horn antenna exhibits a roughly quadratic phase taper across the
        aperture.  In the far-field angular domain this translates to a
        smooth, symmetric phase front centred at boresight (0 deg).

        Parameters
        ----------
        angles_deg : np.ndarray
            Observation angles in degrees (-180 to +180).

        Returns
        -------
        np.ndarray
            Base (clean) phase values in degrees.
        """
        angles_rad = np.deg2rad(angles_deg)
        # Quadratic phase taper (typical horn aperture distribution)
        phase = -120.0 * np.sin(angles_rad) ** 2
        # Add a small cubic asymmetry (feed offset / cross-pol coupling)
        phase += 8.0 * np.sin(angles_rad) ** 3
        return phase

    def _base_magnitude_profile(self, angles_deg: np.ndarray) -> np.ndarray:
        """Simulate a horn antenna magnitude profile (dB).

        Parameters
        ----------
        angles_deg : np.ndarray
            Observation angles in degrees (-180 to +180).

        Returns
        -------
        np.ndarray
            Magnitude values in dB with main lobe peak at 0 dB.
        """
        angles_rad = np.deg2rad(angles_deg)
        # sinc-like main lobe with side-lobes
        u = 5.0 * np.sin(angles_rad)
        with np.errstate(divide="ignore", invalid="ignore"):
            pattern = np.where(
                np.abs(u) < 1e-12,
                1.0,
                np.sin(np.pi * u) / (np.pi * u),
            )
        mag_db = 20.0 * np.log10(np.abs(pattern) + 1e-12)
        # Clamp floor
        mag_db = np.clip(mag_db, -50.0, 0.0)
        return mag_db

    def _thermal_drift(
        self,
        temperature: np.ndarray,
        elapsed_min: np.ndarray,
    ) -> np.ndarray:
        """Compute the non-linear thermal phase drift (ground truth).

        The model is:
            drift = 0.5 * (T - T_ref)^1.5  +  0.02 * t

        where T_ref = temp_start.  This captures:
          - The dominant non-linear cable-dielectric effect (T^1.5 term)
          - A slow monotonic VNA LO drift (linear in time)

        Parameters
        ----------
        temperature : np.ndarray
            Instantaneous chamber temperature (deg C).
        elapsed_min : np.ndarray
            Time since scan start (minutes).

        Returns
        -------
        np.ndarray
            Phase drift in degrees.
        """
        delta_t = np.clip(temperature - self.temp_start, 0.0, None)
        drift = 0.5 * np.power(delta_t, 1.5) + 0.02 * elapsed_min
        return drift

    # -- public API --------------------------------------------------------

    def generate(self) -> pd.DataFrame:
        """Generate a complete synthetic measurement dataset.

        Returns
        -------
        pd.DataFrame
            Columns: motor_angle_deg, ambient_temp_c, elapsed_time_min,
                     raw_phase_deg, raw_mag_dB, phase_drift_error_deg,
                     true_phase_deg, true_mag_dB
        """
        logger.info(
            "Generating %d synthetic samples  (%.0f min scan, %.1f-%.1f C)",
            self.n_samples,
            self.scan_duration_min,
            self.temp_start,
            self.temp_end,
        )
        t0 = time.perf_counter()

        # --- Independent variables ----------------------------------------
        # Motor sweeps continuously from -180 to +180 multiple times
        elapsed = np.linspace(0.0, self.scan_duration_min, self.n_samples)
        n_sweeps = max(1, self.n_samples // 360)
        angles = np.tile(np.linspace(-180.0, 179.0, 360), n_sweeps + 1)[
            : self.n_samples
        ]

        # Temperature: slow sinusoidal fluctuation riding on a linear ramp
        temp_ramp = np.linspace(self.temp_start, self.temp_end, self.n_samples)
        temp_fluct = 0.3 * np.sin(2 * np.pi * elapsed / 60.0)  # 1-hour period
        temperature = temp_ramp + temp_fluct

        # --- Base antenna patterns ----------------------------------------
        true_phase = self._base_phase_profile(angles)
        true_mag = self._base_magnitude_profile(angles)

        # --- Inject drift + noise -----------------------------------------
        drift = self._thermal_drift(temperature, elapsed)
        phase_noise = self._rng.normal(0.0, self.noise_std_phase, self.n_samples)
        mag_noise = self._rng.normal(0.0, self.noise_std_mag, self.n_samples)

        raw_phase = true_phase + drift + phase_noise
        raw_mag = true_mag + mag_noise

        # --- Assemble DataFrame -------------------------------------------
        df = pd.DataFrame(
            {
                "motor_angle_deg": angles,
                "ambient_temp_c": np.round(temperature, 3),
                "elapsed_time_min": np.round(elapsed, 3),
                "raw_phase_deg": np.round(raw_phase, 4),
                "raw_mag_dB": np.round(raw_mag, 4),
                "phase_drift_error_deg": np.round(drift, 4),
                "true_phase_deg": np.round(true_phase, 4),
                "true_mag_dB": np.round(true_mag, 4),
            }
        )

        dt = time.perf_counter() - t0
        logger.info(
            "  [OK] Dataset generated in %.2f s  "
            "(drift range: %.2f - %.2f deg)",
            dt,
            drift.min(),
            drift.max(),
        )
        return df


# ===================================================================
# 2.  CALIBRATION PIPELINE
# ===================================================================

FEATURE_COLS = [
    "motor_angle_deg",
    "ambient_temp_c",
    "elapsed_time_min",
    "raw_phase_deg",
]
TARGET_COL = "phase_drift_error_deg"


@dataclass
class CalibrationPipeline:
    """LightGBM gradient-boosted regression pipeline for phase drift.

    Trains a fast, CPU-optimised regressor to predict the thermal phase
    drift from readily-available measurement metadata (angle, temperature,
    elapsed time, raw phase).

    Parameters
    ----------
    n_folds : int
        Number of cross-validation folds.
    lgb_params : dict | None
        Override LightGBM hyperparameters.  If None, tuned defaults
        are used.
    model_path : str
        File path for saving / loading the trained model.
    """

    n_folds: int = 5
    lgb_params: Optional[dict] = None
    model_path: str = "models/lightgbm_calibration_model.txt"

    # internal state -------------------------------------------------------
    _model: Optional[lgb.LGBMRegressor] = field(
        init=False, default=None, repr=False
    )

    def __post_init__(self) -> None:
        if self.lgb_params is None:
            self.lgb_params = {
                "n_estimators": 800,
                "max_depth": 7,
                "learning_rate": 0.05,
                "num_leaves": 63,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "min_child_samples": 20,
                "reg_alpha": 0.1,
                "reg_lambda": 1.0,
                "random_state": 42,
                "verbose": -1,
                "n_jobs": -1,
            }

    # -- training ----------------------------------------------------------

    def train(self, df: pd.DataFrame) -> pd.DataFrame:
        """Train the LightGBM model with K-Fold cross-validation.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain FEATURE_COLS and TARGET_COL.

        Returns
        -------
        pd.DataFrame
            Per-fold metrics (MAE, RMSE, R2).
        """
        X = df[FEATURE_COLS].values
        y = df[TARGET_COL].values

        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=42)
        fold_metrics: list[dict] = []

        logger.info(
            "=" * 62 + "\n"
            "  Training LightGBM Calibration Model\n"
            "  Folds: %d  |  Features: %s\n"
            "  Estimators: %d  |  LR: %.3f  |  Depth: %d\n"
            + "=" * 62,
            self.n_folds,
            ", ".join(FEATURE_COLS),
            self.lgb_params["n_estimators"],
            self.lgb_params["learning_rate"],
            self.lgb_params["max_depth"],
        )

        t0 = time.perf_counter()

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X), start=1):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr, y_val = y[train_idx], y[val_idx]

            model = lgb.LGBMRegressor(**self.lgb_params)
            model.fit(
                X_tr,
                y_tr,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.log_evaluation(period=0)],  # suppress per-iter
            )

            y_pred = model.predict(X_val)
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            r2 = r2_score(y_val, y_pred)

            fold_metrics.append(
                {"fold": fold_idx, "MAE_deg": mae, "RMSE_deg": rmse, "R2": r2}
            )
            logger.info(
                "  Fold %d/%d  |  MAE: %.4f deg  |  RMSE: %.4f deg  |  R2: %.6f",
                fold_idx,
                self.n_folds,
                mae,
                rmse,
                r2,
            )

        # Final model: retrain on ALL data ---------------------------------
        self._model = lgb.LGBMRegressor(**self.lgb_params)
        self._model.fit(X, y)

        dt = time.perf_counter() - t0
        metrics_df = pd.DataFrame(fold_metrics)

        logger.info("")
        logger.info(
            "  MEAN  |  MAE: %.4f deg  |  RMSE: %.4f deg  |  R2: %.6f",
            metrics_df["MAE_deg"].mean(),
            metrics_df["RMSE_deg"].mean(),
            metrics_df["R2"].mean(),
        )
        logger.info("  Training completed in %.2f s", dt)

        # Save model -------------------------------------------------------
        self._model.booster_.save_model(self.model_path)
        logger.info("  [OK] Model saved -> %s", self.model_path)

        return metrics_df

    # -- loading -----------------------------------------------------------

    def load(self, path: Optional[str] = None) -> None:
        """Load a previously trained model from disk.

        Parameters
        ----------
        path : str | None
            Path to the LightGBM model text file.
        """
        load_path = path or self.model_path
        booster = lgb.Booster(model_file=load_path)
        self._model = lgb.LGBMRegressor(**self.lgb_params)
        self._model._Booster = booster
        self._model.fitted_ = True
        self._model._n_features = len(FEATURE_COLS)
        logger.info("  [OK] Model loaded <- %s", load_path)

    # -- prediction --------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict phase drift for a feature matrix.

        Parameters
        ----------
        X : np.ndarray
            Shape (n_samples, 4) matching FEATURE_COLS.

        Returns
        -------
        np.ndarray
            Predicted phase drift in degrees.

        Raises
        ------
        RuntimeError
            If the model has not been trained or loaded.
        """
        if self._model is None:
            raise RuntimeError(
                "Model not initialised. Call .train() or .load() first."
            )
        return self._model.predict(X)

    @property
    def feature_importances(self) -> Optional[np.ndarray]:
        """Return feature importance array (gain-based)."""
        if self._model is None:
            return None
        return self._model.feature_importances_


# ===================================================================
# 3.  CALIBRATE SWEEP FUNCTION
# ===================================================================


def calibrate_sweep(
    df: pd.DataFrame,
    pipeline: CalibrationPipeline,
    output_path: str = "data/processed/calibrated_sweep.csv",
) -> pd.DataFrame:
    """Apply drift correction to a raw measurement DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain FEATURE_COLS.  Typically loaded from a raw CSV
        with columns: motor_angle_deg, ambient_temp_c, elapsed_time_min,
        raw_phase_deg, (optionally raw_mag_dB).
    pipeline : CalibrationPipeline
        A trained pipeline instance.
    output_path : str
        Path for the corrected CSV output.

    Returns
    -------
    pd.DataFrame
        Input DataFrame augmented with 'predicted_drift_deg' and
        'calibrated_phase_deg' columns.
    """
    logger.info("  Calibrating %d measurement points ...", len(df))

    X = df[FEATURE_COLS].values
    predicted_drift = pipeline.predict(X)

    df = df.copy()
    df["predicted_drift_deg"] = np.round(predicted_drift, 4)

    # Subtract predicted drift and wrap to [-180, +180)
    corrected = df["raw_phase_deg"] - predicted_drift
    corrected = (corrected + 180.0) % 360.0 - 180.0
    df["calibrated_phase_deg"] = np.round(corrected, 4)

    # Export
    df.to_csv(output_path, index=False)
    logger.info("  [OK] Calibrated data saved -> %s", output_path)

    return df


# ===================================================================
# 4.  VISUALISATION
# ===================================================================


def plot_diagnostics(
    df: pd.DataFrame,
    save_path: str = "results/calibration_diagnostic.png",
) -> None:
    """Generate and save a 2-panel diagnostic figure.

    Panel 1: Colour-mapped scatter showing drift vs temperature & time.
    Panel 2: Phase pattern comparison (true vs corrupted vs calibrated).

    Parameters
    ----------
    df : pd.DataFrame
        Must contain: ambient_temp_c, elapsed_time_min,
        phase_drift_error_deg, motor_angle_deg, true_phase_deg,
        raw_phase_deg, calibrated_phase_deg.
    save_path : str
        Output image path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle(
        "LightGBM Phase Drift Calibration -- Diagnostic Report",
        fontsize=15,
        fontweight="bold",
        y=1.01,
    )

    # --- Panel 1: Drift landscape -----------------------------------------
    ax1 = axes[0]
    sc = ax1.scatter(
        df["elapsed_time_min"],
        df["ambient_temp_c"],
        c=df["phase_drift_error_deg"],
        cmap="inferno",
        s=3,
        alpha=0.7,
        edgecolors="none",
    )
    cbar = fig.colorbar(sc, ax=ax1, pad=0.02)
    cbar.set_label("Phase Drift Error (deg)", fontsize=11)
    ax1.set_xlabel("Elapsed Time (min)", fontsize=12)
    ax1.set_ylabel("Ambient Temperature (C)", fontsize=12)
    ax1.set_title("Thermal Phase Drift Landscape", fontsize=13)
    ax1.grid(True, alpha=0.3)

    # --- Panel 2: Before vs After -----------------------------------------
    ax2 = axes[1]

    # Take the LAST full sweep (last 360 points) for a clean comparison
    last_sweep = df.tail(360).copy().sort_values("motor_angle_deg")
    angles = last_sweep["motor_angle_deg"].values

    if "true_phase_deg" in last_sweep.columns:
        ax2.plot(
            angles,
            last_sweep["true_phase_deg"],
            "g-",
            linewidth=2.0,
            label="True (clean) phase",
            zorder=3,
        )

    ax2.plot(
        angles,
        last_sweep["raw_phase_deg"],
        "r-",
        linewidth=1.0,
        alpha=0.6,
        label="Raw (corrupted) phase",
    )

    if "calibrated_phase_deg" in last_sweep.columns:
        ax2.plot(
            angles,
            last_sweep["calibrated_phase_deg"],
            "b--",
            linewidth=1.8,
            label="LightGBM calibrated phase",
            zorder=2,
        )

    ax2.set_xlabel("Motor Angle (deg)", fontsize=12)
    ax2.set_ylabel("Phase (deg)", fontsize=12)
    ax2.set_title("Phase Pattern: Before vs After Calibration", fontsize=13)
    ax2.legend(loc="lower left", fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(-180, 180)

    plt.tight_layout()
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    logger.info("  [OK] Diagnostic plot saved -> %s", save_path)
    plt.show()


# ===================================================================
# 5.  CLI ENTRYPOINT
# ===================================================================


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "LightGBM Phase Drift Calibration for Near-Field "
            "Antenna Measurements (10 GHz)"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python nf_calibration.py --mode train\n"
            "  python nf_calibration.py --mode train --samples 20000\n"
            "  python nf_calibration.py --mode calibrate --input raw.csv\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["train", "calibrate"],
        default="train",
        help="'train' = synthetic training + demo; "
        "'calibrate' = inference on external file.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to raw measurement CSV (required for --mode calibrate).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/lightgbm_calibration_model.txt",
        help="Path to LightGBM model file (default: lightgbm_calibration_model.txt).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Number of synthetic training samples (default: 10000).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/calibrated_sweep.csv",
        help="Output CSV path for calibrated data.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the matplotlib plot window.",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the calibration pipeline."""
    _setup_logging()
    args = _parse_args()

    if args.no_plot:
        matplotlib.use("Agg")

    pipeline = CalibrationPipeline(model_path=args.model)

    # ==================================================================
    # MODE: TRAIN
    # ==================================================================
    if args.mode == "train":
        logger.info("")
        logger.info("  Mode: TRAIN  (synthetic data generation + training)")
        logger.info("")

        # 1. Generate synthetic data
        generator = SyntheticDriftDataGenerator(n_samples=args.samples)
        df = generator.generate()

        # 2. Train with cross-validation
        metrics_df = pipeline.train(df)

        # 3. Print feature importances
        importances = pipeline.feature_importances
        if importances is not None:
            logger.info("")
            logger.info("  Feature Importances (gain-based):")
            for feat, imp in zip(FEATURE_COLS, importances):
                bar = "#" * int(imp / max(importances) * 30)
                logger.info("    %-22s  %6d  %s", feat, imp, bar)

        # 4. Demo: calibrate the training data itself
        logger.info("")
        logger.info("  >> Demo: calibrating synthetic data ...")
        df = calibrate_sweep(df, pipeline, output_path=args.output)

        # 5. Compute residual stats
        if "true_phase_deg" in df.columns:
            residual_before = df["raw_phase_deg"] - df["true_phase_deg"]
            residual_after = df["calibrated_phase_deg"] - df["true_phase_deg"]
            # Wrap residuals to [-180, 180]
            residual_after = (residual_after + 180.0) % 360.0 - 180.0

            logger.info("")
            logger.info("  ----------------------------------------")
            logger.info(
                "  Phase error BEFORE calibration:  "
                "MAE = %.3f deg  |  RMSE = %.3f deg",
                np.mean(np.abs(residual_before)),
                np.sqrt(np.mean(residual_before**2)),
            )
            logger.info(
                "  Phase error AFTER  calibration:  "
                "MAE = %.3f deg  |  RMSE = %.3f deg",
                np.mean(np.abs(residual_after)),
                np.sqrt(np.mean(residual_after**2)),
            )
            improvement = 1.0 - (
                np.mean(np.abs(residual_after))
                / np.mean(np.abs(residual_before))
            )
            logger.info(
                "  Improvement: %.1f%% reduction in phase error",
                improvement * 100,
            )
            logger.info("  ----------------------------------------")

        # 6. Plot diagnostics
        plot_diagnostics(df)

    # ==================================================================
    # MODE: CALIBRATE
    # ==================================================================
    elif args.mode == "calibrate":
        if args.input is None:
            logger.error(
                "  --input is required for calibrate mode. "
                "Usage: --mode calibrate --input raw.csv"
            )
            sys.exit(1)

        input_path = Path(args.input)
        if not input_path.exists():
            logger.error("  Input file not found: %s", input_path)
            sys.exit(1)

        logger.info("")
        logger.info("  Mode: CALIBRATE  (inference on %s)", input_path.name)
        logger.info("")

        # 1. Load model
        model_path = Path(args.model)
        if not model_path.exists():
            logger.error(
                "  Model file not found: %s  "
                "(run --mode train first)",
                model_path,
            )
            sys.exit(1)

        pipeline.load(str(model_path))

        # 2. Load raw data
        df = pd.read_csv(input_path)
        logger.info("  Loaded %d rows from %s", len(df), input_path.name)

        # Validate columns
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            logger.error(
                "  Missing required columns: %s\n"
                "  Expected: %s",
                missing,
                FEATURE_COLS,
            )
            sys.exit(1)

        # 3. Calibrate
        df = calibrate_sweep(df, pipeline, output_path=args.output)

        # 4. Plot if ground truth is available
        if "true_phase_deg" in df.columns:
            plot_diagnostics(df)
            logger.info("  [OK] Diagnostic plot generated.")
        else:
            logger.info(
                "  (No 'true_phase_deg' column -- skipping comparison plot)"
            )

    logger.info("")
    logger.info("  Done.")


if __name__ == "__main__":
    main()
