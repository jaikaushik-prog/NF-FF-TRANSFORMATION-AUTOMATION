#!/usr/bin/env python3
"""Circular Near-Field to Far-Field (NF-FF) transformation.

This module replicates the exact mathematical algorithm from a MATLAB circular
NF-FF transformation script (``untitled2.m``).  It uses a 1-D cylindrical-mode
expansion (CME) with Hankel-function compensation to convert a measured
near-field circular scan into a far-field radiation pattern.

Algorithm overview
------------------
1. Convert measured magnitude (dB) and phase (deg) to a complex E-field.
2. Extract cylindrical modes via ``fftshift(fft(E_NF)) / N``.
3. Compensate each mode with the Hankel function of the second kind
   (``H_n^{(2)}(k * R_probe)``) and an ``(j)^n`` phase factor.
4. Zero-out evanescent modes (``|n| > k * R_probe``).
5. Reconstruct the far-field pattern with ``ifft(ifftshift(modes_FF))``.
6. Normalise to dB relative to the peak.

Dependencies
------------
* numpy
* scipy  (``scipy.special.hankel2``)
* matplotlib  (optional, for plotting)
* pandas  (optional, for CSV reading -- falls back to numpy)

Usage
-----
As a library::

    from nf_ff_circular import CircularNFFFTransformer

    xfm = CircularNFFFTransformer(freq_hz=10e9, r_probe_m=0.5)
    xfm.load_csv("data/processed/cleaned_sweep.csv")
    result = xfm.transform()
    xfm.plot(result)

From the command line::

    python nf_ff_circular.py --input cleaned_sweep.csv --freq 10 --r-probe 0.5
"""

from __future__ import annotations

import argparse
import logging
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
from numpy.typing import NDArray
from scipy.special import hankel2  # Hankel function of the second kind

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CircularFFResult:
    """Container for all intermediate and final NF-FF transformation data.

    Attributes
    ----------
    angles_deg : NDArray[np.float64]
        Azimuthal angles in degrees (0 -- 359 for a 360-point scan).
    nf_mag_dB : NDArray[np.float64]
        Near-field magnitude in dB (normalised to peak = 0 dB).
    ff_mag_dB : NDArray[np.float64]
        Far-field magnitude in dB (normalised to peak = 0 dB).
    E_NF : NDArray[np.complex128]
        Complex near-field E-field phasor.
    E_FF : NDArray[np.complex128]
        Complex far-field E-field phasor (after CME compensation).
    modes_NF : NDArray[np.complex128]
        Near-field cylindrical-mode spectrum (``fftshift(fft(E_NF)) / N``).
    modes_FF : NDArray[np.complex128]
        Far-field cylindrical-mode spectrum (after Hankel compensation and
        evanescent filtering).
    mode_indices : NDArray[np.int64]
        Mode index vector ``n`` (``-N//2 ... (N-1)//2``).
    max_mode : int
        Maximum propagating mode index (``floor(k * R_probe)``).
    """

    angles_deg: NDArray[np.float64]
    nf_mag_dB: NDArray[np.float64]
    ff_mag_dB: NDArray[np.float64]
    E_NF: NDArray[np.complex128]
    E_FF: NDArray[np.complex128]
    modes_NF: NDArray[np.complex128]
    modes_FF: NDArray[np.complex128]
    mode_indices: NDArray[np.int64]
    max_mode: int


# ---------------------------------------------------------------------------
# Transformer class
# ---------------------------------------------------------------------------

class CircularNFFFTransformer:
    """One-dimensional circular NF-FF transformer using cylindrical-mode
    expansion (CME) with Hankel-function probe compensation.

    Parameters
    ----------
    freq_hz : float
        Operating frequency in Hz (e.g. ``10e9`` for 10 GHz).
    r_probe_m : float
        Probe (measurement) radius in metres.

    Raises
    ------
    ValueError
        If ``freq_hz`` or ``r_probe_m`` are not positive.
    """

    # Physical constants
    _C0: float = 3.0e8  # speed of light in vacuum [m/s]

    def __init__(self, freq_hz: float, r_probe_m: float) -> None:
        if freq_hz <= 0:
            raise ValueError(f"freq_hz must be positive, got {freq_hz}")
        if r_probe_m <= 0:
            raise ValueError(f"r_probe_m must be positive, got {r_probe_m}")

        self.freq_hz: float = freq_hz
        self.r_probe_m: float = r_probe_m
        self.k: float = (2.0 * np.pi * freq_hz) / self._C0

        # Spatial Nyquist check (MATLAB: max_physical_mode = k * R_probe)
        self._max_physical_mode: float = self.k * self.r_probe_m
        if self._max_physical_mode > 180:
            warnings.warn(
                "Spatial Nyquist Violation: max_physical_mode "
                f"({self._max_physical_mode:.1f}) exceeds 180.  "
                "Increase angular sampling or reduce probe radius.",
                stacklevel=2,
            )
        logger.info(
            "Transformer initialised: freq=%.3g GHz, R_probe=%.4f m, "
            "k=%.4f rad/m, max_physical_mode=%.1f",
            freq_hz / 1e9,
            r_probe_m,
            self.k,
            self._max_physical_mode,
        )

        # Data placeholders (populated by load_csv / load_arrays)
        self._angles_deg: Optional[NDArray[np.float64]] = None
        self._mag_dB: Optional[NDArray[np.float64]] = None
        self._phase_deg: Optional[NDArray[np.float64]] = None

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_csv(self, filepath: Union[str, Path]) -> None:
        """Load a cleaned near-field CSV file.

        Expected CSV format (with header row)::

            angle_deg,mag_dB,phase_deg
            0, -12.3, 45.6
            1, -11.8, 47.2
            ...

        Parameters
        ----------
        filepath : str or Path
            Path to the CSV file.

        Raises
        ------
        FileNotFoundError
            If the file does not exist.
        ValueError
            If the file does not contain exactly three columns.
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"CSV file not found: {filepath}")

        data: NDArray[np.float64] = np.loadtxt(
            filepath, delimiter=",", skiprows=1, dtype=np.float64
        )

        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"Expected 3 columns (angle_deg, mag_dB, phase_deg), "
                f"got shape {data.shape}"
            )

        self._angles_deg = data[:, 0].copy()
        self._mag_dB = data[:, 1].copy()
        self._phase_deg = data[:, 2].copy()

        logger.info(
            "Loaded %d samples from '%s' (angles %.1f to %.1f deg)",
            len(self._angles_deg),
            filepath,
            self._angles_deg[0],
            self._angles_deg[-1],
        )

    def load_arrays(
        self,
        angles_deg: NDArray[np.float64],
        mag_dB: NDArray[np.float64],
        phase_deg: NDArray[np.float64],
    ) -> None:
        """Load near-field data directly from NumPy arrays.

        Parameters
        ----------
        angles_deg : ndarray
            Azimuthal angles in degrees.
        mag_dB : ndarray
            Measured magnitude in dB.
        phase_deg : ndarray
            Measured phase in degrees.

        Raises
        ------
        ValueError
            If arrays are not 1-D or have mismatched lengths.
        """
        angles_deg = np.asarray(angles_deg, dtype=np.float64).ravel()
        mag_dB = np.asarray(mag_dB, dtype=np.float64).ravel()
        phase_deg = np.asarray(phase_deg, dtype=np.float64).ravel()

        if not (angles_deg.size == mag_dB.size == phase_deg.size):
            raise ValueError(
                "All input arrays must have the same length.  Got "
                f"angles={angles_deg.size}, mag={mag_dB.size}, "
                f"phase={phase_deg.size}."
            )

        self._angles_deg = angles_deg.copy()
        self._mag_dB = mag_dB.copy()
        self._phase_deg = phase_deg.copy()

        logger.info("Loaded %d samples from arrays.", len(self._angles_deg))

    # ------------------------------------------------------------------
    # Core transformation
    # ------------------------------------------------------------------

    def transform(self) -> CircularFFResult:
        """Execute the full CME NF-FF transformation pipeline.

        Returns
        -------
        CircularFFResult
            Dataclass containing all intermediate and final results.

        Raises
        ------
        RuntimeError
            If no data has been loaded yet.

        Notes
        -----
        The algorithm follows the MATLAB reference exactly:

        1. Complex E-field:  ``E_NF = 10^(mag_dB/20) * exp(j * phase_rad)``
        2. Mode extraction:  ``modes_NF = fftshift(fft(E_NF)) / N``
        3. Hankel compensation:
           ``modes_FF = (modes_NF / H_n^{(2)}(kR)) * (j)^n``
        4. Evanescent filtering: zero modes with ``|n| > floor(kR)``
        5. Far-field reconstruction: ``E_FF = ifft(ifftshift(modes_FF))``
        6. dB normalisation: ``FF_dB = 20*log10(|E_FF|) - max(...)``
        """
        if self._angles_deg is None or self._mag_dB is None or self._phase_deg is None:
            raise RuntimeError(
                "No data loaded.  Call load_csv() or load_arrays() first."
            )

        # -- Step 4: Convert to complex E-field --------------------------------
        mag_linear: NDArray[np.float64] = np.power(10.0, self._mag_dB / 20.0)
        phase_rad: NDArray[np.float64] = np.deg2rad(self._phase_deg)
        E_NF: NDArray[np.complex128] = mag_linear * np.exp(1j * phase_rad)

        N: int = len(self._angles_deg)

        # -- Step 5: Mode extraction via 1-D FFT (scaled by 1/N) ---------------
        modes_NF: NDArray[np.complex128] = np.fft.fftshift(np.fft.fft(E_NF)) / N

        # -- Step 6: Mode indices -----------------------------------------------
        # MATLAB: n = -floor(N/2) : floor((N-1)/2)
        mode_indices: NDArray[np.int64] = np.arange(
            -(N // 2), (N - 1) // 2 + 1, dtype=np.int64
        )

        # -- Step 7: CME compensation with Hankel functions ---------------------
        # H_n^{(2)}(k * R_probe)
        H2_near: NDArray[np.complex128] = hankel2(mode_indices, self.k * self.r_probe_m)
        modes_FF: NDArray[np.complex128] = (modes_NF / H2_near) * np.power(1j, mode_indices)

        # -- Step 8: Evanescent mode filtering ----------------------------------
        max_mode: int = int(np.floor(self.k * self.r_probe_m))
        evanescent_mask: NDArray[np.bool_] = np.abs(mode_indices) > max_mode
        modes_FF[evanescent_mask] = 0.0 + 0.0j

        # -- Step 9: Reconstruct far-field --------------------------------------
        E_FF: NDArray[np.complex128] = np.fft.ifft(np.fft.ifftshift(modes_FF))

        # -- Step 10: Normalise to dB -------------------------------------------
        ff_mag_dB: NDArray[np.float64] = 20.0 * np.log10(np.abs(E_FF))
        ff_mag_dB = ff_mag_dB - np.max(ff_mag_dB)

        # Near-field dB (normalised for plotting)
        nf_mag_dB: NDArray[np.float64] = self._mag_dB - np.max(self._mag_dB)

        logger.info(
            "Transform complete: N=%d, max_mode=%d, "
            "FF peak=%.2f dB (normalised to 0 dB)",
            N,
            max_mode,
            0.0,
        )

        return CircularFFResult(
            angles_deg=self._angles_deg.copy(),
            nf_mag_dB=nf_mag_dB,
            ff_mag_dB=ff_mag_dB,
            E_NF=E_NF,
            E_FF=E_FF,
            modes_NF=modes_NF,
            modes_FF=modes_FF,
            mode_indices=mode_indices,
            max_mode=max_mode,
        )

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, result: CircularFFResult) -> None:
        """Generate a two-panel polar plot matching the MATLAB reference.

        * Left panel  -- Near-field pattern (blue)
        * Right panel -- Far-field pattern  (red)

        Parameters
        ----------
        result : CircularFFResult
            Output of :meth:`transform`.
        """
        import matplotlib
        import matplotlib.pyplot as plt

        # Use Agg backend if no display is available
        try:
            fig_test = plt.figure()
            plt.close(fig_test)
        except Exception:
            matplotlib.use("Agg")

        theta_rad: NDArray[np.float64] = np.deg2rad(result.angles_deg)

        fig, (ax_nf, ax_ff) = plt.subplots(
            1, 2, subplot_kw={"projection": "polar"}, figsize=(14, 6)
        )

        # -- Near-field polar plot (blue) --------------------------------------
        ax_nf.plot(theta_rad, result.nf_mag_dB, color="blue", linewidth=1.2)
        ax_nf.set_title("Near-Field (Measured)", va="bottom", fontsize=11)
        ax_nf.set_theta_zero_location("N")  # 'top' in MATLAB
        ax_nf.set_rlim(-40, 0)
        ax_nf.set_rlabel_position(135)

        # -- Far-field polar plot (red) ----------------------------------------
        ax_ff.plot(theta_rad, result.ff_mag_dB, color="red", linewidth=1.2)
        ax_ff.set_title("Far-Field (Transformed)", va="bottom", fontsize=11)
        ax_ff.set_theta_zero_location("N")  # 'top' in MATLAB
        ax_ff.set_rlim(-40, 0)
        ax_ff.set_rlabel_position(135)

        freq_ghz: float = self.freq_hz / 1e9
        fig.suptitle(
            f"1D Circular NF-FF Transformation ({freq_ghz:.0f} GHz)",
            fontsize=13,
            fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_results(
        self,
        result: CircularFFResult,
        filepath: Union[str, Path],
    ) -> None:
        """Save transformation results to a CSV file.

        Output columns: ``angle_deg, nf_mag_dB, ff_mag_dB``

        Parameters
        ----------
        result : CircularFFResult
            Output of :meth:`transform`.
        filepath : str or Path
            Destination CSV path.
        """
        filepath = Path(filepath)
        header = "angle_deg,nf_mag_dB,ff_mag_dB"
        out_data: NDArray[np.float64] = np.column_stack(
            (result.angles_deg, result.nf_mag_dB, result.ff_mag_dB)
        )
        np.savetxt(
            filepath,
            out_data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.6f",
        )
        logger.info("Results saved to '%s'", filepath)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Circular Near-Field to Far-Field (NF-FF) transformation "
            "using cylindrical-mode expansion with Hankel-function "
            "probe compensation."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/processed/cleaned_sweep.csv",
        help="Path to the input near-field CSV file.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="Operating frequency in GHz.",
    )
    parser.add_argument(
        "--r-probe",
        type=float,
        default=0.5,
        help="Probe (measurement) radius in metres.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        help="Suppress the polar plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the output CSV (angles + NF + FF in dB).",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """Entry point for the command-line interface."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    parser = _build_parser()
    args = parser.parse_args(argv)

    freq_hz: float = args.freq * 1e9
    r_probe_m: float = args.r_probe

    logger.info("--- Circular NF-FF Transformation ---")
    logger.info("Frequency : %.3f GHz", args.freq)
    logger.info("Probe radius : %.4f m", r_probe_m)
    logger.info("Input file : %s", args.input)

    # Build transformer and run pipeline
    transformer = CircularNFFFTransformer(freq_hz=freq_hz, r_probe_m=r_probe_m)
    transformer.load_csv(args.input)
    result: CircularFFResult = transformer.transform()

    # Summary statistics
    logger.info("Max propagating mode : %d", result.max_mode)
    logger.info("Number of samples : %d", len(result.angles_deg))
    logger.info(
        "FF dynamic range : %.2f dB",
        float(np.max(result.ff_mag_dB) - np.min(result.ff_mag_dB)),
    )

    # Save results
    if args.output:
        transformer.save_results(result, args.output)
        logger.info("Output saved to: %s", args.output)

    # Plot
    if not args.no_plot:
        transformer.plot(result)

    logger.info("Done.")


if __name__ == "__main__":
    main()
