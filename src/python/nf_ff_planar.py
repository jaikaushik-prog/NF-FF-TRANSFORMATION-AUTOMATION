#!/usr/bin/env python3
"""Planar Near-Field to Far-Field (NF-FF) Transformation.

Replicates the MATLAB planar NF-FF algorithm using the Plane-Wave Spectrum
(PWS) method.  The workflow is:

1. Load CST-exported planar near-field data (or supply NumPy arrays directly).
2. Compute the 2-D FFT of the tangential E-field components to obtain the PWS.
3. Build the U-V direction-cosine grid and apply evanescent-wave filtering.
4. Convert the result to normalised far-field amplitude in dB.
5. Optionally convert to spherical (theta, phi) coordinates and extract
   principal-plane cuts.

Usage
-----
    python nf_ff_planar.py --input Simulated_NF_Data.txt --freq 10 --n-fft 256

Author : auto-generated
License: MIT
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class PlanarFFResult:
    """Container for planar NF-FF transformation results.

    Attributes
    ----------
    u : NDArray[np.float64]
        1-D direction-cosine axis in the x-direction (length ``N_fft``).
    v : NDArray[np.float64]
        1-D direction-cosine axis in the y-direction (length ``N_fft``).
    U_grid : NDArray[np.float64]
        2-D meshgrid of *u* values, shape ``(N_fft, N_fft)``.
    V_grid : NDArray[np.float64]
        2-D meshgrid of *v* values, shape ``(N_fft, N_fft)``.
    farfield_Ey_dB : NDArray[np.float64]
        Normalised far-field Ey amplitude in dB, shape ``(N_fft, N_fft)``.
    farfield_Ex_dB : NDArray[np.float64]
        Normalised far-field Ex amplitude in dB, shape ``(N_fft, N_fft)``.
    visible_region : NDArray[np.bool_]
        Boolean mask for the visible region (U^2 + V^2 <= 1).
    PWS_x : NDArray[np.complex128]
        Plane-wave spectrum of Ex, shape ``(N_fft, N_fft)``.
    PWS_y : NDArray[np.complex128]
        Plane-wave spectrum of Ey, shape ``(N_fft, N_fft)``.
    """

    u: NDArray[np.float64]
    v: NDArray[np.float64]
    U_grid: NDArray[np.float64]
    V_grid: NDArray[np.float64]
    farfield_Ey_dB: NDArray[np.float64]
    farfield_Ex_dB: NDArray[np.float64]
    visible_region: NDArray[np.bool_]
    PWS_x: NDArray[np.complex128]
    PWS_y: NDArray[np.complex128]


@dataclass
class SphericalFFResult:
    """Container for far-field data in spherical coordinates.

    Attributes
    ----------
    theta : NDArray[np.float64]
        Polar angle grid in radians, shape ``(N_fft, N_fft)``.
    phi : NDArray[np.float64]
        Azimuthal angle grid in radians, shape ``(N_fft, N_fft)``.
    E_theta : NDArray[np.complex128]
        Theta component of the far-field, shape ``(N_fft, N_fft)``.
    E_phi : NDArray[np.complex128]
        Phi component of the far-field, shape ``(N_fft, N_fft)``.
    E_theta_dB : NDArray[np.float64]
        Normalised ``|E_theta|`` in dB.
    E_phi_dB : NDArray[np.float64]
        Normalised ``|E_phi|`` in dB.
    """

    theta: NDArray[np.float64]
    phi: NDArray[np.float64]
    E_theta: NDArray[np.complex128]
    E_phi: NDArray[np.complex128]
    E_theta_dB: NDArray[np.float64]
    E_phi_dB: NDArray[np.float64]


# ---------------------------------------------------------------------------
# Main transformer class
# ---------------------------------------------------------------------------


class PlanarNFFFTransformer:
    """Planar near-field to far-field transformer using the PWS method.

    Parameters
    ----------
    freq_hz : float
        Operating frequency in Hz.
    dx_mm : float
        Sample spacing in the x-direction in mm.
    dy_mm : float
        Sample spacing in the y-direction in mm.
    n_fft : int
        Size of the zero-padded 2-D FFT (applied to both dimensions).
    """

    # Class-level physical constant
    C0_M_PER_S: float = 3.0e8

    def __init__(
        self,
        freq_hz: float = 10.0e9,
        dx_mm: float = 10.0,
        dy_mm: float = 10.0,
        n_fft: int = 256,
    ) -> None:
        self.freq_hz: float = freq_hz
        self.dx_mm: float = dx_mm
        self.dy_mm: float = dy_mm
        self.n_fft: int = n_fft

        # Derived quantities
        self.lambda_mm: float = (self.C0_M_PER_S / self.freq_hz) * 1000.0

        # Populated by load_* methods
        self._Ex_2D: Optional[NDArray[np.complex128]] = None
        self._Ey_2D: Optional[NDArray[np.complex128]] = None
        self._x_mm: Optional[NDArray[np.float64]] = None
        self._y_mm: Optional[NDArray[np.float64]] = None

        # Populated by transform / to_spherical
        self._result: Optional[PlanarFFResult] = None
        self._sph_result: Optional[SphericalFFResult] = None

        logger.info(
            "Transformer initialised: freq=%.3g GHz, lambda=%.2f mm, "
            "dx=%.1f mm, dy=%.1f mm, N_fft=%d",
            self.freq_hz / 1.0e9,
            self.lambda_mm,
            self.dx_mm,
            self.dy_mm,
            self.n_fft,
        )

    # ------------------------------------------------------------------
    # Data loading
    # ------------------------------------------------------------------

    def load_cst_data(self, filepath: str | Path) -> None:
        """Load a CST-exported planar near-field text file.

        The file is expected to have **2 header lines** (column names and a
        dashes separator) followed by whitespace-separated numeric data with
        9 columns::

            x  y  z  ExRe  ExIm  EyRe  EyIm  EzRe  EzIm

        Data layout convention: X varies fastest (inner loop), Y varies
        slowest (outer loop).

        Parameters
        ----------
        filepath : str or Path
            Path to the CST data file.

        Raises
        ------
        FileNotFoundError
            If *filepath* does not exist.
        ValueError
            If the data does not contain at least 7 columns.
        """
        filepath = Path(filepath)
        if not filepath.is_file():
            raise FileNotFoundError(f"CST data file not found: {filepath}")

        logger.info("Loading CST data from '%s' ...", filepath)

        # Skip 2 header lines
        raw_matrix: NDArray[np.float64] = np.loadtxt(
            filepath, skiprows=2, dtype=np.float64
        )

        if raw_matrix.ndim != 2 or raw_matrix.shape[1] < 7:
            raise ValueError(
                f"Expected at least 7 columns, got shape {raw_matrix.shape}"
            )

        # Extract columns (0-indexed): x=0, y=1, ExRe=3, ExIm=4, EyRe=5, EyIm=6
        x_vals: NDArray[np.float64] = raw_matrix[:, 0]
        y_vals: NDArray[np.float64] = raw_matrix[:, 1]
        Ex_Re: NDArray[np.float64] = raw_matrix[:, 3]
        Ex_Im: NDArray[np.float64] = raw_matrix[:, 4]
        Ey_Re: NDArray[np.float64] = raw_matrix[:, 5]
        Ey_Im: NDArray[np.float64] = raw_matrix[:, 6]

        Ex: NDArray[np.complex128] = Ex_Re + 1j * Ex_Im
        Ey: NDArray[np.complex128] = Ey_Re + 1j * Ey_Im

        # Determine grid dimensions from unique coordinate values
        x_unique: NDArray[np.float64] = np.unique(x_vals)
        y_unique: NDArray[np.float64] = np.unique(y_vals)
        Nx: int = int(x_unique.size)
        Ny: int = int(y_unique.size)

        logger.info(
            "Grid detected: Nx=%d (x: %.1f to %.1f), Ny=%d (y: %.1f to %.1f)",
            Nx,
            x_unique[0],
            x_unique[-1],
            Ny,
            y_unique[0],
            y_unique[-1],
        )

        # Reshape to 2-D grids.
        # MATLAB: reshape(Ex, [Nx, Ny]).' gives (Ny, Nx).
        # In NumPy (row-major): reshape to (Nx, Ny) with Fortran order, then
        # transpose -- or equivalently reshape(Ny, Nx) with order='F'.
        Ex_2D: NDArray[np.complex128] = Ex.reshape((Nx, Ny), order="F").T
        Ey_2D: NDArray[np.complex128] = Ey.reshape((Nx, Ny), order="F").T

        self._Ex_2D = Ex_2D
        self._Ey_2D = Ey_2D
        self._x_mm = x_unique
        self._y_mm = y_unique

        logger.info(
            "Near-field arrays shaped to (%d, %d) [rows=Ny, cols=Nx]",
            Ny,
            Nx,
        )

    def load_arrays(
        self,
        Ex_2D: NDArray[np.complex128],
        Ey_2D: NDArray[np.complex128],
        x_mm: NDArray[np.float64],
        y_mm: NDArray[np.float64],
    ) -> None:
        """Load near-field data directly from NumPy arrays.

        Parameters
        ----------
        Ex_2D : ndarray, complex, shape (Ny, Nx)
            2-D complex Ex near-field data.
        Ey_2D : ndarray, complex, shape (Ny, Nx)
            2-D complex Ey near-field data.
        x_mm : ndarray, float, shape (Nx,)
            X-axis sample positions in mm.
        y_mm : ndarray, float, shape (Ny,)
            Y-axis sample positions in mm.

        Raises
        ------
        ValueError
            If the shapes of *Ex_2D* and *Ey_2D* are inconsistent.
        """
        if Ex_2D.shape != Ey_2D.shape:
            raise ValueError(
                f"Ex_2D shape {Ex_2D.shape} != Ey_2D shape {Ey_2D.shape}"
            )
        self._Ex_2D = np.asarray(Ex_2D, dtype=np.complex128)
        self._Ey_2D = np.asarray(Ey_2D, dtype=np.complex128)
        self._x_mm = np.asarray(x_mm, dtype=np.float64)
        self._y_mm = np.asarray(y_mm, dtype=np.float64)
        logger.info(
            "Arrays loaded directly: shape (%d, %d)",
            Ex_2D.shape[0],
            Ex_2D.shape[1],
        )

    # ------------------------------------------------------------------
    # Core transform
    # ------------------------------------------------------------------

    def transform(self) -> PlanarFFResult:
        """Compute the planar NF-FF transformation.

        Performs the following steps:

        1. 2-D FFT (zero-padded to ``n_fft x n_fft``) of Ex and Ey to obtain
           the plane-wave spectrum (PWS).
        2. Build the U-V direction-cosine grid from the sampling spacings.
        3. Apply evanescent-wave filtering (``U^2 + V^2 <= 1``).
        4. Normalise and convert to dB.

        Returns
        -------
        PlanarFFResult
            Dataclass containing all transform outputs.

        Raises
        ------
        RuntimeError
            If no near-field data has been loaded yet.
        """
        if self._Ex_2D is None or self._Ey_2D is None:
            raise RuntimeError(
                "No near-field data loaded. Call load_cst_data() or "
                "load_arrays() first."
            )

        N: int = self.n_fft

        # --- Step 1: PWS via 2-D FFT ---
        PWS_y: NDArray[np.complex128] = np.fft.fftshift(
            np.fft.fft2(self._Ey_2D, s=(N, N))
        )
        PWS_x: NDArray[np.complex128] = np.fft.fftshift(
            np.fft.fft2(self._Ex_2D, s=(N, N))
        )
        logger.info("PWS computed via %dx%d FFT.", N, N)

        # --- Step 2: U-V direction-cosine space ---
        max_u: float = self.lambda_mm / (2.0 * self.dx_mm)
        max_v: float = self.lambda_mm / (2.0 * self.dy_mm)

        u: NDArray[np.float64] = np.linspace(-max_u, max_u, N)
        v: NDArray[np.float64] = np.linspace(-max_v, max_v, N)
        U_grid, V_grid = np.meshgrid(u, v)  # indexing='xy' (default)

        # --- Step 3: Evanescent filtering ---
        visible_region: NDArray[np.bool_] = (U_grid ** 2 + V_grid ** 2) <= 1.0

        # --- Step 4: dB normalisation ---
        farfield_Ey_dB = self._to_normalised_dB(
            np.abs(PWS_y), visible_region, floor_dB=-60.0
        )
        farfield_Ex_dB = self._to_normalised_dB(
            np.abs(PWS_x), visible_region, floor_dB=-60.0
        )

        self._result = PlanarFFResult(
            u=u,
            v=v,
            U_grid=U_grid,
            V_grid=V_grid,
            farfield_Ey_dB=farfield_Ey_dB,
            farfield_Ex_dB=farfield_Ex_dB,
            visible_region=visible_region,
            PWS_x=PWS_x,
            PWS_y=PWS_y,
        )

        logger.info("Transform complete.")
        return self._result

    # ------------------------------------------------------------------
    # Spherical conversion
    # ------------------------------------------------------------------

    def to_spherical(self) -> SphericalFFResult:
        """Convert U-V far-field data to spherical coordinates.

        Computes theta and phi from the direction-cosine grid and projects
        the PWS components onto the theta and phi unit vectors using the
        standard Ludwig-3 relations::

            E_theta =  PWS_x * cos(phi) + PWS_y * sin(phi)
            E_phi   = -PWS_x * sin(phi) + PWS_y * cos(phi)

        Returns
        -------
        SphericalFFResult
            Dataclass with theta, phi, E_theta, E_phi and dB versions.

        Raises
        ------
        RuntimeError
            If ``transform()`` has not been called yet.
        """
        if self._result is None:
            raise RuntimeError("Call transform() before to_spherical().")

        r = self._result
        rho: NDArray[np.float64] = np.sqrt(r.U_grid ** 2 + r.V_grid ** 2)

        # Clip rho to 1.0 to avoid NaN in arcsin for evanescent region
        theta: NDArray[np.float64] = np.arcsin(np.clip(rho, 0.0, 1.0))
        phi: NDArray[np.float64] = np.arctan2(r.V_grid, r.U_grid)

        cos_phi: NDArray[np.float64] = np.cos(phi)
        sin_phi: NDArray[np.float64] = np.sin(phi)

        # Ludwig-3 projection
        E_theta: NDArray[np.complex128] = (
            r.PWS_x * cos_phi + r.PWS_y * sin_phi
        )
        E_phi: NDArray[np.complex128] = (
            -r.PWS_x * sin_phi + r.PWS_y * cos_phi
        )

        # Apply visible-region mask
        E_theta = E_theta * r.visible_region
        E_phi = E_phi * r.visible_region

        E_theta_dB = self._to_normalised_dB(
            np.abs(E_theta), r.visible_region, floor_dB=-60.0
        )
        E_phi_dB = self._to_normalised_dB(
            np.abs(E_phi), r.visible_region, floor_dB=-60.0
        )

        self._sph_result = SphericalFFResult(
            theta=theta,
            phi=phi,
            E_theta=E_theta,
            E_phi=E_phi,
            E_theta_dB=E_theta_dB,
            E_phi_dB=E_phi_dB,
        )

        logger.info("Spherical conversion complete.")
        return self._sph_result

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def plot(self, result: PlanarFFResult) -> None:
        """Generate a 2x2 diagnostic figure.

        Subplots
        --------
        (1,1) Near-field |Ex| in dB
        (1,2) Near-field |Ey| in dB
        (2,1) Far-field Ey in U-V space (matches the MATLAB output)
        (2,2) Far-field theta cuts at phi=0 deg and phi=90 deg

        Parameters
        ----------
        result : PlanarFFResult
            Output of :meth:`transform`.
        """
        import matplotlib

        # Use Agg backend if no display is available
        try:
            import tkinter  # noqa: F401

            matplotlib.use("TkAgg")
        except ImportError:
            matplotlib.use("Agg")
            logger.info("No display detected -- using Agg backend.")

        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))
        fig.suptitle(
            "Planar NF-FF Transformation  "
            f"(f = {self.freq_hz / 1e9:.2f} GHz, "
            f"N_fft = {self.n_fft})",
            fontsize=13,
        )

        # -- (1,1) NF |Ex| in dB --
        ax = axes[0, 0]
        nf_Ex_abs = np.abs(self._Ex_2D)
        nf_Ex_dB = self._safe_db(nf_Ex_abs)
        im0 = ax.imshow(
            nf_Ex_dB,
            extent=[
                self._x_mm[0],
                self._x_mm[-1],
                self._y_mm[0],
                self._y_mm[-1],
            ],
            origin="lower",
            aspect="auto",
            cmap="jet",
        )
        ax.set_title("Near-Field |Ex| (dB)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im0, ax=ax, label="dB")

        # -- (1,2) NF |Ey| in dB --
        ax = axes[0, 1]
        nf_Ey_abs = np.abs(self._Ey_2D)
        nf_Ey_dB = self._safe_db(nf_Ey_abs)
        im1 = ax.imshow(
            nf_Ey_dB,
            extent=[
                self._x_mm[0],
                self._x_mm[-1],
                self._y_mm[0],
                self._y_mm[-1],
            ],
            origin="lower",
            aspect="auto",
            cmap="jet",
        )
        ax.set_title("Near-Field |Ey| (dB)")
        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        fig.colorbar(im1, ax=ax, label="dB")

        # -- (2,1) FF Ey in U-V space --
        ax = axes[1, 0]
        im2 = ax.imshow(
            result.farfield_Ey_dB,
            extent=[result.u[0], result.u[-1], result.v[0], result.v[-1]],
            origin="lower",
            aspect="equal",
            cmap="jet",
            vmin=-60,
            vmax=0,
        )
        # Draw the visible-region circle
        circle_theta = np.linspace(0, 2.0 * np.pi, 361)
        ax.plot(
            np.cos(circle_theta),
            np.sin(circle_theta),
            "w--",
            linewidth=1.0,
        )
        ax.set_title("Far-Field |Ey| (dB) -- U-V Space")
        ax.set_xlabel("u = sin(theta)*cos(phi)")
        ax.set_ylabel("v = sin(theta)*sin(phi)")
        fig.colorbar(im2, ax=ax, label="dB")

        # -- (2,2) FF theta cuts at phi=0 and phi=90 --
        ax = axes[1, 1]
        sph = self.to_spherical()

        # phi=0 cut --> row at V~0 (middle row)
        mid_row: int = self.n_fft // 2
        theta_phi0_deg: NDArray[np.float64] = np.degrees(
            sph.theta[mid_row, :]
        )
        # Assign sign based on u direction for symmetric display
        theta_signed_phi0: NDArray[np.float64] = np.where(
            result.U_grid[mid_row, :] >= 0,
            theta_phi0_deg,
            -theta_phi0_deg,
        )
        mask_phi0: NDArray[np.bool_] = result.visible_region[mid_row, :]

        # phi=90 cut --> column at U~0 (middle column)
        mid_col: int = self.n_fft // 2
        theta_phi90_deg: NDArray[np.float64] = np.degrees(
            sph.theta[:, mid_col]
        )
        theta_signed_phi90: NDArray[np.float64] = np.where(
            result.V_grid[:, mid_col] >= 0,
            theta_phi90_deg,
            -theta_phi90_deg,
        )
        mask_phi90: NDArray[np.bool_] = result.visible_region[:, mid_col]

        ax.plot(
            theta_signed_phi0[mask_phi0],
            result.farfield_Ey_dB[mid_row, mask_phi0],
            "b-",
            linewidth=1.5,
            label="phi = 0 deg",
        )
        ax.plot(
            theta_signed_phi90[mask_phi90],
            result.farfield_Ey_dB[mask_phi90, mid_col],
            "r--",
            linewidth=1.5,
            label="phi = 90 deg",
        )
        ax.set_title("Far-Field |Ey| -- Theta Cuts")
        ax.set_xlabel("Theta (deg)")
        ax.set_ylabel("Normalised Amplitude (dB)")
        ax.set_ylim(-60, 5)
        ax.legend()
        ax.grid(True, alpha=0.4)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
        logger.info("Plot displayed.")

    # ------------------------------------------------------------------
    # CSV export
    # ------------------------------------------------------------------

    def save_results(self, result: PlanarFFResult, filepath: str | Path) -> None:
        """Save the far-field result to a CSV file.

        The CSV contains columns: u, v, FF_Ey_dB, FF_Ex_dB, visible.

        Only points inside the visible region are written to keep the file
        size manageable.

        Parameters
        ----------
        result : PlanarFFResult
            Output of :meth:`transform`.
        filepath : str or Path
            Destination CSV path.
        """
        filepath = Path(filepath)
        mask: NDArray[np.bool_] = result.visible_region

        u_flat: NDArray[np.float64] = result.U_grid[mask]
        v_flat: NDArray[np.float64] = result.V_grid[mask]
        ey_db_flat: NDArray[np.float64] = result.farfield_Ey_dB[mask]
        ex_db_flat: NDArray[np.float64] = result.farfield_Ex_dB[mask]

        data: NDArray[np.float64] = np.column_stack(
            [u_flat, v_flat, ey_db_flat, ex_db_flat]
        )

        header: str = "u,v,FF_Ey_dB,FF_Ex_dB"
        np.savetxt(
            filepath,
            data,
            delimiter=",",
            header=header,
            comments="",
            fmt="%.6e",
        )
        logger.info(
            "Results saved to '%s' (%d visible points).", filepath, int(mask.sum())
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_normalised_dB(
        amplitude: NDArray[np.float64],
        visible_region: NDArray[np.bool_],
        floor_dB: float = -60.0,
    ) -> NDArray[np.float64]:
        """Normalise amplitude within the visible region and convert to dB.

        Parameters
        ----------
        amplitude : ndarray
            Linear amplitude array (must be non-negative).
        visible_region : ndarray of bool
            Mask for the visible region.
        floor_dB : float
            Value assigned to evanescent (invisible) samples.

        Returns
        -------
        ndarray
            Normalised amplitude in dB with evanescent samples set to
            *floor_dB*.
        """
        # Apply visible-region mask before finding the peak
        masked_amp: NDArray[np.float64] = amplitude * visible_region
        peak: float = float(np.max(masked_amp))

        if peak == 0.0:
            logger.warning("Peak amplitude is zero -- returning floor dB.")
            return np.full_like(amplitude, floor_dB, dtype=np.float64)

        # Avoid log10(0) by clamping to a tiny positive value
        ratio: NDArray[np.float64] = np.where(
            masked_amp > 0,
            masked_amp / peak,
            10.0 ** (floor_dB / 20.0),
        )
        db_vals: NDArray[np.float64] = 20.0 * np.log10(ratio)

        # Clamp and blank the evanescent region
        db_vals = np.clip(db_vals, floor_dB, 0.0)
        db_vals[~visible_region] = floor_dB
        return db_vals

    @staticmethod
    def _safe_db(amplitude: NDArray[np.float64]) -> NDArray[np.float64]:
        """Convert linear amplitude to dB, normalised to its own peak.

        Parameters
        ----------
        amplitude : ndarray
            Non-negative linear amplitude.

        Returns
        -------
        ndarray
            Amplitude in dB (peak = 0 dB).
        """
        peak: float = float(np.max(amplitude))
        if peak == 0.0:
            return np.full_like(amplitude, -60.0, dtype=np.float64)
        ratio: NDArray[np.float64] = np.where(
            amplitude > 0, amplitude / peak, 1e-30
        )
        return 20.0 * np.log10(ratio)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Planar Near-Field to Far-Field transformation using the "
            "Plane-Wave Spectrum method."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw/Simulated_NF_Data.txt",
        help="Path to the CST near-field data file.",
    )
    parser.add_argument(
        "--freq",
        type=float,
        default=10.0,
        help="Operating frequency in GHz.",
    )
    parser.add_argument(
        "--n-fft",
        type=int,
        default=256,
        help="FFT zero-padding size (applied to both dimensions).",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Suppress the diagnostic plot.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for the output CSV file.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> None:
    """CLI entry point for the planar NF-FF transformer.

    Parameters
    ----------
    argv : list of str or None
        Command-line arguments (defaults to ``sys.argv[1:]``).
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    freq_hz: float = args.freq * 1.0e9

    logger.info("=== Planar NF-FF Transformation ===")
    logger.info("Input file : %s", args.input)
    logger.info("Frequency  : %.3f GHz", args.freq)
    logger.info("N_fft      : %d", args.n_fft)

    transformer = PlanarNFFFTransformer(
        freq_hz=freq_hz,
        dx_mm=10.0,
        dy_mm=10.0,
        n_fft=args.n_fft,
    )

    transformer.load_cst_data(args.input)
    result: PlanarFFResult = transformer.transform()

    if args.output is not None:
        transformer.save_results(result, args.output)

    if not args.no_plot:
        transformer.plot(result)

    logger.info("Done.")


if __name__ == "__main__":
    main()
