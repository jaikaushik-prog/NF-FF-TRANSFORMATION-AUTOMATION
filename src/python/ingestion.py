"""
VNA Data Ingestion Module
=========================
Parses raw VNA CSV exports and CST simulation files into standardised
Pandas DataFrames for the NF-FF metrology pipeline.

Supported formats
-----------------
A) CST planar near-field export  (whitespace-separated, 9 columns)
B) Simple CSV  (angle_deg, mag_dB, phase_deg)
C) Extended VNA CSV  (angle, mag, phase, temperature, elapsed_time)

Author : NF Metrology Pipeline
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ───────────────────────────────────────────────────────────────────
# Result container
# ───────────────────────────────────────────────────────────────────

@dataclass
class IngestedData:
    """Standardised output from the ingestion stage.

    Attributes
    ----------
    df : pd.DataFrame
        Parsed measurement data with standardised column names.
    format_detected : str
        One of ``'cst_planar'``, ``'simple_csv'``, ``'extended_csv'``.
    grid_shape : tuple[int, int] | None
        ``(Ny, Nx)`` for planar grids, ``None`` for 1-D sweeps.
    Ex_2D : np.ndarray | None
        Complex 2-D Ex field grid for planar data.
    Ey_2D : np.ndarray | None
        Complex 2-D Ey field grid for planar data.
    x_mm : np.ndarray | None
        Unique x-axis positions (mm) for planar data.
    y_mm : np.ndarray | None
        Unique y-axis positions (mm) for planar data.
    angles : np.ndarray | None
        1-D angle array (degrees) for sweep data.
    magnitudes : np.ndarray | None
        1-D magnitude array (dB) for sweep data.
    phases : np.ndarray | None
        1-D phase array (degrees) for sweep data.
    has_temperature : bool
        Whether ambient temperature data is available.
    has_elapsed_time : bool
        Whether elapsed-time data is available.
    """

    df: pd.DataFrame
    format_detected: str = "unknown"
    grid_shape: Optional[tuple] = None
    Ex_2D: Optional[np.ndarray] = None
    Ey_2D: Optional[np.ndarray] = None
    x_mm: Optional[np.ndarray] = None
    y_mm: Optional[np.ndarray] = None
    angles: Optional[np.ndarray] = None
    magnitudes: Optional[np.ndarray] = None
    phases: Optional[np.ndarray] = None
    has_temperature: bool = False
    has_elapsed_time: bool = False


# ───────────────────────────────────────────────────────────────────
# Main ingester class
# ───────────────────────────────────────────────────────────────────

class VNADataIngester:
    """Flexible parser for VNA and CST simulation data files.

    Usage
    -----
    >>> ingester = VNADataIngester()
    >>> data = ingester.parse("data/raw/Simulated_NF_Data.txt")
    >>> print(data.format_detected, data.grid_shape)
    """

    # Column name aliases for auto-detection
    _ANGLE_ALIASES = {"angle_deg", "angle", "motor_angle_deg", "theta", "phi"}
    _MAG_ALIASES = {"mag_db", "mag", "magnitude", "amplitude", "raw_mag_db"}
    _PHASE_ALIASES = {"phase_deg", "phase", "raw_phase_deg"}
    _TEMP_ALIASES = {"ambient_temp_c", "temperature", "temp_c", "temp"}
    _TIME_ALIASES = {"elapsed_time_min", "elapsed_min", "time_min", "time"}

    def parse(
        self,
        source: Union[str, Path, io.IOBase],
        format_hint: Optional[str] = None,
    ) -> IngestedData:
        """Parse a data file into a standardised IngestedData object.

        Parameters
        ----------
        source : str, Path, or file-like object
            Path to the data file or an open file object.
        format_hint : str | None
            Force a specific format: ``'cst_planar'``, ``'simple_csv'``,
            or ``'extended_csv'``.  If ``None``, auto-detection is used.

        Returns
        -------
        IngestedData
            Standardised data container.

        Raises
        ------
        ValueError
            If the file cannot be parsed or has an unsupported format.
        """
        # Resolve to string content
        raw_text, filepath_str = self._read_source(source)

        # Auto-detect format
        fmt = format_hint or self._detect_format(raw_text)
        logger.info("Ingesting data  (format: %s)  from: %s", fmt, filepath_str)

        if fmt == "cst_planar":
            return self._parse_cst_planar(raw_text, filepath_str)
        elif fmt in ("simple_csv", "extended_csv"):
            return self._parse_csv(raw_text, filepath_str, fmt)
        else:
            raise ValueError(f"Unsupported data format: {fmt}")

    # ── internal helpers ─────────────────────────────────────────

    @staticmethod
    def _read_source(source) -> tuple:
        """Read source into raw text and return (text, label)."""
        if isinstance(source, (str, Path)):
            path = Path(source)
            if not path.is_file():
                raise FileNotFoundError(f"Data file not found: {path}")
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            return raw_text, str(path)
        elif hasattr(source, "read"):
            raw_text = source.read()
            if isinstance(raw_text, bytes):
                raw_text = raw_text.decode("utf-8", errors="replace")
            name = getattr(source, "name", "<uploaded file>")
            return raw_text, name
        else:
            raise TypeError(f"Unsupported source type: {type(source)}")

    @staticmethod
    def _detect_format(raw_text: str) -> str:
        """Detect the file format from its header."""
        first_line = raw_text.split("\n")[0].strip()

        if "ExRe" in first_line or "x [mm]" in first_line:
            return "cst_planar"

        # Check for CSV-style header
        lower = first_line.lower().replace(" ", "")
        if "angle" in lower or "motor_angle" in lower:
            if "temp" in lower or "elapsed" in lower:
                return "extended_csv"
            return "simple_csv"

        # Fallback: if comma-separated with 3+ numeric columns
        if "," in first_line:
            return "simple_csv"

        # Whitespace-separated with many columns → likely CST
        parts = first_line.split()
        if len(parts) >= 7:
            return "cst_planar"

        return "simple_csv"

    def _parse_cst_planar(self, raw_text: str, label: str) -> IngestedData:
        """Parse a CST planar near-field export."""
        data = np.loadtxt(io.StringIO(raw_text), skiprows=2)

        if data.ndim != 2 or data.shape[1] < 7:
            raise ValueError(
                f"CST data requires >= 7 columns, got shape {data.shape}"
            )

        x = data[:, 0]
        y = data[:, 1]
        Ex_Re, Ex_Im = data[:, 3], data[:, 4]
        Ey_Re, Ey_Im = data[:, 5], data[:, 6]

        Ex = Ex_Re + 1j * Ex_Im
        Ey = Ey_Re + 1j * Ey_Im

        x_unique = np.unique(x)
        y_unique = np.unique(y)
        Nx, Ny = len(x_unique), len(y_unique)

        Ex_2D = Ex.reshape((Nx, Ny), order="F").T
        Ey_2D = Ey.reshape((Nx, Ny), order="F").T

        # Build a centre-row 1-D profile for the autoencoder
        centre_row = Ny // 2
        Ey_centre = Ey_2D[centre_row, :]
        mag_centre = np.abs(Ey_centre)
        mag_dB = 20.0 * np.log10(mag_centre / mag_centre.max() + 1e-12)

        # Map x to observation angle
        z_val = data[:, 2][0]
        theta_deg = np.degrees(np.arctan2(x_unique, z_val)) + 180.0

        df = pd.DataFrame({
            "angle_deg": theta_deg,
            "mag_dB": mag_dB,
            "phase_deg": np.degrees(np.angle(Ey_centre)),
        })

        logger.info(
            "  [OK] CST planar grid: Nx=%d, Ny=%d  (%s)", Nx, Ny, label
        )

        return IngestedData(
            df=df,
            format_detected="cst_planar",
            grid_shape=(Ny, Nx),
            Ex_2D=Ex_2D,
            Ey_2D=Ey_2D,
            x_mm=x_unique,
            y_mm=y_unique,
            angles=theta_deg,
            magnitudes=mag_dB.astype(np.float32),
            phases=np.degrees(np.angle(Ey_centre)).astype(np.float32),
            has_temperature=False,
            has_elapsed_time=False,
        )

    def _parse_csv(self, raw_text: str, label: str, fmt: str) -> IngestedData:
        """Parse a CSV measurement file."""
        df = pd.read_csv(io.StringIO(raw_text))
        cols_lower = {c.lower().strip(): c for c in df.columns}

        # Map standard column names
        angle_col = self._find_col(cols_lower, self._ANGLE_ALIASES)
        mag_col = self._find_col(cols_lower, self._MAG_ALIASES)
        phase_col = self._find_col(cols_lower, self._PHASE_ALIASES)

        if angle_col is None or mag_col is None:
            # Try positional fallback (first 3 columns)
            if df.shape[1] >= 3:
                df.columns = ["angle_deg", "mag_dB", "phase_deg"] + list(
                    df.columns[3:]
                )
                angle_col, mag_col, phase_col = (
                    "angle_deg", "mag_dB", "phase_deg"
                )
            else:
                raise ValueError(
                    f"Cannot identify angle/magnitude columns in {label}. "
                    f"Found columns: {list(df.columns)}"
                )

        angles = df[angle_col].values.astype(np.float64)
        mags = df[mag_col].values.astype(np.float32)
        phases = (
            df[phase_col].values.astype(np.float32)
            if phase_col else np.zeros_like(mags)
        )

        has_temp = self._find_col(cols_lower, self._TEMP_ALIASES) is not None
        has_time = self._find_col(cols_lower, self._TIME_ALIASES) is not None

        logger.info(
            "  [OK] CSV parsed: %d rows, temp=%s, time=%s  (%s)",
            len(df), has_temp, has_time, label,
        )

        return IngestedData(
            df=df,
            format_detected=fmt,
            grid_shape=None,
            angles=angles,
            magnitudes=mags,
            phases=phases,
            has_temperature=has_temp,
            has_elapsed_time=has_time,
        )

    @staticmethod
    def _find_col(cols_lower: dict, aliases: set) -> Optional[str]:
        """Find a column by checking against a set of aliases."""
        for alias in aliases:
            if alias in cols_lower:
                return cols_lower[alias]
        return None
