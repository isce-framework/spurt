"""Load perpendicular baseline data from CSV files."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = [
    "BaselineData",
    "load_baseline_csv",
]


@dataclass
class BaselineData:
    """Container for per-SLC baseline information.

    Parameters
    ----------
    dates : np.ndarray
        Array of dates as datetime64[D], shape (n_slc,).
    bperp_m : np.ndarray
        Perpendicular baseline in meters for each SLC, shape (n_slc,).
        Values are relative to first SLC (reference).
    """

    dates: np.ndarray
    bperp_m: np.ndarray

    def __post_init__(self):
        if self.dates.shape != self.bperp_m.shape:
            errmsg = (
                f"Shape mismatch: dates {self.dates.shape}"
                f" vs bperp_m {self.bperp_m.shape}"
            )
            raise ValueError(errmsg)
        if self.dates.ndim != 1:
            errmsg = f"Expected 1D arrays, got dates.ndim={self.dates.ndim}"
            raise ValueError(errmsg)


def load_baseline_csv(filepath: str | Path) -> BaselineData:
    """Load per-SLC baselines from CSV file.

    Supports two CSV formats:

    1. Per-SLC format with columns: date,bperp_m
       Each row represents one SLC acquisition.

    2. Per-IFG format with columns: reference,secondary,...,bperp_m
       Each row represents one interferogram. This format is automatically
       detected and converted to per-SLC baselines via least-squares,
       with the first SLC as reference (bperp=0).

    Parameters
    ----------
    filepath : str | Path
        Path to CSV file.

    Returns
    -------
    BaselineData
        Container with dates and perpendicular baselines per SLC.
    """
    filepath = Path(filepath)

    with open(filepath) as f:
        header = f.readline().strip().lower()

    columns = [c.strip() for c in header.split(",")]

    if "date" in columns and "bperp_m" in columns:
        return _load_per_slc_csv(filepath, columns)
    if "reference" in columns and "secondary" in columns and "bperp_m" in columns:
        return _load_per_ifg_csv(filepath, columns)
    errmsg = (
        f"Unrecognized CSV format in {filepath}. "
        "Expected either 'date,bperp_m' (per-SLC) or "
        "'reference,secondary,...,bperp_m' (per-IFG) columns."
    )
    raise ValueError(errmsg)


def _load_per_slc_csv(filepath: Path, columns: list[str]) -> BaselineData:
    """Load per-SLC format CSV."""
    date_col = columns.index("date")
    bperp_col = columns.index("bperp_m")

    dates_list = []
    bperp_list = []

    with open(filepath) as f:
        next(f)  # skip header
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or not parts[0]:
                continue
            dates_list.append(_parse_date(parts[date_col]))
            bperp_list.append(float(parts[bperp_col]))

    dates = np.array(dates_list, dtype="datetime64[D]")
    bperp_m = np.array(bperp_list, dtype=np.float64)

    # Sort by date
    sort_idx = np.argsort(dates)
    dates = dates[sort_idx]
    bperp_m = bperp_m[sort_idx]

    return BaselineData(dates=dates, bperp_m=bperp_m)


def _parse_date(date_str: str) -> str:
    """Parse date string to ISO format (YYYY-MM-DD).

    Supports formats: YYYYMMDD, YYYY-MM-DD.
    """
    date_str = date_str.strip()
    if len(date_str) == 8 and date_str.isdigit():
        # YYYYMMDD format
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    return date_str


def _load_per_ifg_csv(filepath: Path, columns: list[str]) -> BaselineData:
    """Load per-IFG format CSV and convert to per-SLC via least-squares.

    The interferometric baseline relationship is:
        bperp_ifg[i,j] = bperp[j] - bperp[i]

    We solve for per-SLC baselines using least-squares with the
    first SLC as reference (bperp=0).
    """
    ref_col = columns.index("reference")
    sec_col = columns.index("secondary")
    bperp_col = columns.index("bperp_m")

    ref_dates = []
    sec_dates = []
    bperp_ifg = []

    with open(filepath) as f:
        next(f)  # skip header
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or not parts[0]:
                continue
            ref_dates.append(_parse_date(parts[ref_col]))
            sec_dates.append(_parse_date(parts[sec_col]))
            bperp_ifg.append(float(parts[bperp_col]))

    # Get unique dates sorted
    all_dates = sorted(set(ref_dates) | set(sec_dates))
    date_to_idx = {d: i for i, d in enumerate(all_dates)}
    n_slc = len(all_dates)
    n_ifg = len(bperp_ifg)

    # Build design matrix: A[ifg, :] has -1 at reference, +1 at secondary
    # bperp_ifg = A @ bperp_slc
    # First column is reference (bperp=0), so we solve for columns 1:
    amat = np.zeros((n_ifg, n_slc - 1), dtype=np.float64)
    bvec = np.array(bperp_ifg, dtype=np.float64)

    for i, (ref, sec) in enumerate(zip(ref_dates, sec_dates)):
        ref_idx = date_to_idx[ref]
        sec_idx = date_to_idx[sec]
        # bperp_ifg[i] = bperp[sec] - bperp[ref]
        # Column 0 is reference with bperp=0, so only adjust for non-reference
        if ref_idx > 0:
            amat[i, ref_idx - 1] = -1.0
        if sec_idx > 0:
            amat[i, sec_idx - 1] = 1.0

    # Solve least-squares
    result, *_ = np.linalg.lstsq(amat, bvec, rcond=None)

    # Prepend zero for reference SLC
    bperp_m = np.zeros(n_slc, dtype=np.float64)
    bperp_m[1:] = result

    dates = np.array(all_dates, dtype="datetime64[D]")

    return BaselineData(dates=dates, bperp_m=bperp_m)
