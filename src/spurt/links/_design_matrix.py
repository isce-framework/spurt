"""Build design matrices for link model estimation."""

from __future__ import annotations

import numpy as np

__all__ = [
    "build_design_matrix",
]


def build_design_matrix(
    ifg_edges: np.ndarray,
    dates: np.ndarray,
    bperp_m: np.ndarray,
    wavelength_m: float,
    slant_range_m: float,
    incidence_rad: float,
) -> np.ndarray:
    """Build design matrix for velocity and DEM error estimation.

    The design matrix relates model parameters (velocity, DEM error) to
    interferometric phase via the linear model:

        phase = amat @ [velocity, dem_error]

    where:
        - Column 0: velocity sensitivity (rad per mm/yr)
        - Column 1: DEM error sensitivity (rad per meter)

    Parameters
    ----------
    ifg_edges : np.ndarray
        Interferogram edges as (nifgs, 2) array of SLC indices.
        Each row [i, j] represents an interferogram from SLC i to SLC j.
    dates : np.ndarray
        SLC acquisition dates as datetime64[D], shape (n_slc,).
    bperp_m : np.ndarray
        Perpendicular baseline in meters per SLC, shape (n_slc,).
    wavelength_m : float
        Radar wavelength in meters (e.g., 0.055465 for Sentinel-1 C-band).
    slant_range_m : float
        Slant range distance in meters (e.g., 900000 for Sentinel-1).
    incidence_rad : float
        Incidence angle in radians (e.g., 0.68 rad = 39 deg for Sentinel-1).

    Returns
    -------
    amat : np.ndarray
        Design matrix of shape (nifgs, 2).
        Column 0: temporal sensitivity for velocity (rad per mm/yr).
        Column 1: baseline sensitivity for DEM error (rad per meter).

    Notes
    -----
    The phase model is:

        phi = (4 * pi / wavelength) * velocity * delta_t / 1000 / 365.25
            + (4 * pi / wavelength) * (bperp / (slant_range * sin(inc))) * dem_error

    where velocity is in mm/yr and dem_error is in meters.

    References
    ----------
    .. [1] Ferretti, A., Prati, C. and Rocca, F., 2001. Permanent scatterers
           in SAR interferometry. IEEE Transactions on geoscience and remote
           sensing, 39(1), pp.8-20.
    """
    nifgs = len(ifg_edges)
    amat = np.zeros((nifgs, 2), dtype=np.float64)

    # Common factor: 4 * pi / wavelength
    phase_factor = 4.0 * np.pi / wavelength_m

    # DEM error factor: bperp / (slant_range * sin(incidence))
    dem_scale = 1.0 / (slant_range_m * np.sin(incidence_rad))

    for ii, (ref_idx, sec_idx) in enumerate(ifg_edges):
        # Temporal baseline in days
        delta_t_days = (dates[sec_idx] - dates[ref_idx]).astype("timedelta64[D]")
        delta_t_days = float(delta_t_days.astype(np.float64))

        # Perpendicular baseline difference in meters
        delta_bperp = bperp_m[sec_idx] - bperp_m[ref_idx]

        # Column 0: velocity sensitivity (rad per mm/yr)
        # Convert days to years, mm to m: delta_t_days / 365.25 * 0.001
        amat[ii, 0] = phase_factor * delta_t_days / 365.25 * 0.001

        # Column 1: DEM error sensitivity (rad per meter)
        amat[ii, 1] = phase_factor * delta_bperp * dem_scale

    return amat
