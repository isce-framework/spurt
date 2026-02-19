"""Tests for design matrix construction."""

import numpy as np

from spurt.links import build_design_matrix


def test_design_matrix_shape():
    """Test that design matrix has correct shape."""
    # 4 SLCs -> 5 IFGs (hop-3 style: 0-1, 0-2, 0-3, 1-2, 1-3, 2-3)
    ifg_edges = np.array([[0, 1], [0, 2], [1, 2], [1, 3], [2, 3]])
    dates = np.array(
        ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06"],
        dtype="datetime64[D]",
    )
    bperp_m = np.array([0.0, 100.0, 200.0, 150.0])

    amat = build_design_matrix(
        ifg_edges=ifg_edges,
        dates=dates,
        bperp_m=bperp_m,
        wavelength_m=0.055465,
        slant_range_m=900000.0,
        look_angle_rad=np.radians(39.0),
    )

    assert amat.shape == (5, 2)


def test_velocity_sensitivity_calculation():
    """Test velocity sensitivity column calculation.

    For velocity in mm/yr, the phase sensitivity is:
        dphi/dv = 4*pi/wavelength * dt_days/365.25 * 0.001
    """
    ifg_edges = np.array([[0, 1]])
    dates = np.array(["2020-01-01", "2021-01-01"], dtype="datetime64[D]")  # 366 days
    bperp_m = np.array([0.0, 0.0])  # No baseline -> no DEM error sensitivity

    wavelength_m = 0.055465

    amat = build_design_matrix(
        ifg_edges=ifg_edges,
        dates=dates,
        bperp_m=bperp_m,
        wavelength_m=wavelength_m,
        slant_range_m=900000.0,
        look_angle_rad=np.radians(39.0),
    )

    # Expected: 4*pi/0.055465 * 366/365.25 * 0.001 = 0.2273 rad/(mm/yr)
    expected_vel_sens = 4.0 * np.pi / wavelength_m * 366.0 / 365.25 * 0.001
    np.testing.assert_almost_equal(amat[0, 0], expected_vel_sens, decimal=6)


def test_dem_error_sensitivity_calculation():
    """Test DEM error sensitivity column calculation.

    For DEM error in meters, the phase sensitivity is:
        dphi/dh = 4*pi/wavelength * bperp / (slant_range * cos(look))
    """
    ifg_edges = np.array([[0, 1]])
    dates = np.array(["2020-01-01", "2020-01-01"], dtype="datetime64[D]")  # Same date
    bperp_m = np.array([0.0, 100.0])  # 100m baseline difference

    wavelength_m = 0.055465
    slant_range_m = 900000.0
    look_angle_rad = np.radians(39.0)

    amat = build_design_matrix(
        ifg_edges=ifg_edges,
        dates=dates,
        bperp_m=bperp_m,
        wavelength_m=wavelength_m,
        slant_range_m=slant_range_m,
        look_angle_rad=look_angle_rad,
    )

    # Velocity sensitivity should be 0 (same date)
    np.testing.assert_almost_equal(amat[0, 0], 0.0, decimal=10)

    # Expected DEM error sensitivity: 4*pi/wavelength * 100 / (900000 * cos(39deg))
    expected_dem_sens = (
        4.0 * np.pi / wavelength_m * 100.0 / (slant_range_m * np.cos(look_angle_rad))
    )
    np.testing.assert_almost_equal(amat[0, 1], expected_dem_sens, decimal=10)


def test_negative_baseline_difference():
    """Test that negative baseline differences are handled correctly."""
    ifg_edges = np.array([[0, 1]])
    dates = np.array(["2020-01-01", "2020-01-13"], dtype="datetime64[D]")
    bperp_m = np.array([100.0, 0.0])  # Secondary has smaller bperp

    amat = build_design_matrix(
        ifg_edges=ifg_edges,
        dates=dates,
        bperp_m=bperp_m,
        wavelength_m=0.055465,
        slant_range_m=900000.0,
        look_angle_rad=np.radians(39.0),
    )

    # DEM error sensitivity should be negative
    assert amat[0, 1] < 0


def test_multiple_ifgs():
    """Test design matrix for multiple interferograms."""
    # Hop3-style edges for 5 SLCs
    ifg_edges = np.array(
        [
            [0, 1],
            [0, 2],
            [0, 3],
            [1, 2],
            [1, 3],
            [1, 4],
            [2, 3],
            [2, 4],
            [3, 4],
        ]
    )
    dates = np.array(
        ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06", "2020-02-18"],
        dtype="datetime64[D]",
    )
    bperp_m = np.array([0.0, 50.0, 100.0, -50.0, 25.0])

    amat = build_design_matrix(
        ifg_edges=ifg_edges,
        dates=dates,
        bperp_m=bperp_m,
        wavelength_m=0.055465,
        slant_range_m=900000.0,
        look_angle_rad=np.radians(39.0),
    )

    assert amat.shape == (9, 2)

    # All velocity sensitivities should be positive (time always increases)
    assert np.all(amat[:, 0] > 0)

    # Longer temporal baselines should have larger velocity sensitivity
    # IFG 0-3 (36 days) should have larger sensitivity than 0-1 (12 days)
    assert amat[2, 0] > amat[0, 0]
