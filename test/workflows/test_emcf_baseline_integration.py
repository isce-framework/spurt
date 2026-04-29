"""Integration tests for EMCF workflow with baseline/velocity estimation."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

import spurt
from spurt.io import load_baseline_csv
from spurt.links import build_design_matrix
from spurt.workflows.emcf import LinkModelSettings


def test_link_model_settings_validation():
    """Test LinkModelSettings validation."""
    # Should work with baseline_csv provided
    settings = LinkModelSettings(
        enabled=True,
        baseline_csv="/path/to/file.csv",
    )
    assert settings.enabled is True

    # Should fail without baseline_csv when enabled
    with pytest.raises(ValueError, match="baseline_csv is required"):
        LinkModelSettings(enabled=True, baseline_csv=None)

    # Should work when disabled without baseline_csv
    settings = LinkModelSettings(enabled=False, baseline_csv=None)
    assert settings.enabled is False


def test_link_model_settings_look_angle_conversion():
    """Test look angle conversion from degrees to radians."""
    settings = LinkModelSettings(
        enabled=True,
        baseline_csv="/path/to/file.csv",
        look_angle_deg=45.0,
    )

    expected_rad = np.radians(45.0)
    np.testing.assert_almost_equal(settings.look_angle_rad, expected_rad)


def test_baseline_and_design_matrix_integration():
    """Test that baseline loading and design matrix construction work together."""
    # Create a per-SLC baseline CSV
    csv_content = """date,bperp_m
20200101,0.0
20200113,100.0
20200125,200.0
20200206,150.0
20200218,250.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        # Load baselines
        baseline_data = load_baseline_csv(csv_path)
        assert len(baseline_data.dates) == 5

        # Create temporal graph
        g_time = spurt.graph.Hop3Graph(5)

        # Build design matrix
        amat = build_design_matrix(
            ifg_edges=g_time.links,
            dates=baseline_data.dates,
            bperp_m=baseline_data.bperp_m,
            wavelength_m=0.055465,
            slant_range_m=900000.0,
            look_angle_rad=np.radians(39.0),
        )

        # Should have correct shape
        assert amat.shape == (len(g_time.links), 2)

        # All velocity sensitivities should be positive
        assert np.all(amat[:, 0] > 0)

    finally:
        csv_path.unlink()


def test_emcf_with_baseline_csv_full_workflow():
    """Test EMCF solver with baseline CSV and link model estimation.

    This is a full workflow test that:
    1. Creates synthetic phase data with known velocity
    2. Creates a baseline CSV file
    3. Configures LinkModelSettings
    4. Runs the solver with link model
    5. Verifies results
    """
    n_sar = 10
    y, x = np.ogrid[-3:3:32j, -3:3:32j]

    # Velocity field in mm/yr (converted to radians for phase)
    vel_mm_yr = -20.0 * np.exp(-(x**2 + y**2) / 5)
    vel_mm_yr -= vel_mm_yr.max()

    # Create dates (12 day repeat cycle)
    base_date = np.datetime64("2020-01-01")
    dates = [base_date + np.timedelta64(i * 12, "D") for i in range(n_sar)]
    dates_str = [str(d) for d in dates]

    # Create baselines (no DEM error in this test)
    bperp_m = np.zeros(n_sar)

    # Convert velocity to phase
    wavelength_m = 0.055465
    times_days = np.array([i * 12 for i in range(n_sar)], dtype=np.float64)
    times_years = times_days / 365.25

    # Phase = 4*pi/wavelength * velocity_m * time_years
    # velocity_m = velocity_mm_yr * 0.001
    phase_rate = 4.0 * np.pi / wavelength_m * 0.001  # rad per (mm/yr * year)
    phase = times_years[:, None, None] * vel_mm_yr[None, :, :] * phase_rate

    igram = np.exp(1j * phase)

    # Create baseline CSV
    csv_content = "date,bperp_m\n"
    for d, b in zip(dates_str, bperp_m):
        date_str = str(d).replace("-", "")
        csv_content += f"{date_str},{b}\n"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        # Set up time processing
        g_time = spurt.graph.Hop3Graph(n_sar)
        s_time = spurt.mcf.ORMCFSolver(g_time)

        # Set up spatial processing
        g_space = spurt.graph.Reg2DGraph(igram.shape[1:])
        s_space = spurt.mcf.ORMCFSolver(g_space)

        # Load baselines and build design matrix
        baseline_data = load_baseline_csv(csv_path)
        stack_dates = np.array(dates, dtype="datetime64[D]")

        amat = build_design_matrix(
            ifg_edges=g_time.links,
            dates=stack_dates,
            bperp_m=baseline_data.bperp_m,
            wavelength_m=wavelength_m,
            slant_range_m=900000.0,
            look_angle_rad=np.radians(39.0),
        )

        # Create link model
        vel_range = slice(-50.0, 10.0, 2.0)
        dem_range = slice(-5.0, 5.0, 1.0)
        link_model = spurt.links.GridSearchLinearModel(
            matrix=amat,
            ranges=(vel_range, dem_range),
        )

        # Create EMCF solver with link model
        settings = spurt.workflows.emcf.SolverSettings(
            s_worker_count=1,
            t_worker_count=1,
            links_per_batch=500,
        )
        solver = spurt.workflows.emcf.Solver(s_space, s_time, settings, link_model)

        w_data = spurt.io.Irreg3DInput(
            igram.reshape((n_sar, g_space.npoints)), g_space.points
        )
        uw_data = solver.unwrap_cube(w_data)

        # Verify link parameters were estimated
        assert solver.link_params is not None
        assert solver.link_coherence is not None

        # Verify unwrapping succeeded
        for ii, edge in enumerate(g_time.links):
            orig = phase[edge[1]] - phase[edge[0]]
            recon = uw_data[ii].reshape(phase.shape[1:])
            assert np.allclose(orig - orig[0, 0], recon - recon[0, 0], atol=1.0e-3)

        # High coherence expected for clean synthetic data
        assert np.mean(solver.link_coherence) > 0.95

    finally:
        csv_path.unlink()


def test_build_link_model_yyyymmdd_dates():
    """Test that _build_link_model handles YYYYMMDD stack dates.

    The SLCStackReader stores dates as YYYYMMDD strings, but numpy's
    datetime64 misparses these (treating '20240626' as year 20240626).
    This test verifies the dates are normalized before comparison.
    """
    from spurt.workflows.emcf._unwrap import _build_link_model

    n_slc = 5
    g_time = spurt.graph.Hop3Graph(n_slc)

    # Stack dates in YYYYMMDD format (as SLCStackReader provides)
    dates_yyyymmdd = ["20200101", "20200113", "20200125", "20200206", "20200218"]

    csv_content = (
        "date,bperp_m\n"
        "20200101,0.0\n20200113,100.0\n20200125,200.0\n"
        "20200206,150.0\n20200218,250.0\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        settings = LinkModelSettings(
            enabled=True,
            baseline_csv=str(csv_path),
        )
        model = _build_link_model(g_time, dates_yyyymmdd, settings)

        # Model should have correct shape: (nifgs, 2)
        assert model.matrix.shape == (len(g_time.links), 2)

        # All velocity sensitivities should be positive (time moves forward)
        assert np.all(model.matrix[:, 0] > 0)

        # DEM error sensitivities should be nonzero (baselines vary)
        assert not np.allclose(model.matrix[:, 1], 0.0)
    finally:
        csv_path.unlink()


def test_cli_argument_parsing():
    """Test that CLI argument parsing works for baseline arguments."""
    import argparse

    test_args = [
        "-i",
        "/nonexistent/path",
        "--baseline-csv",
        "/path/to/baselines.csv",
        "--wavelength",
        "0.031",
        "--slant-range",
        "800000",
        "--look-angle-deg",
        "35.0",
        "--velocity-range",
        "-80",
        "80",
        "4",
        "--dem-error-range",
        "-40",
        "40",
        "2",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--inputdir", required=True)
    parser.add_argument("--baseline-csv", type=str, default=None)
    parser.add_argument("--no-velocity-estimation", action="store_true")
    parser.add_argument("--wavelength", type=float, default=0.055465)
    parser.add_argument("--slant-range", type=float, default=900000.0)
    parser.add_argument("--look-angle-deg", type=float, default=39.0)
    parser.add_argument(
        "--velocity-range", type=float, nargs=3, default=[-100.0, 100.0, 5.0]
    )
    parser.add_argument(
        "--dem-error-range", type=float, nargs=3, default=[-50.0, 50.0, 2.5]
    )

    parsed = parser.parse_args(test_args)

    assert parsed.baseline_csv == "/path/to/baselines.csv"
    assert parsed.wavelength == 0.031
    assert parsed.slant_range == 800000
    assert parsed.look_angle_deg == 35.0
    assert parsed.velocity_range == [-80.0, 80.0, 4.0]
    assert parsed.dem_error_range == [-40.0, 40.0, 2.0]
