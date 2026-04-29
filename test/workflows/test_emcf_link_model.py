"""Test EMCF solver with link model for DEM error and velocity estimation."""

import numpy as np
import pytest

import spurt


def gen_data_with_velocity():
    """Generate a regular 3D dataset with known velocity field."""
    n_collects = 20
    y, x = np.ogrid[-3:3:32j, -3:3:32j]

    # Velocity field in radians per time unit
    vel = -np.pi * np.exp(-(x**2 + y**2) / 5) / 12
    vel -= vel.max()

    times = np.arange(n_collects) * 12
    phase = times[:, None, None] * vel[None, :, :]

    return n_collects, times, phase, vel


def test_emcf_with_link_model():
    """Test EMCF unwrapping with link model for velocity estimation."""
    n_sar, times, phase, true_vel = gen_data_with_velocity()
    igram = np.exp(1j * phase)

    # Set up time processing
    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)

    # Set up spatial processing
    g_space = spurt.graph.Reg2DGraph(igram.shape[1:])
    s_space = spurt.mcf.ORMCFSolver(g_space)

    # Build design matrix for velocity estimation
    # For each interferogram, the temporal sensitivity is delta_time
    nifgs = len(g_time.links)
    amat = np.zeros((nifgs, 1))
    for ii, edge in enumerate(g_time.links):
        amat[ii, 0] = times[edge[1]] - times[edge[0]]

    # Create link model with velocity search range
    vel_range = slice(-0.5, 0.1, 0.02)
    link_model = spurt.links.GridSearchLinearModel(matrix=amat, ranges=(vel_range,))

    # Create EMCF solver with link model
    settings = spurt.workflows.emcf.SolverSettings(
        s_worker_count=1,
        t_worker_count=1,
        links_per_batch=10000,
    )
    solver = spurt.workflows.emcf.Solver(s_space, s_time, settings, link_model)

    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    uw_data = solver.unwrap_cube(w_data)

    # Verify unwrapping succeeded
    for ii, edge in enumerate(g_time.links):
        orig = phase[edge[1]] - phase[edge[0]]
        recon = uw_data[ii].reshape(phase.shape[1:])
        assert np.allclose(orig - orig[0, 0], recon - recon[0, 0], atol=1.0e-3)

    # Verify link parameters were estimated
    assert solver.link_params is not None
    assert solver.link_coherence is not None
    assert solver.link_params.shape == (1, solver.nlinks)
    assert solver.link_coherence.shape == (solver.nlinks,)

    # Coherence should be high for clean synthetic data (no noise added).
    # 0.95 threshold ensures model fits well; lower values indicate estimation issues.
    assert np.mean(solver.link_coherence) > 0.95

    # Test integrate_link_params to get point velocities
    point_vel = solver.integrate_link_params(param_idx=0)
    assert point_vel.shape == (solver.npoints,)

    # Point velocities should correlate well with true velocity field.
    # Exact values may differ due to grid search resolution (0.02) and integration.
    # 0.95 correlation threshold ensures spatial pattern is recovered correctly.
    true_vel_flat = true_vel.flatten()
    point_vel_ref = point_vel - point_vel[0]
    true_vel_ref = true_vel_flat - true_vel_flat[0]
    correlation = np.corrcoef(point_vel_ref, true_vel_ref)[0, 1]
    assert correlation > 0.95


def test_emcf_with_link_model_ifg_input():
    """Test EMCF with link model using interferogram input."""
    n_sar, times, phase, _ = gen_data_with_velocity()

    # Set up time processing
    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)

    # Set up spatial processing
    g_space = spurt.graph.Reg2DGraph(phase.shape[1:])
    s_space = spurt.mcf.ORMCFSolver(g_space)

    # Create interferograms from SLC phases
    nifgs = len(g_time.links)
    ifg_phase = np.zeros((nifgs, *phase.shape[1:]))
    for ii, edge in enumerate(g_time.links):
        ifg_phase[ii] = phase[edge[1]] - phase[edge[0]]
    ifg = np.exp(1j * ifg_phase)

    # Build design matrix for velocity estimation
    amat = np.zeros((nifgs, 1))
    for ii, edge in enumerate(g_time.links):
        amat[ii, 0] = times[edge[1]] - times[edge[0]]

    # Create link model
    vel_range = slice(-0.5, 0.1, 0.02)
    link_model = spurt.links.GridSearchLinearModel(matrix=amat, ranges=(vel_range,))

    # Create EMCF solver with link model
    settings = spurt.workflows.emcf.SolverSettings(
        s_worker_count=1,
        t_worker_count=1,
    )
    solver = spurt.workflows.emcf.Solver(s_space, s_time, settings, link_model)

    w_data = spurt.io.Irreg3DInput(
        ifg.reshape((nifgs, g_space.npoints)), g_space.points
    )
    uw_data = solver.unwrap_cube(w_data)

    # Verify link parameters were estimated
    assert solver.link_params is not None
    assert solver.link_coherence is not None

    # Verify unwrapping succeeded
    for ii in range(nifgs):
        orig = ifg_phase[ii]
        recon = uw_data[ii].reshape(phase.shape[1:])
        assert np.allclose(orig - orig[0, 0], recon - recon[0, 0], atol=1.0e-3)


def test_link_model_not_provided():
    """Test that link_params is None when no link model is provided."""
    n_sar, _, phase, _ = gen_data_with_velocity()
    igram = np.exp(1j * phase)

    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)

    g_space = spurt.graph.Reg2DGraph(igram.shape[1:])
    s_space = spurt.mcf.ORMCFSolver(g_space)

    settings = spurt.workflows.emcf.SolverSettings(
        s_worker_count=1,
        t_worker_count=1,
    )
    solver = spurt.workflows.emcf.Solver(s_space, s_time, settings)

    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    solver.unwrap_cube(w_data)

    assert solver.link_params is None
    assert solver.link_coherence is None


def test_integrate_link_params_without_model():
    """Test that integrate_link_params raises error when no link model was used."""
    n_sar, _, phase, _ = gen_data_with_velocity()
    igram = np.exp(1j * phase)

    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)

    g_space = spurt.graph.Reg2DGraph(igram.shape[1:])
    s_space = spurt.mcf.ORMCFSolver(g_space)

    settings = spurt.workflows.emcf.SolverSettings(
        s_worker_count=1,
        t_worker_count=1,
    )
    solver = spurt.workflows.emcf.Solver(s_space, s_time, settings)

    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    solver.unwrap_cube(w_data)

    with pytest.raises(RuntimeError, match="No link parameters available"):
        solver.integrate_link_params()
