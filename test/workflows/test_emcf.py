import numpy as np

import spurt


def gen_data_real():
    """Generate a regular 3D dataset"""

    n_collects = 20
    y, x = np.ogrid[-3:3:64j, -3:3:64j]

    vel = -np.pi * np.exp(-(x**2 + y**2) / 5) / 12
    vel -= vel.max()
    times = np.arange(n_collects) * 12
    phase = times[:, None, None] * vel[None, :, :]
    return n_collects, phase


def gen_data_snaphu():
    """Generate a regular 3D dataset.

    Derived from snaphu unit test."""

    n_collects = 10
    y, x = np.ogrid[-3:3:64j, -3:3:64j]

    vel = np.pi * (x + y) / 12
    times = np.arange(n_collects) * 12
    phase = times[:, None, None] * vel[None, :, :]
    return n_collects, phase


def test_emcf():
    n_sar, phase = gen_data_real()
    igram = np.exp(1j * phase)

    # Set up time processing
    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)

    # Set up spatial processing
    g_space = spurt.graph.Reg2DGraph(igram.shape[1:])
    s_space = spurt.mcf.ORMCFSolver(g_space)

    # Create EMCF solver
    settings = spurt.workflows.emcf.SolverSettings()
    solver = spurt.workflows.emcf.Solver(s_space, s_time, settings)
    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    uw_data = solver.unwrap_cube(w_data)

    for ii, edge in enumerate(g_time.links):
        orig = phase[edge[1]] - phase[edge[0]]
        recon = uw_data[ii].reshape(phase.shape[1:])

        assert np.allclose(orig, recon, 1.0e-3)


def test_emcf_ramp():
    n_sar, phase = gen_data_snaphu()
    igram = np.exp(1j * phase)

    # Set up time processing
    g_time = spurt.graph.Hop3Graph(n_sar)
    s_time = spurt.mcf.ORMCFSolver(g_time)

    # Set up spatial processing
    g_space = spurt.graph.Reg2DGraph(igram.shape[1:])
    s_space = spurt.mcf.ORMCFSolver(g_space)

    # Create EMCF solver
    solver = spurt.workflows.emcf.Solver(s_space, s_time)
    # Test this setting as well
    solver.worker_count = 1

    uw_data = solver.unwrap_cube(igram.reshape((n_sar, g_space.npoints)))

    for ii, edge in enumerate(g_time.links):
        orig = phase[edge[1]] - phase[edge[0]]
        recon = uw_data[ii].reshape(phase.shape[1:])

        assert np.allclose(orig - orig[0, 0], recon - recon[0, 0], 1.0e-3)
