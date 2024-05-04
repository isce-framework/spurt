import numpy as np

import spurt


def gen_data_real(add_eq: bool | None = None):
    """Generate a regular 3D dataset"""

    if add_eq is None:
        add_eq = False

    n_collects = 20
    y, x = np.ogrid[-3:3:64j, -3:3:64j]

    vel = -np.pi * np.exp(-(x**2 + y**2) / 5) / 12
    vel -= vel.max()
    times = np.arange(n_collects) * 12
    phase = times[:, None, None] * vel[None, :, :]

    if add_eq:
        # Distance from diagonal
        dist = (x + y) / np.sqrt(2)

        # Fault that almost breaks the surface
        disp = 20 * (1 - np.arctan2(dist, 0.5))

        # Spatial weighting to contain fault within image
        disc = x**2 + y**2
        disp *= 1 - disc / disc.max()

        # Add eq to phase
        phase[10:, :, :] += disp

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
    solver = spurt.workflows.emcf.Solver(s_space, s_time)
    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    uw_data = solver.unwrap_cube(w_data)

    for ii, edge in enumerate(g_time.links):
        orig = phase[edge[1]] - phase[edge[0]]
        recon = uw_data[ii].reshape(phase.shape[1:])

        assert np.allclose(orig, recon, atol=1.0e-3)


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
    solver.settings.worker_count = 1

    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    uw_data = solver.unwrap_cube(w_data)

    for ii, edge in enumerate(g_time.links):
        orig = phase[edge[1]] - phase[edge[0]]
        recon = uw_data[ii].reshape(phase.shape[1:])

        assert np.allclose(orig - orig[0, 0], recon - recon[0, 0], atol=1.0e-3)


def test_emcf_eq():
    n_sar, phase = gen_data_real(add_eq=True)
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

    w_data = spurt.io.Irreg3DInput(
        igram.reshape((n_sar, g_space.npoints)), g_space.points
    )
    uw_data = solver.unwrap_cube(w_data)

    for ii, edge in enumerate(g_time.links):
        orig = phase[edge[1]] - phase[edge[0]]
        recon = uw_data[ii].reshape(phase.shape[1:])

        assert np.allclose(orig - orig[0, 0], recon - recon[0, 0], atol=1.0e-3)
