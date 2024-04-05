import numpy as np

import spurt

# Fix the seed for repeatability
np.random.seed(32)


def wrap(x):
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


def gen_data_real():
    """Generate a sparse 2D dataset"""
    npoints = 4000

    # Coordinates of points.
    points = np.random.randn(2 * npoints).reshape((npoints, 2))

    # Data at points -- random with a ramp in x.
    point_data = 0.5 * np.random.randn(npoints)
    c = 8.0 * np.pi / np.ptp(points[:, 0])
    point_data += c * points[:, 0]

    # Connections.
    graph = spurt.graph.DelaunayGraph(points)

    return graph, point_data


def test_flood_fill():
    """Test the flood fill unwrapping implementation."""

    graph, point_data = gen_data_real()
    edges = graph.links

    # The generated data is truth and flows are derived from it
    # Flows are fed back to ensure that flood fill unwraps it correctly
    flows = np.rint((point_data[edges[:, 1]] - point_data[edges[:, 0]]) / (2 * np.pi))

    point_data1 = spurt.mcf.utils.flood_fill(point_data, edges, flows)

    print(point_data.min(), point_data.max())
    assert not np.allclose(np.ptp(point_data), 0.0)
    assert np.allclose(np.ptp(point_data - point_data1), 0.0)


def test_unwrap_one():
    """Test basic 2D unwrapping."""

    graph, point_data = gen_data_real()

    # Hack for now to reference to first pixel
    point_data = point_data - point_data[0]

    edges = graph.links
    solver = spurt.mcf.ORMCFSolver(graph)
    uwdata, _ = solver.unwrap_one(point_data, np.ones(edges.shape[0], dtype=int))

    grads = point_data[edges[:, 1]] - point_data[edges[:, 0]]
    ugrads = uwdata[edges[:, 1]] - uwdata[edges[:, 0]]
    diff = ugrads - grads

    print(diff.min(), diff.max())
    assert np.max(np.abs(diff)) < 1

    assert np.ptp(wrap(diff)) < 1.0e-3


def test_snaphu_data():
    """Borrow this unit test from snaphu."""

    # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
    y, x = np.ogrid[-3:3:512j, -3:3:512j]
    phase = np.pi * (x + y)

    # Hack for now to reference to first pixel
    phase = phase - phase[0, 0]

    igram = np.exp(1j * phase)

    graph = spurt.graph.Reg2DGraph(igram.shape)
    solver = spurt.mcf.ORMCFSolver(graph)
    # Use unit cost - since we dont have smooth / defo
    cost = np.ones(solver.edges.shape[0], dtype=int)
    unw, _ = solver.unwrap_one(igram.flatten(), cost)
    unw = unw.reshape(igram.shape)

    # The unwrapped phase may differ from the true phase by a fixed integer
    # multiple of 2pi.
    mean_diff = np.mean(unw - phase)
    offset = 2.0 * np.pi * np.round(mean_diff / (2.0 * np.pi))
    np.testing.assert_allclose(unw, phase + offset, atol=1e-3)
