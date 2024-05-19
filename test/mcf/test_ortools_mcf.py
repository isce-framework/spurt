import numpy as np

import spurt

# Fix the seed for repeatability
np.random.seed(32)


def wrap(x):
    return x - 2.0 * np.pi * np.round(x / (2.0 * np.pi))


def gen_data_real():
    """Generate a sparse 2D dataset"""
    npoints = 1000

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

    point_data1 = spurt.mcf.utils.flood_fill(point_data, edges, flows, mode="points")

    assert not np.allclose(np.ptp(point_data), 0.0)
    assert np.allclose(np.ptp(point_data - point_data1), 0.0)


def test_flood_fill_gradients():
    """Test the flood fill implementation with gradients."""

    graph, point_data = gen_data_real()
    edges = graph.links

    # The generated data is truth and flows are derived from it
    # Flows are fed back to ensure that flood fill unwraps it correctly
    grads = spurt.mcf.utils.phase_diff(point_data[edges[:, 0]], point_data[edges[:, 1]])
    flows = np.rint((point_data[edges[:, 1]] - point_data[edges[:, 0]]) / (2 * np.pi))

    point_data1 = spurt.mcf.utils.flood_fill(grads, edges, flows, mode="gradients")

    assert not np.allclose(np.ptp(point_data), 0.0)
    assert np.allclose(np.ptp(point_data - point_data1), 0.0)


def test_unwrap_one():
    """Test basic 2D unwrapping."""

    graph, point_data = gen_data_real()

    edges = graph.links
    solver = spurt.mcf.ORMCFSolver(graph)

    # Based on current internal unit test
    # Setting up cost function based on centroid distance
    cost = spurt.mcf.utils.centroid_costs(
        graph.points, solver.cycles, solver.dual_edges
    )
    uwdata, _ = solver.unwrap_one(point_data, cost)

    grads = point_data[edges[:, 1]] - point_data[edges[:, 0]]
    ugrads = uwdata[edges[:, 1]] - uwdata[edges[:, 0]]
    diff = ugrads - grads

    assert solver.npoints == point_data.shape[0]
    assert np.max(np.abs(diff)) > 1
    assert np.ptp(wrap(diff)) < 1.0e-3


def test_snaphu_data():
    """Borrow this unit test data from snaphu."""

    # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)

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


def test_snaphu_sparse():
    """Borrow this unit test data from snaphu."""

    # Simulate interferogram containing a diagonal phase ramp with multiple fringes.
    y, x = np.ogrid[-3:3:256j, -3:3:256j]
    phase = np.pi * (x + y)

    igram = np.exp(1j * phase)

    graph = spurt.graph.DelaunayGraph(spurt.graph.Reg2DGraph(igram.shape).points)
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


def test_unwrap_many():
    """Test basic 2D unwrapping."""

    graph, point_data = gen_data_real()

    solver = spurt.mcf.ORMCFSolver(graph)

    # Based on current internal unit test
    # Setting up cost function based on centroid distance
    cost = spurt.mcf.utils.centroid_costs(
        graph.points, solver.cycles, solver.dual_edges
    )
    resid = solver.compute_residues(point_data)

    ncopies = 20
    nworkers = 4

    residues = np.zeros((ncopies, len(resid)), dtype=int)
    residues[:, :] = resid[None, :]

    flows = solver.residues_to_flows_many(residues, cost, worker_count=nworkers)

    assert np.ptp(np.ptp(flows, axis=0)) == 0


def test_unwrap_many_oneworker():
    """Test basic 2D unwrapping."""

    graph, point_data = gen_data_real()

    solver = spurt.mcf.ORMCFSolver(graph)

    # Based on current internal unit test
    # Setting up cost function based on centroid distance
    cost = spurt.mcf.utils.centroid_costs(
        graph.points, solver.cycles, solver.dual_edges
    )
    resid = solver.compute_residues(point_data)

    ncopies = 4
    nworkers = 1

    residues = np.zeros((ncopies, len(resid)), dtype=int)
    residues[:, :] = resid[None, :]

    flows = solver.residues_to_flows_many(residues, cost, worker_count=nworkers)

    assert np.ptp(np.ptp(flows, axis=0)) == 0


def test_grad_residues():
    """Test residue computation using points and gradients."""
    graph, point_data = gen_data_real()
    solver = spurt.mcf.ORMCFSolver(graph)
    grads = point_data[solver.edges[:, 1]] - point_data[solver.edges[:, 0]]
    grads_resid = solver.compute_residues_from_gradients(grads)

    assert grads_resid.min() == 0
    assert grads_resid.max() == 0


def test_residues():
    """Test residue computation using points and gradients."""
    graph, point_data = gen_data_real()
    solver = spurt.mcf.ORMCFSolver(graph)
    grads = spurt.mcf.utils.phase_diff(
        point_data[solver.edges[:, 0]], point_data[solver.edges[:, 1]]
    )
    pts_resid = solver.compute_residues(point_data)
    grads_resid = solver.compute_residues_from_gradients(grads)
    np.testing.assert_array_equal(pts_resid, grads_resid)
