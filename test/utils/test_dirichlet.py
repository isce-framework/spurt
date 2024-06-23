from __future__ import annotations

import numpy as np

import spurt


def test_dirichlet():

    # Create some random points and per-point data.
    pts = np.random.randn(4000).reshape((2000, 2))

    cutoff = -1

    # Choose points with an x-value below cutoff to be the fixed (boundary)
    # values. Choose 1 for positive y-values and -1 for negative y-values.
    mask = pts[:, 0] < cutoff

    pt_data = np.zeros(pts.shape[0])
    pt_data[pts[0, 1] >= 0] = 1.0
    pt_data[pts[0, 1] < 0] = -1.0

    # Form the link from a Delaunay triangulation.
    graph = spurt.graph.DelaunayGraph(pts)

    # Form a discrete Laplacian
    lap = spurt.graph.graph_laplacian(graph)

    # Solve the Dirichlet problem.
    x, _ = spurt.utils.merge.dirichlet(lap, np.zeros(lap.shape[0]), pt_data, mask)

    # x and pt_data should agree on the mask.
    assert np.all(x[mask] == pt_data[mask])

    # There should be no extrema outside the mask.
    assert np.max(x[mask]) <= np.max(pt_data[mask])
    assert np.min(x[mask]) >= np.min(pt_data[mask])
