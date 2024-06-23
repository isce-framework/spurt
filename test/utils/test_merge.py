import numpy as np
from scipy.sparse import csr_matrix

import spurt


def test_l2_min_dense():
    g = spurt.graph.Hop3Graph(10)
    edges = g.links
    b = np.random.randn(g.npoints)
    amat = np.zeros((len(edges), g.npoints))
    diff = b[edges[:, 1]] - b[edges[:, 0]]

    for ii, ee in enumerate(edges):
        amat[ii, ee[0]] = -1
        amat[ii, ee[1]] = 1

    x, _ = spurt.utils.merge.l2_min(amat, diff)

    assert np.allclose(b - b[0], x - x[0], atol=1.0e-6)


def test_l2_min_sparse():
    g = spurt.graph.Hop3Graph(10)
    edges = g.links
    b = np.random.randn(g.npoints)
    diff = b[edges[:, 1]] - b[edges[:, 0]]

    data = np.ones(2 * len(edges), dtype=int)
    data[0::2] = -1
    amat = csr_matrix(
        (data, (np.arange(2 * len(edges), dtype=int) // 2, edges.flatten()))
    )

    x, _ = spurt.utils.merge.l2_min(amat, diff)
    assert np.allclose(b - b[0], x - x[0], atol=1.0e-6)


def test_common_points():
    img = np.random.rand(100, 100)

    # tile [0, 0, 70, 70]
    pts0 = np.column_stack(np.where(img[:70, :70] < 0.3))
    # tile [40, 40, 100, 100]
    pts1 = np.column_stack(np.where(img[40:, 40:] < 0.3))
    pts1 += 40

    ind0, ind1 = spurt.utils.merge.find_common_points(pts0, pts1)

    assert np.array_equal(pts0[ind0], pts1[ind1])
