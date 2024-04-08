from __future__ import annotations

import numpy as np

import spurt


class Test3D:
    def test_quad(self):

        xy = np.array([[0.0, 0.0], [0.0, 1.0], [1.2, 1.1], [1.0, 0.0]])
        g = spurt.graph.DelaunayGraph(xy)

        assert g.npoints == xy.shape[0]
        assert np.array_equal(g.cycles, np.array([[3, 1, 0], [1, 3, 2]], dtype=int))
        assert np.array_equal(
            g.links, np.array(([(0, 1), (0, 3), (1, 2), (1, 3), (2, 3)]), dtype=int)
        )

    def test_reg2d(self):
        shape = (4, 5)
        g = spurt.graph.Reg2DGraph(shape)

        assert g.npoints == np.prod(shape)
        assert g.boundary.shape == (2 * (shape[0] + shape[1] - 2), 2)
        assert g.cycles.shape == (g.npoints - shape[0] - shape[1] + 1, 4)
        assert g.links.shape == (2 * g.npoints - shape[0] - shape[1], 2)
