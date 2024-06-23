from __future__ import annotations

import numpy as np

import spurt


class TestHop3:
    def test_hop3(self):
        npoints = 5
        xy = np.array(
            [[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
            dtype=int,
        )

        g = spurt.graph.Hop3Graph(npoints)

        assert g.npoints == npoints
        assert np.array_equal(g.links, xy)
        assert len(g.cycles) == 5

        # Euler's formula
        assert (g.npoints - len(g.links) + len(g.cycles) + 1) == 2
        assert np.array_equal(g.boundary, np.array([[0, 1], [1, 3], [3, 0]]))
