"""Delaunay triangulation specific graph."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

from ._interface import PlanarGraphInterface

__all__ = [
    "order_points",
    "DelaunayGraph",
    "Reg2DGraph",
]


def order_points(p: tuple[int, int]) -> tuple[int, int]:
    """Order points/nodes/vertices by index.

    Given a pair of numbers, return a 2-tuple so the first is lower. The use
    case is that the pair contains pairs of indices representing undirected
    links, where (a, b) is the same as (b, a). This ordering, and returning
    a tuple allows us to comparison with ==.

    Parameters
    ----------
    p : array-like[2]
        Data to be ordered, must be comparable.

    Returns
    -------
    (int, int) : Ordered 2-tuple.
    """
    if p[0] <= p[1]:
        return (p[0], p[1])
    return (p[1], p[0])


class DelaunayGraph(PlanarGraphInterface):
    """Class to hold Delaunay triangulation.

    This will be the default class in use in this package.
    This will also be the only class that will interact with
    Minimum Cost Flow (MCF) solvers for irregular grids.
    """

    def __init__(self, xy: np.ndarray, scaling: ArrayLike = (1, 1)):
        """Create Delaunay triangulation with given coordinates and scaling."""
        self._d = Delaunay(xy * scaling)
        assert self._d.npoints == len(xy), "Number of points mismatch"
        self._links: np.ndarray | None = None
        self._create_links()

    @property
    def npoints(self) -> int:
        return self._d.npoints

    @property
    def points(self) -> np.ndarray:
        return self._d.points

    @property
    def cycles(self) -> np.ndarray:
        return self._d.simplices

    @property
    def boundary(self) -> np.ndarray:
        return self._d.convex_hull

    def _create_links(self) -> None:
        arcs: set[tuple[int, int]] = set()
        for s in self.cycles:
            arcs.add(order_points((s[0], s[1])))
            arcs.add(order_points((s[0], s[2])))
            arcs.add(order_points((s[1], s[2])))
        self._links = np.array(sorted(arcs))

    @property
    def links(self) -> np.ndarray:
        return self._links


class Reg2DGraph(PlanarGraphInterface):
    """Class to hold 2D regular grid.

    This is a utility class to test MCF implementation against other
    regular 2D grid solvers. We will walk down the rows of the 2D grid
    to determine the order of points, links and cycles.
    """

    def __init__(self, shape: tuple[int, int]):
        """Create regular 2D graph of given shape."""
        self._shape = shape
        self._links: np.ndarray | None = None
        self._create_links()

    @property
    def npoints(self) -> int:
        return self._shape[0] * self._shape[1]

    @property
    def points(self) -> np.ndarray:
        """Return points in 2D grid.

        Points are returned in row-major order.
        """
        # We will just use indices here for position
        i, j = np.indices(self._shape).reshape(2, -1)
        return np.column_stack((j, i))

    @property
    def cycles(self) -> np.ndarray:
        """Return rectangular loops in 2D grid."""
        ncyc = self.npoints - self._shape[0] - self._shape[1] + 1
        cyc = np.zeros((ncyc, 4), dtype=int)

        ind = 0
        # Iterate over the loops for top-left corner of loop
        for ii in range(self._shape[0] - 1):
            for jj in range(self._shape[1] - 1):
                # cycles will go counter-clockwise like Delaunay
                # top-left -> bottom-left -> bottom-right -> top-right
                cyc[ind, :] = np.ravel_multi_index(
                    ((ii, ii + 1, ii + 1, ii), (jj, jj, jj + 1, jj + 1)), self._shape
                )
                ind += 1

        return cyc

    @property
    def boundary(self) -> np.ndarray:
        """Return boundary edges in 2D grid."""
        narcs = 2 * (self._shape[0] + self._shape[1] - 2)
        arcs = np.zeros((narcs, 2), dtype=int)

        ind = 0
        # Walk along the top edge
        for ii in range(self._shape[1] - 1):
            arcs[ind, :] = (ii, ii + 1)
            ind += 1

        # Walk down the right edge
        for ii in range(self._shape[0] - 1):
            off = ii * self._shape[1] + self._shape[1] - 1
            arcs[ind, :] = (off, off + self._shape[1])
            ind += 1

        # Walk back along bottom edge
        for ii in range(self._shape[1] - 1, 0, -1):
            off = (self._shape[0] - 1) * self._shape[1]
            arcs[ind, :] = (off + ii - 1, off + ii)
            ind += 1

        # Walk back along left edge
        for ii in range(self._shape[0] - 1, 0, -1):
            off = ii * self._shape[1]
            arcs[ind, :] = (off - self._shape[1], off)
            ind += 1

        assert ind == narcs
        return arcs

    def _create_links(self) -> None:
        """Horizontal links followed by vertical links."""
        narcs = 2 * self.npoints - self._shape[0] - self._shape[1]
        arcs = np.zeros((narcs, 2), dtype=int)

        ind = 0
        # Start with horizontal links, iterate over the rows
        for ii in range(self._shape[0]):
            start = ii * self._shape[1] + np.arange(self._shape[1] - 1)
            arcs[ind : ind + start.size, 0] = start
            arcs[ind : ind + start.size, 1] = start + 1
            ind += start.size

        # Now include the vertical links, but walk along rows
        for ii in range(self._shape[0] - 1):
            start = ii * self._shape[1] + np.arange(self._shape[1])
            arcs[ind : ind + start.size, 0] = start
            arcs[ind : ind + start.size, 1] = start + self._shape[1]
            ind += start.size

        self._links = arcs

    @property
    def links(self) -> np.ndarray:
        return self._links
