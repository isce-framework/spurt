"""Delaunay triangulation specific graph."""

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import Delaunay

from ._interface import GraphInterface

__all__ = [
    "order",
    "DelaunayGraph",
]


def order(p: tuple[int, int]) -> tuple[int, int]:
    """Order vertices by index.

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


class DelaunayGraph(GraphInterface):
    """Class to hold delaunay triangulation."""

    def __init__(self, xy: np.ndarray, scaling: ArrayLike = (1, 1)):
        """Create Delaunay triangulation with given coordinates and scaling."""
        self._d = Delaunay(xy * scaling)
        assert self._d.npoints == len(xy), "Number of points mismatch"

    @property
    def npoints(self) -> int:
        return self._d.npoints

    @property
    def points(self) -> np.ndarray:
        return self._d.points

    @property
    def simplices(self) -> np.ndarray:
        return self._d.simplices

    @property
    def boundary(self) -> np.ndarray:
        return self._d.convex_hull

    @property
    def links(self) -> np.ndarray:
        arcs: set[tuple[int, int]] = set()
        for s in self.simplices:
            arcs.add(order((s[0], s[1])))
            arcs.add(order((s[0], s[2])))
            arcs.add(order((s[1], s[2])))
        return np.array(sorted(arcs))
