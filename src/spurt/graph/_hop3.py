"""Build hop-3 planar graphs."""

import numpy as np

from ._interface import PlanarGraphInterface
from .utils import order_points

__all__ = [
    "Hop3Graph",
]


class Hop3Graph(PlanarGraphInterface):
    """Class to hold Hop-3 planar graph.

    This will be a useful class for missions like Sentinel-1
    with narrow baseline tube and when one wants to restrict the
    interferogram network based on temporal baseline alone.
    """

    def __init__(self, npoints: int):
        """Create triangulation for N points and 3 hops."""
        if npoints < 4:
            errmsg = (
                f"Atleast 4 points are needed to build a 3-hop graph. Got {npoints}"
            )
            raise ValueError(errmsg)

        # Create the points as a colinear set
        self._xy = np.zeros((npoints, 2), dtype=int)
        self._xy[:, 0] = np.arange(npoints, dtype=int)

        # Initialize to none
        self._cycles: np.ndarray | None = None
        self._links: np.ndarray | None = None
        self._create_cycles_and_links()

    @property
    def npoints(self) -> int:
        return self._xy.shape[0]

    @property
    def points(self) -> np.ndarray:
        return self._xy

    @property
    def cycles(self) -> np.ndarray:
        return self._cycles

    @property
    def boundary(self) -> np.ndarray:
        return np.ndarray([[0, 1], [1, 3], [3, 0]])

    def _create_cycles_and_links(self) -> None:
        arcs: set[tuple[int, int]] = set()

        # Set size of cycles
        ncycles = 3 + (self.npoints - 4) * 2
        nlinks = 3 * self.npoints - 6
        self._cycles = np.zeros((ncycles, 3), dtype=int)

        # Initialization for first point is fixed
        self._cycles[0, :] = [0, 1, 2]
        self._cycles[1, :] = [0, 2, 3]

        for ii in range(1, self.npoints - 3):
            self._cycles[2 * ii, :] = [ii, ii + 2, ii + 3]
            self._cycles[2 * ii + 1, :] = [ii, ii + 3, ii + 1]

        # Last cycle is fixed
        self._cycles[-1, :] = [self.npoints - 3, self.npoints - 1, self.npoints - 2]

        # Set the links
        for s in self.cycles:
            arcs.add(order_points((s[0], s[1])))
            arcs.add(order_points((s[0], s[2])))
            arcs.add(order_points((s[1], s[2])))
        self._links = np.array(sorted(arcs))
        assert self._links.shape[0] == nlinks, "Not all links captured"

    @property
    def links(self) -> np.ndarray:
        return self._links
