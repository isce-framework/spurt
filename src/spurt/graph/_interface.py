"""Protocol for graphs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "GraphInterface",
]


@runtime_checkable
class GraphInterface(Protocol):
    """
    Interface to 2D graph objects.

    Such objects should provide access to vertices, arcs and convex hull.
    """

    @property
    def points(self) -> np.ndarray:
        """Return vertices as (npts, ndim)."""

    @property
    def simplices(self) -> np.ndarray | list[list[int]]:
        """Return simplices as (nsimplex, ...)."""

    @property
    def boundary(self) -> np.ndarray | list[list[int]]:
        """Return list of arcs forming outer boundary of graph."""

    @property
    def links(self) -> np.ndarray:
        """Return list of links in the graph as (nlinks, 2)."""
