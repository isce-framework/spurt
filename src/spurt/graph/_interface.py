"""Protocol for graphs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "GraphInterface",
    "PlanarGraphInterface",
]


@runtime_checkable
class GraphInterface(Protocol):
    """
    Interface to 2D graph objects.

    Such objects should provide access to points (nodes or vertices),
    links (arcs) and cycles (simplices).

    Attributes
    ----------
    npoints: int
        Number of points/ vertices/ nodes.
    points: np.ndarray
        (npoints, 2) array corresponding to locations of the points.
    cycles: list[list[int]]
        Each entry in the list represents one cycle in the graph.
    links: np.ndarray
        (nlinks, 2) array. Each row contains indices into the points array.
    """

    @property
    def npoints(self) -> int:
        """Return number of points in graph."""

    @property
    def points(self) -> np.ndarray:
        """Return points/ vertices/ nodes as (npts, 2)."""

    @property
    def cycles(self) -> np.ndarray | list[list[int]]:
        """Return cycles/ simplices as (nsimplex, ...)."""

    @property
    def links(self) -> np.ndarray:
        """Return list of links/ arcs in the graph as (nlinks, 2)."""


@runtime_checkable
class PlanarGraphInterface(GraphInterface, Protocol):
    """
    Interface to a 2D planar graph.

    This provides an additional access to the boundary (convex_hull) of the
    graph. This class is specifically meant for use with MCF solvers and will
    provide a common interface for Regular 2D grids and Delaunay
    triangulations. The following assumptions are made regarding our interface
    for ease of use with MCF solvers:
        - The graph is connected.
        - Every edge is a part of one or two cycles utmost.
        - The links are always oriented from lower index into the points array
        to the higher index.
        - The cycles are all oriented in a single direction - either clockwise
        or anti-clockwise.
        - All the cycles returned by the interface are of the same length.
    """

    @property
    def cycles(self) -> np.ndarray:
        """Return cycles as ndarray as cycles are of same length."""

    @property
    def boundary(self) -> np.ndarray | list[list[int]]:
        """Return arcs in form of pair of vertices forming outer boundary of graph.

        This is only a utility and is not necessarily used in the MCF code.
        There is no concept of boundary for non-planar graphs.
        """
