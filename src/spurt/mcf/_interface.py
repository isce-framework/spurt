"""Simple protocol for MCF solvers."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from numpy.typing import ArrayLike

from ..graph import PlanarGraphInterface

__all__ = [
    "MCFSolverInterface",
]


@runtime_checkable
class MCFSolverInterface(Protocol):
    """Interface for an MCF solver implementation.

    A solver can be initialized with a planar graph and
    can be repeatedly invoked with different input.
    """

    def __init__(self, graph: PlanarGraphInterface):
        """Initialize solver with the graph."""

    def unwrap_one(
        self, indata: ArrayLike, cost: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """Solver should return unwrapped phase and flows on the edges."""
