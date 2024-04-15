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
    can be repeatedly invoked with different input. See
    the documentation of PlanarGraphInterface for restrictions
    on supported graphs.
    """

    def __init__(self, graph: PlanarGraphInterface):
        """Initialize solver with the graph."""

    def unwrap_one(
        self, wrapdata: ArrayLike, cost: ArrayLike, revcost: ArrayLike | None = None
    ) -> tuple[ArrayLike, ArrayLike]:
        """Solver should return unwrapped phase and flows on the edges.

        Parameters
        ----------
        wrapdata: ArrayLike
            1D array as input in radians or as complex numbers. Same size as
            number of points in the graph.
        cost: ArrayLike
            1D array of nonnegative integer costs. Same size as number of edges in
            the graph. Represents forward directional cost when used in
            combination with revcost.
        revcost: ArrayLike | None
            1D array of nonnegative integer costs on links in reverse direction.
            Same size as number of edges in the graph. cost is used if not
            provided.

        Returns
        -------
        unwdata: ArrayLike
            1D array of unwrapped phase in radians. Same size as number of
            points in the graph.
        flows: ArrayLike
            1D array of integer flows along each of the edges in the graph.
        """

    def residues_to_flows(
        self, residues: ArrayLike, cost: ArrayLike, revcost: ArrayLike | None = None
    ) -> ArrayLike:
        """Solver should return integer flows corresponding to given residues.

        Parameters
        ----------
        residues: ArrayLike
            1D array of integer residues. Same size as number of cycles in
            graph plus one to accommodate the grounding node. Array must sum to
            zero.
        cost: ArrayLike
            1D array of positive integer costs. Same size as number of edges
            in the graph. Represents forward directional cost when used in
            combination with revcost.
        revcost: ArrayLike | None
            1D array of positive integer costs on links in reverse direction.
            Same size as number of edges in the graph. cost is used if not
            provided.

        Returns
        -------
        flows: ArrayLike
            1D array of integer flows along each of the edges in the graph.
        """

    def compute_residues(self, wrapdata: ArrayLike) -> ArrayLike:
        """Solver should return residues corresponding to input wrapped data.

        Parameters
        ----------
        wrapdata: ArrayLike
            1D array as input in radians or complex numbers. Same size as
            number of points in the graph.

        Returns
        -------
        residues: ArrayLike
            1D array of integer residues corresponding to the cycles in the
            graph. Includes the grounding node.
        """
