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


    Attributes
    ----------
    npoints: int
        Number of points in primal graph.
    nedges: int
        Number of edges in primal graph.
    ncycles: int
        Number of cycles in primal graph.
    cycle_length: int
        Length of cycles in primal graph.
    points: ArrayLike
        2D array of size (npoints, 2) with coordinates.
    edges: ArrayLike
        2D array of size (nedges, 2) where values indicate index into points
        array. Edges always go from lower index to higher index.
    cycles: ArrayLike
        2D array of size (ncycles, cycle_length) where values indicate index
        points array. All cycles are oriented in one direction.
    dual_edges: ArrayLike
        2D array of size (nedges, 2) where values indicate 1-index into cycles
        array. zero indicates a boundary edge in primal graph.
    dual_edge_dir: ArrayLike
        2D array of size (nedges, 2) where values orientation of edge into
        1-indexed cycles array. zero indicates a boundary edge in primal
        graph.
    """

    def __init__(self, graph: PlanarGraphInterface):
        """Initialize solver with the graph."""

    @property
    def npoints(self) -> int:
        """Number of points in primal graph."""

    @property
    def nedges(self) -> int:
        """Number of links in primal graph."""

    @property
    def ncycles(self) -> int:
        """Number of cycles in primal graph."""

    @property
    def cycle_length(self) -> int:
        """Length of cycles in primal graph."""

    @property
    def points(self) -> int:
        """Points in primal graph."""

    @property
    def edges(self) -> ArrayLike:
        """Edges in primal graph."""

    @property
    def cycles(self) -> ArrayLike:
        """Cycles in primal graph."""

    @property
    def dual_edges(self) -> ArrayLike:
        """Returns dual edges of the graph.

        This is of size (nedges, 2)
        """

    @property
    def dual_edge_dir(self) -> ArrayLike:
        """Array containing orientation of edge in cycle.

        This is of size (nedges, 2).
        """

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
            1D array of nonnegative integer costs. Same size as number of edges
            in the graph. Represents forward directional cost when used in
            combination with revcost.
        revcost: ArrayLike | None
            1D array of nonnegative integer costs on links in reverse direction.
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

    def compute_residues_from_gradients(self, graddata: ArrayLike) -> ArrayLike:
        """Solver should return residues corresponding to edge gradients.

        Parameters
        ----------
        graddata: ArrayLike
            1D array as input in radians. Same size as number of edges in the
            primal graph.

        Returns
        -------
        residues: ArrayLike
            1D array of integer residues corresponding to the cycles in the
            graph. Includes the grounding node.
        """

    def residues_to_flows_many(
        self,
        residues: ArrayLike,
        cost: ArrayLike,
        revcost: ArrayLike | None = None,
        worker_count: int | None = None,
        chunksize: int | None = 1,
    ) -> ArrayLike:
        """Solver should return integer flows corresponding to given residues.

        Parameters
        ----------
        residues: ArrayLike
            2D array of integer residues of size (nruns, ncycles + 1). Each row
            of the array must sum to zero.
        cost: ArrayLike
            1D array of nonnegative integer costs. Same size as number of edges
            in the graph. Represents forward directional cost when used in
            combination with revcost.
        revcost: ArrayLike | None
            1D array of nonnegative integer costs on links in reverse direction.
            Same size as number of edges in the graph. cost is used if not
            provided.

        Returns
        -------
        flows: ArrayLike
            2D array of integer flows of along each of the edges in the graph.
            Returned array is of size (nruns, nedges).
        """
