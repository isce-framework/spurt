"""MCF solver implemented using ortools."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from ortools.graph.python import min_cost_flow

from ..graph import PlanarGraphInterface, order_points
from ._interface import MCFSolverInterface
from .utils import flood_fill, phase_diff, sign_nonzero

__all__ = ["ORMCFSolver"]


class ORMCFSolver(MCFSolverInterface):
    """Minimum cost flow solver interface.

    Implementation of MCF solver using OR tools python bindings.
    """

    def __init__(self, graph: PlanarGraphInterface):
        """Initialize solver using a planar graph.

        Initializes with information related to the dual graph as
        well to allow us to efficiently reuse the solver with
        multiple inputs
        """
        # We borrow the arrays and avoid copying
        self.npoints = graph.npoints
        self.edges = graph.links
        self.cycles = graph.cycles

        # These are needed for MCF
        # Edges represent arcs between cycles
        self._dual_edges: np.ndarray = np.zeros((len(self.edges), 2), dtype=np.int32)
        # One-to-one correspondence with _dual_edges and represents
        # relative orientation of an edge within a cycle
        # -1 implies increasing/ fwd direction, 1 implies decreasing/reverse
        # direction and zero denotes an edge to the grounding node
        self._dual_edge_dir: np.ndarray = np.zeros((len(self.edges), 2), dtype=np.int8)
        self._prepare_dual()

    def _prepare_dual(self) -> None:
        """Identify edges of the dual graph.

        Since, we build the problem in the dual space we will create edges
        in the dual space by iterating over cycles and it neighbors.
        """
        # Add a grounding node to go along with the residues
        # We use 1-index for other nodes as this allows
        # us to use sign information to indicate direction.
        # ground_node = 0

        # Iterate over the loops
        # Map edges to cycles they show up in
        edge_to_cycles: dict = {tuple(link): [] for link in self.edges}
        for icyc, cycle in enumerate(self.cycles):
            cycsize = len(cycle)
            for ii in range(cycsize):
                jj = (ii + 1) % cycsize
                edge = order_points((cycle[ii], cycle[jj]))

                # Sign indicates if the edge order in forward / reverse
                # direction. We augment icyc by 1 to use the sign infomation.
                edge_to_cycles[edge].append(
                    (icyc + 1, sign_nonzero(cycle[ii] - cycle[jj]))
                )

        # Now build list of dual_edges
        for ii, kk in enumerate(edge_to_cycles):
            vv = edge_to_cycles[kk]
            ncyc = len(vv)

            # TODO: Use switch-case here if Python version > 3.10
            if ncyc == 2:
                self._dual_edges[ii, :] = [vv[0][0], vv[1][0]]
                self._dual_edge_dir[ii, :] = [vv[0][1], vv[1][1]]
            elif ncyc == 1:
                # Arrays are already initialized to zero
                self._dual_edges[ii, 0] = vv[0][0]
                self._dual_edge_dir[ii, 0] = vv[0][1]
            else:
                errmsg = (
                    "Planar graph contains edges that are part of more than 2 cycles"
                )
                raise ValueError(errmsg)

    @property
    def dual_edges(self) -> np.ndarray:
        return self._dual_edges

    def compute_residues(
        self,
        wrapdata: ArrayLike,
    ) -> ArrayLike:
        """Compute phase residues for one set of input wrapped data."""
        if wrapdata.size != self.npoints:
            errmsg = (
                f"Size mismatch for unwrapping."
                f" Received {wrapdata.shape} with {self.npoints} points"
            )
            raise ValueError(errmsg)

        if revcost is None:
            revcost = cost

        # Residues includes the grounding node at index 0
        residues = np.zeros(len(self.cycles) + 1)
        ndim = len(self.cycles[0])
        for col in range(ndim):
            nn = (col + 1) % ndim
            residues[1:] += phase_diff(
                wrapdata[self.cycles[:, col]], wrapdata[self.cycles[:, nn]]
            )
        residues = np.rint(residues / (2 * np.pi))
        # Set supply of ground_node
        residues[0] = -np.sum(residues[1:])
        return residues

    def unwrap_one(
        self,
        wrapdata: ArrayLike,
        cost: ArrayLike,
        revcost: ArrayLike | None = None,
    ) -> tuple[ArrayLike, ArrayLike]:
        """Get the unwrapped phase solution for one set of input wrapped data."""
        if revcost is None:
            revcost = cost

        # Compute residues
        residues = self.compute_residues(wrapdata)

        # Instantiate the MCF solver and supply the data
        flows = self.residues_to_flows(residues, cost, revcost=revcost)

        # Flood fill with the flows
        unw = flood_fill(wrapdata, self.edges, flows)

        return unw, flows

    def residues_to_flows(
        self,
        residues: np.ndarray,
        cost: np.ndarray,
        revcost: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return flows on the edges corresponding to given set of residues.

        This is exposed to allow for unwrapping with gradients.
        """
        return solve_mcf(self._dual_edges, self._dual_edge_dir, residues, cost, revcost)


def solve_mcf(
    edges: np.ndarray,
    edge_dir: np.ndarray,
    residues: np.ndarray,
    cost: np.ndarray,
    revcost: np.ndarray,
) -> np.ndarray:
    """Solve a single mcf problem.

    Parameters
    ----------
    edges: np.ndarray
        Arcs connecting the supplies, i.e possible cuts in the graph connecting
        residues.
    edge_dir: np.ndarray
        Orientation of edge being cut in the source cycles.
    residues: np.ndarray
        Integer 1D array of residues including grounding node.
    cost: np.ndarray
        Integer cost of unit flow on edge being cut in increasing index /
        forward direction.
    revcost: np.ndarray
        Integer cost of unit flow on edge being cut in decreasing index /
        reverse direction.

    """
    num_edges = len(edges)
    int32_max = np.iinfo(np.int32).max
    start_nodes = edges[:, 0]
    end_nodes = edges[:, 1]

    # Ignoring second_cycle as it is guaranteed to be non-zero
    first_cycle_dir = edge_dir[:, 0]

    # TODO: Make this configurable at some point
    # Snaphu uses a limit for this parameter
    capacities = np.full((num_edges), int32_max, dtype=int)

    if np.sum(residues) != 0:
        errmsg = "MCF unbalanced. Sum of residues in non-zero."
        raise ValueError(errmsg)

    # Set up the solver
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Add arcs in increasing index / forward direction
    arc_cost = cost * (first_cycle_dir == -1) + revcost * (first_cycle_dir == 1)
    smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, arc_cost
    )

    # Add arcs in decreasing index / reverse direction
    arc_cost = cost * (first_cycle_dir == 1) + revcost * (first_cycle_dir == -1)
    smcf.add_arcs_with_capacity_and_unit_cost(
        end_nodes, start_nodes, capacities, arc_cost
    )

    # Add the supplies
    smcf.set_nodes_supplies(np.arange(len(residues), dtype=np.int32), residues)

    if smcf.solve() != smcf.OPTIMAL:
        errmsg = "MCF solver returned a sub-optimal solution."
        raise RuntimeError(errmsg)

    flows = np.zeros(num_edges, dtype=int)
    for ii in range(num_edges):
        # Sign accounts for orientation of edge in cycles
        flows[ii] = first_cycle_dir[ii] * (smcf.flow(ii) - smcf.flow(ii + num_edges))

    return flows
