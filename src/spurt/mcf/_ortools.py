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
        """Initialize solver using a planar graph."""
        # We borrow the arrays and avoid copying
        self.npoints = graph.npoints
        self.edges = graph.links
        self.cycles = np.array(graph.cycles)  # To make mypy happy for now

        # These are needed for MCF
        # Edges represent arcs between cycles
        self.dual_edges: np.ndarray = np.zeros((len(self.edges), 2), dtype=np.int32)
        self._prepare_dual()

    def _prepare_dual(self) -> None:
        """Identify edges of the dual graph.

        Since, we build the problem in the dual space we will create edges
        in the dual space by iterating over cycles and it neighbors.
        """
        # Add a grounding node to go along with the residues
        # We use 1-index for other nodes as this allows
        # us to use sign information to indicate direction.
        ground_node = 0

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
                    (icyc + 1) * sign_nonzero(cycle[ii] - cycle[jj])
                )

        # Now build list of dual_edges
        for ii, kk in enumerate(edge_to_cycles):
            vv = edge_to_cycles[kk]
            ncyc = len(vv)

            # TODO: Use switch-case here if Python version > 3.10
            if ncyc == 2:
                self.dual_edges[ii, :] = vv
            elif ncyc == 1:
                self.dual_edges[ii, :] = [vv[0], ground_node]
            else:
                errmsg = (
                    "Planar graph contains edges that are part of more than 2 cycles"
                )
                raise ValueError(errmsg)

    def unwrap_one(
        self, indata: ArrayLike, cost: ArrayLike
    ) -> tuple[ArrayLike, ArrayLike]:
        """Get integer flow solutions for one set of inputs."""
        if indata.size != self.npoints:
            errmsg = (
                f"Size mismatch for unwrapping."
                f" Received {indata.shape} with {self.npoints} points"
            )
            raise ValueError(errmsg)

        # Residues includes the grounding node at index 0
        residues = np.zeros(len(self.cycles) + 1)
        ndim = self.cycles.shape[1]
        for col in range(ndim):
            nn = (col + 1) % ndim
            residues[1:] += phase_diff(
                indata[self.cycles[:, col]], indata[self.cycles[:, nn]]
            )
        residues = np.rint(residues / (2 * np.pi))
        # Set supply of ground_node
        residues[0] = -np.sum(residues[1:])

        # Instantiate the MCF solver and supply the data
        flows = self.residues_to_flows(residues, cost)

        # Flood fill with the flows
        unw = flood_fill(indata, self.edges, flows)

        return unw, flows

    def residues_to_flows(self, residues: np.ndarray, cost: np.ndarray) -> np.ndarray:
        """Return flows corresponding to given set of residues.

        This is exposed to allow for unwrapping with gradients.
        """
        return solve_mcf(self.dual_edges, residues, cost)


def solve_mcf(edges: np.ndarray, residues: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Solve a single mcf problem."""
    num_edges = len(edges)
    start_nodes = np.abs(edges[:, 0])
    end_nodes = np.abs(edges[:, 1])

    # TODO: Make this configurable at some point
    capacities = np.full((num_edges), 20, dtype=int)

    if np.sum(residues) != 0:
        errmsg = "MCF unbalanced. Sum of residues in non-zero."
        raise ValueError(errmsg)

    # Set up the solver
    smcf = min_cost_flow.SimpleMinCostFlow()

    # Add arcs one way
    # TODO: check if capacities and costs can be scalar
    smcf.add_arcs_with_capacity_and_unit_cost(start_nodes, end_nodes, capacities, cost)

    # Add arcs the other way
    smcf.add_arcs_with_capacity_and_unit_cost(end_nodes, start_nodes, capacities, cost)

    # Add the supplies
    smcf.set_nodes_supplies(np.arange(len(residues), dtype=np.int32), residues)

    if smcf.solve() != smcf.OPTIMAL:
        errmsg = "MCF solver returned a sub-optimal solution."
        raise RuntimeError(errmsg)

    flows = np.zeros(num_edges, dtype=int)
    for ii in range(num_edges):
        # Sign accounts for orientation of edge in cycles
        flows[ii] = sign_nonzero(edges[ii, 0]) * (
            smcf.flow(ii) - smcf.flow(ii + num_edges)
        )

    return flows
