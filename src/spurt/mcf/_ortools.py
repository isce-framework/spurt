"""MCF solver implemented using ortools."""

from __future__ import annotations

from multiprocessing import get_context

import numpy as np
from numpy.typing import ArrayLike
from ortools.graph.python import min_cost_flow

from ..graph import PlanarGraphInterface, order_points
from ..utils import get_cpu_count
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
        self._graph: PlanarGraphInterface = graph

        # These are needed for MCF
        # Edges represent arcs between cycles
        self._dual_edges: np.ndarray = np.zeros((self.nedges, 2), dtype=np.int32)
        # One-to-one correspondence with _dual_edges and represents
        # relative orientation of an edge within a cycle
        # 1 implies increasing/ fwd direction, -1 implies decreasing/reverse
        # direction and zero denotes an edge to the grounding node
        self._dual_edge_dir: np.ndarray = np.zeros((self.nedges, 2), dtype=np.int8)
        self._prepare_dual()

    @property
    def npoints(self) -> int:
        return self._graph.npoints

    @property
    def nedges(self) -> int:
        return len(self._graph.links)

    @property
    def ncycles(self) -> int:
        return len(self._graph.cycles)

    @property
    def edges(self) -> np.ndarray:
        return self._graph.links

    @property
    def cycles(self) -> np.ndarray:
        return self._graph.cycles

    @property
    def cycle_length(self) -> int:
        return len(self.cycles[0])

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
                    (icyc + 1, sign_nonzero(cycle[jj] - cycle[ii]))
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

    @property
    def dual_edge_dir(self) -> np.ndarray:
        return self._dual_edge_dir

    def compute_residues(
        self,
        wrapdata: ArrayLike,
    ) -> ArrayLike:
        """Compute phase residues for one set of input wrapped data."""
        if wrapdata.size != self.npoints:
            errmsg = (
                f"Size mismatch for residue computation."
                f" Received {wrapdata.shape} with {self.npoints} points"
            )
            raise ValueError(errmsg)

        # Residues includes the grounding node at index 0
        residues = np.zeros(self.ncycles + 1)
        ndim = self.cycle_length
        for col in range(ndim):
            nn = (col + 1) % ndim
            residues[1:] += phase_diff(
                wrapdata[self.cycles[:, col]], wrapdata[self.cycles[:, nn]]
            )
        residues = np.rint(residues / (2 * np.pi)).astype(int)
        residues[0] = -np.sum(residues[1:])
        return residues

    def compute_residues_from_gradients(
        self,
        graddata: ArrayLike,
    ) -> ArrayLike:
        """Compute phase residues for one set of real input gradients."""
        if graddata.size != self.nedges:
            errmsg = (
                f"Size mismatch for residue computation."
                f" Received {graddata.shape} with {self.nedges} edges"
            )
            raise ValueError(errmsg)

        cyc0 = np.abs(self.dual_edges[:, 0])
        cyc1 = np.abs(self.dual_edges[:, 1])
        cyc0_dir = self.dual_edge_dir[:, 0]
        cyc1_dir = self.dual_edge_dir[:, 1]
        grad_sum = np.zeros(self.ncycles + 1, dtype=np.float32)
        # add.at to handle repeated indices
        np.add.at(grad_sum, cyc0, cyc0_dir * graddata)
        np.add.at(grad_sum, cyc1, cyc1_dir * graddata)

        residues = np.rint(grad_sum / (2 * np.pi))
        # Set supply of groud_node
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
        # Only solve if necessary
        if not np.any(residues != 0):
            return np.zeros(self.nedges, dtype=np.int32)

        if revcost is None:
            revcost = cost
        return solve_mcf(self._dual_edges, self._dual_edge_dir, residues, cost, revcost)

    def residues_to_flows_many(
        self,
        residues: np.ndarray,
        cost: np.ndarray,
        revcost: np.ndarray | None = None,
        worker_count: int | None = None,
    ) -> np.ndarray:
        """Parallel version of residues_to_flows.

        The worker_count is set to get_cpu_count() - 1 if not provided.
        Treating costs as 1D arrays for now. Can consider 2D at a later time,
        if necessary.
        """
        if (worker_count is None) or (worker_count <= 0):
            worker_count = max(1, get_cpu_count() - 1)

        if revcost is None:
            revcost = cost

        # Get dimensions of the problem
        nruns, nresidues = residues.shape

        if nresidues != self.ncycles + 1:
            errmsg = (
                f"Number of residues {nresidues} does not match number of"
                f" cycles {self.ncycles}"
            )
            raise ValueError(errmsg)

        # Create flows output variable
        flows = np.zeros((nruns, self.nedges), dtype=np.int32)

        # Only use multiprocessing if needed
        if worker_count == 1:
            for ii, res in enumerate(residues):
                if np.any(res != 0):
                    flows[ii, :] = self.residues_to_flows(res, cost, revcost=revcost)

        else:
            print(f"Processing batch of {nruns} with {worker_count} threads")

            def uw_inputs(idxs):
                for ii in idxs:
                    # Only solve if needed
                    if not np.any(residues[ii] != 0):
                        continue

                    yield (
                        ii,
                        self._dual_edges,
                        self._dual_edge_dir,
                        residues[ii],
                        cost,
                        revcost,
                    )

            # Create a pool and dispatch
            with Pool(processes=worker_count, maxtasksperchild=1) as p:
                mp_tasks = p.imap_unordered(wrap_solve_mcf, uw_inputs(range(nruns)))

                # Gather results
                count = 0
                for res in mp_tasks:
                    flows[res[0], :] = res[1]
                    count += 1

            assert count == nruns, "Output size != Input size"

        return flows


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
        flows[ii] = first_cycle_dir[ii] * (smcf.flow(ii + num_edges) - smcf.flow(ii))

    return flows


def wrap_solve_mcf(
    args: tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None],
) -> tuple[int, np.ndarray]:
    """Parallel version of solve_mcf."""
    ind, es, ed, rr, cc, rc = args
    return (ind, solve_mcf(es, ed, rr, cc, rc))
