"""Implementation of Extended Minimum Cost Flow (EMCF) phase unwrapping.

A. Pepe and R. Lanari, "On the Extension of the Minimum Cost Flow Algorithm for
Phase Unwrapping of Multitemporal Differential SAR Interferograms," in IEEE
Transactions on Geoscience and Remote Sensing, vol. 44, no. 9, pp. 2374-2383,
Sept. 2006, doi: 10.1109/TGRS.2006.873207.
"""

from __future__ import annotations

import numpy as np

from spurt.links import LinkModelInterface
from spurt.mcf import MCFSolverInterface, utils

from ._Settings import Settings


class EMCFSolver:
    """Implementation of the EMCF algorithm."""

    def __init__(
        self,
        solver_space: MCFSolverInterface,
        solver_time: MCFSolverInterface,
        settings: Settings | None = None,
        link_model: LinkModelInterface | None = None,
    ):
        """Spatio-temporal unwrapping.

        Parameters
        ----------
        solver_space: MCFSolverInterface
            MCF Solver interface for spatial graph connecting the stable points.
            Typically a Delaunaytriangulation.
        solver_time: MCFSolverInterface
            MCF Solver interface for temporal graph usually representing
            interferograms in time-Bperp space. Typically a Delaunay triangulation.
        model: LinkModel
            Per-link model in time used to correct the gradients before
        unwrapping.
        """
        self._solver_space = solver_space
        self._solver_time = solver_time
        self._settings = settings if settings else Settings()
        self._link_model = link_model

        if link_model is not None:
            errmsg = "Not implemented yet."
            raise NotImplementedError(errmsg)

    @property
    def npoints(self) -> int:
        """Number of points in the spatial network."""
        return self._solver_space.npoints

    @property
    def nlinks(self) -> int:
        """Number of links in the spatial network."""
        return self._solver_space.nedges

    @property
    def nepochs(self) -> int:
        """Number of points in the temporal network."""
        return self._solver_time.npoints

    @property
    def nifgs(self) -> int:
        """Number of links in the temporal network."""
        return self._solver_time.nedges

    @property
    def settings(self) -> Settings:
        """Retrieve settings for the workflow."""
        return self._settings

    @property
    def link_model(self) -> LinkModelInterface | None:
        """Retrieve the link model for the workflow."""
        return self._link_model

    def unwrap_cube(self, wrap_data: np.ndarray) -> np.ndarray:
        """Unwrap a 3D cube of data.

        Parameters
        ----------
        wrap_data: np.ndarray
            2D array of size (nslc, npoints) or (nifg, npoints).

        Returns
        -------
        uw_data: np.ndarray
            2D float32 array of size (nifg, npoints).
        """
        if wrap_data.ndim != 2:
            errmsg = f"Input data has more than two dimension - {wrap_data.ndim}"
            raise ValueError(errmsg)

        if wrap_data.shape[0] == self.nepochs:
            input_is_ifg = False
        elif wrap_data.shape[0] == self.nifgs:
            input_is_ifg = True
        else:
            errmsg = (
                f"Input size {wrap_data.shape[0]} does not match solver"
                f" for {self.nifgs} Ifgs from {self.nepochs} images"
            )
            raise ValueError(errmsg)

        # First unwrap in time to get spatial gradients
        grad_space = self._unwrap_time(wrap_data, input_is_ifg=input_is_ifg)

        # Then unwrap spatial gradients
        return self._unwrap_space(grad_space)

    def _unwrap_time(self, wrap_data: np.ndarray, *, input_is_ifg: bool) -> np.ndarray:
        """Temporally unwrap links in parallel."""
        # First set up temporal cost
        if self.settings.t_cost_type == "unit":
            cost = np.ones(self.nifgs, dtype=int)
        elif self.settings.t_cost_type == "distance":
            cost = utils.distance_costs(
                self._solver_time.points,
                self._solver_time.edges,
                scale=self.settings.t_cost_scale,
            )
        elif self.settings.t_cost_type == "centroid":
            cost = utils.centroid_costs(
                self._solver_time.points,
                self._solver_time.cycles,
                self._solver_time.dual_edges,
                scale=self.settings.t_cost_scale,
            )
        else:
            errmsg = f"Unknown cost type: {self.settings.t_cost_type}"
            raise ValueError(errmsg)

        # Mapping from temporal edges to index in edges array
        edge_to_index = {}
        for ii, edge in enumerate(self._solver_time.edges):
            edge_to_index[(edge[0], edge[1])] = ii

        # Create output array
        grad_space = np.zeros((self.nifgs, self.nlinks), dtype=np.float32)

        # Number of batches to process
        nbatches = (self.nlinks // self.settings.points_per_batch) + 1

        # Iterate over batches
        for bb in range(nbatches):
            i_start = bb * self.settings.points_per_batch
            i_end = min(i_start + self.settings.points_per_batch, self.nlinks)
            links_in_batch = i_end - i_start
            if links_in_batch == 0:
                continue

            # Get indices of points forming links from spatial graph
            inds = self._solver_space.edges[i_start:i_end, :]

            # TODO: Incorporate link_model here when ready
            # Add self._modeled_phase_diff to replace phase_diff

            # Compute spatial gradients for each link
            # If input data is already interferograms
            if input_is_ifg:
                grad_space[:, i_start:i_end] = utils.phase_diff(
                    wrap_data[:, inds[:, 0]], wrap_data[:, inds[:, 1]]
                )
            else:
                ifg_inds = self._solver_time.edges

                # Extract SLC data first
                slc_data0 = wrap_data[:, inds[0]]
                slc_data1 = wrap_data[:, inds[1]]

                # Make interferograms for extracted points
                ifg_data0 = utils.phase_diff(
                    slc_data0[ifg_inds[:, 0], :], slc_data0[ifg_inds[:, 1], :]
                )
                ifg_data1 = utils.phase_diff(
                    slc_data1[ifg_inds[:, 0], :], slc_data1[ifg_inds[:, 1], :]
                )

                # Free memory
                slc_data0 = None
                slc_data1 = None

                grad_space[:, i_start:i_end] = utils.phase_diff(ifg_data0, ifg_data1)

                # Free memory
                ifg_data0 = None
                ifg_data1 = None

            # Compute residues for each cycle in temporal graph
            # Easier to loop over interferograms here
            ncycles = len(self._solver_time.cycles)
            grad_sum = np.zeros((links_in_batch, ncycles + 1), dtype=np.float32)
            for ii in range(self.nifgs):
                # Cycles that ifg contributes to
                cyc = self._solver_time.dual_edges[ii]
                cyc_dir = self._solver_time.dual_edge_dir[ii]
                grad_sum[:, cyc[0]] += (
                    cyc_dir[0] * grad_space[abs(cyc[0]), i_start:i_end]
                )
                if cyc[1] != 0:
                    grad_sum[:, cyc[1]] += (
                        cyc_dir[1] * grad_space[abs(cyc[1]), i_start:i_end]
                    )

            residues = np.rint(grad_sum / (2 * np.pi))
            # Set grounding node
            residues[:, 0] = -np.sum(residues[:, 1:], axis=0)

            # Unwrap the batch
            flows = self._solver_time.residues_to_flows_many(
                residues, cost, worker_count=self.settings.worker_count
            )

            # Update the spatial gradients with estimated flows
            grad_space[:, i_start:i_end] += 2 * np.pi * flows.T

        return grad_space

    def _unwrap_space(self, grad_space: np.ndarray) -> np.ndarray:
        """Spatially unwrap each interferogram sequentially."""
        # First set up temporal cost
        if self.settings.s_cost_type == "unit":
            cost = np.ones(self.nlinks, dtype=int)
        elif self.settings.s_cost_type == "distance":
            cost = utils.distance_costs(
                self._solver_space.points,
                self._solver_space.edges,
                scale=self.settings.s_cost_scale,
            )
        elif self.settings.s_cost_type == "centroid":
            cost = utils.centroid_costs(
                self._solver_space.points,
                self._solver_space.cycles,
                self._solver_space.dual_edges,
                scale=self.settings.s_cost_scale,
            )
        else:
            errmsg = f"Unknown cost type: {self.settings.s_cost_type}"
            raise ValueError(errmsg)

        # Create output array
        uw_data = np.zeros((self.nifgs, self.npoints), dtype=np.float32)

        for ii in range(self.nifgs):
            # Slice per ifg
            ifg_grad = grad_space[ii, :]

            # Compute residues
            residues = self._solver_space.compute_residues_from_gradients(ifg_grad)

            # Unwrap the interferogram - sequential
            flows = self._solver_space.residues_to_flows(residues, cost)

            # Flood fill
            uw_data[ii, :] = utils.flood_fill(ifg_grad, self._solver_space.edges, flows)

        return uw_data
