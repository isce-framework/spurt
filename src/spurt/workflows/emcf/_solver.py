"""Implementation of Extended Minimum Cost Flow (EMCF) phase unwrapping."""

from __future__ import annotations

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from spurt.io import Irreg3DInput
from spurt.links import LinkModelInterface
from spurt.mcf import MCFSolverInterface, utils
from spurt.utils import get_cpu_count, logger

from ._settings import SolverSettings


class EMCFSolver:
    """Implementation of the EMCF algorithm.

    Implements the Extended Minimum Cost Flow (EMCF) algorithm for
    phase unwrapping [@Pepe2006ExtensionMinimumCost]. We only implement the solver
    framework [@Olsen2023ContextualUncertaintyAssessments] as the
    graph generation and cost function implementations are not exactly
    replicable without more details.
    """

    def __init__(
        self,
        solver_space: MCFSolverInterface,
        solver_time: MCFSolverInterface,
        settings: SolverSettings,
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
        settings: SolverSettings
            Settings to be used for setting up the solver like number of
            workers, link batch size etc.
        model: LinkModel | None
            Per-link model in time used to correct the gradients before
            unwrapping.
        """
        self._solver_space = solver_space
        self._solver_time = solver_time
        self._settings = settings
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
    def settings(self) -> SolverSettings:
        """Retrieve settings for the workflow."""
        return self._settings

    @property
    def link_model(self) -> LinkModelInterface | None:
        """Retrieve the link model for the workflow."""
        return self._link_model

    def unwrap_cube(self, wrap_data: Irreg3DInput) -> np.ndarray:
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
            errmsg = f"Input data is not a 2D array - {wrap_data.ndim}"
            raise ValueError(errmsg)

        if wrap_data.time_dim != 0:
            errmsg = "Time must be first dimension in input stack."
            raise NotImplementedError(errmsg)

        input_is_ifg: bool = False
        if wrap_data.data.shape[0] == self.nepochs:
            input_is_ifg = False
        elif wrap_data.data.shape[0] == self.nifgs:
            input_is_ifg = True
        else:
            errmsg = (
                f"Input size {wrap_data.data.shape[0]} does not match solver"
                f" for {self.nifgs} Ifgs from {self.nepochs} images"
            )
            raise ValueError(errmsg)

        # First unwrap in time to get spatial gradients
        grad_space: np.ndarray = self.unwrap_gradients_in_time(
            wrap_data.data, input_is_ifg=input_is_ifg
        )

        # Then unwrap spatial gradients
        return self.unwrap_gradients_in_space(grad_space)

    def unwrap_gradients_in_time(
        self, wrap_data: np.ndarray, *, input_is_ifg: bool
    ) -> np.ndarray:
        """Temporally unwrap links in parallel.

        The output of this step is the temporally unwrapped phase gradients on
        each link of the spatial graph.
        """
        # First set up temporal cost
        if self.settings.t_cost_type == "constant":
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

        # Create output array
        grad_space: np.ndarray = np.zeros((self.nifgs, self.nlinks), dtype=np.float32)

        logger.info(f"Temporal: Number of interferograms: {self.nifgs}")
        logger.info(f"Temporal: Number of links: {self.nlinks}")
        logger.info(f"Temporal: Number of cycles: {self._solver_time.ncycles}")

        # Number of batches to process
        nbatches: int = ((self.nlinks - 1) // self.settings.links_per_batch) + 1

        # Iterate over batches
        for bb in range(nbatches):
            i_start = bb * self.settings.links_per_batch
            i_end = min(i_start + self.settings.links_per_batch, self.nlinks)
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
                logger.info(f"Temporal: Preparing batch {bb + 1}/{nbatches}")
                self._ifg_spatial_gradients_from_slc(
                    wrap_data, inds, grad_space, np.s_[i_start:i_end]
                )

            # Compute residues for each cycle in temporal graph
            # Easier to loop over interferograms here
            ncycles: int = len(self._solver_time.cycles)
            grad_sum: np.ndarray = np.zeros(
                (links_in_batch, ncycles + 1), dtype=np.float32
            )
            for ii in range(self.nifgs):
                # Cycles that ifg contributes to
                cyc = np.abs(self._solver_time.dual_edges[ii])
                cyc_dir = self._solver_time.dual_edge_dir[ii]
                grad_sum[:, cyc[0]] += cyc_dir[0] * grad_space[ii, i_start:i_end]
                if cyc[1] != 0:
                    grad_sum[:, cyc[1]] += cyc_dir[1] * grad_space[ii, i_start:i_end]

            residues = np.rint(grad_sum / (2 * np.pi)).astype(int)
            # Set grounding node
            residues[:, 0] = -np.sum(residues[:, 1:], axis=1)

            # Unwrap the batch
            logger.info(f"Temporal: Unwrapping batch {bb + 1}/{nbatches}")
            flows = self._solver_time.residues_to_flows_many(
                residues,
                cost,
                worker_count=self.settings.t_worker_count,
                chunksize=50000,
            )

            # Update the spatial gradients with estimated flows
            grad_space[:, i_start:i_end] += 2 * np.pi * flows.T

        return grad_space

    def unwrap_gradients_in_space(self, grad_space: np.ndarray) -> np.ndarray:
        """Spatially unwrap each interferogram sequentially."""
        # First set up spatial cost
        if self.settings.s_cost_type == "constant":
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

        logger.info(f"Spatial: Number of interferograms: {self.nifgs}")
        logger.info(f"Spatial: Number of links: {self.nlinks}")
        logger.info(f"Spatial: Number of cycles: {self._solver_space.ncycles}")

        nworkers = self.settings.s_worker_count
        if nworkers < 1:
            nworkers = get_cpu_count() - 1

        mp_context = mp.get_context("fork")
        with ProcessPoolExecutor(
            max_workers=nworkers, mp_context=mp_context
        ) as executor:
            futures = {
                executor.submit(
                    _unwrap_ifg_in_space,
                    grad_space[ii, :],
                    self._solver_space,
                    cost,
                    ii,
                ): ii
                for ii in range(self.nifgs)
            }
            for fut in as_completed(futures):
                ii, data = fut.result()
                futures.pop(fut)
                uw_data[ii, :] = data
        return uw_data

    def _ifg_spatial_gradients_from_slc(
        self,
        wrap_data: np.ndarray,
        edges: np.ndarray,
        grad_space: np.ndarray,
        link_slice: slice,
    ) -> None:
        """Compute interferometric spatial gradients from slc data.

        Parameters
        ----------
        wrap_data: np.ndarray
            Wrapped slc data 2D array for whole graph of shape (nslc, npts)
        edges: np.ndarray
            2D array corresponding to edges in spatial graph. These are a
            subset of all links in the graph.
        grad_space: np.ndarray
            Spatial gradient array for the whole graph of shape (nifg, nlinks).
            This array gets updated in place.
        link_slice: slice
            Slice corresponding to edges within the array of all links.
        """
        # Interferogram edges
        ifg_inds = self._solver_time.edges

        # Extract SLC data first
        slc_data0 = wrap_data[:, edges[:, 0]]
        slc_data1 = wrap_data[:, edges[:, 1]]

        # Make interferograms for extracted points
        ifg_data0 = utils.phase_diff(
            slc_data0[ifg_inds[:, 0], :], slc_data0[ifg_inds[:, 1], :]
        )
        ifg_data1 = utils.phase_diff(
            slc_data1[ifg_inds[:, 0], :], slc_data1[ifg_inds[:, 1], :]
        )

        # Update gradient in place
        grad_space[:, link_slice] = utils.phase_diff(ifg_data0, ifg_data1)


def _unwrap_ifg_in_space(ifg_grad, solver_space, cost, ii):
    # Compute residues
    residues = solver_space.compute_residues_from_gradients(ifg_grad)

    # Unwrap the interferogram - sequential
    flows = solver_space.residues_to_flows(residues, cost)

    # Flood fill
    out = utils.flood_fill(ifg_grad, solver_space.edges, flows, mode="gradients")
    logger.info(f"Completed spatial unwrapping {ii + 1}")
    return ii, out
