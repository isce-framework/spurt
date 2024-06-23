"""Protocol for link models.

This is meant to be used for estimation of DEM errors, velocities etc
on a link-by-link basis.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

__all__ = [
    "LinkModelInterface",
]


@runtime_checkable
class LinkModelInterface(Protocol):
    """
    Interface to correcting link gradients.

    Such objects should provide access to estimated model
    parameters and reconstructed model.

    Attributes
    ----------
    nobs: int
        Number of observations. Can be epochs or interferograms based on
        context.
    ndim: int
        Number of model parameters.
    """

    @property
    def nobs(self) -> int:
        """Return number of observations in model."""

    @property
    def ndim(self) -> int:
        """Return number of model parameters."""

    def fwd_model(self, x: np.ndarray) -> np.ndarray:
        """Return model evaluated at x."""

    def estimate_model(
        self, wrapdata: np.ndarray, weights: np.ndarray | float | None = None
    ) -> tuple[np.ndarray, float]:
        """Return estimated model parameters and quality metric.

        Parameters
        ----------
        wrapdata: np.ndarray
            1D complex array or real valued array in radians.
        weights: np.ndarray
            1D array of same length as wrapdata with weights.
            If scalar, all observations are equally weighted.

        Returns
        -------
        params: np.ndarray
              1D array of length ndim with the estimated model parameters.
        quality: float
              Temporal coherence or a similar quality metric indicating model
              fit.
        """

    def estimate_model_many(
        self,
        wrapdata: np.ndarray,
        weights: np.ndarray | float | None = None,
        worker_count: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate model for many inputs.

        Parameters
        ----------
        wrapdata: np.ndarray
            2D complex array or real valued array in radians.
        weights: np.ndarray
            1D / 2D array of same length as wrapdata with weights.
            If 1D array, same weights are reused for all inputs.
         worker_count: int
            Number of workers to use if inputs can be handled in parallel. The
            implementation determines default in case a number <=0 is provided.

        Returns
        -------
        params: np.ndarray
            2D array of shape (ninputs, ndim) with the estimated model parameters.
        quality: np.ndarray
            1D array of length ninputs representing temporal coherence or a similar
            quality metric indicating model fit.
        """
