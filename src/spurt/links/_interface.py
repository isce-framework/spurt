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

    def estimate_model(
        self, wrapdata: np.ndarray, weights: np.ndarray | None = None
    ) -> tuple[float | np.ndarray, np.ndarray, float]:
        """Return estimated model parameters, reconstructed model and quality metric."""
