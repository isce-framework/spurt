from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import get_context

import numpy as np
from scipy import optimize

from ..utils import get_cpu_count, logger
from ._common import temporal_coherence
from ._interface import LinkModelInterface


@dataclass
class Parameters:
    """Data belonging to the model."""

    # Matrix for the linear model
    matrix: np.ndarray

    # One slice per variable
    ranges: tuple[slice]

    def __post_init__(self):
        assert self.matrix.shape[1] == len(self.ranges)


class GridSearchLinearModel(Parameters, LinkModelInterface):
    """Search parameter grid space for the maxima.

    Implements grid search for best fitting parameters.
    We don't solve a bunch of MCF problems like Pepe and Lanari (2006) but
    instead solve this link-by-link.

    Max: |exp(j * (A * x - b))|^2
    s.t: ranges[i][0] <= x_i <= ranges[i][1]
    """

    @property
    def nobs(self) -> int:
        return self.matrix.shape[0]

    @property
    def ndim(self) -> int:
        return self.matrix.shape[1]

    def fwd_model(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, x)

    def estimate_model(
        self,
        wrapdata: np.ndarray,
        weights: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Grid search followed by fmin.

        Parameters
        ----------
        wrapdata: np.ndarray
            Real-valued array of wrapped phase gradient
        weights: np.ndarray | None
            Real-valued weights - assumed to be normalized to 1.

        Returns
        -------
        params: np.ndarray
            1D array of length ndim
        coh: float
            Temporal coherence
        """
        if wrapdata.ndim != 1:
            errmsg = f"Input data must be a 1D array. Got {wrapdata.shape}."
            raise ValueError(errmsg)

        if weights is None:
            weights = 1.0 / self.nobs

        if isinstance(weights, np.ndarray) and (weights.shape != wrapdata.shape):
            errmsg = f"Weights shape mismatch: {weights.shape} vs {wrapdata.shape}"
            raise ValueError(errmsg)

        return solve(self.matrix, self.ranges, wrapdata, weights)

    def estimate_model_many(
        self,
        wrapdata: np.ndarray,
        weights: np.ndarray | float | None = None,
        worker_count: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Grid search followed by fmin in parallel."""
        if (worker_count is None) or (worker_count <= 0):
            worker_count = max(1, get_cpu_count() - 1)

        if wrapdata.ndim != 2:
            errmsg = f"Input data must be a 2D array. Got {wrapdata.shape}."
            raise ValueError(errmsg)

        if wrapdata.shape[0] != self.nobs:
            errmsg = f"Input shape mismatch. Got {wrapdata.shape} vs {self.nobs}"
            raise ValueError(errmsg)

        const_weights: bool = True
        if weights is None:
            weights = 1.0 / self.nobs
        elif isinstance(weights, np.ndarray):
            if weights.ndim == 2 and (weights.shape != wrapdata.shape):
                errmsg = (
                    f"Weights shape mismatch. Got {weights.shape} vs {wrapdata.shape}"
                )
                raise ValueError(errmsg)
                const_weights = False
                arr_weights: np.ndarray = weights

            if weights.shape[0] != self.nobs:
                errmsg = f"Weights shape mismatch. Got {weights.shape} vs {self.nobs}"
                raise ValueError(errmsg)

        # Return arrays
        nruns: int = wrapdata.shape[1]
        params: np.ndarray = np.zeros((self.ndim, nruns))
        tcoh: np.ndarray = np.zeros(nruns)

        # Run sequentially when only 1 worker available
        if worker_count == 1:
            for ii in range(nruns):
                wts = weights if const_weights else arr_weights[:, ii]
                res = self.estimate_model(
                    wrapdata[:, ii],
                    wts,
                )
                params[:, ii] = res[0]
                tcoh[ii] = res[1]
        else:
            logger.info(f"Modeling batch of {nruns} with {worker_count} threads")

            def inv_inputs(idxs):
                for ii in idxs:
                    wts = weights if const_weights else arr_weights[:, ii]
                    yield (
                        ii,
                        self.matrix,
                        self.ranges,
                        wrapdata[:, ii],
                        wts,
                    )

            # Create a pool and dispatch
            with get_context("fork").Pool(
                processes=worker_count, maxtasksperchild=1
            ) as p:
                mp_tasks = p.imap_unordered(wrap_solve, inv_inputs(range(nruns)))

                # Gather results
                for res in mp_tasks:  # type: ignore[assignment]
                    params[:, res[0]] = res[1]
                    tcoh[res[0]] = res[2]  # type: ignore[misc]

        return params, tcoh


def solve(
    matrix: np.ndarray, rngs: tuple[slice], wdata: np.ndarray, wts: np.ndarray | float
) -> tuple[np.ndarray, float]:
    """Actual call to the solver."""
    resbrute = optimize.brute(
        temporal_coherence,
        rngs,
        args=(matrix, wdata, wts),
        full_output=True,
        finish=optimize.fmin,
    )
    return (resbrute[0], -resbrute[1])


def wrap_solve(
    args: tuple[int, np.ndarray, tuple[slice], np.ndarray, np.ndarray | float],
) -> tuple[int, np.ndarray, float]:
    ind, ma, rg, wd, wt = args
    out = solve(ma, rg, wd, wt)
    return (ind, out[0], out[1])
