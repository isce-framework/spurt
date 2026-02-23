from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._interface import LinkModelInterface


@dataclass
class Parameters:
    """Data belonging to the model."""

    # Matrix for the linear model
    matrix: np.ndarray

    # One slice per variable
    ranges: tuple[slice, ...]

    def __post_init__(self):
        if self.matrix.shape[1] != len(self.ranges):
            errmsg = (
                f"size mismatch: matrix ncol ({self.matrix.shape[1]})"
                f" did not match ranges size ({len(self.ranges)})"
            )
            raise ValueError(errmsg)


class GridSearchLinearModel(Parameters, LinkModelInterface):
    """Search parameter grid space for the maxima.

    Implements grid search for best fitting parameters.
    We don't solve a bunch of MCF problems like Pepe and Lanari (2006) but
    instead solve this link-by-link.

    Max: |exp(j * (A * x - b))|^2
    s.t: ranges[i][0] <= x_i <= ranges[i][1]
    """

    def __post_init__(self):
        super().__post_init__()
        self._precompute()

    def _precompute(self) -> None:
        """Precompute grid, forward model, and complex exponentials.

        These are constant for a given design matrix and search ranges,
        so computing them once avoids redundant work per link.
        """
        axes = [np.arange(s.start, s.stop, s.step) for s in self.ranges]
        self._grid_shape = tuple(len(a) for a in axes)
        grids = np.meshgrid(*axes, indexing="ij")
        self._grid_flat = np.column_stack([g.ravel() for g in grids])
        self._grid_steps = np.array([s.step for s in self.ranges])

        # Clip bounds from actual grid range
        self._param_lo = self._grid_flat.min(axis=0)
        self._param_hi = self._grid_flat.max(axis=0)

        # Forward model for all grid points: (nobs, ngrid)
        self._pred = self.matrix @ self._grid_flat.T

        # Complex exponential of forward model: (nobs, ngrid)
        self._E = np.exp(1j * self._pred)

        # Quadratic refinement setup (2D only)
        if self.ndim == 2:
            self._init_quadratic_refinement()

    def _init_quadratic_refinement(self) -> None:
        """Precompute pseudoinverse for 3x3 quadratic surface fit."""
        offsets = np.array(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        )
        self._stencil_offsets = offsets
        dx = offsets[:, 0].astype(np.float64)
        dy = offsets[:, 1].astype(np.float64)
        design = np.column_stack([np.ones(9), dx, dy, dx**2, dx * dy, dy**2])
        self._refine_pinv = np.linalg.pinv(design)  # (6, 9)

    @property
    def nobs(self) -> int:
        return self.matrix.shape[0]

    @property
    def ndim(self) -> int:
        return self.matrix.shape[1]

    @property
    def ngrid(self) -> int:
        """Total number of grid points in the search space."""
        return len(self._grid_flat)

    def fwd_model(self, x: np.ndarray) -> np.ndarray:
        return np.dot(self.matrix, x)

    def estimate_model(
        self,
        wrapdata: np.ndarray,
        weights: np.ndarray | float | None = None,
    ) -> tuple[np.ndarray, float]:
        """Fit model parameters via grid search + quadratic refinement.

        Parameters
        ----------
        wrapdata : np.ndarray
            Real-valued array of wrapped phase gradient, shape ``(nobs,)``.
        weights : np.ndarray | float | None
            Real-valued weights, assumed normalized to 1.

        Returns
        -------
        params : np.ndarray
            1D array of length ``ndim``.
        coh : float
            Temporal coherence.
        """
        if wrapdata.ndim != 1:
            errmsg = f"Input data must be a 1D array. Got {wrapdata.shape}."
            raise ValueError(errmsg)

        if weights is None:
            weights = 1.0 / self.nobs

        if isinstance(weights, np.ndarray) and (weights.shape != wrapdata.shape):
            errmsg = f"Weights shape mismatch: {weights.shape} vs {wrapdata.shape}"
            raise ValueError(errmsg)

        params, coh = self.estimate_model_many(wrapdata[:, np.newaxis], weights=weights)
        return params[:, 0], float(coh[0])

    def estimate_model_many(
        self,
        wrapdata: np.ndarray,
        weights: np.ndarray | float | None = None,
        worker_count: int | None = None,  # noqa: ARG002
    ) -> tuple[np.ndarray, np.ndarray]:
        """Batched grid search + quadratic refinement for many links.

        Evaluates all grid points for all links via matrix multiply, then
        refines with quadratic interpolation. Links are processed in chunks
        to keep the intermediate ``(ngrid, nlinks)`` coherence matrix within
        a reasonable memory budget.

        Parameters
        ----------
        wrapdata : np.ndarray
            Wrapped phase, shape ``(nobs, nlinks)``.
        weights : np.ndarray | float | None
            Weights. Scalar for uniform, 1D for per-observation,
            2D for per-observation-per-link.
        worker_count : int | None
            Accepted for ``LinkModelInterface`` compatibility but not used.
            The batched approach uses NumPy's internal BLAS threading.

        Returns
        -------
        params : np.ndarray
            Shape ``(ndim, nlinks)``.
        coh : np.ndarray
            Shape ``(nlinks,)``.
        """
        if wrapdata.ndim != 2:
            errmsg = f"Input data must be a 2D array. Got {wrapdata.shape}."
            raise ValueError(errmsg)

        if wrapdata.shape[0] != self.nobs:
            errmsg = f"Input shape mismatch. Got {wrapdata.shape} vs {self.nobs}"
            raise ValueError(errmsg)

        wts: np.ndarray | float
        if weights is None:
            wts = 1.0 / self.nobs
        else:
            wts = weights
            if isinstance(wts, np.ndarray):
                if wts.ndim == 2 and wts.shape != wrapdata.shape:
                    errmsg = (
                        f"Weights shape mismatch."
                        f" Got {wts.shape} vs {wrapdata.shape}"
                    )
                    raise ValueError(errmsg)
                if wts.ndim <= 1 and wts.shape[0] != self.nobs:
                    errmsg = f"Weights shape mismatch. Got {wts.shape} vs {self.nobs}"
                    raise ValueError(errmsg)

        nlinks = wrapdata.shape[1]

        # Weighted conjugate of data: v[k, l] = wts_k * exp(-1j * wdata[k, l])
        d_conj = np.exp(-1j * wrapdata)
        if isinstance(wts, np.ndarray) and wts.ndim == 2:
            weighted_conj = wts * d_conj
        elif isinstance(wts, np.ndarray):
            weighted_conj = wts[:, np.newaxis] * d_conj
        else:
            weighted_conj = wts * d_conj
        del d_conj

        # Chunk links so the intermediate (ngrid, chunk) coherence matrix
        # stays under ~512 MB. Each link needs ngrid * 24 bytes
        # (16 for complex128 matmul result + 8 for float64 abs).
        bytes_per_link = self.ngrid * 24
        chunk_size = max(1, int(512e6 / bytes_per_link))

        params = np.zeros((self.ndim, nlinks), dtype=np.float64)
        coh = np.zeros(nlinks, dtype=np.float64)

        for start in range(0, nlinks, chunk_size):
            end = min(start + chunk_size, nlinks)
            sl = slice(start, end)

            # Grid search for this chunk
            coh_grid = self._E.T @ weighted_conj[:, sl]  # (ngrid, chunk)
            coherence = np.abs(coh_grid)  # (ngrid, chunk)
            del coh_grid

            best_idx = np.argmax(coherence, axis=0)
            chunk_n = end - start

            # Quadratic refinement (2D case)
            if self.ndim == 2 and chunk_n > 0:
                chunk_wts = (
                    wts[:, sl] if isinstance(wts, np.ndarray) and wts.ndim == 2 else wts
                )
                params[:, sl], coh[sl] = self._quadratic_refine_batch(
                    coherence, best_idx, wrapdata[:, sl], chunk_wts
                )
            else:
                params[:, sl] = self._grid_flat[best_idx].T
                coh[sl] = coherence[best_idx, np.arange(chunk_n)]

        return params, coh

    def _quadratic_refine_batch(
        self,
        coherence: np.ndarray,
        best_idx: np.ndarray,
        wrapdata: np.ndarray,
        wts: np.ndarray | float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Refine grid search via 2D quadratic interpolation.

        Fits a quadratic surface to the 3x3 coherence stencil around
        each best grid point and solves for the analytic peak.
        """
        nlinks = len(best_idx)

        # Convert flat grid indices to 2D
        i0, i1 = np.unravel_index(best_idx, self._grid_shape)

        # Build stencil grid indices: (nlinks, 9)
        di = self._stencil_offsets[:, 0]  # (9,)
        dj = self._stencil_offsets[:, 1]  # (9,)
        si = np.clip(i0[:, None] + di[None, :], 0, self._grid_shape[0] - 1)
        sj = np.clip(i1[:, None] + dj[None, :], 0, self._grid_shape[1] - 1)

        # Flat stencil indices and extract coherence
        flat_idx = np.ravel_multi_index((si, sj), self._grid_shape)
        link_col = np.broadcast_to(np.arange(nlinks)[:, None], flat_idx.shape)
        stencil_coh = coherence[flat_idx, link_col]  # (nlinks, 9)

        # Fit quadratic: f(dx,dy) = a + b*dx + c*dy + d*dx^2 + e*dx*dy + f*dy^2
        coeffs = stencil_coh @ self._refine_pinv.T  # (nlinks, 6)

        b_c = coeffs[:, 1]
        c_c = coeffs[:, 2]
        d_c = coeffs[:, 3]
        e_c = coeffs[:, 4]
        f_c = coeffs[:, 5]

        # Solve for quadratic peak: H @ [dx, dy] = -[b, c]
        # where H = [[2d, e], [e, 2f]]
        det = 4.0 * d_c * f_c - e_c**2
        is_max = (d_c < 0) & (det > 0) & (np.abs(det) > 1e-12)

        dx = np.zeros(nlinks)
        dy = np.zeros(nlinks)
        m = is_max
        dx[m] = -(2.0 * f_c[m] * b_c[m] - e_c[m] * c_c[m]) / det[m]
        dy[m] = -(2.0 * d_c[m] * c_c[m] - e_c[m] * b_c[m]) / det[m]

        # Clip refinement to one grid step
        dx = np.clip(dx, -1.0, 1.0)
        dy = np.clip(dy, -1.0, 1.0)

        # Build refined parameters
        params = self._grid_flat[best_idx].T.copy()  # (ndim, nlinks)
        params[0] += dx * self._grid_steps[0]
        params[1] += dy * self._grid_steps[1]

        # Clip to search bounds
        for d in range(self.ndim):
            params[d] = np.clip(params[d], self._param_lo[d], self._param_hi[d])

        # Evaluate actual coherence at refined parameters
        coh = self._eval_coherence(params, wrapdata, wts)

        return params, coh

    def _eval_coherence(
        self,
        params: np.ndarray,
        wrapdata: np.ndarray,
        wts: np.ndarray | float,
    ) -> np.ndarray:
        """Evaluate temporal coherence at given parameters.

        Parameters
        ----------
        params : np.ndarray
            Shape ``(ndim, nlinks)``.
        wrapdata : np.ndarray
            Shape ``(nobs, nlinks)``.
        wts : np.ndarray | float
            Weights.

        Returns
        -------
        np.ndarray
            Coherence values, shape ``(nlinks,)``.
        """
        residuals = self.matrix @ params - wrapdata  # (nobs, nlinks)
        weighted_exp = np.exp(1j * residuals)
        if isinstance(wts, np.ndarray) and wts.ndim == 2:
            weighted_exp *= wts
        elif isinstance(wts, np.ndarray):
            weighted_exp *= wts[:, np.newaxis]
        else:
            weighted_exp *= wts
        return np.abs(weighted_exp.sum(axis=0))


def _vectorized_grid_search(
    matrix: np.ndarray,
    rngs: tuple[slice, ...],
    wdata: np.ndarray,
    wts: np.ndarray | float,
) -> np.ndarray:
    """Find parameters maximizing temporal coherence over a regular grid.

    Evaluates all grid points in a single batched numpy operation
    instead of calling the objective function once per grid point.

    Parameters
    ----------
    matrix : np.ndarray
        Design matrix of shape ``(nifgs, ndim)``.
    rngs : tuple[slice, ...]
        One ``slice(start, stop, step)`` per parameter dimension.
    wdata : np.ndarray
        Wrapped phase data of shape ``(nifgs,)``.
    wts : np.ndarray | float
        Weights, either scalar or array of shape ``(nifgs,)``.

    Returns
    -------
    np.ndarray
        Best-fit parameter vector of shape ``(ndim,)``.
    """
    # Build 1D coordinate arrays for each parameter dimension
    axes = [np.arange(s.start, s.stop, s.step) for s in rngs]

    # Flattened grid of all parameter combinations: (ngrid, ndim)
    grids = np.meshgrid(*axes, indexing="ij")
    grid_flat = np.column_stack([g.ravel() for g in grids])

    # Evaluate all grid points at once
    # matrix @ grid_flat.T: (nifgs, ndim) @ (ndim, ngrid) -> (nifgs, ngrid)
    residuals = matrix @ grid_flat.T - wdata[:, np.newaxis]

    # Temporal coherence for all grid points
    weighted_exp = np.exp(1j * residuals)
    if isinstance(wts, np.ndarray):
        weighted_exp *= wts[:, np.newaxis]
    else:
        weighted_exp *= wts
    coherence = np.abs(weighted_exp.sum(axis=0))

    return grid_flat[np.argmax(coherence)]
