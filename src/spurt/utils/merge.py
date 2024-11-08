"""Utilities for merging point clouds."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import lstsq as lsq_dense
from numpy.linalg import norm as lpnorm
from scipy.sparse import csc_matrix, csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, spilu
from scipy.sparse.linalg import lsqr as lsq_sparse

from ._logger import logger


def find_common_points(c1: np.ndarray, c2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Given 2 2d (positive integer) point clouds, return the set of common points.

    Parameters
    ----------
    c1 : array[n, 2]
        First set of coordinates, must be integers
    c2 : array[m, 2]
        Second set of coordinates, must be integers

    Returns
    -------
    ii : array[int]
        Indices of c1 that overlaps c1
    jj : array[int]
        Indices of c2 that overlaps c2, i.e. c1[ii] == c2[jj]
    """
    if np.min(c1) < 0:
        errmsg = "First set of coordinates contains negative values"
        raise ValueError(errmsg)

    if np.min(c2) < 0:
        errmsg = "Second set of coordinates contains negative values"
        raise ValueError(errmsg)

    m = 1 + max(np.amax(c1[:, 1]), np.amax(c2[:, 1]))
    fc1 = c1[:, 0] * m + c1[:, 1]
    fc2 = c2[:, 0] * m + c2[:, 1]
    return np.intersect1d(fc1, fc2, return_indices=True)[1:]


def pairwise_unwrapped_diff_deciles(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Deciles for pairwise unwrapped differences.

    Given two sets of unwrapped phases that differ by integer cycles of 2pi,
    Create a histogram from 0-100 percentile in steps of 10.

    Parameters
    ----------
    b1: 2D array of (bands, pixels)
        First set of unwrapped phase in radians
    b2: 2D array of (bands, pixels)
        Second set of unwrapped phase in radians

    Returns
    -------
    hist: 2D array of (bands, 11)
        Array of size 11 with percentile values from 0 - 100
    """
    if b1.shape != b2.shape:
        errmsg = f"Shape mismatch: {b1.shape} vs {b2.shape}"
        raise ValueError(errmsg)

    if b1.ndim != 2:
        errmsg = f"Expecting 2D array as input - received {b1.shape}"
        raise ValueError(errmsg)

    if b1.shape[1] < 11:
        errmsg = f"Need atleast 11 elements per band - received {b1.shape}"
        raise ValueError(errmsg)

    # Compute difference
    diff: np.ndarray = np.zeros(b1.shape, dtype=np.float32)

    # Get integer difference
    diff[:, :] = (b2 - b1) / (2 * np.pi)
    nint: np.ndarray = np.rint(diff).astype(np.int16)

    # Check that we are close to integers
    assert np.allclose(diff, nint, atol=0.01), "Arrays differ by non-integer cycles"

    deciles = np.linspace(0.0, 100.0, 11)
    return np.percentile(nint, deciles, axis=-1).T


def l2_min(
    amat: np.ndarray | csr_matrix | csc_matrix,
    b: np.ndarray,
    *,
    enable_logging: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find x so that Ax-b has least L2 norm.

    Parameters
    ----------
    amat : matrix-like
        System to solve. This can be a dense or sparse matrix.
    b : vector-like
        Right-hand side.
    enable_logging : bool
        Optional, Enable logger to capture diagnostic messages.

    Returns
    -------
    x : array
        L2 minimizer of Ax - b
    r : array
        Ax - b
    """
    if np.size(amat) == 0:
        if enable_logging:
            logger.warning("A is empty; returning all zeros")
        return np.zeros(amat.shape[1]), np.zeros(amat.shape[0])

    assert amat.shape[0] == b.shape[0]

    if isinstance(amat, np.ndarray):
        x: np.ndarray = lsq_dense(amat, b, rcond=None)[0]
    else:
        x = lsq_sparse(amat, b)[0]

    return x, b - amat.dot(x)


def l2_min_cg(
    amat: Any,
    b: np.ndarray,
    *,
    enable_logging: bool = False,
    x0: np.ndarray | None = None,
    maxiter: int | None = None,
) -> tuple[np.ndarray, np.ndarray, Any]:
    """L2 minimization.

    Find x so that Ax-b has least L2 norm. Use CG when A == A.T, otherwise
    use CG to solve the normal equations, ie, use CG to solve A.T A x - A.T b.

    Note that if A == A.T, this method will assume A is positive definite,
    but will not verify that this condition is holds. If A is not positive definiite,

    Note also that if A is rank deficient, there is a subspace of solutions, and

    Parameters
    ----------
    A : matrix-like
        Sparse real system to solve.
    b : vector-like
        Right-hand side.
    enable_logging : bool
        Optional, Enable logger to capture diagnostic messages.
    x0 : vector-like
        Optional, initial guess
    maxiter : integer
        Optional, maximum number of iterations

    Returns
    -------
    x : array
        L2 minimizer of Ax - b
    r : array
        Ax - b
    P : scipy.sparse.linalg.SuperLU
        Preconditioner for the trans(A)A
    """
    assert amat.shape[0] == b.shape[0]

    if np.size(amat) == 0:
        if enable_logging:
            logger.warning("A is empty; returning all zeros")
        return np.zeros(amat.shape[1]), np.zeros(amat.shape[0]), None

    if np.all(b == 0):
        if enable_logging:
            logger.info("b is identically zero, returning zero")
        return np.zeros(amat.shape[1]), np.zeros(amat.shape[0]), None

    # If symmetric
    if amat.shape[0] == amat.shape[1] and np.max(np.abs(amat - amat.T)) == 0:
        use_normal_eqs: bool = False
        mat: np.ndarray | csr_matrix | csc_matrix = amat.copy()
        rhs: np.ndarray = b
    else:
        use_normal_eqs = True
        mat = np.dot(np.transpose(amat), amat)
        rhs = np.dot(np.transpose(amat), b)

    # Pre-conditioner
    pre = spilu(mat, fill_factor=100)

    x, info = cg(
        mat,
        rhs,
        rtol=1e-7,
        atol=1e-7,
        x0=x0,
        maxiter=maxiter,
        M=LinearOperator(mat.shape, pre.solve),
    )

    if info < 0 and enable_logging:
        logger.error("Something bad happened with CG!")
        return x0, None, pre

    r: np.ndarray = b - amat.dot(x)

    # Compute the residual of the normal equations. That is compute
    # A.T b - A.T A x, where x is our solution and b is the rhs.
    #
    # Note that the original system A x - b may not have a solution, while
    # the normal equations always have at least one. From this there are two
    # reasons we may not have solved the system, 1, there is not solution, 2,
    # CG performed badly.
    #
    # Since the normal equations always have a solution, and since the residual
    # of the normal equations is zero only for solutions of the normal equations,
    # the normal residual (computed below) allows the caller to distinguish between
    # 1 and 2 above.
    rn = rhs - np.dot(mat, x) if use_normal_eqs else None

    if enable_logging:
        if maxiter is not None:
            logger.info(f"Relative residual size {lpnorm(r, 2) / lpnorm(b, 2)}. ")
            if use_normal_eqs:
                logger.info(
                    "Relative size of residual of normal eqs"
                    f" {lpnorm(rn, 2) / lpnorm(rhs, 2)}, "
                )
        else:
            logger.info(f"Relative residual size {lpnorm(r, 2) / lpnorm(b, 2)}")
            logger.info(
                "Relative size of residual of normal eqs"
                f" {lpnorm(rn, 2) / lpnorm(rhs, 2)}, "
            )

    return x, r, pre


def dirichlet(
    amat: Any,
    b: np.ndarray,
    xf: np.ndarray,
    mask: np.ndarray,
    *,
    enable_logging: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Find x that minimizes Ax - b in an L2 sense, subject to x[mask] == xf[mask].

    Note, this has only been tested for A as a discrete Laplacian.

    Parameters
    ----------
    amat : array-like[m, m]
        Square matrix.
    b : array-like[m]
        Right hand side.
    xf : array-like [m]
        Fixed data, only xf[mask] is significant as input.
    mask : array-like[m] of bool
        mask[i] is true if x[i] should be forced to equal xf[i]
    enable_logging: bool
        Optional, enable logger to capture diagnostic messages.

    Returns
    -------
    x : array-like[m]
        Solution
    res : array-like[n]
        Residual b - Ax
    """
    assert amat.shape[0] == amat.shape[1]

    rhs = b - amat[:, mask].dot(xf[mask])

    x = np.zeros(xf.size)

    x[~mask] = l2_min_cg(
        amat[:, ~mask][~mask, :],
        rhs[~mask],
        maxiter=100,
        enable_logging=enable_logging,
    )[0]
    x[mask] = xf[mask]

    return x, b - amat.dot(x)


def dirichlet_graph(
    amat: csc_matrix,
    xf: np.ndarray,
    mask: np.ndarray,
    maxiter: int | None = 100,
    *,
    enable_logging: bool = False,
) -> np.ndarray:
    """Specialized implementation of dirichlet.

    Specifically to be used for graph Laplacian matrices.

    Parameters
    ----------
    amat : array-like [m, m]
        Square matrix.
    xf : array-like [n, m]
        Fixed data, only xf[:, mask] is significant as input.
    mask : array-like[m] of bool
        mask[i] is true if x[:, i] should be forced to equal xf[:, i]
    enable_logging: bool
        Optional, enable logger to capture diagnostic messages.

    Returns
    -------
    x: array-like [n, m]
    """
    assert amat.shape[0] == amat.shape[1]
    assert amat.shape[0] == xf.shape[1]

    corrections = np.zeros(xf.shape, dtype=np.float32)
    corrections[:, :] = xf[:, :]

    # This part is from l2_min_cg
    # We reuse the pre-conditioner here
    mat = amat[:, ~mask][~mask, :].copy()
    pre = spilu(mat, fill_factor=100)

    for kk in range(xf.shape[0]):
        b = -amat[:, mask].dot(xf[kk, mask])[~mask]

        assert mat.shape[0] == b.shape[0]
        if np.size(amat) == 0 and enable_logging:
            logger.warning("A is empty; returning all zeros")
            corrections[kk, ~mask] = 0
            continue

        if np.all(b == 0) and enable_logging:
            logger.info("b is identically zero, returning zero.")
            corrections[kk, ~mask] = 0
            continue

        x, info = cg(
            mat,
            b,
            rtol=1e-7,
            atol=1e-7,
            maxiter=maxiter,
            M=LinearOperator(mat.shape, pre.solve),
        )

        r: np.ndarray = b - mat.dot(x)
        if enable_logging and (maxiter is not None):
            logger.info(f"Relative residual size {lpnorm(r, 2) / lpnorm(b, 2)}. ")

        corrections[kk, ~mask] = x

    return corrections
