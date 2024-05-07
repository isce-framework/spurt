"""Utilities for merging point clouds."""

import numpy as np


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
    ii : list[int]
        Indices of c1 that overlaps c1
    jj : list[int]
        Indices of c2 that overlaps c2, i.e. c1[ii] == c2[jj]
    """
    assert min(np.min(c1), np.min(c2)) >= 0

    m = 1 + max(np.amax(c1[:, 1]), np.amax(c2[:, 1]))
    fc1 = c1[:, 0] * m + c1[:, 1]
    fc2 = c2[:, 0] * m + c2[:, 1]
    return np.intersect1d(fc1, fc2, return_indices=True)[1:]


def pairwise_unwrapped_diff(b1: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """Stats for pairwise unwrapped differences.

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

    if b1.shape[1] != 11:
        errmsg = f"Need atleast 11 elements per band - received {b1.shape}"
        raise ValueError(errmsg)

    # Compute difference
    diff: np.ndarray = np.zeros(b1.shape, dtype=np.float32)

    # Get integer difference
    diff[:, :] = (b2 - b1) / (2 * np.pi)
    nint: np.ndarray = np.rint(diff).astype(np.int16)

    # Check that we are close to integers
    assert np.allclose(diff, nint, atol=0.01), "Arrays differ by non-integer cycles"

    # Indices into a sorted array
    inds = (np.linspace(0, 100, 11) * b1.shape[1]).astype(np.int)

    # Sort the differences
    diff = np.sort(nint, axis=-1)

    # Return histogram
    return diff[:, inds]
