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
