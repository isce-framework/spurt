import numpy as np


def temporal_coherence(
    x: np.ndarray, *args: tuple[np.ndarray, np.ndarray, np.ndarray | float]
):
    """Temporal coherence function to minimize.

    args[0]: matrixA
    args[1]: b
    args[2]: weight
    """
    res = np.dot(args[0], x) - args[1]
    return -np.abs(np.sum(args[2] * np.exp(1j * res)))
