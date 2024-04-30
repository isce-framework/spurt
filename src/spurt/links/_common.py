from typing import Any

import numpy as np


def temporal_coherence(x: np.ndarray, args: tuple[Any, ...]):
    """Temporal coherence function to minimize.

    args[0]: matrixA
    args[1]: b
    args[2]: weight
    """
    res = np.dot(args[0], x) - args[1]
    return -np.sum(args[2] * np.exp(1j * res))
