import numpy as np


def temporal_coherence(
    x: np.ndarray, *args: tuple[np.ndarray, np.ndarray, np.ndarray | float]
) -> float:
    """Temporal coherence function to minimize.

    args[0]: matrixA
    args[1]: b
    args[2]: weight
    """
    res = np.dot(args[0], x) - args[1]
    return -np.abs(np.sum(args[2] * np.exp(1j * res)))


def temporal_coherence_with_jac(
    x: np.ndarray, *args: tuple[np.ndarray, np.ndarray, np.ndarray | float]
) -> tuple[float, np.ndarray]:
    """Temporal coherence with its Jacobian.

    args[0]: matrixA
    args[1]: b
    args[2]: weight
    """
    phase_diff = np.dot(args[0], x) - args[1]
    cpdw = np.cos(phase_diff) * args[2]
    spdw = np.sin(phase_diff) * args[2]

    t1 = np.sum(cpdw)
    t2 = np.sum(spdw)
    r = -np.sqrt(t1**2 + t2**2)

    t3 = 1.0 / (2.0 * r)
    t4 = t1 * args[0].T.dot(2.0 * -spdw)  # type: ignore[attr-defined]
    t5 = t2 * args[0].T.dot(2.0 * cpdw)  # type: ignore[attr-defined]

    return (r, t3 * (t4 + t5))
