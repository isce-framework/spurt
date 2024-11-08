from __future__ import annotations

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix


def neg_temporal_coherence(
    x: np.ndarray,
    amat: np.ndarray | csr_matrix | csc_matrix,
    b: np.ndarray,
    wts: np.ndarray | float,
) -> float:
    """Negative of temporal coherence function.

    For definition of temporal coherence, see [1]_. We implement the
    function as negative of temporal coherence, for ease of use with
    scipy.minimize.

    References
    ----------
    [1] Ferretti, A., Prati, C. and Rocca, F., 2001. Permanent scatterers
        in SAR interferometry. IEEE Transactions on geoscience and remote
        sensing, 39(1), pp.8-20.
    """
    res = amat.dot(x) - b
    return -np.abs(np.sum(wts * np.exp(1j * res)))


def neg_temporal_coherence_with_jacobian(
    x: np.ndarray,
    amat: np.ndarray | csr_matrix | csc_matrix,
    b: np.ndarray,
    wts: np.ndarray | float,
) -> tuple[float, np.ndarray]:
    """Negative of temporal coherence with its Jacobian.

    We write the function as negative of temporal coherence, for ease
    of use with scipy.minimize
    """
    phase_diff = amat.dot(x) - b
    cpdw = np.cos(phase_diff) * wts
    spdw = np.sin(phase_diff) * wts

    t1 = np.sum(cpdw)
    t2 = np.sum(spdw)
    r = -np.sqrt(t1**2 + t2**2)

    t3 = 1.0 / (2.0 * r)
    t4 = t1 * amat.T.dot(2.0 * -spdw)
    t5 = t2 * amat.T.dot(2.0 * cpdw)

    return (r, t3 * (t4 + t5))
