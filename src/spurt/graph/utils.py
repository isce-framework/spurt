"""Utilities for graph manipulation."""

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix

from ._interface import GraphInterface

__all__ = [
    "order_points",
    "graph_laplacian",
]


def order_points(p: tuple[int, int]) -> tuple[int, int]:
    """Order points/nodes/vertices by index.

    Given a pair of numbers, return a 2-tuple so the first is lower. The use
    case is that the pair contains pairs of indices representing undirected
    links, where (a, b) is the same as (b, a). This ordering, and returning
    a tuple allows us to comparison with ==.

    Parameters
    ----------
    p : array-like[2]
        Data to be ordered, must be comparable.

    Returns
    -------
    (int, int) : Ordered 2-tuple.
    """
    if p[0] <= p[1]:
        return (p[0], p[1])
    return (p[1], p[0])


def graph_laplacian(graph: GraphInterface) -> csc_matrix:
    """Compute the graph laplacian."""
    links = graph.links
    nlinks = len(links)
    data = np.ones(2 * nlinks, dtype=int)
    data[0::2] = -1
    amat = csr_matrix(
        (
            data,
            (np.arange(2 * nlinks, dtype=int) // 2, links.flatten()),
        )
    )
    return amat.T.dot(amat)
