"""Utilities for graph manipulation."""


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
