from __future__ import annotations

import numpy as np


def phase_diff(z0: np.ndarray, z1: np.ndarray) -> np.ndarray:
    """
    Compute the wrapped phase difference for between two numbers in radians.

    Parameters
    ----------
    z0 : np.ndarray
        Can be complex or real
    z1 : np.ndarray
        Same type as z0
    """
    if np.iscomplexobj(z0):
        p0 = np.angle(z0)
        p1 = np.angle(z1)
        d = p1 - p0
    else:
        d = z1 - z0
    return d - np.round(d / (2 * np.pi)) * 2 * np.pi


def sign_nonzero(x: float) -> int:
    """Return +1 if x > 0 and -1 for x < 0.

    Non-zero value should not be passed in.
    """
    if x > 0:
        return 1
    return -1


def flood_fill(indata: np.ndarray, links: np.ndarray, flows: np.ndarray):
    """Flood fill unwrapping.

    Given input data amd links for those links start at an arbitrary point
    and walk along links adding the gradient. When we encounter a cycle,
    make sure that walking either path around the cycle will result in
    the same answer. Return the point values.

    Parameters
    ----------
    indata: np.ndarray
        Input wrapped or approximately unwrapped phase data.
    links : np.ndarray
        Links specifed as tuples of point indices.
    flows : np.ndarray
        Integer cycles to be added to each link.

    Returns
    -------
    unwrapped : array
        Per point unwrapped data.
    """
    if len(links) != len(flows):
        errmsg = f"Dimension mismatch - {links.shape} vs {flows.shape}"
        raise ValueError(errmsg)

    pts = np.arange(len(indata))

    pts_to_nbrs: dict = {pt: [] for pt in pts}
    for ii, link in enumerate(links):
        gradient = phase_diff(indata[link[0]], indata[link[1]]) + 2 * np.pi * flows[ii]
        pts_to_nbrs[link[0]].append((link[1], gradient))
        pts_to_nbrs[link[1]].append((link[0], -gradient))

    done = np.zeros(len(pts))
    to_do = []
    unwrapped = np.zeros(len(pts))
    pts_to_paths: dict = {pt: [] for pt in pts}

    to_do.append(0)
    done[0] = 1
    pts_to_paths[0].append(0)

    multi_paths = []

    while np.any(done == 0):
        i = to_do.pop(0)
        for j, g in pts_to_nbrs[i]:
            u = unwrapped[i] + g
            if done[j] == 0:
                unwrapped[j] = u
                done[j] = 1
                pts_to_paths[j] = pts_to_paths[i] + [j]
                to_do.append(j)
            elif np.abs(u - unwrapped[j]) > 1e-3:
                multi_paths.append(
                    (pts_to_paths[j], pts_to_paths[i] + [j], u - unwrapped[j])
                )

    if len(multi_paths) > 0:
        errmsg = f"Error: Encountered {len(multi_paths)} closure errors"
        raise ValueError(errmsg)

    # Adding the source node value
    if np.iscomplexobj(indata):
        unwrapped += np.angle(indata[links[0, 0]])
    else:
        unwrapped += indata[links[0, 0]]

    return unwrapped


def centroid_costs(
    points: np.ndarray,
    cycles: np.ndarray | list[list[int]],
    dual_edges: np.ndarray,
    scale: float = 100.0,
) -> np.ndarray:
    """Estimate edge costs based on centroid distance.

    Should probably relocate to a common area where cost functions are
    maintained at a later date.
    """
    cost = np.zeros(dual_edges.shape[0], dtype=int)
    centroids = np.zeros((len(cycles), 2))
    for ii, cycle in enumerate(cycles):
        centroids[ii] = np.mean(points[cycle], axis=0)

    for ii, edge in enumerate(dual_edges):
        # If connected to grounding node
        if edge[1] == 0:
            continue

        d = np.linalg.norm(centroids[abs(edge[0]) - 1] - centroids[abs(edge[1]) - 1])
        cost[ii] = np.rint(1 + scale / d)

    return cost


def distance_costs(
    points: np.ndarray,
    edges: np.ndarray,
    scale: float = 100.0,
) -> np.ndarray:
    """Estimate edge costs based on distance between points.

    Should probably relocate to a common area where cost functions are
    maintained at a later date.
    """
    cost = np.zeros(edges.shape[0], dtype=int)
    for ii, edge in enumerate(edges):
        d = np.linalg.norm(points[edge[0]] - points[edge[1]])
        cost[ii] = np.rint(1 + scale / d)

    return cost
