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
    the same answer. Return the point values. This version has a lot of
    debugging friendly features.

    Parameters
    ----------
    indata: np.ndarray
        Input wrapped phase data as 1D array. Same size as number of points in graph.
    links : np.ndarray
        Links specifed as tuples of point indices. The links should represented
        a fully connected graph.
    flows : np.ndarray
        Integer cycles to be added to each link.

    Returns
    -------
    unwrapped : np.ndarray
        Unwrapped phase in radians. Same size as indata.
    """
    if len(links) != len(flows):
        errmsg = f"Dimension mismatch - {links.shape} vs {flows.shape}"
        raise ValueError(errmsg)

    # Indices of points
    pts = np.arange(len(indata))

    # Mapping of points to its immediate neighbors and gradient on the link
    pts_to_nbrs: dict = {pt: [] for pt in pts}

    # Iterate over the links
    for ii, link in enumerate(links):
        # Get the unwrapped phase gradient by adding flows to
        gradient = phase_diff(indata[link[0]], indata[link[1]]) + 2 * np.pi * flows[ii]

        # Add the link in either direction with appropriate gradient sign
        pts_to_nbrs[link[0]].append((link[1], gradient))
        pts_to_nbrs[link[1]].append((link[0], -gradient))

    # To track if a pixel has been unwrapped already
    done = np.zeros(len(pts), dtype=bool)

    # Track indices of point yet to be unwrapped
    to_do = []

    # To store unwrapped value that will be returned
    unwrapped = np.zeros(len(pts))

    # To track the integration path to each point
    pts_to_paths: dict = {pt: [] for pt in pts}

    # Start with point with index 0 - this is the reference
    to_do.append(0)
    done[0] = True
    pts_to_paths[0].append(0)

    # This is for reporting in case things go wrong
    # This should never happen but having this on helps
    # with fast debugging of sign errors
    multi_paths = []

    # Continue till all points are unwrapped
    while not np.all(done):

        # Get the first pixel from the to do list
        i = to_do.pop(0)

        # For each of its neighbors and associated gradient
        for j, g in pts_to_nbrs[i]:

            # Unwrap by adding the gradient
            u = unwrapped[i] + g

            # If not unwrapped, now label as unwrapped
            if not done[j]:
                unwrapped[j] = u
                done[j] = True
                # Track the path to the point
                pts_to_paths[j] = pts_to_paths[i] + [j]
                # Add new point to to do list
                to_do.append(j)

            # If already unwrapped, verify values are numerically compatible
            elif np.abs(u - unwrapped[j]) > 1e-3:
                # If you get here, unwrapping path dependent - track for
                # debugging
                multi_paths.append(
                    (pts_to_paths[j], pts_to_paths[i] + [j], u - unwrapped[j])
                )

    # Report issues if any - this is bad sign if we get here
    if len(multi_paths) > 0:
        errmsg = f"Error: Encountered {len(multi_paths)} closure errors"
        raise ValueError(errmsg)

    # Adding the source node value - we started at index 0
    if np.iscomplexobj(indata):
        unwrapped += np.angle(indata[0])
    else:
        unwrapped += indata[0]

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
