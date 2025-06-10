import h5py
import numpy as np
from numpy.typing import ArrayLike
from ortools.linear_solver import pywraplp

import spurt

from ._settings import GeneralSettings, MergerSettings

logger = spurt.utils.logger

__all__ = ["get_bulk_offsets"]


def get_bulk_offsets(
    stack: spurt.io.SLCStackReader,
    gen_settings: GeneralSettings,
    mrg_settings: MergerSettings,
) -> None:
    """Compute bulk phase offsets between overlapping tiles.

    We compute the bulk offset used to adjust individually unwrapped tiles in
    the merge process. While we computed phase difference deciles, we only use
    the median phase difference here for adjusting tiles.
    """
    # Check if offsets already computed
    if gen_settings.offsets_filename.is_file():
        logger.info("Offsets file already exists. Skipping ...")
        return

    # Load tile info
    tiledata = spurt.utils.TileSet.from_json(gen_settings.tiles_jsonname)

    # Single tile processing
    if tiledata.ntiles == 1:
        logger.info("Single tile used. Skipping bulk offsets ...")
        return

    # Load offset data for inversion
    offsets = []
    olap_counts = []
    overlap_file = str(gen_settings.overlap_filename)
    with h5py.File(str(overlap_file), "r") as fid:
        labels: np.ndarray = fid["conn_comp"][...]
        overlaps: np.ndarray = fid["overlaps"][...]
        for pair in overlaps:
            t1, t2 = pair
            grp_name = gen_settings.overlap_groupname(t1, t2)
            grp = fid[grp_name]

            # Just use the median for now
            olap_counts.append(grp["c1"].shape[0])
            offsets.append(grp["stats"][:, 5])

    if offsets:
        # Get number of bands to merge
        nbands = len(offsets[0])
        ntiles = tiledata.ntiles
        counts = np.zeros(ntiles)

        # Add counts of valid pixels in each tile
        arr = stack.read_temporal_coherence(np.s_[:, :]) > stack.temp_coh_threshold
        for ii, tile in enumerate(tiledata.tiles):
            counts[ii] = np.sum(arr[tile.space])

        # If integer solver requested
        logger.info(f"Solving for bulk offsets with method: {mrg_settings.method}")
        if mrg_settings.bulk_method == "integer":
            bulk_offset: np.ndarray = np.zeros((nbands, ntiles), dtype=np.int32)
            obj: np.ndarray = np.zeros(nbands, dtype=np.int32)

            # Solve band by band
            for ii in range(nbands):
                logger.info(f"Computing bulk offsets for band {ii + 1}")
                off: np.ndarray = np.array([x[ii] for x in offsets])
                bulk_offset[ii, :], obj[ii] = _solve_int_offsets(
                    overlaps, labels, off, counts
                )
        # Use L2 method
        elif mrg_settings.bulk_method == "L2":
            # These can be floating point
            off = np.zeros((len(overlaps), nbands))
            for ii in range(len(overlaps)):
                off[ii, :] = offsets[ii]

            bulk_offset, obj = _solve_l2_min(overlaps, off, ntiles, counts)

        else:
            errmsg = f"Unsupported bulk offset method {mrg_settings.bulk_method}"
            raise RuntimeError(errmsg)

    else:
        # Write empty file as availability of this file marks completion of
        # this stage
        bulk_offset = np.empty(0)
        obj = np.empty(0)

    # Write HDF5 file with bulk offsets
    with h5py.File(str(gen_settings.offsets_filename), "w") as fid:
        grp = fid.create_group(mrg_settings.bulk_method)
        grp["offsets"] = bulk_offset
        grp["residuals"] = obj


def _solve_l2_min(
    olaps: ArrayLike, off: ArrayLike, ntiles: int, counts: ArrayLike
) -> tuple[np.ndarray, float]:
    """Return minimum L2 solution for offsets between tiles.

    Parameters
    ----------
    olaps: array
        Integer array of size (noverlaps, 2) with each row containing tile
        indices.
    off: array
        Integer array of length noverlaps indicating integer offsets for
        corresponding overlap.
    counts: array
        Integer array of length ntiles indicating number of valid pixels in
        each tile.

    Returns
    -------
    x: array
        Floating point array of size ntiles containing number of cycles to be added to
        individually unwrapped tiles before merging.
     val: float
        L1 norm of residuals from each overlapping tile pair. A non-zero value is an
        indication of possible inconsistency.
    """
    nlinks: int = len(olaps)
    # Create the system to solve
    cmat = np.zeros((nlinks, ntiles))

    for ind in range(nlinks):
        ii, jj = olaps[ind, :]
        cmat[ind, ii] = -1
        cmat[ind, jj] = 1

    results: tuple[np.ndarray, np.ndarray] = spurt.utils.merge.l2_min(cmat, off)

    # Set largest component to zero
    connnum = np.argmax(counts)
    logger.info(f"Largest tile by count: {connnum}")
    offset: np.ndarray = results[0] - results[0][connnum, :]

    return np.transpose(offset), np.sum(np.abs(results[1]), axis=0)


def _solve_int_offsets(
    olaps: ArrayLike, labels: ArrayLike, off: ArrayLike, counts: ArrayLike
) -> tuple[np.ndarray, int]:
    """Solve bulk offset problem using MIP solver.

    We use the most generic formulation here similar to edgelist unwrapping as
    the size of the problems are expected to be small and lets us overcome any
    assumptions about planar graphs when looking at all possible overlaps.

    Parameters
    ----------
    olaps: array
        Integer array of size (noverlaps, 2) with each row containing tile indices.
    off: array
        Integer array of length noverlaps indicate integer offsets for corresponding
        overlap.
    counts: array
        Integer array of length ntiles indicating number of valid pixels in each tile.

    Returns
    -------
    x: array
        Integer array of size ntiles containing number of cycles to be added to
        individually unwrapped tiles before merging.
    val: int
        L1 norm of flows for each overlapping tile pair. A non-zero value is an
        indication of possible inconsistency.
    """
    npts: int = len(labels)
    nlinks: int = len(olaps)
    int32_max = np.iinfo(np.int32).max
    solver = pywraplp.Solver.CreateSolver("SAT")

    # Node cycle offsets
    ns = []
    for ii in range(npts):
        varname = f"n_{ii:02d}"
        ns.append(solver.IntVar(-int32_max, int32_max, varname))

    # Flow variables
    pf = []
    nf = []
    for pair in olaps:
        t1, t2 = pair
        pf.append(solver.IntVar(0, int32_max, f"pf_{t1:02d}_{t2:02d}"))
        nf.append(solver.IntVar(0, int32_max, f"nf_{t1:02d}_{t2:02d}"))

    # Add constraints from offsets
    for ind in range(nlinks):
        t1, t2 = olaps[ind, :]
        solver.Add(ns[t2] - ns[t1] + pf[ind] - nf[ind] == off[ind])

    # Add constraint for largest components
    ncomp = np.max(labels) + 1
    for ii in range(ncomp):
        conn = np.where(labels == ii)[0]
        largest_comp = np.argmax(counts[conn])
        connnum = conn[largest_comp]
        solver.Add(ns[connnum] == 0)

    # Set objective function
    obj = solver.Objective()
    for ii in pf:
        obj.SetCoefficient(ii, 1)

    for ii in nf:
        obj.SetCoefficient(ii, 1)
    obj.SetMinimization()

    # Solve
    status = solver.Solve()

    if status != solver.OPTIMAL:
        errmsg = "The problem does not have an optimal solution"
        raise RuntimeError(errmsg)

    val: int = obj.Value()
    logger.info(f"Estimated flow correction: {val}")

    return np.array([x.solution_value() for x in ns], dtype=np.int32), val
