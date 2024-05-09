import json

import h5py
import numpy as np
from numpy.typing import ArrayLike
from ortools.linear_solver import pywraplp

import spurt

from ._settings import GeneralSettings, MergerSettings

logger = spurt.utils.logger


def get_bulk_offsets(
    gen_settings: GeneralSettings,
    mrg_settings: MergerSettings,
) -> None:

    # Check if offsets already computed
    if gen_settings.offsets_filename.is_file():
        logger.info("Offsets file already exists. Skipping ...")
        return

    # Load tile info
    with gen_settings.tiles_jsonname.open(mode="r") as fid:
        tiledata = json.load(fid)

    # Single tile processing
    if len(tiledata["tiles"]) == 1:
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

    # Get number of bands to merge
    nbands = len(offsets[0])
    ntiles = len(tiledata["tiles"])
    counts = np.array([x["count"] for x in tiledata["tiles"]])

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
        bulk_offset = np.zeros((nbands, ntiles))
        obj = np.zeros(nbands)
        off = np.zeros((len(overlaps), nbands))
        for ii in range(len(overlaps)):
            off[ii, :] = offsets[ii]

        bulk_offset, obj = _solve_l2_min(overlaps, off, ntiles)

    else:
        errmsg = f"Unsupported bulk offset method {mrg_settings.bulk_method}"
        raise RuntimeError(errmsg)

    # Write HDF5 file with bulk offsets
    with h5py.File(str(gen_settings.offsets_filename), "w") as fid:
        grp = fid.create_group(mrg_settings.bulk_method)
        grp["offsets"] = bulk_offset
        grp["residues"] = obj


def _solve_l2_min(
    olaps: ArrayLike, off: ArrayLike, ntiles: int
) -> tuple[np.ndarray, float]:
    """Return minimum L2 solution."""
    nlinks: int = len(olaps)
    # Create the system to solve
    cmat = np.zeros((nlinks, ntiles))

    for ind in range(nlinks):
        ii, jj = olaps[ind, :]
        cmat[ind, ii] = -1
        cmat[ind, jj] = 1

    results: tuple[np.ndarray, np.ndarray] = spurt.utils.merge.l2_min(cmat, off)
    return np.transpose(results[0]), np.sum(np.abs(results[1]), axis=0)


def _solve_int_offsets(
    olaps: ArrayLike, labels: ArrayLike, off: ArrayLike, counts: ArrayLike
) -> tuple[np.ndarray, int]:
    """Solve bulk offset problem."""
    npts: int = len(labels)
    nlinks: int = len(olaps)
    solver = pywraplp.Solver.CreateSolver("SAT")

    # Node cycle offsets
    ns = []
    for ii in range(npts):
        varname = f"n_{ii:02d}"
        ns.append(solver.IntVar(-100000, 100000, varname))

    # Flow variables
    pf = []
    nf = []
    for pair in olaps:
        t1, t2 = pair
        pf.append(solver.IntVar(0, 20, f"pf_{t1:02d}_{t2:02d}"))
        nf.append(solver.IntVar(0, 20, f"nf_{t1:02d}_{t2:02d}"))

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
    if val != 0:
        logger.info(f"Non-zero flow correction: {val}")

    return np.array([x.solution_value() for x in ns], dtype=np.int32), val
