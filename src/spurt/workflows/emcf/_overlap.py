import json
from pathlib import Path

import h5py
import numpy as np

import spurt

from ._settings import GeneralSettings, MergerSettings

logger = spurt.utils.logger


def compute_overlap_stats(
    gen_settings: GeneralSettings,
    mrg_settings: MergerSettings | None = None,
) -> None:
    """Compute overlap stats and save to h5."""
    if mrg_settings is None:
        mrg_settings = MergerSettings()

    # I/O files
    pdir = Path(gen_settings.output_folder)
    json_name = pdir / "tiles.json"
    tile_file_tmpl = str(pdir / "uw_tile_{}.h5")
    overlap_file = pdir / "overlaps.h5"

    if overlap_file.is_file():
        logger.info(f"{overlap_file!s} already exists. Skipping ...")
        return

    # Load tile info
    with json_name.open(mode="r") as fid:
        tiledata = json.load(fid)

    t1 = -1
    t2 = -1
    for pair in tiledata["neighbors"]:
        logger.info(f"Processing neighboring pair: {pair}")
        if pair[0] != t1:
            t1 = pair[0]
            bnds1 = tiledata["tiles"][t1]["bounds"]
            pt1, uw1 = _load_uw_tile(tile_file_tmpl.format(f"{t1 + 1:02d}"), bnds1)
            pt1[:, 0] += bnds1[0]
            pt1[:, 1] += bnds1[1]

        if pair[1] != t2:
            t2 = pair[1]
            bnds2 = tiledata["tiles"][t2]["bounds"]
            pt2, uw2 = _load_uw_tile(tile_file_tmpl.format(f"{t2 + 1:02d}"), bnds2)

        # Find common points
        c1, c2 = spurt.utils.merge.find_common_points(pt1, pt2)

        if len(c1) < mrg_settings.min_overlap_points:
            logger.info("Insufficient overlap. Skipping ...")
            continue

        # difference stats
        cuw1 = uw1[:, c1]
        cuw2 = uw2[:, c2]
        stats = spurt.utils.merge.pairwise_unwrapped_diff(cuw1, cuw2)
        grpname = f"{t1:02d}_{t2:02d}"
        with h5py.File(overlap_file, "a") as fid:
            grp = fid.create_group(grpname)
            grp["c1"] = c1.astype(np.int16)
            grp["c2"] = c2.astype(np.int16)
            grp["stats"] = stats.astype(np.int16)


def _load_uw_tile(fname: str, bnds: list[int]) -> tuple[np.ndarray, np.ndarray]:
    """Load data from tile h5 file."""
    with h5py.File(fname, "r") as fid:
        pts = fid["points"][...]
        uw = fid["uw_data"][...]
        off = fid["phase_offset"][...]

    # Apply phase offset to unwrapped phase
    uw += off

    # Apply bounds offset to locations
    pts[:, 0] + bnds[0]
    pts[:, 1] + bnds[1]

    return pts, uw
