import h5py
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

import spurt

from ._settings import GeneralSettings, MergerSettings

logger = spurt.utils.logger

__all__ = ["compute_phasediff_deciles"]


def compute_phasediff_deciles(
    gen_settings: GeneralSettings,
    mrg_settings: MergerSettings,
) -> None:
    """Compute overlap phase difference stats and save to h5.

    We compute histograms of phase difference between overlapping tiles.
    While we only use a constant bulk offset for reconciling differences, these
    histograms can be used for debugging and assessing quality of consistency
    betwen unwrapped results of individual tiles.
    """
    # Check if already processed
    if gen_settings.overlap_filename.is_file():
        logger.info("Overlap file already exists. Skipping ...")
        return

    # Load tile info
    tiledata = spurt.utils.TileSet.from_json(gen_settings.tiles_jsonname)

    # If single tile json - just return
    if tiledata.ntiles == 1:
        logger.info("Single tile used. Skipping overlaps ...")
        return

    t1 = -1
    t2 = -1
    ntiles = tiledata.ntiles
    conn_mat = np.zeros((ntiles, ntiles))
    overlaps = []
    overlap_file: str = str(gen_settings.overlap_filename)
    for pair in tiledata.get_overlaps():
        logger.info(f"Processing neighboring pair: {pair}")
        if pair[0] != t1:
            t1 = pair[0]
            tile1 = tiledata.tiles[t1]
            pt1, uw1 = _load_uw_tile(str(gen_settings.tile_filename(t1)), tile1)

        if pair[1] != t2:
            t2 = pair[1]
            tile2 = tiledata.tiles[t2]
            pt2, uw2 = _load_uw_tile(str(gen_settings.tile_filename(t2)), tile2)

        # Find common points
        c1, c2 = spurt.utils.merge.find_common_points(pt1, pt2)

        if len(c1) < mrg_settings.min_overlap_points:
            logger.info("Insufficient overlap. Skipping ...")
            continue

        # Track overlaps
        overlaps.append([t1, t2])
        conn_mat[t1, t2] = 1

        # difference stats
        cuw1 = uw1[:, c1]
        cuw2 = uw2[:, c2]
        stats = spurt.utils.merge.pairwise_unwrapped_diff_deciles(cuw1, cuw2)
        grpname = gen_settings.overlap_groupname(t1, t2)
        with h5py.File(overlap_file, "a") as fid:
            grp = fid.create_group(grpname)
            grp["c1"] = c1.astype(int)
            grp["c2"] = c2.astype(int)
            grp["stats"] = stats.astype(np.int16)

    # Identify connected components
    graph = csr_matrix(conn_mat)
    n_components, labels = connected_components(
        csgraph=graph, directed=False, return_labels=True
    )

    logger.info(f"Number of connected components: {n_components}")

    # Add list of overlaps at the end
    with h5py.File(overlap_file, "a") as fid:
        fid["conn_comp"] = labels
        fid["overlaps"] = np.array(overlaps, dtype=np.int16)


def _load_uw_tile(fname: str, tile: spurt.utils.BBox) -> tuple[np.ndarray, np.ndarray]:
    """Load data from tile h5 file."""
    with h5py.File(fname, "r") as fid:
        pts = fid["points"][...]
        uw = fid["uw_data"][...]
        off = fid["phase_offset"][...]

    # Apply phase offset to unwrapped phase
    uw += off.flatten()[:, None]

    # Apply bounds offset to locations
    pts[:, 0] = pts[:, 0] + tile.xmin
    pts[:, 1] = pts[:, 1] + tile.ymin

    return pts, uw
