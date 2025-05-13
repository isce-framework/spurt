from pathlib import Path

import h5py
import numpy as np

import spurt

from ._output import Tile, write_merged_band, write_single_tile
from ._settings import GeneralSettings, MergerSettings

logger = spurt.utils.logger

__all__ = ["merge_tiles"]


def merge_tiles(
    stack: spurt.io.SLCStackReader,
    g_time: spurt.graph.GraphInterface,
    gen_settings: GeneralSettings,
    mrg_settings: MergerSettings,
) -> list[Path]:
    """Merge the different tiles."""
    if mrg_settings.method != "dirichlet":
        errmsg = "dirichlet is the only merge method supported."
        raise NotImplementedError(errmsg)

    # Load tile json
    tiledata = spurt.utils.TileSet.from_json(gen_settings.tiles_jsonname)

    # Tile manager for each tile
    tiles: dict[int, Tile] = {}
    for ii in range(tiledata.ntiles):
        # Use band 0 as place holder
        tiles[ii] = Tile(str(gen_settings.tile_filename(ii)))

    # Preparing file names
    ifgs = g_time.links
    dates = stack.dates
    fnames: list[Path] = []
    for ifg in ifgs:
        fnames.append(gen_settings.unw_filename(dates[ifg[0]], dates[ifg[1]]))

    like_slc_file = stack.slc_files[dates[-1]]
    # If we only have to write one tile
    # Nothing to merge - just write to geotiff
    if len(tiles) == 1:
        logger.info(f"Writing single tile output to {gen_settings.output_folder}")
        write_single_tile(tiles[0], fnames, tiledata.shape, like=like_slc_file)
        return fnames

    # Create overlap map for the graph
    overlap_map = _get_overlap_map(gen_settings)
    max_degree = _get_max_degree(tiles, tiledata.shape)
    logger.info(f"Maximum degree for Dirichlet iterations: {max_degree}")

    # Read bulk offsets from hdf5 file
    with h5py.File(str(gen_settings.offsets_filename), "r") as fid:
        grp = fid[mrg_settings.bulk_method]
        bulk_offsets = grp["offsets"][...]

    # Write batch-by-batch
    nifgs = len(fnames)

    batch_size = min(mrg_settings.num_parallel_ifgs, nifgs)
    if batch_size < 1:
        batch_size = nifgs
    logger.info(f"Merging batches of {batch_size} interferograms")

    batch_start = np.arange(0, nifgs, batch_size, dtype=int)

    for bnum, bstart in enumerate(batch_start):
        bend = min(bstart + batch_size, nifgs)
        if (bend - bstart) <= 0:
            continue

        # Process bstart to bend interferograms
        if batch_size > 1:
            logger.info(f"Merging batch {bnum + 1} from {bstart + 1} to {bend}")

        # Check if batch already processed
        idx = []
        for ii in range(bstart, bend):
            fname = fnames[ii]
            if fname.is_file():
                logger.info(
                    f"{fname!s} already exists. Skipping merging band {ii + 1}."
                )
            else:
                idx.append(ii)

        if not idx:
            logger.info(f"Batch {bnum + 1} already processed. Skipping ..")
            continue

        # Initialize each tile for the right band with bulk offset
        for jj, tile in tiles.items():
            tile.reset_band_index(idx)
            if bulk_offsets:
                tile.add_correction(np.float32(-2 * np.pi * bulk_offsets[idx, jj]))
            else:
                # This is to set tile metadata
                tile.add_correction(np.float32(0.0))
            tile.increment_correction()

        if bulk_offsets:
            # Adjust the tiles
            _adjust_tiles(
                tiles, overlap_map, gen_settings, max_degree, debug_stats=False
            )

        # Write file to output band-by-band
        for ii in idx:
            fname = fnames[ii]
            write_merged_band(tiles, fname, ii, tiledata.shape, like=like_slc_file)

    return fnames


def _adjust_tiles(
    tiles: dict[int, Tile],
    overlap_map: dict[int, list[int]],
    gen_settings: GeneralSettings,
    max_degree: int,
    *,
    debug_stats: bool = False,
) -> None:
    """Dirichlet-based tile adjustment described in [1]_.

    References
    ----------
    .. [1] M. T. Calef, Olsen K. M., & Agram P. S., "Merging Point Data for
           InSAR Deformation Processing", in arXiv preprints arXiv:2405.06838, 2024.
    """
    # Start overlap processing
    for overlap_degree in range(int(max_degree), 1, -1):
        logger.info(f"Generating corrections for overlap degree: {overlap_degree}")

        for ii, tile_i in tiles.items():
            logger.info(f"Processing tile: {ii}")
            assert tile_i.correction_level == 1, "Bad correction level in compress mode"
            data = tile_i.uw_phase
            c = np.ones(tile_i.coords.shape[0], dtype=np.int16)
            s = data.astype(np.float32)
            for jj in overlap_map[ii]:
                tile_j = tiles[jj]
                if jj > ii:
                    idx_i, idx_j = _get_common_overlap(gen_settings, ii, jj)
                else:
                    idx_j, idx_i = _get_common_overlap(gen_settings, jj, ii)
                c[idx_i] += 1
                s[:, idx_i] += tile_j.uw_phase[:, idx_j]

            if np.sum(c >= overlap_degree) == 0:
                continue
            overlap_average = s / c
            raw_correction = overlap_average - data

            logger.info("Solving Dirichlet problem")

            correction = spurt.utils.merge.dirichlet_graph(
                tile_i.graph_laplacian,
                raw_correction,
                c >= overlap_degree,
                enable_logging=True,
            )

            tile_i.add_correction(correction)

        # Pop last element to keep corrections for using a lot of memory
        for _, tile in tiles.items():
            tile.compress_corrections()

            # Uncomment to track each level of correction
            # tile.increment_correction()

    if debug_stats:
        # Verification of overlap differences
        for ii, tile_i in tiles.items():
            for jj in overlap_map[ii]:
                if jj <= ii:
                    continue
                idx_i, idx_j = _get_common_overlap(gen_settings, ii, jj)
                diff = tile_i.uw_phase[idx_i] - tiles[jj].uw_phase[idx_j]
                logger.info(f"Overlap between {ii}, {jj} : {np.max(np.abs(diff))}")


def _get_overlap_map(
    gen_settings: GeneralSettings,
) -> dict[int, list[int]]:
    """Return overlap map for each tile.

    Returns
    -------
    x: dict[int, list[int]]
        The key represents the tile index and the value represents the list of
        indices of tiles that the tile overlaps with.
    """
    with h5py.File(str(gen_settings.overlap_filename), "r") as fid:
        npts = fid["conn_comp"].size
        olaps = fid["overlaps"][...]

    # Create output variable
    overlap_map: dict[int, list[int]] = {}
    for ii in range(npts):
        overlap_map[ii] = []

    # Iterate over links
    for edge in olaps:
        overlap_map[edge[0]].append(edge[1])
        overlap_map[edge[1]].append(edge[0])

    return overlap_map


def _get_common_overlap(
    gen_settings: GeneralSettings, ii: int, jj: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return common indices."""
    fname = gen_settings.overlap_filename
    grpname = gen_settings.overlap_groupname(ii, jj)
    with h5py.File(str(fname), "r") as fid:
        grp = fid[grpname]
        c1 = grp["c1"][:]
        c2 = grp["c2"][:]

    return c1, c2


def _get_max_degree(tiles: dict[int, Tile], shape: tuple[int, int]) -> int:
    """Compute max degree of a pixel.

    This is same as maximum number of tiles that a pixel can be a part of.
    """
    count = np.zeros(shape, dtype=np.uint8)
    for tile in tiles.values():
        coords = tile.coords.astype(np.int32)
        count[coords[:, 0], coords[:, 1]] += 1

    return int(count.max())
