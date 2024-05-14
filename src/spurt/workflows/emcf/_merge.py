import json
from pathlib import Path

import h5py
import numpy as np

import spurt

from ._output import Tile, write_merged_band, write_single_tile
from ._settings import GeneralSettings, MergerSettings

logger = spurt.utils.logger


def merge_tiles(
    stack: spurt.io.SLCStackReader,
    g_time: spurt.graph.GraphInterface,
    gen_settings: GeneralSettings,
    mrg_settings: MergerSettings,
) -> None:
    """Merge the different tiles."""
    if mrg_settings.method != "dirichlet":
        errmsg = "dirichlet is the only merge method supported."
        raise NotImplementedError(errmsg)

    # Load tile json
    with gen_settings.tiles_jsonname.open(mode="r") as fid:
        tiledata = json.load(fid)

    # Tile manager for each tile
    tiles: dict[int, Tile] = {}
    for ii in range(len(tiledata["tiles"])):
        # Use band 0 as place holder
        tiles[ii] = Tile(str(gen_settings.tile_filename(ii)), 0)

    # Preparing file names
    ifgs = g_time.links
    dates = stack.dates
    fnames: list[Path] = []
    for ifg in ifgs:
        fnames.append(gen_settings.unw_filename(dates[ifg[0]], dates[ifg[1]]))

    # If we only have to write one tile
    # Nothing to merge - just write to geotiff
    if len(tiles) == 1:
        logger.info(f"Writing single tile output to {gen_settings.output_folder}")
        write_single_tile(tiles[0], fnames, tiledata["shape"])
        return

    # Create overmap lap for the graph
    overlap_map: dict[int, list[int]] = _get_overlap_map(gen_settings)

    # Get maximum degree
    max_degree: int = max([len(vv) for kk, vv in overlap_map.items()])

    # Read bulk offsets from hdf5 file
    with h5py.File(str(gen_settings.offsets_filename), "r") as fid:
        grp = fid[mrg_settings.bulk_method]
        bulk_offsets = grp["offsets"][...]

    # Write band-by-band
    nifgs = len(fnames)
    for ii, fname in enumerate(fnames):
        if fname.is_file():
            logger.info(f"{fname!s} already exists. Skipping merging band {ii + 1}.")
            continue

        logger.info(f"Merging band {ii + 1} of {nifgs}")

        # Initialize each tile for the right band with bulk offset
        for jj, tile in tiles.items():
            tile.reset_band_index(ii)
            tile.add_correction(np.float32(-2 * np.pi * bulk_offsets[ii, jj]))
            tile.increment_correction()

        # Adjust the tiles
        _adjust_tiles(tiles, overlap_map, gen_settings, max_degree)

        # Write file to output
        write_merged_band(tiles, fname, ii, tiledata["shape"])


def _adjust_tiles(
    tiles: dict[int, Tile],
    overlap_map: dict[int, list[int]],
    gen_settings: GeneralSettings,
    max_degree: int,
) -> None:
    # Start overlap processing
    for overlap_degree in range(int(max_degree), 1, -1):
        logger.info(f"Generating corrections for overlap degree: {overlap_degree}")

        for ii, tile_i in tiles.items():
            logger.info(f"Processing tile: {ii}")
            c = np.ones(tile_i.coords.shape[0], dtype=np.int16)
            s = np.copy(tile_i.uw_phase.astype(np.float32))
            for jj in overlap_map[ii]:
                tile_j = tiles[jj]
                if jj > ii:
                    idx_i, idx_j = _get_common_overlap(gen_settings, ii, jj)
                else:
                    idx_j, idx_i = _get_common_overlap(gen_settings, jj, ii)
                c[idx_i] += 1
                s[idx_i] += tile_j.uw_phase[idx_j]

            if np.sum(c >= overlap_degree) == 0:
                continue
            overlap_average = s / c
            raw_correction = overlap_average - tile_i.uw_phase

            logger.info("Solving Dirichlet problem")
            correction = spurt.utils.merge.dirichlet(
                tile_i.graph_laplacian,
                np.zeros(s.size),
                raw_correction,
                c >= overlap_degree,
            )[0]

            tile_i.add_correction(correction.astype(np.float32))

        # Pop last element to keep corrections for using a lot of memory
        for _, tile in tiles.items():
            tile.compress_corrections()

            # Uncomment to track each level of correction
            # tile.increment_correction()


def _get_overlap_map(
    gen_settings: GeneralSettings,
) -> dict[int, list[int]]:
    """Return overlap map for each tile."""
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
