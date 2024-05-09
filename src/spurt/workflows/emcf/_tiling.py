import json
from pathlib import Path

import numpy as np

import spurt

from ._settings import GeneralSettings, TilerSettings

logger = spurt.utils.logger


def get_tiles(
    stack: spurt.io.SLCStackReader,
    gen_settings: GeneralSettings,
    tile_settings: TilerSettings,
) -> None:
    """Get the tiles for given stack."""
    json_file = gen_settings.tiles_jsonname

    # Generate tiles if file doesn't exist
    if not json_file.is_file():
        logger.info("Generating tiles for stack")
        arr = stack.read_temporal_coherence(np.s_[:, :]) > stack.temp_coh_threshold
        logger.info(f"Stack shape for generating tiles: {arr.shape}")

        # No tiles requested
        if not gen_settings.use_tiles:
            _write_single_tile_json(arr, json_file)

        # Generate tiles and write
        else:
            # Get points
            pts = np.column_stack(np.nonzero(arr))
            logger.info(f"Number of points: {pts.shape[0]}")
            logger.info(f"Fraction good:  {pts.shape[0] / arr.size:.3f}")

            # Skip points as needed for tile generation
            skip = max(1, int(len(pts) / tile_settings.target_points_for_generation))
            logger.info(f"Skipping {skip} pixels for tile generation.")

            # Determine max tiles
            ntiles = int(
                np.rint(np.sqrt(len(pts) / tile_settings.target_points_per_tile))
            )
            ntiles = min(max(1, ntiles * ntiles), tile_settings.max_tiles)
            logger.info(f"Generating {ntiles} tiles.")

            if ntiles > 1:
                # Set up tiles
                tiler = spurt.utils.DensityTiler(
                    pts[::skip, :], shape=arr.shape, max_tiles=ntiles
                )

                # Write tiles to json file
                logger.info(f"Writing tiles to: {json_file!s}")
                _write_tile_json(tiler, arr, json_file)
            else:
                _write_single_tile_json(arr, json_file)

    else:
        logger.info(f"Using existing tiles file: {json_file!s}")


def _write_tile_json(
    tiler: spurt.utils.DensityTiler,
    arr: np.ndarray,
    fname: Path,
) -> None:
    """Write a json file with tile extents."""
    jdata = {
        "shape": arr.shape,
        "neighbors": tiler.neighbors.tolist() if tiler.neighbors else [],
        "tiles": [],
    }
    for tile in tiler.tiles:
        tdata = {
            "bounds": tile.tolist(),
            "count": int(np.sum(arr[tile[0] : tile[2], tile[1] : tile[3]])),
        }
        jdata["tiles"].append(tdata)

    with fname.open(mode="x") as fid:
        fid.write(json.dumps(jdata, indent=4))

    return


def _write_single_tile_json(
    arr: np.ndarray,
    fname: Path,
) -> None:
    """Write a json file for single tile."""
    jdata = {"shape": arr.shape, "neighbors": [], "tiles": []}
    tdata = {
        "bounds": [0, 0, arr.shape[0], arr.shape[1]],
        "count": int(np.sum(arr)),
    }
    jdata["tiles"].append(tdata)

    logger.info(f"Writing single tile json to {fname!s}")
    with fname.open(mode="x") as fid:
        fid.write(json.dumps(jdata, indent=4))

    return
