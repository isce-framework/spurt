import numpy as np

import spurt

from ._settings import GeneralSettings, TilerSettings

logger = spurt.utils.logger


def get_tiles(
    stack: spurt.io.SLCStackReader,
    gen_settings: GeneralSettings,
    tile_settings: TilerSettings,
) -> None:
    """Generate tiles based on settings.

    Create a json file with tile information in the intermediate folder.
    Tiles are not regenerated if a json file is already present.
    """
    json_file = gen_settings.tiles_jsonname

    # Generate tiles if file doesn't exist
    if json_file.is_file():
        logger.info(f"Using existing tiles file: {json_file!s}")
        return

    # Actual tile generation
    logger.info("Generating tiles for stack")
    arr = stack.read_temporal_coherence(np.s_[:, :]) > stack.temp_coh_threshold
    logger.info(f"Stack shape for generating tiles: {arr.shape}")

    # No tiles requested
    if not gen_settings.use_tiles:
        tileset = spurt.utils.TileSet.single_tile(arr.shape)

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
        ntiles = int(np.rint(np.sqrt(len(pts) / tile_settings.target_points_per_tile)))
        ntiles = min(max(1, ntiles * ntiles), tile_settings.max_tiles)
        logger.info(f"Generating {ntiles} tiles.")

        # Set up tiles
        tileset = spurt.utils.create_tiles_density(
            pts[::skip, :], shape=arr.shape, max_tiles=ntiles
        )

    # Dilate tiles to create overlaps
    if (tileset.ntiles > 1) and (tile_settings.dilation_factor > 0.0):
        tileset = tileset.dilate(tile_settings.dilation_factor)

    # Write tiles to json file
    logger.info(f"Writing tiles to: {json_file!s}")
    tileset.to_json(json_file)
