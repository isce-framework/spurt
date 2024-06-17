from pathlib import Path

import h5py
import numpy as np

import spurt

from ._settings import GeneralSettings, SolverSettings
from ._solver import EMCFSolver

logger = spurt.utils.logger


def unwrap_tiles(
    stack: spurt.io.SLCStackReader,
    g_time: spurt.graph.PlanarGraphInterface,
    gen_settings: GeneralSettings,
    solv_settings: SolverSettings,
) -> None:
    """Unwrap each tile and save to h5."""
    # Temporal graph
    s_time = spurt.mcf.ORMCFSolver(g_time)  # type: ignore[abstract]

    # Load tile set
    tiledata = spurt.utils.TileSet.from_json(gen_settings.tiles_jsonname)

    # Iterate over tiles
    for tt, tile in enumerate(tiledata.tiles):
        tfname = str(gen_settings.tile_filename(tt))
        if Path(tfname).is_file():
            logger.info(f"Tile {tt+1} already processed. Skipping...")
            continue

        # Select valid pixels from coherence file
        logger.info(f"Processing tile: {tt+1}")
        coh = stack.read_temporal_coherence(tile.space)

        # Create spatial graph and solver
        g_space = spurt.graph.DelaunayGraph(
            np.column_stack(np.nonzero(coh > stack.temp_coh_threshold))
        )
        s_space = spurt.mcf.ORMCFSolver(g_space)  # type: ignore[abstract]

        # EMCF solver
        solver = EMCFSolver(s_space, s_time, solv_settings)
        wrap_data = stack.read_tile(tile.space)
        assert wrap_data.shape[1] == g_space.npoints
        logger.info(f"Time steps: {solver.nifgs}")
        logger.info(f"Number of points: {solver.npoints}")

        uw_data = solver.unwrap_cube(wrap_data)
        logger.info(f"Completed tile: {tt+1}")

        # Unwrapped data above is always referenced to first pixel
        # since we unwrap gradients. Phase offsets for the first
        # pixel are computed and provided separately. When mosaicking,
        # these offsets need to be added to unwrapped tiles to guarantee
        # integer cycle shifts between tiles.
        ifgs = g_time.links
        phase_offset = spurt.mcf.utils.phase_diff(
            wrap_data.data[ifgs[:, 0], 0], wrap_data.data[ifgs[:, 1], 0]
        )

        _dump_tile_to_h5(tfname, uw_data, phase_offset, g_space, tile)
        logger.info(f"Wrote tile {tt + 1} to {tfname}")
        wrap_data = None
        uw_data = None


def _dump_tile_to_h5(
    fname: str,
    uw: np.ndarray,
    off: np.ndarray,
    gspace: spurt.graph.PlanarGraphInterface,
    tile: spurt.utils.BBox,
) -> None:
    with h5py.File(fname, "w") as fid:
        fid["uw_data"] = uw
        fid["points"] = gspace.points.astype(np.int32)
        fid["tile"] = np.array(tile.tolist()).astype(np.int32)
        fid["phase_offset"] = off.astype(np.float32)
