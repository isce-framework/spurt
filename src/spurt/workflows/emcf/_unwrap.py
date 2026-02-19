import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import h5py
import numpy as np

import spurt

from ._settings import GeneralSettings, LinkModelSettings, SolverSettings
from ._solver import EMCFSolver

logger = spurt.utils.logger

__all__ = ["unwrap_tiles"]


def unwrap_tiles(
    stack: spurt.io.SLCStackReader,
    g_time: spurt.graph.PlanarGraphInterface,
    gen_settings: GeneralSettings,
    solv_settings: SolverSettings,
    link_model_settings: LinkModelSettings | None = None,
) -> None:
    """Unwrap each tile and save to h5."""
    # Load tile set
    tile_json = gen_settings.tiles_jsonname
    tiledata = spurt.utils.TileSet.from_json(tile_json)

    mp_context = mp.get_context("fork")
    with ProcessPoolExecutor(
        max_workers=solv_settings.num_parallel_tiles, mp_context=mp_context
    ) as executor:
        futures = {}

        # Iterate over tiles
        for tt in range(tiledata.ntiles):
            tfname = str(gen_settings.tile_filename(tt))
            if Path(tfname).is_file():
                logger.info(f"Tile {tt+1} already processed. Skipping...")
                continue

            futures[
                executor.submit(
                    _unwrap_one_tile,
                    stack,
                    tile_json,
                    tfname,
                    g_time,
                    solv_settings,
                    tt,
                    link_model_settings,
                )
            ] = tt

        for fut in as_completed(futures):
            fut.result()
            futures.pop(fut)


def _unwrap_one_tile(
    stack: spurt.io.SLCStackReader,
    tile_json: Path,
    tile_output: str,
    g_time: spurt.graph.PlanarGraphInterface,
    solv_settings: SolverSettings,
    tile_num: int,
    link_model_settings: LinkModelSettings | None = None,
) -> None:
    """Unwrap tile-by-tile."""
    # Get tile information
    tiledata = spurt.utils.TileSet.from_json(tile_json)
    tile = tiledata.tiles[tile_num]
    tt = tile_num

    # Temporal solver
    s_time = spurt.mcf.ORMCFSolver(g_time)  # type: ignore[abstract]

    # Select valid pixels from coherence file
    logger.info(f"Processing tile: {tt+1}")
    coh = stack.read_temporal_coherence(tile.space)

    # Create spatial graph and solver
    g_space = spurt.graph.DelaunayGraph(
        np.column_stack(np.nonzero(coh > stack.temp_coh_threshold))
    )
    s_space = spurt.mcf.ORMCFSolver(g_space)  # type: ignore[abstract]

    # Build link model if settings provided
    link_model = None
    if link_model_settings is not None and link_model_settings.enabled:
        link_model = _build_link_model(g_time, stack.dates, link_model_settings)

    # EMCF solver
    solver = EMCFSolver(s_space, s_time, solv_settings, link_model)
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

    _dump_tile_to_h5(
        tile_output,
        uw_data,
        phase_offset,
        g_space,
        tile,
        solver.link_params,
        solver.link_coherence,
    )
    logger.info(f"Wrote tile {tt + 1} to {tile_output}")


def _build_link_model(
    g_time: spurt.graph.PlanarGraphInterface,
    dates: list[str],
    settings: LinkModelSettings,
) -> spurt.links.GridSearchLinearModel:
    """Build link model from settings and baseline data."""
    from spurt.io import load_baseline_csv
    from spurt.links import GridSearchLinearModel, build_design_matrix

    assert settings.baseline_csv is not None
    baseline_data = load_baseline_csv(settings.baseline_csv)

    # Convert stack dates to datetime64
    stack_dates = np.array(dates, dtype="datetime64[D]")

    # Build design matrix
    amat = build_design_matrix(
        ifg_edges=g_time.links,
        dates=stack_dates,
        bperp_m=_interpolate_baselines(stack_dates, baseline_data),
        wavelength_m=settings.wavelength_m,
        slant_range_m=settings.slant_range_m,
        incidence_rad=settings.incidence_rad,
    )

    # Create grid search model
    return GridSearchLinearModel(
        matrix=amat,
        ranges=(settings.velocity_slice, settings.dem_error_slice),
    )


def _interpolate_baselines(
    stack_dates: np.ndarray,
    baseline_data: spurt.io.BaselineData,
) -> np.ndarray:
    """Interpolate baselines to match stack dates.

    If stack dates match baseline dates exactly, returns baselines directly.
    Otherwise, linearly interpolates baselines for missing dates.

    Parameters
    ----------
    stack_dates : np.ndarray
        SLC dates from the stack as datetime64[D].
    baseline_data : spurt.io.BaselineData
        Baseline data loaded from CSV.

    Returns
    -------
    np.ndarray
        Perpendicular baselines matched to stack dates.
    """
    # Check for exact match
    if len(stack_dates) == len(baseline_data.dates) and np.all(
        stack_dates == baseline_data.dates
    ):
        return baseline_data.bperp_m

    # Perpendicular baselines depend on orbital geometry, not time, so
    # linear interpolation is only a rough approximation.
    n_missing = np.sum(~np.isin(stack_dates, baseline_data.dates))
    logger.warning(
        f"Baseline dates do not match stack dates ({n_missing} dates missing"
        f" from baseline CSV). Linearly interpolating baselines; this is only"
        f" approximate since Bperp depends on orbital geometry, not time."
    )

    stack_days = stack_dates.astype("datetime64[D]").astype(np.float64)
    baseline_days = baseline_data.dates.astype("datetime64[D]").astype(np.float64)
    return np.interp(stack_days, baseline_days, baseline_data.bperp_m)


def _dump_tile_to_h5(
    fname: str,
    uw: np.ndarray,
    off: np.ndarray,
    gspace: spurt.graph.PlanarGraphInterface,
    tile: spurt.utils.BBox,
    link_params: np.ndarray | None = None,
    link_coherence: np.ndarray | None = None,
) -> None:
    with h5py.File(fname, "w") as fid:
        fid["uw_data"] = uw
        fid["points"] = gspace.points.astype(np.int32)
        fid["tile"] = np.array(tile.tolist()).astype(np.int32)
        fid["phase_offset"] = off.astype(np.float32)

        if link_params is not None:
            fid["link_params"] = link_params.astype(np.float32)
        if link_coherence is not None:
            fid["link_coherence"] = link_coherence.astype(np.float32)
