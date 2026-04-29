import argparse
import logging

import spurt

from ._bulk_offset import get_bulk_offsets
from ._merge import merge_tiles
from ._output import write_link_params
from ._overlap import compute_phasediff_deciles
from ._settings import (
    GeneralSettings,
    LinkModelSettings,
    MergerSettings,
    SolverSettings,
    TilerSettings,
)
from ._tiling import get_tiles
from ._unwrap import unwrap_tiles

logger = spurt.utils.logger


def main(args=None):
    """Top-level entry pint for EMCF workflow."""
    parser = argparse.ArgumentParser(
        prog=__package__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=spurt.__version__)
    parser.add_argument(
        "-i",
        "--inputdir",
        help="Input folder with phase-linked SLC stack.",
        required=True,
    )
    parser.add_argument(
        "-o",
        "--outputdir",
        default="./emcf",
        help="Output folder for final unwrapped raster files.",
    )
    parser.add_argument(
        "--tempdir", default="./emcf_tmp", help="Folder for intermediate outputs."
    )
    parser.add_argument(
        "-w",
        "--t-workers",
        type=int,
        default=0,
        help="Number of workers for temporal unwrapping. <=0 uses ncpus - 1.",
    )
    parser.add_argument(
        "--s-workers",
        type=int,
        default=1,
        help="Number of workers for spatial unwrapping. <=0 uses ncpus - 1.",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=150000,
        help="Links per batch for temporal unwrapping.",
    )
    parser.add_argument(
        "-c",
        "--coh",
        type=float,
        default=0.6,
        help="Temporal coherence threshold for good pixels.",
    )
    parser.add_argument(
        "--t-cost-type",
        choices=["constant", "distance", "centroid"],
        default="constant",
        help="Temporal unwrapping costs.",
    )
    parser.add_argument(
        "--t-cost-scale",
        type=int,
        default=100,
        help="Scale factor used in computing edge costs for temporal unwrapping.",
    )
    parser.add_argument(
        "--pts-per-tile",
        type=int,
        default=800000,
        help="Target points per tile.",
    )
    parser.add_argument(
        "--max-tiles",
        type=int,
        default=49,
        help="Maximum number of tiles allowed during tile formation.",
    )
    parser.add_argument(
        "--merge-parallel-ifgs",
        type=int,
        default=1,
        help="Number of ifgs to merge in parallel.",
    )
    parser.add_argument(
        "--unwrap-parallel-tiles",
        type=int,
        default=1,
        help="Number of tiles to unwrap in parallel.",
    )
    parser.add_argument(
        "--singletile", action="store_true", help="Process as a single tile."
    )
    parser.add_argument(
        "--log-file",
        help="Path to save the log file (in addition to printing to stderr).",
    )
    parser.add_argument(
        "--date-fmt",
        default="%Y%m%d",
        help=(
            "strftime format used to extract acquisition dates from SLC"
            " filenames and to write the date portion of unwrapped output"
            " filenames. Use a longer format such as '%%Y%%m%%d%%H%%M%%S' to"
            " preserve a time-of-day component (e.g. for non-Sentinel cadences"
            " with same-day repeats)."
        ),
    )

    # Link model / velocity estimation arguments
    parser.add_argument(
        "--baseline-csv",
        type=str,
        default=None,
        help="Path to CSV with perpendicular baselines. Enables velocity estimation.",
    )
    parser.add_argument(
        "--no-velocity-estimation",
        action="store_true",
        help="Disable velocity/DEM error estimation even if baseline CSV is provided.",
    )
    parser.add_argument(
        "--wavelength",
        type=float,
        default=0.055465,
        help="Radar wavelength in meters.",
    )
    parser.add_argument(
        "--slant-range",
        type=float,
        default=900000.0,
        help="Slant range distance in meters.",
    )
    parser.add_argument(
        "--look-angle-deg",
        type=float,
        default=39.0,
        help="Look angle in degrees.",
    )
    parser.add_argument(
        "--velocity-range",
        type=float,
        nargs=3,
        default=[-100.0, 100.0, 5.0],
        metavar=("MIN", "MAX", "STEP"),
        help="Velocity search range in mm/yr: min max step.",
    )
    parser.add_argument(
        "--dem-error-range",
        type=float,
        nargs=3,
        default=[-50.0, 50.0, 2.5],
        metavar=("MIN", "MAX", "STEP"),
        help="DEM error search range in meters: min max step.",
    )

    # Parse arguments
    parsed_args = parser.parse_args(args=args)
    if parsed_args.log_file:
        file_handler = logging.FileHandler(parsed_args.log_file)
        _fmt = "%(asctime)s [%(process)d] [%(levelname)s] %(name)-3s: %(message)s"
        file_handler.setFormatter(logging.Formatter(_fmt))
        logger.addHandler(file_handler)

    # Create the stack that is used
    stack = spurt.io.SLCStackReader.from_phase_linked_directory(
        parsed_args.inputdir,
        temp_coh_threshold=parsed_args.coh,
        date_fmt=parsed_args.date_fmt,
    )

    # Create general settings
    gen_settings = GeneralSettings(
        use_tiles=not parsed_args.singletile,
        intermediate_folder=parsed_args.tempdir,
        output_folder=parsed_args.outputdir,
    )

    # Create tile settings
    tile_settings = TilerSettings(
        target_points_per_tile=parsed_args.pts_per_tile,
        max_tiles=parsed_args.max_tiles,
    )

    # Create solver settings
    slv_settings = SolverSettings(
        t_worker_count=parsed_args.t_workers,
        s_worker_count=parsed_args.s_workers,
        links_per_batch=parsed_args.batchsize,
        num_parallel_tiles=parsed_args.unwrap_parallel_tiles,
        t_cost_scale=parsed_args.t_cost_scale,
        t_cost_type=parsed_args.t_cost_type,
    )

    # Create merger settings
    mrg_settings = MergerSettings(
        num_parallel_ifgs=parsed_args.merge_parallel_ifgs,
    )

    # Create link model settings if baseline CSV is provided and not disabled
    link_model_settings: LinkModelSettings | None = None
    if parsed_args.baseline_csv and not parsed_args.no_velocity_estimation:
        link_model_settings = LinkModelSettings(
            enabled=True,
            wavelength_m=parsed_args.wavelength,
            slant_range_m=parsed_args.slant_range,
            look_angle_deg=parsed_args.look_angle_deg,
            velocity_range=tuple(parsed_args.velocity_range),
            dem_error_range=tuple(parsed_args.dem_error_range),
            baseline_csv=parsed_args.baseline_csv,
        )
        logger.info(f"Link model enabled with baselines: {parsed_args.baseline_csv}")

    # Using default Hop3Graph
    logger.info(f"Using Hop3 Graph in time with {len(stack.slc_files)} epochs.")
    g_time = spurt.graph.Hop3Graph(len(stack.slc_files))

    # Run the workflow
    # Generate tiles
    get_tiles(stack, gen_settings, tile_settings)

    # Unwrap tiles
    unwrap_tiles(stack, g_time, gen_settings, slv_settings, link_model_settings)

    # Compute overlap stats
    compute_phasediff_deciles(gen_settings, mrg_settings)

    # Compute bulk offsets
    get_bulk_offsets(stack, gen_settings, mrg_settings)

    # Merge tiles and write output
    merge_tiles(stack, g_time, gen_settings, mrg_settings)

    # Write link model parameters (velocity, DEM error) if available
    if link_model_settings is not None:
        like_slc_file = stack.slc_files[stack.dates[-1]]
        write_link_params(gen_settings, stack.raster_shape, like=like_slc_file)

    logger.info("Completed EMCF workflow.")
