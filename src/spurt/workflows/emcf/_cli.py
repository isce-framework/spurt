import argparse

import spurt

from ._bulk_offset import get_bulk_offsets
from ._merge import merge_tiles
from ._overlap import compute_overlap_stats
from ._settings import GeneralSettings, MergerSettings, SolverSettings, TilerSettings
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
        "-i", "--inputdir", help="Input folder with phase-linked SLC stack."
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
        "--workers",
        type=int,
        default=0,
        help="Number of workers. <=0 uses ncpus - 1.",
    )
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=50000,
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
        "--singletile", action="store_true", help="Process as a single tile."
    )

    # Parse arguments
    parsed_args = parser.parse_args(args=args)

    # Create the stack that is used
    stack = spurt.io.SLCStackReader.from_phase_linked_directory(
        parsed_args.inputdir,
        temp_coh_threshold=parsed_args.coh,
    )

    # Create general settings
    gen_settings = GeneralSettings(
        use_tiles=not parsed_args.singletile,
        intermediate_folder=parsed_args.tempdir,
        output_folder=parsed_args.outputdir,
    )

    # Create tile settings
    tile_settings = TilerSettings()

    # Create solver settings
    slv_settings = SolverSettings(
        worker_count=parsed_args.workers,
        links_per_batch=parsed_args.batchsize,
    )

    # Create merger settings
    mrg_settings = MergerSettings()

    # Using default Hop3Graph
    logger.info(f"Using Hop3 Graph in time with {len(stack.slc_files)} epochs.")
    g_time = spurt.graph.Hop3Graph(len(stack.slc_files))

    # Run the workflow
    # Generate tiles
    get_tiles(stack, gen_settings, tile_settings)

    # Unwrap tiles
    unwrap_tiles(stack, g_time, gen_settings, slv_settings)

    # Compute overlap stats
    compute_overlap_stats(gen_settings, mrg_settings)

    # Compute bulk offsets
    get_bulk_offsets(gen_settings, mrg_settings)

    # Merge tiles and write output
    merge_tiles(stack, g_time, gen_settings, mrg_settings)

    logger.info("Completed EMCF workflow.")
