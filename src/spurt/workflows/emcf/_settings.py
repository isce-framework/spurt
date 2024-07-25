"""Class for handling EMCF settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SolverSettings:
    """Settings associated with Extended Minimum Cost Flow (EMCF) workflow.

    Parameters
    ----------
    t_worker_count: int
        Number of workers for temporal unwrapping in parallel. Set value to <=0
        to let workflow use default workers (ncpus - 1).
    s_worker_count: int
        Number of workers for spatial unwrapping in parallel. Set value to <=0
        to let workflow use (ncpus - 1).
        Defaults to 1 (i.e. unwrap one interferogram in space at a time).
    links_per_batch: int
        Temporal unwrapping operations over spatial links are performed in batches
        and each batch is solved in parallel.
    t_cost_type: str
        Temporal unwrapping costs. Can be one of 'constant', 'distance',
        'centroid'.
    t_cost_scale: float
        Scale factor used in computing edge costs for temporal unwrapping.
    s_cost_type: str
        Spatial unwrapping costs. Can be one of 'constant', 'distance',
        'centroid'.
    s_cost_scale: float
        Scale factor used in computing edge costs for spatial unwrapping.
    num_parallel_tiles: int
        Number of tiles to process in parallel. Set to 0 for all tiles.
    """

    t_worker_count: int = 0
    s_worker_count: int = 1
    links_per_batch: int = 10000
    t_cost_type: str = "constant"
    t_cost_scale: float = 100.0
    s_cost_type: str = "constant"
    s_cost_scale: float = 100.0
    num_parallel_tiles: int = 1

    def __post_init__(self):
        cost_types = {"constant", "distance", "centroid"}
        if self.t_cost_type not in cost_types:
            errmsg = f"t_cost_type must be one of {cost_types}, got {self.t_cost_type}"
            raise ValueError(errmsg)
        if self.s_cost_type not in cost_types:
            errmsg = f"s_cost_type must be one of {cost_types}, got {self.s_cost_type}"
            raise ValueError(errmsg)
        if self.links_per_batch <= 0:
            errmsg = f"links_per_batch must be > 0, got {self.links_per_batch}"
            raise ValueError(errmsg)
        if self.t_cost_scale <= 0.0:
            errmsg = f"t_cost_scale must be > 0, got {self.t_cost_scale}"
            raise ValueError(errmsg)
        if self.s_cost_scale <= 0.0:
            errmsg = f"s_cost_scale must be > 0, got {self.s_cost_scale}"
            raise ValueError(errmsg)


@dataclass
class GeneralSettings:
    """Settings associated with breaking data into tiles.

    Parameters
    ----------
    use_tiles: bool
        Tile up data spatially.
    intermediate_folder: str
        Path to folder where intermediate outputs are created.
    output_folder: str
        Path to output folder.
    """

    use_tiles: bool = True
    intermediate_folder: str = "./emcf_tmp"
    output_folder: str = "./emcf"

    @property
    def tiles_jsonname(self) -> Path:
        return Path(self.intermediate_folder) / "tiles.json"

    def tile_filename(self, num: int) -> Path:
        """Input index is zero-based."""
        return Path(self.intermediate_folder) / f"uw_tile_{num:02d}.h5"

    @property
    def overlap_filename(self) -> Path:
        return Path(self.intermediate_folder) / "overlaps.h5"

    def overlap_groupname(self, ii: int, jj: int) -> str:
        return f"{ii:02d}_{jj:02d}"

    @property
    def offsets_filename(self) -> Path:
        return Path(self.intermediate_folder) / "bulk_offsets.h5"

    def unw_filename(self, d1: str, d2: str) -> Path:
        return Path(self.output_folder) / f"{d1}_{d2}.unw.tif"

    def __post_init__(self):
        p = Path(self.output_folder)
        if not p.is_dir():
            p.mkdir(exist_ok=True)

        p = Path(self.intermediate_folder)
        if not p.is_dir():
            p.mkdir(exist_ok=True)


@dataclass
class TilerSettings:
    """Class for holding tile generation settings.

    Parameters
    ----------
    target_points_per_tile: int
        Target points per tile when generating tiles.
    max_tiles: int
        Maximum number of tiles allowed.
    target_points_for_generation: int
        Number of points used for determining tiles based on density.
        If input has a lot of points, tiling can take a really long time.
        We use this as a guide to downsample inputs to generate tile
        boundaries. The tile boundaries are then used with full set of inputs.
    dilation_factor: float
        Dilation factor of non-overlapping tiles. 0.05 would lead to
        10 percent dilation of the tile.
    """

    target_points_per_tile: int = 800000
    max_tiles: int = 16
    target_points_for_generation: int = 120000
    dilation_factor: float = 0.05

    def __post_init__(self):
        if self.max_tiles < 1:
            errmsg = f"max_tiles must be at least 1, got {self.max_tiles}"
            raise ValueError(errmsg)
        if self.dilation_factor < 0.0:
            errmsg = f"dilation_factor must be >= 0., got {self.dilation_factor}"
            raise ValueError(errmsg)
        if self.target_points_for_generation <= 0:
            errmsg = (
                "target_points_for_generation must be > 0,"
                f" got {self.target_points_for_generation}"
            )
            raise ValueError(errmsg)
        if self.target_points_per_tile <= 0.0:
            errmsg = (
                f"target_points_per_tile must be > 0, got {self.target_points_per_tile}"
            )
            raise ValueError(errmsg)


@dataclass
class MergerSettings:
    """Class for holding tile merging settings.

    Parameters
    ----------
    min_overlap_points: int
        Minimum number of pixels in overlap region for it to be considered
        valid.
    method: str
        Currently, only "dirichlet" is supported.
    bulk_method: str
        Method used to estimate bulk offset between tiles. Supported methods
        are 'integer' and 'L2'.
    num_parallel_ifgs: int
        Number of interferograms to merge in one batch. Use zero to merge all
        interferograms in a single batch.
    """

    min_overlap_points: int = 25
    method: str = "dirichlet"
    bulk_method: str = "L2"
    num_parallel_ifgs: int = 1

    def __post_init__(self):
        bulk_methods = {"integer", "L2"}
        if self.bulk_method not in bulk_methods:
            errmsg = (
                f"bulk_method must be one of {bulk_methods}. got {self.bulk_method}"
            )
            raise ValueError(errmsg)

        if self.method != "dirichlet":
            errmsg = f"'dirichlet' is the only valid option, got {self.method}"
            raise ValueError(errmsg)
