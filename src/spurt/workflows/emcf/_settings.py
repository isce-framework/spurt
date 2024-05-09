"""Class for handling EMCF settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SolverSettings:
    """Settings associated with EMCF workflow.

    Parameters
    ----------
    worker_count: int
        Number of workers for temporal unwrapping in parallel. Set value to <=0
        to let workflow use default workers (ncpus - 1).
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
    """

    worker_count: int = 0
    points_per_batch: int = 10000
    t_cost_type: str = "unit"
    t_cost_scale: float = 100.0
    s_cost_type: str = "constant"
    s_cost_scale: float = 100.0

    def __post_init__(self):
        assert self.t_cost_type in ["unit", "distance", "centroid"]
        assert self.s_cost_type in ["unit", "distance", "centroid"]
        assert self.points_per_batch > 0
        assert self.t_cost_scale > 0
        assert self.s_cost_scale > 0


@dataclass
class GeneralSettings:
    """Settings associated with breaking data into tiles.

    Parameters
    ----------
    use_tiles: bool
        Tile up data spatially.
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
        return Path(self.output_folder) / f"{d1}_{d2}.unw"

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
    """

    max_tiles: int = 16
    target_points_for_generation: int = 120000
    target_points_per_tile: int = 800000


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
        Method used to estimate bulk offset between tiles.
    """

    min_overlap_points: int = 25
    method: str = "dirichlet"
    bulk_method: str = "L2"

    def __post_init__(self):
        assert self.bulk_method in ["integer", "L2"]
        assert self.method == "dirichlet"
