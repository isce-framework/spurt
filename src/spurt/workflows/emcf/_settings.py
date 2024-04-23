"""Class for handling EMCF settings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class SolverSettings:
    """Settings associated with Extended Minimum Cost Flow (EMCF) workflow.

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
    links_per_batch: int = 10000
    t_cost_type: str = "constant"
    t_cost_scale: float = 100.0
    s_cost_type: str = "constant"
    s_cost_scale: float = 100.0

    def __post_init__(self):
        assert self.t_cost_type in ["unit", "distance", "centroid"]
        assert self.s_cost_type in ["unit", "distance", "centroid"]
