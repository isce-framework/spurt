# Python Coding & Editing Guidelines

> **Living document – PRs welcome!**
> Last updated: 2026‑01‑15

## Table of Contents

1. Philosophy
1. Code Style
1. Docstrings & Comments
1. Tools
1. Documentation

---

## Philosophy

- **Readability, reproducibility, performance – in that order.**
- Prefer explicit over implicit; avoid hidden state and global flags.
- Measure before you optimize (`time.perf_counter`, `line_profiler`).
- Each module holds a **single responsibility**; keep public APIs minimal.
- Before merging pull requests, ensure that
  - All tests pass
  - All precommit checks pass
  - Any new functionality has a new test with it
  - Any bug fixes have a new test which fails on the main branch but passes on your PR

## Code Style

- Annotate all public functions (PEP 484).
- In general, write code that will raise an exception early if something isn't expected.
- Raise Exceptions/Errors for user-facing problems. Only use asserts to help fix mypy errors, or to show developer expectations.
- Use Pydantic models over `dataclasses.dataclass`.
- Aim for zero "dead" code: do not leave commented code in unless it is part of a very descriptive comment that illustrates something specific.
- Follow the "parse, don't validate" addage: Parse unknown inputs at the serialization boundaries, not scattered everywhere in the code.
- If you need to add an ignore, ignore a specific check like # type: ignore[specific] . Use sparingly.
- Don't write error handing code or smooth over exceptions/errors unless they are expected as part of control flow.
- Prefer `Protocol` over `ABC`s when only an interface is needed.
- Use `from loguru import logger` for logging instead of `print` statements (`logger.info`).

## Docstrings & Comments

- Style: NumPyDoc.
- Start with a one‑sentence summary in the imperative mood.
- Sections: Parameters, Returns, Raises, Examples, References.
- Use backticks for code or referring to variables (e.g. `xarray.DataArray`).
- Do not use emojis, or non-unicode characters in comments/print statements.
- Cite peer‑reviewed papers with DOI links when relevant.
- Write code that explains itself rather than needs comments.
- For the inline you do add, explain *why*, not what. For example, *don't* write:

```python
# open the file
f = open(filename)
```

- The comments should be things which are not obvious to a reader with typical background knowledge. Aim to write code that explains itself.


## Tools

- You can run `pre-commit run -a` to run all pre-commit hooks and check for style violations
- ruff is uses for most code maintenance, black for formatting, mypy for type checking, pytest for testing



## Documentation

- mkdocs + Jupyter. Hosted on ReadTheDocs.
- Auto API from type hints.
- Provide tutorial notebooks covering common workflows.
- Include examples in docstrings.
- Add high-level guides for key functionality.

---

## Codebase Architecture

SPURT implements Extended Minimum Cost Flow (EMCF) for 3D InSAR phase unwrapping. The algorithm decomposes 3D unwrapping into two sequential 2D MCF problems.

### Module Structure (`src/spurt/`)

| Module | Purpose |
|--------|---------|
| `graph/` | Graph representations (Delaunay, Hop3, Regular2D) for spatial/temporal domains |
| `mcf/` | Minimum Cost Flow solver (OR-Tools based) and utilities |
| `links/` | Per-link model estimation (DEM errors, velocities) via grid search |
| `workflows/emcf/` | Main EMCF algorithm orchestration, tiling, merging |
| `io/` | Input/output interfaces for SLC stacks and 3D data |
| `utils/` | Logging, CPU utilities, tiling helpers |

### Algorithm Flow

**Stage 1: Temporal Unwrapping** (`EMCFSolver.unwrap_gradients_in_time`)
- For each spatial edge (pixel-to-pixel link), unwrap phase gradients across interferograms
- Uses temporal graph `G_t` (typically Hop3 or Delaunay in time-baseline space)
- Outputs temporally-unwrapped spatial gradients

**Stage 2: Spatial Unwrapping** (`EMCFSolver.unwrap_gradients_in_space`)
- For each interferogram, unwrap spatial gradients using MCF
- Uses spatial graph `G_s` (typically Delaunay triangulation)
- Outputs unwrapped interferograms

### Key Files

- `mcf/_ortools.py` - Core MCF solver using OR-Tools (`ORMCFSolver`)
- `mcf/utils.py` - `phase_diff()`, `flood_fill()`, cost functions
- `links/_grid_search.py` - `GridSearchLinearModel` for velocity/DEM error estimation
- `links/_common.py` - Temporal coherence objective function
- `workflows/emcf/_solver.py` - `EMCFSolver` main algorithm class
- `graph/_delaunay.py` - Delaunay triangulation and regular 2D grid graphs
- `graph/_hop3.py` - Hop-3 temporal graph for narrow-baseline time-series

### Data Structures

**Edges/Links**: `(nedges, 2)` integer arrays indexing into points, always ordered `links[i, 0] < links[i, 1]`

**Dual Graph**: Maps primal edges to cycles they participate in:
- `dual_edges[i]` = `[cycle1_idx, cycle2_idx]` (1-indexed, 0 = boundary)
- `dual_edge_dir[i]` = `[+1/-1, +1/-1]` orientation within each cycle

**Design Matrix for Link Models**: `amat` of shape `(nifgs, ndim)` where:
- Column 0: temporal sensitivity (rad per velocity unit, e.g., mm/yr)
- Column 1: baseline sensitivity (rad per DEM error unit, e.g., meters)

### DEM Error / Velocity Estimation

The `links/` module provides per-link model estimation via `GridSearchLinearModel`:
1. Build design matrix `amat` from temporal baselines and perpendicular baselines
2. Define search ranges as `slice(start, stop, step)` for each parameter
3. For each edge, grid search + Nelder-Mead finds parameters maximizing temporal coherence
4. Model prediction: `model_phase = amat @ [velocity, dem_error]`
5. Use `phase_diff(data0, data1, model=model_phase)` to "flatten" edges before MCF

Integration point: `_solver.py` line 181-182 (TODO comment for incorporating link_model)
