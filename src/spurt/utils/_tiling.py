from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "BBox",
    "TileSet",
    "create_tiles_regular",
    "create_tiles_density",
]


@dataclass
class BBox:
    """Utility class for managing tile bounds.

    We follow shapely convention to store the bounds. The bounds should
    also be interpreted similar to python's indexing - i.e, left edge
    inclusive. See https://shapely.readthedocs.io/en/stable/manual.html#object.bounds
    for details on shapely convention.
    """

    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @classmethod
    def from_shapely_bounds(cls, bounds: Sequence[int]):
        """Create using any sequence in shapely convention."""
        if len(bounds) != 4:
            errmsg = f"bounds array must have size == 4, got {len(bounds)}"
            raise ValueError(errmsg)
        return cls(*bounds)

    def intersects(self, box: BBox) -> bool:
        """Check if two boxes intersect."""
        return not (
            (self.xmax < box.xmin)
            or (self.xmin > box.xmax)
            or (self.ymin > box.ymax)
            or (self.ymax < box.ymin)
        )

    def tolist(self) -> list[int]:
        """List of integers in shapely convention."""
        return [self.xmin, self.ymin, self.xmax, self.ymax]

    @property
    def space(self) -> tuple[slice, slice]:
        """Slice notation for use with NumPy arrays."""
        return (slice(self.xmin, self.xmax), slice(self.ymin, self.ymax))

    @property
    def count(self) -> int:
        """Number of pixels in tile."""
        return (self.xmax - self.xmin) * (self.ymax - self.ymin)


class TileSet:
    """Utility class for managing collection of tiles."""

    def __init__(
        self,
        shape: tuple[int, int],
        tiles: list[BBox],
    ):
        self._shape: tuple[int, int] = shape
        self._tiles: list[BBox] = tiles

    @property
    def shape(self) -> tuple[int, int]:
        return self._shape

    @property
    def tiles(self) -> list[BBox]:
        return self._tiles

    @property
    def ntiles(self) -> int:
        return len(self._tiles)

    @classmethod
    def from_json(cls, fname: Path) -> TileSet:
        """Load tiles from a json file."""
        with fname.open(mode="r") as fid:
            jdict = json.load(fid)

        shape = jdict["shape"]
        tiles: list[BBox] = []
        for tt in jdict["tiles"]:
            tiles.append(BBox.from_shapely_bounds(tt["bounds"]))

        return TileSet(shape, tiles)

    def to_json(self, fname: Path) -> None:
        """Write tiles to a json file."""
        tiles = []
        for tt in self.tiles:
            tiles.append({"bounds": tt.tolist()})

        jdict: dict = {
            "shape": self.shape,
            "tiles": tiles,
        }
        with fname.open(mode="w") as fid:
            fid.write(json.dumps(jdict, indent=4))

    @classmethod
    def single_tile(cls, shape: tuple[int, int]) -> TileSet:
        """Return tileset with single tile corresponding to shape."""
        return TileSet(shape, [BBox.from_shapely_bounds((0, 0, shape[0], shape[1]))])

    def get_overlaps(self) -> list[tuple[int, int]]:
        """Return list of pairs of overlapping tiles."""
        olaps: list[tuple[int, int]] = []

        ntiles = self.ntiles
        for ii in range(ntiles - 1):
            box1 = self._tiles[ii]
            for jj in range(ii + 1, ntiles):
                if box1.intersects(self._tiles[jj]):
                    olaps.append((ii, jj))

        return olaps

    def dilate(self, factor: float) -> TileSet:
        """Dilate current tile set."""
        tiles: list[BBox] = []
        shape = self.shape

        # Iterate over and dilate rectangles
        for tt in self.tiles:
            rdiff = tt.xmax - tt.xmin
            r0 = int(max(0, tt.xmin - factor * rdiff))
            r1 = int(min(shape[0], tt.xmax + factor * rdiff))
            cdiff = tt.ymax - tt.ymin
            c0 = int(max(0, tt.ymin - factor * cdiff))
            c1 = int(min(shape[1], tt.ymax + factor * cdiff))

            tiles.append(BBox.from_shapely_bounds([r0, c0, r1, c1]))

        return TileSet(shape, tiles)


def create_tiles_regular(shape: tuple[int, int], max_tiles: int) -> TileSet:
    """Tile set with approximately regularly sized non-overlapping tiles.

    Parameters
    ----------
    shape: tuple[int, int]
        Shape of rectangle to tile up.
    max_tiles: int
        Maximum number of tiles.
        Actual number of tiles is perfect square '<= max_tiles'.
    """
    if (shape[0] <= 0) and (shape[1] <= 0):
        errmsg = f"Invalid shape provided to tiler: {shape}"
        raise ValueError(errmsg)

    if max_tiles <= 0:
        errmsg = f"Maximum tiles should at least be 1. Got {max_tiles}"
        raise ValueError(errmsg)

    # Compute tiles per dimension
    tiles_per_dim: int = int(np.sqrt(max_tiles))

    # If only 1 tile was requested
    if tiles_per_dim == 1:
        return TileSet.single_tile(shape)

    # Generate tile extents
    rows = np.ogrid[0 : shape[0] : 1j * (tiles_per_dim + 1)]  # type: ignore[misc]
    cols = np.ogrid[0 : shape[1] : 1j * (tiles_per_dim + 1)]  # type: ignore[misc]
    tiles: list[BBox] = []

    # Counter for rows
    for ii in range(tiles_per_dim):
        r0 = int(rows[ii])
        r1 = int(rows[ii + 1])
        # Counter for columns
        for jj in range(tiles_per_dim):
            c0 = int(cols[jj])
            c1 = int(cols[jj + 1])
            tiles.append(BBox.from_shapely_bounds((r0, c0, r1, c1)))

    return TileSet(shape, tiles)


def create_tiles_density(
    points: np.ndarray, shape: tuple[int, int], max_tiles: int
) -> TileSet:
    """Tile set with non-overlapping tiles based on density of points.

    This is based on the nice implementation found here:
    https://mathoverflow.net/questions/412127/partitioning-unit-square-with-equal-frequency-rectangles

    Parameters
    ----------
    points: np.ndarray
        2D array with point coordinates within the shape.
    shape: tuple[int, int]
        Shape of rectangle to tile up.
    max_tiles: int
        Maximum number of tiles.
        Actual number of tiles is perfect square '<= max_tiles'.
    """
    if (points.ndim != 2) or (points.shape[1] != 2):
        errmsg = f"Point coordinates must be a Nx2 array. Got {points.shape}"
        raise ValueError(errmsg)

    if (shape[0] <= 0) and (shape[1] <= 0):
        errmsg = f"Invalid shape provided to tiler: {shape}"
        raise ValueError(errmsg)

    if max_tiles <= 0:
        errmsg = f"Maximum tiles should at least be 1. Got {max_tiles}"
        raise ValueError(errmsg)

    # Compute tiles per dim
    tiles_per_dim: int = int(np.sqrt(max_tiles))
    max_tiles = tiles_per_dim * tiles_per_dim

    # If only 1 tile was requested
    if max_tiles == 1:
        return TileSet.single_tile(shape)

    bounds = ((0, shape[0]), (0, shape[1]))
    splits = split_rectangle(points, bounds, max_tiles)
    tiles = []

    for s in splits:
        # Reorder indices
        t = [s[0][0], s[1][0], s[0][1], s[1][1]]
        r0 = int(max(0, t[0]))
        r1 = int(min(shape[0], t[2]))
        c0 = int(max(0, t[1]))
        c1 = int(min(shape[1], t[3]))

        tiles.append(BBox.from_shapely_bounds([r0, c0, r1, c1]))

    return TileSet(shape, tiles)


def _score_rectangle(aa: ArrayLike, bb: ArrayLike, m: int):
    """Compute the score for a rectangle."""
    asum = np.sum(aa)
    bsum = np.sum(bb)
    ii = np.maximum(1, np.minimum(m, (m * asum / bsum) ** 0.5))
    return ((asum * m / ii) ** 2 + (bsum * ii) ** 2) ** 0.5


def _get_splits(z: ArrayLike, m: int, boundz: tuple[int, int]):
    n = int(len(z) / m)
    beg_idx = np.arange(0, np.round(m - 1) * n + 1, n).astype(int)
    end_idx = beg_idx + n - 1
    beg_val = z[beg_idx]
    end_val = z[end_idx]

    # splits are defined to be exactly half-way in between : might not be optimal
    split_cuts = (end_val[:-1] + beg_val[1:]) / 2
    split_cuts = np.concatenate(([boundz[0]], split_cuts, [boundz[1]]))
    split_lens = np.diff(split_cuts)

    return split_cuts, split_lens


def _find_best_split(lens_c: ArrayLike, lens_o: ArrayLike, m: int):
    scores = np.array(
        [
            _score_rectangle(lens_c[:i], lens_o, i)
            + _score_rectangle(lens_c[i:], lens_o, m - i)
            for i in range(1, len(lens_c))
        ]
    )
    best_split = np.argmin(scores)
    return (best_split + 1, scores[best_split])


def split_rectangle(
    points: ArrayLike, bounds: tuple[tuple[int, int], tuple[int, int]], m: int
):
    if m == 1:
        return [bounds]

    x_s, y_s = list(zip(*points))
    x_s = np.array(x_s)
    y_s = np.array(y_s)

    ix_s = np.argsort(x_s)
    iy_s = np.argsort(y_s)

    cuts_x, lens_x = _get_splits(x_s[ix_s], m, bounds[0])
    cuts_y, lens_y = _get_splits(y_s[iy_s], m, bounds[1])

    x_bsplit_i, x_bsplit_s = _find_best_split(lens_x, lens_y, m)
    y_bsplit_i, y_bsplit_s = _find_best_split(lens_y, lens_x, m)

    if x_bsplit_s <= y_bsplit_s:
        best_cut = cuts_x[x_bsplit_i]
        mask = x_s <= best_cut

        newm = (x_bsplit_i, m - x_bsplit_i)

        newbounds = (
            ((bounds[0][0], best_cut), bounds[1]),
            ((best_cut, bounds[0][1]), bounds[1]),
        )
    else:
        best_cut = cuts_y[y_bsplit_i]
        mask = y_s <= best_cut

        newm = (y_bsplit_i, m - y_bsplit_i)

        newbounds = (
            (bounds[0], (bounds[1][0], best_cut)),
            (bounds[0], (best_cut, bounds[1][1])),
        )

    return split_rectangle(
        list(zip(x_s[mask], y_s[mask])), newbounds[0], newm[0]
    ) + split_rectangle(list(zip(x_s[~mask], y_s[~mask])), newbounds[1], newm[1])
