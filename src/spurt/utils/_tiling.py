from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike


def intersects(box1: ArrayLike, box2: ArrayLike) -> bool:
    """Check if two boxes intersect."""
    return not (
        box1[2] < box2[0] or box1[0] > box2[2] or box1[1] > box2[3] or box1[3] < box2[1]
    )


@runtime_checkable
class TilerInterface(Protocol):
    """Rectangular tiler interface."""

    @property
    def ntiles(self) -> int:
        """Number of tiles."""

    @property
    def bounds(self) -> tuple[int, int]:
        """Shape of rectangle being tiled."""

    @property
    def tiles(self) -> ArrayLike:
        """2D array of shape (ntiles, 4) using shapely bbox convention."""

    @property
    def neighbors(self) -> ArrayLike | None:
        """2D array of shape (nedges, 2).

        This represents sorted pair of indices of neighboring tiles similar to
        edges in a graph.
        """


class RegularTiler(TilerInterface):
    """Regularly sized tiles."""

    def __init__(self, shape: tuple[int, int], max_tiles: int, dilation: float = 0.05):
        """Regularly sized tileset.

        Parameters
        ----------
        shape: tuple[int, int]
            Shape of rectangle to tile up.
        max_tiles: int
            Maximum number of tiles. Use of perfect squares recommended.
        dilation: float
            Fraction indicating the factor by which the tiles will be dilated
            for overlaps. 0.05 dilation will result in 10 percent overlap.
        """
        if (shape[0] <= 0) and (shape[1] <= 0):
            errmsg = f"Invalid shape provided to tiler: {shape}"
            raise ValueError(errmsg)

        if max_tiles <= 0:
            errmsg = f"Maximum tiles should atleast be 1. Got {max_tiles}"
            raise ValueError(errmsg)

        self._shape: tuple[int, int] = shape
        self._max_tiles: int = max_tiles
        self._dilation: float = dilation
        self._tiles: np.ndarray | None = None
        self._neighbors: np.ndarray | None = None
        self._generate_tiles()

    @property
    def ntiles(self) -> int:
        return self.tiles.shape[0]

    @property
    def bounds(self) -> tuple[int, int]:
        return self._shape

    @property
    def tiles(self) -> ArrayLike:
        return self._tiles

    @property
    def neighbors(self) -> ArrayLike | None:
        return self._neighbors

    def _generate_tiles(self) -> None:
        """Generate the tiles and populate arrays."""
        # If only 1 tile was requested
        if self._max_tiles == 1:
            self._tiles = np.array([[0, 0, self._shape[0], self._shape[1]]])
            return

        aspect = self._shape[1] / self._shape[0]
        tiles_per_dim = int(
            self._shape[0]
            / np.sqrt(self._shape[0] * self._shape[1] / (self._max_tiles * aspect))
        )

        # If only 1 tile was requested
        if tiles_per_dim == 1:
            self._tiles = np.array([[0, 0, self._shape[0], self._shape[1]]])
            return

        # Generate tile extents
        rows = np.ogrid[0 : self._shape[0] : 1j * (tiles_per_dim + 1)]  # type: ignore[misc]
        cols = np.ogrid[0 : self._shape[1] : 1j * (tiles_per_dim + 1)]  # type: ignore[misc]
        tiles = []

        # Counter for rows
        for ii in range(tiles_per_dim):
            rdiff = rows[ii + 1] - rows[ii]
            r0 = int(max(0, rows[ii] - self._dilation * rdiff))
            r1 = int(min(self._shape[0], rows[ii + 1] + self._dilation * rdiff))
            # Counter for columns
            for jj in range(tiles_per_dim):
                cdiff = cols[jj + 1] - cols[jj]
                c0 = int(max(0, cols[jj] - self._dilation * cdiff))
                c1 = int(min(self._shape[1], cols[jj + 1] + self._dilation * cdiff))
                tiles.append([r0, c0, r1, c1])

        # Track neighbors
        nbrs = []
        ntiles = len(tiles)
        for ii in range(ntiles - 1):
            for jj in range(ii + 1, ntiles):
                if intersects(tiles[ii], tiles[jj]):
                    nbrs.append([ii, jj])

        # Assign tiles
        self._tiles = np.array(tiles, dtype=int)
        self._neighbors = np.array(nbrs, dtype=int)


class DensityTiler(TilerInterface):
    """Tile up points by density.

    This is based on the nice implementation found here:
    https://mathoverflow.net/questions/412127/partitioning-unit-square-with-equal-frequency-rectangles
    """

    def __init__(
        self,
        points: np.ndarray,
        shape: tuple[int, int],
        max_tiles: int,
        dilation: float = 0.05,
    ):
        """Regularly sized tileset.

        Parameters
        ----------
        points: np.ndarray
            2D array with point coordinates within the shape.
        shape: tuple[int, int]
            Shape of rectangle to tile up.
        max_tiles: int
            Maximum number of tiles. Use of perfect squares recommended.
        dilation: float
            Fraction indicating the factor by which the tiles will be dilated
            for overlaps. 0.05 dilation will result in 10 percent overlap.
        """
        if points.ndim != 2:
            errmsg = f"Point coordinates must be a 2D array. Got {points.shape}"
            raise ValueError(errmsg)

        if (shape[0] <= 0) and (shape[1] <= 0):
            errmsg = f"Invalid shape provided to tiler: {shape}"
            raise ValueError(errmsg)

        if max_tiles <= 0:
            errmsg = f"Maximum tiles should atleast be 1. Got {max_tiles}"
            raise ValueError(errmsg)

        self._shape: tuple[int, int] = shape
        self._max_tiles: int = max_tiles
        self._dilation: float = dilation
        self._tiles: np.ndarray | None = None
        self._neighbors: np.ndarray | None = None
        self._generate_tiles(points)

    @property
    def ntiles(self) -> int:
        return self.tiles.shape[0]

    @property
    def bounds(self) -> tuple[int, int]:
        return self._shape

    @property
    def tiles(self) -> ArrayLike:
        return self._tiles

    @property
    def neighbors(self) -> ArrayLike | None:
        return self._neighbors

    def _generate_tiles(self, points: ArrayLike) -> None:
        """Generate the tiles and populate arrays."""
        # If only 1 tile was requested
        if self._max_tiles == 1:
            self._tiles = np.array([[0, 0, self._shape[0], self._shape[1]]])
            return

        bounds = ((0, self._shape[0]), (0, self._shape[1]))
        splits = split_rectangle(points, bounds, self._max_tiles)

        # Reorder indices
        tiles = [[s[0][0], s[1][0], s[0][1], s[1][1]] for s in splits]
        ntiles = len(tiles)

        # Dilate rectangles
        for ii in range(ntiles):
            t = tiles[ii]
            rdiff = t[2] - t[0]
            r0 = int(max(0, t[0] - self._dilation * rdiff))
            r1 = int(min(self._shape[0], t[2] + self._dilation * rdiff))
            cdiff = t[3] - t[1]
            c0 = int(max(0, t[1] - self._dilation * cdiff))
            c1 = int(min(self._shape[1], t[3] + self._dilation * cdiff))

            tiles[ii] = [r0, c0, r1, c1]

        # Track neighbors
        nbrs = []
        ntiles = len(tiles)
        for ii in range(ntiles - 1):
            for jj in range(ii + 1, ntiles):
                if intersects(tiles[ii], tiles[jj]):
                    nbrs.append([ii, jj])

        self._tiles = np.array(tiles)
        self._neighbors = np.array(nbrs)


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

    # splits are defined to be exaclty half-way in between : might not be optimal
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
