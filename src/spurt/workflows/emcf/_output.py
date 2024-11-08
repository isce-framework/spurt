from __future__ import annotations

import numbers
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import spurt

logger = spurt.utils.logger


class Tile:
    """Utility class for managing one unwrapped tile."""

    def __init__(self, fname: str):
        """Handle one tile of unwrapped output.

        Parameters
        ----------
        fname: str
            Filename of intermediate HDF5 file with output for one unwrapped
            tile.
        """
        self._fname: str = fname
        self._idx: list[int] = []
        self._corrections: list[Any] = []
        self._graph_laplacian: Any | None = None
        self._correction_level: int = 0
        self._uwdata: np.ndarray | None = None
        self._coords: np.ndarray | None = None

        self.init_coords()

    @property
    def idx(self) -> list[int]:
        """Band index currently being tracked."""
        return self._idx

    @property
    def correction_level(self) -> int:
        """Correction level being tracked."""
        return self._correction_level

    def init_coords(self) -> None:
        """Point coordinates in global grid."""
        with h5py.File(self._fname, mode="r") as fid:
            pts = fid["/points"][...]
            tile = fid["/tile"][...]

        self._coords = pts + tile[None, :2]

    @property
    def coords(self) -> np.ndarray:
        """Return point coordinates."""
        return self._coords

    def init_uw_data(self) -> None:
        """Initialize with unwrapped tile data."""
        self.check_idx()
        with h5py.File(self._fname, mode="r") as fid:
            uw_data: np.ndarray = fid["/uw_data"][self._idx, :]
            offset: np.ndarray = np.expand_dims(fid["/phase_offset"][self._idx], 1)

        self._uwdata = uw_data + offset

    def check_idx(self) -> None:
        if not self._idx:
            errmsg = "Band indices not initialized."
            raise RuntimeError(errmsg)

    def get_uw_phase(self, level: int) -> np.ndarray:
        """Unwrapped phase with corrections applied to specified level."""
        return self._uwdata + self.get_corrections_sum(level)

    def get_corrections_sum(self, level: int) -> np.ndarray:
        """Corrections to unwrapped phase up to a specified level."""
        return sum(self._corrections[:level])

    @property
    def corrections(self) -> np.ndarray:
        """Correction up to the current level."""
        return self.get_corrections_sum(self._correction_level)

    @property
    def uw_phase(self) -> np.ndarray:
        """Return corrections summed up to the current level."""
        return self.get_uw_phase(self._correction_level)

    @property
    def raw_uw_phase(self) -> np.ndarray:
        """Unwrapped phase without corrections."""
        return self.get_uw_phase(0)

    @property
    def graph_laplacian(self) -> Any:
        """Graph laplacian for the tile."""
        if self._graph_laplacian is not None:
            return self._graph_laplacian

        g_space = spurt.graph.DelaunayGraph(self.coords)
        self._graph_laplacian = spurt.graph.graph_laplacian(g_space)
        return self._graph_laplacian

    def add_correction(self, a) -> None:
        """Add a constant or pixel-by-pixel correction."""
        if isinstance(a, numbers.Number):
            self._corrections.append(a)
        elif isinstance(a, np.ndarray) and a.size == 1:
            self._corrections.append(a.item())
        elif isinstance(a, np.ndarray) and a.ndim == 1 and a.size == len(self._idx):
            self._corrections.append(np.expand_dims(a, 1))
        elif isinstance(a, np.ndarray) and a.shape[-1] == self.coords.shape[0]:
            self._corrections.append(a)
        else:
            errmsg = "Cannot process correction"
            raise ValueError(errmsg)

    def reset_corrections(self) -> None:
        """Reset corrections to reuse object with another band index."""
        self._correction_level = 0
        self._corrections = []

    def increment_correction(self) -> None:
        """Increment current correction level."""
        self._correction_level += 1

    def compress_corrections(self) -> None:
        """Compress corrections to single level to reduce memory usage."""
        if len(self._corrections) > 1:
            self._corrections[0] = self._corrections[0] + self._corrections.pop()

        if len(self._corrections) > 1:
            errmsg = "More than one correction in compress mode"
            raise RuntimeError(errmsg)

    def reset_band_index(self, newidx: list[int]) -> None:
        """Change current band index."""
        self._idx = newidx
        self.init_uw_data()
        # This needs to be reset since we are looking at a new band now
        self.reset_corrections()


def write_single_tile(
    tile: Tile,
    fnames: list[Path],
    shape: tuple[int, int],
    like: str | os.PathLike[str] | None = None,
) -> None:
    """Write a single tile as GeoTIFFs to output file."""
    # Get indices of the interferograms
    coords = tile.coords

    like_raster = None if like is None else spurt.io.Raster(like)
    # For each band
    for ii, fname in enumerate(fnames):
        if fname.is_file():
            logger.info(
                f"{fname!s} for band {ii + 1} already exists. Skipping writing ..."
            )
            continue

        # Reset tile band index
        tile.reset_band_index([ii])

        arr = np.full(shape, np.nan, dtype=np.float32)
        arr[coords[:, 0], coords[:, 1]] = tile.raw_uw_phase
        idx = np.s_[:, :]
        logger.info(f"Writing band {ii + 1} to {fname!s}")
        with spurt.io.Raster.create(
            str(fname),
            width=shape[1],
            height=shape[0],
            dtype=np.float32,
            nodata=np.nan,
            driver="GTiff",
            tiled=True,
            blockxsize=512,
            blockysize=512,
            compress="DEFLATE",
            like=like_raster,
        ) as raster:
            raster[idx] = arr


def write_merged_band(
    tiles: dict[int, Tile],
    fname: Path,
    idx: int,
    shape: tuple[int, int],
    like: str | os.PathLike[str] | None = None,
) -> None:
    """Write a single band after merging all the tiles."""
    if fname.is_file():
        logger.info(
            f"{fname!s} for band {idx + 1} already exists. Skipping writing ..."
        )
        return

    # check that we are looking at the same band in all tiles
    for tile in tiles.values():
        assert idx in tile.idx, "Band index mismatch"
        assert tile.correction_level == 1, "Only one offset supported"

    # Get the index of idx in batch
    ind = tile.idx.index(idx)

    # Create full sized array
    model = np.zeros(shape, dtype=np.float32)
    cnt = np.zeros(shape, dtype=np.int16)
    minval = np.full(shape, np.inf, dtype=np.float32)
    maxval = np.full(shape, -np.inf, dtype=np.float32)

    # Now iterate over tiles and compute average
    for tile in tiles.values():
        coords = tile.coords.astype(np.int32)
        c0 = coords[:, 0]
        c1 = coords[:, 1]

        data = tile.uw_phase[ind]
        minval[c0, c1] = np.minimum(data, minval[c0, c1])
        maxval[c0, c1] = np.maximum(data, maxval[c0, c1])

        model[c0, c1] += data
        cnt[c0, c1] += 1

    mask = cnt != 0
    model[~mask] = np.nan
    model[mask] /= cnt[mask]
    diff = maxval - minval
    diff[cnt < 2] = np.nan
    diff[np.isinf(diff)] = np.nan

    # Now iterate over tiles and wrap around model
    arr = np.full(shape, np.nan, dtype=np.float32)
    for tile in tiles.values():
        coords = tile.coords.astype(np.int32)
        c0 = coords[:, 0]
        c1 = coords[:, 1]
        mmodel = model[c0, c1]
        d = tile.raw_uw_phase[ind] - mmodel

        arr[c0, c1] = mmodel + d - 2 * np.pi * np.round(d / (2 * np.pi))

    # Now write array to file
    sidx = np.s_[:, :]
    like_raster = None if like is None else spurt.io.Raster(like)
    logger.info(f"Writing band {idx + 1} to {fname!s}")
    with spurt.io.Raster.create(
        str(fname),
        width=shape[1],
        height=shape[0],
        dtype=np.float32,
        nodata=np.nan,
        driver="GTiff",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        like=like_raster,
    ) as raster:
        raster[sidx] = arr

    # Now write array to file
    with spurt.io.Raster.create(
        str(fname).replace(".tif", "_diff.tif"),
        width=shape[1],
        height=shape[0],
        dtype=np.float32,
        nodata=np.nan,
        driver="GTiff",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        like=like_raster,
    ) as raster:
        raster[sidx] = diff

    # Now write array to file
    with spurt.io.Raster.create(
        str(fname).replace(".tif", "_model.tif"),
        width=shape[1],
        height=shape[0],
        dtype=np.float32,
        nodata=np.nan,
        driver="GTiff",
        tiled=True,
        blockxsize=512,
        blockysize=512,
        compress="DEFLATE",
        like=like_raster,
    ) as raster:
        raster[sidx] = model
