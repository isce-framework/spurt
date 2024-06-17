import numbers
from typing import Any

import h5py
import numpy as np
from scipy.sparse import csr_matrix

import spurt

from ._settings import GeneralSettings

logger = spurt.utils.logger


class Tile:
    """Utility class for managing one unwrapped tile."""

    def __init__(self, fname: str, bandidx: int):
        self._fname: str = fname
        self._idx: int = bandidx
        self._corrections: list[Any] = []
        self._graph_laplacian: Any | None = None
        self._correction_level: int = 0

    @property
    def idx(self) -> int:
        return self._idx

    @property
    def coords(self) -> np.ndarray:
        with h5py.File(self._fname, mode="r") as fid:
            pts = fid["/points"][...]
            tile = fid["/tile"][...]
            return pts + tile[None, :2]

    def get_uw_phase(self, level: int) -> np.ndarray:
        with h5py.File(self._fname, mode="r") as fid:
            uw_data: np.ndarray = fid["/uw_data"][self._idx, :]
            offset: float = fid["/phase_offset"][self._idx]

        return sum([uw_data + offset] + self._corrections[:level])

    def get_corrections_sum(self, level: int) -> np.ndarray:
        return sum(self._corrections[:level])

    @property
    def corrections(self) -> np.ndarray:
        return self.get_corrections_sum(self._correction_level)

    @property
    def uw_phase(self) -> np.ndarray:
        """Return corrections summed up to a level."""
        return self.get_uw_phase(self._correction_level)

    @property
    def raw_uw_phase(self) -> np.ndarray:
        """Unwrapped phase without corrections."""
        return self.get_uw_phase(0)

    @property
    def graph_laplacian(self) -> Any:
        if self._graph_laplacian is not None:
            return self._graph_laplacian

        g_space = spurt.graph.DelaunayGraph(self.coords)
        links = g_space.links
        nlinks = links.shape[0]
        data = np.ones(2 * nlinks, dtype=int)
        data[0::2] = -1
        amat = csr_matrix(
            (
                data,
                (np.arange(2 * nlinks, dtype=int) // 2, links.flatten()),
            )
        )

        self._graph_laplacian = amat.T.dot(amat)
        return self._graph_laplacian

    def add_correction(self, a) -> None:
        if isinstance(a, numbers.Number) or (
            isinstance(a, np.ndarray) and a.size == self.coords.shape[0]
        ):
            self._corrections.append(a)
        else:
            errmsg = "Cannot process correction"
            raise ValueError(errmsg)

    def reset_corrections(self) -> None:
        self._correction_level = 0
        self._corrections = []

    def increment_correction(self) -> None:
        self._correction_level += 1

    def reset_band_index(self, newidx: int) -> None:
        self._idx = newidx
        # This needs to be reset since we are looking at a new band now
        self.reset_corrections()


def write_single_tile(
    tile: Tile,
    g_time: spurt.graph.GraphInterface,
    shape: tuple[int, int],
    gen_settings: GeneralSettings,
) -> None:
    """Write a single tile as geotiffs to output file."""
    # Get indices of the interferograms
    ifgs = g_time.links

    coords = tile.coords

    # For each band
    for ii, ifg in enumerate(ifgs):
        fname = gen_settings.unw_filename(ifg[0], ifg[1])
        if fname.is_file():
            logger.info(
                f"{fname!s} for band {ii + 1} already exists. Skipping writing ..."
            )
            continue

        # Reset tile band index
        tile.reset_band_index(ii)

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
        ) as raster:
            raster[idx] = arr


def write_merged_band(
    tiles: list[Tile],
    g_time: spurt.graph.GraphInterface,
    idx: int,
    shape: tuple[int, int],
    gen_settings: GeneralSettings,
) -> None:
    """Write a single band after merging all the tiles."""
    ifg = g_time.links[idx]
    fname = gen_settings.unw_filename(ifg[0], ifg[1])
    if fname.is_file():
        logger.info(
            f"{fname!s} for band {idx + 1} already exists. Skipping writing ..."
        )
        return

    # check that we are looking at the same band in all tiles
    for tile in tiles:
        assert tile.idx == idx, "Band index mismatch"

    # Create full sized array
    arr = np.full(shape, np.nan, dtype=np.float32)

    # Now iterate over tiles and fill up the array
    for tile in tiles:
        coords = tile.coords

        int_corr = 2 * np.pi * np.rint(tile.corrections / (2 * np.pi))
        arr[coords[:, 0], coords[:, 1]] = tile.raw_uw_phase + int_corr

    # Now write array to file
    idx = np.s_[:, :]
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
    ) as raster:
        raster[idx] = arr
