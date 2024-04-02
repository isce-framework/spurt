"""Read test SLC stacks during development."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from ._raster import Raster
from ._three_d import Irreg3DInput

__all__ = [
    "SLCStackReader",
]


class SLCStackReader:
    """Read a SLC stack from single files.

    This is a specific type of stack where time domain represents acquisition
    dates and spatial domain represents SAR acquisition pixels. This class
    supports two types of stacks
        - Actual coregistered SLCs, i.e, one raster per date
        - Phase linked SLCs w.r.t first acquisition which is assumed to have
        zero phase.

    An additional temporal coherence or a quality file should be provided for
    pixel selection in the spatial dimension.


    Attributes
    ----------
    dates: list of str
        List of acquisition dates in order

    slc_files: dict
        SLC file corresponding to acquisition date

    temp_coh_file: str
        Temporal coherence file, indicative of quality

    temp_coh_threshold: float
        Minimum value of temporal coherence to consider a pixel stable

    raster_shape: tuple of int
        Shape of rasters in the stack

    """

    def __init__(
        self,
        slc_files: Mapping[str, str | os.PathLike[str] | None],
        temp_coh_file: str | os.PathLike[str],
        temp_coh_threshold: float = 0.6,
    ):
        self.slc_files = slc_files
        self.temp_coh_file = temp_coh_file
        self.temp_coh_threshold = temp_coh_threshold

        # Extract dates by getting a list of slc_files keys
        self.dates: list[str] = sorted(slc_files)

        # Get raster shape from temporal coherence
        with Raster(self.temp_coh_file) as fin:
            self.raster_shape = fin.shape

    @classmethod
    def from_phase_linked_directory(
        cls,
        folder: str | os.PathLike[str],
        temp_coh_threshold: float = 0.6,
    ) -> SLCStackReader:
        """Initialize stack by scanning a folder.

        We only care about temporal coherence and SLC files
        for now. metadata and spatial coherence to be dealt with later.
        This folder structure corresponds to current test data for `spurt`
        and will likely evolve.
        """
        p = Path(folder)

        # First get temporal coherence
        temp_coh_file = (p / "temporal_coherence.tif").absolute()
        if not temp_coh_file.exists():
            errmsg = f"Error scanning {folder}. temporal_coherence.tif not found."
            raise ValueError(errmsg)

        # Then list individual SLCs
        slclist = sorted(p.glob("*.int.tif"))
        first_date = slclist[0].name.split("_")[0][:8]

        # Start with first date - set to None
        # None is special case for reference epoch
        slc_files: dict[str, os.PathLike[str] | None] = {
            first_date: None,
        }

        for slc in slclist:
            if not slc.name.startswith(first_date):
                errmsg = (
                    f"Error scanning {folder}."
                    f" {slc.name} does not conform to phase linking"
                    " naming convention."
                )
                raise ValueError(errmsg)

            acq_date = slc.name.split("_")[1][:8]
            if acq_date in slc_files:
                errmsg = (
                    f"Error scanning {folder}."
                    f" {acq_date} already in list of SLCs. "
                    f"{slc.name} appears to be a duplicate"
                )
                raise ValueError(errmsg)

            slc_files[acq_date] = slc.absolute()

        return SLCStackReader(
            slc_files, temp_coh_file, temp_coh_threshold=temp_coh_threshold
        )

    @classmethod
    def from_slc_directory(
        cls,
        folder: str | os.PathLike[str],
        temp_coh_threshold: float = 0.6,
    ) -> SLCStackReader:
        """Initialize stack by scanning a folder.

        We only care about inverse amplitude dispersion and SLC files
        for now. metadata and spatial coherence to be dealt with later.
        This folder structure corresponds to current test data for `spurt`
        and will likely evolve. This is a totally made up directory structure
        to demonstrate use of same data structure. The temporal coherence file
        could just be a mask file as well for good pixels here.
        """
        p = Path(folder)

        # First get temporal coherence - this will be called inv_amp_dispersion
        temp_coh_file = (p / "inv_amp_dispersion.tif").absolute()
        if not temp_coh_file.exists():
            errmsg = f"Error scanning {folder}. inv_amp_dispersion.tif not found."
            raise ValueError(errmsg)

        # Then list individual SLCs
        slclist = sorted(p.glob("*.slc.tif"))
        slc_files = {}

        for slc in slclist:
            acq_date = slc.name.split("_")[0][:8]
            if acq_date in slc_files:
                errmsg = (
                    f"Error scanning {folder}."
                    f" {acq_date} already in list of SLCs. "
                    f"{slc.name} appears to be a duplicate"
                )
                raise ValueError(errmsg)

            slc_files[acq_date] = slc.absolute()

        return SLCStackReader(
            slc_files, temp_coh_file, temp_coh_threshold=temp_coh_threshold
        )

    def _read_file(
        self,
        infile: str | os.PathLike[str],
        key: slice | tuple[slice, ...],
    ) -> np.ndarray:
        """Read slice of a single file."""
        with Raster(infile) as fin:
            return fin[key]

    def get_valid_count(self):
        """Return number of pixels over qthreshold."""
        return np.sum(
            self._read_file(self.temp_coh_file, np.s_[:, :]) > self.temp_coh_threshold
        )

    def read_tile(
        self,
        space: tuple[slice, ...],
    ) -> Irreg3DInput:
        """Return a tile of 3D sparse data.

        We could potentially add a time slice as well at a later date.
        """
        # First read the quality file to get dimensions
        msk = self._read_file(self.temp_coh_file, space) > self.temp_coh_threshold
        xy = np.column_stack(np.where(msk))

        # Assumed complex64 for now but can read one file and determine
        arr = np.ones((len(self.dates), xy.shape[0]), dtype=np.complex64)

        for ind, acqdate in enumerate(self.dates):

            # Get the name of slc file
            slcname = self.slc_files[acqdate]

            # If reference epoch
            if slcname is None:
                continue

            arr[ind, :] = self._read_file(slcname, space)[msk]

        return Irreg3DInput(arr, xy)
