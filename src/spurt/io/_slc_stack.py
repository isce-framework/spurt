"""Read test SLC stacks during development."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ._raster import Raster
from ._three_d import Irreg3DInput

__all__ = [
    "SLCStackReader",
]


class SLCStackReader:
    """Read SLC stack from single files."""

    def __init__(
        self,
        folder_name: str | os.PathLike[str],
        qthreshold: float = 0.6,
    ):
        """Initialize stack by scanning a folder.

        We only care about temporal coherence and int files
        for now. metdata and coherence to be dealt with later.
        """
        # Here are members that we hold by default
        self.dates: list = []
        self.files: dict = {
            "quality": None,
            "slc": {},
        }
        self.raster_shape: tuple = ()
        self.folder_name = folder_name
        self.qthreshold = qthreshold

        # Discover and populate the files
        self._discover()

    def _discover(self) -> None:
        """Scan folder and set file names."""
        p = Path(self.folder_name)

        # First get temporal coherence
        self.files["quality"] = next(p.glob("temporal_coherence.tif")).absolute()
        # Read in shape
        with Raster(self.files["quality"]) as fin:
            self.raster_shape = fin.shape

        # Then list interferograms
        ifglist = sorted(p.glob("*.int.tif"))
        first_date = ifglist[0].name.split("_")[0]
        self.dates.append(first_date)

        for ifg in ifglist:
            if not ifg.name.startswith(first_date):
                errmsg = (
                    f"Error scanning {self.folder_name}. "
                    f"{ifg.name} does not conform to interferogram"
                    " naming convention."
                )
                raise ValueError(errmsg)

            acq_date = ifg.name.split("_")[1][:8]
            self.dates.append(acq_date)
            if acq_date in self.files["slc"]:
                errmsg = (
                    f"Error scanning {self.folder_name}. "
                    f"{acq_date} already in list of SLCs. "
                    f"{ifg.name} appears to be a duplicate"
                )
                raise ValueError(errmsg)

            self.files["slc"][acq_date] = ifg.absolute()

        # Check that there are files for every date
        assert len(self.files["slc"]) == len(self.dates) - 1

    def summary(self) -> dict:
        """Summary of the stack."""
        # To be determined
        return {}

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
            self._read_file(self.files["quality"], (slice(None), slice(None)))
            > self.qthreshold
        )

    def read_tile(
        self,
        space: tuple[slice, ...],
    ) -> Irreg3DInput:
        """Return a tile of 3D sparse data.

        We could potentially add a time slice as well at a later date.
        """
        # First read the quality file to get dimensions
        with Raster(self.files["quality"]) as fin:
            qlty = fin[space]

        msk = qlty > self.qthreshold
        xy = np.column_stack(np.where(msk))

        # Assumed complex64 for now but can read one file and determine
        arr = np.ones((len(self.dates), xy.shape[0]), dtype=np.complex64)

        for ind, acqdate in enumerate(self.dates):

            # Reference date usually does not have corresponding file
            if ind == 0:
                continue

            # Get the name of slc file
            slcname = self.files["slc"][acqdate]
            with Raster(slcname) as fin:
                arr[ind, :] = fin[space][msk]

        return Irreg3DInput(arr, xy)
