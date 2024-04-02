from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

import spurt


def has_rasterio() -> bool:
    """Check if `rasterio` can be imported."""
    return importlib.util.find_spec("rasterio") is not None


def get_testdata_dir() -> str:
    return os.environ.get(
        "SPURT_TEST_DATA", "/Users/piyushagram/DL/jplopera/spurt_test_data"
    )


def has_testdata() -> bool:
    """Check if test data is available on machine."""
    p = Path(get_testdata_dir())
    return p.is_dir()


@pytest.mark.skipif(
    (not has_rasterio()) or (not has_testdata()),
    reason="Either rasterio or test data not available",
)
class TestStack:
    def test_stack(self):
        stack = spurt.io.SLCStackReader.from_phase_linked_directory(
            get_testdata_dir(),
            temp_coh_threshold=0.65,
        )

        assert len(stack.dates) == 21
        assert len(stack.slc_files) == 21

        arr = stack.read_tile((slice(0, 1024), slice(0, 1024)))
        assert arr.dtype == np.complex64
        assert arr.shape[0] == 21
        assert sum(arr.get_time_slice(0) == 1.0) == arr.shape[1]
        assert stack.temp_coh_threshold == 0.65
