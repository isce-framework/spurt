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


class TestExtractDateStr:
    """Tests for the date-string extractor used during stack scanning."""

    def test_default_format_extracts_eight_chars(self):
        from spurt.io._slc_stack import _date_str_length, _extract_date_str

        date_len = _date_str_length("%Y%m%d")
        assert _extract_date_str("20240709_extra", "%Y%m%d", date_len) == "20240709"

    def test_with_time_of_day(self):
        from spurt.io._slc_stack import _date_str_length, _extract_date_str

        fmt = "%Y%m%d%H%M%S"
        date_len = _date_str_length(fmt)
        token = "20240709040329_secondary"
        assert _extract_date_str(token, fmt, date_len) == "20240709040329"

    def test_invalid_date_raises(self):
        from spurt.io._slc_stack import _date_str_length, _extract_date_str

        date_len = _date_str_length("%Y%m%d")
        with pytest.raises(ValueError, match="does not match format"):
            _extract_date_str("notadate_xx", "%Y%m%d", date_len)

    def test_format_mismatch_raises(self):
        """A format that doesn't match the filename token should fail loudly."""
        from spurt.io._slc_stack import _date_str_length, _extract_date_str

        # Token has compact YYYYMMDD but user passed dashed format of same length
        date_len = _date_str_length("%Y-%m-%d")
        with pytest.raises(ValueError, match="does not match format"):
            _extract_date_str("20240709aa", "%Y-%m-%d", date_len)


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
