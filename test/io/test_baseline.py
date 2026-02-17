"""Tests for baseline CSV loading functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from spurt.io import BaselineData, load_baseline_csv


def test_load_per_slc_csv():
    """Test loading per-SLC format CSV."""
    csv_content = """date,bperp_m
20200101,0.0
20200113,150.5
20200125,-200.3
20200206,50.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        baseline_data = load_baseline_csv(csv_path)

        assert isinstance(baseline_data, BaselineData)
        assert len(baseline_data.dates) == 4
        assert len(baseline_data.bperp_m) == 4

        # Check dates are sorted
        expected_dates = np.array(
            ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(baseline_data.dates, expected_dates)

        # Check baselines
        expected_bperp = np.array([0.0, 150.5, -200.3, 50.0])
        np.testing.assert_array_almost_equal(baseline_data.bperp_m, expected_bperp)
    finally:
        csv_path.unlink()


def test_load_per_slc_csv_unsorted():
    """Test that per-SLC CSV with unsorted dates gets sorted."""
    csv_content = """date,bperp_m
20200125,-200.3
20200101,0.0
20200206,50.0
20200113,150.5
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        baseline_data = load_baseline_csv(csv_path)

        # Check dates are sorted
        expected_dates = np.array(
            ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(baseline_data.dates, expected_dates)

        # Check baselines are reordered to match sorted dates
        expected_bperp = np.array([0.0, 150.5, -200.3, 50.0])
        np.testing.assert_array_almost_equal(baseline_data.bperp_m, expected_bperp)
    finally:
        csv_path.unlink()


def test_load_per_ifg_csv():
    """Test loading per-IFG format CSV and conversion to per-SLC."""
    # Create IFG baselines that correspond to:
    # SLC baselines: [0, 100, 200, 300] (linear progression)
    # IFG 0-1: 100, IFG 1-2: 100, IFG 0-2: 200, IFG 2-3: 100
    csv_content = """reference,secondary,bperp_m
20200101,20200113,100.0
20200113,20200125,100.0
20200101,20200125,200.0
20200125,20200206,100.0
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        baseline_data = load_baseline_csv(csv_path)

        assert len(baseline_data.dates) == 4

        # First SLC should have bperp=0 (reference)
        assert baseline_data.bperp_m[0] == 0.0

        # Check that baselines are consistent with IFG differences
        # bperp[1] - bperp[0] should be ~100
        assert (
            np.abs(baseline_data.bperp_m[1] - baseline_data.bperp_m[0] - 100.0) < 1e-6
        )
        # bperp[2] - bperp[1] should be ~100
        assert (
            np.abs(baseline_data.bperp_m[2] - baseline_data.bperp_m[1] - 100.0) < 1e-6
        )
        # bperp[2] - bperp[0] should be ~200
        assert (
            np.abs(baseline_data.bperp_m[2] - baseline_data.bperp_m[0] - 200.0) < 1e-6
        )
    finally:
        csv_path.unlink()


def test_invalid_csv_raises():
    """Test that invalid CSV format raises ValueError."""
    csv_content = """foo,bar,baz
1,2,3
4,5,6
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(csv_content)
        f.flush()
        csv_path = Path(f.name)

    try:
        with pytest.raises(ValueError, match="Unrecognized CSV format"):
            load_baseline_csv(csv_path)
    finally:
        csv_path.unlink()


def test_baseline_data_shape_mismatch():
    """Test that BaselineData validates shape consistency."""
    dates = np.array(["2020-01-01", "2020-01-13"], dtype="datetime64[D]")
    bperp = np.array([0.0, 100.0, 200.0])  # Wrong size

    with pytest.raises(ValueError, match="Shape mismatch"):
        BaselineData(dates=dates, bperp_m=bperp)


def test_baseline_data_not_1d():
    """Test that BaselineData rejects non-1D arrays."""
    dates = np.array([["2020-01-01"], ["2020-01-13"]], dtype="datetime64[D]")
    bperp = np.array([[0.0], [100.0]])

    with pytest.raises(ValueError, match="Expected 1D arrays"):
        BaselineData(dates=dates, bperp_m=bperp)
