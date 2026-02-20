"""Tests for baseline CSV loading functionality."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from spurt.io import BaselineData, load_baseline_csv

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tmp_csv(content: str) -> Path:
    """Write content to a temporary CSV and return the path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    f.write(content)
    f.flush()
    f.close()
    return Path(f.name)


# ---------------------------------------------------------------------------
# BaselineData validation
# ---------------------------------------------------------------------------


def test_baseline_data_shape_mismatch():
    """Test that BaselineData validates shape consistency."""
    dates = np.array(["2020-01-01", "2020-01-13"], dtype="datetime64[D]")
    bperp = np.array([0.0, 100.0, 200.0])

    with pytest.raises(ValueError, match="Shape mismatch"):
        BaselineData(dates=dates, bperp_m=bperp)


def test_baseline_data_not_1d():
    """Test that BaselineData rejects non-1D arrays."""
    dates = np.array([["2020-01-01"], ["2020-01-13"]], dtype="datetime64[D]")
    bperp = np.array([[0.0], [100.0]])

    with pytest.raises(ValueError, match="Expected 1D arrays"):
        BaselineData(dates=dates, bperp_m=bperp)


# ---------------------------------------------------------------------------
# Invalid / unrecognized format
# ---------------------------------------------------------------------------


def test_invalid_csv_raises():
    """Test that invalid CSV format raises ValueError."""
    csv_path = _write_tmp_csv("foo,bar,baz\n1,2,3\n4,5,6\n")
    try:
        with pytest.raises(ValueError, match="Unrecognized CSV format"):
            load_baseline_csv(csv_path)
    finally:
        csv_path.unlink()


# ---------------------------------------------------------------------------
# Per-SLC format
# ---------------------------------------------------------------------------


def test_load_per_slc_csv():
    """Test loading per-SLC format CSV with YYYYMMDD dates."""
    csv_content = (
        "date,bperp_m\n20200101,0.0\n20200113,150.5\n20200125,-200.3\n20200206,50.0\n"
    )
    csv_path = _write_tmp_csv(csv_content)
    try:
        bd = load_baseline_csv(csv_path)

        assert isinstance(bd, BaselineData)
        assert len(bd.dates) == 4

        expected_dates = np.array(
            ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
        np.testing.assert_array_almost_equal(bd.bperp_m, [0.0, 150.5, -200.3, 50.0])
    finally:
        csv_path.unlink()


def test_load_per_slc_csv_unsorted():
    """Test that per-SLC CSV with unsorted dates gets sorted."""
    csv_content = (
        "date,bperp_m\n20200125,-200.3\n20200101,0.0\n20200206,50.0\n20200113,150.5\n"
    )
    csv_path = _write_tmp_csv(csv_content)
    try:
        bd = load_baseline_csv(csv_path)

        expected_dates = np.array(
            ["2020-01-01", "2020-01-13", "2020-01-25", "2020-02-06"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
        np.testing.assert_array_almost_equal(bd.bperp_m, [0.0, 150.5, -200.3, 50.0])
    finally:
        csv_path.unlink()


# ---------------------------------------------------------------------------
# Per-IFG format - simple YYYYMMDD dates
# ---------------------------------------------------------------------------


def test_load_per_ifg_csv():
    """Test loading per-IFG format CSV and conversion to per-SLC."""
    csv_content = (
        "reference,secondary,bperp_m\n"
        "20200101,20200113,100.0\n"
        "20200113,20200125,100.0\n"
        "20200101,20200125,200.0\n"
        "20200125,20200206,100.0\n"
    )
    csv_path = _write_tmp_csv(csv_content)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 4
        assert bd.bperp_m[0] == 0.0
        np.testing.assert_allclose(
            np.diff(bd.bperp_m), [100.0, 100.0, 100.0], atol=1e-6
        )
    finally:
        csv_path.unlink()


# ---------------------------------------------------------------------------
# Per-IFG format
# ---------------------------------------------------------------------------

_CAPELLA_CSV = """\
reference,secondary,reference_time_utc,secondary_time_utc,btemp_days,bperp_m,bpar_m,btotal_m,bradial_m,bnormal_m,look_angle_diff_rad
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260206022355_20260206022404.tif,2026-01-31T04:30:30.757865503Z,2026-02-06T02:23:59.736190396Z,5.912,79.512,65.021,106.861,18.250,105.291,0.00012
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260209012040_20260209012049.tif,2026-01-31T04:30:30.757865503Z,2026-02-09T01:20:44.804243296Z,8.868,369.187,285.877,469.717,70.814,464.348,0.00053
slcs/CAPELLA_C13_SP_SLC_HH_20260206022355_20260206022404.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260209012040_20260209012049.tif,2026-02-06T02:23:59.736190396Z,2026-02-09T01:20:44.804243296Z,2.956,289.675,220.856,362.856,52.564,359.057,0.00041
"""


def test_load_per_ifg_capella_filenames():
    """Test per-IFG loading with Capella-style filenames and UTC time columns."""
    csv_path = _write_tmp_csv(_CAPELLA_CSV)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 3
        expected_dates = np.array(
            ["2026-01-31", "2026-02-06", "2026-02-09"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)

        # Reference SLC should be zero
        assert bd.bperp_m[0] == 0.0

        # IFG baselines should be self-consistent (redundant network)
        np.testing.assert_allclose(bd.bperp_m[1] - bd.bperp_m[0], 79.512, atol=0.1)
        np.testing.assert_allclose(bd.bperp_m[2] - bd.bperp_m[0], 369.187, atol=0.1)
        np.testing.assert_allclose(bd.bperp_m[2] - bd.bperp_m[1], 289.675, atol=0.1)
    finally:
        csv_path.unlink()


def test_load_per_ifg_capella_without_time_columns():
    """Test that Capella filenames work even without *_time_utc columns.

    Dates should be extracted from the filename's first YYYYMMDD substring.
    """
    csv_content = (
        "reference,secondary,bperp_m\n"
        "slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,"
        "slcs/CAPELLA_C13_SP_SLC_HH_20260206022355_20260206022404.tif,79.512\n"
        "slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,"
        "slcs/CAPELLA_C13_SP_SLC_HH_20260209012040_20260209012049.tif,369.187\n"
    )
    csv_path = _write_tmp_csv(csv_content)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 3
        expected_dates = np.array(
            ["2026-01-31", "2026-02-06", "2026-02-09"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
        assert bd.bperp_m[0] == 0.0
    finally:
        csv_path.unlink()


# ---------------------------------------------------------------------------
# Single-reference IFG format (per-SLC baselines in IFG CSV)
# ---------------------------------------------------------------------------

_SINGLE_REF_CSV = """\
reference,secondary,bperp_m
20230105,20230117,95.3
20230105,20230129,210.7
20230105,20230210,305.1
"""


def test_load_single_reference_ifg():
    """Test single-reference IFG CSV uses direct assignment (no lstsq)."""
    csv_path = _write_tmp_csv(_SINGLE_REF_CSV)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 4
        assert bd.bperp_m[0] == 0.0
        # Values should be exact — no least-squares involved
        np.testing.assert_array_equal(bd.bperp_m, [0.0, 95.3, 210.7, 305.1])
    finally:
        csv_path.unlink()


_SINGLE_REF_CAPELLA_CSV = """\
reference,secondary,reference_time_utc,secondary_time_utc,btemp_days,bperp_m,bpar_m,btotal_m,bradial_m,bnormal_m,look_angle_diff_rad
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260206022355_20260206022404.tif,2026-01-31T04:30:30.757865503Z,2026-02-06T02:23:59.736190396Z,5.912,79.512,65.021,106.861,18.250,105.291,0.00012
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260209012040_20260209012049.tif,2026-01-31T04:30:30.757865503Z,2026-02-09T01:20:44.804243296Z,8.868,369.187,285.877,469.717,70.814,464.348,0.00053
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260212001725_20260212001734.tif,2026-01-31T04:30:30.757865503Z,2026-02-12T00:17:29.220234759Z,11.824,441.233,335.754,557.799,79.465,552.110,0.00063
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260214231409_20260214231418.tif,2026-01-31T04:30:30.757865503Z,2026-02-14T23:14:13.365825911Z,14.780,443.027,325.736,554.367,69.824,549.952,0.00064
slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif,slcs/CAPELLA_C13_SP_SLC_HH_20260217221050_20260217221059.tif,2026-01-31T04:30:30.757865503Z,2026-02-17T22:10:55.094275436Z,17.736,4.588,16.477,38.075,12.147,36.086,4.86e-05
"""


def test_load_single_reference_capella_full():
    """Test the actual Capella single-reference CSV format end-to-end.

    This mirrors the real output from the Capella baseline processor:
    all rows share one reference, with UTC time columns and extra baseline cols.
    """
    csv_path = _write_tmp_csv(_SINGLE_REF_CAPELLA_CSV)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 6
        expected_dates = np.array(
            [
                "2026-01-31",
                "2026-02-06",
                "2026-02-09",
                "2026-02-12",
                "2026-02-14",
                "2026-02-17",
            ],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)

        # Reference should be zero, values should be exact (single-ref path)
        assert bd.bperp_m[0] == 0.0
        np.testing.assert_allclose(bd.bperp_m[1], 79.512, atol=1e-3)
        np.testing.assert_allclose(bd.bperp_m[2], 369.187, atol=1e-3)
        np.testing.assert_allclose(bd.bperp_m[3], 441.233, atol=1e-3)
        np.testing.assert_allclose(bd.bperp_m[4], 443.027, atol=1e-3)
        np.testing.assert_allclose(bd.bperp_m[5], 4.588, atol=1e-3)
    finally:
        csv_path.unlink()


def test_single_ref_gives_same_as_lstsq():
    """Verify single-reference shortcut matches what lstsq would give."""
    # Build a small single-reference CSV that also works as a valid network
    csv_content = (
        "reference,secondary,bperp_m\n"
        "20230101,20230113,100.0\n"
        "20230101,20230125,250.0\n"
        "20230101,20230206,400.0\n"
    )
    csv_path = _write_tmp_csv(csv_content)
    try:
        bd = load_baseline_csv(csv_path)

        # Should be exact (no lstsq noise)
        np.testing.assert_array_equal(bd.bperp_m, [0.0, 100.0, 250.0, 400.0])
    finally:
        csv_path.unlink()


# ---------------------------------------------------------------------------
# _parse_date edge cases
# ---------------------------------------------------------------------------


def test_parse_date_iso_timestamp():
    """Test parsing ISO 8601 timestamps."""
    from spurt.io._baseline import _parse_date

    assert _parse_date("2026-01-31T04:30:30.757865503Z") == "2026-01-31"
    assert _parse_date("2026-02-06T02:23:59.736190396Z") == "2026-02-06"


def test_parse_date_filename():
    """Test date extraction from Capella-style filenames."""
    from spurt.io._baseline import _parse_date

    fname = "slcs/CAPELLA_C13_SP_SLC_HH_20260131043025_20260131043034.tif"
    assert _parse_date(fname) == "2026-01-31"


def test_parse_date_unparseable():
    """Test that unparseable strings raise ValueError."""
    from spurt.io._baseline import _parse_date

    with pytest.raises(ValueError, match="Cannot parse date"):
        _parse_date("no-date-here")


# ---------------------------------------------------------------------------
# OPERA CSLC-S1 filenames
# ---------------------------------------------------------------------------

_OPERA_CSLC_CSV = """\
reference,secondary,bperp_m
OPERA_L2_CSLC-S1_T078-165495-IW2_20230105T120000Z_20230120T000000Z_S1A_VV_v1.0.h5,OPERA_L2_CSLC-S1_T078-165495-IW2_20230117T120000Z_20230201T000000Z_S1A_VV_v1.0.h5,95.3
OPERA_L2_CSLC-S1_T078-165495-IW2_20230105T120000Z_20230120T000000Z_S1A_VV_v1.0.h5,OPERA_L2_CSLC-S1_T078-165495-IW2_20230129T120000Z_20230212T000000Z_S1A_VV_v1.0.h5,210.7
OPERA_L2_CSLC-S1_T078-165495-IW2_20230117T120000Z_20230201T000000Z_S1A_VV_v1.0.h5,OPERA_L2_CSLC-S1_T078-165495-IW2_20230129T120000Z_20230212T000000Z_S1A_VV_v1.0.h5,115.4
"""


def test_load_per_ifg_opera_cslc():
    """Test per-IFG loading with OPERA CSLC-S1 filenames (no time columns)."""
    csv_path = _write_tmp_csv(_OPERA_CSLC_CSV)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 3
        expected_dates = np.array(
            ["2023-01-05", "2023-01-17", "2023-01-29"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
        assert bd.bperp_m[0] == 0.0
        np.testing.assert_allclose(bd.bperp_m[1] - bd.bperp_m[0], 95.3, atol=0.1)
        np.testing.assert_allclose(bd.bperp_m[2] - bd.bperp_m[0], 210.7, atol=0.1)
    finally:
        csv_path.unlink()


def test_parse_date_opera_cslc():
    """Test date extraction from OPERA CSLC-S1 filenames."""
    from spurt.io._baseline import _parse_date

    f = "OPERA_L2_CSLC-S1_T078-165495-IW2_20230105T120000Z_20230120T000000Z_S1A_VV_v1.0.h5"
    assert _parse_date(f) == "2023-01-05"


_OPERA_COMPRESSED_CSLC_CSV = """\
reference,secondary,bperp_m
OPERA_L2_COMPRESSED-CSLC-S1_F23148_T078-165495-IW2_20230105T120000Z_20221001T000000Z_20230101T000000Z_20230115T000000Z_VV_v1.1.h5,OPERA_L2_COMPRESSED-CSLC-S1_F23148_T078-165495-IW2_20230117T120000Z_20230101T000000Z_20230201T000000Z_20230210T000000Z_VV_v1.1.h5,88.2
OPERA_L2_COMPRESSED-CSLC-S1_F23148_T078-165495-IW2_20230105T120000Z_20221001T000000Z_20230101T000000Z_20230115T000000Z_VV_v1.1.h5,OPERA_L2_COMPRESSED-CSLC-S1_F23148_T078-165495-IW2_20230129T120000Z_20230101T000000Z_20230301T000000Z_20230310T000000Z_VV_v1.1.h5,195.0
"""


def test_load_per_ifg_opera_compressed_cslc():
    """Test per-IFG loading with OPERA COMPRESSED-CSLC-S1 filenames."""
    csv_path = _write_tmp_csv(_OPERA_COMPRESSED_CSLC_CSV)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 3
        expected_dates = np.array(
            ["2023-01-05", "2023-01-17", "2023-01-29"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
        assert bd.bperp_m[0] == 0.0
        np.testing.assert_allclose(bd.bperp_m[1] - bd.bperp_m[0], 88.2, atol=0.1)
    finally:
        csv_path.unlink()


def test_parse_date_opera_compressed_cslc():
    """Test date extraction from OPERA COMPRESSED-CSLC-S1 filenames."""
    from spurt.io._baseline import _parse_date

    f = "OPERA_L2_COMPRESSED-CSLC-S1_F23148_T078-165495-IW2_20230105T120000Z_20221001T000000Z_20230101T000000Z_20230115T000000Z_VV_v1.1.h5"
    assert _parse_date(f) == "2023-01-05"


# ---------------------------------------------------------------------------
# Sentinel-1 IW SAFE filenames
# ---------------------------------------------------------------------------

_S1_SAFE_CSV = """\
reference,secondary,bperp_m
S1A_IW_SLC__1SDV_20230105T120000_20230105T120030_046801_059E44_A1B2.SAFE,S1A_IW_SLC__1SDV_20230117T120000_20230117T120030_046976_05A3F1_C3D4.SAFE,102.5
S1A_IW_SLC__1SDV_20230105T120000_20230105T120030_046801_059E44_A1B2.SAFE,S1B_IW_SLC__1SDV_20230129T120000_20230129T120030_047151_05A9BE_E5F6.SAFE,215.8
S1A_IW_SLC__1SDV_20230117T120000_20230117T120030_046976_05A3F1_C3D4.SAFE,S1B_IW_SLC__1SDV_20230129T120000_20230129T120030_047151_05A9BE_E5F6.SAFE,113.3
"""


def test_load_per_ifg_sentinel1_safe():
    """Test per-IFG loading with Sentinel-1 SAFE directory names."""
    csv_path = _write_tmp_csv(_S1_SAFE_CSV)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 3
        expected_dates = np.array(
            ["2023-01-05", "2023-01-17", "2023-01-29"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
        assert bd.bperp_m[0] == 0.0
        np.testing.assert_allclose(bd.bperp_m[1] - bd.bperp_m[0], 102.5, atol=0.1)
        np.testing.assert_allclose(bd.bperp_m[2] - bd.bperp_m[0], 215.8, atol=0.1)
        np.testing.assert_allclose(bd.bperp_m[2] - bd.bperp_m[1], 113.3, atol=0.1)
    finally:
        csv_path.unlink()


def test_parse_date_sentinel1_safe():
    """Test date extraction from Sentinel-1 SAFE names."""
    from spurt.io._baseline import _parse_date

    f = "S1A_IW_SLC__1SDV_20230105T120000_20230105T120030_046801_059E44_A1B2.SAFE"
    assert _parse_date(f) == "2023-01-05"


def test_load_per_ifg_sentinel1_safe_paths():
    """Test with full path-like Sentinel-1 SAFE references."""
    csv_content = (
        "reference,secondary,bperp_m\n"
        "/data/slcs/S1A_IW_SLC__1SDV_20230105T120000_20230105T120030_046801_059E44_A1B2.SAFE/measurement/s1a-iw1-slc-vv.tiff,"
        "/data/slcs/S1A_IW_SLC__1SDV_20230117T120000_20230117T120030_046976_05A3F1_C3D4.SAFE/measurement/s1a-iw1-slc-vv.tiff,102.5\n"
    )
    csv_path = _write_tmp_csv(csv_content)
    try:
        bd = load_baseline_csv(csv_path)

        assert len(bd.dates) == 2
        expected_dates = np.array(
            ["2023-01-05", "2023-01-17"],
            dtype="datetime64[D]",
        )
        np.testing.assert_array_equal(bd.dates, expected_dates)
    finally:
        csv_path.unlink()
