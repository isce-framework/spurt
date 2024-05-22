from __future__ import annotations

import numpy as np

import spurt


def test_cpu_count():
    ncpu = spurt.utils.get_cpu_count()

    # Check that atleast one is returned
    assert ncpu >= 1


def test_tiler_regular():
    tiler = spurt.utils.create_tiles_regular((1000, 1000), max_tiles=16).dilate(0.05)
    assert tiler.ntiles == 16
    for tile in tiler.tiles:
        assert tile.count > (250 * 250)


def test_tiler_density():
    ii, jj = np.meshgrid(np.arange(400), np.arange(400))
    points = np.column_stack((ii.flatten(), jj.flatten()))
    tiler = spurt.utils.create_tiles_density(points, (400, 400), max_tiles=16).dilate(
        0.05
    )
    assert tiler.ntiles == 16
    for tile in tiler.tiles:
        assert tile.count > (100 * 100)
