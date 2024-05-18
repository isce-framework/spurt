from __future__ import annotations

import numpy as np

import spurt


class TestUtils:
    def test_cpu_count(self):
        ncpu = spurt.utils.get_cpu_count()

        # Check that atleast one is returned
        assert ncpu >= 1

    def test_tiler_regular(self):
        tiler = spurt.utils.RegularTiler((1000, 1000), max_tiles=16)
        assert tiler.ntiles == 16
        assert len(tiler.neighbors) > 0
        for tile in tiler.tiles:
            assert (tile[2] - tile[0]) * (tile[3] - tile[1]) > (250 * 250)

    def test_tiler_density(self):
        ii, jj = np.meshgrid(np.arange(400), np.arange(400))
        points = np.column_stack((ii.flatten(), jj.flatten()))
        tiler = spurt.utils.DensityTiler(points, (400, 400), max_tiles=16)
        assert tiler.ntiles == 16
        assert len(tiler.neighbors) > 0
        for tile in tiler.tiles:
            assert (tile[2] - tile[0]) * (tile[3] - tile[1]) > (100 * 100)
