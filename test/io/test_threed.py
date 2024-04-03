from __future__ import annotations

import numpy as np

import spurt


class Test3D:
    def test_reg3d_input(self):
        data = np.ones((2, 4, 5), dtype=np.complex64)
        stack = spurt.io.Reg3DInput(data)

        assert stack.dtype == data.dtype
        assert stack.ndim == 2
        assert stack.time_dim == 0
        assert stack.space_dim == 1
        assert stack.shape == (data.shape[0], data.shape[1] * data.shape[2])
        assert stack.get_time_slice(0).size == data.shape[1] * data.shape[2]
        assert stack.get_spatial_slice(1).size == data.shape[0]

    def test_irreg3d_input(self):
        data = np.ones((2, 15), dtype=np.complex64)
        xy = np.arange(data.size).reshape((15, 2))
        stack = spurt.io.Irreg3DInput(data, xy)

        assert stack.dtype == data.dtype
        assert stack.ndim == 2
        assert stack.time_dim == 0
        assert stack.space_dim == 1
        assert stack.shape == data.shape
        assert stack.get_time_slice(1).size == data.shape[1]
        assert stack.get_spatial_slice(5).size == data.shape[0]
