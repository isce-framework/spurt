"""Handle 3D interfaces for unwrapping."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from ._interface import InputInterface, InputStackInterface

__all__ = [
    "Reg3DInput",
    "Irreg3DInput",
]


class Reg3DInput(InputStackInterface):
    """A single numpy 3D array as input.

    A 2D image will just be a 3D array with size of time axis set to 1. This is
    largely a helper class to assist in development. `time_dim` is an optional
    input to indicate time axis.
    """

    def __init__(
        self,
        data: InputInterface,
        time_dim: int = 0,
    ):
        """Initialize a stack interface for a numpy array.

        Representing 3D data in a cube - regularly spaced in spatial dimension.


        Parameters
        ----------
        data: np.ndarray
            Numpy 3D array as input

        time_dim: int, optional
            Axis of the numpy array that represents the time dimension
        """
        # Check number of dimensions of input array
        if data.ndim != 3:
            clsname = type(self).__name__
            errmsg = (
                f"{clsname} only supports 3D arrays. Unsupported shape {data.shape}"
            )
            raise ValueError(errmsg)

        # Check if time_dim is within limits
        if time_dim >= data.ndim:
            clsname = type(self).__name__
            errmsg = f"{clsname} received time dimension outside array dimensions"
            raise ValueError(errmsg)

        # We only hold a reference to the data
        self._data = data
        self._time_dim = time_dim

    @property
    def dtype(self) -> DTypeLike:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return 2

    @property
    def time_dim(self) -> int:
        """We flatten spatial dimensions."""
        return int(self._time_dim > 0)

    @property
    def space_dim(self) -> int:
        """We flatten spatial dimensions."""
        return int(self._time_dim == 0)

    @property
    def _space_shape(self) -> tuple[int, ...]:
        return tuple(
            ss for ii, ss in enumerate(self._data.shape) if ii != self._time_dim
        )

    @property
    def _space_size(self) -> int:
        return int(np.prod(self._space_shape))

    @property
    def _time_size(self) -> int:
        return self._data.shape[self._time_dim]

    @property
    def shape(self) -> tuple[int, ...]:
        if self.time_dim:
            return (self._space_size, self._time_size)

        return (self._time_size, self._space_size)

    @property
    def xcoords(self) -> ArrayLike:
        """Return the x-coordinate of the points as an array.

        This is to mimic an irregular grid.
        """
        return np.arange(np.prod(self._data.shape), dtype=int) % self._space_shape[1]

    @property
    def ycoords(self) -> ArrayLike:
        """Return the y-coordinate of the points as an array.

        This is to mimic an irregular grid.
        """
        return np.arange(np.prod(self._data.shape), dtype=int) // self._space_shape[1]

    def get_time_slice(self, key: int) -> ArrayLike:
        """Read a block of data in time.

        Only makes a copy if absolutely needed.
        """
        ind = tuple(
            slice(None) if ii != self._time_dim else int(key)
            for ii in range(self._data.ndim)
        )
        return self._data[ind].ravel()  # type: ignore[arg-type, index]

    def get_spatial_slice(self, key: int) -> ArrayLike:
        """Read a block of data in space.

        Only makes a copy if absolutely needed.
        """
        ind = list(np.unravel_index(key, self._space_shape))
        ind.insert(self._time_dim, slice(None))

        return self._data[tuple(ind)].ravel()  # type: ignore[arg-type, index]


class Irreg3DInput(InputStackInterface):
    """A single numpy 2D array as input.

    One axis represents time and the other some flattened representation
    in space. `time_dim` is an optional input to indicate time axis.
    """

    def __init__(
        self,
        data: InputInterface,
        xy: InputInterface,
        time_dim: int = 0,
    ):
        """Initialize a stack interface for a 2D numpy array.

        One axis represents time axis and the other spatial axis.

        Parameters
        ----------
        data: np.ndarray
            Numpy 2D array as input

        xy: np.ndarray
            Numpy npts x 2 array as input

        time_dim: int, optional
            Axis of the numpy array that represents the time dimension
        """
        # Check number of dimensions of input array
        if data.ndim != 2:
            clsname = type(self).__name__
            errmsg = (
                f"{clsname} only supports 2D arrays. Unsupported shape {data.shape}"
            )
            raise ValueError(errmsg)

        if xy.ndim != 2:
            clsname = type(self).__name__
            errmsg = f"{clsname} accepts xy as a 2D array. Unsupported shape {xy.shape}"
            raise ValueError(errmsg)

        # Check if time_dim is within limits
        if time_dim >= data.ndim:
            clsname = type(self).__name__
            errmsg = f"{clsname} received time dimension outside array dimensions"
            raise ValueError(errmsg)

        if xy.shape[0] != data.shape[1 - time_dim]:
            clsname = type(self).__name__
            errmsg = (
                f"{clsname} mismatch of data and xy arrays"
                f"{data.shape} vs {xy.shape} with time axis {time_dim}"
            )
            raise ValueError(errmsg)

        # We only hold a reference to the data
        self._data = data
        self._time_dim = time_dim
        self._xy = xy

    @property
    def dtype(self) -> DTypeLike:
        return self._data.dtype

    @property
    def ndim(self) -> int:
        return 2

    @property
    def time_dim(self) -> int:
        return self._time_dim

    @property
    def space_dim(self) -> int:
        return int(1 - self._time_dim)

    @property
    def _space_size(self) -> int:
        return self._data.shape[self.space_dim]

    @property
    def _time_size(self) -> int:
        return self._data.shape[self._time_dim]

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    @property
    def xcoords(self) -> ArrayLike:
        """Return the x-coordinate of the points as an array."""
        return self._xy[:, slice(0, 1)]

    @property
    def ycoords(self) -> ArrayLike:
        """Return the y-coordinate of the points as an array."""
        return self._xy[:, slice(1, 2)]

    def get_time_slice(self, key: int) -> ArrayLike:
        """Read a block of data in time.

        Only makes a copy if absolutely needed.

        Parameters
        ----------
        key: int
            Index on the time axis
        """
        ind = tuple(
            slice(None) if ii != self._time_dim else key
            for ii in range(self._data.ndim)
        )
        return self._data[ind].ravel()  # type: ignore[arg-type, index]

    def get_spatial_slice(self, key: int) -> ArrayLike:
        """Read a block of data in space.

        Only makes a copy if absolutely needed.

        Parameters
        ----------
        key: int
            Index on the space axis
        """
        ind = tuple(
            slice(None) if ii != self.space_dim else key
            for ii in range(self._data.ndim)
        )
        return self._data[tuple(ind)].ravel()  # type: ignore[arg-type, index]
