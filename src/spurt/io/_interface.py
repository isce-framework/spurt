"""Protocols for input and output stack interfaces from unwrapping modules."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "InputInterface",
    "OutputInterface",
    "InputStackInterface",
    "OutputStackInterface",
]


@runtime_checkable
class InputInterface(Protocol):
    """
    Common stack interface for input and output.

    Such objects must export NumPy-like `dtype`, `shape` and `ndim`
    attributes.
    """

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype : Data-type of the array's elements."""

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int : Tuple of array dimensions."""  # noqa: D403

    @property
    def ndim(self) -> int:
        """int : Number of array dimension."""  # noqa: D403

    def __getitem__(self, key: slice | tuple[slice, ...], /) -> ArrayLike:
        """Read a block of data."""


@runtime_checkable
class OutputInterface(Protocol):
    """
    An array-like interface for writing output datasets.

    `OutputInterface` must export NumPy-like `dtype`, `shape`, and `ndim`
    attributes and must support NumPy-style slice-based indexing.
    """

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype : Data-type of the array's elements."""

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int : Tuple of array dimensions."""  # noqa: D403

    @property
    def ndim(self) -> int:
        """int : Number of array dimensions."""  # noqa: D403

    def __setitem__(self, key: slice | tuple[slice, ...], value: np.ndarray, /) -> None:
        """Write a block of data."""


@runtime_checkable
class StackInterface(Protocol):
    """
    Common stack interface for input and output.

    They should indicate which dimension represents time and space axes.
    Time dimension could represent a single acquisition time or a pair of
    times for interferograms. Spatial dimension could represent a single
    point in space or an arc connecting two points in space. This interface
    is meant to be generic and will be used as a base class of specific type
    of stacks based on context in which they are used in the processing
    pipeline.
    """

    @property
    def dtype(self) -> np.dtype:
        """numpy.dtype : Data-type of the array's elements."""

    @property
    def shape(self) -> tuple[int, ...]:
        """tuple of int : Tuple of array dimensions."""  # noqa: D403

    @property
    def ndim(self) -> int:
        """int : Number of array dimension."""  # noqa: D403

    @property
    def time_dim(self) -> int:
        """int: Returns the dimension of array corresponding to time."""

    @property
    def space_dim(self) -> int:
        """int: Returns the dimension of array corresponding to space."""


@runtime_checkable
class InputStackInterface(StackInterface, Protocol):
    """
    An array-like interface for reading input datasets.

    `InputStackInterface` defines the abstract interface that types must
    conform to in order to be valid inputs to ``spurt.unwrap()` function.
    """

    def get_time_slice(self, key: int) -> ArrayLike:
        """Read a block of data in time."""

    def get_spatial_slice(self, key: int) -> ArrayLike:
        """Read a block of data in space."""


@runtime_checkable
class OutputStackInterface(StackInterface, Protocol):
    """
    An array-like interface for writing output datasets.

    `OutputStackInterface` defines the abstract interface that types must
    conform to in order to be valid inputs to ``spurt.unwrap()` function.
    """

    def set_time_slice(self, key: int, array: ArrayLike) -> None:
        """Write a block of data in time."""

    def set_spatial_slice(self, key: int, array: ArrayLike) -> None:
        """Write a block of data in space."""
