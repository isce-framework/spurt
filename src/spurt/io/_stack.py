from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike

__all__ = [
    "InputStackDataset",
    "OutputStackDataset",
]


@runtime_checkable
class StackDataset(Protocol):
    """
    Common stack interface for input and output.

    Such objects must exportc Numpy-like `dtype`, `shape` and `ndim`
    attributes. They should also indicate which dimension represents
    time and space axes.
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
class InputStackDataset(StackDataset, Protocol):
    """
    An array-like interface for reading input datasets.

    `InputStackDataset` defines the abstract interface that types must conform
    to in order to be valid inputs to ``spurt.unwrap()` function.
    """

    def get_time_slice(self, key: int) -> ArrayLike:
        """Read a block of data in time."""

    def get_spatial_slice(self, key: int) -> ArrayLike:
        """Read a block of data in space."""


@runtime_checkable
class OutputStackDataset(StackDataset, Protocol):
    """
    An array-like interface for writing output datasets.

    `InputStackDataset` defines the abstract interface that types must conform
    to in order to be valid inputs to ``spurt.unwrap()` function.
    """

    def set_time_slice(self, key: int, array: ArrayLike) -> None:
        """Write a block of data in time."""

    def spatial_slice(self, key: int, array: ArrayLike) -> None:
        """Write a block of data in space."""
