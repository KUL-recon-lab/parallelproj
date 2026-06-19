"""Data utilities for memory-efficient iterative reconstruction."""

from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import array_api_compat

from ._backend import Array, to_numpy_array


class SubsetArrayMmap:
    """Memory-mapped array split into equal subsets on disk.

    The file on disk has shape ``(num_subsets, *subset_shape)`` in
    C-contiguous order, so each subset occupies a single contiguous block.
    The OS memory-maps the file and loads pages on demand; pages are
    evicted automatically when RAM is scarce.

    Parameters
    ----------
    path : str or Path
        Path to a binary file produced by :func:`to_subset_mmap`
        (or any code that writes data in the expected on-disk layout).
    num_subsets : int
        Number of subsets stored in the file.
    subset_shape : tuple of int
        Shape of a single subset's data array, e.g.
        ``(num_rad, views_per_subset, num_planes, num_tofbins)`` for a
        TOF sinogram in RVP order.
    dtype : numpy dtype, optional
        Element type of the stored data.  Default ``float32``.
    mode : str, optional
        :func:`numpy.memmap` open mode.  Use ``'r'`` (default) for
        read-only access.  Pass ``'r+'`` to allow in-place writes, or
        ``'w+'`` to create / overwrite the file.

    Examples
    --------
    >>> mmap = SubsetArrayMmap("y_subsets.bin", num_subsets=24,
    ...                        subset_shape=(171, 4, 19, 13))
    >>> y_k = mmap[3]   # owned float32 ndarray of shape (171, 4, 19, 13)
    >>> del y_k          # RAM freed immediately
    """

    def __init__(
        self,
        path: str | os.PathLike,
        num_subsets: int,
        subset_shape: tuple[int, ...],
        *,
        dtype: np.dtype = np.float32,
        mode: str = "r",
    ) -> None:
        self._path = Path(path)
        self._num_subsets = int(num_subsets)
        self._subset_shape = tuple(subset_shape)
        self._dtype = np.dtype(dtype)
        self._mmap = np.memmap(
            self._path,
            dtype=self._dtype,
            mode=mode,
            shape=(self._num_subsets,) + self._subset_shape,
        )

    # ------------------------------------------------------------------
    # sequence interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._num_subsets

    def __getitem__(self, k: int) -> np.ndarray:
        """Return an *owned* NumPy copy of subset *k*.

        The returned array is a plain ``ndarray`` (not a ``memmap``), so
        its lifetime is fully controlled by Python's reference counter.
        The underlying OS pages become eviction-eligible as soon as the
        array is deleted.
        """
        return np.array(self._mmap[k])

    # ------------------------------------------------------------------
    # metadata
    # ------------------------------------------------------------------

    @property
    def shape(self) -> tuple[int, ...]:
        """Full shape of the on-disk array: ``(num_subsets, *subset_shape)``."""
        return (self._num_subsets,) + self._subset_shape

    @property
    def subset_shape(self) -> tuple[int, ...]:
        """Shape of a single subset's data."""
        return self._subset_shape

    @property
    def num_subsets(self) -> int:
        """Number of subsets stored in the file."""
        return self._num_subsets

    @property
    def path(self) -> Path:
        """Path to the binary file on disk."""
        return self._path

    def nbytes_per_subset(self) -> int:
        """Bytes occupied in RAM when one subset is loaded."""
        return math.prod(self._subset_shape) * self._dtype.itemsize

    def nbytes_total(self) -> int:
        """Total size of the on-disk file in bytes."""
        return self._num_subsets * self.nbytes_per_subset()


# ---------------------------------------------------------------------------
# factory
# ---------------------------------------------------------------------------


def to_subset_mmap(
    full_array: np.ndarray,
    subset_slices: list[tuple],
    path: str | os.PathLike,
    dtype: np.dtype = np.float32,
) -> SubsetArrayMmap:
    """Write a full array to a subset-contiguous binary file.

    Each non-contiguous subset slice is gathered from ``full_array`` and
    written as a contiguous block so that a subsequent ``mmap[k]`` read is
    a single sequential I/O operation.  The resulting access pattern lets
    the OS prefetch the next subset from disk while the algorithm computes
    on the current one.

    Parameters
    ----------
    full_array : numpy ndarray
        Full data array.  Any axis order is accepted; the subset shape is
        inferred from ``full_array[subset_slices[0]]``.
    subset_slices : list of tuple
        Index tuples that select each subset from ``full_array``.  For PET
        sinograms these are returned by
        :meth:`.RegularPolygonPETLORDescriptor.get_distributed_views_and_slices`.
        All slices must select the same number of elements.
    path : str or Path
        Destination binary file (created or overwritten).
    dtype : numpy dtype, optional
        On-disk element type.  Default ``float32``.

    Returns
    -------
    SubsetArrayMmap
        Read-mode wrapper around the newly written file.

    Notes
    -----
    ``full_array`` must be a CPU NumPy array.  When working with a GPU or
    PyTorch backend, convert first with :func:`parallelproj.to_numpy_array`::

        from parallelproj import to_numpy_array
        from parallelproj.data import to_subset_mmap

        y_mmap = to_subset_mmap(to_numpy_array(y), subset_slices, "y.bin")
    """
    path = Path(path)
    num_subsets = len(subset_slices)
    subset_shape = tuple(full_array[subset_slices[0]].shape)

    mmap = np.memmap(
        path,
        dtype=np.dtype(dtype),
        mode="w+",
        shape=(num_subsets,) + subset_shape,
    )
    for k, sl in enumerate(subset_slices):
        mmap[k] = full_array[sl].astype(dtype)
    mmap.flush()
    del mmap  # closes the write handle and flushes OS buffers

    return SubsetArrayMmap(path, num_subsets, subset_shape, dtype=dtype, mode="r")


def count_event_multiplicity(events: Array) -> Array:
    """Count how many times each row appears in a 2-D event array.

    Parameters
    ----------
    events : Array
        2-D integer array of shape ``(N, M)`` where each row represents one
        event and the columns are event attributes (e.g. crystal indices).

    Returns
    -------
    Array
        1-D integer array of length ``N``.  Element ``i`` is the number of
        rows in *events* that are identical to row ``i``.

    Raises
    ------
    ValueError
        If *events* is not a 2-D array.
    """
    xp = array_api_compat.get_namespace(events)

    if events.ndim != 2:
        raise ValueError("events must be a 2D array")

    if array_api_compat.is_torch_namespace(xp):
        return _count_event_multiplicity_torch(events, xp)
    elif array_api_compat.is_cupy_namespace(xp):
        return _count_event_multiplicity_cupy(events, xp)
    else:
        return _count_event_multiplicity_numpy_fallback(events, xp)


def _count_event_multiplicity_torch(events: Array, xp) -> Array:
    torch_mod = _native_torch_module(xp)
    _, inverse, counts = torch_mod.unique(
        events,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse].reshape(-1)


def _count_event_multiplicity_cupy(events: Array, xp) -> Array:
    cupy_mod = xp
    _, inverse, counts = cupy_mod.unique(
        events,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse].reshape(-1)


def _count_event_multiplicity_numpy_fallback(events: Array, xp) -> Array:
    x_np = to_numpy_array(events)
    _, inverse, counts = np.unique(
        x_np,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return xp.asarray(counts[inverse].reshape(-1))


def _native_torch_module(xp):
    return xp if xp.__name__ == "torch" else xp.torch
