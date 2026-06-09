"""Tests for parallelproj.data — disk-backed subset array utilities."""
from __future__ import annotations

import math

import pytest

# parallelproj._backend loads numpy's C extension fully; this import must
# come before any bare `import numpy` to avoid a double-init error when the
# test file is collected in isolation.
from parallelproj.data import SubsetArrayMmap, to_subset_mmap

import numpy as np


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(name="arr")
def fixture_arr() -> np.ndarray:
    """Small float32 array with shape (views=6, radial=4, planes=3)."""
    return np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)


@pytest.fixture(name="slices")
def fixture_slices() -> list[tuple]:
    """Two subsets: even views [0,2,4] and odd views [1,3,5]."""
    return [
        (slice(0, None, 2), slice(None), slice(None)),
        (slice(1, None, 2), slice(None), slice(None)),
    ]


# ---------------------------------------------------------------------------
# to_subset_mmap
# ---------------------------------------------------------------------------


def test_to_subset_mmap_creates_file(arr, slices, tmp_path):
    """to_subset_mmap writes a binary file at the given path."""
    path = tmp_path / "y.bin"
    to_subset_mmap(arr, slices, path)
    assert path.exists()


def test_to_subset_mmap_returns_subset_array_mmap(arr, slices, tmp_path):
    """to_subset_mmap returns a SubsetArrayMmap instance."""
    result = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    assert isinstance(result, SubsetArrayMmap)


def test_to_subset_mmap_shape(arr, slices, tmp_path):
    """shape is (num_subsets, *subset_shape)."""
    num_subsets = len(slices)
    subset_shape = arr[slices[0]].shape
    mmap = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    assert mmap.shape == (num_subsets,) + subset_shape


def test_to_subset_mmap_roundtrip(arr, slices, tmp_path):
    """Every subset read back from disk matches the original slice."""
    mmap = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    for k, sl in enumerate(slices):
        np.testing.assert_array_equal(mmap[k], arr[sl])


def test_to_subset_mmap_dtype_cast(arr, slices, tmp_path):
    """Input array is cast to the requested on-disk dtype."""
    arr_f64 = arr.astype(np.float64)
    mmap = to_subset_mmap(arr_f64, slices, tmp_path / "y.bin", dtype=np.float32)
    assert mmap[0].dtype == np.float32


def test_to_subset_mmap_str_path(arr, slices, tmp_path):
    """Path can be supplied as a plain string."""
    mmap = to_subset_mmap(arr, slices, str(tmp_path / "y.bin"))
    assert mmap.path == (tmp_path / "y.bin")


# ---------------------------------------------------------------------------
# SubsetArrayMmap — sequence interface
# ---------------------------------------------------------------------------


def test_len(arr, slices, tmp_path):
    """len() returns num_subsets."""
    mmap = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    assert len(mmap) == len(slices)


def test_getitem_returns_owned_ndarray(arr, slices, tmp_path):
    """mmap[k] returns a plain ndarray (not a memmap subclass) that owns its buffer."""
    mmap = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    item = mmap[0]
    assert isinstance(item, np.ndarray)
    assert not isinstance(item, np.memmap)
    assert item.base is None


# ---------------------------------------------------------------------------
# SubsetArrayMmap — properties
# ---------------------------------------------------------------------------


def test_properties(arr, slices, tmp_path):
    """num_subsets, subset_shape, shape, and path properties return correct values."""
    path = tmp_path / "y.bin"
    num_subsets = len(slices)
    subset_shape = arr[slices[0]].shape
    mmap = to_subset_mmap(arr, slices, path)
    assert mmap.num_subsets == num_subsets
    assert mmap.subset_shape == subset_shape
    assert mmap.shape == (num_subsets,) + subset_shape
    assert mmap.path == path


# ---------------------------------------------------------------------------
# SubsetArrayMmap — byte-count helpers
# ---------------------------------------------------------------------------


def test_nbytes_per_subset(arr, slices, tmp_path):
    """nbytes_per_subset returns element count times itemsize."""
    mmap = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    expected = math.prod(mmap.subset_shape) * np.dtype(np.float32).itemsize
    assert mmap.nbytes_per_subset() == expected


def test_nbytes_total(arr, slices, tmp_path):
    """nbytes_total equals num_subsets times nbytes_per_subset."""
    mmap = to_subset_mmap(arr, slices, tmp_path / "y.bin")
    assert mmap.nbytes_total() == len(mmap) * mmap.nbytes_per_subset()
