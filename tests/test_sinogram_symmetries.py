"""Tests for parallelproj.sinogram_symmetries — 100 % line coverage."""

from __future__ import annotations

import numpy as np
import pytest
from types import ModuleType

from parallelproj import to_numpy_array
from parallelproj.sinogram_symmetries import (
    is_interior_ring,
    axially_mirrored_plane,
    swapped_plane,
    axial_block_shifted_planes,
    plane_orbit,
    compute_sinogram_plane_symmetries,
    build_plane_class_indices,
    build_view_class_indices,
    build_radial_class_indices,
    build_bin_to_class,
    reduce_sinogram_by_symmetry_class,
    expand_sinogram_by_symmetry_class,
)

from .config import pytestmark  # noqa: F401  (applied as module-level mark)


# ── helpers ───────────────────────────────────────────────────────────────────


def _make_span1_table(num_rings: int, max_ring_diff: int | None = None) -> np.ndarray:
    """Build a simple span-1 plane_for_ring_pair_table."""
    if max_ring_diff is None:
        max_ring_diff = num_rings - 1
    table = np.full((num_rings, num_rings), -1, dtype=np.int64)
    plane_idx = 0
    for r1 in range(num_rings):
        for r2 in range(num_rings):
            if abs(r1 - r2) <= max_ring_diff:
                table[r1, r2] = plane_idx
                plane_idx += 1
    return table


# ── 1. is_interior_ring ───────────────────────────────────────────────────────


def test_is_interior_ring_no_edge(xp: ModuleType, dev: str) -> None:
    """n_edge=0 → every ring is interior (including edges)."""
    assert is_interior_ring(0, 10, 0) is True
    assert is_interior_ring(9, 10, 0) is True
    assert is_interior_ring(5, 10, 0) is True


def test_is_interior_ring_left_edge(xp: ModuleType, dev: str) -> None:
    """Ring 0 is an edge ring when n_edge=2."""
    assert is_interior_ring(0, 10, 2) is False


def test_is_interior_ring_interior(xp: ModuleType, dev: str) -> None:
    """Ring 3 is interior for N=10, n_edge=2."""
    assert is_interior_ring(3, 10, 2) is True


def test_is_interior_ring_right_edge(xp: ModuleType, dev: str) -> None:
    """Ring 9 is an edge ring for N=10, n_edge=2."""
    assert is_interior_ring(9, 10, 2) is False


def test_is_interior_ring_negative_n_edge(xp: ModuleType, dev: str) -> None:
    """Negative n_edge is treated the same as 0 (all interior)."""
    assert is_interior_ring(0, 10, -1) is True


# ── 2. axially_mirrored_plane ─────────────────────────────────────────────────


def test_axially_mirrored_plane_basic(xp: ModuleType, dev: str) -> None:
    """(0, 2) with 5 rings → (2, 4)."""
    assert axially_mirrored_plane(0, 2, 5) == (2, 4)


def test_axially_mirrored_plane_diagonal(xp: ModuleType, dev: str) -> None:
    """A diagonal plane (r, r) maps to the symmetric diagonal."""
    num_rings = 6
    r = 2
    result = axially_mirrored_plane(r, r, num_rings)
    assert result == (num_rings - 1 - r, num_rings - 1 - r)


def test_axially_mirrored_plane_order_reversed(xp: ModuleType, dev: str) -> None:
    """Mirror of (r1, r2) gives (num_rings-1-r2, num_rings-1-r1)."""
    num_rings = 8
    r1, r2 = 1, 5
    mr1, mr2 = axially_mirrored_plane(r1, r2, num_rings)
    assert mr1 == num_rings - 1 - r2
    assert mr2 == num_rings - 1 - r1


# ── 3. swapped_plane ──────────────────────────────────────────────────────────


def test_swapped_plane(xp: ModuleType, dev: str) -> None:
    """(1, 3) → (3, 1)."""
    assert swapped_plane(1, 3) == (3, 1)


def test_swapped_plane_diagonal(xp: ModuleType, dev: str) -> None:
    """(r, r) → (r, r) — no change for diagonal planes."""
    assert swapped_plane(4, 4) == (4, 4)


# ── 4. axial_block_shifted_planes ─────────────────────────────────────────────


def test_axial_block_shifted_no_edge(xp: ModuleType, dev: str) -> None:
    """n_edge=0: all valid shifts within bounds are returned."""
    block_size = 3
    num_rings = 9  # 3 blocks
    planes = axial_block_shifted_planes(0, 1, block_size, num_rings, n_edge=0)
    assert (0, 1) in planes
    assert (3, 4) in planes
    assert (6, 7) in planes
    for r1, r2 in planes:
        assert 0 <= r1 < num_rings
        assert 0 <= r2 < num_rings


def test_axial_block_shifted_edge_ring_only_k0(xp: ModuleType, dev: str) -> None:
    """Edge ring with n_edge=2: only k=0 returned (shift leaves edge category)."""
    block_size = 4
    num_rings = 8  # 2 blocks; edge rings: 0,1 and 6,7
    # ring 0 is edge; shifting by block_size=4 → ring 4 which is interior
    planes = axial_block_shifted_planes(0, 0, block_size, num_rings, n_edge=2)
    assert planes == [(0, 0)]


def test_axial_block_shifted_interior_ring_multiple_shifts(
    xp: ModuleType, dev: str
) -> None:
    """Interior ring with n_edge=1 can shift to other interior rings."""
    block_size = 3
    num_rings = 9
    n_edge = 1
    # interior rings: 1-7; edge: 0, 8
    planes = axial_block_shifted_planes(1, 2, block_size, num_rings, n_edge=n_edge)
    for r1, r2 in planes:
        assert (
            is_interior_ring(r1, num_rings, n_edge)
            == is_interior_ring(1, num_rings, n_edge)
        )
        assert (
            is_interior_ring(r2, num_rings, n_edge)
            == is_interior_ring(2, num_rings, n_edge)
        )
    assert len(planes) > 1


# ── 5. plane_orbit ────────────────────────────────────────────────────────────


def test_plane_orbit_contains_seed(xp: ModuleType, dev: str) -> None:
    """The orbit always contains the seed plane."""
    orbit = plane_orbit(1, 3, 2, 8, n_edge=0)
    assert (1, 3) in orbit


def test_plane_orbit_contains_mirror(xp: ModuleType, dev: str) -> None:
    """The orbit contains the axially mirrored plane."""
    block_size = 2
    num_rings = 8
    r1, r2 = 1, 3
    mirror = axially_mirrored_plane(r1, r2, num_rings)
    orbit = plane_orbit(r1, r2, block_size, num_rings, n_edge=0)
    assert mirror in orbit


def test_plane_orbit_contains_swapped(xp: ModuleType, dev: str) -> None:
    """The orbit contains the endpoint-swapped plane."""
    block_size = 2
    num_rings = 8
    r1, r2 = 1, 3
    orbit = plane_orbit(r1, r2, block_size, num_rings, n_edge=0)
    assert swapped_plane(r1, r2) in orbit


def test_plane_orbit_no_edge_larger_than_with_edge(xp: ModuleType, dev: str) -> None:
    """n_edge=0 orbit is at least as large as n_edge>0 orbit."""
    block_size = 3
    num_rings = 9
    n_edge = 1
    r1, r2 = 0, 0  # edge ring
    orbit_no_edge = plane_orbit(r1, r2, block_size, num_rings, n_edge=0)
    orbit_with_edge = plane_orbit(r1, r2, block_size, num_rings, n_edge=n_edge)
    assert len(orbit_no_edge) >= len(orbit_with_edge)


def test_plane_orbit_sorted(xp: ModuleType, dev: str) -> None:
    """plane_orbit returns a sorted list."""
    orbit = plane_orbit(2, 4, 2, 8, n_edge=0)
    assert orbit == sorted(orbit)


# ── 6. compute_sinogram_plane_symmetries ──────────────────────────────────────


def test_compute_plane_symmetries_all_planes_covered(xp: ModuleType, dev: str) -> None:
    """Every valid plane appears in exactly one class."""
    block_size = 2
    num_blocks = 3
    plane_to_class, _, _ = compute_sinogram_plane_symmetries(block_size, num_blocks)
    num_rings = block_size * num_blocks
    all_planes = {(r1, r2) for r1 in range(num_rings) for r2 in range(num_rings)}
    assert set(plane_to_class.keys()) == all_planes


def test_compute_plane_symmetries_consistency(xp: ModuleType, dev: str) -> None:
    """plane_to_class and class_to_planes are inverses of each other."""
    block_size = 2
    num_blocks = 3
    plane_to_class, class_to_planes, _ = compute_sinogram_plane_symmetries(
        block_size, num_blocks
    )
    for plane, cls in plane_to_class.items():
        assert plane in class_to_planes[cls]
    for cls, planes in class_to_planes.items():
        for plane in planes:
            assert plane_to_class[plane] == cls


def test_compute_plane_symmetries_num_classes(xp: ModuleType, dev: str) -> None:
    """num_classes equals the length of class_to_planes."""
    block_size = 2
    num_blocks = 3
    _, class_to_planes, num_classes = compute_sinogram_plane_symmetries(
        block_size, num_blocks
    )
    assert num_classes == len(class_to_planes)


def test_compute_plane_symmetries_max_ring_diff_none(xp: ModuleType, dev: str) -> None:
    """max_ring_diff=None uses num_rings-1 (same as explicit full range)."""
    block_size = 2
    num_blocks = 3
    num_rings = block_size * num_blocks
    p2c_none, _, _ = compute_sinogram_plane_symmetries(
        block_size, num_blocks, max_ring_diff=None
    )
    p2c_full, _, _ = compute_sinogram_plane_symmetries(
        block_size, num_blocks, max_ring_diff=num_rings - 1
    )
    assert set(p2c_none.keys()) == set(p2c_full.keys())


def test_compute_plane_symmetries_max_ring_diff_restricted(
    xp: ModuleType, dev: str
) -> None:
    """max_ring_diff=1 includes only diagonal and ±1 planes."""
    block_size = 2
    num_blocks = 3
    plane_to_class, _, _ = compute_sinogram_plane_symmetries(
        block_size, num_blocks, max_ring_diff=1
    )
    for r1, r2 in plane_to_class:
        assert abs(r1 - r2) <= 1


# ── 7. build_plane_class_indices ─────────────────────────────────────────────


def test_build_plane_class_indices_normal(xp: ModuleType, dev: str) -> None:
    """Normal span-1 table returns correct list of arrays."""
    block_size = 2
    num_blocks = 2
    num_rings = block_size * num_blocks
    _, class_to_planes, num_classes = compute_sinogram_plane_symmetries(
        block_size, num_blocks
    )
    table = _make_span1_table(num_rings)
    indices = build_plane_class_indices(table, class_to_planes, num_classes)

    assert len(indices) == num_classes
    all_plane_ids = np.concatenate(indices)
    assert len(all_plane_ids) == len(np.unique(all_plane_ids))
    assert np.all(all_plane_ids >= 0)


def test_build_plane_class_indices_duplicate_raises(xp: ModuleType, dev: str) -> None:
    """Table with duplicate plane indices (span > 1) raises ValueError."""
    num_rings = 4
    table = np.full((num_rings, num_rings), -1, dtype=np.int64)
    table[0, 0] = 0
    table[1, 1] = 0  # duplicate — simulates span > 1
    table[0, 1] = 1
    _, class_to_planes, num_classes = compute_sinogram_plane_symmetries(2, 2)
    with pytest.raises(ValueError):
        build_plane_class_indices(table, class_to_planes, num_classes)


def test_build_plane_class_indices_minus_one_omitted(xp: ModuleType, dev: str) -> None:
    """Ring pairs with table entry -1 are silently omitted."""
    block_size = 2
    num_blocks = 2
    num_rings = block_size * num_blocks
    max_ring_diff = 1
    _, class_to_planes, num_classes = compute_sinogram_plane_symmetries(
        block_size, num_blocks, max_ring_diff=max_ring_diff
    )
    table = _make_span1_table(num_rings, max_ring_diff=max_ring_diff)
    indices = build_plane_class_indices(table, class_to_planes, num_classes)

    for arr in indices:
        assert np.all(arr >= 0)


# ── 8. build_view_class_indices ───────────────────────────────────────────────


def test_build_view_class_indices_length(xp: ModuleType, dev: str) -> None:
    """Returns exactly view_period classes."""
    num_views = 20
    view_period = 4
    result = build_view_class_indices(num_views, view_period)
    assert len(result) == view_period


def test_build_view_class_indices_class_content(xp: ModuleType, dev: str) -> None:
    """Class c contains views c, c+period, c+2*period, …"""
    num_views = 12
    view_period = 3
    result = build_view_class_indices(num_views, view_period)
    for c in range(view_period):
        expected = np.arange(c, num_views, view_period, dtype=np.int64)
        np.testing.assert_array_equal(result[c], expected)


def test_build_view_class_indices_all_views_covered(xp: ModuleType, dev: str) -> None:
    """Union of all class arrays covers range(num_views) exactly once."""
    num_views = 16
    view_period = 4
    result = build_view_class_indices(num_views, view_period)
    all_views = np.concatenate(result)
    np.testing.assert_array_equal(
        np.sort(all_views), np.arange(num_views, dtype=np.int64)
    )


def test_build_view_class_indices_class_size(xp: ModuleType, dev: str) -> None:
    """Each class has num_views // view_period views."""
    num_views = 20
    view_period = 5
    result = build_view_class_indices(num_views, view_period)
    for arr in result:
        assert len(arr) == num_views // view_period


# ── 9. build_radial_class_indices ─────────────────────────────────────────────


def test_build_radial_class_indices_length(xp: ModuleType, dev: str) -> None:
    """Returns (num_rad+1)//2 classes."""
    num_rad = 7
    result = build_radial_class_indices(num_rad)
    assert len(result) == (num_rad + 1) // 2


def test_build_radial_class_indices_last_is_singleton(xp: ModuleType, dev: str) -> None:
    """Last class is a singleton (the centre bin) for odd num_rad."""
    num_rad = 9
    result = build_radial_class_indices(num_rad)
    centre_bin = (num_rad - 1) // 2
    assert len(result[-1]) == 1
    assert result[-1][0] == centre_bin


def test_build_radial_class_indices_pair_symmetry(xp: ModuleType, dev: str) -> None:
    """For each non-centre class, the two indices sum to num_rad - 1."""
    num_rad = 11
    result = build_radial_class_indices(num_rad)
    for arr in result[:-1]:  # exclude centre singleton
        assert len(arr) == 2
        assert arr[0] + arr[1] == num_rad - 1


def test_build_radial_class_indices_singleton(xp: ModuleType, dev: str) -> None:
    """num_rad=1 → single singleton class."""
    result = build_radial_class_indices(1)
    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0] == 0


def test_build_radial_class_indices_even_raises(xp: ModuleType, dev: str) -> None:
    """Even num_rad raises ValueError."""
    with pytest.raises(ValueError):
        build_radial_class_indices(8)


# ── 10. build_bin_to_class ────────────────────────────────────────────────────


def test_build_bin_to_class_correct_mapping(xp: ModuleType, dev: str) -> None:
    """bin_to_class[idx[j]] == c for all c, j."""
    num_views = 12
    view_period = 3
    class_indices = build_view_class_indices(num_views, view_period)
    bin_to_class = build_bin_to_class(class_indices, num_views)

    for c, indices in enumerate(class_indices):
        for idx in indices:
            assert bin_to_class[idx] == c


def test_build_bin_to_class_roundtrip(xp: ModuleType, dev: str) -> None:
    """For each class c, all bins where bin_to_class==c appear in class_indices[c]."""
    num_rad = 9
    class_indices = build_radial_class_indices(num_rad)
    bin_to_class = build_bin_to_class(class_indices, num_rad)

    for c, indices in enumerate(class_indices):
        recovered = np.where(bin_to_class == c)[0]
        np.testing.assert_array_equal(np.sort(recovered), np.sort(indices))


def test_build_bin_to_class_shape(xp: ModuleType, dev: str) -> None:
    """Output has shape (num_bins,) and dtype int64."""
    class_indices = build_view_class_indices(16, 4)
    result = build_bin_to_class(class_indices, 16)
    assert result.shape == (16,)
    assert result.dtype == np.int64


# ── 11. reduce_sinogram_by_symmetry_class ─────────────────────────────────────


def test_reduce_sinogram_view_axis_sum(xp: ModuleType, dev: str) -> None:
    """Reduce on axis=1 (view) with sum: output is class_size * (class+1)."""
    num_rad = 5
    num_views = 12
    num_planes = 3
    view_period = 3
    class_indices = build_view_class_indices(num_views, view_period)

    data = np.zeros((num_rad, num_views, num_planes), dtype=np.float32)
    for c, idx in enumerate(class_indices):
        data[:, idx, :] = float(c + 1)

    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=1)

    assert tuple(reduced.shape) == (num_rad, view_period, num_planes)
    class_size = num_views // view_period
    for c in range(view_period):
        expected = float(class_size * (c + 1))
        slice_np = to_numpy_array(reduced[:, c, :])
        np.testing.assert_allclose(slice_np.flatten(), expected, rtol=1e-5)


def test_reduce_sinogram_view_axis_mean(xp: ModuleType, dev: str) -> None:
    """Reduce with mean: output equals class+1 for uniform-within-class sinogram."""
    num_rad = 4
    num_views = 8
    num_planes = 2
    view_period = 4
    class_indices = build_view_class_indices(num_views, view_period)

    data = np.zeros((num_rad, num_views, num_planes), dtype=np.float64)
    for c, idx in enumerate(class_indices):
        data[:, idx, :] = float(c + 1)

    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=1, reduction=xp.mean)

    assert tuple(reduced.shape) == (num_rad, view_period, num_planes)
    for c in range(view_period):
        expected = float(c + 1)
        slice_np = to_numpy_array(reduced[:, c, :])
        np.testing.assert_allclose(slice_np.flatten(), expected, rtol=1e-5)


def test_reduce_sinogram_axis0(xp: ModuleType, dev: str) -> None:
    """Reduce on axis=0 (radial) produces correct output shape."""
    num_rad = 9
    num_views = 4
    num_planes = 3
    class_indices = build_radial_class_indices(num_rad)
    num_classes = len(class_indices)

    data = np.ones((num_rad, num_views, num_planes), dtype=np.float32)
    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=0)

    assert tuple(reduced.shape) == (num_classes, num_views, num_planes)


def test_reduce_sinogram_axis2(xp: ModuleType, dev: str) -> None:
    """Reduce on axis=2 (plane) produces correct output shape."""
    num_rad = 3
    num_views = 4
    num_planes = 12
    view_period = 4
    class_indices = build_view_class_indices(num_planes, view_period)
    num_classes = len(class_indices)

    data = np.ones((num_rad, num_views, num_planes), dtype=np.float32)
    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=2)

    assert tuple(reduced.shape) == (num_rad, num_views, num_classes)


def test_reduce_sinogram_default_reduction_is_sum(xp: ModuleType, dev: str) -> None:
    """Default reduction (None) behaves identically to explicit xp.sum."""
    num_rad = 3
    num_views = 6
    num_planes = 2
    view_period = 3
    class_indices = build_view_class_indices(num_views, view_period)

    data = np.ones((num_rad, num_views, num_planes), dtype=np.float32)
    sino = xp.asarray(data, device=dev)

    reduced_default = reduce_sinogram_by_symmetry_class(
        sino, class_indices, axis=1, reduction=None
    )
    reduced_sum = reduce_sinogram_by_symmetry_class(
        sino, class_indices, axis=1, reduction=xp.sum
    )

    np.testing.assert_allclose(
        to_numpy_array(reduced_default), to_numpy_array(reduced_sum), rtol=1e-6
    )


# ── 12. expand_sinogram_by_symmetry_class ────────────────────────────────────


def test_expand_sinogram_roundtrip_mean(xp: ModuleType, dev: str) -> None:
    """Round-trip reduce(mean) → expand gives back original values."""
    num_rad = 5
    num_views = 12
    num_planes = 3
    view_period = 3
    class_indices = build_view_class_indices(num_views, view_period)

    data = np.zeros((num_rad, num_views, num_planes), dtype=np.float64)
    for c, idx in enumerate(class_indices):
        data[:, idx, :] = float(c + 1)

    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=1, reduction=xp.mean)
    expanded = expand_sinogram_by_symmetry_class(reduced, class_indices, num_views, axis=1)

    assert tuple(expanded.shape) == tuple(sino.shape)
    np.testing.assert_allclose(to_numpy_array(expanded), data, rtol=1e-5)


def test_expand_sinogram_axis0(xp: ModuleType, dev: str) -> None:
    """Expand on axis=0 restores original shape."""
    num_rad = 9
    num_views = 4
    num_planes = 3
    class_indices = build_radial_class_indices(num_rad)

    data = np.ones((num_rad, num_views, num_planes), dtype=np.float32)
    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=0)
    expanded = expand_sinogram_by_symmetry_class(reduced, class_indices, num_rad, axis=0)

    assert tuple(expanded.shape) == (num_rad, num_views, num_planes)


def test_expand_sinogram_axis1(xp: ModuleType, dev: str) -> None:
    """Expand on axis=1 restores original shape."""
    num_rad = 3
    num_views = 8
    num_planes = 2
    view_period = 4
    class_indices = build_view_class_indices(num_views, view_period)

    data = np.ones((num_rad, num_views, num_planes), dtype=np.float32)
    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=1)
    expanded = expand_sinogram_by_symmetry_class(reduced, class_indices, num_views, axis=1)

    assert tuple(expanded.shape) == (num_rad, num_views, num_planes)


def test_expand_sinogram_axis2(xp: ModuleType, dev: str) -> None:
    """Expand on axis=2 restores original shape."""
    num_rad = 3
    num_views = 4
    num_planes = 9
    class_indices = build_radial_class_indices(num_planes)

    data = np.ones((num_rad, num_views, num_planes), dtype=np.float32)
    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=2)
    expanded = expand_sinogram_by_symmetry_class(reduced, class_indices, num_planes, axis=2)

    assert tuple(expanded.shape) == (num_rad, num_views, num_planes)


def test_expand_sinogram_sum_each_bin_gets_class_sum(xp: ModuleType, dev: str) -> None:
    """After reduce(sum) + expand, each bin contains the class sum."""
    num_rad = 3
    num_views = 6
    num_planes = 2
    view_period = 3
    class_indices = build_view_class_indices(num_views, view_period)
    class_size = num_views // view_period  # 2

    data = np.zeros((num_rad, num_views, num_planes), dtype=np.float64)
    for c, idx in enumerate(class_indices):
        data[:, idx, :] = float(c + 1)

    sino = xp.asarray(data, device=dev)
    reduced = reduce_sinogram_by_symmetry_class(sino, class_indices, axis=1)
    expanded = expand_sinogram_by_symmetry_class(reduced, class_indices, num_views, axis=1)

    expected = np.zeros_like(data)
    for c, idx in enumerate(class_indices):
        expected[:, idx, :] = float(class_size * (c + 1))

    np.testing.assert_allclose(to_numpy_array(expanded), expected, rtol=1e-5)
