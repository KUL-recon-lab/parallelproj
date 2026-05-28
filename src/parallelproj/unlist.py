"""Listmode-to-sinogram histogrammer for RegularPolygonPETScannerGeometry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ._backend import to_numpy_array
from .pet_lors import RegularPolygonPETLORDescriptor

if TYPE_CHECKING:
    pass


def _build_inring_luts(
    lor_descriptor: RegularPolygonPETLORDescriptor,
) -> tuple[np.ndarray, np.ndarray]:
    """Build in-ring lookup tables from a LOR descriptor.

    Parameters
    ----------
    lor_descriptor : RegularPolygonPETLORDescriptor

    Returns
    -------
    inring_lut : np.ndarray, shape (n, n), dtype int32
        ``inring_lut[d1, d2]`` is the flat ``view * num_rad + rad`` sinogram
        index for the crystal pair, or ``-1`` if the pair is not a valid LOR.
    inring_tof_sign : np.ndarray, shape (n, n), dtype int8
        ``inring_tof_sign[d1, d2]`` is ``+1`` if d1 is the canonical *start*
        crystal (the xstart endpoint used by the sinogram convention) and
        ``-1`` if d1 is the end crystal.  Zero on the diagonal.
    """
    n = lor_descriptor.scanner.num_lor_endpoints_per_ring
    num_rad = lor_descriptor.num_rad
    num_views = lor_descriptor.num_views

    inring_lut = np.full((n, n), -1, dtype=np.int32)
    inring_tof_sign = np.zeros((n, n), dtype=np.int8)

    # start/end in-ring crystal indices: shape (num_views, num_rad)
    ds = to_numpy_array(lor_descriptor.start_in_ring_index).ravel().astype(np.intp)
    de = to_numpy_array(lor_descriptor.end_in_ring_index).ravel().astype(np.intp)

    v_idx = np.repeat(np.arange(num_views, dtype=np.intp), num_rad)
    r_idx = np.tile(np.arange(num_rad, dtype=np.intp), num_views)
    flat_vr = (v_idx * num_rad + r_idx).astype(np.int32)

    inring_lut[ds, de] = flat_vr
    inring_lut[de, ds] = flat_vr
    inring_tof_sign[ds, de] = 1
    inring_tof_sign[de, ds] = -1

    # Self-pairs (start == end crystal) are mathematical artifacts of the
    # zig-zag sinogram parameterisation; no physical coincidence can produce
    # them.  Invalidate explicitly so they are dropped by the validity check.
    np.fill_diagonal(inring_lut, -1)
    np.fill_diagonal(inring_tof_sign, 0)

    return inring_lut, inring_tof_sign


def regular_polygon_events_to_sinogram(
    lor_descriptor: RegularPolygonPETLORDescriptor,
    events: Any,
    num_tof_bins: int | None = None,
    tof_bin_sign: int = 1,
) -> np.ndarray:
    """Histogram listmode events into a sinogram.

    Parameters
    ----------
    lor_descriptor : RegularPolygonPETLORDescriptor
        LOR descriptor defining the sinogram geometry.
    events : array-like, shape (N, 4) or (N, 5)
        Listmode events.  Each row is ``(d1, r1, d2, r2)`` for non-TOF or
        ``(d1, r1, d2, r2, tof_bin)`` for TOF, where

        - ``d1``, ``d2`` are in-ring crystal indices
          (0 … num_lor_endpoints_per_ring - 1)
        - ``r1``, ``r2`` are ring indices (0 … num_rings - 1)
        - ``tof_bin`` is an **unsigned** TOF bin index with the convention
          set by ``tof_bin_sign``

        Events outside the sinogram FOV (invalid crystal pair, ring pair
        beyond ``max_ring_difference``, out-of-range indices) are silently
        discarded.
    num_tof_bins : int or None
        Number of TOF bins.  Required when ``events`` has 5 columns; must
        be ``None`` when ``events`` has 4 columns.
    tof_bin_sign : {+1, -1}, optional
        TOF bin direction convention in the input events:

        * ``+1`` (default): bin 0 is closest to detector d1.  This matches
          the parallelproj sinogram convention (bin 0 = closest to xstart)
          when d1 is the start crystal.
        * ``-1``: bin 0 is closest to detector d2.

        A flip is applied automatically so that the output sinogram always
        uses the parallelproj convention (bin 0 = closest to xstart).

    Returns
    -------
    sinogram : np.ndarray
        Histogram sinogram.  Shape is ``spatial_sinogram_shape`` for non-TOF
        or ``(*spatial_sinogram_shape, num_tof_bins)`` for TOF.
        Dtype is ``int32``.
    """
    if tof_bin_sign not in (1, -1):
        raise ValueError("tof_bin_sign must be +1 or -1")

    events_np = np.asarray(to_numpy_array(events), dtype=np.int32)

    if events_np.ndim != 2:
        raise ValueError("events must be a 2D array")

    n_events, n_cols = events_np.shape

    if n_cols == 4:
        if num_tof_bins is not None:
            raise ValueError(
                "events has 4 columns (non-TOF) but num_tof_bins was specified"
            )
        tof_mode = False
    elif n_cols == 5:
        if num_tof_bins is None:
            raise ValueError(
                "events has 5 columns (TOF) but num_tof_bins was not specified"
            )
        tof_mode = True
    else:
        raise ValueError(
            f"events must have 4 (non-TOF) or 5 (TOF) columns, got {n_cols}"
        )

    shape_spatial = lor_descriptor.spatial_sinogram_shape

    if n_events == 0:
        if tof_mode:
            return np.zeros((*shape_spatial, num_tof_bins), dtype=np.int32)
        return np.zeros(shape_spatial, dtype=np.int32)

    d1 = events_np[:, 0]
    r1 = events_np[:, 1]
    d2 = events_np[:, 2]
    r2 = events_np[:, 3]
    if tof_mode:
        tof_raw = events_np[:, 4]

    inring_lut, inring_tof_sign_lut = _build_inring_luts(lor_descriptor)
    ring_pair_table = lor_descriptor.michelogram.plane_for_ring_pair_table

    n_crystals = lor_descriptor.scanner.num_lor_endpoints_per_ring
    num_rings = lor_descriptor.scanner.num_rings
    num_rad = lor_descriptor.num_rad

    # primary validity: all indices within bounds
    valid = (
        (d1 >= 0)
        & (d1 < n_crystals)
        & (d2 >= 0)
        & (d2 < n_crystals)
        & (r1 >= 0)
        & (r1 < num_rings)
        & (r2 >= 0)
        & (r2 < num_rings)
    )

    # safe fallback indices for out-of-bounds events (they will be masked out)
    d1s = np.where(valid, d1, 0)
    d2s = np.where(valid, d2, 0)

    inring_flat = inring_lut[d1s, d2s]
    tof_sign_vals = inring_tof_sign_lut[d1s, d2s]

    # the crystal pair must form a valid LOR
    valid &= inring_flat >= 0

    # canonical ring ordering: +1 means d1 is xstart, so r1 is the start ring.
    # Use safe fallback (0) for out-of-range ring indices so the table lookup
    # never receives negative or OOB values (numpy wraps negative indices).
    r1_safe = np.where(valid, r1, 0)
    r2_safe = np.where(valid, r2, 0)
    is_d1_start = tof_sign_vals == 1
    r_start = np.where(is_d1_start, r1_safe, r2_safe)
    r_end = np.where(is_d1_start, r2_safe, r1_safe)

    # plane lookup
    plane_idx = ring_pair_table[r_start, r_end]
    valid &= plane_idx >= 0

    if tof_mode:
        valid &= (tof_raw >= 0) & (tof_raw < num_tof_bins)
        tof_raw_safe = np.where(valid, tof_raw, 0)
        # flip the bin when the canonical direction does not match tof_bin_sign
        flip = (tof_bin_sign * tof_sign_vals) == -1
        sinogram_tof = np.where(flip, num_tof_bins - 1 - tof_raw_safe, tof_raw_safe)

    # decompose the in-ring flat index into view and radial indices
    safe_flat = np.where(valid, inring_flat, 0)
    view_idx = safe_flat // num_rad
    rad_idx = safe_flat % num_rad

    # compute the flat sinogram index respecting sinogram_order
    p_ax = lor_descriptor.plane_axis_num
    v_ax = lor_descriptor.view_axis_num
    r_ax = lor_descriptor.radial_axis_num
    strides = [int(np.prod(shape_spatial[i + 1 :])) for i in range(3)]

    safe_plane = np.where(valid, plane_idx, 0)
    flat_sino = (
        rad_idx.astype(np.int64) * strides[r_ax]
        + view_idx.astype(np.int64) * strides[v_ax]
        + safe_plane.astype(np.int64) * strides[p_ax]
    )

    if tof_mode:
        flat_sino = flat_sino * num_tof_bins + sinogram_tof.astype(np.int64)
        total_bins = int(np.prod(shape_spatial)) * num_tof_bins
        output_shape = (*shape_spatial, num_tof_bins)
    else:
        total_bins = int(np.prod(shape_spatial))
        output_shape = shape_spatial

    flat_sino_valid = flat_sino[valid]
    sino_flat = np.bincount(flat_sino_valid, minlength=total_bins).astype(np.int32)
    return sino_flat.reshape(output_shape)
