"""Listmode-to-sinogram histogramming for regular-polygon PET scanners.

Converts per-event crystal and ring indices into a binned sinogram array,
for both non-TOF and TOF acquisitions.  A companion function converts raw
detection-time differences (in nanoseconds) to projector-convention TOF bin
indices ready for histogramming.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import array_api_compat

from ._backend import Array, to_numpy_array
from .pet_lors import RegularPolygonPETLORDescriptor
from .tof import C_MM_PER_NS

if TYPE_CHECKING:
    from .projectors import RegularPolygonPETProjector


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
        ``inring_lut[d_red, d_blue]`` is the flat ``view * num_rad + rad``
        sinogram index for the crystal pair, or ``-1`` if the pair is not a
        valid LOR.
    inring_tof_sign : np.ndarray, shape (n, n), dtype int8
        ``inring_tof_sign[d_red, d_blue]`` is ``+1`` if ``d_red`` is the
        canonical *xstart* crystal and ``-1`` if ``d_red`` is the *xend*
        crystal.  Zero on the diagonal.
    """
    n = lor_descriptor.scanner.num_lor_endpoints_per_ring
    num_rad = lor_descriptor.num_rad
    num_views = lor_descriptor.num_views

    inring_lut = np.full((n, n), -1, dtype=np.int32)
    inring_tof_sign = np.zeros((n, n), dtype=np.int8)

    # xstart / xend in-ring crystal indices: shape (num_views, num_rad)
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
    projector: "RegularPolygonPETProjector",
    d_red: Any,
    r_red: Any,
    d_blue: Any,
    r_blue: Any,
    unsigned_sinogram_tof_bin: Any | None = None,
) -> Array:
    """Histogram listmode events into a sinogram.

    Parameters
    ----------
    projector : RegularPolygonPETProjector
        The PET projector whose LOR descriptor defines the sinogram geometry.
        For TOF mode (``unsigned_sinogram_tof_bin`` provided),
        ``projector.tof_parameters`` must be set and its ``num_tofbins``
        determines the TOF axis size.
    d_red : array-like, shape (N,), dtype int32
        In-ring crystal indices for the **red** detector
        (0 ... num_lor_endpoints_per_ring - 1).
    r_red : array-like, shape (N,), dtype int32
        Ring indices for the **red** detector (0 ... num_rings - 1).
    d_blue : array-like, shape (N,), dtype int32
        In-ring crystal indices for the **blue** detector.
    r_blue : array-like, shape (N,), dtype int32
        Ring indices for the **blue** detector.
    unsigned_sinogram_tof_bin : array-like, shape (N,), dtype int32, or None
        Unsigned TOF bin numbers in the **projector convention**
        (bin 0 = closest to xstart).  Use :func:`detection_times_to_tof_bin`
        to convert raw detection-time differences to this convention.
        Pass ``None`` for non-TOF histogramming.

        Events outside the sinogram FOV (invalid crystal pair, ring pair
        beyond ``max_ring_difference``, out-of-range indices, or negative
        ``unsigned_sinogram_tof_bin`` values) are silently discarded.

    Returns
    -------
    sinogram : Array
        Histogram sinogram on the same device as the input arrays.
        Shape is ``spatial_sinogram_shape`` for non-TOF or
        ``(*spatial_sinogram_shape, num_tof_bins)`` for TOF.
        Dtype is ``int32``.

    Raises
    ------
    NotImplementedError
        If the array backend does not provide ``bincount``
        (e.g. ``array_api_strict``).  Supported backends: numpy, cupy, torch.
    ValueError
        If ``unsigned_sinogram_tof_bin`` is provided but
        ``projector.tof_parameters`` is ``None``.
    """
    xp = array_api_compat.get_namespace(d_red)
    dev = array_api_compat.device(d_red)

    if not hasattr(xp, "bincount"):
        raise NotImplementedError(
            "regular_polygon_events_to_sinogram requires a backend with "
            "bincount support (numpy, cupy, torch); "
            f"got {xp.__name__!r}"
        )

    lor_descriptor = projector.lor_descriptor

    tof_mode = unsigned_sinogram_tof_bin is not None
    if tof_mode:
        if projector.tof_parameters is None:
            raise ValueError(
                "unsigned_sinogram_tof_bin provided but projector.tof_parameters is None"
            )
        num_tof_bins = projector.tof_parameters.num_tofbins

    d_red_xp = xp.asarray(d_red, dtype=xp.int32, device=dev)
    r_red_xp = xp.asarray(r_red, dtype=xp.int32, device=dev)
    d_blue_xp = xp.asarray(d_blue, dtype=xp.int32, device=dev)
    r_blue_xp = xp.asarray(r_blue, dtype=xp.int32, device=dev)

    n_events = d_red_xp.shape[0]

    if tof_mode:
        tof_bin_xp = xp.asarray(unsigned_sinogram_tof_bin, dtype=xp.int32, device=dev)

    shape_spatial = lor_descriptor.spatial_sinogram_shape

    if n_events == 0:
        if tof_mode:
            return xp.zeros((*shape_spatial, num_tof_bins), dtype=xp.int32, device=dev)
        return xp.zeros(shape_spatial, dtype=xp.int32, device=dev)

    inring_lut_np, inring_tof_sign_lut_np = _build_inring_luts(lor_descriptor)
    inring_lut = xp.asarray(inring_lut_np, device=dev)
    inring_tof_sign_lut = xp.asarray(inring_tof_sign_lut_np, device=dev)
    ring_pair_table = xp.asarray(
        lor_descriptor.michelogram.plane_for_ring_pair_table, device=dev
    )

    n_crystals = lor_descriptor.scanner.num_lor_endpoints_per_ring
    num_rings = lor_descriptor.scanner.num_rings
    num_rad = lor_descriptor.num_rad

    zero = xp.zeros(1, dtype=xp.int32, device=dev)

    # primary validity: all indices within bounds
    valid = (
        (d_red_xp >= 0)
        & (d_red_xp < n_crystals)
        & (d_blue_xp >= 0)
        & (d_blue_xp < n_crystals)
        & (r_red_xp >= 0)
        & (r_red_xp < num_rings)
        & (r_blue_xp >= 0)
        & (r_blue_xp < num_rings)
    )

    d_red_s = xp.where(valid, d_red_xp, zero)
    d_blue_s = xp.where(valid, d_blue_xp, zero)

    inring_flat = inring_lut[d_red_s, d_blue_s]
    tof_sign_vals = inring_tof_sign_lut[d_red_s, d_blue_s]

    valid = valid & (inring_flat >= 0)

    # Canonical ring ordering from the sinogram definition.
    # tof_sign_vals = +1  ->  d_red is xstart  ->  r_red  is r_start
    # tof_sign_vals = -1  ->  d_blue is xstart ->  r_blue is r_start
    r_red_s = xp.where(valid, r_red_xp, zero)
    r_blue_s = xp.where(valid, r_blue_xp, zero)
    is_red_start = tof_sign_vals == 1
    r_start = xp.where(is_red_start, r_red_s, r_blue_s)
    r_end = xp.where(is_red_start, r_blue_s, r_red_s)

    plane_idx = ring_pair_table[r_start, r_end]
    valid = valid & (plane_idx >= 0)

    if tof_mode:
        # unsigned_sinogram_tof_bin is already in the projector convention
        # (bin 0 = closest to xstart) -- no flip required.  Negative values
        # signal invalid events (e.g. detection_times_to_tof_bin returning -1).
        valid = valid & (tof_bin_xp >= 0) & (tof_bin_xp < num_tof_bins)
        tof_bin_safe = xp.where(valid, tof_bin_xp, zero)

    # decompose the in-ring flat index into view and radial indices
    safe_flat = xp.where(valid, inring_flat, zero)
    view_idx = safe_flat // num_rad
    rad_idx = safe_flat % num_rad

    # compute the flat sinogram index respecting sinogram_order
    p_ax = lor_descriptor.plane_axis_num
    v_ax = lor_descriptor.view_axis_num
    r_ax = lor_descriptor.radial_axis_num
    strides = [int(np.prod(shape_spatial[i + 1 :])) for i in range(3)]

    safe_plane = xp.where(valid, plane_idx, zero)
    flat_sino = (
        xp.astype(rad_idx, xp.int64) * strides[r_ax]
        + xp.astype(view_idx, xp.int64) * strides[v_ax]
        + xp.astype(safe_plane, xp.int64) * strides[p_ax]
    )

    if tof_mode:
        flat_sino = flat_sino * num_tof_bins + xp.astype(tof_bin_safe, xp.int64)
        total_bins = int(np.prod(shape_spatial)) * num_tof_bins
        output_shape = (*shape_spatial, num_tof_bins)
    else:
        total_bins = int(np.prod(shape_spatial))
        output_shape = shape_spatial

    flat_sino_valid = flat_sino[valid]
    if flat_sino_valid.shape[0] == 0:
        return xp.zeros(output_shape, dtype=xp.int32, device=dev)
    sino_flat = xp.astype(xp.bincount(flat_sino_valid, minlength=total_bins), xp.int32)
    return xp.reshape(sino_flat, output_shape)


def detection_times_to_tof_bin(
    d_red: Any,
    d_blue: Any,
    dt_blue_minus_red: Any,
    projector: "RegularPolygonPETProjector",
) -> Array:
    """Convert raw detection-time differences to projector-convention TOF bins.

    Each coincidence event is characterised by two crystal hits and the
    **signed arrival-time difference** ``t_blue - t_red`` (in nanoseconds).
    This function maps that physical timing to the **unsigned integer TOF bin**
    used by :func:`regular_polygon_events_to_sinogram`, taking into account
    whether the projector's canonical ray direction runs from *red* to *blue*
    or vice versa.

    Parameters
    ----------
    d_red : array-like, shape (N,), dtype int32
        In-ring crystal indices for the **red** detector.
    d_blue : array-like, shape (N,), dtype int32
        In-ring crystal indices for the **blue** detector.
    dt_blue_minus_red : array-like, shape (N,), dtype float
        ``t_blue - t_red`` in **nanoseconds**.
        Positive = blue photon arrived later = emission closer to the red side.
        Internally cast to ``float32``; values outside the ``float32`` range
        will silently lose precision.
    projector : RegularPolygonPETProjector
        TOF projector that defines the bin grid.  Must have
        ``tof_parameters`` set.

    Returns
    -------
    tof_bin : Array, shape (N,), dtype int32
        Unsigned TOF bin numbers in the projector convention
        (bin 0 = closest to xstart).
        Returns ``-1`` for events whose emission falls outside the sinogram's
        TOF window, or for invalid crystal pairs (self-pairs / out-of-bounds).
        These ``-1`` values are silently discarded by
        :func:`regular_polygon_events_to_sinogram`.

    Raises
    ------
    ValueError
        If ``projector.tof_parameters`` is ``None``.

    Notes
    -----
    The signed spatial off-centre displacement (positive toward the red
    detector) is

    .. math::

        \\Delta x_{\\text{blue->red}} = \\frac{c}{2}\\,(t_{\\text{blue}} - t_{\\text{red}})

    where :math:`c` = :data:`C_MM_PER_NS` mm/ns.  Letting
    :math:`s = \\text{sign}[d_{\\text{red}}, d_{\\text{blue}}]`
    (``+1`` if ``d_red`` is the canonical xstart, ``-1`` otherwise) and
    :math:`W` = ``tofbin_width``, the bin index is

    .. math::

        k = \\operatorname{round}\\!\\left(
              \\frac{N-1}{2}
              - \\frac{s\\,\\Delta x_{\\text{blue->red}}
                       + \\Delta_{\\text{center}}}{W}
            \\right)

    where :math:`\\Delta_{\\text{center}}` = ``tofcenter_offset``.
    """
    if projector.tof_parameters is None:
        raise ValueError("projector.tof_parameters is None")

    num_tof_bins = projector.tof_parameters.num_tofbins
    tofbin_width = projector.tof_parameters.tofbin_width
    tofcenter_offset = projector.tof_parameters.tofcenter_offset

    lor_desc = projector.lor_descriptor
    lut_np, sign_lut_np = _build_inring_luts(lor_desc)

    xp = array_api_compat.get_namespace(dt_blue_minus_red)
    dev = array_api_compat.device(dt_blue_minus_red)

    lut = xp.asarray(lut_np, device=dev)
    sign_lut = xp.asarray(sign_lut_np, device=dev)

    n_crystals = lor_desc.scanner.num_lor_endpoints_per_ring

    d_red_xp = xp.asarray(d_red, dtype=xp.int32, device=dev)
    d_blue_xp = xp.asarray(d_blue, dtype=xp.int32, device=dev)
    dt_xp = xp.asarray(dt_blue_minus_red, dtype=xp.float32, device=dev)

    zero = xp.zeros(1, dtype=xp.int32, device=dev)

    # bounds check + valid-LOR check (catches self-pairs via lut[d,d] == -1)
    in_bounds = (
        (d_red_xp >= 0)
        & (d_red_xp < n_crystals)
        & (d_blue_xp >= 0)
        & (d_blue_xp < n_crystals)
    )
    d_red_s = xp.where(in_bounds, d_red_xp, zero)
    d_blue_s = xp.where(in_bounds, d_blue_xp, zero)

    valid = in_bounds & (lut[d_red_s, d_blue_s] >= 0)

    # sign = +1 if d_red is xstart, -1 if d_blue is xstart
    sign = xp.astype(sign_lut[d_red_s, d_blue_s], xp.float32)

    # dx_blue_red: positive means emission is toward the red detector
    dx_blue_red = dt_xp * xp.asarray(C_MM_PER_NS / 2.0, dtype=xp.float32, device=dev)

    # bin-centre position from LOR midpoint (toward xend) = -sign * dx_blue_red
    # so k = round((N-1)/2 - (sign*dx + tofcenter_offset) / W)
    half_n = xp.asarray((num_tof_bins - 1) / 2.0, dtype=xp.float32, device=dev)
    offset = xp.asarray(tofcenter_offset / tofbin_width, dtype=xp.float32, device=dev)
    scale = xp.asarray(1.0 / tofbin_width, dtype=xp.float32, device=dev)

    k_float = half_n - sign * dx_blue_red * scale - offset
    k_int = xp.astype(xp.round(k_float), xp.int32)

    in_range = valid & (k_int >= 0) & (k_int < num_tof_bins)
    invalid = xp.asarray(np.array([-1], dtype=np.int32), device=dev)
    return xp.where(in_range, k_int, invalid)
