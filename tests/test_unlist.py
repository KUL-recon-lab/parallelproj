"""Tests for parallelproj.unlist — listmode-to-sinogram histogrammer."""
from __future__ import annotations

import numpy as np
import pytest
from types import ModuleType

import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
import parallelproj.projectors as pp_proj
import parallelproj.tof as ppt
from parallelproj import to_numpy_array
from parallelproj.unlist import (
    _build_inring_luts,
    regular_polygon_events_to_sinogram as _sinogram_native,
)
try:
    from parallelproj.unlist import detection_times_to_tof_bin
except ImportError:
    detection_times_to_tof_bin = None  # type: ignore[assignment]

from .config import xp_dev_list

# Only run against backends that provide bincount (numpy, torch, cupy).
# array_api_strict is intentionally excluded; a dedicated test below checks
# that NotImplementedError is raised for backends without bincount.
pytestmark = pytest.mark.parametrize(
    "xp,dev", [(xp, dev) for xp, dev in xp_dev_list if hasattr(xp, "bincount")]
)

# Backends that do NOT have bincount — used by the NotImplementedError test.
_no_bincount_xp_dev = [
    (xp, dev) for xp, dev in xp_dev_list if not hasattr(xp, "bincount")
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_scanner(xp: ModuleType, dev: str, num_rings: int = 3):
    """4 sides × 4 endpoints/side = 16 crystals/ring."""
    return pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=100.0,
        num_sides=4,
        num_lor_endpoints_per_side=4,
        lor_spacing=5.0,
        ring_positions=xp.linspace(-10.0, 10.0, num_rings, device=dev),
        symmetry_axis=2,
    )


def _lor_desc(
    scanner: pps.RegularPolygonPETScannerGeometry,
    span: int = 1,
    max_ring_difference: int | None = None,
    sinogram_order: ppl.SinogramSpatialAxisOrder = ppl.SinogramSpatialAxisOrder.RVP,
) -> ppl.RegularPolygonPETLORDescriptor:
    """Build a LOR descriptor for a scanner."""
    if max_ring_difference is None:
        max_ring_difference = scanner.num_rings - 1
    return ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(
            scanner.num_rings,
            max_ring_difference=max_ring_difference,
            span=span,
        ),
        radial_trim=1,
        sinogram_order=sinogram_order,
    )


def _first_valid_vr(sc: np.ndarray, ec: np.ndarray) -> tuple[int, int]:
    """Return (v, r) of the first non-self-pair position (sc[v,r] != ec[v,r])."""
    idxs = np.argwhere(sc != ec)
    return int(idxs[0, 0]), int(idxs[0, 1])


def _small_proj(scanner, span=1, max_ring_difference=None,
                sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
                tof_params=None):
    """Build a minimal RegularPolygonPETProjector for use in tests."""
    lor_desc = _lor_desc(scanner, span=span, max_ring_difference=max_ring_difference,
                         sinogram_order=sinogram_order)
    proj = pp_proj.RegularPolygonPETProjector(
        lor_desc, img_shape=(4, 4, 4), voxel_size=(4.0, 4.0, 4.0)
    )
    if tof_params is not None:
        proj.tof_parameters = tof_params
    return proj


def _unlist(proj, events_np):
    """Unpack a stacked numpy events array and call regular_polygon_events_to_sinogram.

    events_np shape: (N, 4) for non-TOF or (N, 5) for TOF.
    All column arrays are passed on the same xp/dev as the scanner.
    """
    xp = proj.xp
    dev = proj.lor_descriptor.dev
    cols = [xp.asarray(events_np[:, i], dtype=xp.int32, device=dev)
            for i in range(events_np.shape[1])]
    if events_np.shape[1] == 4:
        return to_numpy_array(_sinogram_native(proj, *cols))
    return to_numpy_array(
        _sinogram_native(proj, *cols[:4], unsigned_sinogram_tof_bin=cols[4])
    )


# ---------------------------------------------------------------------------
# LUT structure tests
# ---------------------------------------------------------------------------

def test_inring_lut_symmetry(xp: ModuleType, dev: str) -> None:
    """inring_lut is symmetric; inring_tof_sign is antisymmetric off-diagonal;
    diagonal is invalidated after self-pair fill."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    lut, sign = _build_inring_luts(desc)
    assert np.array_equal(lut, lut.T), "inring_lut must be symmetric"
    assert np.all(lut.diagonal() == -1), "self-pair LUT entries must be -1"
    assert np.all(sign.diagonal() == 0), "self-pair sign entries must be 0"
    n = scanner.num_lor_endpoints_per_ring
    off = ~np.eye(n, dtype=bool)
    assert np.array_equal(sign[off], -sign.T[off]), "sign must be antisymmetric off-diagonal"


def test_inring_lut_valid_pairs_covered(xp: ModuleType, dev: str) -> None:
    """Every non-self-pair (view, rad) maps to the expected flat LUT index."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    lut, _ = _build_inring_luts(desc)

    ds = to_numpy_array(desc.start_in_ring_index).ravel().astype(int)
    de = to_numpy_array(desc.end_in_ring_index).ravel().astype(int)
    for k, (d1, d2) in enumerate(zip(ds, de)):
        if d1 == d2:
            assert lut[d1, d2] == -1, f"self-pair at k={k} must be invalidated"
        else:
            assert lut[d1, d2] == k, f"LUT entry mismatch at k={k}"
            assert lut[d2, d1] == k, f"LUT reverse entry mismatch at k={k}"


# ---------------------------------------------------------------------------
# Non-TOF round-trip
# ---------------------------------------------------------------------------

def test_non_tof_round_trip(xp: ModuleType, dev: str) -> None:
    """Non-self-pair sinogram bins each hit exactly once; self-pair bins stay 0."""
    scanner = _small_scanner(xp, dev, num_rings=3)
    proj = _small_proj(scanner, span=1)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)   # (num_views, num_rad)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)      # (num_planes,)
    er = to_numpy_array(desc.end_plane_index)

    # only include events for non-self-pair (v, r) positions
    valid_vr = sc != ec  # (num_views, num_rad) bool mask
    rows = [
        [int(sc[v, r]), int(sr[p]), int(ec[v, r]), int(er[p])]
        for p in range(desc.num_planes)
        for v in range(desc.num_views)
        for r in range(desc.num_rad)
        if valid_vr[v, r]
    ]
    events_np = np.array(rows, dtype=np.int32)
    sino = _unlist(proj, events_np)

    assert sino.shape == desc.spatial_sinogram_shape
    assert np.max(sino) == 1.0, "no bin should accumulate more than one count"
    assert float(sino.sum()) == float(len(rows))


def test_non_tof_accumulation(xp: ModuleType, dev: str) -> None:
    """Duplicate events accumulate correctly."""
    scanner = _small_scanner(xp, dev, num_rings=2)
    proj = _small_proj(scanner)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    row = [int(sc[v0, r0]), int(sr[0]), int(ec[v0, r0]), int(er[0])]
    events_np = np.array([row, row, row], dtype=np.int32)
    sino = _unlist(proj, events_np)
    assert sino.sum() == 3.0


# ---------------------------------------------------------------------------
# Crystal-swap invariance (non-TOF)
# ---------------------------------------------------------------------------

def test_crystal_swap_non_tof(xp: ModuleType, dev: str) -> None:
    """(d1,r1,d2,r2) and (d2,r2,d1,r1) must produce the same sinogram."""
    scanner = _small_scanner(xp, dev, num_rings=2)
    proj = _small_proj(scanner)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
    r1, r2 = int(sr[0]), int(er[0])

    ev_fwd = np.array([[d1, r1, d2, r2]], dtype=np.int32)
    ev_bwd = np.array([[d2, r2, d1, r1]], dtype=np.int32)

    assert np.allclose(
        _unlist(proj, ev_fwd),
        _unlist(proj, ev_bwd),
    ), "Crystal swap must not change the sinogram bin"


# ---------------------------------------------------------------------------
# TOF round-trip
# ---------------------------------------------------------------------------

def test_tof_single_event_lands_in_correct_bin(xp: ModuleType, dev: str) -> None:
    """A single TOF event with d_red=xstart lands in the expected TOF bin."""
    num_tof_bins = 7
    scanner = _small_scanner(xp, dev, num_rings=2)
    tof_params = ppt.TOFParameters(num_tofbins=num_tof_bins, tofbin_width=40.0,
                                   sigma_tof=5.0, tofcenter_offset=0.0)
    proj = _small_proj(scanner, tof_params=tof_params)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])   # d1 is xstart → no flip
    r1, r2 = int(sr[0]), int(er[0])
    tof_bin = 2

    events_np = np.array([[d1, r1, d2, r2, tof_bin]], dtype=np.int32)
    sino = _unlist(proj, events_np)

    assert sino.shape == (*desc.spatial_sinogram_shape, num_tof_bins)
    assert sino.sum() == 1.0

    nz = np.argwhere(sino > 0)
    assert len(nz) == 1
    # TOF bin axis is trailing — no flip because d1 is xstart
    assert nz[0, -1] == tof_bin, "TOF bin must be stored unflipped when d_red is xstart"


def test_tof_round_trip(xp: ModuleType, dev: str) -> None:
    """Non-self-pair TOF sinogram bins each hit exactly once."""
    num_tof_bins = 5
    scanner = _small_scanner(xp, dev, num_rings=2)
    tof_params = ppt.TOFParameters(num_tofbins=num_tof_bins, tofbin_width=40.0,
                                   sigma_tof=5.0, tofcenter_offset=0.0)
    proj = _small_proj(scanner, span=1, tof_params=tof_params)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    valid_vr = sc != ec
    rows = [
        [int(sc[v, r]), int(sr[p]), int(ec[v, r]), int(er[p]), t]
        for p in range(desc.num_planes)
        for v in range(desc.num_views)
        for r in range(desc.num_rad)
        if valid_vr[v, r]
        for t in range(num_tof_bins)
    ]
    events_np = np.array(rows, dtype=np.int32)
    sino = _unlist(proj, events_np)

    assert sino.shape == (*desc.spatial_sinogram_shape, num_tof_bins)
    assert np.max(sino) == 1.0
    assert float(sino.sum()) == float(len(rows))


# ---------------------------------------------------------------------------
# TOF direction: crystal-swap with mirrored TOF bin
# ---------------------------------------------------------------------------

def test_tof_crystal_swap_same_bin(xp: ModuleType, dev: str) -> None:
    """Swapping d_red / d_blue with the *same* projector-convention TOF bin
    must produce the identical sinogram.

    In the projector convention, bin 0 is always closest to xstart regardless
    of which crystal is labeled red or blue.  The function determines xstart
    from the LOR descriptor, so the ring ordering is correct in both cases
    and no TOF-bin flip is needed at the call site.

    Event (d_xstart, r1, d_xend, r2, bin=k)  and
    Event (d_xend,   r2, d_xstart, r1, bin=k)
    must both land in the same sinogram bin at TOF index k.
    """
    num_tof_bins = 7
    scanner = _small_scanner(xp, dev, num_rings=2)
    tof_params = ppt.TOFParameters(num_tofbins=num_tof_bins, tofbin_width=40.0,
                                   sigma_tof=5.0, tofcenter_offset=0.0)
    proj = _small_proj(scanner, tof_params=tof_params)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d_xstart, d_xend = int(sc[v0, r0]), int(ec[v0, r0])
    r_start, r_end = int(sr[0]), int(er[0])
    tof_bin = 2   # projector-convention bin (bin 0 = closest to xstart)

    # same bin in both events — no mirror needed
    ev_fwd = np.array([[d_xstart, r_start, d_xend,   r_end,   tof_bin]], dtype=np.int32)
    ev_bwd = np.array([[d_xend,   r_end,   d_xstart, r_start, tof_bin]], dtype=np.int32)

    sino_fwd = _unlist(proj, ev_fwd)
    sino_bwd = _unlist(proj, ev_bwd)

    assert np.allclose(sino_fwd, sino_bwd), (
        "Crystal swap with the same projector-convention TOF bin must produce "
        "the same sinogram"
    )
    nz_fwd = np.argwhere(sino_fwd > 0)
    nz_bwd = np.argwhere(sino_bwd > 0)
    assert nz_fwd[0, -1] == tof_bin
    assert nz_bwd[0, -1] == tof_bin


# ---------------------------------------------------------------------------
# Out-of-FOV events silently dropped
# ---------------------------------------------------------------------------

def test_out_of_fov_events_dropped(xp: ModuleType, dev: str) -> None:
    """Events outside the FOV are silently discarded — sinogram stays zero."""
    scanner = _small_scanner(xp, dev, num_rings=3)
    proj = _small_proj(scanner, max_ring_difference=1)

    n_crystals = scanner.num_lor_endpoints_per_ring
    num_rings = scanner.num_rings

    invalid_rows = [
        [n_crystals, 0, 0, 0],           # d_red out of range
        [-1, 0, 0, 0],                    # d_red negative
        [0, 0, 1, num_rings],             # r_blue out of range
        [0, 0, 1, -1],                    # r_blue negative
        [0, 0, 1, 2],                     # ring difference (=2) > max_ring_difference (=1)
        [0, 0, 0, 0],                     # same crystal → invalidated by fill_diagonal
    ]
    events_np = np.array(invalid_rows, dtype=np.int32)
    sino = _unlist(proj, events_np)
    assert np.all(sino == 0), "All out-of-FOV events must be silently dropped"


def test_out_of_range_tof_bin_dropped(xp: ModuleType, dev: str) -> None:
    """TOF events with out-of-range bins are dropped."""
    num_tof_bins = 5
    scanner = _small_scanner(xp, dev, num_rings=2)
    tof_params = ppt.TOFParameters(num_tofbins=num_tof_bins, tofbin_width=40.0,
                                   sigma_tof=5.0, tofcenter_offset=0.0)
    proj = _small_proj(scanner, tof_params=tof_params)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
    r1, r2 = int(sr[0]), int(er[0])

    invalid_tof_rows = np.array(
        [
            [d1, r1, d2, r2, num_tof_bins],      # bin == num_tof_bins → out of range
            [d1, r1, d2, r2, -1],                # negative bin
        ],
        dtype=np.int32,
    )
    sino = _unlist(proj, invalid_tof_rows)
    assert np.all(sino == 0)


# ---------------------------------------------------------------------------
# Span > 1
# ---------------------------------------------------------------------------

def test_span3_round_trip(xp: ModuleType, dev: str) -> None:
    """Span-3: total count equals number of contributing (ring-pair, view, rad) triples."""
    scanner = _small_scanner(xp, dev, num_rings=5)
    proj = _small_proj(scanner, span=3, max_ring_difference=3)
    desc = proj.lor_descriptor

    m = desc.michelogram
    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    valid_vr = sc != ec

    rows = [
        [int(sc[v, r]), int(m.plane_start_rings[p, k]), int(ec[v, r]), int(m.plane_end_rings[p, k])]
        for p in range(desc.num_planes)
        for k in range(int(m.plane_multiplicity[p]))
        for v in range(desc.num_views)
        for r in range(desc.num_rad)
        if valid_vr[v, r]
    ]
    events_np = np.array(rows, dtype=np.int32)
    sino = _unlist(proj, events_np)

    assert sino.shape == desc.spatial_sinogram_shape
    assert float(sino.sum()) == float(len(rows))


def test_span3_single_event_plane(xp: ModuleType, dev: str) -> None:
    """A single span-3 event lands in the correct sinogram plane."""
    scanner = _small_scanner(xp, dev, num_rings=5)
    proj = _small_proj(scanner, span=3, max_ring_difference=3)
    desc = proj.lor_descriptor

    m = desc.michelogram
    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)

    v0, r0 = _first_valid_vr(sc, ec)
    target_plane = 3
    r1 = int(m.plane_start_rings[target_plane, 0])
    r2 = int(m.plane_end_rings[target_plane, 0])
    d1 = int(sc[v0, r0])
    d2 = int(ec[v0, r0])

    events_np = np.array([[d1, r1, d2, r2]], dtype=np.int32)
    sino = _unlist(proj, events_np)

    assert sino.sum() == 1.0
    nz = np.argwhere(sino > 0)
    assert len(nz) == 1
    assert nz[0, desc.plane_axis_num] == target_plane


# ---------------------------------------------------------------------------
# Sinogram order variants
# ---------------------------------------------------------------------------

def test_sinogram_order_variants(xp: ModuleType, dev: str) -> None:
    """All sinogram axis orderings give the correct shape and total count."""
    scanner = _small_scanner(xp, dev, num_rings=2)

    for order in [
        ppl.SinogramSpatialAxisOrder.RVP,
        ppl.SinogramSpatialAxisOrder.PVR,
        ppl.SinogramSpatialAxisOrder.VRP,
    ]:
        proj = _small_proj(scanner, sinogram_order=order)
        desc = proj.lor_descriptor
        sc = to_numpy_array(desc.start_in_ring_index)
        ec = to_numpy_array(desc.end_in_ring_index)
        sr = to_numpy_array(desc.start_plane_index)
        er = to_numpy_array(desc.end_plane_index)

        v0, r0 = _first_valid_vr(sc, ec)
        d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
        r1, r2 = int(sr[0]), int(er[0])

        events_np = np.array([[d1, r1, d2, r2]], dtype=np.int32)
        sino = _unlist(proj, events_np)

        assert sino.shape == desc.spatial_sinogram_shape, f"Wrong shape for {order}"
        assert sino.sum() == 1.0, f"Wrong total count for {order}"


# ---------------------------------------------------------------------------
# Empty events
# ---------------------------------------------------------------------------

def test_empty_events_non_tof(xp: ModuleType, dev: str) -> None:
    """Empty non-TOF event arrays return a zero sinogram of the right shape."""
    scanner = _small_scanner(xp, dev)
    proj = _small_proj(scanner)
    desc = proj.lor_descriptor
    dev_ = desc.dev
    d_red = xp.zeros(0, dtype=xp.int32, device=dev_)
    r_red = xp.zeros(0, dtype=xp.int32, device=dev_)
    d_blue = xp.zeros(0, dtype=xp.int32, device=dev_)
    r_blue = xp.zeros(0, dtype=xp.int32, device=dev_)
    sino = to_numpy_array(_sinogram_native(proj, d_red, r_red, d_blue, r_blue))
    assert sino.shape == desc.spatial_sinogram_shape
    assert np.all(sino == 0)


def test_empty_events_tof(xp: ModuleType, dev: str) -> None:
    """Empty TOF event arrays return a zero sinogram of the right shape."""
    num_tof_bins = 5
    scanner = _small_scanner(xp, dev)
    tof_params = ppt.TOFParameters(num_tofbins=num_tof_bins, tofbin_width=40.0,
                                   sigma_tof=5.0, tofcenter_offset=0.0)
    proj = _small_proj(scanner, tof_params=tof_params)
    desc = proj.lor_descriptor
    dev_ = desc.dev
    d_red = xp.zeros(0, dtype=xp.int32, device=dev_)
    r_red = xp.zeros(0, dtype=xp.int32, device=dev_)
    d_blue = xp.zeros(0, dtype=xp.int32, device=dev_)
    r_blue = xp.zeros(0, dtype=xp.int32, device=dev_)
    unsigned_sinogram_tof_bin = xp.zeros(0, dtype=xp.int32, device=dev_)
    sino = to_numpy_array(
        _sinogram_native(
            proj, d_red, r_red, d_blue, r_blue,
            unsigned_sinogram_tof_bin=unsigned_sinogram_tof_bin,
        )
    )
    assert sino.shape == (*desc.spatial_sinogram_shape, num_tof_bins)
    assert np.all(sino == 0)


# ---------------------------------------------------------------------------
# detection_times_to_tof_bin
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    detection_times_to_tof_bin is None,
    reason="detection_times_to_tof_bin not yet implemented in parallelproj.unlist",
)
def test_detection_times_to_tof_bin(xp: ModuleType, dev: str) -> None:
    """detection_times_to_tof_bin maps physical detection time differences to bins."""
    num_tof_bins = 5
    tof_bin_width = 40.0  # mm
    scanner = _small_scanner(xp, dev, num_rings=1)
    tof_params = ppt.TOFParameters(num_tofbins=num_tof_bins, tofbin_width=tof_bin_width,
                                   sigma_tof=5.0, tofcenter_offset=0.0)
    proj = _small_proj(scanner, tof_params=tof_params)
    desc = proj.lor_descriptor

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d_red_val  = int(sc[v0, r0])   # xstart crystal
    d_blue_val = int(ec[v0, r0])

    # Scenario 1: dt=0 → emission at LOR midpoint → center bin = (N-1)//2
    center_bin = (num_tof_bins - 1) // 2
    dt_center = np.array([0.0], dtype=np.float32)

    d_red_arr = xp.asarray([d_red_val], dtype=xp.int32, device=dev)
    d_blue_arr = xp.asarray([d_blue_val], dtype=xp.int32, device=dev)
    dt_arr = xp.asarray(dt_center, device=dev)

    bin_center = to_numpy_array(
        detection_times_to_tof_bin(d_red_arr, d_blue_arr, dt_arr, proj)
    )
    assert int(bin_center[0]) == center_bin, (
        f"dt=0 should give center bin {center_bin}, got {bin_center[0]}"
    )

    # Scenario 2: emission fully toward xstart (bin 0).
    # With d_red=xstart (sign=+1):
    #   k = round((N-1)/2 - dx_blue_red / W)  →  k=0  ⟺  dx_blue_red = (N-1)/2 * W
    # dx_blue_red = (c/2) * dt  →  dt = (N-1)/2 * W / (c/2)   [nanoseconds]
    from parallelproj.unlist import C_MM_PER_NS
    dt_bin0_ns = float((num_tof_bins - 1) / 2.0 * tof_bin_width / (C_MM_PER_NS / 2.0))
    dt_bin0_arr = xp.asarray(np.array([dt_bin0_ns], dtype=np.float32), device=dev)

    bin0 = to_numpy_array(
        detection_times_to_tof_bin(d_red_arr, d_blue_arr, dt_bin0_arr, proj)
    )
    assert int(bin0[0]) == 0, (
        f"dt corresponding to bin 0 should give bin 0, got {bin0[0]}"
    )


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not _no_bincount_xp_dev, reason="no non-bincount backends available")
def test_error_no_bincount_backend(xp: ModuleType, dev: str) -> None:  # noqa: ARG001
    """Backends without bincount must raise NotImplementedError.

    The module-level (xp, dev) are from a bincount backend and are ignored;
    the test unconditionally exercises array_api_strict.
    """
    import array_api_strict as xp_strict
    import numpy as np_plain

    scanner = _small_scanner(np_plain, "cpu")
    proj = _small_proj(scanner)

    d_red = xp_strict.asarray([0], dtype=xp_strict.int32)
    r_red = xp_strict.asarray([0], dtype=xp_strict.int32)
    d_blue = xp_strict.asarray([1], dtype=xp_strict.int32)
    r_blue = xp_strict.asarray([0], dtype=xp_strict.int32)

    with pytest.raises(NotImplementedError, match="bincount"):
        _sinogram_native(proj, d_red, r_red, d_blue, r_blue)
