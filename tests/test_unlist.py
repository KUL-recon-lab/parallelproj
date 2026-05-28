"""Tests for parallelproj.unlist — listmode-to-sinogram histogrammer."""
from __future__ import annotations

import numpy as np
import pytest
from types import ModuleType

import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
from parallelproj import to_numpy_array
from parallelproj.unlist import _build_inring_luts, regular_polygon_events_to_sinogram as _sinogram_native

def regular_polygon_events_to_sinogram(*args, **kwargs):
    """Thin wrapper that always returns a numpy array for test assertions."""
    return to_numpy_array(_sinogram_native(*args, **kwargs))

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
    desc = _lor_desc(scanner, span=1)

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
    events = xp.asarray(rows, dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events)

    assert sino.shape == desc.spatial_sinogram_shape
    assert np.max(sino) == 1.0, "no bin should accumulate more than one count"
    assert float(sino.sum()) == float(len(rows))


def test_non_tof_accumulation(xp: ModuleType, dev: str) -> None:
    """Duplicate events accumulate correctly."""
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner)

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    row = [int(sc[v0, r0]), int(sr[0]), int(ec[v0, r0]), int(er[0])]
    events = xp.asarray([row, row, row], dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events)
    assert sino.sum() == 3.0


# ---------------------------------------------------------------------------
# Crystal-swap invariance (non-TOF)
# ---------------------------------------------------------------------------

def test_crystal_swap_non_tof(xp: ModuleType, dev: str) -> None:
    """(d1,r1,d2,r2) and (d2,r2,d1,r1) must produce the same sinogram."""
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner)

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
    r1, r2 = int(sr[0]), int(er[0])

    ev_fwd = xp.asarray([[d1, r1, d2, r2]], dtype=xp.int32, device=dev)
    ev_bwd = xp.asarray([[d2, r2, d1, r1]], dtype=xp.int32, device=dev)

    assert np.allclose(
        regular_polygon_events_to_sinogram(desc, ev_fwd),
        regular_polygon_events_to_sinogram(desc, ev_bwd),
    ), "Crystal swap must not change the sinogram bin"


# ---------------------------------------------------------------------------
# TOF round-trip
# ---------------------------------------------------------------------------

def test_tof_single_event_lands_in_correct_bin(xp: ModuleType, dev: str) -> None:
    """A single TOF event with d1=xstart lands in the expected TOF bin."""
    num_tof_bins = 7
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner)

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])   # d1 is xstart → no flip
    r1, r2 = int(sr[0]), int(er[0])
    tof_bin = 2

    events = xp.asarray([[d1, r1, d2, r2, tof_bin]], dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events, num_tof_bins=num_tof_bins)

    assert sino.shape == (*desc.spatial_sinogram_shape, num_tof_bins)
    assert sino.sum() == 1.0

    nz = np.argwhere(sino > 0)
    assert len(nz) == 1
    # TOF bin axis is trailing — no flip because d1 is xstart and tof_bin_sign=+1
    assert nz[0, -1] == tof_bin, "TOF bin must be stored unflipped when d1 is xstart"


def test_tof_round_trip(xp: ModuleType, dev: str) -> None:
    """Non-self-pair TOF sinogram bins each hit exactly once."""
    num_tof_bins = 5
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner, span=1)

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
    events = xp.asarray(rows, dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events, num_tof_bins=num_tof_bins)

    assert sino.shape == (*desc.spatial_sinogram_shape, num_tof_bins)
    assert np.max(sino) == 1.0
    assert float(sino.sum()) == float(len(rows))


# ---------------------------------------------------------------------------
# TOF direction: crystal-swap with mirrored TOF bin
# ---------------------------------------------------------------------------

def test_tof_crystal_swap_mirrored_bin(xp: ModuleType, dev: str) -> None:
    """Swapping (d1,r1,d2,r2) and mirroring the TOF bin gives the same sinogram.

    Convention: bin 0 = closest to d1 (tof_bin_sign=+1, default).

    Event (d1_start, r1, d2_end, r2, bin=k) and
    Event (d2_end, r2, d1_start, r1, bin=(n-1-k))
    must both land in the same sinogram bin at TOF index k.
    """
    num_tof_bins = 7
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner)

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])   # d1 = xstart
    r1, r2 = int(sr[0]), int(er[0])
    tof_bin = 2
    tof_bin_mirror = num_tof_bins - 1 - tof_bin   # = 4

    ev_fwd = xp.asarray([[d1, r1, d2, r2, tof_bin]], dtype=xp.int32, device=dev)
    ev_bwd = xp.asarray([[d2, r2, d1, r1, tof_bin_mirror]], dtype=xp.int32, device=dev)

    sino_fwd = regular_polygon_events_to_sinogram(desc, ev_fwd, num_tof_bins=num_tof_bins)
    sino_bwd = regular_polygon_events_to_sinogram(desc, ev_bwd, num_tof_bins=num_tof_bins)

    assert np.allclose(sino_fwd, sino_bwd), (
        "Crystal swap with mirrored TOF bin must produce the same sinogram"
    )
    # Both should land at TOF index tof_bin (not tof_bin_mirror)
    nz_fwd = np.argwhere(sino_fwd > 0)
    nz_bwd = np.argwhere(sino_bwd > 0)
    assert nz_fwd[0, -1] == tof_bin
    assert nz_bwd[0, -1] == tof_bin


# ---------------------------------------------------------------------------
# tof_bin_sign=-1
# ---------------------------------------------------------------------------

def test_tof_bin_sign_minus1(xp: ModuleType, dev: str) -> None:
    """tof_bin_sign=-1 (bin 0 closest to d2): mirrored raw bin gives same sinogram.

    With sign=-1, physical offset k from d1 corresponds to raw bin (n-1-k)
    because bin 0 counts from d2 instead of d1.
    """
    num_tof_bins = 7
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner)

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
    r1, r2 = int(sr[0]), int(er[0])
    phys_bin = 2   # physical sinogram bin index

    # sign=+1: raw bin = phys_bin (bin 0 from d1)
    ev_p1 = xp.asarray([[d1, r1, d2, r2, phys_bin]], dtype=xp.int32, device=dev)
    # sign=-1: bin 0 is from d2, so phys_bin from d1 = (n-1-phys_bin) from d2
    raw_m1 = num_tof_bins - 1 - phys_bin
    ev_m1 = xp.asarray([[d1, r1, d2, r2, raw_m1]], dtype=xp.int32, device=dev)

    sino_p1 = regular_polygon_events_to_sinogram(
        desc, ev_p1, num_tof_bins=num_tof_bins, tof_bin_sign=1
    )
    sino_m1 = regular_polygon_events_to_sinogram(
        desc, ev_m1, num_tof_bins=num_tof_bins, tof_bin_sign=-1
    )

    assert np.allclose(sino_p1, sino_m1), (
        "tof_bin_sign=-1 with mirrored raw bin must give the same sinogram"
    )


# ---------------------------------------------------------------------------
# Out-of-FOV events silently dropped
# ---------------------------------------------------------------------------

def test_out_of_fov_events_dropped(xp: ModuleType, dev: str) -> None:
    """Events outside the FOV are silently discarded — sinogram stays zero."""
    scanner = _small_scanner(xp, dev, num_rings=3)
    desc = _lor_desc(scanner, max_ring_difference=1)

    n_crystals = scanner.num_lor_endpoints_per_ring
    num_rings = scanner.num_rings

    invalid_rows = [
        [n_crystals, 0, 0, 0],           # d1 out of range
        [-1, 0, 0, 0],                    # d1 negative
        [0, 0, 1, num_rings],             # r2 out of range
        [0, 0, 1, -1],                    # r2 negative
        [0, 0, 1, 2],                     # ring difference (=2) > max_ring_difference (=1)
        [0, 0, 0, 0],                     # same crystal → invalidated by fill_diagonal
    ]
    events = xp.asarray(invalid_rows, dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events)
    assert np.all(sino == 0), "All out-of-FOV events must be silently dropped"


def test_out_of_range_tof_bin_dropped(xp: ModuleType, dev: str) -> None:
    """TOF events with out-of-range bins are dropped."""
    num_tof_bins = 5
    scanner = _small_scanner(xp, dev, num_rings=2)
    desc = _lor_desc(scanner)

    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)
    sr = to_numpy_array(desc.start_plane_index)
    er = to_numpy_array(desc.end_plane_index)

    v0, r0 = _first_valid_vr(sc, ec)
    d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
    r1, r2 = int(sr[0]), int(er[0])

    events = xp.asarray(
        [
            [d1, r1, d2, r2, num_tof_bins],      # bin == num_tof_bins → out of range
            [d1, r1, d2, r2, -1],                # negative bin
        ],
        dtype=xp.int32,
        device=dev,
    )
    sino = regular_polygon_events_to_sinogram(desc, events, num_tof_bins=num_tof_bins)
    assert np.all(sino == 0)


# ---------------------------------------------------------------------------
# Span > 1
# ---------------------------------------------------------------------------

def test_span3_round_trip(xp: ModuleType, dev: str) -> None:
    """Span-3: total count equals number of contributing (ring-pair, view, rad) triples."""
    scanner = _small_scanner(xp, dev, num_rings=5)
    desc = _lor_desc(scanner, span=3, max_ring_difference=3)

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
    events = xp.asarray(rows, dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events)

    assert sino.shape == desc.spatial_sinogram_shape
    assert float(sino.sum()) == float(len(rows))


def test_span3_single_event_plane(xp: ModuleType, dev: str) -> None:
    """A single span-3 event lands in the correct sinogram plane."""
    scanner = _small_scanner(xp, dev, num_rings=5)
    desc = _lor_desc(scanner, span=3, max_ring_difference=3)

    m = desc.michelogram
    sc = to_numpy_array(desc.start_in_ring_index)
    ec = to_numpy_array(desc.end_in_ring_index)

    v0, r0 = _first_valid_vr(sc, ec)
    target_plane = 3
    r1 = int(m.plane_start_rings[target_plane, 0])
    r2 = int(m.plane_end_rings[target_plane, 0])
    d1 = int(sc[v0, r0])
    d2 = int(ec[v0, r0])

    events = xp.asarray([[d1, r1, d2, r2]], dtype=xp.int32, device=dev)
    sino = regular_polygon_events_to_sinogram(desc, events)

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
        desc = _lor_desc(scanner, sinogram_order=order)
        sc = to_numpy_array(desc.start_in_ring_index)
        ec = to_numpy_array(desc.end_in_ring_index)
        sr = to_numpy_array(desc.start_plane_index)
        er = to_numpy_array(desc.end_plane_index)

        v0, r0 = _first_valid_vr(sc, ec)
        d1, d2 = int(sc[v0, r0]), int(ec[v0, r0])
        r1, r2 = int(sr[0]), int(er[0])

        events = xp.asarray([[d1, r1, d2, r2]], dtype=xp.int32, device=dev)
        sino = regular_polygon_events_to_sinogram(desc, events)

        assert sino.shape == desc.spatial_sinogram_shape, f"Wrong shape for {order}"
        assert sino.sum() == 1.0, f"Wrong total count for {order}"


# ---------------------------------------------------------------------------
# Empty events
# ---------------------------------------------------------------------------

def test_empty_events_non_tof(xp: ModuleType, dev: str) -> None:
    """Empty non-TOF event array returns a zero sinogram of the right shape."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events_np = np.zeros((0, 4), dtype=np.int32)
    sino = regular_polygon_events_to_sinogram(desc, events_np)
    assert sino.shape == desc.spatial_sinogram_shape
    assert np.all(sino == 0)


def test_empty_events_tof(xp: ModuleType, dev: str) -> None:
    """Empty TOF event array returns a zero sinogram of the right shape."""
    num_tof_bins = 5
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events_np = np.zeros((0, 5), dtype=np.int32)
    sino = regular_polygon_events_to_sinogram(desc, events_np, num_tof_bins=num_tof_bins)
    assert sino.shape == (*desc.spatial_sinogram_shape, num_tof_bins)
    assert np.all(sino == 0)


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_error_bad_tof_bin_sign(xp: ModuleType, dev: str) -> None:
    """tof_bin_sign values other than ±1 must raise ValueError."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events = xp.asarray([[0, 0, 1, 0, 0]], dtype=xp.int32, device=dev)
    with pytest.raises(ValueError, match="tof_bin_sign"):
        regular_polygon_events_to_sinogram(desc, events, num_tof_bins=5, tof_bin_sign=0)


def test_error_missing_num_tof_bins(xp: ModuleType, dev: str) -> None:
    """5-column events without num_tof_bins must raise ValueError."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events = xp.asarray([[0, 0, 1, 0, 0]], dtype=xp.int32, device=dev)
    with pytest.raises(ValueError, match="num_tof_bins"):
        regular_polygon_events_to_sinogram(desc, events)


def test_error_spurious_num_tof_bins(xp: ModuleType, dev: str) -> None:
    """4-column events with num_tof_bins specified must raise ValueError."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events = xp.asarray([[0, 0, 1, 0]], dtype=xp.int32, device=dev)
    with pytest.raises(ValueError, match="num_tof_bins"):
        regular_polygon_events_to_sinogram(desc, events, num_tof_bins=5)


def test_error_wrong_col_count(xp: ModuleType, dev: str) -> None:
    """Event arrays with column count other than 4 or 5 must raise ValueError."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events = xp.asarray([[0, 0, 1]], dtype=xp.int32, device=dev)
    with pytest.raises(ValueError, match="columns"):
        regular_polygon_events_to_sinogram(desc, events)


def test_error_1d_events(xp: ModuleType, dev: str) -> None:
    """1-D event array must raise ValueError."""
    scanner = _small_scanner(xp, dev)
    desc = _lor_desc(scanner)
    events = xp.asarray([0, 0, 1, 0], dtype=xp.int32, device=dev)
    with pytest.raises(ValueError, match="2D"):
        regular_polygon_events_to_sinogram(desc, events)


@pytest.mark.skipif(not _no_bincount_xp_dev, reason="no non-bincount backends available")
def test_error_no_bincount_backend(xp: ModuleType, dev: str) -> None:  # noqa: ARG001
    """Backends without bincount must raise NotImplementedError.

    The module-level (xp, dev) are from a bincount backend and are ignored;
    the test unconditionally exercises array_api_strict.
    """
    import array_api_strict as xp_strict

    import numpy as np_plain
    scanner = _small_scanner(np_plain, "cpu")
    desc = _lor_desc(scanner)
    events = xp_strict.asarray([[0, 0, 1, 0]])
    with pytest.raises(NotImplementedError, match="bincount"):
        regular_polygon_events_to_sinogram(desc, events)
