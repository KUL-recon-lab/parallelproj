"""Analytic correctness tests for the TOF forward projection.

These complement the low-level backend test in ``parallelproj_core``
(``tests/test_tof_sinogram_projection.py``) by checking the **public**
``parallelproj`` projector API.  Since parallelproj-core 3.0, the TOF kernel
used by the Joseph projector is a Gaussian **truncated at** ``num_sigmas``
**sigma**, integrated over the part of a TOF bin inside that window and
normalised to unit total weight.  For a voxel whose distance to the centre of a
TOF bin is ``dx`` (in mm), with bin width ``W`` and spatial resolution
``sigma`` (both in mm) and ``L = num_sigmas * sigma``,

    w(dx) = [Phi(min(dx + W/2, L) / sigma) - Phi(max(dx - W/2, -L) / sigma)]
            / [Phi(num_sigmas) - Phi(-num_sigmas)]        (0 if the bin is
                                                            outside the window)

where ``Phi`` is the standard normal CDF.  The weights of all bins sum to one
(if the TOF bins cover the window), so summing the TOF forward projection over
all TOF bins reproduces the (non-TOF) line integral, and bins entirely outside
the window get exactly zero weight.  (parallelproj-core < 3.0 used the full
Gaussian bin integrals of the bins overlapping the window instead.)

We drive this through :class:`parallelproj.projectors.ListmodePETProjector`,
which accepts arbitrary LOR endpoints (the high-level sinogram projectors only
expose scanner-defined LORs).  Its ``event_tofbins`` use the *unsigned* bin
convention ``0 .. num_tofbins - 1`` (the same indices produced by
:meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode`); for an
LOR with zero ``tofcenter_offset`` the LOR centre maps to bin
``(num_tofbins - 1) / 2``.
"""

from __future__ import annotations

import math
from types import ModuleType

import array_api_compat.numpy as np

import parallelproj.projectors as ppp
import parallelproj.tof as ppt
from parallelproj import to_numpy_array

from .config import pytestmark
import pytest


def _phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def effective_tof_kernel(
    dx: float, sigma: float, tofbin_width: float, num_sigmas: float
) -> float:
    """Weight of a TOF bin of width ``W`` whose centre is ``dx`` away from the
    voxel: the Gaussian of width ``sigma``, truncated at ``num_sigmas`` sigma,
    integrated over the part of the bin inside the window and normalised to
    unit total weight (the kernel of parallelproj-core >= 3.0)."""
    half_window = num_sigmas * sigma
    lo = max(dx - 0.5 * tofbin_width, -half_window)
    hi = min(dx + 0.5 * tofbin_width, half_window)
    if hi <= lo:
        return 0.0
    norm = _phi(num_sigmas) - _phi(-num_sigmas)
    return (_phi(hi / sigma) - _phi(lo / sigma)) / norm


@pytest.mark.parametrize(
    "direction, sigma_tof, num_tofbins",
    [
        (0, 2.0, 27),
        (1, 2.0, 27),
        (2, 2.0, 27),
        (0, 1.5, 27),
        (0, 3.5, 41),
    ],
)
def test_tof_listmode_fwd_single_voxel(
    xp: ModuleType,
    dev: str,
    direction: int,
    sigma_tof: float,
    num_tofbins: int,
    tofbin_width: float = 0.8,
    num_sigmas: float = 3.0,
    nvox: int = 19,
) -> None:
    """A single central voxel on a centre-crossing, axis-aligned LOR must give
    a TOF spectrum equal to the analytic truncated kernel in every bin
    (including exact zeros outside the +/- num_sigmas window): symmetric,
    peaked at the central bin, and summing to the line integral (the voxel
    size).  The parameters cover the window edge on a bin edge (sigma = 2.0:
    +/- 7.5 bins) and inside a bin (sigma = 1.5, 3.5)."""

    assert num_tofbins % 2 == 1  # so the LOR centre falls exactly on a bin

    voxel_size = (2.2, 2.5, 2.7)
    cf = voxel_size[direction]  # axis-aligned -> intersection length = voxel size

    # image with a single hot voxel at the centre (odd nvox -> world coord 0)
    img_shape = tuple(nvox if i == direction else 1 for i in range(3))
    center_vox = nvox // 2
    img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    idx = [0, 0, 0]
    idx[direction] = center_vox
    img[tuple(idx)] = 1.0

    # axis-aligned LOR through the centre, long enough to fully cross the image
    start_row = [0.0, 0.0, 0.0]
    end_row = [0.0, 0.0, 0.0]
    start_row[direction] = 60.0
    end_row[direction] = -60.0

    # one "event" per (unsigned) TOF bin so the projector returns the spectrum
    bins = list(range(num_tofbins))
    n_events = len(bins)

    xstart = xp.asarray([start_row] * n_events, dtype=xp.float32, device=dev)
    xend = xp.asarray([end_row] * n_events, dtype=xp.float32, device=dev)

    proj = ppp.ListmodePETProjector(xstart, xend, img_shape, voxel_size)
    proj.tof_parameters = ppt.TOFParameters(
        num_tofbins=num_tofbins,
        tofbin_width=tofbin_width,
        sigma_tof=sigma_tof,
        num_sigmas=num_sigmas,
    )
    proj.event_tofbins = xp.asarray(bins, dtype=xp.int16, device=dev)
    proj.tof = True

    spectrum = to_numpy_array(proj(img))  # shape (num_tofbins,), bin j <-> bins[j]

    # ---- analytic reference: truncated, normalised Gaussian bin weights ----
    # the central voxel sits exactly at the LOR centre -> fractional bin it_f.
    it_f = (num_tofbins - 1) / 2.0
    ref = cf * np.asarray(
        [
            effective_tof_kernel(
                (it - it_f) * tofbin_width, sigma_tof, tofbin_width, num_sigmas
            )
            for it in bins
        ],
        dtype=np.float64,
    )

    # (a) every bin (inside and outside the window) matches the reference;
    #     single-precision projector -> float32-appropriate tolerance
    np.testing.assert_allclose(spectrum, ref, rtol=1e-4, atol=1e-6)

    # (a') bins entirely outside the +/- num_sigmas window get exactly zero
    outside = ref == 0.0
    assert outside.any()
    assert bool(np.all(spectrum[outside] == 0.0))

    # (b) the truncated kernel is normalised: the TOF spectrum sums to the
    #     non-TOF line integral (voxel size)
    assert math.isclose(float(spectrum.sum()), cf, rel_tol=1e-5)

    # (c) symmetric and peaked at the central TOF bin
    assert int(np.argmax(spectrum)) == (num_tofbins - 1) // 2
    np.testing.assert_allclose(spectrum, spectrum[::-1], atol=1e-5)


@pytest.mark.parametrize("sigma_tof, num_tofbins", [(2.0, 31), (3.0, 41)])
def test_tof_listmode_fwd_sum_equals_nontof_oblique(
    xp: ModuleType,
    dev: str,
    sigma_tof: float,
    num_tofbins: int,
    tofbin_width: float = 1.0,
    num_sigmas: float = 3.0,
    nvox: int = 11,
) -> None:
    """For a general (oblique) LOR, summing the TOF forward projection over all
    TOF bins must reproduce the non-TOF forward projection (the TOF weights sum
    to one).  A single central voxel keeps all TOF weight near the central bin
    (no edge truncation), so the equality is exact.  This also exercises the
    oblique intersection length (``cos(theta) != 1``).  Convention-independent."""

    voxel_size = (2.0, 2.0, 2.0)
    img_shape = (nvox, nvox, nvox)

    # single hot voxel at the image centre (odd nvox -> world coord (0,0,0))
    c = nvox // 2
    img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    img[c, c, c] = 1.0

    # an oblique LOR through the centre of the (centred) image
    start_row = [12.0, 8.0, 5.0]
    end_row = [-12.0, -8.0, -5.0]

    # non-TOF reference projection (single event)
    xstart1 = xp.asarray([start_row], dtype=xp.float32, device=dev)
    xend1 = xp.asarray([end_row], dtype=xp.float32, device=dev)
    nontof = float(
        to_numpy_array(
            ppp.ListmodePETProjector(xstart1, xend1, img_shape, voxel_size)(img)
        )[0]
    )

    # TOF projection: one event per (unsigned) bin -> full spectrum, then sum
    bins = list(range(num_tofbins))
    n_events = len(bins)
    xstart = xp.asarray([start_row] * n_events, dtype=xp.float32, device=dev)
    xend = xp.asarray([end_row] * n_events, dtype=xp.float32, device=dev)

    proj = ppp.ListmodePETProjector(xstart, xend, img_shape, voxel_size)
    proj.tof_parameters = ppt.TOFParameters(
        num_tofbins=num_tofbins,
        tofbin_width=tofbin_width,
        sigma_tof=sigma_tof,
        num_sigmas=num_sigmas,
    )
    proj.event_tofbins = xp.asarray(bins, dtype=xp.int16, device=dev)
    proj.tof = True

    tof_sum = float(to_numpy_array(proj(img)).sum())

    assert math.isclose(tof_sum, nontof, rel_tol=1e-5, abs_tol=1e-6)


# ---------------------------------------------------------------------------
# tabulated (non-Gaussian) kernels and per-LOR TOF parameters
# ---------------------------------------------------------------------------

import numpy as onp

import parallelproj.pet_lors as ppl
import parallelproj.pet_scanners as pps


def _asymmetric_kernel() -> ppt.TOFKernel:
    """asymmetric kernel: short linear rise for u < 0, long exponential tail for u > 0"""
    u = onp.linspace(-1.0, 3.0, 401)
    return ppt.TOFKernel.from_pdf(u, onp.where(u < 0, 1 + u, onp.exp(-u)))


def _single_voxel_lm(xp, dev, nb, tof_parameters, voxel_size=(2.2, 2.5, 2.7), nvox=19):
    """TOF spectrum (one event per TOF bin) of a single central voxel on an
    axis-aligned LOR from +60 to -60 mm (direction 0), per unit line integral"""
    shape = (nvox, 1, 1)
    img = xp.zeros(shape, dtype=xp.float32, device=dev)
    img[nvox // 2, 0, 0] = 1.0
    xs = xp.asarray([[60.0, 0.0, 0.0]] * nb, dtype=xp.float32, device=dev)
    xe = xp.asarray([[-60.0, 0.0, 0.0]] * nb, dtype=xp.float32, device=dev)
    proj = ppp.ListmodePETProjector(xs, xe, shape, voxel_size)
    proj.tof_parameters = tof_parameters
    proj.event_tofbins = xp.arange(nb, dtype=xp.int16, device=dev)
    proj.tof = True
    return to_numpy_array(proj(img)) / voxel_size[0]


@pytest.mark.parametrize("scale, offset", [(1.5, 0.0), (0.7, 1.1)])
def test_tof_listmode_asymmetric_kernel_single_voxel(
    xp: ModuleType, dev: str, scale: float, offset: float
) -> None:
    """An asymmetric tabulated kernel gives the TOF spectrum TOFKernel.weights
    predicts: u = TOF bin position - emission position along xstart -> xend,
    scaled by kernel_scale, with the bin grid shifted by tofcenter_offset."""
    nb, w = 21, 0.8
    k = _asymmetric_kernel()
    p = ppt.TOFParameters(
        num_tofbins=nb, tofbin_width=w, kernel=k, kernel_scale=scale, tofcenter_offset=offset
    )
    spec = _single_voxel_lm(xp, dev, nb, p)
    d = (onp.arange(nb) - (nb - 1) / 2) * w + offset
    ref = k.weights(d - w / 2, d + w / 2, scale)
    onp.testing.assert_allclose(spec, ref, atol=2e-6)
    assert math.isclose(float(spec.sum()), 1.0, rel_tol=1e-5)


def test_tof_kernel_cdf_normalisation_irrelevant(xp: ModuleType, dev: str) -> None:
    """The projectors normalise the kernel: a * F + b gives the same projection."""
    nb, w = 15, 1.0
    k = _asymmetric_kernel()
    k2 = ppt.TOFKernel(3.0 * k.cdf + 7.0, k.u_min, k.u_max)
    s1 = _single_voxel_lm(xp, dev, nb, ppt.TOFParameters(num_tofbins=nb, tofbin_width=w, kernel=k))
    s2 = _single_voxel_lm(xp, dev, nb, ppt.TOFParameters(num_tofbins=nb, tofbin_width=w, kernel=k2))
    onp.testing.assert_allclose(s1, s2, atol=1e-6)


def test_tof_gaussian_default_equals_explicit_table(xp: ModuleType, dev: str) -> None:
    """sigma_tof is the Gaussian table scaled by sigma_tof."""
    nb, w, sig = 17, 0.9, 1.7
    s1 = _single_voxel_lm(xp, dev, nb, ppt.TOFParameters(num_tofbins=nb, tofbin_width=w, sigma_tof=sig))
    s2 = _single_voxel_lm(
        xp, dev, nb,
        ppt.TOFParameters(
            num_tofbins=nb, tofbin_width=w, kernel=ppt.TOFKernel.gaussian(3.0), kernel_scale=sig
        ),
    )
    onp.testing.assert_array_equal(s1, s2)


def test_tof_one_element_array_is_scalar(xp: ModuleType, dev: str) -> None:
    """1-element arrays (any namespace) for sigma / offset act as scalars."""
    nb, w = 11, 1.0
    s1 = _single_voxel_lm(
        xp, dev, nb,
        ppt.TOFParameters(num_tofbins=nb, tofbin_width=w, sigma_tof=1.7, tofcenter_offset=0.4),
    )
    s2 = _single_voxel_lm(
        xp, dev, nb,
        ppt.TOFParameters(
            num_tofbins=nb,
            tofbin_width=w,
            sigma_tof=xp.asarray([1.7], dtype=xp.float32, device=dev),
            tofcenter_offset=onp.asarray([0.4], dtype=onp.float32),
        ),
    )
    onp.testing.assert_array_equal(s1, s2)


def test_tof_show_tof_bins_per_lor_offset(xp: ModuleType, dev: str) -> None:
    """show_tof_bins supports only a scalar tofcenter_offset."""
    proj = _small_polygon_projector(xp, dev)
    tp = ppt.TOFParameters(
        num_tofbins=5,
        tofbin_width=10.0,
        sigma_tof=8.0,
        tofcenter_offset=onp.zeros(proj.lor_descriptor.spatial_sinogram_shape, onp.float32),
    )
    with pytest.raises(NotImplementedError, match="scalar tofcenter_offset"):
        proj.lor_descriptor.show_tof_bins(None, tp)


def test_tof_listmode_per_event_parameters(xp: ModuleType, dev: str) -> None:
    """Per-event sigma / offset give the same projections (forward and back) as
    projecting each group of events with the corresponding scalar parameters."""
    rng = onp.random.default_rng(1)
    nb, w = 11, 6.0
    shape, vs = (12, 13, 5), (4.0, 4.0, 4.0)
    img_np = rng.uniform(0, 1, shape).astype(onp.float32)
    n = 40
    v = rng.normal(size=(n, 3)); v[:, 2] *= 0.1
    xs_np = (60 * v / onp.linalg.norm(v, axis=1, keepdims=True)).astype(onp.float32)
    v = rng.normal(size=(n, 3)); v[:, 2] *= 0.1
    xe_np = (60 * v / onp.linalg.norm(v, axis=1, keepdims=True)).astype(onp.float32)
    tb_np = rng.integers(0, nb, n).astype(onp.int16)
    groups = [(slice(0, 15), 5.0, 0.0), (slice(15, 30), 9.0, 2.5), (slice(30, 40), 3.0, -4.0)]
    sig_np = onp.concatenate([onp.full(g.stop - g.start, s) for g, s, _ in groups]).astype(onp.float32)
    off_np = onp.concatenate([onp.full(g.stop - g.start, o) for g, _, o in groups]).astype(onp.float32)

    img = xp.asarray(img_np, device=dev)
    y_np = rng.uniform(0, 1, n).astype(onp.float32)

    proj = ppp.ListmodePETProjector(
        xp.asarray(xs_np, device=dev), xp.asarray(xe_np, device=dev), shape, vs
    )
    proj.tof_parameters = ppt.TOFParameters(
        num_tofbins=nb,
        tofbin_width=w,
        sigma_tof=xp.asarray(sig_np, device=dev),
        tofcenter_offset=xp.asarray(off_np, device=dev),
    )
    proj.event_tofbins = xp.asarray(tb_np, device=dev)
    proj.tof = True
    fwd = to_numpy_array(proj(img))
    back = to_numpy_array(proj.adjoint(xp.asarray(y_np, device=dev)))

    back_ref = onp.zeros(shape, onp.float32)
    for g, s, o in groups:
        pg = ppp.ListmodePETProjector(
            xp.asarray(xs_np[g], device=dev), xp.asarray(xe_np[g], device=dev), shape, vs
        )
        pg.tof_parameters = ppt.TOFParameters(
            num_tofbins=nb, tofbin_width=w, sigma_tof=s, tofcenter_offset=o
        )
        pg.event_tofbins = xp.asarray(tb_np[g], device=dev)
        pg.tof = True
        onp.testing.assert_allclose(fwd[g], to_numpy_array(pg(img)), rtol=1e-5, atol=1e-6)
        back_ref += to_numpy_array(pg.adjoint(xp.asarray(y_np[g], device=dev)))
    onp.testing.assert_allclose(back, back_ref, rtol=1e-5, atol=1e-5)

    # wrong per-event shape
    proj.tof_parameters = ppt.TOFParameters(
        num_tofbins=nb, tofbin_width=w, sigma_tof=xp.asarray(sig_np[:-1], device=dev)
    )
    proj.tof = True
    with pytest.raises(ValueError, match="per-LOR"):
        proj(img)


def _small_polygon_projector(xp, dev, views=None):
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings=2, symmetry_axis=2)
    lor_desc = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, max_ring_difference=1, span=1),
        radial_trim=260,
    )
    return ppp.RegularPolygonPETProjector(
        lor_desc, img_shape=(20, 20, 2), voxel_size=(4.0, 4.0, 4.0), views=views
    )


def test_tof_sinogram_per_lor_parameters_views(xp: ModuleType, dev: str) -> None:
    """Per-LOR offset / kernel scale (full spatial sinogram shape) with a views
    subset: the projector selects its views; the result equals the full
    projection restricted to those views.  Per-LOR arrays with all values
    equal give the scalar result, and the TOF projector stays adjoint."""
    rng = onp.random.default_rng(2)
    full = _small_polygon_projector(xp, dev)
    desc = full.lor_descriptor
    sino_shape = desc.spatial_sinogram_shape
    nb = 7
    k = _asymmetric_kernel()
    scale_np = rng.uniform(4.0, 8.0, sino_shape).astype(onp.float32)
    off_np = rng.uniform(-5.0, 5.0, sino_shape).astype(onp.float32)
    p = ppt.TOFParameters(
        num_tofbins=nb,
        tofbin_width=12.0,
        kernel=k,
        kernel_scale=xp.asarray(scale_np, device=dev),
        tofcenter_offset=xp.asarray(off_np, device=dev),
    )
    full.tof_parameters = p
    full.tof = True
    img = xp.asarray(rng.uniform(0, 1, full.in_shape).astype(onp.float32), device=dev)
    y_full = to_numpy_array(full(img))

    views = xp.asarray([1, 4, 5], device=dev)
    sub = _small_polygon_projector(xp, dev, views=views)
    sub.tof_parameters = p
    sub.tof = True
    y_sub = to_numpy_array(sub(img))
    onp.testing.assert_allclose(
        y_sub, onp.take(y_full, [1, 4, 5], axis=desc.view_axis_num), rtol=1e-5, atol=1e-6
    )
    assert sub.adjointness_test(xp, dev, dtype=xp.float32)

    # all values equal -> same as scalars
    p_const = ppt.TOFParameters(
        num_tofbins=nb, tofbin_width=12.0, kernel=k,
        kernel_scale=xp.full(sino_shape, 6.0, dtype=xp.float32, device=dev),
        tofcenter_offset=xp.full(sino_shape, 1.5, dtype=xp.float32, device=dev),
    )
    p_scalar = ppt.TOFParameters(
        num_tofbins=nb, tofbin_width=12.0, kernel=k, kernel_scale=6.0, tofcenter_offset=1.5
    )
    full.tof_parameters = p_const
    a = to_numpy_array(full(img))
    full.tof_parameters = p_scalar
    onp.testing.assert_allclose(a, to_numpy_array(full(img)), rtol=1e-6, atol=1e-7)

    # per-LOR arrays must have the full spatial sinogram shape
    sub.tof_parameters = ppt.TOFParameters(
        num_tofbins=nb,
        tofbin_width=12.0,
        sigma_tof=xp.ones(sub.out_shape[:-1], dtype=xp.float32, device=dev),
    )
    with pytest.raises(ValueError, match="per-LOR"):
        sub(img)


def test_tof_equalblock_per_lor_chunks(xp: ModuleType, dev: str) -> None:
    """EqualBlockPETProjector: per-LOR values (num_block_pairs,
    num_lors_per_block_pair) give the same result with 1 and 2 chunks, and the
    projector stays adjoint."""
    rng = onp.random.default_rng(3)
    mods = []
    for r in (10.0, -10.0):
        aff = xp.eye(4, device=dev, dtype=xp.float32)
        aff[1, -1] = r
        mods.append(pps.BlockPETScannerModule(xp, dev, (2, 2, 2), (4.0, 3.0, 2.0), affine_transformation_matrix=aff))
    aff = xp.eye(4, device=dev, dtype=xp.float32)
    aff[0, -1] = 12.0
    mods.append(pps.BlockPETScannerModule(xp, dev, (2, 2, 2), (4.0, 3.0, 2.0), affine_transformation_matrix=aff))
    scanner = pps.ModularizedPETScannerGeometry(mods)
    desc = ppl.EqualBlockPETLORDescriptor(scanner, xp.asarray([[0, 1], [0, 2], [1, 2]], device=dev))
    shape = (desc.num_block_pairs, desc.num_lors_per_block_pair)
    p = ppt.TOFParameters(
        num_tofbins=5,
        tofbin_width=3.0,
        sigma_tof=xp.asarray(rng.uniform(2.0, 4.0, shape).astype(onp.float32), device=dev),
        tofcenter_offset=xp.asarray(rng.uniform(-2.0, 2.0, shape).astype(onp.float32), device=dev),
    )
    img = xp.asarray(rng.uniform(0, 1, (14, 10, 3)).astype(onp.float32), device=dev)
    res = []
    for chunks in (1, 2):
        proj = ppp.EqualBlockPETProjector(desc, (14, 10, 3), (1.0, 1.0, 1.0), num_chunks=chunks)
        proj.tof_parameters = p
        proj.tof = True
        res.append(to_numpy_array(proj(img)))
        assert proj.adjointness_test(xp, dev, dtype=xp.float32)
    onp.testing.assert_allclose(res[0], res[1], rtol=1e-6, atol=1e-7)
