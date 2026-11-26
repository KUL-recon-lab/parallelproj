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
