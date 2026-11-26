"""Tests for :class:`parallelproj.tof.TOFParameters` (validation)."""

from __future__ import annotations

import math

import pytest

from parallelproj.tof import (
    TOFParameters,
    C_MM_PER_NS,
    get_tof_parameters_G1,
    get_tof_parameters_G2,
)


def test_tofparameters_valid() -> None:
    """A well-formed parameter set is accepted and stores its fields."""
    p = TOFParameters(
        num_tofbins=29, tofbin_width=25.4, sigma_tof=24.5, num_sigmas=3.0
    )
    assert p.num_tofbins == 29
    assert p.tofbin_width == 25.4
    assert p.sigma_tof == 24.5
    assert p.num_sigmas == 3.0
    assert p.tofcenter_offset == 0

    # a negative center offset is allowed (systematic timing shift)
    TOFParameters(num_tofbins=1, tofbin_width=1.0, sigma_tof=1.0, tofcenter_offset=-2.5)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(num_tofbins=0, tofbin_width=2.0, sigma_tof=2.0),  # zero bins
        dict(num_tofbins=-3, tofbin_width=2.0, sigma_tof=2.0),  # negative bins
        dict(num_tofbins=10.5, tofbin_width=2.0, sigma_tof=2.0),  # non-integer bins
        dict(num_tofbins=10, tofbin_width=0.0, sigma_tof=2.0),  # zero bin width
        dict(num_tofbins=10, tofbin_width=-2.0, sigma_tof=2.0),  # negative bin width
        dict(num_tofbins=10, tofbin_width=2.0, sigma_tof=0.0),  # zero sigma
        dict(num_tofbins=10, tofbin_width=2.0, sigma_tof=-2.0),  # negative sigma
        dict(num_tofbins=10, tofbin_width=2.0, sigma_tof=2.0, num_sigmas=0.0),
        dict(num_tofbins=10, tofbin_width=2.0, sigma_tof=2.0, num_sigmas=-1.0),
        dict(num_tofbins=10, tofbin_width=math.nan, sigma_tof=2.0),  # NaN width
        dict(
            num_tofbins=10,
            tofbin_width=2.0,
            sigma_tof=2.0,
            tofcenter_offset=math.inf,
        ),  # non-finite offset
    ],
)
def test_tofparameters_invalid(kwargs: dict) -> None:
    """Out-of-range / nonsensical parameters raise ``ValueError``."""
    with pytest.raises(ValueError):
        TOFParameters(**kwargs)


def test_demo_tof_parameters_g_family() -> None:
    """The G1/G2 TOF-parameter getters build the expected defaults (G1 and G2
    differ only in num_tofbins) and stay fully overridable via keyword."""
    exp_w = 0.169 * 0.5 * C_MM_PER_NS
    exp_s = 0.385 * 0.5 * C_MM_PER_NS / 2.3548

    g1 = get_tof_parameters_G1()
    g2 = get_tof_parameters_G2()

    assert g1.num_tofbins == 27
    assert g2.num_tofbins == 29
    for p in (g1, g2):
        assert abs(p.tofbin_width - exp_w) < 1e-9
        assert abs(p.sigma_tof - exp_s) < 1e-9
        assert p.num_sigmas == 3.0
        assert p.tofcenter_offset == 0

    # keyword overrides (raw TOFParameters fields)
    assert get_tof_parameters_G1(num_tofbins=15).num_tofbins == 15
    assert get_tof_parameters_G2(num_sigmas=4.0).num_sigmas == 4.0
    assert get_tof_parameters_G1(sigma_tof=20.0).sigma_tof == 20.0
    assert get_tof_parameters_G2(tofbin_width=30.0).tofbin_width == 30.0
    assert get_tof_parameters_G2(tofcenter_offset=-3.0).tofcenter_offset == -3.0


# ---------------------------------------------------------------------------
# TOFKernel
# ---------------------------------------------------------------------------

import numpy as np
import scipy.special

from parallelproj.tof import TOFKernel


def test_tofkernel_gaussian() -> None:
    """The Gaussian table is the standard normal CDF on +/- num_sigmas, cached."""
    k = TOFKernel.gaussian(num_sigmas=2.5, num_points=101)
    assert k.num_points == 101
    assert (k.u_min, k.u_max) == (-2.5, 2.5)
    np.testing.assert_allclose(k.cdf, scipy.special.ndtr(np.linspace(-2.5, 2.5, 101)))
    assert TOFKernel.gaussian(2.5, 101) is k  # cached
    assert TOFKernel.gaussian().num_points == 2001
    assert not k.cdf.flags.writeable


def test_tofkernel_weights_truncated_gaussian() -> None:
    """weights() integrates the truncated, normalised kernel over intervals."""
    k = TOFKernel.gaussian(num_sigmas=3.0)
    sigma, w = 2.0, 0.7
    edges = np.arange(-8.4, 8.41, w)
    wts = k.weights(edges[:-1], edges[1:], scale=sigma)
    assert math.isclose(wts.sum(), 1.0, rel_tol=1e-12)
    phi = scipy.special.ndtr
    lo = np.clip(edges[:-1], -3 * sigma, 3 * sigma)
    hi = np.clip(edges[1:], -3 * sigma, 3 * sigma)
    ref = (phi(hi / sigma) - phi(lo / sigma)) / (phi(3.0) - phi(-3.0))
    np.testing.assert_allclose(wts, ref, atol=1e-6)  # linear interpolation of the 2001-point table


def test_tofkernel_from_pdf_and_cdf() -> None:
    """from_pdf (unnormalised, non-equidistant samples) and from_cdf build the
    same kernel as the analytic Gaussian table (up to the quadrature error)."""
    u = np.unique(np.concatenate([np.linspace(-3, 3, 3001), [0.1234, -1.7071]]))
    k_pdf = TOFKernel.from_pdf(u, 5.0 * np.exp(-0.5 * u**2))
    k_ref = TOFKernel.gaussian(3.0)
    assert k_pdf.num_points == 2001 and (k_pdf.u_min, k_pdf.u_max) == (-3.0, 3.0)
    e = np.linspace(-3, 3, 13)
    np.testing.assert_allclose(k_pdf.weights(e[:-1], e[1:]), k_ref.weights(e[:-1], e[1:]), atol=1e-6)

    # equidistant CDF samples are used as they are, others are resampled
    ue = np.linspace(-1.0, 2.0, 7)
    assert TOFKernel.from_cdf(ue, np.linspace(0, 1, 7)).num_points == 7
    assert TOFKernel.from_cdf(ue, np.linspace(0, 1, 7), num_points=11).num_points == 11
    un = np.array([-1.0, 0.0, 0.5, 2.0])
    assert TOFKernel.from_cdf(un, np.array([0.0, 0.3, 0.6, 1.0])).num_points == 2001


@pytest.mark.parametrize(
    "args",
    [
        (np.array([0.0, 1.0, 0.5]), -1.0, 1.0),  # decreasing
        (np.array([0.5, 0.5, 0.5]), -1.0, 1.0),  # constant
        (np.array([0.0, 1.0]), 1.0, 1.0),  # u_min == u_max
        (np.array([0.0, np.nan]), -1.0, 1.0),  # nan
        (np.array([0.0]), -1.0, 1.0),  # one point
    ],
)
def test_tofkernel_invalid(args) -> None:
    with pytest.raises(ValueError):
        TOFKernel(*args)


def test_tofkernel_from_pdf_invalid() -> None:
    with pytest.raises(ValueError):
        TOFKernel.from_pdf(np.array([0.0, 1.0, 2.0]), np.array([1.0, -1.0, 1.0]))
    with pytest.raises(ValueError):
        TOFKernel.from_pdf(np.array([0.0, 2.0, 1.0]), np.array([1.0, 1.0, 1.0]))


# ---------------------------------------------------------------------------
# TOFParameters: kernel, kernel_scale, per-LOR values
# ---------------------------------------------------------------------------


def test_tofparameters_kernel_and_scale() -> None:
    """Gaussian default vs. tabulated kernel: tof_kernel / scale / is_lor_dependent."""
    p = TOFParameters(num_tofbins=5, tofbin_width=2.0, sigma_tof=3.0, num_sigmas=4.0)
    assert p.tof_kernel is TOFKernel.gaussian(4.0)
    assert p.scale == 3.0
    assert not p.is_lor_dependent

    k = TOFKernel.from_cdf(np.linspace(-1, 2, 4), np.array([0.0, 0.5, 0.9, 1.0]))
    p2 = TOFParameters(num_tofbins=5, tofbin_width=2.0, kernel=k, kernel_scale=1.5)
    assert p2.tof_kernel is k and p2.scale == 1.5

    p3 = TOFParameters(
        num_tofbins=5, tofbin_width=2.0, sigma_tof=np.array([1.0, 2.0]), tofcenter_offset=np.zeros(2)
    )
    assert p3.is_lor_dependent
    # 1-element arrays count as scalars
    assert not TOFParameters(num_tofbins=5, tofbin_width=2.0, sigma_tof=np.array([2.0])).is_lor_dependent


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(),  # neither sigma_tof nor kernel
        dict(sigma_tof=2.0, kernel=TOFKernel.gaussian()),  # both
        dict(kernel=TOFKernel.gaussian(), kernel_scale=0.0),
        dict(kernel=TOFKernel.gaussian(), kernel_scale=np.array([1.0, -1.0])),
        dict(sigma_tof=np.array([1.0, np.nan])),
        dict(sigma_tof=np.array([1.0, 0.0])),
        dict(sigma_tof=2.0, tofcenter_offset=np.array([0.0, np.inf])),
    ],
)
def test_tofparameters_kernel_invalid(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        TOFParameters(num_tofbins=5, tofbin_width=2.0, **kwargs)


def test_tofparameters_kernel_type() -> None:
    with pytest.raises(TypeError):
        TOFParameters(num_tofbins=5, tofbin_width=2.0, kernel="gauss")


def test_tofkernel_more_invalid_and_repr() -> None:
    """argument checks of from_pdf / from_cdf / gaussian, and the repr"""
    with pytest.raises(ValueError):
        TOFKernel.from_pdf(np.zeros(3), np.zeros(4))  # shape mismatch
    with pytest.raises(ValueError):
        TOFKernel.from_cdf(np.zeros((2, 2)), np.zeros((2, 2)))  # not 1D
    with pytest.raises(ValueError):
        TOFKernel.from_cdf(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.5, 1.0]))
    with pytest.raises(ValueError):
        TOFKernel.gaussian(num_sigmas=0.0)
    with pytest.raises(ValueError):
        TOFKernel.gaussian(num_sigmas=3.0, num_points=1)
    assert repr(TOFKernel.gaussian(2.0, 5)) == "TOFKernel(num_points=5, u_min=-2, u_max=2)"
