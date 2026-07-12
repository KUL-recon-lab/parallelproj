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
