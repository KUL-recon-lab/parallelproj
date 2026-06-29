"""Tests for :class:`parallelproj.tof.TOFParameters` (validation)."""

from __future__ import annotations

import math

import pytest

from parallelproj.tof import TOFParameters


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
