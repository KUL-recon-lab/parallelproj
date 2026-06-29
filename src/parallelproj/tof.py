"""Time-of-flight (TOF) parameter dataclass for PET scanners.

Defines the bin width, Gaussian kernel width, truncation threshold, and
centre offset that parameterise the TOF response model used by the projectors
in :mod:`parallelproj.projectors`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

#: Speed of light in mm per nanosecond.
C_MM_PER_NS: float = 299.792458


@dataclass
class TOFParameters:
    """TOF kernel parameters for a PET scanner.

    All spatial quantities are in **mm**.  To convert a timing quantity
    :math:`t` (in ns) to a spatial displacement along the LOR, use

    .. math::

        d \\;[\\text{mm}] = \\frac{c}{2} \\cdot t \\;[\\text{ns}]

    where :math:`c` = :data:`C_MM_PER_NS` mm/ns (= 299.792 mm/ns).
    The factor of 1/2 arises because both photons travel simultaneously
    toward the two detectors.

    Parameters
    ----------
    num_tofbins : int
        Number of TOF bins covering the full LOR.
    tofbin_width : float
        Width of each TOF bin in mm.
    sigma_tof : float
        Standard deviation of the Gaussian TOF kernel in mm.
    num_sigmas : float, optional
        Number of sigmas at which the Gaussian kernel is truncated. default ``3.0``.
    tofcenter_offset : float, optional
        Global shift of the TOF bin grid centre from the LOR midpoint in mm.
        This is a scalar approximation (identical for all LORs); default ``0``.
        A non-zero value is needed when systematic timing offsets are present,
        e.g. due to unequal cable lengths or electronics delays.

    Raises
    ------
    ValueError
        If ``num_tofbins`` is not a positive integer, if ``tofbin_width``,
        ``sigma_tof`` or ``num_sigmas`` is not strictly positive, or if any
        value is not finite.  A common cause is passing timing quantities in
        seconds/ps instead of the expected spatial **mm** (see the conversion
        above).

    Example
    -------
    A scanner with 385 ps FWHM TOF resolution and 13 fine bins of 0.01302 ns
    each grouped into one sinogram bin::

        # sigma_tof: FWHM [ns] / 2.355 * c/2 [mm/ns]
        sigma_tof = (385e-3 / 2.355) * (C_MM_PER_NS / 2)   # ~ 24.5 mm

        # tofbin_width: 13 fine bins * 0.01302 ns * c/2
        tofbin_width = 13 * 0.01302 * (C_MM_PER_NS / 2)    # ~ 25.4 mm

        p = TOFParameters(
            num_tofbins=29,
            tofbin_width=tofbin_width,
            sigma_tof=sigma_tof,
            num_sigmas=3.0,
        )
    """

    num_tofbins: int
    tofbin_width: float
    sigma_tof: float
    num_sigmas: float = 3.0
    tofcenter_offset: float = 0

    def __post_init__(self) -> None:
        if int(self.num_tofbins) != self.num_tofbins or self.num_tofbins < 1:
            raise ValueError(
                f"num_tofbins must be a positive integer, got {self.num_tofbins!r}"
            )
        if not (self.tofbin_width > 0):
            raise ValueError(
                f"tofbin_width must be > 0 (mm), got {self.tofbin_width!r}"
            )
        if not (self.sigma_tof > 0):
            raise ValueError(f"sigma_tof must be > 0 (mm), got {self.sigma_tof!r}")
        if not (self.num_sigmas > 0):
            raise ValueError(f"num_sigmas must be > 0, got {self.num_sigmas!r}")
        if not math.isfinite(self.tofcenter_offset):
            raise ValueError(
                f"tofcenter_offset must be finite (mm), got {self.tofcenter_offset!r}"
            )
