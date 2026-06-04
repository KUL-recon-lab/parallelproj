"""Time-of-flight (TOF) parameter dataclass for PET scanners.

Defines the bin width, Gaussian kernel width, truncation threshold, and
centre offset that parameterise the TOF response model used by the projectors
in :mod:`parallelproj.projectors`.
"""

from __future__ import annotations

from dataclasses import dataclass

#: Speed of light in mm per nanosecond.
C_MM_PER_NS: float = 299.792458


@dataclass
class TOFParameters:
    """TOF kernel parameters for a PET scanner.

    Defaults correspond to a scanner with 385 ps FWHM TOF resolution.
    Fields: ``num_tofbins`` (int), ``tofbin_width`` (mm), ``sigma_tof`` (mm),
    ``num_sigmas``, ``tofcenter_offset`` (mm).
    """

    num_tofbins: int = 29
    # 13 TOF "small" TOF bins of 0.01302[ns] * (speed of light / 2) [mm/ns]
    tofbin_width: float = 13 * 0.01302 * C_MM_PER_NS / 2
    sigma_tof: float = (C_MM_PER_NS / 2) * (
        0.385 / 2.355
    )  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355
    num_sigmas: float = 3.0
    tofcenter_offset: float = 0
