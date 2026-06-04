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

    Attributes
    ----------
    num_tofbins : int
        Number of TOF bins.
    tofbin_width : float
        Width of each TOF bin in mm.
    sigma_tof : float
        Standard deviation of the Gaussian TOF kernel in mm.
    num_sigmas : float
        Number of sigmas at which the TOF kernel is truncated.
    tofcenter_offset : float
        Offset of the centre of the central TOF bin from the LOR midpoint in mm.
    """

    num_tofbins: int = 29
    # 13 TOF "small" TOF bins of 0.01302[ns] * (speed of light / 2) [mm/ns]
    tofbin_width: float = 13 * 0.01302 * C_MM_PER_NS / 2
    sigma_tof: float = (C_MM_PER_NS / 2) * (
        0.385 / 2.355
    )  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355
    num_sigmas: float = 3.0
    tofcenter_offset: float = 0
