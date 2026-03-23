"""PET time-of-flight (TOF) related classes and functions"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TOFParameters:
    """
    generic time of flight (TOF) parameters for a scanner with 385ps FWHM TOF

    num_tofbins: int
        number of time of flight bins
    tofbin_width: float
        width of the TOF bin in spatial units (mm)
    sigma_tof: float
        standard deviation of Gaussian TOF kernel in spatial units (mm)
    num_sigmas: float
        number of sigmas after which TOF kernel is truncated
    tofcenter_offset: float
        offset of center of central TOF bin from LOR center in spatial units (mm)

    Examples
    --------
    .. minigallery:: parallelproj.tof.TOFParameters
    """

    num_tofbins: int = 29
    # 13 TOF "small" TOF bins of 0.01302[ns] * (speed of light / 2) [mm/ns]
    tofbin_width: float = 13 * 0.01302 * 299.792 / 2
    sigma_tof: float = (299.792 / 2) * (
        0.385 / 2.355
    )  # (speed_of_light [mm/ns] / 2) * TOF FWHM [ns] / 2.355
    num_sigmas: float = 3.0
    tofcenter_offset: float = 0
