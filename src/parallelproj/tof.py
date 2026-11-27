"""Time-of-flight (TOF) parameters and TOF kernels for PET scanners.

Defines the bin width, Gaussian kernel width, truncation threshold, and
centre offset that parameterise the TOF response model used by the projectors
in :mod:`parallelproj.projectors`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import scipy.special

from ._backend import to_numpy_array

#: Speed of light in mm per nanosecond.
C_MM_PER_NS: float = 299.792458


class TOFKernel:
    """Tabulated TOF kernel: the cumulative distribution of the TOF kernel.

    The kernel is given by its cumulative distribution function (CDF)
    :math:`F` sampled at ``num_points`` **equidistant** positions
    :math:`u_0 = u_\\text{min}, \\ldots, u_{n-1} = u_\\text{max}`, where
    :math:`u` is the position of a TOF bin minus the emission position,
    measured along the LOR in the ``xstart -> xend`` direction, in units of
    the kernel *scale* (see :attr:`TOFParameters.kernel_scale`).  For an
    asymmetric kernel, a long tail at :math:`u > 0` thus means counts
    shifted towards ``xend``.  The support of the kernel
    is :math:`[u_\\text{min}, u_\\text{max}]`; between the table points
    :math:`F` is interpolated linearly.  The projectors normalise the kernel
    to unit total weight, so :math:`F` does not need to run from 0 to 1, only
    increase (it must not decrease, and :math:`F(u_\\text{max}) >
    F(u_\\text{min})`).  Asymmetric and non-Gaussian kernels are allowed.

    The weight of a TOF bin is the kernel integrated over the part of the bin
    inside the support, i.e. a difference of two (interpolated) CDF values.
    This is the form the projectors of ``parallelproj_core`` (>= 3.0) use for
    all TOF kernels; a Gaussian kernel is a table of the normal CDF, see
    :meth:`gaussian`.

    Use the constructors :meth:`gaussian`, :meth:`from_pdf` or
    :meth:`from_cdf` rather than the class directly.

    Parameters
    ----------
    cdf : array-like
        CDF values at ``num_points >= 2`` equidistant positions from
        ``u_min`` to ``u_max`` (non-decreasing, ``cdf[-1] > cdf[0]``).
    u_min, u_max : float
        First and last table position (``u_min < u_max``).
    """

    def __init__(self, cdf, u_min: float, u_max: float) -> None:
        cdf = np.asarray(cdf, dtype=np.float64)
        if cdf.ndim != 1 or cdf.shape[0] < 2:
            raise ValueError("cdf must be a 1D array with at least 2 values")
        if not np.all(np.isfinite(cdf)):
            raise ValueError("cdf must contain finite values")
        if np.any(np.diff(cdf) < 0):
            raise ValueError("cdf (the cumulative TOF kernel) must not decrease")
        if not cdf[-1] > cdf[0]:
            raise ValueError("cdf must increase: cdf[-1] > cdf[0]")
        if not (math.isfinite(u_min) and math.isfinite(u_max) and u_min < u_max):
            raise ValueError("u_min and u_max must be finite with u_min < u_max")
        self._cdf = cdf
        self._cdf.setflags(write=False)
        self._u_min = float(u_min)
        self._u_max = float(u_max)

    @property
    def cdf(self) -> np.ndarray:
        """CDF values at the (equidistant) table positions (read-only, float64)."""
        return self._cdf

    @property
    def u_min(self) -> float:
        """Start of the kernel support (first table position)."""
        return self._u_min

    @property
    def u_max(self) -> float:
        """End of the kernel support (last table position)."""
        return self._u_max

    @property
    def num_points(self) -> int:
        """Number of table points."""
        return int(self._cdf.shape[0])

    @property
    def u(self) -> np.ndarray:
        """The (equidistant) table positions."""
        return np.linspace(self._u_min, self._u_max, self.num_points)

    def weights(self, u_lo, u_hi, scale: float = 1.0) -> np.ndarray:
        """Normalised kernel weight of the intervals ``[u_lo, u_hi]``.

        Reference implementation (NumPy, float64) of the TOF weights: the
        kernel with scale ``scale`` (:math:`F_s(u) = F(u / s)`), integrated over
        the part of each interval inside the support and normalised to unit
        total weight.  Useful for plots and tests; the projectors compute the
        same in the backend.
        """
        lo = np.clip(np.asarray(u_lo, dtype=np.float64) / scale, self._u_min, self._u_max)
        hi = np.clip(np.asarray(u_hi, dtype=np.float64) / scale, self._u_min, self._u_max)
        u = self.u
        total = self._cdf[-1] - self._cdf[0]
        return np.maximum(np.interp(hi, u, self._cdf) - np.interp(lo, u, self._cdf), 0.0) / total

    @classmethod
    def gaussian(cls, num_sigmas: float = 3.0, num_points: int = 2001) -> "TOFKernel":
        """Gaussian kernel with unit standard deviation, truncated at
        ``+/- num_sigmas``.

        Use it with ``kernel_scale = sigma`` (the standard deviation in mm).
        This is what :class:`TOFParameters` uses for ``sigma_tof``.  With
        the default ``num_points = 2001``, the TOF projections differ from
        the closed-form (erf) Gaussian of ``parallelproj_core`` by < 1e-6
        (relative to the maximum), i.e. at float32 precision.
        """
        return _gaussian_kernel(float(num_sigmas), int(num_points))

    @classmethod
    def from_pdf(cls, u, pdf, num_points: int = 2001) -> "TOFKernel":
        """Kernel from samples of its density (PDF).

        ``u`` must be increasing, ``pdf >= 0``.  The PDF is integrated with the
        trapezoidal rule and resampled to ``num_points`` equidistant points
        over ``[u[0], u[-1]]``; it does not need to be normalised.
        """
        u = np.asarray(u, dtype=np.float64)
        pdf = np.asarray(pdf, dtype=np.float64)
        if u.ndim != 1 or u.shape != pdf.shape or u.shape[0] < 2:
            raise ValueError("u and pdf must be 1D arrays of the same length >= 2")
        if np.any(np.diff(u) <= 0):
            raise ValueError("u must be strictly increasing")
        if np.any(pdf < 0) or not np.all(np.isfinite(pdf)):
            raise ValueError("pdf must be finite and >= 0")
        cdf = np.concatenate([[0.0], np.cumsum(0.5 * (pdf[1:] + pdf[:-1]) * np.diff(u))])
        return cls.from_cdf(u, cdf, num_points=num_points)

    @classmethod
    def from_cdf(cls, u, cdf, num_points: int | None = None) -> "TOFKernel":
        """Kernel from samples of its CDF at increasing positions ``u``.

        If ``num_points`` is given (or ``u`` is not equidistant), the CDF is
        resampled (linear interpolation) to ``num_points`` (default: 2001)
        equidistant points over ``[u[0], u[-1]]``.
        """
        u = np.asarray(u, dtype=np.float64)
        cdf = np.asarray(cdf, dtype=np.float64)
        if u.ndim != 1 or u.shape != cdf.shape or u.shape[0] < 2:
            raise ValueError("u and cdf must be 1D arrays of the same length >= 2")
        if np.any(np.diff(u) <= 0):
            raise ValueError("u must be strictly increasing")
        equidistant = np.allclose(np.diff(u), (u[-1] - u[0]) / (u.shape[0] - 1), rtol=1e-9, atol=0)
        if num_points is None and equidistant:
            return cls(cdf, u[0], u[-1])
        n = 2001 if num_points is None else int(num_points)
        u_eq = np.linspace(u[0], u[-1], n)
        return cls(np.interp(u_eq, u, cdf), u[0], u[-1])

    def __repr__(self) -> str:
        return (
            f"TOFKernel(num_points={self.num_points}, u_min={self._u_min:g}, "
            f"u_max={self._u_max:g})"
        )


@functools.lru_cache(maxsize=32)
def _gaussian_kernel(num_sigmas: float, num_points: int) -> TOFKernel:
    if not (num_sigmas > 0 and math.isfinite(num_sigmas)):
        raise ValueError(f"num_sigmas must be > 0, got {num_sigmas!r}")
    if num_points < 2:
        raise ValueError(f"num_points must be >= 2, got {num_points!r}")
    u = np.linspace(-num_sigmas, num_sigmas, num_points)
    return TOFKernel(scipy.special.ndtr(u), -num_sigmas, num_sigmas)


def _is_scalar(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) or (
        hasattr(v, "shape") and tuple(v.shape) in ((), (1,))
    )


def _scalar_value(v) -> float:
    """float value of a scalar (Python number or 1-element array of any namespace)"""
    if hasattr(v, "shape"):
        return float(to_numpy_array(v).reshape(()))
    return float(v)


def _check_values(name: str, v, positive: bool) -> None:
    """finite (and > 0) check for scalars and arrays of any namespace"""
    if _is_scalar(v):
        x = _scalar_value(v)
        ok = math.isfinite(x) and (x > 0 if positive else True)
    else:
        a = to_numpy_array(v)
        ok = bool(np.all(np.isfinite(a)) and (np.all(a > 0) if positive else True))
    if not ok:
        cond = "> 0 and finite" if positive else "finite"
        raise ValueError(f"{name} must be {cond} (mm), got {v!r}")


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
    sigma_tof : float or Array, optional
        Standard deviation of the Gaussian TOF kernel in mm, scalar or one
        value per LOR (see Notes).  Required unless ``kernel`` is given.
    num_sigmas : float, optional
        Number of sigmas at which the Gaussian kernel is truncated. default ``3.0``.
    tofcenter_offset : float or Array, optional
        Shift of the TOF bin grid centre from the LOR midpoint in mm, scalar
        (the same for all LORs, default ``0``) or one value per LOR (see
        Notes).  A non-zero value is needed when systematic timing offsets
        are present, e.g. due to unequal cable lengths or electronics delays.
        The offset is measured along the ``xstart -> xend`` direction of each
        LOR, so if the LOR endpoint convention is reversed (see
        :class:`~parallelproj.pet_lors.LOREndpointOrder` ``END_START``) a
        non-zero offset must be negated to keep the same physical meaning.
    kernel : TOFKernel or None, optional
        Tabulated TOF kernel with unit scale, for non-Gaussian kernels.
        Mutually exclusive with ``sigma_tof``.  Default ``None`` (Gaussian
        from ``sigma_tof`` and ``num_sigmas``).
    kernel_scale : float or Array, optional
        Scale :math:`s > 0` of ``kernel`` in mm, scalar or one value per LOR:
        :math:`F_s(u) = F(u / s)`.  Default ``1.0``.  Only used with
        ``kernel``.

    Notes
    -----
    **TOF kernel.**  The projectors always use a tabulated kernel
    (:class:`TOFKernel`, see :attr:`tof_kernel` and :attr:`scale`).  By
    default it is a Gaussian with unit standard deviation truncated at
    ``num_sigmas`` (:meth:`TOFKernel.gaussian`, 2001 points), scaled by
    ``sigma_tof``.  Its projections agree with the closed-form Gaussian to
    float32 precision.

    **Per-LOR values.**  ``sigma_tof``, ``kernel_scale`` and
    ``tofcenter_offset`` can be scalars (the same for all LORs) or arrays
    with one value per LOR, with the shape of the LOR grid of the projector:

    - :class:`~parallelproj.projectors.RegularPolygonPETProjector`: the
      spatial sinogram shape of the LOR descriptor, for **all** views
      (``lor_descriptor.spatial_sinogram_shape``); the projector selects its
      ``views``,
    - :class:`~parallelproj.projectors.EqualBlockPETProjector`:
      ``(num_block_pairs, num_lors_per_block_pair)``,
    - :class:`~parallelproj.projectors.ListmodePETProjector`: ``(num_events,)``.

    Scalars cost nothing extra.  Per-LOR arrays are moved to the device of
    the projector in every call (no copy if they are already there).  The TOF
    unlisting helpers and ``show_tof_bins`` only support a scalar
    ``tofcenter_offset``.

    Raises
    ------
    ValueError
        If ``num_tofbins`` is not a positive integer, if ``tofbin_width``,
        ``sigma_tof``, ``kernel_scale`` or ``num_sigmas`` is not strictly
        positive, if any value is not finite, or if neither / both of
        ``sigma_tof`` and ``kernel`` are given.  A common cause is passing timing quantities in
        seconds/ps instead of the expected spatial **mm** (see the conversion
        above).

    Example
    -------
    A scanner with 385 ps FWHM TOF resolution and 13 fine bins of 0.01302 ns
    each grouped into one sinogram bin::

        # sigma_tof: FWHM [ns] / 2.3548 * c/2 [mm/ns]
        sigma_tof = (385e-3 / 2.3548) * (C_MM_PER_NS / 2)   # ~ 24.5 mm

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
    sigma_tof: Any = None
    num_sigmas: float = 3.0
    tofcenter_offset: Any = 0.0
    kernel: TOFKernel | None = None
    kernel_scale: Any = 1.0

    def __post_init__(self) -> None:
        if int(self.num_tofbins) != self.num_tofbins or self.num_tofbins < 1:
            raise ValueError(
                f"num_tofbins must be a positive integer, got {self.num_tofbins!r}"
            )
        if not (self.tofbin_width > 0):
            raise ValueError(
                f"tofbin_width must be > 0 (mm), got {self.tofbin_width!r}"
            )
        if not (self.num_sigmas > 0):
            raise ValueError(f"num_sigmas must be > 0, got {self.num_sigmas!r}")
        if self.kernel is None:
            if self.sigma_tof is None:
                raise ValueError("either sigma_tof or kernel must be given")
            _check_values("sigma_tof", self.sigma_tof, positive=True)
        else:
            if not isinstance(self.kernel, TOFKernel):
                raise TypeError("kernel must be a TOFKernel or None")
            if self.sigma_tof is not None:
                raise ValueError(
                    "sigma_tof and kernel are mutually exclusive (use kernel_scale "
                    "to scale a tabulated kernel)"
                )
            _check_values("kernel_scale", self.kernel_scale, positive=True)
        _check_values("tofcenter_offset", self.tofcenter_offset, positive=False)

    @property
    def tof_kernel(self) -> TOFKernel:
        """The tabulated (unit-scale) kernel the projectors use."""
        if self.kernel is not None:
            return self.kernel
        return TOFKernel.gaussian(self.num_sigmas)

    @property
    def scale(self) -> Any:
        """Scale of :attr:`tof_kernel` in mm (``sigma_tof`` for the Gaussian
        default, else ``kernel_scale``); scalar or per LOR."""
        return self.sigma_tof if self.kernel is None else self.kernel_scale

    @property
    def is_lor_dependent(self) -> bool:
        """``True`` if the scale or the centre offset is given per LOR."""
        return not (_is_scalar(self.scale) and _is_scalar(self.tofcenter_offset))


def get_tof_parameters_G1(**kwargs) -> TOFParameters:
    """Demo :class:`TOFParameters` **G1** (matches the G1 scanner).

    Defaults to 27 TOF bins with a 385 ps coincidence timing resolution
    (``sigma_tof``) and a 169 ps TOF-bin width, converted from ns to mm via
    ``x = t * C_MM_PER_NS / 2``.  Pair it with
    :func:`parallelproj.pet_lors.get_lor_descriptor_G1`.  Any
    :class:`TOFParameters` field (``num_tofbins``, ``tofbin_width``,
    ``sigma_tof``, ``num_sigmas``, ``tofcenter_offset``) can be overridden via
    keyword.
    """
    kwargs.setdefault("num_tofbins", 27)
    kwargs.setdefault("tofbin_width", 0.169 * 0.5 * C_MM_PER_NS)
    kwargs.setdefault("sigma_tof", 0.385 * 0.5 * C_MM_PER_NS / 2.3548)
    return TOFParameters(**kwargs)


def get_tof_parameters_G2(**kwargs) -> TOFParameters:
    """Demo :class:`TOFParameters` **G2** (matches the G2 scanner).

    Identical to :func:`get_tof_parameters_G1` but with 29 TOF bins by default.
    Pair it with :func:`parallelproj.pet_lors.get_lor_descriptor_G2`.  Any
    :class:`TOFParameters` field can be overridden via keyword.
    """
    kwargs.setdefault("num_tofbins", 29)
    kwargs.setdefault("tofbin_width", 0.169 * 0.5 * C_MM_PER_NS)
    kwargs.setdefault("sigma_tof", 0.385 * 0.5 * C_MM_PER_NS / 2.3548)
    return TOFParameters(**kwargs)
