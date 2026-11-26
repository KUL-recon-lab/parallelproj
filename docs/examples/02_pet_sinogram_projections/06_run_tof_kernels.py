"""
Non-Gaussian TOF kernels and per-LOR TOF parameters
===================================================

The TOF projectors model the uncertainty of the TOF measurement with a *TOF
kernel*: the probability density of the measured TOF position around the true
emission position along a LOR.  By default this is a Gaussian with standard
deviation ``sigma_tof``, truncated at ``num_sigmas`` sigma.  Real scanners can
have non-Gaussian kernels, e.g. with a tail on one side.

parallelproj handles every TOF kernel as a table of its cumulative
distribution (:class:`.TOFKernel`).  This example shows how to

1. build a non-Gaussian kernel from samples of its density,
2. use it in a projector (:class:`.TOFParameters` with ``kernel``),
3. set the TOF centre offset (and the kernel scale) **per LOR**.

.. note::

    All lengths are in mm.  A time difference :math:`t` (ns) corresponds to
    :math:`t \\cdot c / 2` mm along the LOR
    (:data:`~parallelproj.tof.C_MM_PER_NS`).
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj._examples_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")
xp, dev = suggest_array_backend_and_device(None, None)

# %%
# Two TOF kernels with the same core
# ----------------------------------
#
# The Gaussian kernel of a scanner with 385 ps FWHM coincidence timing
# resolution has :math:`\sigma \approx 24.5` mm.  As a non-Gaussian example we
# add an exponential tail on the ``u > 0`` side to the same Gaussian core (15 %
# of the counts, decay length 40 mm).  :meth:`.TOFKernel.from_pdf` integrates
# samples of the density (which need not be normalised) to the table the
# projectors use.
#
# ``u`` is the position of a TOF bin minus the emission position, measured
# along the LOR from ``xstart`` to ``xend``: a tail at ``u > 0`` shifts counts
# towards ``xend``.

sigma = (0.385 / 2.3548) * (parallelproj.tof.C_MM_PER_NS / 2)  # ~24.5 mm
u = np.linspace(-3 * sigma, 8 * sigma, 2001)
core = np.exp(-0.5 * (u / sigma) ** 2)
tail_frac, tail_len = 0.15, 40.0
# smooth onset of the tail (Gaussian CDF) so that the density is continuous
tail = np.exp(-u / tail_len) * 0.5 * (1 + np.tanh(u / sigma))
pdf_tail = (1 - tail_frac) * core / core.sum() + tail_frac * tail / tail.sum()

kernel_gauss = parallelproj.tof.TOFKernel.gaussian(num_sigmas=3.0)  # unit sigma
kernel_tail = parallelproj.tof.TOFKernel.from_pdf(u, pdf_tail)  # in mm (scale 1)
print(kernel_gauss)
print(kernel_tail)

# %%
# The density of a kernel follows from its table by differentiation; for the
# Gaussian table, the positions are in units of sigma (``kernel_scale = sigma``).

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
for k, s, label, color in [
    (kernel_gauss, sigma, "Gaussian (sigma_tof = 24.5 mm, 3 sigma)", "#257de0"),
    (kernel_tail, 1.0, "Gaussian core + exponential tail", "#f47230"),
]:
    uk = k.u * s
    dens = np.gradient(k.cdf, uk) / (k.cdf[-1] - k.cdf[0])
    for a in ax:
        a.plot(uk, dens, color=color, lw=2, label=label)
ax[1].set_yscale("log")
ax[1].set_ylim(1e-5, 0.05)
for a in ax:
    a.set_xlabel("u = TOF position - emission position [mm]")
    a.set_ylabel("kernel density [1/mm]")
    a.grid(alpha=0.3)
ax[0].set_title("TOF kernels (linear)")
ax[1].set_title("TOF kernels (log)")
ax[0].legend(fontsize="small")
fig.show()

# %%
# TOF spectrum of a point source
# ------------------------------
#
# We project a single voxel (at the centre of a LOR along the x axis) with a
# listmode projector that has one event per TOF bin: the result is the TOF
# spectrum, i.e. the weights of the TOF bins.  A Gaussian kernel is given by
# ``sigma_tof``, any other kernel by ``kernel`` (and optionally
# ``kernel_scale``).  :meth:`.TOFKernel.weights` computes the same weights in
# NumPy (dots), e.g. for checks and plots.

num_tofbins, tofbin_width = 25, 12.0
img_shape, voxel_size = (41, 1, 1), (2.0, 2.0, 2.0)
img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
img[img_shape[0] // 2, 0, 0] = 1.0 / voxel_size[0]  # unit line integral


def tof_spectrum(tof_parameters, offsets=0.0):
    """TOF spectrum of the central voxel on a LOR from x = -200 to x = +200 mm"""
    n = tof_parameters.num_tofbins
    xstart = xp.asarray([[-200.0, 0.0, 0.0]] * n, dtype=xp.float32, device=dev)
    xend = xp.asarray([[200.0, 0.0, 0.0]] * n, dtype=xp.float32, device=dev)
    proj = parallelproj.projectors.ListmodePETProjector(
        xstart, xend, img_shape, voxel_size
    )
    proj.tof_parameters = tof_parameters
    proj.event_tofbins = xp.arange(n, dtype=xp.int16, device=dev)
    proj.tof = True
    return to_numpy_array(proj(img))


tof_gauss = parallelproj.tof.TOFParameters(
    num_tofbins=num_tofbins, tofbin_width=tofbin_width, sigma_tof=sigma
)
tof_tail = parallelproj.tof.TOFParameters(
    num_tofbins=num_tofbins, tofbin_width=tofbin_width, kernel=kernel_tail
)

# TOF bin centres relative to the emission position (the LOR centre)
d = (np.arange(num_tofbins) - (num_tofbins - 1) / 2) * tofbin_width

fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
for tp, k, s, label, color in [
    (tof_gauss, kernel_gauss, sigma, "Gaussian", "#257de0"),
    (tof_tail, kernel_tail, 1.0, "Gaussian core + tail", "#f47230"),
]:
    ax.step(d, tof_spectrum(tp), where="mid", color=color, lw=2, label=label)
    ax.plot(
        d,
        k.weights(d - tofbin_width / 2, d + tofbin_width / 2, s),
        ".",
        color="k",
        label="TOFKernel.weights (NumPy)" if k is kernel_tail else None,
    )
ax.set_xlabel("TOF bin centre - emission position [mm]")
ax.set_ylabel("TOF bin weight")
ax.set_title(f"TOF spectrum of a point source ({tofbin_width:g} mm TOF bins)")
ax.legend()
ax.grid(alpha=0.3)
fig.show()

# %%
# Per-LOR (per-event) TOF centre offsets
# --------------------------------------
#
# ``tofcenter_offset`` (and ``sigma_tof`` / ``kernel_scale``) can be given per
# LOR, e.g. from a timing calibration.  For a listmode projector, there is one
# value per event.  Here, all events of the spectrum get an offset of +30 mm:
# the TOF bin grid moves by 30 mm towards ``xend``, so the spectrum of the
# point source moves by -30 mm relative to the bins.  Scalars (the default)
# cost nothing extra; per-LOR arrays should live on the device of the
# projector.

offsets = xp.full(num_tofbins, 30.0, dtype=xp.float32, device=dev)
tof_tail_offset = parallelproj.tof.TOFParameters(
    num_tofbins=num_tofbins,
    tofbin_width=tofbin_width,
    kernel=kernel_tail,
    tofcenter_offset=offsets,
)

fig, ax = plt.subplots(figsize=(7, 3.8), layout="constrained")
ax.step(d, tof_spectrum(tof_tail), where="mid", color="#f47230", lw=2, label="offset 0")
ax.step(
    d, tof_spectrum(tof_tail_offset), where="mid", color="#00b894", lw=2,
    label="offset +30 mm (per event)",
)
ax.set_xlabel("(TOF bin - central bin) x bin width [mm]")
ax.set_ylabel("TOF bin weight")
ax.set_title("TOF centre offset per event")
ax.legend()
ax.grid(alpha=0.3)
fig.show()

# %%
# Sinogram projectors
# -------------------
#
# For the sinogram projectors, per-LOR values have the spatial sinogram shape
# of the LOR descriptor (all views); the projector selects its ``views``
# itself.  As an example, the timing calibration of the scanner below has an
# offset that grows linearly with the radial position of the LOR.

scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-4, 4, 3, device=dev),
    symmetry_axis=2,
)
lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=1, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)
sino_shape = lor_desc.spatial_sinogram_shape  # (radial, views, planes)
radial = np.linspace(-1, 1, sino_shape[0], dtype=np.float32)
offset_per_lor = np.ascontiguousarray(
    np.broadcast_to(5.0 * radial[:, None, None], sino_shape)
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=(40, 40, 7), voxel_size=(2.0, 2.0, 2.0)
)
# 45 TOF bins of 12 mm cover the LORs (~130 mm) plus the long tail of the
# kernel (up to ~200 mm), so no counts fall outside the TOF bins
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=45,
    tofbin_width=tofbin_width,
    kernel=kernel_tail,
    tofcenter_offset=xp.asarray(offset_per_lor, device=dev),
)
proj.tof = True
print(f"TOF sinogram shape: {proj.out_shape}")

x = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
x[20, 20, 3] = 1.0
x[10, 26, 3] = 1.0
y = proj(x)

# if the TOF bins cover the kernel support, the TOF projection summed over the
# TOF bins is the non-TOF projection, for any kernel (it is normalised) and
# any offset
proj_nontof = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=(40, 40, 7), voxel_size=(2.0, 2.0, 2.0)
)
print(
    "max |sum over TOF bins - non-TOF| =",
    float(xp.max(xp.abs(xp.sum(y, axis=-1) - proj_nontof(x)))),
)

# %%
# Notes
# -----
#
# - The projectors always use the tabulated kernel; ``sigma_tof`` is turned
#   into a 2001-point table of the Gaussian (:meth:`.TOFKernel.gaussian`),
#   whose projections agree with the closed-form Gaussian to float32
#   precision.
# - The kernel is normalised by the projectors: the table does not need to
#   run from 0 to 1.  The TOF bins only get weight inside the support
#   ``[u_min, u_max]`` of the table.
# - ``kernel_scale`` (scalar or per LOR) stretches a kernel around ``u = 0``:
#   :math:`F_s(u) = F(u / s)`, e.g. for LOR-dependent timing resolution.
