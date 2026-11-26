"""
PET TOF sinogram projector
==========================

:bdg-primary:`projectors` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Time-of-flight (TOF) PET measures the difference of the arrival times of the
two annihilation photons, which localizes the emission along each line of
response (LOR) to within a few centimetres.  A TOF sinogram therefore has an
additional TOF-bin axis.  In this example we set up a TOF sinogram projector
together with a resolution model and attenuation, and show how the convention
for the start and end point of an LOR fixes the direction of the TOF bins.

**You will learn**

- how to turn a :class:`.RegularPolygonPETProjector` into a TOF projector by
  assigning :class:`.TOFParameters`,
- why the attenuation sinogram is computed with the non-TOF projector, and how
  to broadcast it over the TOF bins,
- how to combine resolution model, TOF projector and attenuation in a
  :class:`.CompositeLinearOperator`,
- how :class:`.LOREndpointOrder` reverses the direction of the TOF-bin axis.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_01_run_pet_non_tof_sinogram_projector.py`
(the non-TOF forward model).
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj._examples_utils import (
    show_vol_cuts,
    suggest_array_backend_and_device,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()


def view_from_front(ax) -> None:
    """3D view from in front of the scanner (looking along +x2, +x1 down)"""
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")
    # the scanner is only 8 mm long along x2: one tick avoids overlapping labels
    ax.set_zticks([0.0])


# %%
# 1. Scanner and sinogram definition
# ----------------------------------
#
# We use a small regular polygon scanner with 3 rings and a sinogram with all
# ring pairs with a ring difference of up to 1 (span 1), 10 radial bins
# trimmed at each edge, and the axis order (radial, view, plane) (``RVP``).

num_rings = 3
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-4, 4, num_rings, device=dev),
    symmetry_axis=2,
)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=1, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

# %%
# 2. Setting up the base projector (non-TOF initially)
# ----------------------------------------------------
#
# :class:`.RegularPolygonPETProjector` combines the scanner, LOR and image
# geometry.  It starts in non-TOF mode; TOF is enabled in section 5 by
# assigning ``tof_parameters``.  The image has 40 x 40 x 7 voxels of
# 2 x 2 x 2 mm, and its centre is at world coordinate (0, 0, 0).

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=(40, 40, 7), voxel_size=(2.0, 2.0, 2.0)
)

fig = plt.figure(figsize=(5.5, 5.5), layout="constrained")
ax = fig.add_subplot(projection="3d")
view_from_front(ax)
proj.show_geometry(ax)
ax.set_title("scanner and image field of view")
fig.show()

# %%
# The image field of view (box) lies inside the 3-ring scanner and covers
# most of its bore.

# %%
# 3. Image-based resolution model
# -------------------------------
#
# As in the non-TOF example, a Gaussian filter with a FWHM of 4.5 mm
# (``sigma`` in voxels) models the spatial resolution in image space.

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[4.5 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

# %%
# 4. Non-TOF attenuation sinogram
# -------------------------------
#
# The attenuation of a photon pair depends only on its LOR, not on where along
# the LOR the annihilation happened.  The attenuation factors
# :math:`\exp(-\int \mu \, dl)` are therefore the same for all TOF bins of an
# LOR, and we compute them with the projector while it is still in non-TOF
# mode, from an image of water (:math:`\mu` = 0.01 / mm).

x_att = xp.full(proj.in_shape, 0.01, device=dev, dtype=xp.float32)
att_sino = xp.exp(-proj(x_att))

# %%
# 5. Adding time-of-flight to the projector
# -----------------------------------------
#
# Assigning :class:`.TOFParameters` switches the projector to TOF mode.  The
# parameters define the number and width (in mm, along the LOR) of the TOF
# bins and the TOF resolution: ``sigma_tof`` is the standard deviation (in mm)
# of the Gaussian TOF kernel, which is truncated at ``num_sigmas`` standard
# deviations.  The values below are those of a clinical scanner: a 385 ps FWHM
# timing resolution gives ``sigma_tof`` = 24.4 mm, and 9 bins of 78 mm cover a
# clinical bore of about 700 mm.  For our small scanner with a bore of 130 mm,
# only the central TOF bins receive counts.

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=9,
    tofbin_width=78.0,
    sigma_tof=24.4,  # (385e-3 ns / 2.355) * (C_MM_PER_NS / 2)
    num_sigmas=3.0,
)

# %%
# 6. Combining resolution model, TOF projector and attenuation
# ------------------------------------------------------------
#
# The attenuation sinogram is non-TOF with shape (159, 90, 7), while the
# projector output is now a TOF sinogram with the TOF bins as last axis,
# shape (159, 90, 7, 9):

print(f"attenuation sinogram shape {att_sino.shape}")
print(f"TOF projector output shape {proj.out_shape}")

# %%
# ``expand_dims`` adds a trailing singleton axis to the attenuation sinogram,
# and ``broadcast_to`` repeats it over the TOF-bin axis without copying data
# (a zero-stride view).  The complete forward operator is
# :math:`A = \text{diag}(a) P_\text{TOF} G`.

att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
)

proj_with_att_and_res_model = parallelproj.operators.CompositeLinearOperator(
    (att_op, proj, res_model)
)

# %%
# 7. Forward and back projection of a test image
# ----------------------------------------------
#
# The test image contains three "hot rods" parallel to ``x1``: one through the
# image centre and one near the left image border in the central axial slice
# 3, and a shorter one through the image centre in slice 4.

x = xp.zeros(proj.in_shape, device=dev, dtype=xp.float32)
x[proj.in_shape[0] // 2, :, proj.in_shape[2] // 2] = 1.0
x[4, 3:, proj.in_shape[2] // 2] = 1.0
x[proj.in_shape[0] // 2, :-3, 4] = 1.0

x_fwd = proj_with_att_and_res_model(x)
x_fwd_back = proj_with_att_and_res_model.adjoint(x_fwd)

# %%
# The TOF sinogram has the shape (r, v, p, t).  ``show_vol_cuts`` shows a 4D
# array as cuts through the last three axes at the central index of the first
# axis, so we move the TOF-bin axis to the front and look at TOF bin 4, the
# central one.

x_fwd_np = to_numpy_array(x_fwd).transpose(3, 0, 1, 2)

fig, _, _ = show_vol_cuts(
    x_fwd_np,
    axis_labels=("t", "r", "v", "p"),
    fig_title="TOF sinogram",
    cmap="magma",
)
fig.show()

# %%
# The central TOF bin looks much like a non-TOF sinogram: the LORs that run
# along the rods (around view 7) give bright peaks.  The next figure compares
# plane 1 (the direct plane of the central ring) in the three central TOF bins
# with a common, square-root colour scale (``PowerNorm``).

plane = 1
norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(x_fwd_np.max()))
fig, axes = plt.subplots(
    1, 3, figsize=(10, 3.8), sharex=True, sharey=True, layout="constrained"
)
for ax, tof_bin in zip(axes, (3, 4, 5)):
    im = ax.imshow(
        x_fwd_np[tof_bin, :, :, plane].T,
        cmap="magma",
        norm=norm,
        aspect="auto",
    )
    ax.set_title(f"TOF bin {tof_bin}")
    ax.set_xlabel("radial bin")
axes[0].set_ylabel(f"view (plane {plane})")
fig.colorbar(im, ax=axes, shrink=0.9)
fig.show()

# %%
# TOF bin 4 covers the central 78 mm of every LOR and contains most counts.
# TOF bins 3 and 5 receive the activity more than 39 mm from the LOR centre,
# towards the start and the end point (blurred by the TOF kernel); they show
# only parts of the sinogram structures, and these parts differ between the
# two bins, which is the TOF information.  TOF bins 0 to 2 and 6 to 8 are
# (almost) empty in this small scanner.
#
# The back projection :math:`A^T A x` spreads each TOF bin only over its part
# of the LOR:

fig, _, _ = show_vol_cuts(
    x_fwd_back, axis_labels=("x0", "x1", "x2"), fig_title="TOF back projection"
)
fig.show()

# %%
# Compared with the non-TOF back projection of the previous example, the
# activity is concentrated more closely around the rods: the central rod now
# has the highest values, and the shorter rod in slice 4 is visible in the
# cut perpendicular to ``x1`` (bottom right).

# %%
# 8. LOR start / end convention and the TOF-bin direction
# -------------------------------------------------------
#
# A sinogram bin defines an LOR between two detectors, but *which* of the two
# endpoints is the "start" (``xstart``) and which is the "end" (``xend``) is a
# convention.  For **non-TOF** projections it is irrelevant -- the line
# integral is the same in either direction.  For **TOF** it matters: the
# TOF-bin axis is defined *along* ``xstart -> xend``, so the first TOF bin sits
# near ``xstart``.  Different vendors adopt different start/end conventions,
# which flips where the first TOF bin lives.
#
# :class:`.LOREndpointOrder` on :class:`.RegularPolygonPETLORDescriptor` lets
# you choose: ``START_END`` (default) or ``END_START`` (endpoints swapped).
# We make the effect visible by back projecting a sinogram that is zero
# everywhere except for a few LORs (3 central radial bins in 4 views) in an
# **off-centre TOF bin**.  For a clear picture we use a transaxial FOV that
# covers the bore and finer TOF bins than above (9 bins of 16 mm), so that the
# off-centre bin falls *inside* the image.

LOREndpointOrder = parallelproj.pet_lors.LOREndpointOrder


def tof_lor_descriptor(order):
    return parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
        scanner,
        parallelproj.pet_lors.Michelogram(
            scanner.num_rings, max_ring_difference=1, span=1
        ),
        radial_trim=10,
        sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
        lor_endpoint_order=order,
    )


demo_img_shape = (80, 80, 7)
demo_voxel_size = (1.6, 1.6, 2.0)  # 128 mm transaxial FOV (covers the bore)
demo_tof = parallelproj.tof.TOFParameters(
    num_tofbins=9, tofbin_width=16.0, sigma_tof=6.0, num_sigmas=3.0
)

proj_se = parallelproj.projectors.RegularPolygonPETProjector(
    tof_lor_descriptor(LOREndpointOrder.START_END), demo_img_shape, demo_voxel_size
)
proj_es = parallelproj.projectors.RegularPolygonPETProjector(
    tof_lor_descriptor(LOREndpointOrder.END_START), demo_img_shape, demo_voxel_size
)
proj_se.tof_parameters = demo_tof
proj_es.tof_parameters = demo_tof

# sinogram: 3 central radial bins in 4 views, all in the same off-centre TOF
# bin (bin 1 of 9; the central bin is bin 4)
y_tof = xp.zeros(proj_se.out_shape, device=dev, dtype=xp.float32)
r_center = proj_se.out_shape[0] // 2
plane_center = proj_se.out_shape[2] // 2
tof_bin = 1
for view in (0, 20, 40, 60):
    for rad in (r_center - 1, r_center, r_center + 1):
        y_tof[rad, view, plane_center, tof_bin] = 1.0

# back project with both conventions; maximum intensity projection over x2
bp_se = to_numpy_array(proj_se.adjoint(y_tof)).max(axis=2)
bp_es = to_numpy_array(proj_es.adjoint(y_tof)).max(axis=2)

half_fov = 0.5 * demo_img_shape[0] * demo_voxel_size[0]
extent = (-half_fov, half_fov, half_fov, -half_fov)  # x1 points down
vmax = float(max(bp_se.max(), bp_es.max()))

fig, axes = plt.subplots(
    1, 2, figsize=(9, 4.6), sharex=True, sharey=True, layout="constrained"
)
for ax, bp, title in (
    (axes[0], bp_se, "LOREndpointOrder.START_END"),
    (axes[1], bp_es, "LOREndpointOrder.END_START"),
):
    ax.imshow(bp.T, cmap="Greys", vmax=vmax, extent=extent)
    ax.axhline(0.0, lw=0.8, ls=":")
    ax.axvline(0.0, lw=0.8, ls=":")
    ax.set_title(title)
    ax.set_xlabel("x0 [mm]")
axes[0].set_ylabel("x1 [mm]")
fig.suptitle(f"Back projection of TOF bin {tof_bin} of {demo_tof.num_tofbins}")
fig.show()

# %%
# The back-projected activity is localized near TOF bin 1 along each of the 4
# LOR bundles, i.e. 3 bins (48 mm) from the LOR centre towards ``xstart``.
# Swapping the endpoints moves every blob to the opposite side of the scanner
# centre (a point reflection through the centre), because TOF bin ``k`` of one
# convention is TOF bin ``num_tofbins - 1 - k`` of the other.

# %%
# Summary
# -------
#
# - Assigning :class:`.TOFParameters` to ``tof_parameters`` turns a
#   :class:`.RegularPolygonPETProjector` into a TOF projector; its output gets
#   a trailing TOF-bin axis.
# - Attenuation factors do not depend on the TOF bin: compute them with the
#   non-TOF projector and broadcast them over the TOF axis with
#   ``expand_dims`` and ``broadcast_to``.
# - Resolution model, TOF projector and attenuation are combined with
#   :class:`.CompositeLinearOperator`, exactly as in the non-TOF case.
# - The TOF-bin axis runs from ``xstart`` to ``xend``; use
#   :class:`.LOREndpointOrder` to match the start/end convention of other
#   software.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_03_run_equalblock_projector.py`
#    -- non-TOF and TOF projections for scanners made of block modules.

# sphinx_gallery_thumbnail_number = 5
