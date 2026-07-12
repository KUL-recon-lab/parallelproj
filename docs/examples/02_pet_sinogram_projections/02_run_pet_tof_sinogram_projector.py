"""
PET TOF sinogram projector
==========================

In this example we will show how to setup and use a TOF PET sinogram projector
consisting of a geometrical TOF forward projector (Joseph's method),
a resolution model and the modeling of attenuation.

.. note::

   To run this example locally, download
   `example_utils.py <https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/example_utils.py>`_
   into the **same folder** as this script. Make sure ``parallelproj`` is installed.
"""

# %%
import matplotlib.pyplot as plt
from example_utils import show_vol_cuts

import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
import parallelproj.operators
import parallelproj.tof
from parallelproj import to_numpy_array

# %%
from example_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)


# %%
# setup a small regular polygon PET scanner with 3 rings (polygons)

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

# %%
# setup the LOR descriptor that defines the sinogram

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=1, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

# %%
# Setting up the base projector (non-TOF initially)
# -------------------------------------------------
#
# :class:`.RegularPolygonPETProjector` combines the scanner, LOR and image
# geometry.  It starts in non-TOF mode; TOF is enabled in the next step by
# assigning ``tof_parameters``.  The image geometry is defined by the image
# shape, the voxel size, and the image origin.

# define a first projector using an image with 40x8x40 voxels of size 2x2x2 mm
# where the image center is at world coordinate (0, 0, 0)
proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=(40, 40, 7), voxel_size=(2.0, 2.0, 2.0)
)


# %%
# Visualize the scanner and image geometry
# ----------------------------------------
#
# :meth:`.RegularPolygonPETProjector.show_geometry` can be used
# to visualize the scanner and image geometry

fig = plt.figure(figsize=(8, 8))
ax1 = fig.add_subplot(111, projection="3d")
ax1.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")
proj.show_geometry(ax1)
fig.tight_layout()
fig.show()

# %%
# Adding an image-based resolution model
# --------------------------------------

# setup a simple image-based resolution model with an Gaussian FWHM of 4.5mm
res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[4.5 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

# %%
# Calculation of the non-TOF attenuation sinogram
# -----------------------------------------------

# setup an attenuation image containing the attenuation coeff. of water
# (in 1/mm)
x_att = xp.full(proj.in_shape, 0.01, device=dev, dtype=xp.float32)

# forward project the attenuation image
x_att_fwd = proj(x_att)

# calculate the attenuation sinogram
att_sino = xp.exp(-x_att_fwd)

# %%
# Adding time-of-flight to the projector
# --------------------------------------

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=9,
    tofbin_width=78.0,  # 9 bins x 78 mm covers the ~700 mm scanner diameter
    sigma_tof=24.4,  # 385 ps FWHM: (385e-3 / 2.355) * (C_MM_PER_NS / 2)
    num_sigmas=3.0,
)

# %%
# Combining resolution model, TOF projector and attenuation model
# ---------------------------------------------------------------
#
# The attenuation sinogram is non-TOF with shape = (159, 90, 7) while the projector
# output is a TOF sinogram with shape = (159, 90, 7, num_tofbins).
# We use broadcast_to to add a trailing singleton dimension to att_sino and broadcast
# it over the TOF-bins axis without copying data (zero-stride view).

print(f"atten. sino shape {att_sino.shape}")
print(f"proj output shape {proj.out_shape}")

att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
)

# setup a forward projector containing the attenuation and resolution
proj_with_att_and_res_model = parallelproj.operators.CompositeLinearOperator(
    (att_op, proj, res_model)
)


# %%
# setup a simple test image containing a few "hot rods"
# -----------------------------------------------------

# setup a simple test image containing a few "hot rods"
x = xp.zeros(proj.in_shape, device=dev, dtype=xp.float32)
x[proj.in_shape[0] // 2, :, proj.in_shape[2] // 2] = 1.0
x[4, 3:, proj.in_shape[2] // 2] = 1.0
x[proj.in_shape[0] // 2, :-3, 4] = 1.0


# %%
# forward and back project the image
# ----------------------------------

x_fwd = proj_with_att_and_res_model(x)
x_fwd_back = proj_with_att_and_res_model.adjoint(x_fwd)

# %%
# visualize the forward and the back projection
# ---------------------------------------------

# TOF sinogram shape is (r, v, p, t) -- transpose to (t, r, v, p) so the
# TOF-bin axis becomes the leading slider in the 4-D viewer
x_fwd_np = to_numpy_array(x_fwd).transpose(3, 0, 1, 2)

# %%
fig, _, widgets = show_vol_cuts(
    x_fwd_np, axis_labels=("t", "r", "v", "p"), fig_title="TOF sinogram"
)
fig.show()

# %%
# visualize the back projection
fig2, _, widgets2 = show_vol_cuts(
    to_numpy_array(x_fwd_back), fig_title="back projection"
)
fig2.show()

# %%
# LOR start / end convention and the TOF-bin direction
# ----------------------------------------------------
#
# A sinogram bin defines a line of response (LOR) between two detectors, but
# *which* of the two endpoints is the "start" (``xstart``) and which is the
# "end" (``xend``) is a convention.  For **non-TOF** projections it is
# irrelevant -- the line integral is the same in either direction.  For **TOF**
# it matters: the TOF-bin axis is defined *along* ``xstart -> xend``, so the
# first TOF bin sits near ``xstart``.  Different vendors adopt different
# start/end conventions, which flips where the first TOF bin lives.
#
# :class:`.LOREndpointOrder` on :class:`.RegularPolygonPETLORDescriptor` lets
# you choose: ``START_END`` (default) or ``END_START`` (endpoints swapped).
# Below we make the effect visible by back-projecting a sinogram that is zero
# everywhere except a few LORs (in several views) at an **off-center TOF bin**.
# The back-projected activity is localised near that TOF position along each
# LOR; swapping the endpoints reflects every blob to the opposite side (a
# point reflection through the scanner centre).
#
# For a clear picture we use a transaxial FOV that covers the bore and a finer
# TOF binning than above (so an off-center bin falls *inside* the image).

LOREndpointOrder = parallelproj.pet_lors.LOREndpointOrder


def _tof_descriptor(order):
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
demo_voxel_size = (1.6, 1.6, 2.0)  # ~128 mm transaxial FOV (covers the bore)
demo_tof = parallelproj.tof.TOFParameters(
    num_tofbins=9, tofbin_width=16.0, sigma_tof=6.0, num_sigmas=3.0
)

proj_se = parallelproj.projectors.RegularPolygonPETProjector(
    _tof_descriptor(LOREndpointOrder.START_END), demo_img_shape, demo_voxel_size
)
proj_es = parallelproj.projectors.RegularPolygonPETProjector(
    _tof_descriptor(LOREndpointOrder.END_START), demo_img_shape, demo_voxel_size
)
proj_se.tof_parameters = demo_tof
proj_es.tof_parameters = demo_tof

# sinogram: a few central-radial LORs across several views, all set at the
# same off-center TOF bin (bin 1 of 9; the central bin is bin 4)
y_tof = xp.zeros(proj_se.out_shape, device=dev, dtype=xp.float32)
r_center = proj_se.out_shape[0] // 2
plane_center = proj_se.out_shape[2] // 2
tof_bin = 1  # off-center
for view in (0, 20, 40, 60):
    for rad in (r_center - 1, r_center, r_center + 1):
        y_tof[rad, view, plane_center, tof_bin] = 1.0

# back project with both conventions and show the transaxial max-intensity
# projection (over the axial axis)
bp_se = to_numpy_array(proj_se.adjoint(y_tof)).max(axis=2)
bp_es = to_numpy_array(proj_es.adjoint(y_tof)).max(axis=2)

fig3, ax3 = plt.subplots(1, 2, figsize=(9, 4.6), tight_layout=True)
vmax = float(max(bp_se.max(), bp_es.max()))
for a, bp, title in (
    (ax3[0], bp_se, "LOREndpointOrder.START_END"),
    (ax3[1], bp_es, "LOREndpointOrder.END_START"),
):
    a.imshow(bp.T, origin="lower", cmap="Greys", vmax=vmax)
    a.axhline(demo_img_shape[1] / 2 - 0.5, color="w", lw=0.4, ls=":")
    a.axvline(demo_img_shape[0] / 2 - 0.5, color="w", lw=0.4, ls=":")
    a.set_title(title, fontsize="medium")
    a.set_xlabel("x0 voxel")
    a.set_ylabel("x1 voxel")
fig3.suptitle(
    "Back projection of a single off-center TOF bin: swapping the LOR\n"
    "start/end reflects the TOF-localised activity through the centre"
)
fig3.show()
