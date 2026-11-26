"""
PET non-TOF sinogram projector
==============================

:bdg-primary:`projectors` :bdg-secondary:`beginner` :bdg-light:`CPU or GPU`

The forward model of PET maps an activity image to the expected data in the
sinogram bins.  In this example we build such a model step by step for a
non-TOF sinogram: a geometrical forward projector (line integrals computed
with Joseph's method), an image-based resolution model, and the attenuation of
the annihilation photons.  We also compute the corresponding back projections
(the adjoint operators), which every iterative reconstruction needs.

**You will learn**

- how to set up a :class:`.RegularPolygonPETProjector` from a scanner, a
  sinogram definition and an image geometry (shape, voxel size, origin),
- how to visualize the scanner and the image field of view,
- how to forward project an image and back project a sinogram with
  ``proj(x)`` and ``proj.adjoint(y)``,
- how to add a Gaussian resolution model and attenuation by composing linear
  operators with :class:`.CompositeLinearOperator`.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_01_run_regular_polygon_pet_scanner.py`
and :ref:`sphx_glr_auto_examples_01_pet_geometry_02_run_regular_polygon_pet_sino.py`
(scanner geometry and sinogram definition).
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
    # the scanner is only 20 mm long along x2: two ticks avoid overlapping labels
    ax.set_zticks([-10.0, 10.0])


# %%
# 1. Scanner and sinogram definition
# ----------------------------------
#
# We use a small regular polygon scanner with 12 sides, 15 LOR endpoints per
# side and 5 rings.  The :class:`.RegularPolygonPETLORDescriptor` defines the
# sinogram: all ring pairs with a ring difference of up to 2 (span 1), 10
# radial bins trimmed at each edge, and the axis order (radial, view, plane)
# (``RVP``).

num_rings = 5
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

# %%
# 2. Defining a non-TOF projector
# -------------------------------
#
# :class:`.RegularPolygonPETProjector` combines the scanner and the sinogram
# definition (both in the LOR descriptor) with the image geometry, given by the
# image shape, the voxel size (in mm) and the image origin.  The image origin
# is the world coordinate of the centre of voxel ``[0, 0, 0]``; by default it
# is chosen such that the image centre is at world coordinate (0, 0, 0).
#
# The first projector uses an image of 40 x 40 x 8 voxels of 2 x 2 x 2 mm,
# centred in the scanner.  The second uses an image of 20 x 30 x 8 voxels of
# 3 x 2 x 2 mm with an explicit origin, which shifts it off-centre.

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=(40, 40, 8), voxel_size=(2.0, 2.0, 2.0)
)

proj_off_centre = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc,
    img_shape=(20, 30, 8),
    voxel_size=(3.0, 2.0, 2.0),
    img_origin=(-19, -19, -7),
)

# %%
# 3. Scanner and image geometry
# -----------------------------
#
# :meth:`.RegularPolygonPETProjector.show_geometry` draws the LOR endpoints of
# the scanner and the image field of view (FOV) as a box.

fig, axes = plt.subplots(
    1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax in axes:
    view_from_front(ax)
proj.show_geometry(axes[0])
proj_off_centre.show_geometry(axes[1], color=(0, 0, 1))
axes[0].set_title("centred image, 40 x 40 x 8 voxels")
axes[1].set_title("off-centre image, 20 x 30 x 8 voxels")
fig.show()

# %%
# The centred image covers most of the bore.  The second image is smaller and
# shifted towards ``+x0`` and ``+x1`` (right and down in the front view).
# All projections below use the centred image.

# %%
# 4. A test image
# ---------------
#
# The test image contains two "hot rods" parallel to ``x1``, both in axial
# slice 4: one through the image centre and one near the left image border.

x = xp.zeros(proj.in_shape, device=dev, dtype=xp.float32)
x[proj.in_shape[0] // 2, :, proj.in_shape[2] // 2] = 1.0
x[4, :, proj.in_shape[2] // 2] = 1.0

fig, _, _ = show_vol_cuts(x, axis_labels=("x0", "x1", "x2"), fig_title="Test image")
fig.show()

# %%
# The transaxial slice (left) shows the two rods as vertical lines; the cut
# perpendicular to ``x1`` (bottom right) shows their cross-sections.

# %%
# 5. Geometrical forward and back projection
# ------------------------------------------
#
# Calling the projector, :meth:`.RegularPolygonPETProjector.__call__`,
# computes the geometrical forward projection :math:`P x`: the line integrals
# of the image along all LORs of the sinogram (Joseph's method).  The result
# has the shape ``proj.out_shape`` = (radial bins, views, planes).
#
# The geometrical back projection is the **transpose** :math:`P^T` of the
# forward projector.  Because the projectors operate on real-valued images and
# sinograms, the transpose coincides with the adjoint, so it is computed with
# the general :meth:`.RegularPolygonPETProjector.adjoint` method (equivalently
# the ``.H`` property, e.g. ``proj.H(x_fwd)``).

x_fwd = proj(x)
x_fwd_back = proj.adjoint(x_fwd)

# %%
# 6. Adding an image-based resolution model
# -----------------------------------------
#
# :class:`.GaussianFilterOperator` models the limited spatial resolution of
# the scanner as a Gaussian filter :math:`G` applied in image space, here with
# a FWHM of 4.5 mm (``sigma`` is given in voxels: FWHM / (2.35 voxel size)).
# :class:`.CompositeLinearOperator` chains operators (the last one in the tuple
# is applied first), giving the forward operator :math:`A = P G`.  Its
# transpose :math:`A^T = G^T P^T` is implemented by
# :meth:`.CompositeLinearOperator.adjoint`.

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[4.5 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

proj_with_res_model = parallelproj.operators.CompositeLinearOperator((proj, res_model))

x_fwd2 = proj_with_res_model(x)
x_fwd2_back = proj_with_res_model.adjoint(x_fwd2)

# %%
# 7. Adding the effect of attenuation
# -----------------------------------
#
# Attenuation reduces the number of detected coincidences along each LOR by the
# factor :math:`a = \exp(-\int \mu \, dl)`, the exponential of the negative
# line integral of the attenuation coefficients :math:`\mu`.  We obtain these
# line integrals by forward projecting an attenuation image (here water,
# :math:`\mu` = 0.01 / mm, everywhere in the image).  Attenuation is then an
# element-wise multiplication in the sinogram domain,
# :class:`.ElementwiseMultiplicationOperator`, and the complete forward
# operator becomes :math:`A = \text{diag}(a) P G`.

x_att = xp.full(proj.in_shape, 0.01, device=dev, dtype=xp.float32)
att_sino = xp.exp(-proj(x_att))
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_sino)

proj_with_att_and_res_model = parallelproj.operators.CompositeLinearOperator(
    (att_op, proj, res_model)
)

x_fwd3 = proj_with_att_and_res_model(x)
x_fwd3_back = proj_with_att_and_res_model.adjoint(x_fwd3)

# %%
# 8. Comparing the sinograms
# --------------------------
#
# The figure shows the three forward projections with a common, square-root
# colour scale (``PowerNorm``) that makes weak values visible.  Top: plane 2,
# the direct (ring difference 0) plane of the central ring at ``x2`` = 0, as a
# function of radial bin and view.  Bottom: the central radial bin (79) as a
# function of view and plane.

sinograms = {
    "geometrical, P x": x_fwd,
    "+ resolution model, P G x": x_fwd2,
    "+ attenuation, diag(a) P G x": x_fwd3,
}
plane, rad_bin = 2, proj.out_shape[0] // 2
norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=float(xp.max(x_fwd)))

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10, 6),
    sharex="row",
    sharey="row",
    height_ratios=(2.2, 1),
    layout="constrained",
)
for col, (title, sino) in enumerate(sinograms.items()):
    sino_np = to_numpy_array(sino)
    im_top = axes[0, col].imshow(
        sino_np[:, :, plane].T, cmap="magma", norm=norm, aspect="auto"
    )
    im_bottom = axes[1, col].imshow(
        sino_np[rad_bin, :, :].T,
        cmap="magma",
        norm=norm,
        aspect="auto",
        origin="lower",
    )
    axes[0, col].set_title(title)
    axes[0, col].set_xlabel("radial bin")
    axes[1, col].set_xlabel("view")
axes[0, 0].set_ylabel(f"view (plane {plane})")
axes[1, 0].set_ylabel(f"plane (radial bin {rad_bin})")
fig.colorbar(im_top, ax=axes[0, :], shrink=0.9)
fig.colorbar(im_bottom, ax=axes[1, :], shrink=0.9)
fig.show()

# %%
# The rods lie in the transaxial plane.  The LORs of plane 2 that run along a
# rod (around view 7, at radial bins 80 and about 50 for the central and the
# off-centre rod) integrate over its full length and give the two bright peaks;
# all other LORs cross a rod at most once and form the weak diamond-shaped
# bands.  In the bottom row, only the planes whose LORs cross the axial slice
# of the rods contain counts.  The resolution model blurs the sinograms and
# lowers the peak from 32.6 to 10.1; attenuation reduces all values further
# (by factors down to 0.33 for the longest paths through the water image),
# to a peak of 4.5.

# %%
# 9. Comparing the back projections
# ---------------------------------
#
# The back projections :math:`P^T P x`, :math:`(PG)^T P G x` and
# :math:`A^T A x` are blurry images of the rods.  The figure shows their
# transaxial slice 4 (the slice of the rods), each with its own colour scale.

back_projections = {
    "P^T P x": x_fwd_back,
    "(P G)^T P G x": x_fwd2_back,
    "A^T A x (with attenuation)": x_fwd3_back,
}
fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), layout="constrained")
for ax, (title, bp) in zip(axes, back_projections.items()):
    im = ax.imshow(to_numpy_array(bp[:, :, proj.in_shape[2] // 2]).T, cmap="Greys")
    ax.set_title(title)
    ax.set_xlabel("x0 voxel")
    fig.colorbar(im, ax=ax, shrink=0.8)
axes[0].set_ylabel("x1 voxel")
fig.show()

# %%
# The back projection without resolution model shows the two rods as narrow
# lines on a smooth background; the resolution model widens them.  With
# attenuation, the values are about 4 to 5 times lower, and the central rod is
# weakened more than the rod near the border (ratio of the rod values 1.35
# instead of 1.09): the attenuation factors are applied twice (in :math:`A`
# and in :math:`A^T`), and LORs through the image centre are attenuated most.
#
# All three cuts of the back projection including the attenuation and
# resolution model:

fig, _, _ = show_vol_cuts(
    x_fwd3_back,
    axis_labels=("x0", "x1", "x2"),
    fig_title="Back projection with attenuation and resolution model",
)
fig.show()

# %%
# Summary
# -------
#
# - :class:`.RegularPolygonPETProjector` combines a LOR descriptor with an
#   image geometry (shape, voxel size, origin); ``show_geometry`` draws both.
# - ``proj(x)`` computes the geometrical forward projection (Joseph's method)
#   and ``proj.adjoint(y)`` (or ``proj.H(y)``) its transpose, the back
#   projection.
# - A resolution model (:class:`.GaussianFilterOperator`) and attenuation
#   (:class:`.ElementwiseMultiplicationOperator` with the attenuation sinogram
#   :math:`\exp(-P\mu)`) are added with :class:`.CompositeLinearOperator`,
#   which also provides the adjoint of the complete model.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
#    -- the same forward model with time-of-flight (TOF) information.

# sphinx_gallery_thumbnail_number = 3
