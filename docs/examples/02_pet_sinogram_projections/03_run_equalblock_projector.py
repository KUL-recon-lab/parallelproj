"""
Non-TOF and TOF projections for a modularized (block) PET scanner
=================================================================

:bdg-primary:`projectors` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Scanners built from identical detector blocks at arbitrary positions do not
have the cylindrical symmetry needed for a sinogram.  Their data are instead
organized by block pairs: for every pair of blocks in coincidence, all LORs
between the LOR endpoints of the two blocks.  In this example we set up an
:class:`.EqualBlockPETProjector` for such a scanner and compute non-TOF and
TOF forward and back projections.

**You will learn**

- how to build a :class:`.ModularizedPETScannerGeometry` from block modules
  placed with affine transformation matrices,
- how to define the LORs from a list of block pairs with an
  :class:`.EqualBlockPETLORDescriptor`,
- how to set up an :class:`.EqualBlockPETProjector` and check it with
  ``adjointness_test``,
- the data layout of non-TOF and TOF projections: (block pair, LOR in block
  pair[, TOF bin]).

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_03_run_block_scanner.py`
(block scanner geometry) and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF projectors).
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import math

import matplotlib.pyplot as plt

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
    # the scanner is only 1.7 mm long along x2: one tick avoids overlapping labels
    ax.set_zticks([0.0])


# %%
# 1. Input parameters
# -------------------
#
# A block module is a regular 3D grid of LOR endpoints, defined by its shape
# and the spacing of the endpoints along the three axes (in mm).

block_shape = (3, 2, 2)
block_spacing = (1.5, 1.2, 1.7)
scanner_radius = 10

# %%
# 2. A modularized PET scanner geometry
# -------------------------------------
#
# We place 6 block modules on a circle with a radius of 10 mm, on 6 of the 12
# sides of a regular 12-sided polygon: three neighbouring sides at the bottom
# and the three opposite sides at the top.  All block modules must be
# identical, but can be anywhere in space: the position of each copy is set by
# an affine transformation matrix, here a translation by the scanner radius
# along ``x1`` followed by a rotation by the angle ``phi`` about the ``x2``
# axis.

delta_phi = 2 * xp.pi / 12

aff_mat_trans = xp.eye(4, device=dev)
aff_mat_trans[1, -1] = scanner_radius

modules = []
for phi in [
    -delta_phi,
    0,
    delta_phi,
    5 * delta_phi,
    6 * delta_phi,
    7 * delta_phi,
]:
    aff_mat_rot = xp.asarray(
        [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        device=dev,
    )
    modules.append(
        parallelproj.pet_scanners.BlockPETScannerModule(
            xp,
            dev,
            block_shape,
            block_spacing,
            affine_transformation_matrix=(aff_mat_rot @ aff_mat_trans),
        )
    )

scanner = parallelproj.pet_scanners.ModularizedPETScannerGeometry(modules)

# %%
# 3. An LOR descriptor made of block pairs
# ----------------------------------------
#
# Once the LOR endpoints are defined, we define the LORs by listing the pairs
# of blocks that are in coincidence: each block pair contributes all LORs
# between the endpoints of its two blocks.  Here we use the 9 pairs between
# the 3 bottom blocks (0, 1, 2) and the 3 top blocks (3, 4, 5); more pairs
# would be possible.

lor_desc = parallelproj.pet_lors.EqualBlockPETLORDescriptor(
    scanner,
    xp.asarray(
        [
            [0, 3],
            [0, 4],
            [0, 5],
            [1, 3],
            [1, 4],
            [1, 5],
            [2, 3],
            [2, 4],
            [2, 5],
        ],
        device=dev,
    ),
)

# %%
# 4. A non-TOF projector
# ----------------------
#
# :class:`.EqualBlockPETProjector` combines the LOR descriptor with an image
# of 28 x 20 x 3 voxels of 0.5 x 0.5 x 1 mm.
# :meth:`.EqualBlockPETProjector.adjointness_test` checks numerically that
# the back projection is the adjoint of the forward projection (``xp`` and
# ``dev`` are taken from the projector; only the dtype is needed).

img_shape = (28, 20, 3)
voxel_size = (0.5, 0.5, 1.0)

proj = parallelproj.projectors.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
assert proj.adjointness_test(dtype=xp.float32)

# %%
# The figure shows the scanner and the image FOV (left) and, in addition, all
# LORs of the 9 block pairs (right).

fig, axes = plt.subplots(
    1, 2, figsize=(9, 4.8), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax in axes:
    view_from_front(ax)
    proj.show_geometry(ax)
lor_desc.show_block_pair_lors(axes[1], block_pair_nums=None, color=plt.cm.tab10(0))
axes[0].set_title("block modules and image FOV")
axes[1].set_title("all LORs of the 9 block pairs")
fig.show()

# %%
# All LORs connect a bottom block with a top block and cross the image FOV.

# %%
# 5. Non-TOF forward and back projection
# --------------------------------------
#
# The forward projection of an image full of ones has the shape
# (number of block pairs, number of LORs per block pair); each block pair has
# 12 x 12 = 144 LORs, one for every pair of its 12 + 12 LOR endpoints.

img = xp.ones(img_shape, dtype=xp.float32, device=dev)

img_fwd = proj(img)
print(img_fwd.shape)

# %%
# Back projecting a "histogram" full of ones gives the "sensitivity image"
# when attenuation and normalization are ignored.  It has the image shape.

ones_back = proj.adjoint(xp.ones(proj.out_shape, dtype=xp.float32, device=dev))
print(ones_back.shape)

# %%
# For an image of ones, each value of the forward projection is the length (in
# mm) of the LOR inside the image.

fig, ax = plt.subplots(figsize=(9, 2.6), layout="constrained")
im = ax.imshow(to_numpy_array(img_fwd), cmap="magma", aspect="auto")
ax.set_xlabel("LOR number in block pair")
ax.set_ylabel("block pair")
ax.set_title("forward projection of ones")
fig.colorbar(im, ax=ax, label="length in image [mm]")
fig.show()

# %%
# All values lie between about 10 and 13 mm.  Block pairs 0 (blocks 0 and 3)
# and 8 (blocks 2 and 5) connect opposite blocks whose LORs are tilted by 30°
# to ``x1``; they cross the 10 mm high image along the longest paths.

fig, _, _ = show_vol_cuts(
    ones_back,
    voxel_size=voxel_size,
    axis_labels=("x0", "x1", "x2"),
    fig_title="back projection of ones",
)
fig.show()

# %%
# The sensitivity is highest in the image centre, where the LORs of all block
# pairs cross, and shows a pattern of streaks along the discrete LOR
# directions.

# %%
# 6. A TOF projector
# ------------------
#
# As for sinogram projectors, assigning :class:`.TOFParameters` turns a second
# projector with the same LOR descriptor and image into a TOF projector, here
# with 27 TOF bins of 0.8 mm and a TOF kernel with a standard deviation of
# 2 mm.

proj_tof = parallelproj.projectors.EqualBlockPETProjector(
    lor_desc, img_shape, voxel_size
)
proj_tof.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=27, tofbin_width=0.8, sigma_tof=2.0, num_sigmas=3.0
)

assert proj_tof.adjointness_test(dtype=xp.float32)

# %%
# 7. TOF forward and back projection
# ----------------------------------
#
# The TOF forward projection of the image of ones has the shape
# (number of block pairs, number of LORs per block pair, number of TOF bins).

img_fwd_tof = proj_tof(img)
print(img_fwd_tof.shape)

# %%
# Back projecting a "TOF histogram" full of ones gives the TOF sensitivity
# image (when attenuation and normalization are ignored).

ones_back_tof = proj_tof.adjoint(
    xp.ones(proj_tof.out_shape, dtype=xp.float32, device=dev)
)
print(ones_back_tof.shape)

# %%
# The TOF profile of a single LOR shows how its line integral is distributed
# over the TOF bins.

fig, ax = plt.subplots(figsize=(6, 3), layout="constrained")
ax.plot(to_numpy_array(img_fwd_tof[0, 0, :]), ".-")
ax.set_xlabel("TOF bin")
ax.set_ylabel("forward projection")
ax.set_title("TOF profile of LOR 0 in block pair 0")
fig.show()

# %%
# The profile is the image of ones along the LOR (a box of the LOR's length
# in the image) blurred by the Gaussian TOF kernel; the sum over all TOF bins
# is (up to 0.2 %) the non-TOF value of the LOR.

fig, _, _ = show_vol_cuts(
    ones_back_tof,
    voxel_size=voxel_size,
    axis_labels=("x0", "x1", "x2"),
    fig_title="TOF back projection of ones",
)
fig.show()

# %%
# The TOF back projection of ones is practically identical to the non-TOF one
# (the maximum difference is 0.3 % of the maximum value): the 27 TOF bins
# cover the whole LOR, so the TOF kernels of every point sum to one over the
# TOF bins.

# %%
# Summary
# -------
#
# - A :class:`.ModularizedPETScannerGeometry` is made of identical
#   :class:`.BlockPETScannerModule` objects, placed with affine
#   transformation matrices.
# - An :class:`.EqualBlockPETLORDescriptor` defines the LORs from a list of
#   block pairs; the projection data have the shape (block pair, LOR in block
#   pair) and, with TOF, an additional trailing TOF-bin axis.
# - :class:`.EqualBlockPETProjector` is used like the sinogram projectors:
#   ``proj(x)``, ``proj.adjoint(y)``, and ``tof_parameters`` for TOF.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_04_run_pet_listmode_projector.py`
#    -- projections of individual events (listmode).

# sphinx_gallery_thumbnail_number = 3
