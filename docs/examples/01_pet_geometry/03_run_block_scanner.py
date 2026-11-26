"""
Modularized (block) PET scanner geometry
========================================

:bdg-primary:`geometry` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Not every PET system is a closed cylinder: prototypes, organ-specific and
open scanners are often built from identical detector blocks placed at
arbitrary positions.  parallelproj models such a system as a
:class:`.ModularizedPETScannerGeometry` made of :class:`.BlockPETScannerModule`
objects, each a regular grid of LOR endpoints.  Its LORs are defined by an
explicit list of block pairs in coincidence.

**You will learn**

- how to create a block module and place copies of it in space with affine
  transformation matrices,
- how to combine the modules into a :class:`.ModularizedPETScannerGeometry`,
- how to define the LORs with an :class:`.EqualBlockPETLORDescriptor` from a
  list of block pairs,
- how to plot the LORs of block pairs and get their world coordinates.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_01_run_regular_polygon_pet_scanner.py`
and :ref:`sphx_glr_auto_examples_01_pet_geometry_02_run_regular_polygon_pet_sino.py`
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
from parallelproj._examples_utils import suggest_array_backend_and_device

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
# (number of endpoints along ``x0``, ``x1``, ``x2``) and the spacing between
# the endpoints.

block_shape = (3, 2, 2)  # grid shape of the LOR endpoints in a block module
block_spacing = (1.5, 1.2, 1.7)  # spacing between the LOR endpoints
scanner_radius = 10  # distance of the block centres from the scanner axis

# %%
# 2. A modularized PET scanner geometry
# -------------------------------------
#
# We place 7 block modules on a circle with radius 10.  The arrangement
# follows a regular polygon with 12 sides (30 degree steps), leaving some of
# the sides empty.  All block modules must be identical, but they can be
# anywhere in space: the location of each module is set with a 4 x 4 affine
# transformation matrix.  Here, every module is first translated from the
# centre to the radius along ``x1``, and then rotated by ``phi`` about the
# ``x2`` axis.

delta_phi = 2 * xp.pi / 12

# translation of a block module from the centre to the scanner radius
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
    8 * delta_phi,
]:
    # rotation of the block module about the x2 axis
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

# create the scanner geometry from a list of identical block modules at
# different locations in space
scanner = parallelproj.pet_scanners.ModularizedPETScannerGeometry(modules)

# %%
# :meth:`.ModularizedPETScannerGeometry.show_lor_endpoints` plots the LOR
# endpoints of all modules.  With ``show_linear_index=False``, every endpoint
# is annotated as ``module number, endpoint number within the module``.

fig = plt.figure(figsize=(5.5, 5.5), layout="constrained")
ax = fig.add_subplot(projection="3d")
view_from_front(ax)
scanner.show_lor_endpoints(ax, annotation_fontsize=5, show_linear_index=False)
ax.set_title("7 block modules with 3 x 2 x 2 LOR endpoints")
fig.show()

# %%
# Blocks 0, 1 and 2 (at -30, 0 and 30 degrees) sit at the bottom of the front
# view (``+x1``), blocks 3 to 6 (at 150 to 240 degrees) in the upper half,
# from top left to right.

# %%
# 3. An LOR descriptor from block pairs
# -------------------------------------
#
# Once the geometry of the LOR endpoints is defined, the LORs are defined by
# specifying which block pairs are in coincidence.  For a modularized
# scanner, this list of pairs of block numbers is set up manually and passed
# to :class:`.EqualBlockPETLORDescriptor`.  Every LOR between an endpoint of
# the first and an endpoint of the second block of a pair is used.  Here we
# define 12 block pairs (blocks 0-2 with blocks 3-6); more pairs would be
# possible.

lor_desc = parallelproj.pet_lors.EqualBlockPETLORDescriptor(
    scanner,
    xp.asarray(
        [
            [0, 3],
            [0, 4],
            [0, 5],
            [0, 6],
            [1, 3],
            [1, 4],
            [1, 5],
            [1, 6],
            [2, 3],
            [2, 4],
            [2, 5],
            [2, 6],
        ]
    ),
)

# %%
# 4. Visualising the LORs of block pairs
# --------------------------------------
#
# :meth:`.EqualBlockPETLORDescriptor.show_block_pair_lors` draws all LORs of
# the given block pairs (``block_pair_nums=None`` draws all of them).  The
# left panel shows 3 block pairs in different colours:
#
# * block pair 0: block 0 and block 3,
# * block pair 5: block 1 and block 4,
# * block pair 11: block 2 and block 6.
#
# The right panel shows the LORs of all 12 defined block pairs.

fig, axes = plt.subplots(
    1, 2, figsize=(10, 5.5), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax in axes:
    view_from_front(ax)
    scanner.show_lor_endpoints(ax, annotation_fontsize=4, show_linear_index=False)

for i, block_pair in enumerate([0, 5, 11]):
    lor_desc.show_block_pair_lors(
        axes[0],
        block_pair_nums=xp.asarray([block_pair], device=dev),
        color=plt.cm.tab10(i),
    )
axes[0].set_title("block pairs 0, 5 and 11")

lor_desc.show_block_pair_lors(axes[1], block_pair_nums=None, color=plt.cm.tab10(0))
axes[1].set_title("all 12 block pairs")
fig.show()

# %%
# Each block pair contributes 12 x 12 = 144 LORs.  The LORs only connect the
# two groups of blocks; there are no LORs within a group.

# %%
# 5. World coordinates of the LORs
# --------------------------------
#
# :meth:`.EqualBlockPETLORDescriptor.get_lor_coordinates` returns the start
# and end coordinates of the LORs of selected block pairs, or of all block
# pairs if no argument is given.

# all LORs of block pair 0 (connecting block 0 and block 3)
xstart0, xend0 = lor_desc.get_lor_coordinates(xp.asarray([0], device=dev))

# all LORs of block pair 4 (connecting block 1 and block 3)
xstart4, xend4 = lor_desc.get_lor_coordinates(xp.asarray([4], device=dev))

# all LORs of all block pairs
xstart, xend = lor_desc.get_lor_coordinates()

print("block pair 0:", xstart0.shape, xend0.shape)
print("block pair 4:", xstart4.shape, xend4.shape)
print("all block pairs:", xstart.shape, xend.shape)

# %%
# Each row holds the 3 world coordinates of the start (or end) point of one
# LOR: 144 LORs per block pair, 12 x 144 = 1728 LORs in total.

# %%
# Summary
# -------
#
# - A :class:`.BlockPETScannerModule` is a regular grid of LOR endpoints
#   (``shape``, ``spacing``), placed in space with an
#   ``affine_transformation_matrix``.
# - A :class:`.ModularizedPETScannerGeometry` combines a list of identical
#   modules; ``show_lor_endpoints(ax)`` plots them.
# - An :class:`.EqualBlockPETLORDescriptor` defines the LORs from an explicit
#   list of block pairs; ``show_block_pair_lors()`` plots them and
#   ``get_lor_coordinates()`` returns their world coordinates.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_01_pet_geometry_04_run_michelogram_and_compression.py`
#    -- axial compression of regular polygon sinograms with a Michelogram.

# sphinx_gallery_thumbnail_number = 2
