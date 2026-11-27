"""
Regular polygon PET scanners
============================

:bdg-primary:`geometry` :bdg-secondary:`beginner` :bdg-light:`CPU or GPU`

Most clinical and pre-clinical PET scanners are cylinders of detector rings.
parallelproj models such a scanner as a **stack of regular polygons**: every
ring is a polygon with ``num_sides`` sides, and every side holds the same
pattern of *LOR endpoints* (the detector crystals between which lines of
response, LORs, are measured).

**You will learn**

- how parallelproj's world axes ``x0``, ``x1``, ``x2`` relate to the scanner,
- how to create a scanner with :class:`.RegularPolygonPETScannerGeometry`,
  with its symmetry (axial) axis along any of the three axes,
- how to get the world coordinates of LOR endpoints and how to plot them,
- how to build open geometries, change the endpoint ordering, and model
  detector modules with gaps (non-uniform crystal spacing).

**Prerequisites:** none, this is the first example.
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
import numpy as np

import parallelproj.pet_scanners
from parallelproj._examples_utils import suggest_array_backend_and_device

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()


def view_from_front(ax) -> None:
    """3D view from in front of the scanner (looking along +x2, +x1 down)"""
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")


# %%
# 1. The scanner coordinate system
# --------------------------------
#
# parallelproj labels the three world axes ``x0``, ``x1``, ``x2`` (instead of
# ``x``, ``y``, ``z``), in the order of the axes of a 3D image array
# ``img[i0, i1, i2]``.  For a scanner with ``symmetry_axis=2``, the cylinder
# axis is ``x2``.  Standing in front of the scanner and looking into the bore:
#
# * ``x0`` runs **left to right**,
# * ``x1`` runs **top to bottom** (``+x1`` points down),
# * ``x2`` runs **away from you**, into the scanner.
#
# This right-handed system matches the DICOM patient axes (LPS) of a
# head-first supine patient.  All 3D plots below are drawn from this point of
# view (``view_from_front``).

# %%
# 2. Four scanners
# ----------------
#
# A :class:`.RegularPolygonPETScannerGeometry` is defined by the radius of the
# polygons, the number of sides, the LOR endpoints per side and their spacing,
# the axial ring positions, and the symmetry axis.  We create two small
# scanners (12 sides, 3 rings) and two large ones (32 sides, 36 rings), each
# with the symmetry axis along two different world axes.


def make_scanner(radius, num_sides, num_per_side, spacing, rings, symmetry_axis):
    return parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=radius,
        num_sides=num_sides,
        num_lor_endpoints_per_side=num_per_side,
        lor_spacing=spacing,
        ring_positions=rings,
        symmetry_axis=symmetry_axis,
    )


small_rings = xp.linspace(-16, 16, 3, device=dev)
large_rings = xp.linspace(-70, 70, 36, device=dev)

scanners = {
    "small, symmetry axis x2": make_scanner(65.0, 12, 8, 4.0, small_rings, 2),
    "small, symmetry axis x1": make_scanner(65.0, 12, 8, 4.0, small_rings, 1),
    "large, symmetry axis x2": make_scanner(400.0, 32, 16, 4.3, large_rings, 2),
    "large, symmetry axis x0": make_scanner(400.0, 32, 16, 4.3, large_rings, 0),
}
for name, sc in scanners.items():
    n = sc.num_lor_endpoints_per_ring
    print(f"{name}: {sc.num_rings} rings, {n} endpoints per ring")

# %%
# 3. World coordinates of LOR endpoints
# -------------------------------------
#
# An LOR endpoint is addressed by its ring index and its index within the
# ring.  :meth:`.RegularPolygonPETScannerGeometry.get_lor_endpoints` returns
# their world coordinates (in mm), here of endpoint 3 in ring 0 and endpoint
# 4 in ring 1:

ring_indices = xp.asarray([0, 1], device=dev)
endpoint_indices = xp.asarray([3, 4], device=dev)

sc = scanners["small, symmetry axis x2"]
print(sc.get_lor_endpoints(ring_indices, endpoint_indices))

# %%
# The last coordinate (``x2``) is the axial ring position, -16 mm and 0 mm.
# For the scanner with symmetry axis ``x1``, the ring position is the second
# coordinate:

sc = scanners["small, symmetry axis x1"]
print(sc.get_lor_endpoints(ring_indices, endpoint_indices))

# %%
# 4. Plotting the LOR endpoints
# -----------------------------
#
# :meth:`.RegularPolygonPETScannerGeometry.show_lor_endpoints` draws all
# endpoints of a scanner in a 3D axes, coloured by ring.

fig = plt.figure(figsize=(9, 8), layout="constrained")
for i, (name, sc) in enumerate(scanners.items()):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")
    view_from_front(ax)
    sc.show_lor_endpoints(ax)
    ax.set_title(name)
fig.show()

# %%
# The symmetry axis only changes the orientation of the cylinder: the two
# small scanners are identical up to a rotation, as are the two large ones.

# %%
# 5. Open geometries
# ------------------
#
# The ``phis`` argument sets the azimuthal angles of the polygon sides
# explicitly.  Leaving out sides gives an *open* scanner, e.g. for a scanner
# with room for a second modality.  Here we keep 6 of the 12 sides of a
# 12-sided scanner, in two opposite groups of 3.

open_scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=6,
    num_lor_endpoints_per_side=8,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-4, 4, 3, device=dev),
    symmetry_axis=2,
    phis=(2 * xp.pi / 12) * xp.asarray([-1, 0, 1, 5, 6, 7], device=dev),
)

fig = plt.figure(figsize=(5, 5), layout="constrained")
ax = fig.add_subplot(projection="3d")
view_from_front(ax)
open_scanner.show_lor_endpoints(ax)
ax.set_title("open scanner: 6 of 12 sides")
fig.show()

# %%
# 6. Endpoint ordering and the start angle
# ----------------------------------------
#
# By default, the endpoint indices within a ring increase **clockwise** in the
# front view, starting at the top (``-x1``).  :class:`.RingEndpointOrdering`
# switches to **counterclockwise** ordering, and ``phi0`` rotates the first
# side (in radians, a right-hand rotation about the symmetry axis; ignored if
# ``phis`` is given).  The figure shows all four combinations for an 8-sided
# ring with 2 endpoints per side, with the endpoint indices.

RO = parallelproj.pet_scanners.RingEndpointOrdering
variants = [
    (RO.CLOCKWISE, 0.0, "clockwise, phi0 = 0"),
    (RO.COUNTERCLOCKWISE, 0.0, "counterclockwise, phi0 = 0"),
    (RO.CLOCKWISE, math.pi / 8, "clockwise, phi0 = pi/8"),
    (RO.COUNTERCLOCKWISE, math.pi / 8, "counterclockwise, phi0 = pi/8"),
]

fig, axes = plt.subplots(
    2, 2, figsize=(9, 9), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax, (ordering, phi0, title) in zip(axes.flat, variants):
    ring = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=8,
        num_lor_endpoints_per_side=2,
        lor_spacing=20.0,
        ring_positions=xp.asarray([0.0], device=dev),
        symmetry_axis=2,
        ring_endpoint_ordering=ordering,
        phi0=phi0,
    )
    view_from_front(ax)
    ring.show_lor_endpoints(ax, show_linear_index=True, annotation_fontsize=9)
    ax.set_title(title)
fig.show()

# %%
# 7. Detector modules with gaps
# -----------------------------
#
# Real detector modules often consist of sub-blocks with a gap between them.
# ``lor_endpoint_positions`` takes the positions of the endpoints along a side
# (in mm, centred at 0) instead of a uniform spacing.  Here: 16 crystals per
# side in 4 sub-blocks of 4 crystals, 4 mm apart within a sub-block and 6 mm
# between sub-blocks:
#
# .. code-block:: text
#
#   sub-block 1      sub-block 2      sub-block 3      sub-block 4
#   -33 -29 -25 -21  -15 -11  -7  -3  +3  +7 +11 +15  +21 +25 +29 +33
#
# For the radial symmetry of the sinogram, the positions must be symmetric
# about 0 (``pos[i] == -pos[N - 1 - i]``).

pitch, gap = 4.0, 2.0  # mm: crystal pitch, extra gap between sub-blocks
offsets = pitch * (np.arange(4) - 1.5)  # within a sub-block: -6, -2, 2, 6
centres = (3 * pitch + pitch + gap) * (np.arange(4) - 1.5)  # -27, -9, 9, 27
positions = (centres[:, None] + offsets[None, :]).ravel()
print("endpoint positions along a side [mm]:", positions)

ring0 = xp.asarray([0.0], dtype=xp.float32, device=dev)
blocks = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=70.0,
    num_sides=6,
    ring_positions=ring0,
    symmetry_axis=2,
    lor_endpoint_positions=xp.asarray(positions, dtype=xp.float32, device=dev),
)
uniform = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=70.0,
    num_sides=6,
    num_lor_endpoints_per_side=16,
    lor_spacing=pitch,
    ring_positions=ring0,
    symmetry_axis=2,
)

fig, axes = plt.subplots(
    1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax, sc, title in zip(
    axes, (uniform, blocks), ("uniform spacing (4 mm)", "4 sub-blocks with 2 mm gaps")
):
    view_from_front(ax)
    sc.show_lor_endpoints(ax)
    ax.set_title(title)
fig.suptitle("6-sided scanner, 16 crystals per side")
fig.show()

# %%
# Summary
# -------
#
# - A :class:`.RegularPolygonPETScannerGeometry` is a stack of regular polygons
#   of LOR endpoints; ``symmetry_axis`` chooses the cylinder axis.
# - ``get_lor_endpoints(ring_indices, endpoint_indices)`` gives world
#   coordinates in mm; ``show_lor_endpoints(ax)`` plots them.
# - ``phis`` builds open geometries, ``ring_endpoint_ordering`` and ``phi0``
#   control the endpoint numbering, and ``lor_endpoint_positions`` models
#   modules with gaps.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_01_pet_geometry_02_run_regular_polygon_pet_sino.py`
#    -- how the LORs between these endpoints are sorted into sinograms.
#    Scanners made of arbitrary block modules are shown in
#    :ref:`sphx_glr_auto_examples_01_pet_geometry_03_run_block_scanner.py`.

# sphinx_gallery_thumbnail_number = 1
