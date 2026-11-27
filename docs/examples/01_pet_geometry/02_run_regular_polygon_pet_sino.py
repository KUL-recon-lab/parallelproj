"""
LOR descriptors and sinogram definition
=======================================

:bdg-primary:`geometry` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

In a scanner with cylindrical symmetry, all lines of response (LORs) between
two LOR endpoints can be sorted into a **sinogram** with a radial, a view and
a plane dimension.  :class:`.RegularPolygonPETLORDescriptor` defines this
sorting for a :class:`.RegularPolygonPETScannerGeometry`: which ring pairs are
used, how they are combined into planes (axial compression), and how the
sinogram bins are numbered.

**You will learn**

- how to create a :class:`.RegularPolygonPETLORDescriptor` and choose the
  order of the sinogram axes,
- how to get the world coordinates of the LOR start and end points,
- how a :class:`.Michelogram` with ``span`` and ``max_ring_difference``
  (or the GE-style layout) defines the sinogram planes,
- which arguments fix the mapping between sinogram bins ``(view, radial)``
  and detector pairs, and how to match a vendor convention.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_01_run_regular_polygon_pet_scanner.py`
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# helper functions below are used for the plots in this example.

import matplotlib.pyplot as plt
import numpy as np

import parallelproj
import parallelproj.pet_lors
import parallelproj.pet_scanners
from parallelproj._examples_utils import suggest_array_backend_and_device

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()


def view_from_front(ax) -> None:
    """3D view from in front of the scanner (looking along +x2, +x1 down)"""
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")


def central_plane_seg0(lor_desc) -> int:
    """Return the index of the central plane of segment 0."""
    seg = np.asarray(lor_desc.plane_segment.tolist())
    idx = np.where(seg == 0)[0]
    return int(idx[len(idx) // 2])


def last_plane_highest_seg(lor_desc) -> int:
    """Return the index of the last plane of the highest-magnitude segment."""
    seg = np.asarray(lor_desc.plane_segment.tolist())
    idx = np.where(np.abs(seg) == int(np.abs(seg).max()))[0]
    return int(idx[-1])


def show_two_planes(lor_desc, scanner, title: str) -> None:
    """3D plot of two (view, plane) combinations of a sinogram."""
    p0 = central_plane_seg0(lor_desc)
    ph = last_plane_highest_seg(lor_desc)
    vh = lor_desc.num_views // 2
    panels = [
        (0, p0, f"view 0, central plane of seg. 0 (plane {p0})"),
        (vh, ph, f"view {vh}, last plane of highest seg. (plane {ph})"),
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"}, layout="constrained"
    )
    for ax, (view, plane, panel_title) in zip(axes, panels):
        view_from_front(ax)
        scanner.show_lor_endpoints(ax)
        lor_desc.show_views(
            ax,
            views=xp.asarray([view], device=dev),
            planes=xp.asarray([plane], device=dev),
            lw=0.5,
            color="k",
        )
        ax.set_title(panel_title)
    fig.suptitle(title)
    fig.show()


def show_michelogram(lor_desc) -> None:
    """Michelogram (start ring vs. end ring) with the sinogram plane indices."""
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    lor_desc.show_michelogram(ax)
    fig.show()


def show_segments(lor_desc, title: str) -> None:
    """Side view of the ring pairs (LORs) contributing to every segment."""
    seg = np.asarray(lor_desc.plane_segment.tolist())
    num_cols = len(np.unique(np.abs(seg)))
    num_rows = 2 if (seg < 0).any() else 1
    width = min(10, 2.5 * num_cols)
    row_height = 1.05 * width / num_cols + 0.4  # panels are about square
    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(width, row_height * num_rows + 0.4),
        squeeze=False,
        layout="constrained",
    )
    lor_desc.show_segment_lors(axs=axs)
    fig.suptitle(title)
    fig.show()


# %%
# 1. A small regular polygon scanner
# ----------------------------------
#
# We use a small scanner with 12 sides, 4 LOR endpoints per side and 11 rings
# (polygons), spaced 4.4 mm apart along the symmetry axis ``x2``.

num_rings = 11
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=4,
    lor_spacing=8.0,
    ring_positions=2 * num_rings * xp.linspace(-1, 1, num_rings, device=dev),
    symmetry_axis=2,
)

# %%
# 2. Defining a sinogram with an LOR descriptor
# ---------------------------------------------
#
# :class:`.RegularPolygonPETLORDescriptor` orders all combinations of LOR
# endpoints into a sinogram with a radial, a view and a plane dimension.
#
# * The ring pairs that form valid LORs and their grouping into planes are
#   defined by a :class:`.Michelogram` (second argument).  Without it, all
#   ring pairs are used without axial compression (span 1, maximum ring
#   difference ``num_rings - 1``).
# * ``radial_trim`` is the number of radial bins removed at each edge of the
#   sinogram (LORs far from the centre that do not pass through the FOV).
# * ``sinogram_order`` (a :class:`.SinogramSpatialAxisOrder`) sets the order
#   of the sinogram axes, e.g. ``RVP`` = [radial, view, plane] or ``PRV`` =
#   [plane, radial, view].
#
# We create two descriptors that differ only in the axis order.


def print_sinogram_info(lor_desc) -> None:
    print(lor_desc)
    print(f"sinogram order: {lor_desc.sinogram_order.name}")
    print(f"sinogram shape: {lor_desc.spatial_sinogram_shape}")
    print(
        f"num rad: {lor_desc.num_rad}  num views: {lor_desc.num_views}  "
        f"num planes: {lor_desc.num_planes}"
    )
    print(
        f"radial axis num: {lor_desc.radial_axis_num}  "
        f"view axis num: {lor_desc.view_axis_num}  "
        f"plane axis num: {lor_desc.plane_axis_num}"
    )


lor_desc_rvp = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)
print_sinogram_info(lor_desc_rvp)

# %%

lor_desc_prv = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.PRV,
)
print_sinogram_info(lor_desc_prv)

# %%
# Both sinograms have 27 radial bins, 24 views and 121 planes (11 x 11 ring
# pairs); only the position of the axes in the array differs.

# %%
# 3. World coordinates of LOR start and end points
# ------------------------------------------------
#
# Every LOR is defined by two LOR endpoints.
# :meth:`.RegularPolygonPETLORDescriptor.get_lor_coordinates` returns their
# world coordinates (for all views or a subset of views) as two arrays with
# the sinogram shape plus a trailing axis of length 3.  Here we print the LOR
# in the 3rd radial bin, the 1st view and the 2nd plane.

lor_start_rvp, lor_end_rvp = lor_desc_rvp.get_lor_coordinates()
print(lor_start_rvp.shape, lor_end_rvp.shape)

# RVP: index order [radial, view, plane]
print("start:", lor_start_rvp[2, 0, 1, :])
print("end:  ", lor_end_rvp[2, 0, 1, :])

# %%
# For the descriptor with sinogram order ``PRV``, **the same LOR has a
# different index**, [plane, radial, view]:

lor_start_prv, lor_end_prv = lor_desc_prv.get_lor_coordinates()
print(lor_start_prv.shape, lor_end_prv.shape)

# PRV: index order [plane, radial, view]
print("start:", lor_start_prv[1, 2, 0, :])
print("end:  ", lor_end_prv[1, 2, 0, :])

# %%
# 4. Visualising views and planes
# -------------------------------
#
# :meth:`.RegularPolygonPETLORDescriptor.show_views` draws all LORs of given
# views and planes on top of the LOR endpoints.  We show view 0 in the
# central plane of segment 0 (a direct plane, both endpoints in the same
# ring), and the middle view in the last plane of the most oblique segment.

show_two_planes(lor_desc_rvp, scanner, "span 1, no maximum ring difference")

# %%
# The LORs of one view are (almost) parallel.  Their endpoints follow a
# zig-zag pattern: neighbouring radial bins alternately move the start or the
# end point by one crystal, which samples the view with half the crystal
# spacing.  The oblique plane on the right connects the first and the last
# ring.

# %%
# 5. Michelogram and segments (span 1)
# ------------------------------------
#
# The Michelogram shows every valid (start ring, end ring) pair as a dot,
# labelled with the index of the sinogram plane it is stored in.  Ring pairs
# with the same ring difference form a **segment**.

show_michelogram(lor_desc_rvp)

# %%
# Without axial compression, every ring pair has its own plane (121 planes).
# The segment diagram below shows, for every segment, the LORs between the
# rings in a side view (axial direction vertical), with the Michelogram as an
# inset (bottom left).  The panel titles give the number of compressed /
# uncompressed planes.

show_segments(lor_desc_rvp, "span 1, no maximum ring difference")

# %%
# With span 1, each of the 21 segments contains one ring difference, and the
# compressed and uncompressed LORs coincide.

# %%
# 6. Axial compression with span 5
# --------------------------------
#
# The ``span`` argument of the :class:`.Michelogram` enables axial
# compression: with ``span=5``, ring pairs whose ring differences fall into
# the same segment and that share the same axial midpoint are merged into a
# single sinogram plane.  ``max_ring_difference=scanner.num_rings - 1``
# includes all ring pairs.

span = 5

lor_desc_s5 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(
        scanner.num_rings, max_ring_difference=scanner.num_rings - 1, span=span
    ),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(lor_desc_s5)
print(f"sinogram shape: {lor_desc_s5.spatial_sinogram_shape}")
print(f"num planes: {lor_desc_s5.num_planes}  (span={span}, no max ring diff)")

# %%

show_michelogram(lor_desc_s5)

# %%
# Grey lines connect ring pairs that are merged into the same plane.  The
# number of planes drops from 121 to 61.

show_segments(lor_desc_s5, f"span {span}, no maximum ring difference")

# %%
# Segment 0 now holds ring differences -2 to +2, segments +/-1 hold ring
# differences 3 to 7 and segments +/-2 the remaining ones.  The dashed
# (compressed) LORs are the axial averages of the merged ring pairs.

show_two_planes(lor_desc_s5, scanner, f"span {span}, no maximum ring difference")

# %%
# 7. Limiting the maximum ring difference
# ---------------------------------------
#
# Additionally setting ``max_ring_difference=7`` excludes the most oblique
# ring pairs, which reduces the number of segments and sinogram planes
# compared to the unrestricted span-5 case above.

max_ring_difference = 7

lor_desc_s5_mrd7 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(
        scanner.num_rings, max_ring_difference=max_ring_difference, span=span
    ),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(lor_desc_s5_mrd7)
print(f"sinogram shape: {lor_desc_s5_mrd7.spatial_sinogram_shape}")
print(
    f"num planes: {lor_desc_s5_mrd7.num_planes}  "
    f"(span={span}, max ring diff={max_ring_difference})"
)

# %%

show_michelogram(lor_desc_s5_mrd7)

# %%
# The ring pairs in the corners of the Michelogram (ring differences above 7)
# are no longer used.

title_mrd7 = f"span {span}, maximum ring difference {max_ring_difference}"
show_segments(lor_desc_s5_mrd7, title_mrd7)

# %%

show_two_planes(lor_desc_s5_mrd7, scanner, title_mrd7)

# %%
# The most oblique plane now has a smaller axial extent than in the
# unrestricted case.

# %%
# 8. GE-style plane ordering
# --------------------------
#
# Instead of a single (odd) span, GE-style scanners use a *mixed* axial
# layout: segment 0 collects ring differences ``{-1, 0, +1}`` (the ``+/-1``
# cross planes are merged into virtual direct planes), while every oblique
# segment collects a ring-difference *pair* ``{+/-2k, +/-(2k+1)}`` laid out
# as a staircase.  Select it with the :meth:`.Michelogram.ge` constructor
# (equivalently ``layout=MichelogramLayout.GE``); ``span`` is then ignored and
# :attr:`.Michelogram.span` returns ``None``.  Choose ``num_rings`` and
# ``max_ring_difference`` to match the GE scanner of interest.

lor_desc_ge = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram.ge(
        scanner.num_rings, max_ring_difference=scanner.num_rings - 1
    ),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(lor_desc_ge)
print(f"sinogram shape: {lor_desc_ge.spatial_sinogram_shape}")
print(f"num planes: {lor_desc_ge.num_planes}  (GE layout, span={lor_desc_ge.span})")

# %%

show_michelogram(lor_desc_ge)

# %%

show_segments(lor_desc_ge, "GE-style layout")

# %%
# Segment 0 holds three ring differences (31 ring pairs in 21 planes).  The
# oblique segments hold two ring differences each, except segments +/-5,
# which only contain ring difference +/-10 (the maximum ring difference).

# %%
# 9. Sinogram indexing conventions
# --------------------------------
#
# The mapping between a sinogram bin ``(view, radial)`` and the underlying
# pair of detectors is fixed by a small set of independent arguments.  By
# default, **the central radial bin of view 0 connects detector 0 and
# detector N/2** (diametrically opposite).  The arguments then allow to
# reproduce any vendor's convention:
#
# * ``ring_endpoint_ordering`` (on the *scanner*): direction of the physical
#   crystal numbering around the ring (``CLOCKWISE`` / ``COUNTERCLOCKWISE``).
# * ``phi0`` (on the *scanner*): azimuth of module 0.  The default ``0``
#   places module 0 on the ``-x1`` axis (top of the front view) for
#   ``symmetry_axis=2``.
# * ``zig_zag_order``: which endpoint takes the interleaving half-step
#   (``END_FIRST`` / ``START_FIRST``).
# * ``view_direction``: direction in which the view index advances
#   (``PLUS`` / ``MINUS``).
# * ``radial_direction``: direction in which the radial index advances.
#
# ``view_direction`` / ``radial_direction`` flip the sinogram *bin* layout
# while ``ring_endpoint_ordering`` flips the physical crystal *numbering*;
# together, these independent choices span every regular-polygon convention.
#
# To print the complete detector <-> (view, radial) tables, we use a
# minimal 1-ring scanner with 4 sides x 2 endpoints = 8 detectors.  Each
# table entry is ``start-end`` detector index.

mini = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=30.0,
    num_sides=4,
    num_lor_endpoints_per_side=2,
    lor_spacing=12.0,
    ring_positions=xp.asarray([0.0], device=dev),
    symmetry_axis=2,
)
num_det = mini.num_lor_endpoints_per_ring  # = 8


def print_table(lor_desc, title: str) -> None:
    """Print the full (view, radial) -> ``start-end`` detector-index table."""
    s = np.asarray(parallelproj.to_numpy_array(lor_desc.start_in_ring_index))
    e = np.asarray(parallelproj.to_numpy_array(lor_desc.end_in_ring_index))
    print(f"\n{title}   (view 0 central bin -> detectors (0, {num_det // 2}))")
    print("  radial bin :", " ".join(f"{r:>3d}" for r in range(lor_desc.num_rad)))
    for v in range(lor_desc.num_views):
        row = " ".join(f"{s[v, r]}-{e[v, r]}" for r in range(lor_desc.num_rad))
        print(f"  view {v}     :", row)


lors = parallelproj.pet_lors
variants = {
    "default": {},
    "view_direction=MINUS": {"view_direction": lors.ViewDirection.MINUS},
    "radial_direction=MINUS": {"radial_direction": lors.RadialDirection.MINUS},
    "zig_zag_order=START_FIRST": {
        "zig_zag_order": lors.SinogramZigZagOrder.START_FIRST
    },
}
for title, kwargs in variants.items():
    lor_desc = lors.RegularPolygonPETLORDescriptor(mini, radial_trim=0, **kwargs)
    print_table(lor_desc, title)

# %%
# ``view_direction=MINUS`` reverses the order of the views (view 1 is rotated
# the other way), ``radial_direction=MINUS`` reverses each row, and
# ``zig_zag_order=START_FIRST`` shifts the half-step to the start point.
#
# ``ring_endpoint_ordering`` is a *scanner* argument: it changes the physical
# crystal numbering (mirrors the detector positions) but not the
# (view, radial) index formula, so the tables above are unchanged while
# detector 1 moves to the other side:

ccw = parallelproj.pet_scanners.RingEndpointOrdering.COUNTERCLOCKWISE
mini_ccw = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=30.0,
    num_sides=4,
    num_lor_endpoints_per_side=2,
    lor_spacing=12.0,
    ring_positions=xp.asarray([0.0], device=dev),
    symmetry_axis=2,
    ring_endpoint_ordering=ccw,
)
endpoints_cw = parallelproj.to_numpy_array(mini.all_lor_endpoints).reshape(-1, 3)
endpoints_ccw = parallelproj.to_numpy_array(mini_ccw.all_lor_endpoints).reshape(-1, 3)
print(
    "\ndetector 1 position (x0, x1):  CW =",
    np.round(endpoints_cw[1, :2], 1),
    "  CCW =",
    np.round(endpoints_ccw[1, :2], 1),
)

# %%
# 10. The four view and radial direction combinations
# ---------------------------------------------------
#
# The figure shows the minimal scanner for all four combinations of
# :class:`.ViewDirection` and :class:`.RadialDirection`.  Detector numbers are
# annotated; **view 0** is drawn in black and **view 1** in red, each with its
# radial-bin labels in the matching colour, so that the effect of
# ``view_direction`` (the direction in which the views advance) is visible.


def draw_directions_panel(ax, view_direction, radial_direction) -> None:
    view_from_front(ax)
    mini.show_lor_endpoints(ax, annotation_fontsize=9, show_linear_index=True)
    lor_desc = lors.RegularPolygonPETLORDescriptor(
        mini,
        radial_trim=0,
        view_direction=view_direction,
        radial_direction=radial_direction,
    )
    xs, xe = lor_desc.get_lor_coordinates(views=xp.asarray([0, 1], device=dev))
    xs = parallelproj.to_numpy_array(xs)
    xe = parallelproj.to_numpy_array(xe)
    for view, color in ((0, "k"), (1, "tab:red")):
        for r in range(lor_desc.num_rad):
            idx = [0, 0, 0]
            idx[lor_desc.radial_axis_num] = r
            idx[lor_desc.view_axis_num] = view
            idx[lor_desc.plane_axis_num] = 0
            a, b = xs[tuple(idx)], xe[tuple(idx)]
            ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], color=color, lw=0.5)
            # shift the labels of the two views apart so that they don't overlap
            m = 0.5 * (a + b) + (2.0 if view == 1 else -2.0)
            ax.text(m[0], m[1], m[2], f"r{r}", color=color, fontsize=8)
    ax.set_zticks([])  # single ring: the x2 axis carries no information
    ax.set_zlabel("")
    ax.set_title(
        f"view_direction={view_direction.name}, "
        f"radial_direction={radial_direction.name}"
    )


fig, axes = plt.subplots(
    2, 2, figsize=(9, 9), subplot_kw={"projection": "3d"}, layout="constrained"
)
combos = [
    (lors.ViewDirection.PLUS, lors.RadialDirection.PLUS),
    (lors.ViewDirection.PLUS, lors.RadialDirection.MINUS),
    (lors.ViewDirection.MINUS, lors.RadialDirection.PLUS),
    (lors.ViewDirection.MINUS, lors.RadialDirection.MINUS),
]
for ax, (view_direction, radial_direction) in zip(axes.flat, combos):
    draw_directions_panel(ax, view_direction, radial_direction)
fig.suptitle("view 0 (black) and view 1 (red) with their radial-bin labels")
fig.show()

# %%
# In all four panels, the central bin r3 of view 0 connects detector 0 and
# detector 4.  ``radial_direction`` reverses the radial labels, ``view_direction``
# changes whether view 1 is rotated clockwise or counterclockwise from
# view 0.

# %%
# 11. Rotating view 0 with ``view0_shift``
# ----------------------------------------
#
# By default, the central radial bin of view 0 connects detectors
# ``(0, N/2)``.  The ``view0_shift=m`` argument (a non-negative integer)
# rotates the detector anchor of *every* view by ``m`` crystals, so the
# central bin of view 0 instead connects ``((0+m) mod N, (N/2+m) mod N)``.
# Below we draw the *complete* view 0 (all radial bins) of an 8-module x
# 5-crystal ring for ``m=0`` (default) and ``m=2``.

mini_shift = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=100.0,
    num_sides=8,
    num_lor_endpoints_per_side=5,
    lor_spacing=10.0,
    ring_positions=xp.asarray([0.0], device=dev),
    symmetry_axis=2,
)

fig, axes = plt.subplots(
    1, 2, figsize=(10, 5.5), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax, shift in zip(axes, (0, 2)):
    view_from_front(ax)
    mini_shift.show_lor_endpoints(ax, annotation_fontsize=8, show_linear_index=True)
    lor_desc = lors.RegularPolygonPETLORDescriptor(
        mini_shift, radial_trim=0, view0_shift=shift
    )
    lor_desc.show_views(
        ax,
        views=xp.asarray([0], device=dev),
        planes=xp.asarray([0], device=dev),
        lw=1.0,
        color="tab:red",
    )
    central = (lor_desc.num_rad - 1) // 2
    s_idx = int(parallelproj.to_numpy_array(lor_desc.start_in_ring_index)[0, central])
    e_idx = int(parallelproj.to_numpy_array(lor_desc.end_in_ring_index)[0, central])
    ax.set_zticks([])  # single ring: the x2 axis carries no information
    ax.set_zlabel("")
    ax.set_title(f"view0_shift = {shift}, central bin = ({s_idx}, {e_idx})")
fig.suptitle(f"complete view 0 (N = {mini_shift.num_lor_endpoints_per_ring})")
fig.show()

# %%
# The whole set of parallel LORs of view 0 rotates by 2 crystals, while the
# rest of the sinogram indexing is unchanged.

# %%
# 12. Matching vendor conventions
# -------------------------------
#
# By default, the central bin of view 0 connects detectors ``(0, N/2)`` and
# module 0 sits on ``-x1`` (top of the front view).  To line a descriptor up
# with the sinograms of a specific vendor, you set a few arguments; there is
# **no single "vendor" setting**, because e.g. different vendors use
# opposite handedness:
#
# * ``phi0`` (scanner): rotate module 0 to the azimuth of the vendor's
#   physical crystal 0.
# * ``ring_endpoint_ordering`` (scanner): match the vendor's crystal
#   numbering direction.
# * ``view_direction`` / ``radial_direction``: match the vendor's sinogram
#   bin ordering (vendors differ here).
# * ``zig_zag_order``: match the interleaving of adjacent LOR angles.
#
# Rather than trusting a preset, **verify** the combination on your data,
# e.g. by backprojecting a sparse sinogram.

# %%
# Summary
# -------
#
# - :class:`.RegularPolygonPETLORDescriptor` sorts the LORs of a regular
#   polygon scanner into a sinogram; ``sinogram_order`` sets the axis order
#   and ``radial_trim`` removes outer radial bins.
# - ``get_lor_coordinates()`` returns the LOR start and end points in world
#   coordinates; ``show_views()`` plots selected views and planes.
# - A :class:`.Michelogram` (``span``, ``max_ring_difference``, or
#   :meth:`.Michelogram.ge`) defines the sinogram planes;
#   ``show_michelogram()`` and ``show_segment_lors()`` visualise it.
# - ``view_direction``, ``radial_direction``, ``zig_zag_order`` and
#   ``view0_shift`` (descriptor) together with ``ring_endpoint_ordering`` and
#   ``phi0`` (scanner) fix the mapping between sinogram bins and detectors.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_01_pet_geometry_03_run_block_scanner.py`
#    -- scanners made of arbitrary block modules.
#    Axial compression is covered in more detail in
#    :ref:`sphx_glr_auto_examples_01_pet_geometry_04_run_michelogram_and_compression.py`.

# sphinx_gallery_thumbnail_number = 1
