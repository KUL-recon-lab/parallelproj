"""
LOR descriptors and sinogram definition
=======================================

In a scanner with "cylindrical symmetry", all possible lines of response (LORs)
between two LOR endpoints can be sorted into a sinogram containing a radial,
view and plane dimension.
This example shows how this can be done using the :class:`.RegularPolygonPETLORDescriptor`

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
import numpy as np
import parallelproj.pet_scanners
import parallelproj.pet_lors
import matplotlib.pyplot as plt

# %%
from importlib import import_module, util
import parallelproj_core as ppc


# choose array backend and a device (CPU or CUDA GPU)
if util.find_spec("torch") is not None:
    xp = import_module("array_api_compat.torch")
    dev = "cuda" if xp.cuda.is_available() and ppc.cuda_enabled == 1 else "cpu"
elif util.find_spec("cupy") is not None and ppc.cupy_enabled == 1:
    xp = import_module("array_api_compat.cupy")
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
else:
    xp = import_module("array_api_compat.numpy")
    # using numpy, device must be cpu
    dev = "cpu"

print(f"Using array API: {xp.__name__}, device: {dev}")


# %%
def _central_plane_seg0(lor_desc: parallelproj.pet_lors.RegularPolygonPETLORDescriptor) -> int:
    """Return the plane index of the central plane belonging to segment 0."""
    seg = np.asarray(lor_desc.plane_segment.tolist())
    idx = np.where(seg == 0)[0]
    return int(idx[len(idx) // 2])


def _last_plane_highest_seg(lor_desc: parallelproj.pet_lors.RegularPolygonPETLORDescriptor) -> int:
    """Return the last plane index belonging to the highest-magnitude segment."""
    seg = np.asarray(lor_desc.plane_segment.tolist())
    idx = np.where(np.abs(seg) == int(np.abs(seg).max()))[0]
    return int(idx[-1])


# %%
# setup a small regular polygon PET scanner with 5 rings (polygons)

num_rings = 11
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=4,
    lor_spacing=8.0,
    ring_positions=2 * num_rings * xp.linspace(-1, 1, num_rings, device=dev),
    symmetry_axis=1,
)

# %%
# Defining a sinogram using an LOR descriptor
# -------------------------------------------
#
# :class:`.RegularPolygonPETLORDescriptor` can be used to order all possible
# combinations of LOR endpoints into a sinogram with a radial, view and plane dimension.
#
# `max_ring_difference` defines the maximum ring (polygon) difference between of a valid LOR
# and `radial_trim` defines the number of radial bins to be trimmed from the sinogram edges.
#
# `sinogram_order` of type :class:`.SinogramSpatialAxisOrder` defines the order of the sinogram dimensions
# (e.g. RVP -> [radial, view, plane], PRV -> [plane, radial, view])

lor_desc1 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(lor_desc1)
print(f"sinogram order: {lor_desc1.sinogram_order.name}")
print(f"sinogram shape: {lor_desc1.spatial_sinogram_shape}")
print(
    f"num rad: {lor_desc1.num_rad}  num views: {lor_desc1.num_views}  num planes: {lor_desc1.num_planes}"
)
print(
    f"radial axis num: {lor_desc1.radial_axis_num}  view axis num: {lor_desc1.view_axis_num}  plane axis num: {lor_desc1.plane_axis_num}"
)

# %%
# Define a 2nd LOR descriptor with sinogram order "PRV"

lor_desc2 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.PRV,
)

print(lor_desc2)
print(f"sinogram order: {lor_desc2.sinogram_order.name}")
print(f"sinogram shape: {lor_desc2.spatial_sinogram_shape}")
print(
    f"num rad: {lor_desc2.num_rad}  num views: {lor_desc2.num_views}  num planes: {lor_desc2.num_planes}"
)
print(
    f"radial axis num: {lor_desc2.radial_axis_num}  view axis num: {lor_desc2.view_axis_num}  plane axis num: {lor_desc2.plane_axis_num}"
)

# %%
# Obtaining world coordinates of LOR start and endpoints
# ------------------------------------------------------
#
# Every LOR is defined by two LOR endpoints.
# :meth:`.RegularPolygonPETLORDescriptor.get_lor_coordinates` can be used to
# to obtain the 3 world coordinates of them (for all views or a subset of
# views).

lor_start_points1, lor_end_points1 = lor_desc1.get_lor_coordinates()
print(lor_start_points1.shape, lor_end_points1.shape)

# print the start and end coordinates of the LOR corresponding to the 1st view
# the 2nd plane and the 3rd radial bin
print(lor_start_points1[2, 0, 1, :])
print(lor_end_points1[2, 0, 1, :])

# %%
# Do the same for the 2nd LOR descriptor that uses sinogram order "PRV"
# **The indexing has to be different compared to "RVP" to get the same LOR.**

lor_start_points2, lor_end_points2 = lor_desc2.get_lor_coordinates()
print(lor_start_points2.shape, lor_end_points2.shape)

# print the start and end coordinates of the LOR corresponding to the 1st view
# the 2nd plane and the 3rd radial bin
print(lor_start_points2[1, 2, 0, :])
print(lor_end_points2[1, 2, 0, :])

# %%
# Visualize the defined LOR endpoints
# -----------------------------------
#
# :meth:`.RegularPolygonPETScannerGeometry.show_lor_endpoints` can be used
# to visualize the defined LOR endpoints. Note that a zig-zag sampling pattern
# is used to define a view.

_p0 = _central_plane_seg0(lor_desc1)
_ph = _last_plane_highest_seg(lor_desc1)

fig = plt.figure(figsize=(16, 8), tight_layout=True)
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")
scanner.show_lor_endpoints(ax1)
lor_desc1.show_views(
    ax1,
    views=xp.asarray([0], device=dev),
    planes=xp.asarray([_p0], device=dev),
    lw=0.5,
    color="k",
)
ax1.set_title(f"view 0, central plane of seg 0 (plane {_p0})")
scanner.show_lor_endpoints(ax2)
lor_desc1.show_views(
    ax2,
    views=xp.asarray([lor_desc1.num_views // 2], device=dev),
    planes=xp.asarray([_ph], device=dev),
    lw=0.5,
    color="k",
)
ax2.set_title(
    f"view {lor_desc1.num_views // 2}, last plane of highest seg (plane {_ph})"
)
fig.show()

# %%
# Michelogram for span-1 (no max ring diff)

fig_m0, ax_m0 = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
lor_desc1.show_michelogram(ax_m0)
fig_m0.show()

# %%
# Segment side-view diagram for span-1 (no max ring diff)

fig_seg0 = lor_desc1.show_segment_lors()
fig_seg0.tight_layout()
fig_seg0.show()

# %%
# Span-5 sinogram without max ring difference limitation
# -------------------------------------------------------
#
# :class:`.RegularPolygonPETLORDescriptor` supports axial compression via the ``span``
# parameter.  With ``span=5`` ring pairs whose ring difference falls in the same segment
# and share the same axial midpoint are merged into a single sinogram plane.
# Setting ``max_ring_difference=None`` (the default) includes all ring pairs.

span = 5

lor_desc_s7 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    span=span,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(lor_desc_s7)
print(f"sinogram shape: {lor_desc_s7.spatial_sinogram_shape}")
print(f"num planes: {lor_desc_s7.num_planes}  (span={span}, no max ring diff)")

# %%
# Michelogram for span-5 (no max ring diff)

fig_m1, ax_m1 = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
lor_desc_s7.show_michelogram(ax_m1)
fig_m1.show()

# %%
# Segment side-view diagram for span-5 (no max ring diff)

fig_seg1 = lor_desc_s7.show_segment_lors()
fig_seg1.tight_layout()
fig_seg1.show()

# %%
# 3D visualisation of two planes – span-5 (no max ring diff)

_p0_s7 = _central_plane_seg0(lor_desc_s7)
_ph_s7 = _last_plane_highest_seg(lor_desc_s7)

fig_3d1 = plt.figure(figsize=(16, 8), tight_layout=True)
ax3d1a = fig_3d1.add_subplot(121, projection="3d")
ax3d1b = fig_3d1.add_subplot(122, projection="3d")
scanner.show_lor_endpoints(ax3d1a)
lor_desc_s7.show_views(
    ax3d1a,
    views=xp.asarray([0], device=dev),
    planes=xp.asarray([_p0_s7], device=dev),
    lw=0.5,
    color="k",
)
ax3d1a.set_title(f"span {span} | view 0, central plane of seg 0 (plane {_p0_s7})")
scanner.show_lor_endpoints(ax3d1b)
lor_desc_s7.show_views(
    ax3d1b,
    views=xp.asarray([lor_desc_s7.num_views // 2], device=dev),
    planes=xp.asarray([_ph_s7], device=dev),
    lw=0.5,
    color="k",
)
ax3d1b.set_title(
    f"span {span} | view {lor_desc_s7.num_views // 2}, last plane of highest seg (plane {_ph_s7})"
)
fig_3d1.show()

# %%
# Span-5 sinogram with max ring difference = 7
# ---------------------------------------------
#
# By additionally setting ``max_ring_difference=7`` we restrict the included
# ring pairs, reducing the number of segments and sinogram planes compared to
# the unrestricted span-5 case above.

max_ring_difference = 7

lor_desc_s7_mrd9 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    span=span,
    max_ring_difference=max_ring_difference,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(lor_desc_s7_mrd9)
print(f"sinogram shape: {lor_desc_s7_mrd9.spatial_sinogram_shape}")
print(
    f"num planes: {lor_desc_s7_mrd9.num_planes}  (span={span}, max ring diff={max_ring_difference})"
)

# %%
# Michelogram for span-5 with max ring diff = 7

fig_m2, ax_m2 = plt.subplots(1, 1, figsize=(6, 6), tight_layout=True)
lor_desc_s7_mrd9.show_michelogram(ax_m2)
fig_m2.show()

# %%
# Segment side-view diagram for span-5 with max ring diff = 7

fig_seg2 = lor_desc_s7_mrd9.show_segment_lors()
fig_seg2.tight_layout()
fig_seg2.show()

# %%
# 3D visualisation of two planes – span-5, max ring diff = 7

_p0_s7_mrd9 = _central_plane_seg0(lor_desc_s7_mrd9)
_ph_s7_mrd9 = _last_plane_highest_seg(lor_desc_s7_mrd9)

fig_3d2 = plt.figure(figsize=(16, 8), tight_layout=True)
ax3d2a = fig_3d2.add_subplot(121, projection="3d")
ax3d2b = fig_3d2.add_subplot(122, projection="3d")
scanner.show_lor_endpoints(ax3d2a)
lor_desc_s7_mrd9.show_views(
    ax3d2a,
    views=xp.asarray([0], device=dev),
    planes=xp.asarray([_p0_s7_mrd9], device=dev),
    lw=0.5,
    color="k",
)
ax3d2a.set_title(
    f"span {span} mrd {max_ring_difference} | view 0, central plane of seg 0 (plane {_p0_s7_mrd9})"
)
scanner.show_lor_endpoints(ax3d2b)
lor_desc_s7_mrd9.show_views(
    ax3d2b,
    views=xp.asarray([lor_desc_s7_mrd9.num_views // 2], device=dev),
    planes=xp.asarray([_ph_s7_mrd9], device=dev),
    lw=0.5,
    color="k",
)
ax3d2b.set_title(
    f"span {span} mrd {max_ring_difference} | view {lor_desc_s7_mrd9.num_views // 2}, last plane of highest seg (plane {_ph_s7_mrd9})"
)
fig_3d2.show()
