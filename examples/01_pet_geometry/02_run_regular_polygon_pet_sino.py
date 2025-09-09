"""
LOR descriptors and sinogram definition
=======================================

In a scanner with "cylindrical symmetry", all possible lines of response (LORs)
between two LOR endpoints can be sorted into a sinogram containing a radial,
view and plane dimension.
This example shows how this can be done using the :class:`.RegularPolygonPETLORDescriptor`

.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
import matplotlib.pyplot as plt

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = "cuda"


# %%
# setup a small regular polygon PET scanner with 5 rings (polygons)

num_rings = 5
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=4,
    lor_spacing=8.0,
    ring_positions=xp.linspace(-8, 8, num_rings, device=dev),
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

lor_desc1 = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
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

lor_desc2 = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.PRV,
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
# Definition of plane numbers
# ---------------------------
#
# The plane number definition corresponds to a span 1 Michelogram.

for i in range(lor_desc1.num_planes):
    st_pl = lor_desc1.start_plane_index[i]
    end_pl = lor_desc1.end_plane_index[i]
    print(
        f"plane num: {i:02} - start ring {st_pl} - end ring {end_pl} - ring diff. {(end_pl-st_pl):>2}"
    )

# %%
# Visualize the defined LOR endpoints
# -----------------------------------
#
# :meth:`.RegularPolygonPETScannerGeometry.show_lor_endpoints` can be used
# to visualize the defined LOR endpoints. Note that a zig-zag sampling pattern
# is used to define a view.

fig = plt.figure(figsize=(16, 8), tight_layout=True)
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")
scanner.show_lor_endpoints(ax1)
lor_desc1.show_views(
    ax1,
    views=xp.asarray([0], device=dev),
    planes=xp.asarray([num_rings // 2], device=dev),
    lw=0.5,
    color="k",
)
ax1.set_title(f"view 0, plane {num_rings // 2}")
scanner.show_lor_endpoints(ax2)
lor_desc1.show_views(
    ax2,
    views=xp.asarray([lor_desc1.num_views // 2], device=dev),
    planes=xp.asarray([lor_desc1.num_planes - 1], device=dev),
    lw=0.5,
    color="k",
)
ax2.set_title(f"view {lor_desc1.num_views // 2}, plane {lor_desc1.num_planes - 1}")
fig.show()

# %%
# Defining sinograms in open PET scanner geometries
# -------------------------------------------------
#
# :class:`.RegularPolygonPETLORDescriptor` can also be used with
# "open" PET scanner geometries. Note, however, that the definition
# of "views" is not trivial due to the presence of "missing" sides (gaps).
# The view definition still uses the "zig-zag" sampling which leads to
# unconvential (very non-parallel) views in the sinogram as shown below.

open_scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=6,
    num_lor_endpoints_per_side=4,
    lor_spacing=8.0,
    ring_positions=xp.linspace(-8, 8, num_rings, device=dev),
    symmetry_axis=1,
    phis=(2 * xp.pi / 12) * xp.asarray([-1, 0, 1, 5, 6, 7]),
)

open_lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    open_scanner,
    radial_trim=1,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

fig2 = plt.figure(figsize=(16, 8), tight_layout=True)
ax2a = fig2.add_subplot(121, projection="3d")
ax2b = fig2.add_subplot(122, projection="3d")
open_scanner.show_lor_endpoints(ax2a)
open_lor_desc.show_views(
    ax2a,
    views=xp.asarray([0], device=dev),
    planes=xp.asarray([num_rings // 2], device=dev),
    lw=0.5,
    color="k",
)
ax2a.set_title(f"view 0, plane {num_rings // 2}")
open_scanner.show_lor_endpoints(ax2b)
open_lor_desc.show_views(
    ax2b,
    views=xp.asarray([open_lor_desc.num_views // 2], device=dev),
    planes=xp.asarray([num_rings // 2], device=dev),
    lw=0.5,
    color="k",
)
ax2b.set_title(f"view {open_lor_desc.num_views // 2}, plane {num_rings // 2}")
fig2.show()
