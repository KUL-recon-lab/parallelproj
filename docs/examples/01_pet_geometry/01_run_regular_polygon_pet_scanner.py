"""
Regular polygon PET scanner geometry
====================================

This example shows how to create and visualize PET scanners where the LOR
endpoints can be modeled as a stack of regular polygons.
"""

# %%
import parallelproj.pet_scanners
import matplotlib.pyplot as plt

# %%
from array_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)

# %%
# Define four different PET scanners with different geometries
# ------------------------------------------------------------
# :class:`.RegularPolygonPETScannerGeometry` can be used to create the
# geometry of PET scanners where the LOR endpoints can be modeled as a stack of
# regular polygons.
#
# Here we create four different PET scanners with different geometries.
# Note that `symmetry_axis` can be used to define which of the three axis is
# used as the cylinder (symmetry) axis.

scanner1 = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=8,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-4, 4, 3, device=dev),
    symmetry_axis=2,
)

scanner2 = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=8,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-4, 4, 3, device=dev),
    symmetry_axis=1,
)

scanner3 = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=400.0,
    num_sides=32,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.3,
    ring_positions=xp.linspace(-70, 70, 36, device=dev),
    symmetry_axis=2,
)

scanner4 = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=400.0,
    num_sides=32,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.3,
    ring_positions=xp.linspace(-70, 70, 36, device=dev),
    symmetry_axis=0,
)

# %%
# Obtaining world coordinates of LOR endpoints
# --------------------------------------------
# :meth:`.RegularPolygonPETScannerGeometry.get_lor_endpoints` can be used
# to obtain the world coordinates of the LOR endpoints

# get the word coordinates of the 4th LOR endpoint in the 1st "ring" (polygon)
# and the 5th LOR endpoint in the 2nd "ring" (polygon)
print("scanner1")
print(
    scanner1.get_lor_endpoints(
        xp.asarray([0, 1], device=dev), xp.asarray([3, 4], device=dev)
    )
)
print("scanner2")
print(
    scanner2.get_lor_endpoints(
        xp.asarray([0, 1], device=dev), xp.asarray([3, 4], device=dev)
    )
)

# %%
# Visualize the defined LOR endpoints
# -----------------------------------
#
# :meth:`.RegularPolygonPETScannerGeometry.show_lor_endpoints` can be used
# to visualize the defined LOR endpoints

fig = plt.figure(figsize=(8, 8), tight_layout=True)
ax1 = fig.add_subplot(221, projection="3d")
ax2 = fig.add_subplot(222, projection="3d")
ax3 = fig.add_subplot(223, projection="3d")
ax4 = fig.add_subplot(224, projection="3d")
scanner1.show_lor_endpoints(ax1)
scanner2.show_lor_endpoints(ax2)
scanner3.show_lor_endpoints(ax3)
scanner4.show_lor_endpoints(ax4)
fig.show()


# %%
# Defining an open PET scanner geometry
# -------------------------------------
#
# The `phis` argument can be used to manually define the azimuthal angles of the
# polygon "sides". This can be used to create open PET scanner geometries.
# Here we create an open geometry with 6 sides and 3 rings corresponding to
# a full geometry using 12 sides where 6 sides were removed.

open_scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=6,
    num_lor_endpoints_per_side=8,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-4, 4, 3),
    symmetry_axis=2,
    phis=(2 * xp.pi / 12) * xp.asarray([-1, 0, 1, 5, 6, 7], device=dev),
)

fig2 = plt.figure(figsize=(8, 8), tight_layout=True)
ax2a = fig2.add_subplot(111, projection="3d")
open_scanner.show_lor_endpoints(ax2a)
fig2.show()
