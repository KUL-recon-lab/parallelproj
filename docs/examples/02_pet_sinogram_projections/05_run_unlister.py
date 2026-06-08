"""
Listmode to sinogram unlisting
==============================

This example demonstrates :func:`.regular_polygon_events_to_sinogram`,
which histograms listmode events into a sinogram.

Red / blue detector naming
--------------------------
Each coincidence event involves two crystal detectors.  We label them
**red** and **blue** -- arbitrary colour names with *no* implied ordering.
The older *d1 / d2* labels were dropped because "1" suggests
"first-to-fire", but in raw PET data either crystal can fire first.  The
colour labels make it unambiguous that only *identity*, not
*chronological order*, is encoded.

.. note::

   :func:`.detection_times_to_tof_bin` always expects the arrival-time
   difference as ``t_blue - t_red`` (blue minus red, in nanoseconds).
   The sign tells you which detector fired *later*:

   * ``t_blue - t_red > 0`` -> blue fired later -> emission is closer to
     the **red** detector.
   * ``t_blue - t_red < 0`` -> red fired later -> emission is closer to
     the **blue** detector.

   You **must** pass the difference in this fixed order.  Swapping to
   ``t_red - t_blue`` silently mirrors every TOF bin assignment.

Projector direction and the need for a sign check
--------------------------------------------------
The TOF sinogram projector always traces each LOR in one fixed direction:
from the **canonical xstart** crystal to the **canonical xend** crystal.
These are defined once by the LOR descriptor and never change.  The TOF
bin numbering follows the projector direction: **bin 0 = closest to
xstart**.

Because the colour labels are arbitrary, the (d_red, d_blue) pair in a
listmode file may or may not match the (xstart, xend) order the projector
uses for that LOR:

* ``d_red = xstart, d_blue = xend`` -> ``t_blue - t_red > 0`` means
  emission closer to xstart -> low bin numbers.  No additional flip
  needed.
* ``d_red = xend,   d_blue = xstart`` -> the same physical event now has
  ``t_blue - t_red < 0``, and the mapping to sinogram TOF bins is
  **mirrored**.

:func:`.detection_times_to_tof_bin` resolves this per event by looking
up whether d_red is xstart or xend from the LOR descriptor, and applies
the correct sign flip automatically.

Simulation pipeline
-------------------
1. **Forward-project** an ``elliptic_cylinder_phantom`` into a **TOF
   sinogram**.
2. **Add Poisson noise** to obtain ``y_tof``.
3. Derive the non-TOF sinogram ``y_span1`` by **summing over the TOF
   axis**.
4. **Convert** ``y_tof`` to crystal-index events with
   :meth:`.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`.
5. Verify **non-TOF span-1 round-trip**.
6. Verify **span-3 round-trip** (``max_ring_difference = 2``).
7. Verify **TOF round-trip**.
8. **TOF sign illustration** -- six events on two LORs, with both orderings
   of (d_red, d_blue) relative to (xstart, xend), show that
   :func:`.detection_times_to_tof_bin` handles the sign flip correctly.

.. note::

   To run this example locally, download
   `example_utils.py <https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/example_utils.py>`_
   into the **same folder** as this script. Make sure ``parallelproj`` is installed.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import parallelproj.pet_scanners
import parallelproj.pet_lors as ppl
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj.unlist import (
    detection_times_to_tof_bin,
    regular_polygon_events_to_sinogram,
)
from example_utils import elliptic_cylinder_phantom

# %%
from example_utils import suggest_array_backend_and_device

xp, dev = suggest_array_backend_and_device(None, None)

# %%
# Simulation setup
# ----------------

# %%
# Scanner and LOR descriptor
# ~~~~~~~~~~~~~~~~~~~~~~~~~~

num_rings = 5
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=16,
    num_lor_endpoints_per_side=12,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

img_shape = (40, 40, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = ppl.RegularPolygonPETLORDescriptor(
    scanner,
    ppl.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

print(
    f"Scanner  : {scanner.num_lor_endpoints_per_ring} crystals/ring"
    f" x {scanner.num_rings} rings"
)
print(f"Sinogram : {lor_desc.spatial_sinogram_shape}  (rad x views x planes)")

# %%
# Simulate TOF data
# ~~~~~~~~~~~~~~~~~

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)
num_tof_bins = proj.tof_parameters.num_tofbins

x_true = elliptic_cylinder_phantom(xp, dev)

np.random.seed(0)
y_tof = xp.asarray(
    np.random.poisson(to_numpy_array(proj(x_true))).astype(np.int32),
    device=dev,
    dtype=xp.int32,
)

y_span1 = xp.sum(y_tof, axis=-1)

print(
    f"\nTOF sinogram     : shape={tuple(y_tof.shape)},"
    f" total counts={int(xp.sum(y_tof))}"
)
print(
    f"Non-TOF sinogram : shape={tuple(y_span1.shape)},"
    f" total counts={int(xp.sum(y_span1))}"
)

# %%
# Listmode events
# ~~~~~~~~~~~~~~~
#
# :meth:`~.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`
# returns a numpy array with columns
# ``(d_red, r_red, d_blue, r_blue, unsigned_sinogram_tof_bin)``.

_events_np = proj.convert_sinogram_to_crystal_index_events(y_tof, shuffle=True)

d_red_tof = xp.asarray(_events_np[:, 0], dtype=xp.int32, device=dev)
r_red_tof = xp.asarray(_events_np[:, 1], dtype=xp.int32, device=dev)
d_blue_tof = xp.asarray(_events_np[:, 2], dtype=xp.int32, device=dev)
r_blue_tof = xp.asarray(_events_np[:, 3], dtype=xp.int32, device=dev)
unsigned_sinogram_tof_bin = xp.asarray(_events_np[:, 4], dtype=xp.int32, device=dev)

# Non-TOF events share the same detector columns.
d_red, r_red, d_blue, r_blue = d_red_tof, r_red_tof, d_blue_tof, r_blue_tof

print(f"\nTotal events : {d_red.shape[0]}")

# %%
# Round-trip verification
# -----------------------

# %%
# Non-TOF round-trip
# ~~~~~~~~~~~~~~~~~~

sino_unlisted = regular_polygon_events_to_sinogram(proj, d_red, r_red, d_blue, r_blue)
assert bool(xp.all(sino_unlisted == y_span1)), "Span-1 round-trip failed"
print("\nSpan-1 round-trip: OK")

# %%
# Span-3 comparison
# ~~~~~~~~~~~~~~~~~

op_compress = ppl.SinogramAxialCompressionOperator(lor_desc, target_span=3)
span3_desc = op_compress.out_lor_descriptor

# Build a span-3 projector so we can pass it to the unlisting function.
proj_span3 = parallelproj.projectors.RegularPolygonPETProjector(
    span3_desc, img_shape=img_shape, voxel_size=voxel_size
)

y_span3 = xp.astype(op_compress(xp.astype(y_span1, xp.float32)), xp.int32)
sino_span3_unlisted = regular_polygon_events_to_sinogram(
    proj_span3, d_red, r_red, d_blue, r_blue
)

assert bool(xp.all(sino_span3_unlisted == y_span3)), "Span-3 round-trip failed"
print("Span-3 round-trip: OK")

# %%
# TOF round-trip
# ~~~~~~~~~~~~~~

sino_tof_unlisted = regular_polygon_events_to_sinogram(
    proj,
    d_red_tof,
    r_red_tof,
    d_blue_tof,
    r_blue_tof,
    unsigned_sinogram_tof_bin=unsigned_sinogram_tof_bin,
)
assert bool(xp.all(sino_tof_unlisted == y_tof)), "TOF round-trip failed"
print("TOF  round-trip: OK")

# %%
# Event vs sinogram TOF sign illustration
# ----------------------------------------
#
# A dual-ring scanner with 32 detectors (8 modules x 4 crystals) and
# 25 TOF bins of 24 mm each covering the full 600 mm diameter.
#
# Six events on two LORs demonstrate that :func:`.detection_times_to_tof_bin`
# assigns the correct sinogram TOF bin regardless of which colour label is
# assigned to which physical detector:
#
# * **Events 0-4** are on the d\ :sub:`7`\-d\ :sub:`24` LOR.
#   Events 0-1 have ``d_red = xstart``; events 2-4 have ``d_red = xend``
#   (reversed relative to the projector direction).
# * **Event 5** is on the d\ :sub:`1`\-d\ :sub:`29` LOR and illustrates
#   that out-of-sinogram events (trimmed by ``radial_trim``) are silently
#   dropped (``unsigned_sinogram_tof_bin = -1``).
#
radius_sign = 300.0  # mm  ->  diameter = 600 mm
num_tof_bins_s = 25
tofbin_width_s = 24.0  # mm  (25 x 24 mm = 600 mm = full diameter)

scanner_sign = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=radius_sign,
    num_sides=8,
    num_lor_endpoints_per_side=4,
    lor_spacing=50.0,
    ring_positions=xp.asarray([0.0, 50.0], device=dev),
    symmetry_axis=2,
)

lor_desc_sign = ppl.RegularPolygonPETLORDescriptor(
    scanner_sign,
    ppl.Michelogram(scanner_sign.num_rings, max_ring_difference=0, span=1),
    radial_trim=7,
    sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
)

proj_sign = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc_sign, img_shape=(1, 1, 1), voxel_size=(1.0, 1.0, 1.0)
)
proj_sign.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=num_tof_bins_s,
    tofbin_width=tofbin_width_s,
    sigma_tof=10.0,
    tofcenter_offset=0.0,
)

# %%
# Setup of 6 coincidence events
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

d_red_arr = xp.asarray([7, 7, 24, 24, 24, 1], dtype=xp.int32, device=dev)
d_blue_arr = xp.asarray([24, 24, 7, 7, 7, 29], dtype=xp.int32, device=dev)

num_events = d_red_arr.shape[0]

r_red_arr = xp.zeros(num_events, dtype=xp.int32, device=dev)
r_blue_arr = xp.zeros(num_events, dtype=xp.int32, device=dev)

# dt = t_blue - t_red for each event (nanoseconds)
dt_arr = xp.asarray([+1.0, -1.0, +1.0, -1.0, 0.0, 0.0], device=dev)

# %%
# Unlist events
# ~~~~~~~~~~~~~
#
# :func:`.detection_times_to_tof_bin` converts each event's ``t_blue - t_red``
# into the unsigned TOF bin index expected by the projector.
#
# .. note::
#
#    Events 0 and 2 are on the same physical LOR with the same
#    ``|t_blue - t_red|``, but their (d_red, d_blue) order is reversed.
#    Because the projector direction is fixed, this reversal maps them to
#    **symmetric TOF bins on opposite sides of the LOR midpoint**.
#
# .. note::
#
#    The total count in the unlisted sinogram can be less than the number
#    of input events when ``radial_trim > 0`` removes edge LORs from the
#    sinogram.  Events that fall on a trimmed LOR receive
#    ``unsigned_sinogram_tof_bin = -1`` and are silently dropped by
#    :func:`.regular_polygon_events_to_sinogram`.

ev_sino_tof_bins = detection_times_to_tof_bin(d_red_arr, d_blue_arr, dt_arr, proj_sign)

for i, (dr, db, dt, tb) in enumerate(
    zip(d_red_arr, d_blue_arr, dt_arr, ev_sino_tof_bins)
):
    print(
        f"event {i}: d_red={int(dr):2d}, d_blue={int(db):2d},"
        f" sino tof bin={int(tb):2d}, dt={float(dt):+.1f} ns"
    )

unlisted_tof_sino = regular_polygon_events_to_sinogram(
    proj_sign,
    d_red_arr,
    r_red_arr,
    d_blue_arr,
    r_blue_arr,
    unsigned_sinogram_tof_bin=ev_sino_tof_bins,
)

print(f"number of listmode events:           {num_events}")
print(f"num counts in unlisted TOF sinogram: {int(xp.sum(unlisted_tof_sino))}")


# %%
# Visualisation
# ~~~~~~~~~~~~~
#
# All LORs in view 0 are drawn as coloured segments -- one colour per TOF bin.
# TOF bin 0 (dark blue) is always at the **xstart** end of each LOR.

fig_sign = plt.figure(figsize=(10, 8))
ax_sign = fig_sign.add_subplot(111, projection="3d")
lor_desc_sign.show_tof_bins(
    ax=ax_sign, tof_parameters=proj_sign.tof_parameters, views=0, show_colorbar=True
)

lim = radius_sign * 1.3
ax_sign.set_xlim(-lim, lim)
ax_sign.set_ylim(-lim, lim)
ax_sign.set_zlim(0, 50)
ax_sign.view_init(elev=45, azim=-45)
fig_sign.tight_layout()
fig_sign.show()

plt.show()
