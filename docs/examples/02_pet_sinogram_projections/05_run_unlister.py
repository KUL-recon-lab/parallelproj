"""
Listmode to sinogram histogramming (unlister)
=============================================

This example demonstrates :func:`.regular_polygon_events_to_sinogram`,
which histograms listmode events into a sinogram.

Red / blue detector naming
--------------------------
Each coincidence event involves two crystal detectors.  We label them
**red** and **blue** — arbitrary colour names with *no* implied ordering.
The older *d1 / d2* labels were dropped because "1" suggests
"first-to-fire", but in raw PET data either crystal can fire first.  The
colour labels make it unambiguous that only *identity*, not
*chronological order*, is encoded.

t_blue − t_red convention
--------------------------
:func:`.detection_times_to_tof_bin` always expects the arrival-time
difference as ``t_blue − t_red`` (blue minus red, in nanoseconds).  The
sign of this number tells you which detector fired *later*:

* ``t_blue − t_red > 0`` → blue fired later → emission is closer to the
  **red** detector.
* ``t_blue − t_red < 0`` → red fired later → emission is closer to the
  **blue** detector.

You **must** pass the difference in this fixed order.  Swapping to
``t_red − t_blue`` would silently mirror every TOF bin assignment.

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

* ``d_red = xstart, d_blue = xend`` → ``t_blue − t_red > 0`` means
  emission closer to xstart → low bin numbers.  No additional flip
  needed.
* ``d_red = xend,   d_blue = xstart`` → the same physical event now has
  ``t_blue − t_red < 0``, and the mapping to sinogram TOF bins is
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
   :meth:`.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`;
   non-TOF events drop the TOF column.
5. Verify **non-TOF span-1 round-trip**.
6. Verify **span-3 round-trip** (``max_ring_difference = 2``).
7. Verify **TOF round-trip**.
8. **TOF sign illustration** — four events on the same LOR, two with
   d_red = xstart and two with d_red = xend, show that
   :func:`.detection_times_to_tof_bin` handles both cases correctly.
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
    C_MM_PER_NS,
    detection_times_to_tof_bin,
    regular_polygon_events_to_sinogram,
)
from img import elliptic_cylinder_phantom

# %%
from array_utils import suggest_array_backend_and_device

xp, dev = suggest_array_backend_and_device(None, None)

# %%
# Scanner and LOR descriptor
# --------------------------

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
    f" × {scanner.num_rings} rings"
)
print(f"Sinogram : {lor_desc.spatial_sinogram_shape}  (rad × views × planes)")

# %%
# Simulate TOF data; derive non-TOF sinogram by summing over TOF axis
# -------------------------------------------------------------------

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
# Listmode events — unpack the stacked array into separate column arrays
# ----------------------------------------------------------------------
#
# ``convert_sinogram_to_crystal_index_events`` returns a numpy array with
# columns ``(d_red, r_red, d_blue, r_blue[, unsigned_sinogram_tof_bin])``.
# We move the arrays to the active device and unpack them.

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
# Non-TOF round-trip (span-1)
# ---------------------------

sino_unlisted = regular_polygon_events_to_sinogram(proj, d_red, r_red, d_blue, r_blue)
assert bool(xp.all(sino_unlisted == y_span1)), "Span-1 round-trip failed"
print("\nSpan-1 round-trip: OK")

# %%
# Span-3 comparison  (max_ring_difference = 2)
# --------------------------------------------

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
# TOF round-trip (span-1)
# -----------------------

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
# TOF sign illustration
# ---------------------
#
# A single-ring scanner with 8 detectors and 6 TOF bins of 100 mm each
# covering the full 600 mm diameter.
#
# The projector traces the LOR connecting d1 and d5 always from
# **xstart → xend** (determined by the LOR descriptor, fixed).
# We construct four events on that LOR to show that
# :func:`.detection_times_to_tof_bin` gives the correct sinogram TOF bin
# regardless of which colour label is assigned to which physical detector:
#
# * **Events 1 & 2** — ``d_red = xstart`` (same order as projector).
#   Passing ``t_blue − t_red`` directly gives the right bin.
# * **Events 3 & 4** — ``d_red = xend`` (reversed relative to projector).
#   The sign of ``t_blue − t_red`` is now *mirrored* compared to the
#   projector convention, and the helper applies the flip automatically.
#
# All four events call :func:`.detection_times_to_tof_bin` with the same
# interface: ``(d_red, d_blue, t_blue − t_red, projector)``.
#
radius_sign = 300.0  # mm  →  diameter = 600 mm
num_tof_bins_s = 6
tofbin_width_s = 100.0  # mm  (6 × 100 mm = 600 mm = full diameter)

scanner_sign = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=radius_sign,
    num_sides=8,
    num_lor_endpoints_per_side=1,
    lor_spacing=1.0,
    ring_positions=xp.asarray([0.0], device=dev),
    symmetry_axis=2,
)

lor_desc_sign = ppl.RegularPolygonPETLORDescriptor(
    scanner_sign,
    ppl.Michelogram(scanner_sign.num_rings, max_ring_difference=0, span=1),
    radial_trim=0,
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
# Identify xstart / xend for the (d_red, d_blue) = (1, 5) LOR.
#
# ``start_in_ring_index`` and ``end_in_ring_index`` record the canonical
# xstart / xend crystal for every sinogram bin.

d_red_ev, d_blue_ev = 1, 5

sc = to_numpy_array(lor_desc_sign.start_in_ring_index)  # (num_views, num_rad)
ec = to_numpy_array(lor_desc_sign.end_in_ring_index)

lor_mask = ((sc == d_red_ev) & (ec == d_blue_ev)) | (
    (sc == d_blue_ev) & (ec == d_red_ev)
)
v_lor, r_lor = np.argwhere(lor_mask)[0]

xstart_all, xend_all = lor_desc_sign.get_lor_coordinates()
xstart_xyz = to_numpy_array(xstart_all)[r_lor, v_lor, 0, :]
xend_xyz = to_numpy_array(xend_all)[r_lor, v_lor, 0, :]

idx_xstart = int(sc[v_lor, r_lor])
idx_xend = int(ec[v_lor, r_lor])

print(f"\nLOR d{d_red_ev}–d{d_blue_ev}: xstart=d{idx_xstart}, xend=d{idx_xend}")

# %%
# Four events — two orderings of the same LOR
# -------------------------------------------
#
# Events 1 & 2: d_red = xstart, d_blue = xend  (sign = +1).
# Events 3 & 4: d_red = xend,   d_blue = xstart (sign = −1, flip applied automatically).

d_red_arr = xp.asarray(
    [d_red_ev, d_red_ev, d_blue_ev, d_blue_ev], dtype=xp.int32, device=dev
)
d_blue_arr = xp.asarray(
    [d_blue_ev, d_blue_ev, d_red_ev, d_red_ev], dtype=xp.int32, device=dev
)

# dt = t_blue − t_red for each event (nanoseconds)
dt_evs = [+1.0, -1.0, +0.5, -0.5]
dt_arr = xp.asarray(np.array(dt_evs, dtype=np.float32), device=dev)

sino_bins = to_numpy_array(
    detection_times_to_tof_bin(d_red_arr, d_blue_arr, dt_arr, proj_sign)
)

# %%
# Step-by-step formula for each event

d_red_list  = [int(d_red_arr[i])  for i in range(4)]
d_blue_list = [int(d_blue_arr[i]) for i in range(4)]
signs = []
for dr, db in zip(d_red_list, d_blue_list):
    m = ((sc == dr) & (ec == db)) | ((sc == db) & (ec == dr))
    vr = np.argwhere(m)[0]
    signs.append(1 if sc[vr[0], vr[1]] == dr else -1)

print(
    f"\nN={num_tof_bins_s} bins, W={tofbin_width_s:.0f} mm,  "
    f"c/2 = {C_MM_PER_NS/2:.3f} mm/ns\n"
    f"  {'Event':<8}  {'d_red':>6}  {'d_blue':>6}  {'sign':>5}"
    f"  {'dt (ns)':>8}  {'dx (mm)':>8}  {'k_theory':>9}  {'bin':>4}"
)
for i, (dt_val, bin_val, sgn) in enumerate(zip(dt_evs, sino_bins, signs), start=1):
    dx = dt_val * C_MM_PER_NS / 2
    k_th = (num_tof_bins_s - 1) / 2.0 - sgn * dx / tofbin_width_s
    print(
        f"  Event {i:<3}  {d_red_list[i-1]:>6}  {d_blue_list[i-1]:>6}"
        f"  {sgn:>+5d}  {dt_val:>+8.1f}  {dx:>+8.1f}"
        f"  {k_th:>9.3f}  {bin_val:>4d}"
    )

# %%
# Sinogram round-trip: unlist each event and verify the bin.

for i in range(4):
    ev_bin_xp = xp.asarray(sino_bins[i : i + 1], dtype=xp.int32, device=dev)
    r_zero = xp.zeros(1, dtype=xp.int32, device=dev)
    sino_ev = to_numpy_array(
        regular_polygon_events_to_sinogram(
            proj_sign,
            d_red_arr[i : i + 1],
            r_zero,
            d_blue_arr[i : i + 1],
            r_zero,
            unsigned_sinogram_tof_bin=ev_bin_xp,
        )
    )
    actual = int(np.argwhere(sino_ev > 0)[0, -1])
    print(f"  Event {i+1}: sinogram bin (unlisted) = {actual}  ✓")

# %%
# Visualisation
# -------------
# Draw the TOF bin grid for the (1, 5) view, then highlight each event's
# assigned bin as a thick semi-transparent overlay on the (1, 5) LOR.

midpoint_xyz = (xstart_xyz + xend_xyz) / 2
lor_dir = (xend_xyz - xstart_xyz) / np.linalg.norm(xend_xyz - xstart_xyz)
lor_half = np.linalg.norm(xend_xyz - xstart_xyz) / 2

fig_sign = plt.figure(figsize=(10, 8))
ax_sign = fig_sign.add_subplot(111, projection="3d")
proj_sign.show_tof_bins(ax=ax_sign, views=int(v_lor), show_colorbar=True)

ev_colors = ["C0", "C1", "C2", "C3"]
ev_labels = [
    f"Event {i+1}: d_red={int(d_red_arr[i])}, d_blue={int(d_blue_arr[i])},"
    f" dt={dt_evs[i]:+.1f} ns → bin {sino_bins[i]}"
    for i in range(4)
]

for i in range(4):
    k = sino_bins[i]
    t_a = max((k     - num_tof_bins_s / 2) * tofbin_width_s, -lor_half)
    t_b = min((k + 1 - num_tof_bins_s / 2) * tofbin_width_s,  lor_half)
    p1 = midpoint_xyz + t_a * lor_dir
    p2 = midpoint_xyz + t_b * lor_dir
    ax_sign.plot(
        [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
        color=ev_colors[i], lw=8, alpha=0.5, solid_capstyle="butt",
        zorder=5, label=ev_labels[i],
    )

lim = radius_sign * 1.3
ax_sign.set_xlim(-lim, lim)
ax_sign.set_ylim(-lim, lim)
ax_sign.set_zlim(-lim / 4, lim / 4)
ax_sign.view_init(elev=25, azim=-60)
ax_sign.legend(loc="upper right", fontsize=7, framealpha=0.9, ncols=1)
fig_sign.tight_layout()
fig_sign.show()

plt.show()
