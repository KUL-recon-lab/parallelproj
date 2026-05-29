"""
Listmode to sinogram histogramming (unlister)
=============================================

This example demonstrates :func:`.regular_polygon_events_to_sinogram`,
which histograms listmode events into a sinogram.

Each event is described by four arrays ``(d_red, r_red, d_blue, r_blue)``
— in-ring crystal and ring indices for the two coincidence detectors —
plus an optional ``unsigned_sinogram_tof_bin`` array in the **projector convention**
(bin 0 = closest to the canonical xstart crystal).

The *red / blue* labels have no physical meaning beyond identifying the two
detectors; they replace the older *d1 / d2* notation to avoid implying
chronological ordering.

We build a full simulation pipeline:

1. **Forward-project** an ``elliptic_cylinder_phantom`` into a **TOF sinogram**.
2. **Add Poisson noise** to obtain ``y_tof``.
3. Derive the non-TOF sinogram ``y_span1`` by **summing over the TOF axis**.
4. **Convert** ``y_tof`` to crystal-index events with
   :meth:`.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`;
   non-TOF events drop the TOF column.
5. Verify **non-TOF span-1 round-trip**.
6. Verify **span-3 round-trip** (``max_ring_difference = 2``).
7. Verify **TOF round-trip**.
8. **TOF sign illustration** — shows why a per-event sign check is required
   when converting raw detection times to projector-convention TOF bins.
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
    _build_inring_luts,
    detection_times_to_tof_bin,
    regular_polygon_events_to_sinogram,
)
from img import elliptic_cylinder_phantom
from vis import show_vol_cuts

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
# A single-ring scanner with 8 detectors and 6 TOF bins (100 mm each)
# covering the full 600 mm diameter.
#
# We look at two events on the LOR between d1 and d5:
#
# * **Event 1** — emission 150 mm from centre toward d_red (d1):
#   ``t_blue − t_red = +1 ns`` (blue photon arrives later).
# * **Event 2** — emission 150 mm from centre toward d_blue (d5):
#   ``t_blue − t_red = −1 ns`` (blue photon arrives first).
#
# :func:`.detection_times_to_tof_bin` looks up whether d_red is xstart or xend
# and applies the correct sign so that both events land in the expected bin.
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
# Which crystal is xstart for the LOR d1–d5?

lut_sign, sign_lut_sign = _build_inring_luts(lor_desc_sign)
d_red_ev, d_blue_ev = 1, 5
assert lut_sign[d_red_ev, d_blue_ev] >= 0, "LOR (1,5) not in sinogram FOV"

sign_val = int(sign_lut_sign[d_red_ev, d_blue_ev])
idx_xstart = d_red_ev if sign_val == 1 else d_blue_ev
idx_xend = d_blue_ev if sign_val == 1 else d_red_ev

print(
    f"\nLOR d{d_red_ev}–d{d_blue_ev}: xstart=d{idx_xstart}, xend=d{idx_xend}"
    f"  (sign[{d_red_ev},{d_blue_ev}]={sign_val:+d})"
)

# World (x, y) coordinates
all_coords_s = to_numpy_array(scanner_sign.all_lor_endpoints)[:, :2]
xstart_xy = all_coords_s[idx_xstart]
xend_xy = all_coords_s[idx_xend]
midpoint_xy = (xstart_xy + xend_xy) / 2
lor_dir = (xend_xy - xstart_xy) / np.linalg.norm(xend_xy - xstart_xy)

# TOF bin-centre positions (bin 0 = closest to xstart)
tof_centers_s = np.array(
    [
        midpoint_xy + (k - (num_tof_bins_s - 1) / 2) * tofbin_width_s * lor_dir
        for k in range(num_tof_bins_s)
    ]
)

# %%
# Two events on the same LOR, 150 mm off-centre in opposite directions.

dt_evs = [+1.0, -1.0, +0.5, -0.5]

d_red_arr = xp.asarray(
    [d_red_ev, d_red_ev, d_blue_ev, d_blue_ev], dtype=xp.int32, device=dev
)
d_blue_arr = xp.asarray(
    [d_blue_ev, d_blue_ev, d_red_ev, d_red_ev], dtype=xp.int32, device=dev
)
dt_arr = xp.asarray(np.array(dt_evs, dtype=np.float32), device=dev)

sino_bins = to_numpy_array(
    detection_times_to_tof_bin(d_red_arr, d_blue_arr, dt_arr, proj_sign)
)

# Step-by-step calculation
print(
    f"\nN={num_tof_bins_s} bins, W={tofbin_width_s:.0f} mm,  "
    f"c/2 = {C_MM_PER_NS/2:.3f} mm/ns"
)
for i, (dt_val, bin_val) in enumerate(zip(dt_evs, sino_bins), start=1):
    dx = dt_val * C_MM_PER_NS / 2
    k_th = (num_tof_bins_s - 1) / 2.0 - sign_val * dx / tofbin_width_s
    print(
        f"  Event {i}: dt={dt_val:+.1f} ns"
        f"  dx_blue_red = {dx:+.1f} mm"
        f"  k = round({(num_tof_bins_s-1)/2:.1f}"
        f" − ({sign_val:+d})×{dx:.0f}/{tofbin_width_s:.0f})"
        f" = round({k_th:.2f}) = bin {bin_val}"
    )

# Physical emission positions along the LOR
ev_xy = [midpoint_xy + (-sign_val * dt * C_MM_PER_NS / 2) * lor_dir for dt in dt_evs]

# Also unlist the events and verify the sinogram bin
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

fig_sign, ax_sign = plt.subplots(figsize=(9, 9))

# Scanner ring
theta = np.linspace(0, 2 * np.pi, 300)
ax_sign.plot(
    radius_sign * np.cos(theta),
    radius_sign * np.sin(theta),
    color="lightgray",
    lw=1,
    zorder=0,
)

# All 8 detectors
for d in range(8):
    ax_sign.scatter(*all_coords_s[d], color="steelblue", s=180, zorder=5)
    offset = all_coords_s[d] / np.linalg.norm(all_coords_s[d]) * 26
    ax_sign.annotate(
        f"d{d}",
        all_coords_s[d] + offset,
        ha="center",
        va="center",
        fontsize=11,
    )

# LOR arrow (xstart → xend)
ax_sign.annotate(
    "",
    xy=xend_xy,
    xytext=xstart_xy,
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 2.0},
)
ax_sign.scatter(
    *xstart_xy,
    color="green",
    s=50,
    zorder=6,
    marker="D",
    label=f"xstart = d{idx_xstart}",
)
ax_sign.scatter(
    *xend_xy,
    color="firebrick",
    s=50,
    zorder=6,
    marker="D",
    label=f"xend   = d{idx_xend}",
)

# TOF bin edges as coloured line segments perpendicular to the LOR.
# There are N+1 edges; colours run from blue (xstart end) to red (xend end).
perp_dir = np.array([-lor_dir[1], lor_dir[0]])  # unit vector ⊥ to LOR
seg_half = 40.0  # mm  →  80 mm total segment length

# Edge j sits at (j − N/2) × W from the LOR midpoint toward xend.
edge_xy = np.array(
    [
        midpoint_xy + (j - num_tof_bins_s / 2) * tofbin_width_s * lor_dir
        for j in range(num_tof_bins_s + 1)
    ]
)
edge_colors = plt.cm.coolwarm(np.linspace(0, 1, num_tof_bins_s + 1))

for j in range(num_tof_bins_s + 1):
    p1 = edge_xy[j] - seg_half * perp_dir
    p2 = edge_xy[j] + seg_half * perp_dir
    ax_sign.plot(
        [p1[0], p2[0]],
        [p1[1], p2[1]],
        # color=edge_colors[j],
        color="k",
        lw=1.0,
        solid_capstyle="round",
    )

# Bin labels placed at the bin centres, offset beyond the segment end.
label_side = perp_dir * (seg_half + 18)
for k in range(num_tof_bins_s):
    ax_sign.annotate(
        f"sino TOF bin {k}",
        tof_centers_s[k] + label_side,
        ha="center",
        va="center",
        fontsize=9,
    )

# Two events (stars) with dashed line to their assigned bin
ev_colors = ["C0", "C1", "C2", "C3"]
ev_labels = [
    f"Event {i+1} d_red={int(d_red_arr[i])} d_blue={int(d_blue_arr[i])} (t_blue - t_red)={dt_evs[i]}ns"
    for i in range(4)
]

for i in range(4):
    ax_sign.scatter(
        *ev_xy[i], color=ev_colors[i], marker="*", s=200, zorder=7, label=ev_labels[i]
    )
    ax_sign.plot(
        [ev_xy[i][0], tof_centers_s[sino_bins[i]][0]],
        [ev_xy[i][1], tof_centers_s[sino_bins[i]][1]],
        color=ev_colors[i],
        lw=1.2,
        linestyle="--",
        zorder=3,
    )

ax_sign.set_aspect("equal")
ax_sign.set_xlim(-500, 500)
ax_sign.set_ylim(-500, 500)
ax_sign.legend(loc="upper right", fontsize=9, framealpha=0.9, ncols=2)
ax_sign.set_xlabel("x (mm)")
ax_sign.set_ylabel("y (mm)")
fig_sign.tight_layout()
fig_sign.show()

## %%
## Main sinograms
## --------------
#
## %%
# _, _, _w1 = show_vol_cuts(
#    to_numpy_array(y_span1),
#    axis_labels=("rad", "view", "plane"),
#    fig_title="y_span1  (non-TOF span-1, sum over TOF bins)",
# )
#
## %%
# _, _, _w2 = show_vol_cuts(
#    to_numpy_array(y_span3),
#    axis_labels=("rad", "view", "plane"),
#    fig_title="y_span3  (span-3, max_ring_difference=2)",
# )
#
## %%
# _, _, _w3 = show_vol_cuts(
#    to_numpy_array(y_tof).transpose(3, 0, 1, 2),
#    axis_labels=("tof", "rad", "view", "plane"),
#    fig_title="y_tof  (TOF span-1)",
# )

plt.show()
