"""
Listmode to sinogram histogramming (unlister)
=============================================

This example demonstrates :func:`.regular_polygon_events_to_sinogram`,
which histograms listmode crystal-pair events into a sinogram.

We build a full simulation pipeline:

1. **Forward-project** an ``elliptic_cylinder_phantom`` with a realistic PET
   scanner into a span-1 sinogram.
2. **Add Poisson noise** to obtain an integer sinogram ``y_span1``.
3. **Convert** ``y_span1`` to crystal-index listmode events
   ``(d1, r1, d2, r2)`` — one event row per detected photon-pair — using
   :meth:`.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`.
4. **Unlist into span-1** and verify that the recovered sinogram is
   *identical* to ``y_span1`` (exact integer round-trip).
5. **Unlist directly into span-3** and verify that the result equals
   :class:`.SinogramAxialCompressionOperator` applied to ``y_span1`` — no
   re-binning step is needed; the unlister uses the span-3 ring-pair table
   directly.
6. Repeat steps 3–4 for a **TOF sinogram** to confirm that TOF bins are
   round-tripped correctly as well.

The span-3 equality demonstrates a useful property: instead of first
histogramming into span-1 and then compressing, you can bin straight into
any odd-span sinogram in a single pass over the event list.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import parallelproj.pet_scanners
import parallelproj.pet_lors as ppl
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj.unlist import regular_polygon_events_to_sinogram
from img import elliptic_cylinder_phantom

# %%
from array_utils import suggest_array_backend_and_device

xp, dev = suggest_array_backend_and_device(None, None)

# %%
# Scanner and LOR descriptor
# --------------------------
#
# We use the same scanner geometry as the listmode reconstruction example:
# 16 sides × 12 crystals per side = 192 crystals per ring, 5 rings.

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

print(f"Scanner  : {scanner.num_lor_endpoints_per_ring} crystals/ring × {scanner.num_rings} rings")
print(f"Sinogram : {lor_desc.spatial_sinogram_shape}  (num_rad × num_views × num_planes)")

# %%
# Simulate non-TOF PET data
# -------------------------
#
# Forward-project the phantom and add Poisson noise.

x_true = elliptic_cylinder_phantom(xp, dev)
noise_free = proj(x_true)

np.random.seed(0)
y_span1 = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free)).astype(np.int32),
    device=dev,
    dtype=xp.int32,
)

total_counts = int(xp.sum(y_span1))
print(f"\nNon-TOF span-1 sinogram : shape={tuple(y_span1.shape)}, total counts={total_counts}")

# %%
# Non-TOF round-trip (span-1)
# ---------------------------
#
# Convert the integer sinogram to crystal-index events and unlist back into
# span-1.  With ``radial_trim=10`` the scanner has no self-pair bins, so
# every count is round-tripped exactly.

events = proj.convert_sinogram_to_crystal_index_events(y_span1, shuffle=True)
print(f"\nNumber of events : {len(events)}")

sino_unlisted = regular_polygon_events_to_sinogram(lor_desc, events)
y_span1_np = to_numpy_array(y_span1).astype(np.float32)

print(f"Span-1 round-trip exact match : {np.array_equal(sino_unlisted, y_span1_np)}")
print(f"Max absolute difference       : {float(np.max(np.abs(sino_unlisted - y_span1_np))):.0f}")

# %%
# Span-3 comparison
# -----------------
#
# :class:`.SinogramAxialCompressionOperator` sums span-1 planes that share
# the same span-3 segment and axial midpoint.  Unlisting the *same events*
# directly into the span-3 descriptor must produce the identical result,
# because the span-3 ring-pair lookup table collapses those ring pairs in
# exactly the same way.

op_compress = ppl.SinogramAxialCompressionOperator(lor_desc, target_span=3)
span3_desc = op_compress.out_lor_descriptor

y_span3_np = to_numpy_array(op_compress(xp.astype(y_span1, xp.float32)))
sino_span3_unlisted = regular_polygon_events_to_sinogram(span3_desc, events)

print(f"\nSpan-3 sinogram shape (operator) : {y_span3_np.shape}")
print(f"Span-3 sinogram shape (unlisted) : {sino_span3_unlisted.shape}")
print(f"Span-3 exact match               : {np.array_equal(sino_span3_unlisted, y_span3_np)}")
print(f"Max absolute difference          : {float(np.max(np.abs(sino_span3_unlisted - y_span3_np))):.0f}")

# %%
# TOF simulation
# --------------
#
# Enable TOF on the projector and re-simulate.

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)
num_tof_bins = proj.tof_parameters.num_tofbins

noise_free_tof = proj(x_true)

np.random.seed(1)
y_tof = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_tof)).astype(np.int32),
    device=dev,
    dtype=xp.int32,
)

print(f"\nTOF sinogram shape : {tuple(y_tof.shape)},  total counts={int(xp.sum(y_tof))}")

# %%
# TOF round-trip (span-1)
# -----------------------
#
# ``convert_sinogram_to_crystal_index_events`` detects the trailing TOF
# dimension automatically (4-D input) and returns ``(d1, r1, d2, r2, tof_bin)``
# rows, where bin 0 is the bin closest to d1 (the xstart crystal).

events_tof = proj.convert_sinogram_to_crystal_index_events(y_tof, shuffle=True)
print(f"\nNumber of TOF events : {len(events_tof)}")

sino_tof_unlisted = regular_polygon_events_to_sinogram(
    lor_desc, events_tof, num_tof_bins=num_tof_bins
)
y_tof_np = to_numpy_array(y_tof).astype(np.float32)

print(f"TOF round-trip exact match : {np.array_equal(sino_tof_unlisted, y_tof_np)}")
print(f"Max absolute difference    : {float(np.max(np.abs(sino_tof_unlisted - y_tof_np))):.0f}")

# %%
# Visualisation
# -------------
#
# Non-TOF: ground-truth span-1 sinogram, unlisted span-1 sinogram, difference
# and the span-3 comparison.

v_ax = lor_desc.view_axis_num

fig, axes = plt.subplots(2, 3, figsize=(13, 8))

# --- row 0: span-1 round-trip ---
vmax1 = float(np.max(y_span1_np))

ax = axes[0, 0]
ax.imshow(y_span1_np.sum(axis=v_ax), aspect="auto", vmin=0, vmax=vmax1)
ax.set_title("y_span1 (ground truth,\nsummed over views)")
ax.set_xlabel("planes")
ax.set_ylabel("radial")

ax = axes[0, 1]
ax.imshow(sino_unlisted.sum(axis=v_ax), aspect="auto", vmin=0, vmax=vmax1)
ax.set_title("Unlisted span-1\n(summed over views)")
ax.set_xlabel("planes")

ax = axes[0, 2]
diff1 = sino_unlisted - y_span1_np
im = ax.imshow(diff1.sum(axis=v_ax), aspect="auto", cmap="bwr", vmin=-1, vmax=1)
ax.set_title("Difference\n(must be all zeros)")
ax.set_xlabel("planes")
fig.colorbar(im, ax=ax)

# --- row 1: span-3 comparison ---
v_ax3 = span3_desc.view_axis_num
vmax3 = float(np.max(y_span3_np))

ax = axes[1, 0]
ax.imshow(y_span3_np.sum(axis=v_ax3), aspect="auto", vmin=0, vmax=vmax3)
ax.set_title("y_span3 via\nSinogramAxialCompressionOperator")
ax.set_xlabel("planes")
ax.set_ylabel("radial")

ax = axes[1, 1]
ax.imshow(sino_span3_unlisted.sum(axis=v_ax3), aspect="auto", vmin=0, vmax=vmax3)
ax.set_title("Unlisted directly\ninto span-3")
ax.set_xlabel("planes")

ax = axes[1, 2]
diff3 = sino_span3_unlisted - y_span3_np
im3 = ax.imshow(diff3.sum(axis=v_ax3), aspect="auto", cmap="bwr", vmin=-1, vmax=1)
ax.set_title("Span-3 difference\n(must be all zeros)")
ax.set_xlabel("planes")
fig.colorbar(im3, ax=ax)

fig.suptitle("Non-TOF sinogram round-trips  (radial × planes, summed over views)")
fig.tight_layout()
fig.show()

# %%
# TOF comparison: sinogram summed over TOF bins.

fig2, axes2 = plt.subplots(1, 3, figsize=(13, 4))

y_tof_spatial = y_tof_np.sum(axis=-1)
sino_tof_spatial = sino_tof_unlisted.sum(axis=-1)
vmax_tof = float(np.max(y_tof_spatial))

ax = axes2[0]
ax.imshow(y_tof_spatial.sum(axis=v_ax), aspect="auto", vmin=0, vmax=vmax_tof)
ax.set_title("y_tof (ground truth,\nTOF-summed, view-summed)")
ax.set_xlabel("planes")
ax.set_ylabel("radial")

ax = axes2[1]
ax.imshow(sino_tof_spatial.sum(axis=v_ax), aspect="auto", vmin=0, vmax=vmax_tof)
ax.set_title("Unlisted TOF span-1\n(TOF-summed, view-summed)")
ax.set_xlabel("planes")

ax = axes2[2]
diff_tof = sino_tof_unlisted - y_tof_np
im_tof = ax.imshow(
    diff_tof.sum(axis=(-1, v_ax)), aspect="auto", cmap="bwr", vmin=-1, vmax=1
)
ax.set_title("TOF difference\n(must be all zeros)")
ax.set_xlabel("planes")
fig2.colorbar(im_tof, ax=ax)

fig2.suptitle(
    "TOF sinogram round-trip  (radial × planes, summed over views and TOF bins)"
)
fig2.tight_layout()
fig2.show()
