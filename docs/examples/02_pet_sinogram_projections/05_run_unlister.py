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
from vis import show_vol_cuts

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
y_span1_np = to_numpy_array(y_span1)

assert np.array_equal(sino_unlisted, y_span1_np), "Span-1 round-trip failed"
print("Span-1 round-trip: OK")

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

y_span3_np = to_numpy_array(op_compress(xp.astype(y_span1, xp.float32))).astype(np.int32)
sino_span3_unlisted = regular_polygon_events_to_sinogram(span3_desc, events)

assert np.array_equal(sino_span3_unlisted, y_span3_np), "Span-3 round-trip failed"
print("Span-3 round-trip: OK")

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
y_tof_np = to_numpy_array(y_tof)

assert np.array_equal(sino_tof_unlisted, y_tof_np), "TOF round-trip failed"
print("TOF round-trip: OK")

# %%
# Visualisation
# -------------
#
# :func:`.show_vol_cuts` handles both 3-D (non-TOF) and 4-D (TOF) sinograms
# with the same call.  For the 4-D case the TOF axis is moved to the front so
# the full-width leading-axis slider browses individual TOF bins.

# %%
_, _, _w1 = show_vol_cuts(
    y_span1_np,
    axis_labels=("rad", "view", "plane"),
    fig_title="y_span1  (non-TOF span-1)",
)

# %%
_, _, _w2 = show_vol_cuts(
    y_span3_np,
    axis_labels=("rad", "view", "plane"),
    fig_title="y_span3  (span-3 via SinogramAxialCompressionOperator)",
)

# %%
# Transpose from (rad, view, plane, tof) → (tof, rad, view, plane) so the
# full-width slider browses TOF bins.
_, _, _w3 = show_vol_cuts(
    y_tof_np.transpose(3, 0, 1, 2),
    axis_labels=("tof", "rad", "view", "plane"),
    fig_title="y_tof  (TOF span-1)",
)

plt.show()
