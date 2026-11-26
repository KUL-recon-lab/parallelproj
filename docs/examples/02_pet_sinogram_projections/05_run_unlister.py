"""
Listmode to sinogram unlisting
==============================

:bdg-primary:`projectors` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

*Unlisting* histograms listmode events into a sinogram, so that listmode data
can be reconstructed with sinogram projectors.
:func:`.regular_polygon_events_to_sinogram` does this for regular polygon
scanners, with or without TOF and for any axial compression (span).  For TOF
data, the measured arrival-time difference of an event must first be
converted to the TOF bin of the sinogram, whose numbering follows the fixed
direction of each sinogram LOR, with :func:`.detection_times_to_tof_bin`.

**You will learn**

- how to convert a sinogram to crystal-index listmode events with
  :meth:`.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`,
- how to unlist events into non-TOF span-1, span-3 and TOF sinograms with
  :func:`.regular_polygon_events_to_sinogram`, and verify the round trip,
- the *red / blue* naming of the two detectors of an event and the sign
  convention ``t_blue - t_red`` of the arrival-time difference,
- how :func:`.detection_times_to_tof_bin` maps arrival-time differences to
  sinogram TOF bins, independent of the order of the two detectors.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_04_run_michelogram_and_compression.py`
(span and axial compression),
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF sinograms) and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_04_run_pet_listmode_projector.py`
(listmode events).
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import matplotlib.pyplot as plt
import numpy as np

import parallelproj.pet_lors as ppl
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    suggest_array_backend_and_device,
)
from parallelproj.unlist import (
    detection_times_to_tof_bin,
    regular_polygon_events_to_sinogram,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()


def view_from_front(ax) -> None:
    """3D view from in front of the scanner (looking along +x2, +x1 down)"""
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")


# %%
# 1. Scanner, LOR descriptor and projector
# ----------------------------------------
#
# We simulate data for a small scanner with 5 rings and a span-1 sinogram
# with a maximum ring difference of 2.

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
# 2. Simulated TOF data
# ---------------------
#
# We forward project an elliptic cylinder phantom into a TOF sinogram with 13
# TOF bins and add Poisson noise to get integer counts ``y_tof``.  Summing
# over the TOF axis gives the non-TOF span-1 sinogram ``y_span1`` with the
# same total number of counts.

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

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
# 3. Listmode events from the sinogram
# ------------------------------------
#
# :meth:`~.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`
# turns every count of a sinogram into one event and returns a NumPy array
# with the columns ``(d_red, r_red, d_blue, r_blue, unsigned_sinogram_tof_bin)``:
# the in-ring index ``d`` and ring index ``r`` of the two detectors and the
# TOF bin of the event in the sinogram convention.  With ``shuffle=True`` the
# events are in random order, as in a real acquisition.

events = proj.convert_sinogram_to_crystal_index_events(y_tof, shuffle=True)

d_red = xp.asarray(events[:, 0], dtype=xp.int32, device=dev)
r_red = xp.asarray(events[:, 1], dtype=xp.int32, device=dev)
d_blue = xp.asarray(events[:, 2], dtype=xp.int32, device=dev)
r_blue = xp.asarray(events[:, 3], dtype=xp.int32, device=dev)
unsigned_sinogram_tof_bin = xp.asarray(events[:, 4], dtype=xp.int32, device=dev)

print(f"\nTotal events : {d_red.shape[0]}")

# %%
# 4. Non-TOF span-1 unlisting
# ---------------------------
#
# :func:`.regular_polygon_events_to_sinogram` histograms the events into the
# sinogram of the given projector.  Without TOF bins, the result must equal
# the non-TOF sinogram ``y_span1``.

sino_unlisted = regular_polygon_events_to_sinogram(proj, d_red, r_red, d_blue, r_blue)
assert bool(xp.all(sino_unlisted == y_span1)), "Span-1 round-trip failed"
print("\nSpan-1 round-trip: OK")

# %%
# The figure shows the central direct plane (ring 2 with itself) of both
# sinograms.

plane = 2
fig, axes = plt.subplots(1, 2, figsize=(8, 4.8), layout="constrained")
for ax, sino, title in zip(
    axes,
    (y_span1, sino_unlisted),
    ("simulated non-TOF sinogram", "unlisted from the events"),
):
    im = ax.imshow(
        to_numpy_array(sino[:, :, plane]).T, cmap="magma", aspect="auto"
    )
    ax.set_title(title)
    ax.set_xlabel("radial bin")
    ax.set_ylabel("view")
fig.colorbar(im, ax=axes, label="counts", shrink=0.8)
fig.suptitle(f"span-1 sinogram plane {plane}")
fig.show()

# %%
# Both noisy sinograms are identical: unlisting restores every count to the
# sinogram bin it was drawn from, independent of the order of the events.

# %%
# 5. Span-3 unlisting
# -------------------
#
# The same events can be unlisted directly into an axially compressed
# sinogram: we pass a projector with a span-3 LOR descriptor.  The result
# must equal the span-1 sinogram compressed with
# :class:`.SinogramAxialCompressionOperator`.

op_compress = ppl.SinogramAxialCompressionOperator(lor_desc, target_span=3)
span3_desc = op_compress.out_lor_descriptor

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
# 6. TOF unlisting
# ----------------
#
# With ``unsigned_sinogram_tof_bin``, the events are unlisted into the TOF
# sinogram, which must equal ``y_tof``.

sino_tof_unlisted = regular_polygon_events_to_sinogram(
    proj,
    d_red,
    r_red,
    d_blue,
    r_blue,
    unsigned_sinogram_tof_bin=unsigned_sinogram_tof_bin,
)
assert bool(xp.all(sino_tof_unlisted == y_tof)), "TOF round-trip failed"
print("TOF  round-trip: OK")

# %%
# 7. Red / blue detectors and the TOF sign
# ----------------------------------------
#
# The events above already carry their sinogram TOF bin.  Raw listmode data
# instead stores the two detectors and the difference of their photon
# arrival times.
#
# **Detector naming.** We label the two detectors of an event **red** and
# **blue**: arbitrary colour names with *no* implied ordering.  The older
# labels *d1 / d2* were dropped because "1" suggests "first to fire", but in
# raw PET data either detector can fire first.  The colour labels make clear
# that only the *identity* of the detectors is encoded, not the
# *chronological order*.
#
# .. note::
#
#    :func:`.detection_times_to_tof_bin` always expects the arrival-time
#    difference as ``t_blue - t_red`` (blue minus red, in ns).  Its sign tells
#    which detector fired *later*:
#
#    * ``t_blue - t_red > 0``: blue fired later, so the emission is closer to
#      the **red** detector.
#    * ``t_blue - t_red < 0``: red fired later, so the emission is closer to
#      the **blue** detector.
#
#    You **must** pass the difference in this fixed order.  Passing
#    ``t_red - t_blue`` silently mirrors every TOF bin assignment.
#
# **Projector direction.** The TOF sinogram projector traces each LOR in one
# fixed direction, from its **canonical xstart** detector to its **canonical
# xend** detector.  Both are defined once by the LOR descriptor and never
# change, and the TOF bins are numbered along this direction: **bin 0 is
# closest to xstart**.
#
# Because the colour labels are arbitrary, the (d_red, d_blue) pair of a
# listmode event may or may not match the (xstart, xend) order of its LOR:
#
# * ``d_red = xstart, d_blue = xend``: ``t_blue - t_red > 0`` means that the
#   emission is closer to xstart, i.e. low TOF bin numbers.  No flip is
#   needed.
# * ``d_red = xend, d_blue = xstart``: the same physical event now has
#   ``t_blue - t_red < 0``, and the mapping to sinogram TOF bins is
#   **mirrored**.
#
# :func:`.detection_times_to_tof_bin` looks up for every event whether d_red
# is xstart or xend of its LOR and applies the sign flip automatically.

# %%
# 8. A scanner for the TOF sign test
# ----------------------------------
#
# To check the convention we use a two-ring scanner with 32 detectors per
# ring (8 modules x 4 crystals) and a diameter of 600 mm, and 25 TOF bins of
# 24 mm that cover the full diameter.  Only the TOF bin assignment matters
# here, so the image has a single voxel.

radius_sign = 300.0  # mm, diameter 600 mm
num_tof_bins_sign = 25
tofbin_width_sign = 24.0  # mm, 25 x 24 mm = 600 mm

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
    num_tofbins=num_tof_bins_sign,
    tofbin_width=tofbin_width_sign,
    sigma_tof=10.0,
    tofcenter_offset=0.0,
)

# %%
# 9. Six test events
# ------------------
#
# Six events on two LORs, all in ring 0:
#
# * **Events 0-4** are on the LOR between detectors 7 and 24, for which the
#   LOR descriptor defines xstart = 7 and xend = 24.  Events 0-1 have
#   ``d_red = xstart``; events 2-4 have ``d_red = xend`` (reversed relative
#   to the projector direction).
# * **Event 5** is on the LOR between detectors 1 and 29, which is removed
#   from the sinogram by ``radial_trim``.
#
# :func:`.detection_times_to_tof_bin` converts ``t_blue - t_red`` of each
# event into the unsigned sinogram TOF bin expected by the projector.  A
# time difference of 1 ns corresponds to about 150 mm along the LOR, i.e.
# about 6 TOF bins of 24 mm away from the central bin 12.

d_red_sign = xp.asarray([7, 7, 24, 24, 24, 1], dtype=xp.int32, device=dev)
d_blue_sign = xp.asarray([24, 24, 7, 7, 7, 29], dtype=xp.int32, device=dev)

num_events = d_red_sign.shape[0]

r_red_sign = xp.zeros(num_events, dtype=xp.int32, device=dev)
r_blue_sign = xp.zeros(num_events, dtype=xp.int32, device=dev)

# dt = t_blue - t_red for each event (ns)
dt_sign = xp.asarray([+1.0, -1.0, +1.0, -1.0, 0.0, 0.0], device=dev)

ev_sino_tof_bins = detection_times_to_tof_bin(
    d_red_sign, d_blue_sign, dt_sign, proj_sign
)

for i, (dr, db, dt, tb) in enumerate(
    zip(d_red_sign, d_blue_sign, dt_sign, ev_sino_tof_bins)
):
    print(
        f"event {i}: d_red={int(dr):2d}, d_blue={int(db):2d},"
        f" sino tof bin={int(tb):2d}, dt={float(dt):+.1f} ns"
    )

unlisted_tof_sino = regular_polygon_events_to_sinogram(
    proj_sign,
    d_red_sign,
    r_red_sign,
    d_blue_sign,
    r_blue_sign,
    unsigned_sinogram_tof_bin=ev_sino_tof_bins,
)

print(f"number of listmode events:           {num_events}")
print(f"num counts in unlisted TOF sinogram: {int(xp.sum(unlisted_tof_sino))}")

# %%
# * Events 0 and 2 are on the same LOR with the same ``|t_blue - t_red|``,
#   but with the (d_red, d_blue) order reversed.  Since the projector
#   direction is fixed, they end up in **symmetric TOF bins (6 and 18) on
#   opposite sides of the LOR centre**.
# * Events 0 and 3 describe the same physical event (labels swapped *and*
#   sign of the time difference flipped) and therefore get the same TOF bin
#   6, as do events 1 and 2 (bin 18).  Event 4 (``dt = 0``) lies in the
#   central bin 12.
# * Event 5 is on a trimmed LOR and gets ``unsigned_sinogram_tof_bin = -1``.
#   Such events are silently dropped by
#   :func:`.regular_polygon_events_to_sinogram`, so the unlisted sinogram
#   contains 5 of the 6 events.

# %%
# 10. TOF bins along the LORs
# ---------------------------
#
# :meth:`.RegularPolygonPETLORDescriptor.show_tof_bins` draws all LORs of a
# sinogram view as coloured segments, one colour per TOF bin.  The left
# panel shows view 0, the right panel view 8, which contains the LOR 7-24 of
# events 0-4.  Detector indices of ring 0 (blue dots) run from 0 to 31, those
# of ring 1 (orange dots) from 32 to 63.

lim = radius_sign * 1.3
fig, axes = plt.subplots(
    1, 2, figsize=(10, 5.2), subplot_kw={"projection": "3d"}, layout="constrained"
)
for ax, view in zip(axes, (0, 8)):
    view_from_front(ax)
    lor_desc_sign.show_tof_bins(
        ax=ax,
        tof_parameters=proj_sign.tof_parameters,
        views=view,
        show_colorbar=ax is axes[-1],
    )
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, 50)
    ax.set_title(f"view {view}")
fig.axes[-1].set_yticks(range(0, num_tof_bins_sign, 4))  # TOF bin colorbar
fig.show()

# %%
# TOF bin 0 (dark blue) is always at the **xstart** end of each LOR, and the
# bin numbers increase (towards red) in the direction of xend.  In view 8,
# the LOR between detectors 7 and 24 starts at detector 7, so TOF bin 6
# (events 0 and 3) lies on the detector-7 side of the LOR centre.

# %%
# Summary
# -------
#
# - :meth:`~.RegularPolygonPETProjector.convert_sinogram_to_crystal_index_events`
#   turns a (TOF) sinogram into crystal-index events
#   ``(d_red, r_red, d_blue, r_blue, unsigned_sinogram_tof_bin)``.
# - :func:`.regular_polygon_events_to_sinogram` unlists events into the
#   sinogram of any regular polygon projector: non-TOF, TOF, span 1 or an
#   axially compressed span; unlisting the converted events restores the
#   original sinograms exactly.
# - Red and blue only identify the two detectors.  Arrival-time differences
#   are always passed as ``t_blue - t_red`` in ns.
# - Sinogram TOF bins are numbered from the canonical xstart of each LOR;
#   :func:`.detection_times_to_tof_bin` handles events whose red detector is
#   xend, and returns -1 for events outside the sinogram, which are dropped.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_06_run_tof_kernels.py`
#    -- non-Gaussian TOF kernels and per-LOR TOF parameters.

# sphinx_gallery_thumbnail_number = 1
