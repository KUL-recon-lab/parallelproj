"""
PET listmode projector (non-TOF and TOF)
========================================

:bdg-primary:`projectors` :bdg-secondary:`beginner` :bdg-light:`CPU or GPU`

In listmode, every detected coincidence is stored as an individual event
instead of being histogrammed into a sinogram.  A listmode projector maps an
image to one value per event (forward projection) and a list of event values
back to an image (back projection).  This is efficient for acquisitions with
fewer events than sinogram bins, e.g. short time frames or TOF data.

**You will learn**

- how to create a :class:`.ListmodePETProjector` from the world coordinates
  of the two LOR endpoints of every event,
- how to forward and back project in listmode without TOF,
- how to switch the projector to TOF with :class:`.TOFParameters` and one
  TOF bin per event,
- how to combine the listmode projector with an image-based resolution model
  and per-event attenuation factors.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_01_run_regular_polygon_pet_scanner.py`
(scanner geometry and LOR endpoints) and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF parameters, resolution and attenuation model for sinograms).
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

import parallelproj.operators
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj._examples_utils import suggest_array_backend_and_device

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()


def view_from_front(ax) -> None:
    """3D view from in front of the scanner (looking along +x2, +x1 down)"""
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")


# %%
# 1. A small scanner
# ------------------
#
# We use a small regular polygon scanner with 12 sides, 15 LOR endpoints per
# side (180 per ring) and 4 rings between -10 mm and +10 mm.

num_rings = 4
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

# %%
# 2. Four listmode events
# -----------------------
#
# An event is defined by the ring and in-ring indices of its two detectors.
# :meth:`.RegularPolygonPETScannerGeometry.get_lor_endpoints` converts them to
# world coordinates (in mm), which is all the listmode projector needs.  We
# use 4 arbitrary events; the printed arrays hold the start and end points
# (one row ``(x0, x1, x2)`` per event).

start_ring = xp.asarray([2, 1, 0, 3], device=dev)
start_xtal = xp.asarray([0, 59, 143, 75], device=dev)

end_ring = xp.asarray([2, 0, 1, 3], device=dev)
end_xtal = xp.asarray([79, 140, 33, 147], device=dev)

event_start_coordinates = scanner.get_lor_endpoints(start_ring, start_xtal)
event_end_coordinates = scanner.get_lor_endpoints(end_ring, end_xtal)

print(event_start_coordinates)
print(event_end_coordinates)

# %%
# The figure shows the LOR endpoints of the scanner (coloured by ring) and the
# LORs of the 4 events.

xstart = to_numpy_array(event_start_coordinates)
xend = to_numpy_array(event_end_coordinates)

fig = plt.figure(figsize=(6, 6), layout="constrained")
ax = fig.add_subplot(projection="3d")
view_from_front(ax)
scanner.show_lor_endpoints(ax)
for i in range(xstart.shape[0]):
    ax.plot(*zip(xstart[i], xend[i]), lw=2)
ax.set_title("scanner and the LORs of 4 listmode events")
fig.show()

# %%
# Events 0 and 3 connect two detectors in the same ring, events 1 and 2 are
# oblique LORs between rings 0 and 1.

# %%
# 3. Non-TOF listmode projector
# -----------------------------
#
# :class:`.ListmodePETProjector` takes the event start and end coordinates,
# the image shape and the voxel size (in mm).  The image covers
# 80 mm x 80 mm x 27 mm.  As test image we use a uniform image of ones.

img_shape = (40, 40, 9)
voxel_size = (2.0, 2.0, 3.0)

lm_proj = parallelproj.projectors.ListmodePETProjector(
    event_start_coordinates, event_end_coordinates, img_shape, voxel_size
)

x = xp.ones(img_shape, dtype=xp.float32, device=dev)

# %%
# The forward projection gives one value per event: for an image of ones, the
# intersection length (in mm) of each LOR with the image volume.

x_fwd = lm_proj(x)
print(x_fwd)

# %%
# The adjoint (back projection) of a list of ones adds a 1 along each event
# LOR to the image.

ones_list = xp.ones(lm_proj.num_events, dtype=xp.float32, device=dev)
y_back = lm_proj.adjoint(ones_list)

# %%
# 4. TOF listmode projector
# -------------------------
#
# :class:`.TOFParameters` define the timing resolution and the number and
# width of the TOF bins along each LOR.  Here we use a TOF resolution of
# 30 mm FWHM (about 200 ps) and 59 TOF bins of 12.5 mm.  In listmode, every
# event has a single TOF bin: valid values are 0 to ``num_tofbins - 1``, with
# bin 0 closest to the start point of the LOR.  Bin 29 is the central bin.

tof_params = parallelproj.tof.TOFParameters(
    sigma_tof=30.0 / 2.3548, num_tofbins=59, tofbin_width=12.5
)
event_tof_bins = xp.asarray([29, 30, 29, 28], device=dev, dtype=xp.int16)

# set the TOF parameters and event TOF bins, then enable TOF
lm_proj.tof_parameters = tof_params
lm_proj.event_tofbins = event_tof_bins
lm_proj.tof = True

# %%
# The TOF forward projection of the uniform image is the weighted line
# integral over a TOF kernel centred on the event's TOF bin.  It is close to
# the TOF bin width (12.5 mm) for all events, since all these TOF bins lie
# well inside the image.

x_fwd_tof = lm_proj(x)
print(x_fwd_tof)

y_back_tof = lm_proj.adjoint(ones_list)

# %%
# 5. Resolution and attenuation model
# -----------------------------------
#
# A complete listmode model applies an image-based resolution model before
# the projection and multiplies each event by its attenuation factor after
# it.  :class:`.CompositeLinearOperator` chains the operators (applied from
# right to left): a Gaussian filter with 4.5 mm FWHM, the TOF listmode
# projector, and an element-wise multiplication with 4 arbitrary attenuation
# factors.

res_model = parallelproj.operators.GaussianFilterOperator(
    lm_proj.in_shape, sigma=[4.5 / (2.3548 * float(vs)) for vs in lm_proj.voxel_size]
)

att_list = xp.asarray([0.3, 0.4, 0.2, 0.6], device=dev)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_list)

lm_proj_with_res_model_and_att = parallelproj.operators.CompositeLinearOperator(
    (att_op, lm_proj, res_model)
)

x_fwd2 = lm_proj_with_res_model_and_att(x)
print(x_fwd2)

y_back2 = lm_proj_with_res_model_and_att.adjoint(ones_list)

# %%
# Each value is the TOF forward projection from section 4 times the
# attenuation factor of the event (e.g. 0.3 x 12.48 = 3.75): smoothing a
# uniform image does not change it away from the image border.

# %%
# 6. The three back projections
# -----------------------------
#
# The LORs are thin lines that do not all pass through the central slices of
# the image, so instead of single cuts we show the sum of each back
# projection along ``x2`` (front view of the scanner, ``x1`` down) and along
# ``x1`` (``x2`` vertical).  Interactively, ``show_vol_cuts`` from
# ``parallelproj._examples_utils`` lets you scroll through all slices.

back_projections = {
    "non-TOF": y_back,
    "TOF": y_back_tof,
    "TOF + resolution + attenuation": y_back2,
}
half_x0 = img_shape[0] * voxel_size[0] / 2
half_x1 = img_shape[1] * voxel_size[1] / 2
half_x2 = img_shape[2] * voxel_size[2] / 2

fig, axes = plt.subplots(
    2,
    3,
    figsize=(10, 5.2),
    height_ratios=(2 * half_x1, 2 * half_x2),
    layout="constrained",
)
for col, (name, vol) in enumerate(back_projections.items()):
    vol = to_numpy_array(vol)
    axes[0, col].imshow(
        vol.sum(axis=2).T,
        cmap="Greys",
        extent=(-half_x0, half_x0, half_x1, -half_x1),
    )
    axes[1, col].imshow(
        vol.sum(axis=1).T,
        cmap="Greys",
        origin="lower",
        extent=(-half_x0, half_x0, -half_x2, half_x2),
    )
    axes[0, col].set_title(name)
    axes[0, col].set_ylabel("x1 [mm]")
    axes[1, col].set_ylabel("x2 [mm]")
    axes[1, col].set_xlabel("x0 [mm]")
fig.suptitle("back projections of a list of ones")
fig.show()

# %%
# Without TOF, the back projection spreads each event uniformly along its
# LOR.  With TOF, each event is concentrated around its TOF bin (here close to
# the LOR centre), and the resolution model blurs it further.  The attenuation
# factors weight the events: event 3 (factor 0.6, the LOR running from the
# lower right to the left) is the brightest one in the front view of the last
# column.

# %%
# Summary
# -------
#
# - :class:`.ListmodePETProjector` projects along the LORs given by the world
#   coordinates of the start and end point of every event; its output has one
#   value per event.
# - Setting ``tof_parameters`` and ``event_tofbins`` and enabling ``tof`` turns
#   it into a TOF listmode projector (TOF bin 0 is closest to the start point).
# - :class:`.CompositeLinearOperator` combines it with a resolution model
#   (:class:`.GaussianFilterOperator`) and per-event attenuation factors
#   (:class:`.ElementwiseMultiplicationOperator`); forward and adjoint work on
#   the combined operator.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_05_run_unlister.py`
#    -- converting listmode events to sinograms (unlisting) and the TOF sign
#    convention of listmode events.

# sphinx_gallery_thumbnail_number = 2
