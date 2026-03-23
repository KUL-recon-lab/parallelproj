"""
PET listmode projector (non-TOF and TOF)
========================================

In this example we show how to setup and use a PET listmode projector
for both non-TOF and TOF acquisitions, including geometrical forward
projection in listmode, image-based resolution model, and a listmode
attenuation model.
"""

# %%
import matplotlib.pyplot as plt
from vis import show_vol_cuts

import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.operators
import parallelproj.tof
from parallelproj import to_numpy_array

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
# Setup a small regular polygon PET scanner with 4 rings
# -------------------------------------------------------

num_rings = 4
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=1,
)

# %%
# Generate 4 arbitrary listmode events
# ------------------------------------

start_ring = xp.asarray([2, 1, 0, 3], device=dev)
start_xtal = xp.asarray([0, 59, 143, 75], device=dev)

end_ring = xp.asarray([2, 0, 1, 3], device=dev)
end_xtal = xp.asarray([79, 140, 33, 147], device=dev)

event_start_coordinates = scanner.get_lor_endpoints(start_ring, start_xtal)
event_end_coordinates = scanner.get_lor_endpoints(end_ring, end_xtal)

print(event_start_coordinates)
print(event_end_coordinates)

# %%
# Show the scanner geometry and the events
# ----------------------------------------

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
scanner.show_lor_endpoints(ax)
for i in range(event_start_coordinates.shape[0]):
    ax.plot(
        [float(event_start_coordinates[i, 0]), float(event_end_coordinates[i, 0])],
        [float(event_start_coordinates[i, 1]), float(event_end_coordinates[i, 1])],
        [float(event_start_coordinates[i, 2]), float(event_end_coordinates[i, 2])],
    )
fig.tight_layout()
fig.show()


# %%
# Setup a non-TOF listmode projector and a test image
# ---------------------------------------------------

img_shape = (40, 9, 40)
voxel_size = (2.0, 3.0, 2.0)

lm_proj = parallelproj.projectors.ListmodePETProjector(
    event_start_coordinates, event_end_coordinates, img_shape, voxel_size
)

x = xp.ones(img_shape, dtype=xp.float32, device=dev)

# %%
# Perform non-TOF listmode forward and back projections
# -----------------------------------------------------

x_fwd = lm_proj(x)
print(x_fwd)

# back project a list of ones
ones_list = xp.ones(lm_proj.num_events, dtype=xp.float32, device=dev)
y_back = lm_proj.adjoint(ones_list)

# %%
# Show the non-TOF backprojected list of ones
# -------------------------------------------

fig2, _, widgets2 = show_vol_cuts(
    to_numpy_array(y_back), fig_title="non-TOF back projection of ones"
)
fig2.show()

# %%
# Extend the projector to TOF
# ---------------------------
#
# TOF parameters define the timing resolution and the number/width of TOF bins.
# Here we use a TOF resolution FWHM of 30mm (approximately 200ps).

tof_params = parallelproj.tof.TOFParameters(sigma_tof=30.0 / 2.35, num_tofbins=59, tofbin_width=12.5)

# event TOF bins - valid values are between 0 and num_tofbins-1,
# with 0 being closest to the start of the LOR
event_tof_bins = xp.asarray([29, 30, 29, 28], device=dev, dtype=xp.int16)

# set the TOF parameters and event TOF bins on the projector, then enable TOF
lm_proj.tof_parameters = tof_params
lm_proj.event_tofbins = event_tof_bins
lm_proj.tof = True

# %%
# Perform TOF listmode forward and back projections
# -------------------------------------------------

x_fwd_tof = lm_proj(x)
print(x_fwd_tof)

# back project a list of ones
y_back_tof = lm_proj.adjoint(ones_list)

# %%
# Show the TOF backprojected list of ones
# ---------------------------------------

fig3, _, widgets3 = show_vol_cuts(
    to_numpy_array(y_back_tof), fig_title="TOF back projection of ones"
)
fig3.show()

# %%
# Combine the TOF listmode projector with a resolution and attenuation model
# --------------------------------------------------------------------------

# setup a simple image-based resolution model with a Gaussian FWHM of 4.5mm
res_model = parallelproj.operators.GaussianFilterOperator(
    lm_proj.in_shape, sigma=4.5 / (2.35 * lm_proj.voxel_size)
)

# define arbitrary attenuation factors
att_list = xp.asarray([0.3, 0.4, 0.2, 0.6], device=dev)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_list)

lm_proj_with_res_model_and_att = parallelproj.operators.CompositeLinearOperator(
    (att_op, lm_proj, res_model)
)

x_fwd2 = lm_proj_with_res_model_and_att(x)
print(x_fwd2)

y_back2 = lm_proj_with_res_model_and_att.adjoint(ones_list)

# %%
# Show the backprojected list of ones with resolution and attenuation model
# -------------------------------------------------------------------------

fig4, _, widgets4 = show_vol_cuts(
    to_numpy_array(y_back2), fig_title="TOF back projection with resolution and attenuation model"
)
fig4.show()
