"""
TOF listmode MLEM with projection data
======================================

This example demonstrates the use of the listmode MLEM algorithm to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \\sum_{i=1}^m \\bar{y}_i (x) - \\bar{y}_i (x) \\log(y_i)

subject to

.. math::
    x \\geq 0

using the listmode linear forward model

.. math::
    \\bar{y}_{LM}(x) = A_{LM} x + s

and data stored in listmode format (event by event).
"""

# %%
from __future__ import annotations
from array_api_compat import size
from vis import show_vol_cuts
from img import elliptic_cylinder_phantom
import numpy as np

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array
from parallelproj.functions import (
    NegPoissonLogL,
    NegPoissonLogLListmode,
    C2AffineObjective,
    C1Function,
)

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
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated listmode data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram that can be converted to listmode data.

# %%
# Sinogram forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_epochs = 50
sens_factor = 0.1

num_rings = 5
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

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

x_true = elliptic_cylinder_phantom(xp, dev)


# %%
# Attenuation image and sinogram setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# %%
# Complete PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

# For TOF, att_sino has no TOF-bins dimension while the projector output does.
# broadcast_to adds a trailing singleton via expand_dims and broadcasts it over
# the TOF-bins axis without copying data (zero-stride view).
att_values = (
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
    if proj.tof
    else att_sino
)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    sens_factor * att_values
)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# Simulation of sinogram projection data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a contant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.int16,
)

# %%
# Conversion of the emission sinogram to listmode
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` we can convert an
# integer non-TOF or TOF sinogram to an event list for listmode processing.
#
# **Note:** The create event list is sorted and should be shuffled running LM-MLEM.

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y
)

# %%
# Setup of the LM projector and LM forward model
# ----------------------------------------------

lm_proj = parallelproj.projectors.ListmodePETProjector(
    event_start_coords,
    event_end_coords,
    img_shape,
    voxel_size,
    proj.img_origin,
)

# recalculate the attenuation factor for all LM events
# this needs to be a non-TOF projection
att_list = xp.exp(-lm_proj(x_att))
lm_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    sens_factor * att_list
)

# enable TOF in the LM projector
lm_proj.tof_parameters = proj.tof_parameters
if proj.tof:
    lm_proj.event_tofbins = event_tofbins
    lm_proj.tof = proj.tof

# create the contamination list
contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=xp.float32,
)

lm_pet_lin_op = parallelproj.operators.CompositeLinearOperator(
    (lm_att_op, lm_proj, res_model)
)

# %%
# Setup sinogram and listmode negative Poisson log-likelihood functions
# ---------------------------------------------------------------------

sinogram_neg_logL = C2AffineObjective(NegPoissonLogL(y), pet_lin_op, contamination)

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)
lm_neg_logL = NegPoissonLogLListmode(
    lm_pet_lin_op, adjoint_ones, contamination_list, float(xp.sum(contamination))
)
#
# %%
# Calculate the gradients of the negative Poisson log-likelihood functions in sinogram and listmode
# -------------------------------------------------------------------------------------------------

x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)

df_sino = sinogram_neg_logL(x_init)
df_lm = lm_neg_logL(x_init)

print(
    f"Negative Poisson log-likelihood at x_init: {df_sino:.4E} (sinogram), {df_lm:.4E} (listmode)"
)

grad_sinogram = sinogram_neg_logL.gradient(x_init)
grad_lm = lm_neg_logL.gradient(x_init)

# %%
vmax = float(xp.max(xp.abs(x_init / adjoint_ones) * grad_sinogram))

fig_gs, _, widgets_gs = show_vol_cuts(
    to_numpy_array((x_init / adjoint_ones) * grad_sinogram),
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
)
fig_gs.show()

# %%
fig_glm, _, widgets_glm = show_vol_cuts(
    to_numpy_array((x_init / adjoint_ones) * grad_lm),
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
)
fig_glm.show()


# %%
# MLEM reconstruction (in sinogram and listmode)
# ----------------------------------------------


def em_update(
    x_cur: Array,
    negpoissonlogl: C1Function,
    adj_ones: Array,
) -> Array:
    """EM update re-written as preconditioned GD step"""
    em_diag_precond = x_cur / adj_ones

    return x_cur - em_diag_precond * negpoissonlogl.gradient(x_cur)


# run MLEM with sinogram data
x_mlem_sino = xp.asarray(x_init, copy=True)
for i in range(num_epochs):
    print(f"MLEM epoch {(i + 1):04} / {num_epochs:04}", end="\r")
    x_mlem_sino = em_update(x_mlem_sino, sinogram_neg_logL, adjoint_ones)
print()

# run MLEM with listmode data
x_mlem_lm = xp.asarray(x_init, copy=True)
for i in range(num_epochs):
    print(f"LM-MLEM epoch {(i + 1):04} / {num_epochs:04}", end="\r")
    x_mlem_lm = em_update(x_mlem_lm, lm_neg_logL, adjoint_ones)
print()

# %%

vmax = float(xp.max(x_mlem_sino))

fig_xsino, _, widgets_xsino = show_vol_cuts(
    to_numpy_array(x_mlem_sino),
    vmin=0,
    vmax=vmax,
)
fig_xsino.show()

# %%
fig_xlm, _, widgets_xlm = show_vol_cuts(
    to_numpy_array(x_mlem_lm),
    vmin=0,
    vmax=vmax,
)
fig_xlm.show()
