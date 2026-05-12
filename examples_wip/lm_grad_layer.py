"""
Neg Poisson logL gradient layer (in listmode)
=============================================

This example demonstrates how to calculate the gradient of the negative Poisson log-likelihood
using a listmode projector and a sinogram projector.
Moreover, it shows how to calculate the Hessian(x) applied to an image needed in the backward pass
of an autograd layer.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
from __future__ import annotations
from parallelproj import Array

import array_api_compat.numpy as np
import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device, size
import array_api_compat.numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from copy import copy

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    if parallelproj.cuda_present:
        dev = "cuda"
    else:
        dev = "cpu"


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

num_rings = 5
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# setup a simple test image containing a few "hot rods"
x_true = xp.ones(proj.in_shape, device=dev, dtype=xp.float32)
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_true[(c0 - 2) : (c0 + 2), (c1 - 2) : (c1 + 2), :] = 5.0
x_true[4, c1, 2:] = 5.0
x_true[c0, 4, :-2] = 5.0

x_true[:2, :, :] = 0
x_true[-2:, :, :] = 0
x_true[:, :2, :] = 0
x_true[:, -2:, :] = 0


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
proj.tof_parameters = parallelproj.TOFParameters(
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
att_op = parallelproj.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=[4.5 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

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
    np.random.poisson(parallelproj.to_numpy_array(noise_free_data)),
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

lm_proj = parallelproj.ListmodePETProjector(
    event_start_coords,
    event_end_coords,
    proj.in_shape,
    proj.voxel_size,
    proj.img_origin,
)

# recalculate the attenuation factor for all LM events
# this needs to be a non-TOF projection
att_list = xp.exp(-lm_proj(x_att))
lm_att_op = parallelproj.ElementwiseMultiplicationOperator(att_list)

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

lm_pet_lin_op = parallelproj.CompositeLinearOperator((lm_att_op, lm_proj, res_model))

# %%
# calculate what is needed for a pytorch negative Poisson logL gradient descent layer
# note that pet_lin_op in the current form, because of attenuation, is object dependent
# the same holds for adjoint_ones (should be precomputed and saved to disk)

# calculate the gradient of the negative Poisson log-likelihood for a random image x
np.random.seed(1)
x = xp.asarray(np.random.rand(*proj.in_shape), device=dev, dtype=xp.float32)

# affine forward model evaluated at x
z_sino = pet_lin_op(x) + contamination

adjoint_ones = pet_lin_op.adjoint(xp.ones(z_sino.shape, device=dev, dtype=xp.float32))
sino_grad = adjoint_ones - pet_lin_op.adjoint(y / z_sino)

# %%
# calculate the gradient of the negative Poisson log-likelihood using LM data and projector

z_lm = lm_pet_lin_op(x) + contamination_list
lm_grad = adjoint_ones - lm_pet_lin_op.adjoint(1 / z_lm)

# now sino_grad and lm_grad should be numerically very close demonstrating that
# both approaches are equivalent
# but for 3D real world low count PET data, the 2nd approach is much faster and
# more memory efficient

assert xp.allclose(
    lm_grad, sino_grad, atol=1e-2
)  # lower limit to the abs tolerance is needed

# to minimize computation time, we should keep z_sino / z_lm in memory (using the ctx object) after the fwd pass
# however, if memory is limited, we could also recompute it in the backward pass


# %%
# calculate the Hessian(x) applied to another random image w
# if the forwad pass computes the gradient of the negative Poisson log-likelihood
# the backward pass needs to compute the Hessian(x) applied to an image w

# grad_output next to ctx is usually the input passed to the backward pass
grad_output = xp.asarray(np.random.rand(*proj.in_shape), device=dev, dtype=xp.float32)
hess_app_grad_output = pet_lin_op.adjoint(y * pet_lin_op(grad_output) / (z_sino**2))
hess_app_grad_output_lm = lm_pet_lin_op.adjoint(lm_pet_lin_op(grad_output) / z_lm**2)

# again both ways of computing the Hessian to grad_output should be numerically very close
# the 2nd way should be faster and more memory efficient for real world low count PET data

assert xp.allclose(
    hess_app_grad_output, hess_app_grad_output_lm, atol=1e-2
)  # lower limit to the abs tolerance is needed

# the only thing that is now left is to properly wrap everything in a pytorch autograd layer
# as done in ../examples/07_torch/01_run_projection_layer.py
