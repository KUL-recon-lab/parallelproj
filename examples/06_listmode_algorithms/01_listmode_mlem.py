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

.. tip::
    parallelproj is python array API compatible meaning it supports different 
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
from __future__ import annotations
from parallelproj import Array

import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

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
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
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

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
if proj.tof:
    att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino
    )
else:
    att_op = parallelproj.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
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
# LM MLEM reconstruction
# ----------------------
#
# The EM update that can be used in LM-MLEM is given by
#
# .. math::
#     x^+ = \frac{x}{A^H 1} A_{LM}^H \frac{1}{A_{LM} x + s_{LM}}
#
# to calculate the minimizer of :math:`f(x)` iteratively.


def lm_em_update(
    x_cur: Array,
    op: parallelproj.LinearOperator,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """LM EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    op : parallelproj.LinearOperator
        listmode linear forward operator
    s : Array
        contamination list
    adjoint_ones : Array
        adjoint of ones of the non-LM (the complete) operator

    Returns
    -------
    Array
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(1 / ybar) / adjoint_ones


# %%
# Run LM MLEM iterations
# ----------------------

# number of MLEM iterations
num_iter = 10

# initialize x
x = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter):
    print(f"MLEM iteration {(i + 1):03} / {num_iter:03}", end="\r")
    x = lm_em_update(x, lm_pet_lin_op, contamination_list, adjoint_ones)

# %%
# Calculate the negative Poisson log-likelihood function of the reconstruction
# ----------------------------------------------------------------------------

# calculate the negative Poisson log-likelihood function of the reconstruction
exp = pet_lin_op(x) + contamination
# calculate the relative cost and distance to the optimal point
cost = float(xp.sum(exp - xp.astype(y, xp.float32) * xp.log(exp)))
print(f"\nMLEM cost {cost:.6E} after {num_iter:03} iterations")

# %%
# Visualize the results
# ---------------------


def _update_img(i):
    img0.set_data(x_true_np[:, :, i])
    img1.set_data(x_np[:, :, i])
    ax[0].set_title(f"true image - plane {i:02}")
    ax[1].set_title(f"LM MLEM iteration {num_iter} - plane {i:02}")
    return (img0, img1)


x_true_np = parallelproj.to_numpy_array(x_true)
x_np = parallelproj.to_numpy_array(x)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
vmax = x_np.max()
img0 = ax[0].imshow(x_true_np[:, :, 0], cmap="Greys", vmin=0, vmax=vmax)
img1 = ax[1].imshow(x_np[:, :, 0], cmap="Greys", vmin=0, vmax=vmax)
ax[0].set_title(f"true image - plane {0:02}")
ax[1].set_title(f"LM MLEM iteration {num_iter} - plane {0:02}")
fig.tight_layout()
ani = animation.FuncAnimation(fig, _update_img, x_np.shape[2], interval=200, blit=False)
fig.show()
