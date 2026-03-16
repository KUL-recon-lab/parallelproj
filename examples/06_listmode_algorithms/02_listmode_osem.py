"""
TOF listmode OSEM with projection data
======================================

This example demonstrates the use of the listmode OSEM algorithm to minimize the negative Poisson log-likelihood function.

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
import matplotlib.pyplot as plt
from matplotlib import animation
from array_api_compat import size
import numpy as np

import parallelproj.operators as ppo
import parallelproj.tof as ppt
import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
import parallelproj.projectors as ppp
from parallelproj import to_numpy_array, Array

# %%
from importlib import import_module, util


# choose array backend and a device (CPU or CUDA GPU)
if util.find_spec("torch") is not None:
    xp = import_module("array_api_compat.torch")
    dev = "cuda" if xp.cuda.is_available() else "cpu"
elif util.find_spec("cupy") is not None:
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
# Setup of the sinogram forward model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_rings = 5
scanner = pps.RegularPolygonPETScannerGeometry(
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

lor_desc = ppl.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
)

proj = ppp.RegularPolygonPETProjector(
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
# Complete sinogram PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = ppt.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
if proj.tof:
    att_op = ppo.TOFNonTOFElementwiseMultiplicationOperator(proj.out_shape, att_sino)
else:
    att_op = ppo.ElementwiseMultiplicationOperator(att_sino)

res_model = ppo.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = ppo.CompositeLinearOperator((att_op, proj, res_model))

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
# Setup of the LM subset projectors and LM subset forward models
# --------------------------------------------------------------

num_subsets = 10
subset_slices = [slice(i, None, num_subsets) for i in range(num_subsets)]

lm_pet_subset_linop_seq = []

for i, sl in enumerate(subset_slices):
    subset_lm_proj = ppp.ListmodePETProjector(
        1
        * event_start_coords[
            sl, :
        ],  # need to copy since we need contiguous arrays for the subset projectors
        1 * event_end_coords[sl, :],
        proj.in_shape,
        proj.voxel_size,
        proj.img_origin,
    )

    # recalculate the attenuation factor for all LM events
    # this needs to be a non-TOF projection
    subset_att_list = xp.exp(-subset_lm_proj(x_att))

    # enable TOF in the LM projector
    subset_lm_proj.tof_parameters = proj.tof_parameters
    if proj.tof:
        # we need to make a copy of the 1D subset event_tofbins array
        # stupid way of doing this, but torch asarray copy doesn't seem to work
        subset_lm_proj.event_tofbins = 1 * event_tofbins[sl]
        subset_lm_proj.tof = proj.tof

    subset_lm_att_op = ppo.ElementwiseMultiplicationOperator(subset_att_list)

    lm_pet_subset_linop_seq.append(
        ppo.CompositeLinearOperator((subset_lm_att_op, subset_lm_proj, res_model))
    )

lm_pet_subset_linop_seq = ppo.LinearOperatorSequence(lm_pet_subset_linop_seq)

# create the contamination list
contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=xp.float32,
)


# %%
# LM OSEM reconstruction
# ----------------------
#
# The EM update that can be used in LM-OSEM is given by
#
# .. math::
#     x^+ = \frac{x}{(A^k)^H 1} (A_{LM}^k)^H \frac{1}{A_{LM}^k x + s_{LM}^k}
#
# to calculate the minimizer of :math:`f(x)` iteratively.


def lm_em_update(
    x_cur: Array,
    op: ppo.LinearOperator,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """LM EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    op : ppo.LinearOperator
        subset listmode linear forward operator
    s : Array
        subset contamination list
    adjoint_ones : Array
        adjoint of ones of the non-LM (the complete) operator
        divided by the number of subsets

    Returns
    -------
    Array
        _description_
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(1 / ybar) / adjoint_ones


# %%
# Run LM OSEM iterations
# ----------------------

# number of MLEM iterations
num_iter = 100 // num_subsets

# initialize x
x = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter):
    for k, sl in enumerate(subset_slices):
        print(f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_iter:03}", end="\r")
        x = lm_em_update(
            x,
            lm_pet_subset_linop_seq[k],
            contamination_list[sl],
            adjoint_ones / num_subsets,
        )

# %%
# Calculate the negative Poisson log-likelihood function of the reconstruction
# ----------------------------------------------------------------------------

# calculate the negative Poisson log-likelihood function of the reconstruction
exp = pet_lin_op(x) + contamination
# calculate the relative cost and distance to the optimal point
cost = float(xp.sum(exp - xp.astype(y, xp.float32) * xp.log(exp)))
print(
    f"\nLM-OSEM cost {cost:.6E} after {num_iter:03} iterations with {num_subsets} subsets"
)


# %%
# Visualize the results
# ---------------------


def _update_img(i):
    img0.set_data(x_true_np[:, :, i])
    img1.set_data(x_np[:, :, i])
    ax[0].set_title(f"true image - plane {i:02}")
    ax[1].set_title(
        f"LM OSEM iteration {num_iter} - {num_subsets} subsets - plane {i:02}"
    )
    return (img0, img1)


x_true_np = to_numpy_array(x_true)
x_np = to_numpy_array(x)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
vmax = x_np.max()
img0 = ax[0].imshow(x_true_np[:, :, 0], cmap="Greys", vmin=0, vmax=vmax)
img1 = ax[1].imshow(x_np[:, :, 0], cmap="Greys", vmin=0, vmax=vmax)
ax[0].set_title(f"true image - plane {0:02}")
ax[1].set_title(f"LM OSEM iteration {num_iter} - {num_subsets} subsets - plane {0:02}")
fig.tight_layout()
ani = animation.FuncAnimation(fig, _update_img, x_np.shape[2], interval=200, blit=False)
fig.show()
