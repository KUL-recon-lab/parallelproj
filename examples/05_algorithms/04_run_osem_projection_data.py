"""
TOF OSEM with projection data
=============================

This example demonstrates the use of the MLEM algorithm to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \\sum_{i=1}^m \\bar{y}_i (x) - y_i \\log(\\bar{y}_i (x))

subject to

.. math::
    x \\geq 0

using the linear forward model

.. math::
    \\bar{y}(x) = A x + s

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
from array_api_compat import to_device
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
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

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

# %%
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
# ------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# %%
# Complete PET forward model setup
# --------------------------------
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
# Simulation of projection data
# -----------------------------
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
    dtype=xp.float64,
)

# %%
# Splitting of the forward model into subsets :math:`A^k`
# -------------------------------------------------------
#
# Calculate the view numbers and slices for each subset.
# We will use the subset views to setup a sequence of projectors projecting only
# a subset of views. The slices can be used to extract the corresponding subsets
# from full data or corrections sinograms.

num_subsets = 10

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear the cached LOR endpoints since we will create many copies of the projector
proj.clear_cached_lor_endpoints()
pet_subset_linop_seq = []

# we setup a sequence of subset forward operators each constisting of
# (1) image-based resolution model
# (2) subset projector
# (3) multiplication with the corresponding subset of the attenuation sinogram
for i in range(num_subsets):
    print(f"subset {i:02} containing views {subset_views[i]}")

    # make a copy of the full projector and reset the views to project
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    if subset_proj.tof:
        subset_att_op = parallelproj.TOFNonTOFElementwiseMultiplicationOperator(
            subset_proj.out_shape, att_sino[subset_slices_non_tof[i]]
        )
    else:
        subset_att_op = parallelproj.ElementwiseMultiplicationOperator(
            att_sino[subset_slices_non_tof[i]]
        )

    # add the resolution model and multiplication with a subset of the attenuation sinogram
    pet_subset_linop_seq.append(
        parallelproj.CompositeLinearOperator(
            [
                subset_att_op,
                subset_proj,
                res_model,
            ]
        )
    )

pet_subset_linop_seq = parallelproj.LinearOperatorSequence(pet_subset_linop_seq)

# %%
# EM update to minimize :math:`f(x)`
# ----------------------------------
#
# The EM update that can be used in MLEM or OSEM is given by cite:p:`Dempster1977` :cite:p:`Shepp1982` :cite:p:`Lange1984` :cite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{A^H 1} A^H \frac{y}{A x + s}
#
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# To monitor the convergence we calculate the relative cost
#
# .. math::
#    \frac{f(x) - f(x^*)}{|f(x^*)|}
#
# and the distance to the optimal point
#
# .. math::
#    \frac{\|x - x^*\|}{\|x^*\|}.
#
#
# We setup a function that calculates a single MLEM/OSEM
# update given the current solution, a linear forward operator,
# data, contamination and the adjoint of ones.


def em_update(
    x_cur: Array,
    data: Array,
    op: parallelproj.LinearOperator,
    s: Array,
    adjoint_ones: Array,
) -> Array:
    """EM update

    Parameters
    ----------
    x_cur : Array
        current solution
    data : Array
        data
    op : parallelproj.LinearOperator
        linear forward operator
    s : Array
        contamination
    adjoint_ones : Array
        adjoint of ones

    Returns
    -------
    Array
    """
    ybar = op(x_cur) + s
    return x_cur * op.adjoint(data / ybar) / adjoint_ones


# %%
# Run the OSEM iterations
# -----------------------
#
# Note that the OSEM iterations are almost the same as the MLEM iterations.
# The only difference is that in every subset update, we pass an operator
# that projects a subset, a subset of the data and a subset of the contamination.
#
# .. math::
#     x^+ = \frac{x}{(A^k)^H 1} (A^k)^H \frac{y^k}{A^k x + s^k}
#
# The "sensitivity" images are also calculated separately for each subset.

# number of OSEM iterations
num_iter = 20 // len(pet_subset_linop_seq)

# initialize x
x = xp.ones(pet_lin_op.in_shape, dtype=xp.float64, device=dev)

# calculate A_k^H 1 for all subsets k
subset_adjoint_ones = [
    x.adjoint(xp.ones(x.out_shape, dtype=xp.float64, device=dev))
    for x in pet_subset_linop_seq
]

# OSEM iterations
for i in range(num_iter):
    for k, sl in enumerate(subset_slices):
        print(f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_iter:03}", end="\r")
        x = em_update(
            x, y[sl], pet_subset_linop_seq[k], contamination[sl], subset_adjoint_ones[k]
        )

# %%
# Calculation of the negative Poisson log-likelihood function of the reconstruction
# ---------------------------------------------------------------------------------

# calculate the negative Poisson log-likelihood function of the reconstruction
exp = pet_lin_op(x) + contamination
# calculate the relative cost and distance to the optimal point
cost = float(xp.sum(exp - y * xp.log(exp)))
print(
    f"\nOSEM cost {cost:.6E} after {num_iter:03} iterations with {num_subsets} subsets"
)


# %%
# Visualize the results
# ---------------------


def _update_img(i):
    img0.set_data(x_true_np[:, :, i])
    img1.set_data(x_np[:, :, i])
    ax[0].set_title(f"true image - plane {i:02}")
    ax[1].set_title(f"OSEM iteration {num_iter} - {num_subsets} subsets - plane {i:02}")
    return (img0, img1)


x_true_np = parallelproj.to_numpy_array(x_true)
x_np = parallelproj.to_numpy_array(x)

fig, ax = plt.subplots(1, 2, figsize=(10, 5))
vmax = x_np.max()
img0 = ax[0].imshow(x_true_np[:, :, 0], cmap="Greys", vmin=0, vmax=vmax)
img1 = ax[1].imshow(x_np[:, :, 0], cmap="Greys", vmin=0, vmax=vmax)
ax[0].set_title(f"true image - plane {0:02}")
ax[1].set_title(f"OSEM iteration {num_iter} - {num_subsets} subsets - plane {0:02}")
fig.tight_layout()
ani = animation.FuncAnimation(fig, _update_img, x_np.shape[2], interval=200, blit=False)
fig.show()
