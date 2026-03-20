"""
TOF OSEM with projection data
=============================

This example demonstrates the use of the SVRG algorithm to minimize the negative Poisson log-likelihood function

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
from collections.abc import Sequence
import matplotlib.pyplot as plt
from matplotlib import animation
from copy import copy
import numpy as np

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array
from parallelproj.functions import NegPoissonLogL, AffineObjective, C1Function

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

# %%
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
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

# setup the attenuation multiplication operator which is different
# for TOF and non-TOF since the attenuation sinogram is always non-TOF
if proj.tof:
    att_op = parallelproj.operators.TOFNonTOFElementwiseMultiplicationOperator(
        proj.out_shape, att_sino
    )
else:
    att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=2.0 / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

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
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

# %%
# Splitting of the forward model into subsets :math:`A^k`
# -------------------------------------------------------
#
# Calculate the view numbers and slices for each subset.
# We will use the subset views to setup a sequence of projectors projecting only
# a subset of views. The slices can be used to extract the corresponding subsets
# from full data or corrections sinograms.

num_subsets = 24

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
    # make a copy of the full projector and reset the views to project
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    if subset_proj.tof:
        subset_att_op = (
            parallelproj.operators.TOFNonTOFElementwiseMultiplicationOperator(
                subset_proj.out_shape, att_sino[subset_slices_non_tof[i]]
            )
        )
    else:
        subset_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
            att_sino[subset_slices_non_tof[i]]
        )

    # add the resolution model and multiplication with a subset of the attenuation sinogram
    pet_subset_linop_seq.append(
        parallelproj.operators.CompositeLinearOperator(
            [
                subset_att_op,
                subset_proj,
                res_model,
            ]
        )
    )

pet_subset_linop_seq = parallelproj.operators.LinearOperatorSequence(
    pet_subset_linop_seq
)

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


def em_update(x_cur: Array, data_fidelity: C1Function, adjoint_ones: Array) -> Array:
    """EM update re-written as preconditioned GD step"""
    em_diag_precond = x_cur / adjoint_ones
    return x_cur - em_diag_precond * data_fidelity.gradient(x_cur)


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
num_epochs = 480 // num_subsets

# initialize x
x_osem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)

# calculate A_k^H 1 for all subsets k, stored as (num_subsets, *in_shape)
subset_adjoint_ones = xp.zeros(
    (num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, op in enumerate(pet_subset_linop_seq):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

subset_data_fidelities = [
    AffineObjective(NegPoissonLogL(y[sl]), pet_subset_linop_seq[k], contamination[sl])
    for k, sl in enumerate(subset_slices)
]

full_data_fidelity = AffineObjective(NegPoissonLogL(y), pet_lin_op, contamination)

df_osem = xp.zeros(num_epochs, dtype=xp.float32, device=dev)

# OSEM iterations
for i in range(num_epochs):
    for k in range(len(subset_slices)):
        print(f"OSEM iteration {(k+1):03} / {(i + 1):03} / {num_epochs:03}", end="\r")
        x_osem = em_update(x_osem, subset_data_fidelities[k], subset_adjoint_ones[k])

    df_osem[i] = full_data_fidelity(x_osem)

# %%
# Run the SVRG iterations
# -----------------------
#
# Each SVRG epoch consists of two phases:
#
# 1. Anchor phase: compute and store all subset gradients at the current point
# 2. Subset updates: for each subset, use the variance-reduced gradient
#
# .. math::
#     g^{VR} = m \left( \nabla f_k(x) - \tilde{g}_k \right) + \sum_{k=1}^m \tilde{g}_k
#
# where :math:`\tilde{g}_k` are the stored subset gradients at the anchor point.


def svrg_calc_snapshot_gradients(
    x_cur: Array,
    subset_obj_functions: Sequence[C1Function],
) -> tuple[Array, Array]:
    """Store all subset gradients at the current anchor point and return their sum.

    Returns
    -------
    stored_grads : Array, shape (m, *x_cur.shape)
        Stacked subset gradients evaluated at the anchor point.
    full_grad : Array, shape x_cur.shape
        Sum of all subset gradients.
    """
    m = len(subset_data_fidelities)
    stored_grads = xp.zeros((m,) + x_cur.shape, dtype=x_cur.dtype, device=dev)
    for k, df in enumerate(subset_obj_functions):
        stored_grads[k] = df.gradient(x_cur)
    full_grad = xp.sum(stored_grads, axis=0)
    return stored_grads, full_grad


def svrg_update(
    x_cur: Array,
    subset_idx: int,
    subset_obj_functions: Sequence[C1Function],
    stored_subset_gradients: Array,
    full_gradient: Array,
    precond: Array,
    step_size: float = 1.0,
) -> Array:
    """Single SVRG subset update with variance-reduced gradient."""
    m = len(subset_obj_functions)
    grad_k = subset_obj_functions[subset_idx].gradient(x_cur)
    approx_grad = m * (grad_k - stored_subset_gradients[subset_idx]) + full_gradient
    return xp.clip(x_cur - step_size * precond * approx_grad, 0, None)


x_svrg = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)

# full sensitivity image: sum of all subset adjoint ones
adjoint_ones = xp.sum(subset_adjoint_ones, axis=0)

svrg_step_size = 1.0
df_svrg = xp.zeros(num_epochs, dtype=xp.float32, device=dev)

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond = x_svrg / adjoint_ones

        stored_grads, full_grad = svrg_calc_snapshot_gradients(
            x_svrg, subset_data_fidelities
        )

        x_svrg = xp.clip(x_svrg - svrg_step_size * svrg_precond * full_grad, 0, None)

    for k in range(num_subsets):
        print(
            f"SVRG epoch {(epoch+1):03} / {num_epochs:03}, subset {(k+1):03} / {num_subsets:03}",
            end="\r",
        )
        x_svrg = svrg_update(
            x_svrg,
            k,
            subset_data_fidelities,
            stored_grads,
            full_grad,
            svrg_precond,
            step_size=svrg_step_size,
        )

    df_svrg[epoch] = full_data_fidelity(x_svrg)

# %%
# Plot the convergence of OSEM and SVRG
# --------------------------------------

epochs = np.arange(1, num_epochs + 1)

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.plot(epochs, to_numpy_array(df_osem), label="OSEM", marker="o")
ax.plot(epochs, to_numpy_array(df_svrg), label="SVRG", marker="o")
ax.set_ylim(min(float(xp.min(df_osem)), float(xp.min(df_svrg))), float(df_osem[1]))
ax.legend()
ax.grid(ls=":")
ax.set_xlabel("Epoch")
ax.set_ylabel("Data fidelity")
fig.show()
