"""
Convergence comparison: MLEM vs OSEM vs SVRG
=============================================

This example compares the convergence speed (per epoch) of three algorithms
for minimising the negative Poisson log-likelihood

.. math::
    f(x) = \\sum_i \\bar{y}_i - y_i \\log \\bar{y}_i, \\qquad \\bar{y}(x) = A x + s

subject to :math:`x \\geq 0`:

* **MLEM** -- expectation-maximisation on the full data; one iteration = one
  full data pass; guaranteed to converge but slow per epoch.
* **OSEM** -- ordered-subsets EM; one epoch = :math:`m` subset updates ~= one
  full data pass; fast empirical convergence but *no* convergence guarantee.
* **SVRG** -- stochastic variance-reduced gradient with subsets; one epoch =
  :math:`m` variance-reduced subset updates; provably convergent like MLEM
  while achieving the fast per-epoch progress of OSEM.

.. note::
    For SVRG, one epoch requires **two** full data passes: one to compute
    the snapshot gradients at the anchor point, and one for the
    :math:`m` variance-reduced subset updates.  The epoch axis in the
    convergence plot therefore understates the true computational cost of
    SVRG relative to OSEM by a factor of roughly two.
"""

# %%
from __future__ import annotations
from collections.abc import Sequence
import matplotlib.pyplot as plt
from copy import copy
import numpy as np

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array
from parallelproj.functions import NegPoissonLogL, C2AffineObjective, C1Function

from vis import show_vol_cuts
from img import elliptic_cylinder_phantom

# %%
from array_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)

# %%

# number of subsets for OSEM and SVRG
num_subsets = 24

# if run on a CPU limit the number of "OSEM / SVRG epochs"
num_epochs_mlem = 120 if dev == "cpu" else 1200
num_epochs = num_epochs_mlem // num_subsets

# run MLEM only on GPU - it is too slow on CPU for the number of iterations used here
run_mlem = dev != "cpu"

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
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# setup a simple test image containing a few "hot rods"
x_true = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

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

# For TOF, att_sino has no TOF-bins dimension while the projector output does.
# broadcast_to adds a trailing singleton via expand_dims and broadcasts it over
# the TOF-bins axis without copying data (zero-stride view).
att_values = (
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
    if proj.tof
    else att_sino
)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[2.0 / (2.35 * float(vs)) for vs in proj.voxel_size]
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

    # same TOF/non-TOF broadcasting as for the full operator above
    att_values_k = (
        xp.broadcast_to(
            xp.expand_dims(att_sino[subset_slices_non_tof[i]], axis=-1),
            subset_proj.out_shape,
        )
        if subset_proj.tof
        else att_sino[subset_slices_non_tof[i]]
    )
    subset_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
        att_values_k
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
# EM update
# ---------
#
# The EM update used in MLEM and OSEM is :cite:p:`Dempster1977`
# :cite:p:`Shepp1982` :cite:p:`Lange1984` :cite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{A^H 1} A^H \frac{y}{A x + s}
#
# which can be rewritten as a preconditioned gradient descent step with
# diagonal preconditioner :math:`D = \operatorname{diag}(x / (A^H 1))`:
#
# .. math::
#     x^+ = x - D \, \nabla_x f(x).
#
# We implement this as a single function used by both MLEM and OSEM.


def em_update(
    x_cur: Array,
    negpoissonlogl: C1Function,
    adj_ones: Array,
) -> Array:
    """EM update re-written as preconditioned GD step"""
    em_diag_precond = x_cur / adj_ones
    return x_cur - em_diag_precond * negpoissonlogl.gradient(x_cur)


# %%
# Setup of objective functions and sensitivity images
# ---------------------------------------------------
#
# We define one :class:`.C2AffineObjective` per subset (for OSEM and SVRG)
# and one for the full data (for MLEM and objective evaluation).
# The sensitivity image :math:`A^H 1` and its per-subset counterparts
# :math:`(A^k)^H 1` are precomputed once.

# calculate A_k^H 1 for all subsets k, stored as (num_subsets, *in_shape)
subset_adjoint_ones = xp.zeros(
    (num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, op in enumerate(pet_subset_linop_seq):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

subset_data_fidelities = [
    C2AffineObjective(NegPoissonLogL(y[sl]), pet_subset_linop_seq[k], contamination[sl])
    for k, sl in enumerate(subset_slices)
]

full_data_fidelity = C2AffineObjective(NegPoissonLogL(y), pet_lin_op, contamination)

# run 1 OSEM epoch as a common warm-start for MLEM, OSEM and SVRG
x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
for k in range(len(subset_slices)):
    print(f"warm-start OSEM subset {(k+1):03} / {num_subsets:03}", end="\r")
    x_init = em_update(x_init, subset_data_fidelities[k], subset_adjoint_ones[k])
print()

# full sensitivity image: A^H 1 = sum of all subset adjoint ones
adjoint_ones = xp.sum(subset_adjoint_ones, axis=0)

# %%
# MLEM
# ----
#
# One MLEM iteration uses the full data (:math:`A`, :math:`y`, :math:`s`)
# and the full sensitivity image :math:`A^H 1`.  It is guaranteed to
# converge to the maximum-likelihood solution but requires one full data
# pass per iteration.
#
# .. note::
#     MLEM is only run when ``run_mlem`` is ``True`` (i.e. on CUDA devices).
#     On CPU it is skipped because the large number of iterations is too slow.
if run_mlem:
    df_mlem = xp.zeros(num_epochs_mlem, dtype=xp.float32, device=dev)
    x_mlem = xp.asarray(x_init, copy=True)
    for i in range(num_epochs_mlem):
        print(f"MLEM epoch {(i + 1):04} / {num_epochs_mlem:04}", end="\r")
        x_mlem = em_update(x_mlem, full_data_fidelity, adjoint_ones)
        df_mlem[i] = full_data_fidelity(x_mlem)
    print()

# %%
# OSEM
# ----
#
# One OSEM epoch cycles through all :math:`m` subsets, each using the
# subset operator :math:`A^k`, subset data :math:`y^k`, contamination
# :math:`s^k`, and subset sensitivity :math:`(A^k)^H 1`.  Fast empirical
# convergence but no convergence guarantee.

df_osem = xp.zeros(num_epochs, dtype=xp.float32, device=dev)
x_osem = xp.asarray(x_init, copy=True)
for i in range(num_epochs):
    for k in range(len(subset_slices)):
        print(
            f"OSEM epoch {(i+1):04} / {num_epochs:04}, subset {(k+1):04} / {num_subsets:04}",
            end="\r",
        )
        x_osem = em_update(x_osem, subset_data_fidelities[k], subset_adjoint_ones[k])

    df_osem[i] = full_data_fidelity(x_osem)
print()

# %%
# SVRG
# ----
#
# Each SVRG epoch consists of two phases:
#
# 1. **Anchor phase** (every other epoch): compute and store all :math:`m`
#    subset gradients at the current point :math:`\tilde{x}`, then take a
#    full gradient step.
# 2. **Variance-reduced subset updates**: for each subset :math:`k`, form
#    the variance-reduced gradient
#
#    .. math::
#        g^{VR}_k = m \left( \nabla f_k(x) - \tilde{g}_k \right)
#                   + \sum_{j=1}^m \tilde{g}_j
#
#    where :math:`\tilde{g}_k = \nabla f_k(\tilde{x})` are the stored
#    anchor gradients.
#
# .. note::
#     The anchor phase requires one full pass through all subsets to
#     compute :math:`\tilde{g}_1, \ldots, \tilde{g}_m`.  Therefore, each
#     SVRG epoch that includes an anchor phase costs **two full data
#     passes** (one for the snapshot, one for the :math:`m` subset
#     updates), compared to one full data pass for OSEM or MLEM.
#     The epoch axis in the convergence plot understates SVRG's
#     computational cost relative to OSEM by a factor of roughly two.


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
    m = len(subset_obj_functions)
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


# start SVRG from the same warm-start as OSEM and MLEM
x_svrg = xp.asarray(x_init, copy=True)

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
            f"SVRG epoch {(epoch+1):04} / {num_epochs:04}, subset {(k+1):04} / {num_subsets:04}",
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
# Convergence comparison
# ----------------------
#
# We plot the negative Poisson log-likelihood vs epoch for OSEM and SVRG,
# and overlay two horizontal reference lines showing where MLEM stands after
# ``num_epochs`` and ``num_epochs_mlem`` iterations respectively.
# One epoch of OSEM or SVRG corresponds to one cycle through all subsets
# (roughly one full data pass for OSEM, roughly two for SVRG).

# data-pass counts
# OSEM:  1 full data pass per epoch
# SVRG:  2 passes on anchor epochs (snapshot + subset updates),
#         1 pass on non-anchor epochs (subset updates only)
# MLEM:  1 pass per iteration
epochs = np.arange(1, num_epochs + 1)
osem_passes = epochs.copy()

svrg_passes_per_epoch = np.where(np.arange(num_epochs) % 2 == 0, 2, 1)
svrg_cumulative_passes = np.cumsum(svrg_passes_per_epoch)

max_passes = int(svrg_cumulative_passes[-1])
if run_mlem:
    df_mlem_trimmed = to_numpy_array(df_mlem[:max_passes])

if run_mlem:
    df_min = min(float(xp.min(df_mlem)), float(xp.min(df_osem)), float(xp.min(df_svrg)))
else:
    df_min = min(float(xp.min(df_osem)), float(xp.min(df_svrg)))
df_max = float(df_osem[0])  # use first OSEM epoch as upper limit

osem_label = f"OSEM ({num_subsets} subsets)"
svrg_label = f"SVRG ({num_subsets} subsets, step={svrg_step_size:.1f})"
mlem_label = "MLEM"

fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

# --- left: vs epoch ---
axs[0].plot(epochs, to_numpy_array(df_osem), label=osem_label, marker="o")
axs[0].plot(epochs, to_numpy_array(df_svrg), label=svrg_label, marker="o")
if run_mlem:
    axs[0].axhline(
        float(df_mlem[50 - 1]),
        label=f"{mlem_label} (50 iter.)",
        ls="--",
        color="gray",
    )
    axs[0].axhline(
        float(df_mlem[100 - 1]),
        label=f"{mlem_label} (100 iter.)",
        ls="--",
        color="gray",
    )
    axs[0].axhline(
        float(df_mlem[-1]),
        label=f"{mlem_label} ({num_epochs_mlem} iter.)",
        ls="--",
        color="black",
    )
axs[0].set_ylim(df_min, df_max)
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Negative Poisson log-likelihood")
axs[0].legend()
axs[0].grid(ls=":")

# --- right: vs full data passes ---
axs[1].plot(osem_passes, to_numpy_array(df_osem), label=osem_label, marker="o")
axs[1].plot(
    svrg_cumulative_passes, to_numpy_array(df_svrg), label=svrg_label, marker="o"
)
if run_mlem:
    axs[1].axhline(
        float(df_mlem[50 - 1]),
        label=f"{mlem_label} (50 iter.)",
        ls=":",
        color="gray",
    )
    axs[1].axhline(
        float(df_mlem[100 - 1]),
        label=f"{mlem_label} (100 iter.)",
        ls="--",
        color="gray",
    )
    axs[1].axhline(
        float(df_mlem[-1]),
        label=f"{mlem_label} ({num_epochs_mlem} iter.)",
        ls="--",
        color="black",
    )
axs[1].set_ylim(df_min, df_max)
axs[1].set_xlabel("Full data passes")
axs[1].set_ylabel("Negative Poisson log-likelihood")
axs[1].legend()
axs[1].grid(ls=":")

fig.show()

# %%
fig, axs, widgets = show_vol_cuts(
    to_numpy_array(x_osem), voxel_size=voxel_size, fig_title="OSEM result"
)
fig.show()

# %%
fig2, axs2, widgets2 = show_vol_cuts(
    to_numpy_array(x_svrg), voxel_size=voxel_size, fig_title="SVRG result"
)
fig2.show()
