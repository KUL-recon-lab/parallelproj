"""
Convergence comparison: SGD vs SVRG with logcosh regularization
================================================================

This example compares the convergence speed (per epoch) of two algorithms
for minimising the regularised negative Poisson log-likelihood

.. math::
    F(x) = \\sum_i \\bar{y}_i - y_i \\log \\bar{y}_i
           + \\beta \\, R(x),
    \\qquad \\bar{y}(x) = A x + s

where the edge-preserving logcosh penalty is

.. math::
    R(x) = \\delta \\sum_i \\log\\!\\cosh\\!\\left(\\frac{(Gx)_i}{\\delta}\\right)

and :math:`G` is the finite forward-difference operator.  The :math:`\\delta`
prefactor ensures the asymptotic gradient magnitude equals 1 regardless of
:math:`\\delta`, so the regularisation strength :math:`\\beta` retains the
same meaning across different choices of :math:`\\delta`.  The scale
:math:`\\delta` itself controls the transition between two regimes:

* **Quadratic** for :math:`|(Gx)_i| \\ll \\delta`:
  :math:`R(x) \\approx \\tfrac{1}{2\\delta}\\|Gx\\|_2^2`.
* **Linear** for :math:`|(Gx)_i| \\gg \\delta`:
  :math:`R(x) \\approx \\|Gx\\|_1 - n\\,\\delta\\log 2 \\approx \\|Gx\\|_1`.

Setting :math:`\\delta` well below the typical edge gradient places true
edges in the linear regime (edge-preserving) while penalising smooth-region
deviations quadratically.
The objective is decomposed into :math:`m` subset functions

.. math::
    f_k(x) = \\underbrace{\\sum_{i \\in S_k} \\bar{y}_i - y_i \\log \\bar{y}_i}_{\\text{subset data fidelity}}
             + \\frac{\\beta}{m} R(x),
    \\qquad k = 1, \\ldots, m,

so that :math:`F(x) = \\sum_{k=1}^m f_k(x)` exactly.  Both algorithms below
exploit this splitting:

* **SGD** -- stochastic gradient descent using ordered subsets;
  one epoch = :math:`m` subset updates ~= one full data pass; fast empirical
  convergence but *no* convergence guarantee.
* **SVRG** -- stochastic variance-reduced gradient with subsets; one epoch =
  :math:`m` variance-reduced subset updates; provably convergent while
  achieving the fast per-epoch progress of SGD.

.. note::
    For SVRG, one epoch requires **two** full data passes: one to compute
    the snapshot gradients at the anchor point, and one for the
    :math:`m` variance-reduced subset updates.  The epoch axis in the
    convergence plot therefore understates the true computational cost of
    SVRG relative to SGD by a factor of roughly two.

.. note::

   To run this example locally, download
   `example_utils.py <https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/example_utils.py>`_
   into the **same folder** as this script. Make sure ``parallelproj`` is installed.
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
from parallelproj.functions import (
    NegPoissonLogL,
    LogCosh,
    C2AffineObjective,
    C1Function,
)

from example_utils import show_vol_cuts
from example_utils import elliptic_cylinder_phantom

# %%
from example_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)

# %%

# number of subsets for SGD and SVRG
num_subsets = 12

# if run on a CPU limit the number of epochs
num_epochs = (120 if dev == "cpu" else 240) // num_subsets

# regularisation weight beta
beta = 1.0
# delta value relative to max of ground truth image for logcosh prior
delta_rel = 0.1

# step size for SGD and SVRG updates
step_size = 1.0

# factor that scales the ground truth image (also reconstruction) and the number of counts
count_factor = 1.0

# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model.

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
x_true = count_factor * elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)


# %%
# Attenuation image and sinogram setup
# ------------------------------------

x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
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

pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noisy data :math:`y` by adding Poisson noise.

noise_free_data = pet_lin_op(x_true)

contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

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

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

proj.clear_cached_lor_endpoints()
pet_subset_linop_seq = []

for i in range(num_subsets):
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
# Regularisation and subset objective functions
# ---------------------------------------------
#
# The logcosh penalty
# :math:`R(x) = \delta \sum_i \log\cosh\!\left((Gx)_i/\delta\right)` is
# built from the :class:`.FiniteForwardDifference` operator :math:`G` and
# :class:`.LogCosh`.
# The full regulariser ``reg`` (weight :math:`\beta`) is used only for
# the total objective evaluation.  Each subset function
#
# .. math::
#     f_k(x) = \sum_{i \in S_k} \left( \bar{y}_i(x) - y_i \log \bar{y}_i(x) \right) + \frac{\beta}{m} R(x)
#
# is formed by adding a :class:`.LogCosh` scaled by :math:`\beta / m` to
# the subset data fidelity, so that :math:`\sum_k f_k(x) = F(x)`.
#
# ``delta`` is set to ``delta_rel`` times the maximum of the ground truth
# image.  With ``delta_rel = 0.1`` edges with gradient equal to the image
# maximum have :math:`|(Gx)|/\delta = 10`, placing them firmly in the
# linear regime (:math:`\tanh(10) \approx 1`), while smooth-region
# gradients near zero remain quadratic.

G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)

delta = float(xp.max(x_true)) * delta_rel

reg = C2AffineObjective(LogCosh(delta=delta, beta=beta), G)

# %%
# Setup of objective functions and sensitivity image
# --------------------------------------------------
#
# We define one subset objective :math:`f_k` per subset and one full
# objective :math:`F` for evaluation, as well as the sensitivity image :math:`A^H 1`

# sensitivity image (adjoint of all-ones vector)
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

# reg/m term shared by all subset objectives
reg_per_subset = C2AffineObjective(LogCosh(delta=delta, beta=beta / num_subsets), G)

# f_k = data_fidelity_k + (beta/m) * R(x)
subset_objectives = [
    C2AffineObjective(NegPoissonLogL(y[sl]), pet_subset_linop_seq[k], contamination[sl])
    + reg_per_subset
    for k, sl in enumerate(subset_slices)
]

full_data_fidelity = C2AffineObjective(NegPoissonLogL(y), pet_lin_op, contamination)


# also setup the full objective F for evaluation of the iterates
total_objective = full_data_fidelity + reg


# %%
# Warm start
# ----------
#
# Run one SGD epoch with regularisation as a common warm-start.

x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
for k in range(num_subsets):
    print(f"warm-start SGD subset {(k+1):03} / {num_subsets:03}", end="\r")
    init_precond = x_init / (
        adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_init, x_init)
    )
    x_init = xp.clip(
        x_init - init_precond * (num_subsets * subset_objectives[k].gradient(x_init)),
        0,
        None,
    )
print()

# %%
# SGD with regularisation
# -----------------------
#
# Each SGD epoch cycles through all :math:`m` subsets.  Because
# :math:`F(x) = \sum_k f_k(x)`, the full gradient is approximated as
#
# .. math::
#     \nabla F(x) \approx m\,\nabla f_k(x)
#
# and a gradient step with a diagonal preconditioner :math:`D` is taken:
#
# .. math::
#     x^+ = \left(x - D\, m\,\nabla f_k(x)\right)_+.

df_sgd = xp.zeros(num_epochs + 1, dtype=xp.float32, device=dev)
x_sgd = xp.asarray(x_init, copy=True)
df_sgd[0] = total_objective(x_sgd)
sgd_recons = xp.zeros((num_epochs + 1,) + img_shape)
sgd_recons[0, ...] = x_sgd

for i in range(num_epochs):
    if i % 2 == 0 and i <= 4:
        sgd_precond = x_sgd / (
            adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_sgd, x_sgd)
        )
    for k in range(num_subsets):
        print(
            f"SGD epoch {(i+1):04} / {num_epochs:04}, subset {(k+1):04} / {num_subsets:04}",
            end="\r",
        )
        approx_grad = num_subsets * subset_objectives[k].gradient(x_sgd)
        x_sgd = xp.clip(x_sgd - step_size * sgd_precond * approx_grad, 0, None)

    df_sgd[i + 1] = total_objective(x_sgd)
    sgd_recons[i + 1, ...] = x_sgd
print()

# %%
# SVRG with regularisation
# -------------------------
#
# Each SVRG epoch consists of two phases:
#
# 1. **Anchor phase** (every other epoch): compute and store all :math:`m`
#    subset gradients :math:`\tilde{g}_k = \nabla f_k(\tilde{x})` at the
#    current point :math:`\tilde{x}`, then take a full gradient step using
#    :math:`\nabla F(\tilde{x}) = \sum_k \tilde{g}_k`.
# 2. **Variance-reduced subset updates**: for each subset :math:`k`, form
#    the variance-reduced gradient
#
#    .. math::
#        g^{VR}_k = m \left( \nabla f_k(x) - \tilde{g}_k \right)
#                   + \sum_{j=1}^m \tilde{g}_j
#


def svrg_calc_snapshot_gradients(
    x_cur: Array,
    subset_obj_functions: Sequence[C1Function],
) -> tuple[Array, Array]:
    """Store all subset gradients at the current anchor point and return their sum."""
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
    stored_snapshot_subset_gradients: Array,
    full_snapshot_gradient: Array,
    precond: Array,
    step_size: float = 1.0,
) -> Array:
    """Single SVRG subset update with variance-reduced gradient."""
    m = len(subset_obj_functions)
    grad_k = subset_obj_functions[subset_idx].gradient(x_cur)
    approx_grad = (
        m * (grad_k - stored_snapshot_subset_gradients[subset_idx])
        + full_snapshot_gradient
    )
    return xp.clip(x_cur - step_size * precond * approx_grad, 0, None)


x_svrg = xp.asarray(x_init, copy=True)
svrg_recons = xp.zeros((num_epochs + 1,) + img_shape)
svrg_recons[0, ...] = x_svrg

df_svrg = xp.zeros(num_epochs + 1, dtype=xp.float32, device=dev)
df_svrg[0] = total_objective(x_svrg)

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond = x_svrg / (
                adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_svrg, x_svrg)
            )

        stored_grads, full_grad = svrg_calc_snapshot_gradients(
            x_svrg, subset_objectives
        )
        x_svrg = xp.clip(x_svrg - step_size * svrg_precond * full_grad, 0, None)

    for k in range(num_subsets):
        print(
            f"SVRG epoch {(epoch+1):04} / {num_epochs:04}, subset {(k+1):04} / {num_subsets:04}",
            end="\r",
        )
        x_svrg = svrg_update(
            x_svrg,
            k,
            subset_objectives,
            stored_grads,
            full_grad,
            svrg_precond,
            step_size=step_size,
        )

    df_svrg[epoch + 1] = total_objective(x_svrg)
    svrg_recons[epoch + 1, ...] = x_svrg

# %%
# Convergence comparison
# ----------------------
#
# We plot the total objective :math:`F(x)` vs epoch (left) and vs
# full data passes (right).  One epoch of SGD corresponds to one cycle
# through all subsets (roughly one full data pass).  One SVRG epoch on
# an anchor phase costs two full data passes (snapshot + subset updates),
# and one full pass otherwise.

epochs = np.arange(num_epochs + 1)
osem_passes = epochs.copy()

svrg_passes_per_epoch = np.concatenate(
    [[0], np.where(np.arange(num_epochs) % 2 == 0, 2, 1)]
)
svrg_cumulative_passes = np.cumsum(svrg_passes_per_epoch)

df_min = min(float(xp.min(df_sgd)), float(xp.min(df_svrg)))
df_max = float(df_sgd[0])

sgd_label = f"SGD ({num_subsets} subsets, step={step_size:.1f})"
svrg_label = f"SVRG ({num_subsets} subsets, step={step_size:.1f})"

fig, axs = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")

# --- left: vs epoch ---
axs[0].plot(epochs, to_numpy_array(df_sgd), label=sgd_label, marker="o")
axs[0].plot(epochs, to_numpy_array(df_svrg), label=svrg_label, marker="o")
axs[0].set_ylim(df_min, df_max)
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel(r"$F(x) = \sum_i(\bar{y}_i - y_i \log \bar{y}_i) + \beta R(x)$")
axs[0].set_title(rf"Convergence vs epoch ($\beta={beta}$)")
axs[0].legend()
axs[0].grid(ls=":")

# --- right: vs full data passes ---
axs[1].plot(osem_passes, to_numpy_array(df_sgd), label=sgd_label, marker="o")
axs[1].plot(
    svrg_cumulative_passes, to_numpy_array(df_svrg), label=svrg_label, marker="o"
)
axs[1].set_ylim(df_min, df_max)
axs[1].set_xlabel("Full data passes")
axs[1].set_ylabel(r"$F(x) = \sum_i(\bar{y}_i - y_i \log \bar{y}_i) + \beta R(x)$")
axs[1].set_title(rf"Convergence vs data passes ($\beta={beta}$)")
axs[1].legend()
axs[1].grid(ls=":")

fig.show()

# %%
fig, axs, widgets = show_vol_cuts(
    to_numpy_array(sgd_recons), voxel_size=voxel_size, fig_title="SGD result"
)
fig.show()

# %%
fig2, axs2, widgets = show_vol_cuts(
    to_numpy_array(svrg_recons), voxel_size=voxel_size, fig_title="SVRG result"
)
fig2.show()
