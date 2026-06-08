"""
Convergence comparison: SGD vs SVRG with regularization (sinogram and listmode)
================================================================================

This example compares sinogram and listmode variants of SGD and SVRG for
minimising the regularised negative Poisson log-likelihood

.. math::
    F(x) = \\underbrace{\\sum_i \\bar{y}_i - y_i \\log \\bar{y}_i}_{\\text{data fidelity}}
           + \\beta \\, R(x),
    \\qquad \\bar{y}(x) = A x + s

where the quadratic penalty is

.. math::
    R(x) = \\frac{1}{2} \\| G x \\|_2^2

and :math:`G` is the finite forward-difference operator.  The same objective
is expressed in two equivalent ways:

* **Sinogram** -- via :class:`.C2AffineObjective` wrapping :class:`.NegPoissonLogL`
  (operates on predicted counts :math:`\\bar{y}`).
* **Listmode** -- via :class:`.NegPoissonLogLListmode` (operates directly on the
  image :math:`x` with the forward model built in).

The objective is decomposed into :math:`m` subset functions

.. math::
    f_k(x) = \\underbrace{\\sum_{i \\in S_k} \\bar{y}_i - y_i \\log \\bar{y}_i}_{\\text{subset data fidelity}}
             + \\frac{\\beta}{m} R(x),
    \\qquad k = 1, \\ldots, m,

so that :math:`F(x) = \\sum_{k=1}^m f_k(x)` exactly.  Both algorithms exploit
this splitting:

* **SGD** -- stochastic gradient descent using ordered subsets; one epoch =
  :math:`m` subset updates ~= one full data pass; fast empirical convergence
  but *no* convergence guarantee.
* **SVRG** -- stochastic variance-reduced gradient with subsets; one epoch =
  :math:`m` variance-reduced subset updates; provably convergent while
  achieving the fast per-epoch progress of SGD.

.. note::
    For SVRG, one epoch requires **two** full data passes: one to compute the
    snapshot gradients at the anchor point, and one for the :math:`m`
    variance-reduced subset updates.  The epoch axis in the convergence plot
    therefore understates the true computational cost of SVRG relative to SGD
    by a factor of roughly two.

**Key learning goal**: Once the subset objective functions are defined (either
as sinogram :class:`.C2AffineObjective` or listmode :class:`.NegPoissonLogLListmode`
objects), the SGD and SVRG loops are *identical* -- they only depend on the
:class:`.C1Function` interface (``gradient`` method).  Regularisation is added
transparently via the :meth:`~.C1Function.__add__` operator.

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
from array_api_compat import size

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array
from parallelproj.functions import (
    NegPoissonLogL,
    NegPoissonLogLListmode,
    HalfSquaredL2Deviation,
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
num_subsets = 24

# if run on a CPU limit the number of epochs
num_epochs = (72 if dev == "cpu" else 240) // num_subsets

# sens factor
sens_factor = 0.2

# regularisation weight beta
beta = 10.0

# step size for SGD and SVRG updates
step_size = 1.0

# %%
# Sinogram forward model setup
# ----------------------------
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

# setup a simple test image
x_true = sens_factor * elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# %%
# Attenuation image and sinogram setup

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
    dtype=xp.int16,
)

# %%
# Conversion of the emission sinogram to listmode
# -----------------------------------------------
#
# Using :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` we
# convert the integer sinogram to a shuffled event list.

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y, shuffle=True
)

# %%
# Setup of the LM projector and LM forward model
# -----------------------------------------------

lm_proj = parallelproj.projectors.ListmodePETProjector(
    event_start_coords,
    event_end_coords,
    img_shape,
    voxel_size,
    proj.img_origin,
)

# recalculate the attenuation factor for all LM events (non-TOF projection)
att_list = xp.exp(-lm_proj(x_att))
lm_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_list)

# enable TOF in the LM projector
lm_proj.tof_parameters = proj.tof_parameters
if proj.tof:
    lm_proj.event_tofbins = event_tofbins
    lm_proj.tof = proj.tof

# create the contamination list (uniform, same mean as sinogram contamination)
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
# Sensitivity image and full objective functions
# ----------------------------------------------
#
# The sensitivity image :math:`A^T\mathbf{1}` is required by
# :class:`.NegPoissonLogLListmode` and serves as the EM diagonal preconditioner.
# The full objectives (sinogram and listmode) are used only for evaluation.

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

# %%
# Regularisation :math:`R(x) = \frac{1}{2} \| G x \|_2^2`
# ---------------------------------------------------------
#
# The quadratic penalty is built from the :class:`.FiniteForwardDifference`
# operator :math:`G`.  The full regulariser ``reg`` (weight :math:`\beta`)
# is used for total objective evaluation.  The per-subset regulariser
# (weight :math:`\beta/m`) is added to each subset data-fidelity function so
# that :math:`\sum_k f_k(x) = F(x)`.

G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)
reg = C2AffineObjective(HalfSquaredL2Deviation(beta=beta), G)
reg_per_subset = C2AffineObjective(HalfSquaredL2Deviation(beta=beta / num_subsets), G)

# full sinogram data fidelity (for evaluation)
sinogram_data_fidelity = C2AffineObjective(NegPoissonLogL(y), pet_lin_op, contamination)

# full listmode data fidelity (for evaluation)
lm_data_fidelity = NegPoissonLogLListmode(
    lm_pet_lin_op, adjoint_ones, contamination_list, float(xp.sum(contamination))
)

# total objectives (data fidelity + regularisation) -- used only for evaluation
total_objective_sino = sinogram_data_fidelity + reg
total_objective_lm = lm_data_fidelity + reg

# %%
# Sinogram subset objectives :math:`f_k^{\text{sino}}`
# -----------------------------------------------------
#
# The sinogram is split into ``num_subsets`` subsets of regularly-spaced views.
# Each subset objective is a sinogram data-fidelity term plus :math:`\beta/m \cdot R(x)`.

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)
_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

proj.clear_cached_lor_endpoints()
sino_subset_linop_list = []

for k in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[k]
    att_values_k = (
        xp.broadcast_to(
            xp.expand_dims(att_sino[subset_slices_non_tof[k]], axis=-1),
            subset_proj.out_shape,
        )
        if subset_proj.tof
        else att_sino[subset_slices_non_tof[k]]
    )
    sino_subset_linop_list.append(
        parallelproj.operators.CompositeLinearOperator(
            [
                parallelproj.operators.ElementwiseMultiplicationOperator(att_values_k),
                subset_proj,
                res_model,
            ]
        )
    )

# f_k^sino = data_fidelity_k + (beta/m) * R(x)
sino_subset_objectives: list[C1Function] = [
    C2AffineObjective(
        NegPoissonLogL(y[sl]), sino_subset_linop_list[k], contamination[sl]
    )
    + reg_per_subset
    for k, sl in enumerate(subset_slices)
]

# sensitivity images (A^k)^T 1 -- one per subset, used as SGD/SVRG preconditioner
subset_adjoint_ones = xp.zeros(
    (num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, op in enumerate(sino_subset_linop_list):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

# %%
# Listmode subset objectives :math:`f_k^{\text{LM}}`
# ---------------------------------------------------
#
# The event list is partitioned into ``num_subsets`` subsets by taking every
# :math:`m`-th event: subset :math:`k` uses events at indices
# :math:`k, k{+}m, k{+}2m, \ldots`
#
# Since each subset holds approximately :math:`1/m` of all events, its
# expected sensitivity is :math:`A^T\mathbf{1} / m`.  The same
# :math:`\beta/m` regulariser is added to each listmode subset objective via
# the :meth:`~.C1Function.__add__` operator.

lm_subset_adj_ones = adjoint_ones / num_subsets
lm_subset_contamination_sum = float(xp.sum(contamination)) / num_subsets

lm_subset_objectives: list[C1Function] = []
for k in range(num_subsets):
    lm_proj_k = parallelproj.projectors.ListmodePETProjector(
        xp.asarray(event_start_coords[k::num_subsets], copy=True),
        xp.asarray(event_end_coords[k::num_subsets], copy=True),
        img_shape,
        voxel_size,
        proj.img_origin,
    )
    att_list_k = xp.exp(-lm_proj_k(x_att))
    lm_att_op_k = parallelproj.operators.ElementwiseMultiplicationOperator(att_list_k)
    lm_proj_k.tof_parameters = proj.tof_parameters
    if proj.tof:
        assert event_tofbins is not None
        lm_proj_k.event_tofbins = xp.asarray(event_tofbins[k::num_subsets], copy=True)
        lm_proj_k.tof = proj.tof

    lm_subset_objectives.append(
        NegPoissonLogLListmode(
            parallelproj.operators.CompositeLinearOperator(
                (lm_att_op_k, lm_proj_k, res_model)
            ),
            lm_subset_adj_ones,
            contamination_list[k::num_subsets],
            lm_subset_contamination_sum,
        )
        + reg_per_subset
    )

# %%
# SVRG helper functions
# ---------------------
#
# Both helpers operate on any list of :class:`.C1Function` objects, so the
# same code drives both sinogram and listmode SVRG.


def svrg_calc_snapshot_gradients(
    x_cur: Array,
    subset_obj_functions: Sequence[C1Function],
) -> tuple[Array, Array]:
    """Compute and store all subset gradients at the current anchor point.

    Returns
    -------
    stored_grads : Array, shape (m, *x_cur.shape)
        Stacked subset gradients evaluated at the anchor point.
    full_grad : Array, shape x_cur.shape
        Sum of all subset gradients (approximates the full gradient of F).
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
    stored_snapshot_subset_gradients: Array,
    full_snapshot_gradient: Array,
    precond: Array,
    step_size: float = 1.0,
) -> Array:
    """Single SVRG subset update with variance-reduced gradient.

    The variance-reduced gradient for subset :math:`k` is

    .. math::

        g^{VR}_k = m\\bigl(\\nabla f_k(x) - \\tilde{g}_k\\bigr) + \\sum_j \\tilde{g}_j

    where :math:`\\tilde{g}_k` are the stored anchor gradients.
    """
    m = len(subset_obj_functions)
    grad_k = subset_obj_functions[subset_idx].gradient(x_cur)
    approx_grad = (
        m * (grad_k - stored_snapshot_subset_gradients[subset_idx])
        + full_snapshot_gradient
    )
    return xp.clip(x_cur - step_size * precond * approx_grad, 0, None)


# %%
# Warm start
# ----------
#
# Run one LM-SGD epoch as a common warm-start for all four reconstructions.

x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
init_precond = x_init / (adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_init, x_init))

num_init_updates = 4

for k in range(num_init_updates):
    print(f"warm-start LM-SGD subset {(k+1):03} / {num_init_updates:03}", end="\r")
    init_precond = x_init / (
        adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_init, x_init)
    )
    x_init = xp.clip(
        x_init
        - init_precond * (num_subsets * lm_subset_objectives[k].gradient(x_init)),
        0,
        None,
    )
print()

# %%
# SGD with regularisation (sinogram and listmode)
# -----------------------------------------------
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
#     x^+ = \left(x - D\, m\,\nabla f_k(x)\right)_+
#
# The preconditioner is updated infrequently to amortise its cost.
# For the sinogram variant, :math:`D^k = (A^k)^T\mathbf{1} + 2\,\text{diag}(H_R)`,
# using the exact subset sensitivity.  For the listmode variant,
# :math:`D = (A^T\mathbf{1}/m) + 2\,\text{diag}(H_R)`, using the scaled
# full sensitivity.

# --- sinogram SGD ---
df_sgd_sino: list[float] = []
x_sgd_sino = xp.asarray(x_init, copy=True)
sgd_precond_sino = x_sgd_sino / (
    adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_sgd_sino, x_sgd_sino)
)

for i in range(num_epochs):
    if i % 2 == 0 and i <= 4:
        # Use mean subset sensitivity as a single shared preconditioner
        sgd_precond_sino = x_sgd_sino / (
            adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_sgd_sino, x_sgd_sino)
        )
    for k in range(num_subsets):
        print(
            f"Sino-SGD epoch {(i+1):04} / {num_epochs:04}, "
            f"subset {(k+1):03} / {num_subsets:03}",
            end="\r",
        )
        approx_grad = num_subsets * sino_subset_objectives[k].gradient(x_sgd_sino)
        x_sgd_sino = xp.clip(
            x_sgd_sino - step_size * sgd_precond_sino * approx_grad, 0, None
        )
    df_sgd_sino.append(float(total_objective_sino(x_sgd_sino)))
print()

# --- listmode SGD ---
df_sgd_lm: list[float] = []
x_sgd_lm = xp.asarray(x_init, copy=True)
sgd_precond_lm = x_sgd_lm / (
    adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_sgd_lm, x_sgd_lm)
)

for i in range(num_epochs):
    if i % 2 == 0 and i <= 4:
        sgd_precond_lm = x_sgd_lm / (
            adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_sgd_lm, x_sgd_lm)
        )
    for k in range(num_subsets):
        print(
            f"LM-SGD epoch {(i+1):04} / {num_epochs:04}, "
            f"subset {(k+1):03} / {num_subsets:03}",
            end="\r",
        )
        approx_grad = num_subsets * lm_subset_objectives[k].gradient(x_sgd_lm)
        x_sgd_lm = xp.clip(x_sgd_lm - step_size * sgd_precond_lm * approx_grad, 0, None)
    df_sgd_lm.append(float(total_objective_sino(x_sgd_lm)))
print()

# %%
# SVRG with regularisation (sinogram and listmode)
# -------------------------------------------------
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
# The same :func:`svrg_calc_snapshot_gradients` and :func:`svrg_update`
# functions are used for both sinogram and listmode because both expose the
# :class:`.C1Function` interface -- the regularisation term is already folded
# into every subset objective via :meth:`~.C1Function.__add__`.

# --- sinogram SVRG ---
df_svrg_sino: list[float] = []
x_svrg_sino = xp.asarray(x_init, copy=True)
svrg_precond_sino = x_svrg_sino / (
    adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_svrg_sino, x_svrg_sino)
)
stored_grads_sino: Array
full_grad_sino: Array

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond_sino = x_svrg_sino / (
                adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_svrg_sino, x_svrg_sino)
            )
        stored_grads_sino, full_grad_sino = svrg_calc_snapshot_gradients(
            x_svrg_sino, sino_subset_objectives
        )
        x_svrg_sino = xp.clip(
            x_svrg_sino - step_size * svrg_precond_sino * full_grad_sino, 0, None
        )
    for k in range(num_subsets):
        print(
            f"Sino-SVRG epoch {(epoch+1):04} / {num_epochs:04}, "
            f"subset {(k+1):03} / {num_subsets:03}",
            end="\r",
        )
        x_svrg_sino = svrg_update(
            x_svrg_sino,
            k,
            sino_subset_objectives,
            stored_grads_sino,
            full_grad_sino,
            svrg_precond_sino,
            step_size=step_size,
        )
    df_svrg_sino.append(float(total_objective_sino(x_svrg_sino)))
print()

# --- listmode SVRG ---
df_svrg_lm: list[float] = []
x_svrg_lm = xp.asarray(x_init, copy=True)
svrg_precond_lm = x_svrg_lm / (
    adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_svrg_lm, x_svrg_lm)
)
stored_grads_lm: Array
full_grad_lm: Array

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond_lm = x_svrg_lm / (
                adjoint_ones + 2 * reg.hessian_diag_vec_prod(x_svrg_lm, x_svrg_lm)
            )
        stored_grads_lm, full_grad_lm = svrg_calc_snapshot_gradients(
            x_svrg_lm, lm_subset_objectives
        )
        x_svrg_lm = xp.clip(
            x_svrg_lm - step_size * svrg_precond_lm * full_grad_lm, 0, None
        )
    for k in range(num_subsets):
        print(
            f"LM-SVRG epoch {(epoch+1):04} / {num_epochs:04}, "
            f"subset {(k+1):03} / {num_subsets:03}",
            end="\r",
        )
        x_svrg_lm = svrg_update(
            x_svrg_lm,
            k,
            lm_subset_objectives,
            stored_grads_lm,
            full_grad_lm,
            svrg_precond_lm,
            step_size=step_size,
        )
    df_svrg_lm.append(float(total_objective_sino(x_svrg_lm)))
print()

# %%
# Convergence comparison
# ----------------------
#
# We plot the total objective :math:`F(x) = \sum_i(\bar{y}_i - y_i\log\bar{y}_i) + \beta R(x)`
# vs epoch (left) and vs full data passes (right).
#
# Data-pass counts:
#
# * **SGD**: 1 full data pass per epoch.
# * **SVRG**: 2 passes on anchor epochs (snapshot + subset updates),
#   1 pass on non-anchor epochs.
#
# Both sinogram and listmode variants are compared, demonstrating that the
# choice of data representation does not affect convergence when the same
# subset splitting and step size are used.

epochs = np.arange(1, num_epochs + 1)
sgd_passes = epochs.copy()

svrg_passes_per_epoch = np.where(np.arange(num_epochs) % 2 == 0, 2, 1)
svrg_cumulative_passes = np.cumsum(svrg_passes_per_epoch)

all_values = df_sgd_sino + df_sgd_lm + df_svrg_sino + df_svrg_lm
df_min = min(all_values)
df_max = max(df_sgd_sino[0], df_sgd_lm[0])

sgd_label_sino = f"Sino-SGD ({num_subsets} subsets, step={step_size:.1f})"
sgd_label_lm = f"LM-SGD ({num_subsets} subsets, step={step_size:.1f})"
svrg_label_sino = f"Sino-SVRG ({num_subsets} subsets, step={step_size:.1f})"
svrg_label_lm = f"LM-SVRG ({num_subsets} subsets, step={step_size:.1f})"

fig, axs = plt.subplots(1, 2, figsize=(13, 4.5), layout="constrained")

for ax, xvals_sgd, xvals_svrg, xlabel in (
    (axs[0], epochs, epochs, "Epoch"),
    (axs[1], sgd_passes, svrg_cumulative_passes, "Full data passes"),
):
    ax.plot(xvals_sgd, df_sgd_sino, label=sgd_label_sino, marker="o")
    ax.plot(xvals_sgd, df_sgd_lm, label=sgd_label_lm, marker="s")
    ax.plot(xvals_svrg, df_svrg_sino, label=svrg_label_sino, marker="^")
    ax.plot(xvals_svrg, df_svrg_lm, label=svrg_label_lm, marker="v")
    ax.set_ylim(df_min, df_max)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(r"$F(x) = \sum_i(\bar{y}_i - y_i \log \bar{y}_i) + \beta R(x)$")
    ax.set_title(rf"Convergence ($\beta={beta}$)")
    ax.legend(fontsize="small")
    ax.grid(ls=":")

fig.show()

# %%
# Reconstruction results
# ----------------------

vmax = float(xp.max(x_sgd_sino))

# %%
fig_sgd_sino, _, widgets_sgd_sino = show_vol_cuts(
    to_numpy_array(x_sgd_sino),
    vmin=0,
    vmax=vmax,
    fig_title=f"Sino-SGD ({num_subsets} subsets, {num_epochs} epochs)",
)
fig_sgd_sino.show()

# %%
fig_sgd_lm, _, widgets_sgd_lm = show_vol_cuts(
    to_numpy_array(x_sgd_lm),
    vmin=0,
    vmax=vmax,
    fig_title=f"LM-SGD ({num_subsets} subsets, {num_epochs} epochs)",
)
fig_sgd_lm.show()

# %%
fig_svrg_sino, _, widgets_svrg_sino = show_vol_cuts(
    to_numpy_array(x_svrg_sino),
    vmin=0,
    vmax=vmax,
    fig_title=f"Sino-SVRG ({num_subsets} subsets, {num_epochs} epochs)",
)
fig_svrg_sino.show()

# %%
fig_svrg_lm, _, widgets_svrg_lm = show_vol_cuts(
    to_numpy_array(x_svrg_lm),
    vmin=0,
    vmax=vmax,
    fig_title=f"LM-SVRG ({num_subsets} subsets, {num_epochs} epochs)",
)
fig_svrg_lm.show()
