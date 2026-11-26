"""
Convergence comparison of SGD and SVRG with a logcosh prior
===========================================================

:bdg-primary:`reconstruction` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

Adding an edge-preserving prior to the negative Poisson log-likelihood
suppresses the noise of maximum-likelihood reconstructions, but the EM
update no longer applies.  This example minimises such a regularised
objective with two subset-based gradient methods: preconditioned
**stochastic gradient descent (SGD)**, fast in the first epochs but without
convergence guarantee, and **SVRG** (stochastic variance-reduced gradient),
which converges.  We compare their convergence per epoch and per full data
pass.

**You will learn**

- how to build a smooth, edge-preserving logcosh prior from
  :class:`.FiniteForwardDifference` and :class:`.LogCosh`,
- how to split a regularised objective exactly into subset functions,
- how to implement preconditioned SGD and SVRG for this objective,
- how to compare the convergence of SGD and SVRG per epoch and per data pass.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(forward model, subsets, OSEM and SVRG without prior).

.. note::

    To keep the documentation build short, this example runs **fewer
    iterations on the CPU** than on a GPU (see the printed note below).
    The results shown on CPU-only builds (e.g. on Read the Docs) are
    therefore less converged.  Run the example on a GPU, or set the environment
    variable ``PARALLELPROJ_EXAMPLES_FULL_ITERATIONS=1``, for converged
    results.
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

from __future__ import annotations

from collections.abc import Sequence
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    iterations_for_device,
    show_vol_cuts,
    suggest_array_backend_and_device,
)
from parallelproj.functions import (
    C1Function,
    C2AffineObjective,
    LogCosh,
    NegPoissonLogL,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Reconstruction settings
# --------------------------
#
# SGD and SVRG split the data into ``num_subsets`` subsets; an **epoch** is
# one cycle through all subsets.  ``beta`` is the weight of the prior,
# ``delta_rel`` its transition scale relative to the maximum of the ground
# truth (see section 6).  ``count_factor`` scales the ground truth image, and
# with it the number of counts (and the reconstructed values).

num_subsets = 12

# fewer epochs on the CPU
num_epochs = iterations_for_device(dev, cpu=5, gpu=240 // num_subsets, what="epochs")

beta = 1.0
delta_rel = 0.1

# step size of the SGD and SVRG updates
step_size = 1.0

count_factor = 1.0

# %%
# 2. Scanner, sinogram and image
# ------------------------------
#
# The same small scanner (16 sides, 5 rings), span-1 sinogram and
# 55 x 55 x 8 image (2 mm voxels) as in the previous example.  The ground
# truth is an elliptic cylinder with hot and cold spherical inserts.

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

img_shape = (55, 55, 8)
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

x_true = count_factor * elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# %%
# 3. Forward model :math:`\bar{y}(x) = A x + s`
# ---------------------------------------------
#
# :math:`A` combines an image-based resolution model (Gaussian, 2 mm FWHM), a
# TOF projector (13 TOF bins of 12 mm) and attenuation (water,
# :math:`\mu` = 0.01 / mm, inside the phantom).  The attenuation sinogram has
# no TOF axis; for TOF, ``xp.broadcast_to`` repeats it over the TOF bins
# without copying data.  Leave out the ``proj.tof_parameters`` line to
# reconstruct non-TOF data.

x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
att_sino = xp.exp(-proj(x_att))

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

att_values = (
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
    if proj.tof
    else att_sino
)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[2.0 / (2.3548 * float(vs)) for vs in proj.voxel_size]
)

pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# 4. Simulated data
# -----------------
#
# Noise-free data :math:`A x_{true} + s` with a constant contamination
# :math:`s` of half the mean of :math:`A x_{true}`, and a Poisson
# realisation :math:`y` of them.

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
# 5. Subset forward models :math:`A^k`
# ------------------------------------
#
# The sinogram views are distributed over the subsets.  Each subset operator
# :math:`A^k` chains the resolution model, a copy of the projector that
# projects only the views of the subset, and the matching subset of the
# attenuation sinogram.  The slices extract the subsets of the data and
# contamination sinograms.

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)
# the attenuation sinogram has no TOF axis: slices for 3 dimensions
_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear the cached LOR endpoints since we create many copies of the projector
proj.clear_cached_lor_endpoints()

subset_linops = []
for k in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[k]

    subset_att_sino = att_sino[subset_slices_non_tof[k]]
    subset_att_values = (
        xp.broadcast_to(xp.expand_dims(subset_att_sino, axis=-1), subset_proj.out_shape)
        if subset_proj.tof
        else subset_att_sino
    )
    subset_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
        subset_att_values
    )
    subset_linops.append(
        parallelproj.operators.CompositeLinearOperator(
            [subset_att_op, subset_proj, res_model]
        )
    )

pet_subset_linop_seq = parallelproj.operators.LinearOperatorSequence(subset_linops)

# %%
# 6. The logcosh prior and the subset objective functions
# -------------------------------------------------------
#
# We minimise the regularised negative Poisson log-likelihood
#
# .. math::
#     F(x) = \sum_i \bar{y}_i - y_i \log \bar{y}_i
#            + \beta \, R(x),
#     \qquad \bar{y}(x) = A x + s
#
# subject to :math:`x \geq 0`, where the edge-preserving logcosh penalty is
#
# .. math::
#     R(x) = \delta \sum_i \log\!\cosh\!\left(\frac{(Gx)_i}{\delta}\right)
#
# and :math:`G` is the finite forward-difference operator
# (:class:`.FiniteForwardDifference`, combined with :class:`.LogCosh`).  The
# :math:`\delta` prefactor makes the asymptotic gradient magnitude equal 1
# regardless of :math:`\delta`, so the regularisation strength :math:`\beta`
# keeps the same meaning for different choices of :math:`\delta`.  The scale
# :math:`\delta` itself controls the transition between two regimes:
#
# * **Quadratic** for :math:`|(Gx)_i| \ll \delta`:
#   :math:`R(x) \approx \tfrac{1}{2\delta}\|Gx\|_2^2`.
# * **Linear** for :math:`|(Gx)_i| \gg \delta`:
#   :math:`R(x) \approx \|Gx\|_1 - n\,\delta\log 2 \approx \|Gx\|_1`.
#
# Setting :math:`\delta` well below the typical edge gradient places true
# edges in the linear regime (edge-preserving) while penalising smooth-region
# deviations quadratically.  Here ``delta`` is ``delta_rel`` times the
# maximum of the ground truth image: with ``delta_rel = 0.1``, edges with a
# gradient equal to the image maximum have :math:`|(Gx)_i|/\delta = 10`,
# firmly in the linear regime (:math:`\tanh(10) \approx 1`), while
# smooth-region gradients near zero remain quadratic.
#
# Both algorithms use the decomposition into :math:`m` subset functions
#
# .. math::
#     f_k(x) = \underbrace{\sum_{i \in S_k} \bar{y}_i - y_i \log \bar{y}_i}_{
#              \text{subset data fidelity}} + \frac{\beta}{m} R(x),
#     \qquad k = 1, \ldots, m,
#
# so that :math:`F(x) = \sum_{k=1}^m f_k(x)` exactly.  The full prior ``reg``
# (weight :math:`\beta`) is only used to evaluate :math:`F` and the
# preconditioner; the subset functions use ``reg_per_subset`` (weight
# :math:`\beta / m`).

G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)

delta = float(xp.max(x_true)) * delta_rel

reg = C2AffineObjective(LogCosh(delta=delta, beta=beta), G)
reg_per_subset = C2AffineObjective(LogCosh(delta=delta, beta=beta / num_subsets), G)

# %%
# 7. Sensitivity image and objective functions
# --------------------------------------------
#
# We define the subset objectives :math:`f_k`, the full objective :math:`F`
# (to evaluate the iterates) and the sensitivity image :math:`A^T 1` (for the
# preconditioner).
#
# .. note::
#     By default :class:`.NegPoissonLogL` evaluates a "safe epsilon"
#     (shifted Poisson) surrogate: a tiny ``eps = rel_eps * mean(y)`` is
#     added to the measured and the expected data.  This is finite for any
#     non-negative expectation (never ``nan`` / ``inf``), at the price of a
#     tiny (~``rel_eps``) bias that vanishes at the fit.  Since our
#     contamination is strictly positive, the expected data ``A x + s`` is
#     positive in every bin and we can use ``exact=True`` to evaluate the
#     unmodified log-likelihood instead (bins with ``y = 0`` are still
#     handled exactly).  Keep the default whenever the expected data can
#     reach zero in bins with counts, e.g. with zero contamination and a
#     mismatched forward model.

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

exact_mode = bool(xp.min(contamination) > 0)

# f_k = subset data fidelity + (beta / m) R(x)
subset_objectives = [
    C2AffineObjective(
        NegPoissonLogL(y[sl], exact=exact_mode),
        pet_subset_linop_seq[k],
        contamination[sl],
    )
    + reg_per_subset
    for k, sl in enumerate(subset_slices)
]

full_data_fidelity = C2AffineObjective(
    NegPoissonLogL(y, exact=exact_mode), pet_lin_op, contamination
)
total_objective = full_data_fidelity + reg

# %%
# 8. Preconditioner, FOV mask and warm start
# ------------------------------------------
#
# SGD and SVRG take preconditioned gradient steps.  The diagonal
# preconditioner extends the EM preconditioner :math:`x / (A^T 1)` by the
# curvature of the prior,
#
# .. math::
#     D = \operatorname{diag}\left(\frac{x}{A^T 1 + 2 \beta\, G^T
#         \operatorname{diag}\!\left(\psi''(Gx)\right) G x}\right),
#     \qquad \psi(t) = \delta \log\cosh(t / \delta),
#
# computed with ``reg.hessian_diag_vec_prod(x, x)``.
#
# The scanner's cylindrical field of view (FOV) does not cover every voxel of
# the image grid.  Voxels outside the FOV are never intersected by any LOR, so
# their sensitivity :math:`(A^T 1)_i = 0`.  We set the initial image and the
# preconditioner to zero outside the FOV, so these voxels stay at zero
# throughout the reconstruction.
# :meth:`.RegularPolygonPETProjector.fov_mask` returns a boolean array that is
# ``True`` inside the FOV.  ``fov_mask`` is set to ``None`` when every image
# voxel is inside the FOV (no masking needed).
#
# The warm start for both algorithms is one SGD epoch from a uniform image.

cyl_mask = proj.fov_mask()
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask


def preconditioner(x: Array) -> Array:
    """Diagonal preconditioner x / (A^T 1 + 2 beta G^T H G x), 0 outside FOV"""
    denom = adjoint_ones + 2 * reg.hessian_diag_vec_prod(x, x)
    if fov_mask is None:
        return x / denom
    return xp.where(fov_mask, x / denom, xp.zeros_like(x))


x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
if fov_mask is not None:
    x_init = xp.where(fov_mask, x_init, xp.zeros_like(x_init))
for k in range(num_subsets):
    print(f"warm-start SGD subset {(k + 1):03} / {num_subsets:03}", end="\r")
    init_precond = preconditioner(x_init)
    x_init = xp.clip(
        x_init - init_precond * (num_subsets * subset_objectives[k].gradient(x_init)),
        0,
        None,
    )
print()

# %%
# 9. SGD
# ------
#
# Each SGD epoch cycles through all :math:`m` subsets.  Because
# :math:`F(x) = \sum_k f_k(x)`, the full gradient is approximated as
#
# .. math::
#     \nabla F(x) \approx m\,\nabla f_k(x)
#
# and a gradient step with the diagonal preconditioner :math:`D` is taken,
# followed by a projection onto :math:`x \geq 0`:
#
# .. math::
#     x^+ = \left(x - D\, m\,\nabla f_k(x)\right)_+.
#
# The preconditioner is updated at the start of epochs 1, 3 and 5 and kept
# fixed afterwards.

df_sgd = xp.zeros(num_epochs + 1, dtype=xp.float32, device=dev)
x_sgd = xp.asarray(x_init, copy=True)
df_sgd[0] = total_objective(x_sgd)

for i in range(num_epochs):
    if i % 2 == 0 and i <= 4:
        sgd_precond = preconditioner(x_sgd)
    for k in range(num_subsets):
        print(
            f"SGD epoch {(i + 1):04} / {num_epochs:04}, "
            f"subset {(k + 1):04} / {num_subsets:04}",
            end="\r",
        )
        approx_grad = num_subsets * subset_objectives[k].gradient(x_sgd)
        x_sgd = xp.clip(x_sgd - step_size * sgd_precond * approx_grad, 0, None)

    df_sgd[i + 1] = total_objective(x_sgd)
print()

# %%
# 10. SVRG
# --------
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
#    and take a preconditioned, projected gradient step.
#
# As for SGD, the preconditioner is updated in epochs 1, 3 and 5 (in the
# anchor phase).
#
# .. note::
#     An SVRG epoch with an anchor phase costs **two** full data passes (one
#     for the snapshot gradients, one for the :math:`m` subset updates), the
#     other epochs one.  With an anchor every other epoch, an SVRG epoch
#     costs 1.5 data passes on average, compared to one per SGD epoch.


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
df_svrg = xp.zeros(num_epochs + 1, dtype=xp.float32, device=dev)
df_svrg[0] = total_objective(x_svrg)

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond = preconditioner(x_svrg)

        stored_grads, full_grad = svrg_calc_snapshot_gradients(
            x_svrg, subset_objectives
        )
        x_svrg = xp.clip(x_svrg - step_size * svrg_precond * full_grad, 0, None)

    for k in range(num_subsets):
        print(
            f"SVRG epoch {(epoch + 1):04} / {num_epochs:04}, "
            f"subset {(k + 1):04} / {num_subsets:04}",
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

# %%
# 11. Convergence comparison
# --------------------------
#
# We plot the objective :math:`F(x)` vs epoch (left) and vs full data passes
# (right), starting at the warm start (epoch 0).  One SGD epoch is one full
# data pass; one SVRG epoch is two passes with an anchor phase and one
# otherwise.

epochs = np.arange(num_epochs + 1)
svrg_passes = np.cumsum(
    np.concatenate([[0], np.where(np.arange(num_epochs) % 2 == 0, 2, 1)])
)

df_min = min(float(xp.min(df_sgd)), float(xp.min(df_svrg)))
df_max = float(df_sgd[0])

sgd_label = f"SGD ({num_subsets} subsets, step {step_size:.1f})"
svrg_label = f"SVRG ({num_subsets} subsets, step {step_size:.1f})"

fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axs[0].plot(epochs, to_numpy_array(df_sgd), marker="o", label=sgd_label)
axs[0].plot(epochs, to_numpy_array(df_svrg), marker="o", label=svrg_label)
axs[1].plot(epochs, to_numpy_array(df_sgd), marker="o", label=sgd_label)
axs[1].plot(svrg_passes, to_numpy_array(df_svrg), marker="o", label=svrg_label)
axs[0].set_xlabel("epoch")
axs[0].set_ylabel("regularised objective F(x)")
axs[0].set_title(f"Convergence per epoch (beta = {beta})")
axs[1].set_xlabel("full data passes")
axs[1].set_title(f"Convergence per data pass (beta = {beta})")
for ax in axs:
    ax.set_ylim(df_min, df_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend()
    ax.grid(ls=":")
fig.show()

# %%
# Both algorithms decrease the objective most in the first epoch.  Per
# epoch, SVRG stays below SGD from then on.  Per data pass, SGD is ahead at
# first, because the first SVRG epoch costs two passes, but SVRG overtakes it
# after about three passes and keeps the lead.

# %%
# 12. Reconstructed images
# ------------------------
#
# The final SGD and SVRG images, with the same colour scale.

vmax = float(xp.max(x_svrg))

fig, _, _ = show_vol_cuts(
    x_sgd,
    voxel_size=voxel_size,
    fig_title=f"SGD, {num_epochs} epochs",
    vmin=0,
    vmax=vmax,
)
fig.show()

# %%

fig, _, _ = show_vol_cuts(
    x_svrg,
    voxel_size=voxel_size,
    fig_title=f"SVRG, {num_epochs} epochs",
    vmin=0,
    vmax=vmax,
)
fig.show()

# %%
# The two images look very similar.  Thanks to the prior, they are much less
# noisy than the unregularised OSEM and SVRG images of the previous example,
# while the hot and cold inserts keep sharp edges.

# %%
# Summary
# -------
#
# - The logcosh prior :math:`R(x) = \delta \sum_i \log\cosh((Gx)_i / \delta)`
#   is quadratic for small and linear for large differences (edge-preserving);
#   it is built from :class:`.FiniteForwardDifference` and :class:`.LogCosh`.
# - Adding :math:`\beta R / m` to each subset data fidelity splits the
#   regularised objective exactly into subset functions,
#   :math:`F = \sum_k f_k`.
# - SGD uses :math:`m \nabla f_k` as the gradient; SVRG corrects it with
#   gradients stored at an anchor point and reaches lower objective values.
# - An SVRG epoch with an anchor phase costs two data passes: compare
#   algorithms per data pass, not only per epoch.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_02_run_pdhg_spdhg_dtv.py`
#    -- PDHG and SPDHG for a non-smooth directional TV prior.

# sphinx_gallery_thumbnail_number = 3
