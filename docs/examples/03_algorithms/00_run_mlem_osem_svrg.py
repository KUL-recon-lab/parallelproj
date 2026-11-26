"""
Convergence comparison of MLEM, OSEM and SVRG
=============================================

:bdg-primary:`reconstruction` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Maximum-likelihood PET reconstruction minimises the negative Poisson
log-likelihood of the measured data.  This example compares three iterative
algorithms for this problem on simulated TOF data: **MLEM** (convergent but
slow), **OSEM** (fast, but without convergence guarantee) and **SVRG**
(stochastic variance-reduced gradient, fast *and* convergent).  We compare how
quickly they decrease the objective per epoch and per full pass through the
data, the main cost of every update.

**You will learn**

- how to set up a complete TOF forward model :math:`\\bar{y}(x) = A x + s`
  and split it into subsets of views,
- how the EM update can be written as a preconditioned gradient step and
  implemented with :class:`.C2AffineObjective` and :class:`.NegPoissonLogL`,
- how to implement MLEM, OSEM and SVRG with the same building blocks,
- how to compare the convergence of algorithms per epoch and per data pass.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_01_run_pet_non_tof_sinogram_projector.py`
(projector, resolution model and attenuation) and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF projector).

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
from parallelproj.functions import C1Function, C2AffineObjective, NegPoissonLogL

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Reconstruction settings
# --------------------------
#
# OSEM and SVRG split the data into ``num_subsets`` subsets.  An **epoch** is
# one cycle through all subsets.  To compare MLEM with the same number of
# subset updates, MLEM runs ``num_epochs * num_subsets`` iterations.  MLEM is
# only run on a GPU: on the CPU this many iterations would be too slow.

num_subsets = 24

# fewer OSEM / SVRG epochs on the CPU
num_epochs = iterations_for_device(dev, cpu=3, gpu=1200 // num_subsets, what="epochs")
num_epochs_mlem = num_epochs * num_subsets

run_mlem = dev != "cpu"

# %%
# 2. Scanner, sinogram and image
# ------------------------------
#
# We use a small regular polygon scanner (16 sides, 5 rings), a span-1
# sinogram with a maximum ring difference of 2, and an image of
# 55 x 55 x 8 voxels of 2 mm.  The ground truth :math:`x_{true}` is an
# elliptic cylinder with hot and cold spherical inserts.

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

x_true = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# %%
# 3. Forward model :math:`\bar{y}(x) = A x + s`
# ---------------------------------------------
#
# The linear operator :math:`A` combines an image-based resolution model
# (Gaussian, 2 mm FWHM), a TOF projector (13 TOF bins of 12 mm) and
# attenuation (water, :math:`\mu` = 0.01 / mm, inside the phantom).  The
# attenuation sinogram is computed with the projector *before* TOF is enabled,
# so it has no TOF axis; for TOF, ``xp.broadcast_to`` repeats it over the TOF
# bins without copying data (a zero-stride view).
#
# .. note::
#     The algorithms below work with any operator that subclasses
#     :class:`.LinearOperator` (e.g. the high-level projectors).  Leave out
#     the ``proj.tof_parameters`` line to reconstruct non-TOF data.

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
# The noise-free data are :math:`A x_{true} + s`, with a constant
# contamination :math:`s` (scatter and randoms) of half the mean of
# :math:`A x_{true}`.  The measured data :math:`y` are a Poisson realisation.

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
# ``get_distributed_views_and_slices`` distributes the sinogram views over the
# subsets.  The views define a subset projector (a copy of the full projector
# that projects only these views); the slices extract the matching subset of
# the data, contamination and attenuation sinograms.  Each subset operator
# :math:`A^k` chains the resolution model, the subset projector and the
# subset of the attenuation sinogram.

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
# 6. The EM update as a preconditioned gradient step
# --------------------------------------------------
#
# All three algorithms minimise the negative Poisson log-likelihood
#
# .. math::
#     f(x) = \sum_i \bar{y}_i - y_i \log \bar{y}_i, \qquad \bar{y}(x) = A x + s
#
# subject to :math:`x \geq 0`.  The EM update used in MLEM and OSEM is
# :footcite:p:`Dempster1977` :footcite:p:`Shepp1982` :footcite:p:`Lange1984`
# :footcite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{A^T 1} A^T \frac{y}{A x + s}
#
# which can be rewritten as a preconditioned gradient descent step with
# diagonal preconditioner :math:`D = \operatorname{diag}(x / (A^T 1))`:
#
# .. math::
#     x^+ = x - D \, \nabla_x f(x).
#
# We implement this as a single function used by both MLEM and OSEM.


def em_update(
    x_cur: Array,
    data_fidelity: C1Function,
    adj_ones: Array,
    img_mask: Array | None = None,
) -> Array:
    """One EM update rewritten as a preconditioned gradient descent step.

    Computes :math:`x^+ = x - D \\nabla f(x)` where the diagonal
    preconditioner is :math:`D = \\operatorname{diag}(x / (A^T 1))`.
    Voxels outside the FOV are excluded via ``img_mask`` to avoid
    division by the zero sensitivity values in ``adj_ones``.

    Parameters
    ----------
    x_cur : Array
        Current image estimate.
    data_fidelity : C1Function
        Differentiable data-fidelity term whose gradient is evaluated at
        ``x_cur``.
    adj_ones : Array
        Sensitivity image :math:`A^T 1` (or subset variant
        :math:`(A^k)^T 1`).
    img_mask : Array or None, optional
        Boolean FOV mask (``True`` inside the FOV).  Preconditioner is
        zeroed outside the FOV so that zero-sensitivity voxels do not
        produce NaN / Inf.  Pass ``None`` when every voxel is in the FOV.

    Returns
    -------
    Array
        Updated image :math:`x^+`, same shape as ``x_cur``.
    """
    if img_mask is None:
        d = x_cur / adj_ones
    else:
        d = xp.where(img_mask, x_cur / adj_ones, xp.zeros_like(x_cur))
    return x_cur - d * data_fidelity.gradient(x_cur)


# %%
# 7. Sensitivity images and FOV mask
# ----------------------------------
#
# The subset sensitivity images :math:`(A^k)^T 1` and the full sensitivity
# image :math:`A^T 1 = \sum_k (A^k)^T 1` are computed once.
#
# The scanner's cylindrical field of view (FOV) does not cover every voxel of
# the image grid.  Voxels outside the FOV are never intersected by any LOR, so
# their sensitivity :math:`(A^T 1)_i = 0`, and dividing by it in the EM
# preconditioner would produce NaN / Inf values that corrupt the
# reconstruction.  :meth:`.RegularPolygonPETProjector.fov_mask` returns a
# boolean array that is ``True`` inside the FOV.  ``fov_mask`` is set to
# ``None`` when every image voxel is inside the FOV (no masking needed).

subset_adjoint_ones = xp.zeros(
    (num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, op in enumerate(pet_subset_linop_seq):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

adjoint_ones = xp.sum(subset_adjoint_ones, axis=0)

cyl_mask = proj.fov_mask()
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask

# %%
# 8. Data-fidelity terms and warm start
# -------------------------------------
#
# We define one :class:`.C2AffineObjective` :math:`f_k` per subset (for OSEM
# and SVRG) and one for the full data (for MLEM and to evaluate the
# objective of all algorithms).
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
#
# All algorithms start from the same warm start: one OSEM epoch from a
# uniform image (zero outside the FOV).

exact_mode = bool(xp.min(contamination) > 0)

subset_data_fidelities = [
    C2AffineObjective(
        NegPoissonLogL(y[sl], exact=exact_mode),
        pet_subset_linop_seq[k],
        contamination[sl],
    )
    for k, sl in enumerate(subset_slices)
]

full_data_fidelity = C2AffineObjective(
    NegPoissonLogL(y, exact=exact_mode), pet_lin_op, contamination
)

x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
if fov_mask is not None:
    x_init = xp.where(fov_mask, x_init, xp.zeros_like(x_init))

for k in range(num_subsets):
    print(f"warm-start OSEM subset {(k + 1):03} / {num_subsets:03}", end="\r")
    x_init = em_update(
        x_init, subset_data_fidelities[k], subset_adjoint_ones[k], fov_mask
    )
print()

# %%
# 9. MLEM
# -------
#
# One MLEM iteration uses the full data (:math:`A`, :math:`y`, :math:`s`)
# and the full sensitivity image :math:`A^T 1`.  It is guaranteed to
# converge to the maximum-likelihood solution but requires one full data
# pass per iteration.  It only runs when ``run_mlem`` is ``True`` (on a GPU).

if run_mlem:
    df_mlem = xp.zeros(num_epochs_mlem, dtype=xp.float32, device=dev)
    x_mlem = xp.asarray(x_init, copy=True)
    for i in range(num_epochs_mlem):
        print(f"MLEM epoch {(i + 1):04} / {num_epochs_mlem:04}", end="\r")
        x_mlem = em_update(x_mlem, full_data_fidelity, adjoint_ones, fov_mask)
        df_mlem[i] = full_data_fidelity(x_mlem)
    print()

# %%
# 10. OSEM
# --------
#
# One OSEM epoch cycles through all :math:`m` subsets, each update using the
# subset operator :math:`A^k`, subset data :math:`y^k`, contamination
# :math:`s^k` and subset sensitivity :math:`(A^k)^T 1`.  OSEM converges fast
# in the first epochs, but has no convergence guarantee.

df_osem = xp.zeros(num_epochs, dtype=xp.float32, device=dev)
x_osem = xp.asarray(x_init, copy=True)
for i in range(num_epochs):
    for k in range(num_subsets):
        print(
            f"OSEM epoch {(i + 1):04} / {num_epochs:04}, "
            f"subset {(k + 1):04} / {num_subsets:04}",
            end="\r",
        )
        x_osem = em_update(
            x_osem, subset_data_fidelities[k], subset_adjoint_ones[k], fov_mask
        )

    df_osem[i] = full_data_fidelity(x_osem)
print()

# %%
# 11. SVRG
# --------
#
# SVRG reduces the variance of the subset gradients with gradients stored at
# an anchor point :math:`\tilde{x}`.  Each epoch consists of two phases:
#
# 1. **Anchor phase** (every other epoch): compute and store all :math:`m`
#    subset gradients :math:`\tilde{g}_k = \nabla f_k(\tilde{x})` at the
#    current point :math:`\tilde{x}`, then take a full gradient step with
#    :math:`\nabla f(\tilde{x}) = \sum_k \tilde{g}_k`.
# 2. **Variance-reduced subset updates**: for each subset :math:`k`, form
#    the variance-reduced gradient
#
#    .. math::
#        g^{VR}_k = m \left( \nabla f_k(x) - \tilde{g}_k \right)
#                   + \sum_{j=1}^m \tilde{g}_j
#
#    and take a preconditioned gradient step, followed by a projection onto
#    :math:`x \geq 0`.
#
# The preconditioner is the EM preconditioner
# :math:`\operatorname{diag}(x / (A^T 1))`, updated in the anchor phases of
# the first three anchor epochs (epochs 1, 3 and 5) and kept fixed afterwards.
#
# .. note::
#     The anchor phase requires one full pass through all subsets to
#     compute :math:`\tilde{g}_1, \ldots, \tilde{g}_m`.  Therefore, each
#     SVRG epoch with an anchor phase costs **two full data passes** (one
#     for the snapshot, one for the :math:`m` subset updates), the other
#     epochs one, compared to one full data pass per OSEM epoch or MLEM
#     iteration.  With an anchor every other epoch, an SVRG epoch costs 1.5
#     data passes on average.


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


x_svrg = xp.asarray(x_init, copy=True)

svrg_step_size = 1.0
df_svrg = xp.zeros(num_epochs, dtype=xp.float32, device=dev)

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            if fov_mask is None:
                svrg_precond = x_svrg / adjoint_ones
            else:
                svrg_precond = xp.where(
                    fov_mask, x_svrg / adjoint_ones, xp.zeros_like(x_svrg)
                )

        stored_grads, full_grad = svrg_calc_snapshot_gradients(
            x_svrg, subset_data_fidelities
        )

        x_svrg = xp.clip(x_svrg - svrg_step_size * svrg_precond * full_grad, 0, None)

    for k in range(num_subsets):
        print(
            f"SVRG epoch {(epoch + 1):04} / {num_epochs:04}, "
            f"subset {(k + 1):04} / {num_subsets:04}",
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
# 12. Convergence comparison
# --------------------------
#
# We plot the negative Poisson log-likelihood :math:`f(x)` of the OSEM and
# SVRG iterates vs epoch (left) and vs the number of full data passes
# (right): one per OSEM epoch, two per SVRG epoch with an anchor phase and
# one otherwise.  On a GPU, horizontal lines show the value MLEM reaches
# after 50, 100 and ``num_epochs_mlem`` iterations, and the right panel also
# shows the MLEM curve for the same number of data passes.

epochs = np.arange(1, num_epochs + 1)
svrg_passes = np.cumsum(np.where(np.arange(num_epochs) % 2 == 0, 2, 1))

df_min = min(float(xp.min(df_osem)), float(xp.min(df_svrg)))
if run_mlem:
    df_min = min(df_min, float(xp.min(df_mlem)))
df_max = float(df_osem[0])  # first OSEM epoch as upper limit

osem_label = f"OSEM ({num_subsets} subsets)"
svrg_label = f"SVRG ({num_subsets} subsets, step {svrg_step_size:.1f})"

fig, axs = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
axs[0].plot(epochs, to_numpy_array(df_osem), marker="o", label=osem_label)
axs[0].plot(epochs, to_numpy_array(df_svrg), marker="o", label=svrg_label)
axs[1].plot(epochs, to_numpy_array(df_osem), marker="o", label=osem_label)
axs[1].plot(svrg_passes, to_numpy_array(df_svrg), marker="o", label=svrg_label)

if run_mlem:
    # MLEM reference values: grey lines, line style by number of iterations
    for n_iter, ls in ((50, ":"), (100, "-."), (num_epochs_mlem, "--")):
        axs[0].axhline(
            float(df_mlem[n_iter - 1]),
            color="0.4",
            ls=ls,
            label=f"MLEM ({n_iter} iterations)",
        )
    max_passes = int(svrg_passes[-1])
    axs[1].plot(
        np.arange(1, max_passes + 1),
        to_numpy_array(df_mlem[:max_passes]),
        label="MLEM",
    )
    axs[1].axhline(
        float(df_mlem[-1]),
        color="0.4",
        ls="--",
        label=f"MLEM ({num_epochs_mlem} iterations)",
    )

axs[0].set_xlabel("epoch")
axs[0].set_ylabel("negative Poisson log-likelihood f(x)")
axs[0].set_title("Convergence per epoch")
axs[1].set_xlabel("full data passes")
axs[1].set_title("Convergence per data pass")
for ax in axs:
    ax.set_ylim(df_min, df_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize="small")
    ax.grid(ls=":")
fig.show()

# %%
# SVRG reaches lower values than OSEM, per epoch and also per data pass,
# although its anchor phases cost extra passes and narrow the gap.  With more
# epochs (on a GPU), OSEM levels off at a higher value, as it does not
# converge to the minimum, while SVRG keeps decreasing the objective.  The
# grey MLEM lines show how many MLEM iterations reach a comparable value.

# %%
# 13. Reconstructed images
# ------------------------
#
# The final OSEM and SVRG images, with the same colour scale.

vmax = float(xp.max(x_svrg))

fig, _, _ = show_vol_cuts(
    x_osem,
    voxel_size=voxel_size,
    fig_title=f"OSEM, {num_epochs} epochs",
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
# The OSEM and SVRG images look very similar: both show the elliptic cylinder
# with its hot and cold inserts, and both are noisy.  Without regularisation,
# the noise increases further as the iterates approach the
# maximum-likelihood solution; the next example adds a prior.

# %%
# Summary
# -------
#
# - The EM update is a preconditioned gradient step with preconditioner
#   :math:`\operatorname{diag}(x / (A^T 1))`; MLEM applies it to the full
#   data, OSEM to one subset at a time.
# - SVRG combines subset updates with stored gradients at an anchor point.
#   Like MLEM it converges, but it decreases the objective much faster per
#   epoch.
# - An SVRG epoch with an anchor phase costs two full data passes: compare
#   algorithms per data pass, not only per epoch.
# - Voxels outside the scanner FOV have zero sensitivity and must be masked
#   in the preconditioner.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_01_run_sgd_svrg.py`
#    -- SGD and SVRG for a regularised objective (logcosh prior).
#
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_number = 1
