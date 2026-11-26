"""
Listmode SGD and SVRG with a logcosh prior
==========================================

:bdg-primary:`reconstruction` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

An edge-preserving prior suppresses the noise of maximum-likelihood
reconstructions, but the EM update no longer applies.  This example minimises
the regularised negative Poisson log-likelihood of simulated TOF data with
preconditioned **stochastic gradient descent (SGD)** and **SVRG** (stochastic
variance-reduced gradient), each run on a sinogram and on an event list.
Once the subset objective functions are defined, the SGD and SVRG loops are
*identical* for both data representations: they only use the
:class:`.C1Function` interface, and the prior is added with ``+``.

**You will learn**

- how to add a logcosh prior to sinogram and listmode subset objectives
  so that they sum exactly to the regularised objective,
- how to build a preconditioner from the sensitivity image and the curvature
  of a separable quadratic majorizer of the prior,
- how to run preconditioned SGD and SVRG on sinogram and listmode subsets
  with the same code,
- how sinogram and listmode variants compare per epoch and per data pass.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_04_listmode_algorithms_00_run_listmode_mlem_osem_svrg.py`
(listmode data, objective and subsets) and
:ref:`sphx_glr_auto_examples_03_algorithms_01_run_sgd_svrg.py`
(SGD and SVRG with a logcosh prior for sinograms).

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
from array_api_compat import size
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
    NegPoissonLogLListmode,
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
# truth (see section 7).  ``sens_factor`` scales the ground truth image, and
# with it the number of counts.

num_subsets = 24

# fewer epochs on the CPU
num_epochs = iterations_for_device(dev, cpu=2, gpu=240 // num_subsets, what="epochs")

sens_factor = 0.2

beta = 1.0
delta_rel = 0.05

# step size of the SGD and SVRG updates
step_size = 1.0

# %%
# 2. Scanner, sinogram and image
# ------------------------------
#
# The same small scanner (16 sides, 5 rings), span-1 sinogram and
# 40 x 40 x 8 image (2 mm voxels) as in the previous example.  The ground
# truth is an elliptic cylinder with hot and cold spherical inserts.
#
# The scanner's cylindrical field of view (FOV) does not cover every voxel of
# the image grid.  Voxels outside the FOV are never intersected by any LOR, so
# their sensitivity :math:`(A^T \mathbf{1})_i = 0`.  Zeroing the initial image
# and the preconditioner outside the FOV keeps those voxels at zero throughout
# the reconstruction.  ``fov_mask`` is ``True`` inside the FOV and ``None``
# when every image voxel is inside the FOV (no masking needed).

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

cyl_mask = proj.fov_mask()
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask

x_true = sens_factor * elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# %%
# 3. Sinogram forward model :math:`\bar{y}(x) = A x + s`
# ------------------------------------------------------
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
# 4. Simulated sinogram data
# --------------------------
#
# Noise-free data :math:`A x_{true} + s` with a constant contamination
# :math:`s` of half the mean of :math:`A x_{true}`, and a Poisson
# realisation :math:`y` of them, stored as integers.

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
# 5. Conversion of the sinogram to listmode
# -----------------------------------------
#
# :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` converts the
# integer sinogram into a shuffled event list (LOR endpoints and TOF bin of
# every event).

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y, shuffle=True
)

# %%
# 6. Listmode forward model
# -------------------------
#
# As in the previous example, the listmode operator :math:`A_\text{LM}`
# chains the resolution model, a :class:`.ListmodePETProjector` and the
# (non-TOF) attenuation factor of every event.  The contamination of every
# event is the constant contamination of the sinogram bins.


def setup_listmode_projector(
    start_coords: Array, end_coords: Array, tofbins: Array | None
) -> tuple[parallelproj.projectors.ListmodePETProjector, Array]:
    """TOF listmode projector for an event list and its attenuation factors"""
    lm_proj = parallelproj.projectors.ListmodePETProjector(
        start_coords, end_coords, img_shape, voxel_size, proj.img_origin
    )
    # attenuation factors of all events (non-TOF projection)
    att_list = xp.exp(-lm_proj(x_att))
    lm_proj.tof_parameters = proj.tof_parameters
    if proj.tof:
        assert tofbins is not None
        lm_proj.event_tofbins = tofbins
        lm_proj.tof = proj.tof
    return lm_proj, att_list


lm_proj, att_list = setup_listmode_projector(
    event_start_coords, event_end_coords, event_tofbins
)
lm_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_list)

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
# 7. The logcosh prior and the subset decomposition
# -------------------------------------------------
#
# We minimise the regularised negative Poisson log-likelihood
#
# .. math::
#     F(x) = \underbrace{\sum_i \bar{y}_i - y_i \log \bar{y}_i}_{
#            \text{data fidelity}} + \beta \, R(x),
#     \qquad \bar{y}(x) = A x + s
#
# subject to :math:`x \geq 0`, where the edge-preserving logcosh penalty is
#
# .. math::
#     R(x) = \delta \sum_i \log\!\cosh\!\left(\frac{(Gx)_i}{\delta}\right)
#
# and :math:`G` is the finite forward-difference operator
# (:class:`.FiniteForwardDifference`, combined with :class:`.LogCosh`).  The
# scale ``delta`` is ``delta_rel`` times the maximum of the ground truth
# image.  With ``delta_rel = 0.05``, edges with a gradient equal to the image
# maximum are placed firmly in the linear (edge-preserving) regime, while
# smooth-region gradients near zero are penalised quadratically.
#
# The data fidelity is expressed in two equivalent ways (same gradient, see
# the previous example):
#
# * **Sinogram** -- :class:`.C2AffineObjective` wrapping
#   :class:`.NegPoissonLogL` (operates on the predicted counts
#   :math:`\bar{y}`).
# * **Listmode** -- :class:`.NegPoissonLogLListmode` (operates directly on
#   the image :math:`x` with the forward model built in).
#
# Both algorithms use the decomposition into :math:`m` subset functions
#
# .. math::
#     f_k(x) = \underbrace{\sum_{i \in S_k} \bar{y}_i - y_i \log \bar{y}_i}_{
#              \text{subset data fidelity}} + \frac{\beta}{m} R(x),
#     \qquad k = 1, \ldots, m,
#
# so that :math:`F(x) = \sum_{k=1}^m f_k(x)` exactly.  The full prior ``reg``
# (weight :math:`\beta`) is used to evaluate :math:`F` and the
# preconditioner; the subset functions use ``reg_per_subset`` (weight
# :math:`\beta / m`), added with the :meth:`~.C1Function.__add__` operator.

G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)

delta = float(xp.max(x_true)) * delta_rel

reg = C2AffineObjective(LogCosh(delta=delta, beta=beta), G)
reg_per_subset = C2AffineObjective(LogCosh(delta=delta, beta=beta / num_subsets), G)

# %%
# 8. Sensitivity image and full objective
# ---------------------------------------
#
# The sensitivity image :math:`A^T\mathbf{1}` is required by
# :class:`.NegPoissonLogLListmode` and the preconditioner.  The full
# regularised sinogram objective :math:`F` is used to evaluate the iterates
# of all variants.
#
# .. note::
#     By default :class:`.NegPoissonLogL` evaluates a "safe epsilon"
#     (shifted Poisson) surrogate: a tiny ``eps = rel_eps * mean(y)`` is
#     added to the measured and the expected data.  This is finite for any
#     non-negative expectation (never ``nan`` / ``inf``), at the price of a
#     tiny (~``rel_eps``) bias that vanishes at the fit.  Since our
#     contamination is strictly positive, the expected data ``A x + s`` are
#     positive in every bin and we can use ``exact=True`` to evaluate the
#     unmodified log-likelihood instead (bins with ``y = 0`` are still
#     handled exactly).  Keep the default whenever the expected data can
#     reach zero in bins with counts.

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

exact_mode = bool(xp.min(contamination) > 0)

sinogram_data_fidelity = C2AffineObjective(
    NegPoissonLogL(y, exact=exact_mode), pet_lin_op, contamination
)
total_objective_sino = sinogram_data_fidelity + reg

# %%
# 9. Sinogram subset objectives :math:`f_k^{\text{sino}}`
# -------------------------------------------------------
#
# The sinogram views are distributed over ``num_subsets`` subsets.  Each
# subset objective is a sinogram data-fidelity term with the subset operator
# :math:`A^k` plus :math:`\beta/m \cdot R(x)`.

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)
# the attenuation sinogram has no TOF axis: slices for 3 dimensions
_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear the cached LOR endpoints since we create many copies of the projector
proj.clear_cached_lor_endpoints()

sino_subset_linops = []
for k in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[k]

    subset_att_sino = att_sino[subset_slices_non_tof[k]]
    subset_att_values = (
        xp.broadcast_to(xp.expand_dims(subset_att_sino, axis=-1), subset_proj.out_shape)
        if subset_proj.tof
        else subset_att_sino
    )
    sino_subset_linops.append(
        parallelproj.operators.CompositeLinearOperator(
            [
                parallelproj.operators.ElementwiseMultiplicationOperator(
                    subset_att_values
                ),
                subset_proj,
                res_model,
            ]
        )
    )

# f_k^sino = data_fidelity_k + (beta/m) * R(x)
sino_subset_objectives: list[C1Function] = [
    C2AffineObjective(
        NegPoissonLogL(y[sl], exact=exact_mode),
        sino_subset_linops[k],
        contamination[sl],
    )
    + reg_per_subset
    for k, sl in enumerate(subset_slices)
]

# %%
# 10. Listmode subset objectives :math:`f_k^{\text{LM}}`
# ------------------------------------------------------
#
# The event list is partitioned into ``num_subsets`` subsets by taking every
# :math:`m`-th event: subset :math:`k` uses the events with indices
# :math:`k, k{+}m, k{+}2m, \ldots`.  Since each subset holds approximately
# :math:`1/m` of all events, its sensitivity image is approximated as
# :math:`A^T\mathbf{1} / m`, and the contamination constant is scaled by the
# same factor.  The same :math:`\beta/m` prior is added to each listmode
# subset objective.

lm_subset_adj_ones = adjoint_ones / num_subsets
lm_subset_contamination_sum = float(xp.sum(contamination)) / num_subsets

lm_subset_objectives: list[C1Function] = []
for k in range(num_subsets):
    lm_proj_k, att_list_k = setup_listmode_projector(
        xp.asarray(event_start_coords[k::num_subsets], copy=True),
        xp.asarray(event_end_coords[k::num_subsets], copy=True),
        xp.asarray(event_tofbins[k::num_subsets], copy=True) if proj.tof else None,
    )
    lm_att_op_k = parallelproj.operators.ElementwiseMultiplicationOperator(att_list_k)
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
# 11. Preconditioner and warm start
# ---------------------------------
#
# SGD and SVRG take preconditioned gradient steps with the diagonal
# preconditioner
#
# .. math::
#     D = \operatorname{diag}\left(\frac{x}{S + \beta\, c_R(x)\, x}\right),
#
# with the curvature :math:`\beta\, c_R(x)` of the separable quadratic
# majorizer of the prior (``reg.separable_majorizer_curvature(x)``, an upper
# bound of the prior, safe also for strong priors, see
# :ref:`sphx_glr_auto_examples_03_algorithms_08_run_svrg_preconditioners.py`)
# and the full sensitivity image :math:`S = A^T\mathbf{1}` for both the
# sinogram and the listmode variants (the scaled subset gradient
# :math:`m \nabla f_k` approximates the gradient of the *full* objective).
# The preconditioner is zero outside the FOV.
#
# The warm start for all four reconstructions is four preconditioned LM-SGD
# subset updates from a uniform image (zero outside the FOV).


def preconditioner(x: Array) -> Array:
    """Diagonal preconditioner x / (A^T 1 + beta c_R(x) x), 0 outside the FOV"""
    denom = adjoint_ones + reg.separable_majorizer_curvature(x) * x
    if fov_mask is None:
        return x / denom
    return xp.where(fov_mask, x / denom, xp.zeros_like(x))


x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
if fov_mask is not None:
    x_init = xp.where(fov_mask, x_init, xp.zeros_like(x_init))

num_init_updates = 4

for k in range(num_init_updates):
    print(f"warm-start LM-SGD subset {(k + 1):03} / {num_init_updates:03}", end="\r")
    init_precond = preconditioner(x_init)
    x_init = xp.clip(
        x_init
        - init_precond * (num_subsets * lm_subset_objectives[k].gradient(x_init)),
        0,
        None,
    )
print()

# %%
# 12. SGD (sinogram and listmode)
# -------------------------------
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
# The preconditioner is updated infrequently to amortise its cost: at the
# start of epochs 1, 3 and 5, and kept fixed afterwards.  The same function
# runs the sinogram and the listmode variant; every iterate is evaluated with
# the regularised *sinogram* objective :math:`F`.


def run_sgd(
    subset_objectives: Sequence[C1Function], label: str
) -> tuple[Array, list[float]]:
    """SGD from x_init; returns the image and the objective F after each epoch"""
    df: list[float] = []
    x = xp.asarray(x_init, copy=True)
    for i in range(num_epochs):
        if i % 2 == 0 and i <= 4:
            precond = preconditioner(x)
        for k in range(num_subsets):
            print(
                f"{label} epoch {(i + 1):04} / {num_epochs:04}, "
                f"subset {(k + 1):03} / {num_subsets:03}",
                end="\r",
            )
            approx_grad = num_subsets * subset_objectives[k].gradient(x)
            x = xp.clip(x - step_size * precond * approx_grad, 0, None)
        df.append(float(total_objective_sino(x)))
    print()
    return x, df


x_sgd_sino, df_sgd_sino = run_sgd(sino_subset_objectives, "Sino-SGD")
x_sgd_lm, df_sgd_lm = run_sgd(lm_subset_objectives, "LM-SGD")

# %%
# 13. SVRG (sinogram and listmode)
# --------------------------------
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
# anchor phase).  The helper functions only use the :class:`.C1Function`
# interface; the prior is already folded into every subset objective.
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
    """Store all subset gradients at the current anchor point and return their sum.

    Returns
    -------
    stored_grads : Array, shape (m, *x_cur.shape)
        Stacked subset gradients evaluated at the anchor point.
    full_grad : Array, shape x_cur.shape
        Sum of all subset gradients (the full gradient of F).
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
    """Single SVRG subset update with variance-reduced gradient."""
    m = len(subset_obj_functions)
    grad_k = subset_obj_functions[subset_idx].gradient(x_cur)
    approx_grad = (
        m * (grad_k - stored_snapshot_subset_gradients[subset_idx])
        + full_snapshot_gradient
    )
    return xp.clip(x_cur - step_size * precond * approx_grad, 0, None)


def run_svrg(
    subset_objectives: Sequence[C1Function], label: str
) -> tuple[Array, list[float]]:
    """SVRG from x_init; returns the image and the objective F after each epoch"""
    df: list[float] = []
    x = xp.asarray(x_init, copy=True)
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            if epoch <= 4:
                precond = preconditioner(x)
            stored_grads, full_grad = svrg_calc_snapshot_gradients(x, subset_objectives)
            x = xp.clip(x - step_size * precond * full_grad, 0, None)
        for k in range(num_subsets):
            print(
                f"{label} epoch {(epoch + 1):04} / {num_epochs:04}, "
                f"subset {(k + 1):03} / {num_subsets:03}",
                end="\r",
            )
            x = svrg_update(
                x,
                k,
                subset_objectives,
                stored_grads,
                full_grad,
                precond,
                step_size=step_size,
            )
        df.append(float(total_objective_sino(x)))
    print()
    return x, df


x_svrg_sino, df_svrg_sino = run_svrg(sino_subset_objectives, "Sino-SVRG")
x_svrg_lm, df_svrg_lm = run_svrg(lm_subset_objectives, "LM-SVRG")

# %%
# 14. Convergence comparison
# --------------------------
#
# We plot the regularised objective
# :math:`F(x) = \sum_i(\bar{y}_i - y_i\log\bar{y}_i) + \beta R(x)` of all
# variants vs epoch (left) and vs full data passes (right): one per SGD epoch,
# two per SVRG epoch with an anchor phase and one otherwise.

epochs = np.arange(1, num_epochs + 1)
svrg_passes = np.cumsum(np.where(np.arange(num_epochs) % 2 == 0, 2, 1))

df_min = min(df_sgd_sino + df_sgd_lm + df_svrg_sino + df_svrg_lm)
df_max = max(df_sgd_sino[0], df_sgd_lm[0])

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

for ax, svrg_xvals in ((axs[0], epochs), (axs[1], svrg_passes)):
    ax.plot(epochs, df_sgd_sino, marker="o", label="Sino-SGD")
    ax.plot(epochs, df_sgd_lm, marker="s", label="LM-SGD")
    ax.plot(svrg_xvals, df_svrg_sino, marker="^", label="Sino-SVRG")
    ax.plot(svrg_xvals, df_svrg_lm, marker="v", label="LM-SVRG")
    ax.set_ylim(df_min, df_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(ls=":")

axs[0].set_xlabel("epoch")
axs[0].set_ylabel("regularised objective F(x)")
axs[0].set_title(f"Convergence per epoch (beta = {beta})")
axs[1].set_xlabel("full data passes")
axs[1].set_title(f"Convergence per data pass (beta = {beta})")
fig.legend(
    *axs[0].get_legend_handles_labels(),
    loc="outside lower center",
    ncols=4,
    title=f"{num_subsets} subsets, step size {step_size:.1f}",
    fontsize="small",
)
fig.show()

# %%
# SVRG reaches clearly lower objective values than SGD, per epoch and per
# data pass.  Sinogram and listmode variants of each algorithm follow the
# same course; the small offsets between them come from the different subset
# partitions (views vs. every :math:`m`-th event) and subset sensitivities,
# which change the individual subset updates.

# %%
# 15. Reconstructed images
# ------------------------
#
# The final images of all four variants, with the same colour scale.

vmax = float(xp.max(x_sgd_sino))

for img, title in (
    (x_sgd_sino, "Sinogram SGD"),
    (x_sgd_lm, "Listmode SGD"),
    (x_svrg_sino, "Sinogram SVRG"),
    (x_svrg_lm, "Listmode SVRG"),
):
    fig, _, _ = show_vol_cuts(
        img,
        voxel_size=voxel_size,
        fig_title=f"{title} ({num_subsets} subsets, {num_epochs} epochs)",
        vmin=0,
        vmax=vmax,
    )
    fig.show()

# %%
# Sinogram and listmode images of the same algorithm look alike.  Thanks to
# the prior, the images are much less noisy than the unregularised
# reconstructions of the previous example, and the inserts are visible.

# %%
# Summary
# -------
#
# - Adding :math:`\beta R / m` to each sinogram or listmode subset data
#   fidelity splits the regularised objective exactly into subset functions,
#   :math:`F = \sum_k f_k`.
# - The preconditioner :math:`x / (A^T\mathbf{1} + \beta c_R(x) x)` combines
#   the EM preconditioner with the curvature of a separable quadratic
#   majorizer of the prior.
# - SGD and SVRG only use the :class:`.C1Function` interface: the same code
#   reconstructs sinogram and listmode data with similar convergence.
# - SVRG reaches lower objective values than SGD; an SVRG epoch with an anchor
#   phase costs two data passes, so compare per data pass, not only per epoch.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_04_listmode_algorithms_02_lm_spdhg.py`
#    -- listmode SPDHG with a non-smooth total variation prior.

# sphinx_gallery_thumbnail_number = 1
