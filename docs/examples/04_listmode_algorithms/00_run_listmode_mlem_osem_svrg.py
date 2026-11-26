"""
Listmode MLEM, OSEM and SVRG
============================

:bdg-primary:`reconstruction` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

In listmode, every detected event is stored individually instead of being
histogrammed into a sinogram, which is efficient for TOF data and short time
frames with fewer events than sinogram bins.  This example reconstructs the
same simulated TOF data from a sinogram and from an event list with **MLEM**,
**OSEM** and **SVRG**, and compares all six variants via the negative Poisson
log-likelihood.  The key point: written as a preconditioned gradient step, the
EM update is *agnostic of the data representation and the subset
partitioning*, so the same update code runs for sinograms and listmode.

**You will learn**

- how to convert a sinogram into an event list and set up the matching
  listmode forward model with a :class:`.ListmodePETProjector`,
- how :class:`.NegPoissonLogLListmode` relates to the sinogram objective
  :class:`.NegPoissonLogL` (same gradient, constant offset in the value),
- how to split an event list into subsets for listmode OSEM and SVRG,
- how one EM / SVRG implementation, written against the
  :class:`.C1Function` interface, runs on sinogram and listmode objectives.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(MLEM, OSEM and SVRG for sinograms) and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_04_run_pet_listmode_projector.py`
(listmode projector).

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
    NegPoissonLogL,
    NegPoissonLogLListmode,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Reconstruction settings
# --------------------------
#
# OSEM and SVRG split the data into ``num_subsets`` subsets; an **epoch** is
# one cycle through all subsets.  For the same number of updates, MLEM runs
# ``num_epochs * num_subsets`` iterations, which is too slow on the CPU, so
# MLEM only runs on a GPU.  ``sens_factor`` scales the sensitivity of the
# scanner, and with it the number of counts.

num_subsets = 24
num_epochs_full = 96 // num_subsets
num_epochs = iterations_for_device(dev, cpu=2, gpu=num_epochs_full, what="epochs")
num_epochs_mlem = num_epochs * num_subsets

# the MLEM reference only runs with the full number of iterations: on a GPU,
# or with PARALLELPROJ_EXAMPLES_FULL_ITERATIONS=1
run_mlem = num_epochs == num_epochs_full

sens_factor = 0.2

# %%
# 2. Scanner, sinogram and image
# ------------------------------
#
# A small regular polygon scanner (16 sides, 5 rings), a span-1 sinogram with
# a maximum ring difference of 2, and an image of 40 x 40 x 8 voxels of 2 mm.
# The ground truth :math:`x_{true}` is an elliptic cylinder with hot and cold
# spherical inserts.
#
# The scanner's cylindrical field of view (FOV) does not cover every voxel of
# the image grid.  Voxels outside the FOV are never intersected by any LOR, so
# their sensitivity :math:`(A^T \mathbf{1})_i = 0`, and dividing by it in the
# EM preconditioner would produce NaN / Inf values that corrupt the
# reconstruction.  ``fov_mask`` is ``True`` inside the FOV and ``None`` when
# every image voxel is inside the FOV (no masking needed).

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

x_true = elliptic_cylinder_phantom(xp, dev)

# %%
# 3. Sinogram forward model :math:`\bar{y}(x) = A x + s`
# ------------------------------------------------------
#
# The linear operator :math:`A` combines an image-based resolution model
# (Gaussian, 2 mm FWHM), a TOF projector (13 TOF bins of 12 mm) and
# attenuation (water, :math:`\mu` = 0.01 / mm, inside the phantom) times
# ``sens_factor``.  The attenuation sinogram is computed before TOF is
# enabled, so it has no TOF axis; for TOF, ``xp.broadcast_to`` repeats it over
# the TOF bins without copying data (a zero-stride view).  Leave out the
# ``proj.tof_parameters`` line to reconstruct non-TOF data.

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
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    sens_factor * att_values
)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[2.0 / (2.3548 * float(vs)) for vs in voxel_size]
)

pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# 4. Simulated sinogram data
# --------------------------
#
# The noise-free data are :math:`A x_{true} + s`, with a constant
# contamination :math:`s` (scatter and randoms) of half the mean of
# :math:`A x_{true}`.  The measured data :math:`y` are a Poisson realisation,
# stored as integers so that they can be converted to an event list.

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
# :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` converts an
# integer non-TOF or TOF sinogram into an event list: the world coordinates of
# the two LOR endpoints and the TOF bin of every event.  The events are
# created bin by bin, i.e. sorted; ``shuffle=True`` puts them into random
# order, as in a real acquisition, which matters when the list is split into
# subsets.

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y, shuffle=True
)

# %%
# 6. Listmode forward model
# -------------------------
#
# The listmode operator :math:`A_\text{LM}` maps an image to the expected
# counts of every event's sinogram bin.  It chains the same resolution model,
# a :class:`.ListmodePETProjector` for the event list and the attenuation
# factor of every event.  The attenuation factors are non-TOF line integrals,
# so they are computed before TOF is enabled in the listmode projector.  The
# contamination :math:`s_\text{LM}` of every event is the (constant)
# contamination of its sinogram bin.


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
lm_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    sens_factor * att_list
)

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
# 7. Sinogram and listmode objective functions
# --------------------------------------------
#
# Both objective functions from :mod:`parallelproj.functions` expose the same
# :class:`.C1Function` interface (``__call__`` for the value, ``gradient`` for
# the gradient w.r.t. the image :math:`x`).
#
# The sinogram objective wraps :class:`.NegPoissonLogL`, which operates in
# **prediction space** :math:`\bar{y}`, with :class:`.C2AffineObjective` to
# pull the gradient back through the forward model :math:`\bar{y}(x) = Ax + s`:
#
# .. math::
#
#     f_\text{sino}(x) = \sum_i \bigl[(Ax+s)_i - y_i\log(Ax+s)_i\bigr]
#                        + \text{const},
#     \quad \nabla_x f_\text{sino} = A^T\!\left(1 - \tfrac{y}{Ax+s}\right)
#
# (:class:`.NegPoissonLogL` evaluates the *deviance* form, which adds the
# constant :math:`\sum_i y_i \log y_i - y_i`).  The listmode objective
# :class:`.NegPoissonLogLListmode` operates directly in **image space**
# :math:`x` and has the forward model built in:
#
# .. math::
#
#     f_\text{LM}(x) = \langle A^T \mathbf{1},\, x \rangle + c_\text{sino}
#                      - N_\text{ev}
#                      - \sum_{e=1}^{N_\text{ev}}
#                        \log\bigl((A_\text{LM}\,x)_e + s_e\bigr),
#     \quad
#     \nabla_x f_\text{LM}(x) = A^T\mathbf{1}
#         - A_\text{LM}^T\!\left(\frac{1}{A_\text{LM} x + s_\text{LM}}\right)
#
# with :math:`c_\text{sino} = \sum_i s_i` and the number of events
# :math:`N_\text{ev}`.  Each detected event from sinogram bin :math:`i`
# contributes the same :math:`\log\bar{y}_i` term, so
# :math:`\sum_e \log\bar{y}_{j_e} = \sum_i y_i\log\bar{y}_i`: both objectives
# have **identical gradients** and differ only by the constant
# :math:`\sum_i y_i \log y_i`, which cannot be computed from the event list
# alone (it is 0 if no bin contains more than one event).  The sensitivity
# image :math:`A^T \mathbf{1}`, the back projection of ones with the
# *sinogram* forward model, enters the listmode objective explicitly.
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

exact_mode = bool(xp.min(contamination) > 0)

sinogram_neg_logL = C2AffineObjective(
    NegPoissonLogL(y, exact=exact_mode), pet_lin_op, contamination
)

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)
lm_neg_logL = NegPoissonLogLListmode(
    lm_pet_lin_op, adjoint_ones, contamination_list, float(xp.sum(contamination))
)

# %%
# 8. Sinogram and listmode subsets
# --------------------------------
#
# *Sinogram subsets*: ``get_distributed_views_and_slices`` distributes the
# sinogram views over :math:`m` subsets.  Each subset :math:`k` gets a
# sub-operator :math:`A^k` (a copy of the projector that projects only these
# views), sub-data :math:`y^k`, contamination :math:`s^k`, its own
# :class:`.C2AffineObjective` and the exact subset sensitivity image
# :math:`(A^k)^T\mathbf{1}`.

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
    subset_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
        sens_factor * subset_att_values
    )
    sino_subset_linops.append(
        parallelproj.operators.CompositeLinearOperator(
            [subset_att_op, subset_proj, res_model]
        )
    )

# (A^k)^T 1 for every subset k, stored as (num_subsets, *in_shape)
subset_adjoint_ones = xp.zeros(
    (num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, op in enumerate(sino_subset_linops):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

sino_subset_neg_logL = [
    C2AffineObjective(
        NegPoissonLogL(y[sl], exact=exact_mode),
        sino_subset_linops[k],
        contamination[sl],
    )
    for k, sl in enumerate(subset_slices)
]

# %%
# *Listmode subsets*: the (shuffled) event list is split by taking every
# :math:`m`-th event, i.e. subset :math:`k` contains the events with indices
# :math:`k, k{+}m, k{+}2m, \ldots`.  Since each subset holds approximately
# :math:`1/m` of all events, its sensitivity image is approximated as
# :math:`A^T\mathbf{1} / m`, and the contamination constant
# :math:`c_\text{sino}` is scaled by the same factor.  Every subset gets its
# own :class:`.ListmodePETProjector` and :class:`.NegPoissonLogLListmode`.

# shared per-subset sensitivity and contamination constant (both scaled by 1/m)
lm_subset_adj_ones = adjoint_ones / num_subsets
lm_subset_contamination_sum = float(xp.sum(contamination)) / num_subsets

lm_subset_neg_logL = []
for k in range(num_subsets):
    lm_proj_k, att_list_k = setup_listmode_projector(
        xp.asarray(event_start_coords[k::num_subsets], copy=True),
        xp.asarray(event_end_coords[k::num_subsets], copy=True),
        xp.asarray(event_tofbins[k::num_subsets], copy=True) if proj.tof else None,
    )
    lm_att_op_k = parallelproj.operators.ElementwiseMultiplicationOperator(
        sens_factor * att_list_k
    )
    lm_subset_neg_logL.append(
        NegPoissonLogLListmode(
            parallelproj.operators.CompositeLinearOperator(
                (lm_att_op_k, lm_proj_k, res_model)
            ),
            lm_subset_adj_ones,
            contamination_list[k::num_subsets],
            lm_subset_contamination_sum,
        )
    )

# %%
# 9. The EM update as a preconditioned gradient step
# --------------------------------------------------
#
# The classical MLEM update
#
# .. math::
#
#     x^{(k+1)} = \frac{x^{(k)}}{A^T \mathbf{1}} \odot
#                 A^T\!\left(\frac{y}{A x^{(k)} + s}\right)
#
# can equivalently be written as a preconditioned gradient descent step:
#
# .. math::
#
#     x^{(k+1)} = x^{(k)} - \underbrace{\frac{x^{(k)}}{A^T \mathbf{1}}}_{
#                 \text{preconditioner}} \odot \nabla_x f(x^{(k)})
#
# This formulation is **data-representation agnostic**: the same
# :func:`em_update` below works for any objective :math:`f` that provides a
# gradient :math:`\nabla_x f` via the :class:`.C1Function` interface --
# :class:`.C2AffineObjective` wrapping :class:`.NegPoissonLogL` (sinogram
# data) or :class:`.NegPoissonLogLListmode` (listmode data), for the full data
# or a subset.  The preconditioner is set to zero outside the FOV.


def em_preconditioner(x: Array, adj_ones: Array) -> Array:
    """EM preconditioner x / adj_ones, 0 outside the FOV"""
    if fov_mask is None:
        return x / adj_ones
    return xp.where(fov_mask, x / adj_ones, xp.zeros_like(x))


def em_update(x_cur: Array, negpoissonlogl: C1Function, adj_ones: Array) -> Array:
    """One EM update rewritten as a preconditioned gradient descent step.

    Computes :math:`x^+ = x - (x / A^T\\mathbf{1}) \\odot \\nabla_x f(x)` for
    any objective ``negpoissonlogl`` that implements the :class:`.C1Function`
    interface, in particular for both :class:`.C2AffineObjective` (sinogram)
    and :class:`.NegPoissonLogLListmode` (listmode).

    Parameters
    ----------
    x_cur : Array
        Current image estimate :math:`x^{(k)}`.
    negpoissonlogl : C1Function
        Negative Poisson log-likelihood (full data or subset) whose gradient
        w.r.t. :math:`x` is evaluated at ``x_cur``.
    adj_ones : Array
        Sensitivity image :math:`A^T\\mathbf{1}` (or its subset variant)
        used in the diagonal preconditioner.

    Returns
    -------
    Array
        Updated image estimate :math:`x^{(k+1)}`.
    """
    d = em_preconditioner(x_cur, adj_ones)
    return x_cur - d * negpoissonlogl.gradient(x_cur)


# %%
# 10. Warm start and gradient check
# ---------------------------------
#
# All reconstructions start from the same warm start: one LM-OSEM epoch from a
# uniform image (zero outside the FOV).  At the warm start we evaluate both
# full objectives and their gradients.  As explained in section 7, the
# function values differ by a constant, but the gradients agree up to
# floating-point precision.

x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
if fov_mask is not None:
    x_init = xp.where(fov_mask, x_init, xp.zeros_like(x_init))
for k in range(num_subsets):
    print(f"warm-start LM-OSEM subset {(k + 1):03} / {num_subsets:03}", end="\r")
    x_init = em_update(x_init, lm_subset_neg_logL[k], lm_subset_adj_ones)
print()

df_sino = sinogram_neg_logL(x_init)
df_lm = lm_neg_logL(x_init)

print(
    f"Negative Poisson log-likelihood at x_init: {df_sino:.4E} (sinogram), "
    f"{df_lm:.4E} (listmode)"
)

grad_sinogram = sinogram_neg_logL.gradient(x_init)
grad_lm = lm_neg_logL.gradient(x_init)

rel_grad_diff = float(
    xp.linalg.vector_norm(grad_lm - grad_sinogram)
    / xp.linalg.vector_norm(grad_sinogram)
)
print(f"Relative difference of the gradients at x_init: {rel_grad_diff:.2E}")

# %%
# The preconditioned gradient :math:`(x / A^T\mathbf{1}) \odot \nabla_x f`
# equals :math:`x - x^+`, i.e. the negative of a single EM update step: blue
# voxels would increase, red voxels decrease after one MLEM iteration.  We show
# it for the sinogram and the listmode objective with the same colour scale.

precond_grad_sino = em_preconditioner(x_init, adjoint_ones) * grad_sinogram
precond_grad_lm = em_preconditioner(x_init, adjoint_ones) * grad_lm
vmax = float(xp.max(xp.abs(precond_grad_sino)))

fig, _, _ = show_vol_cuts(
    precond_grad_sino,
    voxel_size=voxel_size,
    fig_title="Preconditioned gradient, sinogram objective",
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
)
fig.show()

# %%

fig, _, _ = show_vol_cuts(
    precond_grad_lm,
    voxel_size=voxel_size,
    fig_title="Preconditioned gradient, listmode objective",
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
)
fig.show()

# %%
# The two preconditioned gradients are indistinguishable, as expected from
# the printed small relative difference (float32 rounding).  After the warm
# start, the next EM step would still decrease most voxels inside the phantom
# (red) and increase some voxels near its boundary (blue).

# %%
# 11. MLEM (sinogram and listmode)
# --------------------------------
#
# One MLEM iteration uses the full data and the full sensitivity image
# :math:`A^T \mathbf{1}`.  The sinogram and the listmode loops are identical;
# only the objective passed to :func:`em_update` changes.  We record the
# objective at half and at the full number of iterations.  MLEM only runs when
# ``run_mlem`` is ``True`` (full number of iterations, e.g. on a GPU).


def run_mlem_recon(neg_logL: C1Function, label: str) -> tuple[Array, dict[int, float]]:
    """MLEM from x_init; returns the image and the sinogram objective at checkpoints"""
    df: dict[int, float] = {}
    x = xp.asarray(x_init, copy=True)
    for i in range(num_epochs_mlem):
        print(f"{label} epoch {(i + 1):04} / {num_epochs_mlem:04}", end="\r")
        x = em_update(x, neg_logL, adjoint_ones)
        if (i + 1) in mlem_checkpoints:
            df[i + 1] = float(sinogram_neg_logL(x))
    print()
    return x, df


# iterations at which to record the MLEM objective value
mlem_checkpoints = [num_epochs_mlem // 2, num_epochs_mlem]

if run_mlem:
    x_mlem_sino, df_mlem_sino = run_mlem_recon(sinogram_neg_logL, "MLEM")
    x_mlem_lm, df_mlem_lm = run_mlem_recon(lm_neg_logL, "LM-MLEM")

# %%
# 12. OSEM (sinogram and listmode)
# --------------------------------
#
# One OSEM epoch cycles through all :math:`m` subsets in sequence.  Each
# subset step is the same EM update as in MLEM, but with the subset objective
# and the corresponding subset sensitivity image:
#
# * *Sinogram OSEM* uses a :class:`.C2AffineObjective` per subset together
#   with the exact subset sensitivity :math:`(A^k)^T\mathbf{1}`.
# * *LM-OSEM* uses a :class:`.NegPoissonLogLListmode` per subset together
#   with the approximated subset sensitivity :math:`A^T\mathbf{1}/m`.
#
# Both variants run through the same function, once more demonstrating that
# :func:`em_update` is agnostic of the underlying data representation.  To
# compare all variants on the same scale, every iterate is evaluated with the
# *sinogram* objective.


def run_osem(
    subset_neg_logL: Sequence[C1Function], subset_adj_ones: Sequence[Array], label: str
) -> tuple[Array, list[float]]:
    """OSEM from x_init; returns the image and the sinogram objective per epoch"""
    df: list[float] = []
    x = xp.asarray(x_init, copy=True)
    for i in range(num_epochs):
        for k in range(num_subsets):
            print(
                f"{label} epoch {(i + 1):04} / {num_epochs:04}, "
                f"subset {(k + 1):03} / {num_subsets:03}",
                end="\r",
            )
            x = em_update(x, subset_neg_logL[k], subset_adj_ones[k])
        df.append(float(sinogram_neg_logL(x)))
    print()
    return x, df


x_osem_sino, df_osem_sino = run_osem(sino_subset_neg_logL, subset_adjoint_ones, "OSEM")
x_osem_lm, df_osem_lm = run_osem(
    lm_subset_neg_logL, num_subsets * [lm_subset_adj_ones], "LM-OSEM"
)

# %%
# 13. SVRG (sinogram and listmode)
# --------------------------------
#
# SVRG reduces the variance of the subset gradients with gradients stored at
# an anchor point :math:`\tilde{x}`.  Each epoch consists of two phases:
#
# 1. **Anchor phase** (every other epoch): compute and store all :math:`m`
#    subset gradients :math:`\tilde{g}_k = \nabla f_k(\tilde{x})` at the
#    current point :math:`\tilde{x}`, then take a full gradient step with
#    :math:`\sum_k \tilde{g}_k`.
# 2. **Variance-reduced subset updates**: for each subset :math:`k` form
#
#    .. math::
#
#        g^{VR}_k = m\bigl(\nabla f_k(x) - \tilde{g}_k\bigr) + \sum_j \tilde{g}_j
#
#    and take a preconditioned gradient step, followed by a projection onto
#    :math:`x \geq 0`.
#
# The preconditioner is the EM preconditioner
# :math:`x / (A^T \mathbf{1})`, updated in the anchor phases of epochs 1, 3
# and 5 and kept fixed afterwards.  Both helper functions operate on any list
# of :class:`.C1Function` objects, so the same code runs for the sinogram
# subsets (:class:`.C2AffineObjective`) and the listmode subsets
# (:class:`.NegPoissonLogLListmode`).

svrg_step_size = 1.0


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


def run_svrg(
    subset_neg_logL: Sequence[C1Function], label: str
) -> tuple[Array, list[float]]:
    """SVRG from x_init; returns the image and the sinogram objective per epoch"""
    df: list[float] = []
    x = xp.asarray(x_init, copy=True)
    for epoch in range(num_epochs):
        if epoch % 2 == 0:
            if epoch <= 4:
                precond = em_preconditioner(x, adjoint_ones)
            stored_grads, full_grad = svrg_calc_snapshot_gradients(x, subset_neg_logL)
            x = xp.clip(x - svrg_step_size * precond * full_grad, 0, None)
        for k in range(num_subsets):
            print(
                f"{label} epoch {(epoch + 1):04} / {num_epochs:04}, "
                f"subset {(k + 1):03} / {num_subsets:03}",
                end="\r",
            )
            x = svrg_update(
                x,
                k,
                subset_neg_logL,
                stored_grads,
                full_grad,
                precond,
                step_size=svrg_step_size,
            )
        df.append(float(sinogram_neg_logL(x)))
    print()
    return x, df


x_svrg_sino, df_svrg_sino = run_svrg(sino_subset_neg_logL, "Sino-SVRG")
x_svrg_lm, df_svrg_lm = run_svrg(lm_subset_neg_logL, "LM-SVRG")

# %%
# 14. Convergence comparison
# --------------------------
#
# We plot the **sinogram** negative Poisson log-likelihood of all variants vs
# epoch (left) and vs full data passes (right):
#
# * **OSEM**: one full data pass per epoch.
# * **SVRG**: two passes on anchor epochs (snapshot + subset updates), one
#   pass on the other epochs.
# * **MLEM** (GPU only): one pass per iteration; the values after half and
#   after all ``num_epochs_mlem`` iterations are shown as horizontal lines
#   (darker for sinogram, lighter for listmode).

epochs = np.arange(1, num_epochs + 1)
svrg_passes = np.cumsum(np.where(np.arange(num_epochs) % 2 == 0, 2, 1))

df_min = min(df_osem_sino + df_osem_lm + df_svrg_sino + df_svrg_lm)
if run_mlem:
    df_min = min([df_min, *df_mlem_sino.values(), *df_mlem_lm.values()])
df_max = max(df_osem_sino[0], df_osem_lm[0])

fig, axs = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")

for ax, svrg_xvals in ((axs[0], epochs), (axs[1], svrg_passes)):
    ax.plot(epochs, df_osem_sino, marker="o", label="Sino-OSEM")
    ax.plot(epochs, df_osem_lm, marker="s", label="LM-OSEM")
    ax.plot(svrg_xvals, df_svrg_sino, marker="^", label="Sino-SVRG")
    ax.plot(svrg_xvals, df_svrg_lm, marker="v", label="LM-SVRG")
    if run_mlem:
        # MLEM reference values: grey lines, line style by number of iterations
        for n_iter, ls in zip(mlem_checkpoints, ("-.", "--")):
            ax.axhline(
                df_mlem_sino[n_iter],
                color="0.3",
                ls=ls,
                label=f"Sino-MLEM ({n_iter} iterations)",
            )
            ax.axhline(
                df_mlem_lm[n_iter],
                color="0.6",
                ls=ls,
                label=f"LM-MLEM ({n_iter} iterations)",
            )
    ax.set_ylim(df_min, df_max)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.grid(ls=":")

axs[0].set_xlabel("epoch")
axs[0].set_ylabel("negative Poisson log-likelihood f(x)")
axs[0].set_title("Convergence per epoch")
axs[1].set_xlabel("full data passes")
axs[1].set_title("Convergence per data pass")
fig.legend(
    *axs[0].get_legend_handles_labels(),
    loc="outside lower center",
    ncols=4,
    title=f"{num_subsets} subsets, SVRG step size {svrg_step_size:.1f}",
    fontsize="small",
)
fig.show()

# %%
# The sinogram and listmode SVRG curves lie on top of each other, although
# the subsets are formed differently (views vs. every :math:`m`-th event).
# The two OSEM curves differ slightly: OSEM does not converge to the
# minimum, so its iterates depend on the subset partition (and LM-OSEM uses
# the approximate subset sensitivity :math:`A^T\mathbf{1}/m`).  SVRG reaches
# lower objective values than OSEM per epoch and also per data pass.

# %%
# 15. Reconstructed images
# ------------------------
#
# The final images of all variants, with the same colour scale.

vmax = float(xp.max(x_mlem_sino if run_mlem else x_osem_sino))

recons = [
    (x_osem_sino, f"Sinogram OSEM ({num_subsets} subsets, {num_epochs} epochs)"),
    (x_osem_lm, f"Listmode OSEM ({num_subsets} subsets, {num_epochs} epochs)"),
    (x_svrg_sino, f"Sinogram SVRG ({num_subsets} subsets, {num_epochs} epochs)"),
    (x_svrg_lm, f"Listmode SVRG ({num_subsets} subsets, {num_epochs} epochs)"),
]
if run_mlem:
    recons = [
        (x_mlem_sino, f"Sinogram MLEM ({num_epochs_mlem} iterations)"),
        (x_mlem_lm, f"Listmode MLEM ({num_epochs_mlem} iterations)"),
    ] + recons

for img, title in recons:
    fig, _, _ = show_vol_cuts(
        img, voxel_size=voxel_size, fig_title=title, vmin=0, vmax=vmax
    )
    fig.show()

# %%
# Sinogram and listmode reconstructions of the same algorithm look alike.
# At this low count level and without regularisation, all images are very
# noisy, and the noise increases further as the iterates approach the
# maximum-likelihood solution; the next example adds a prior.

# %%
# Summary
# -------
#
# - :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` turns an
#   integer sinogram into an event list; a :class:`.ListmodePETProjector`
#   with per-event attenuation factors gives the listmode forward model.
# - :class:`.NegPoissonLogLListmode` has the same gradient w.r.t. the image as
#   the sinogram :class:`.NegPoissonLogL` composed with the forward model;
#   their values differ only by a data-dependent constant.
# - Listmode subsets take every :math:`m`-th event of a shuffled list and use
#   the approximated subset sensitivity :math:`A^T\mathbf{1}/m`.
# - Written as a preconditioned gradient step, the EM update -- and SVRG built
#   on top of it -- only needs the :class:`.C1Function` interface: the same
#   code reconstructs sinogram and listmode data, and listmode SVRG converges
#   like its sinogram counterpart.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_04_listmode_algorithms_01_run_listmode_sgd_svrg.py`
#    -- listmode SGD and SVRG for a regularised objective (logcosh prior).

# sphinx_gallery_thumbnail_number = 3
