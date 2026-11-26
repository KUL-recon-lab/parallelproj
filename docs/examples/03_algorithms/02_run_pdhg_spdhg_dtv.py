"""
PDHG and SPDHG with a directional total variation prior
=======================================================

:bdg-primary:`reconstruction` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

Edge-preserving priors such as total variation (TV) are not differentiable, so
they cannot be used with MLEM-type or gradient-based algorithms.  The
primal-dual hybrid gradient algorithm (PDHG) and its stochastic variant (SPDHG)
solve such problems using only proximal operators.  In this example we use them
to reconstruct simulated TOF PET data with a **directional TV (DTV)** prior,
which uses a structural image (e.g. an MR image) to favour edges in the PET
image where the structural image has edges.

**You will learn**

- how the PET reconstruction problem with a DTV prior is written as a sum of
  functions with simple proximal operators,
- how PDHG and SPDHG work, implemented in a single update function,
- how to choose the step sizes of both algorithms and the selection
  probabilities of SPDHG,
- how to split a TOF sinogram forward model into subsets for SPDHG,
- how the convergence of PDHG and SPDHG compares per epoch.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF sinogram forward model) and
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(MLEM, subsets and epochs).

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

import parallelproj.functions
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

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. The reconstruction problem
# -----------------------------
#
# We solve the regularized PET reconstruction problem
#
# .. math::
#     \min_x \; f_\text{data}(Ax + s) + \beta \, f_\text{reg}(Dx) + g(x)
#
# where
#
# - :math:`f_\text{data} = \text{NegPoissonLogL}` -- the negative Poisson
#   log-likelihood,
# - :math:`f_\text{reg} = \text{MixedL21Norm}` -- the isotropic mixed L2-L1
#   norm (TV semi-norm),
# - :math:`g = \iota_{\geq 0}` -- the indicator function of the non-negative
#   orthant,
# - :math:`D = P_{\xi} G` -- the projected finite-difference gradient operator
#   implementing a directional total variation (DTV) structural prior,
# - :math:`A` -- the composite PET forward operator (resolution model, TOF
#   projector, attenuation), and
# - :math:`s` -- the contamination sinogram.
#
# All three functions have simple proximal operators (or proximal operators of
# their convex conjugates), which is all PDHG and SPDHG need.
#
# Both algorithms are implemented through the single :func:`spdhg_update`
# function below.  Passing ``probs=None`` performs a **full PDHG** update: all
# dual blocks are updated every epoch and the over-relaxed variable is scaled
# by 1.  Passing per-block selection probabilities activates **SPDHG**: only
# one randomly selected block is updated per mini-iteration and the
# over-relaxed variable is scaled by :math:`1/p_i`, where :math:`p_i` is the
# selection probability of that block.
#
# SPDHG splits the forward operator :math:`A` into :math:`n` sinogram subsets
# :math:`A = A^1 + \ldots + A^n` and updates the image after every subset.
# One epoch (one pass over all data) is about as expensive as a PDHG
# iteration, but contains many more image updates, so SPDHG usually needs
# fewer epochs to converge.
#
# See :footcite:p:`Ehrhardt2016` and :footcite:p:`Ehrhardt2019` for details on
# the DTV prior and the SPDHG algorithm (Algorithm 2), and
# :footcite:p:`Schramm2022` for the step-size rules used here.

# %%
# 2. A unified PDHG / SPDHG update function
# -----------------------------------------
#
# .. admonition:: Unified PDHG / SPDHG algorithm
#
#   Minimize :math:`\sum_{k=1}^{n+1} f_k(K_k x + c_k) + g(x)` where the
#   first :math:`n` blocks are data subsets and block :math:`n{+}1` is the
#   regularizer.
#
#   | **Input** data :math:`d`, operators :math:`K_1,\ldots,K_{n+1}`,
#     probabilities :math:`p_1,\ldots,p_{n+1}` (or ``None`` for full PDHG)
#   | **Initialize** primal :math:`x`, duals :math:`y_1,\ldots,y_{n+1}`;
#     step sizes :math:`S_i`, :math:`T`
#   | **Preprocessing** :math:`z = \bar{z} = \sum_i K_i^T y_i`
#   | **Repeat** until stopping criterion is met
#   |     :math:`x \;\gets\; \mathrm{prox}_{T g}(x - T\bar{z})`
#   |     **if** ``probs`` is ``None`` **(full PDHG)**:
#   |         update all
#     :math:`y_i^+ \gets \mathrm{prox}_{S_i f_i^*}(y_i + S_i (K_i x + c_i))`
#   |         :math:`\Delta z \gets \sum_i K_i^T(y_i^+ - y_i)`,
#     :math:`\quad \bar{z} \gets z + \Delta z`
#   |     **else (SPDHG)**:
#   |         Select :math:`i \in \{1,\ldots,n{+}1\}` with probabilities
#     :math:`(p_i)`
#   |         :math:`y_i^+ \gets \mathrm{prox}_{S_i f_i^*}(y_i + S_i (K_i x + c_i))`
#   |         :math:`\Delta z \gets K_i^T(y_i^+ - y_i)`,
#     :math:`\quad \bar{z} \gets z + \Delta z / p_i`
#   |     :math:`z \gets z + \Delta z`
#   | **Return** :math:`x`
#
# Passing ``probs=None`` performs a full PDHG update (all blocks updated every
# call, scale factor 1).  Passing per-block probabilities activates SPDHG
# (Algorithm 2 from :footcite:p:`Ehrhardt2019`), which touches only one block
# per mini-iteration and scales :math:`\bar{z}` by :math:`1/p_i`.
#
# .. admonition:: Step sizes
#
#  :math:`S^k = \gamma \, \text{diag}\!\left(\frac{\rho}{A^k \mathbf{1}}\right)`
#  for data subsets
#
#  :math:`S_D = \gamma \, \frac{\rho}{\|D\|}` for regularization
#
#  :math:`T^k = \gamma^{-1} \, \frac{\rho \, p_k}{(A^k)^T \mathbf{1}}` for data
#  subsets
#
#  :math:`T_D = \gamma^{-1} \, \frac{\rho \, p_D}{\|D\|}` for regularization
#
#  :math:`T = \min(T^1, \ldots, T^n, T_D)` elementwise
#
# See :footcite:p:`Ehrhardt2019` and :footcite:p:`Schramm2022` for more details.


def spdhg_update(
    x: Array,
    dual_vars: list[Array],  # modified in-place
    z_array: Array,
    zbar_array: Array,
    f_functions: Sequence[parallelproj.functions.FunctionWithConjProx],
    ops: Sequence[parallelproj.operators.LinearOperator],
    contams: Sequence[Array | None],
    g_function: parallelproj.functions.FunctionWithProx,
    dual_step_sizes: Sequence[float | Array],
    primal_step_size: float | Array,
    probs: Sequence[float] | None = None,
) -> tuple[Array, Array, Array]:
    """Unified PDHG / SPDHG update for problems with multiple linear operators.

    Minimizes ``sum_i f_i(K_i x + c_i) + g(x)`` where each ``f_i`` has a
    known proximal operator of its convex conjugate and ``g`` has a known
    proximal operator.

    The primal variable is updated first::

        x <- prox_{T g}(x - T * zbar)

    Then the dual variable(s) are updated depending on the mode:

    **Full PDHG mode** (``probs=None``):
        All dual variables are updated each call.  For every block i::

            y_i+ <- prox_{S_i f_i*}(y_i + S_i (K_i x + c_i))
            delta_z += K_i^T (y_i+ - y_i)
            y_i <- y_i+

        The auxiliary variables are then updated as::

            z    <- z + delta_z
            zbar <- z + delta_z          (scale factor 1, i.e. p_i = 1)

    **SPDHG mode** (``probs`` is a sequence of floats):
        One block i_sub is drawn with probabilities ``probs``.  Only that
        dual variable is updated::

            y_i+ <- prox_{S_i f_i*}(y_i + S_i (K_i x + c_i))
            delta_z = K_i^T (y_i+ - y_i)
            y_i <- y_i+

        The auxiliary variables are then updated as::

            z    <- z + delta_z
            zbar <- z + delta_z / p_i   (scale factor 1/p_i)

    Parameters
    ----------
    x :
        Current primal variable.  A new array is returned for the updated
        primal variable; the original is not mutated.
    dual_vars :
        List of current dual variables ``y_i``.  In PDHG mode all entries are
        updated in-place; in SPDHG mode only the selected entry is updated.
    z_array :
        Auxiliary variable ``z = sum_i K_i^T y_i``.
    zbar_array :
        Over-relaxed auxiliary variable ``zbar``.
    f_functions :
        Functions ``f_i``, each exposing a proximal operator of their convex
        conjugate via
        :meth:`~parallelproj.functions.FunctionWithConjProx.prox_convex_conj`.
    ops :
        Linear operators ``K_i``, one per function in ``f_functions``.
    contams :
        Additive offsets ``c_i`` applied after each forward projection
        ``K_i x``.  Pass ``None`` for terms without an offset.
    g_function :
        Proximal-friendly constraint or regularization function ``g``,
        exposing :meth:`~parallelproj.functions.FunctionWithProx.prox`.
    dual_step_sizes :
        Dual step sizes ``S_i``, one per function/operator pair.
    primal_step_size :
        Primal step size ``T``.
    probs :
        Selection probabilities ``p_i`` for each block.  When ``None`` (the
        default) a full PDHG update is performed (all blocks updated, scale
        factor 1).  When provided, one block is selected at random and the
        over-relaxation is scaled by ``1 / p_i`` (SPDHG mode).

    Returns
    -------
    x :
        Updated primal variable.
    z_array :
        Updated auxiliary variable ``z``.
    zbar_array :
        Updated over-relaxed auxiliary variable ``zbar``.
    """

    # primal update: prox of g (e.g. non-negativity indicator)
    x -= primal_step_size * zbar_array
    x = g_function.prox(x, primal_step_size)

    if probs is None:
        # full PDHG: update all dual variables
        delta_z = xp.zeros_like(z_array)
        for i, (f, op, contam, S) in enumerate(
            zip(f_functions, ops, contams, dual_step_sizes)
        ):
            fwd = op(x)
            if contam is not None:
                fwd += contam
            y_plus = f.prox_convex_conj(dual_vars[i] + S * fwd, S)
            delta_z += op.adjoint(y_plus - dual_vars[i])
            dual_vars[i] = y_plus

        z_array += delta_z
        zbar_array = z_array + delta_z  # scale factor 1

    else:
        # SPDHG: update one randomly selected block
        i_sub = np.random.choice(len(f_functions), p=probs)

        fwd = ops[i_sub](x)
        if contams[i_sub] is not None:
            fwd += contams[i_sub]

        y_plus = f_functions[i_sub].prox_convex_conj(
            dual_vars[i_sub] + dual_step_sizes[i_sub] * fwd, dual_step_sizes[i_sub]
        )

        delta_z = ops[i_sub].adjoint(y_plus - dual_vars[i_sub])
        dual_vars[i_sub] = y_plus

        z_array += delta_z
        zbar_array = z_array + delta_z / probs[i_sub]  # scale factor 1/p_i

    return x, z_array, zbar_array


# %%
# 3. Input parameters
# -------------------
#
# The number of counts, the number of iterations and subsets, the
# regularization weight :math:`\beta`, the step-size parameters
# :math:`\gamma` and :math:`\rho`, and the SPDHG selection probabilities.

# image scale (can be used to simulate more or less counts)
img_scale = 0.1
# number of MLEM epochs used to initialize PDHG and SPDHG
num_epochs_mlem = 10
# number of SPDHG epochs (each = 2 * num_subsets mini-iterations)
num_epochs_spdhg = iterations_for_device(dev, cpu=10, gpu=20, what="SPDHG epochs")
# number of sinogram subsets for SPDHG
num_subsets = 28
# number of PDHG epochs
num_epochs_pdhg = iterations_for_device(
    dev, cpu=10, gpu=num_epochs_spdhg * num_subsets, what="PDHG iterations"
)
# regularization weight
beta = 6.0
# step size ratio (used by both PDHG and SPDHG)
gamma = 10.0 / img_scale
# rho parameter controlling the step size margin (used by both PDHG and SPDHG)
rho = 0.9999
# contamination in every sinogram bin relative to mean of trues sinogram
contam = 1.0
# probability of the regularization (gradient) block update per mini-iteration.
# Chosen as 0.5 so that each outer SPDHG epoch (2*num_subsets mini-iterations)
# produces on average num_subsets data-subset updates and num_subsets reg updates:
#   E[data subset visits] = p_a * 2 * num_subsets = 2 * (1 - p_g) = 1  per subset
#   E[reg visits]         = p_g * 2 * num_subsets = num_subsets
p_g = 0.5
# probability of each data subset block update per mini-iteration
p_a = (1 - p_g) / num_subsets
# evaluate the cost function after every epoch
track_cost = True

# %%
# 4. Scanner and sinogram projector
# ---------------------------------
#
# A small scanner with 28 sides and 2 rings, a sinogram in (radial, view,
# plane) order and an image of 40 x 40 x 4 voxels of 4 x 4 x 2.5 mm.

num_rings = 2
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=350.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-2.5, 2.5, num_rings, device=dev),
    symmetry_axis=2,
)

img_shape = (40, 40, 4)
voxel_size = (4.0, 4.0, 2.5)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=170,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# %%
# 5. Phantom and structural prior image
# -------------------------------------
#
# The ground truth :math:`x_{true}` is an elliptic cylinder with hot and cold
# spherical inserts.  The structural prior image, from which the DTV operator
# takes its edge directions, is derived from it: it has an inverted contrast
# (like an anatomical image of a different modality), and the insert with
# contrast 3 is missing (set to the background value), so the prior has no
# edges there.

x_true = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# setup a structural prior image
x_struct = -1.0 * xp.sqrt(x_true)
x_struct[x_true == 3] = -1.0

# scale image to get more counts
x_true *= img_scale

fig_true, _, _ = show_vol_cuts(
    x_true, voxel_size=voxel_size, vmin=0, fig_title="True activity image"
)
fig_true.show()

fig_struct, _, _ = show_vol_cuts(
    x_struct, voxel_size=voxel_size, fig_title="Structural prior image"
)
fig_struct.show()

# %%
# The structural image is darkest (0) outside the cylinder and in the cold
# insert with zero activity, and brightest in the hot inserts.  The hot insert
# with contrast 3 (near the bottom of the transaxial slice) is not visible in
# it.

# %%
# 6. Complete TOF forward model
# -----------------------------
#
# The attenuation sinogram (water, 0.01 / mm, inside the cylinder) is computed
# with the non-TOF projector.  We then enable TOF and combine an image-based
# resolution model (4 mm FWHM), the TOF projector and the attenuation into a
# single linear operator :math:`A`.

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the (non-TOF) attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# enable TOF - remove these lines to run non-TOF reconstructions
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=10, tofbin_width=24.0, sigma_tof=24.0
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
    proj.in_shape, sigma=[4.0 / (2.3548 * float(vs)) for vs in proj.voxel_size]
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# 7. Simulation of noisy data
# ---------------------------
#
# We forward project the ground truth, add a constant contamination
# :math:`s` (equal to the mean of the noise-free true sinogram) and draw
# Poisson noise to obtain the data :math:`d`.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a constant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    contam * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
d = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

# %%
# 8. Splitting the forward model into subsets :math:`A^k`
# -------------------------------------------------------
#
# Each subset contains every ``num_subsets``-th view.  The subset views define
# a sequence of projectors that project only these views; the slices extract
# the corresponding subsets from the full data and contamination sinograms
# (and from the non-TOF attenuation sinogram).

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear cached LOR endpoints before creating many copies of the projector
proj.clear_cached_lor_endpoints()

# sequence of subset forward operators: resolution model, subset projector and
# attenuation
pet_subset_linop_seq = []

for i in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

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
            [subset_att_op, subset_proj, res_model]
        )
    )

pet_subset_linop_seq = parallelproj.operators.LinearOperatorSequence(
    pet_subset_linop_seq
)

# %%
# 9. MLEM initialization
# ----------------------
#
# A few MLEM epochs provide a warm start for both PDHG and SPDHG.

x_mlem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_epochs_mlem):
    print(f"MLEM epoch {(i + 1):03} / {num_epochs_mlem:03}", end="\r")
    dbar = pet_lin_op(x_mlem) + contamination
    x_mlem *= pet_lin_op.adjoint(d / dbar) / adjoint_ones

# %%
# 10. DTV operator and function objects
# -------------------------------------
#
# The finite-difference gradient operator :math:`G` is projected by
# :math:`P_{\xi}` to obtain the DTV operator :math:`D = P_{\xi} G`, where
# :math:`\xi` is the normalized gradient field of the structural image.
# The following function objects handle all prox evaluations during PDHG and
# SPDHG:
#
# - ``data_fid_subsets`` -- list of :class:`.NegPoissonLogL`, one per subset
#   (used by SPDHG); ``data_fid_full`` is the full-sinogram version used by
#   PDHG and for cost evaluation,
# - ``nonneg`` -- :class:`.NonNegativeIndicator`, implements
#   :math:`g = \iota_{\geq 0}`,
# - ``reg`` -- :class:`.MixedL21Norm` (weighted by ``beta``), implements
#   :math:`\beta f_\text{reg}`.
#
# .. note::
#     By default :class:`.NegPoissonLogL` evaluates a "safe epsilon"
#     (shifted Poisson) surrogate: a tiny ``eps = rel_eps * mean(y)`` is
#     added to the measured and the expected data, making all evaluations
#     finite at the price of a tiny (~``rel_eps``) bias.  Prox-driven
#     algorithms like PDHG / SPDHG never divide by the expected data -- the
#     closed-form dual prox is stable even for zero-count bins -- so with
#     our strictly positive contamination we use ``exact=True`` and solve
#     the exact problem.  Keep the default whenever the expected data can
#     reach zero in bins with counts.

# setup the finite-difference gradient operator
G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)
# calculate the joint vector field from the structural prior image
joint_vector_field = G(x_struct)
# setup the projected gradient (DTV) operator
P = parallelproj.operators.GradientFieldProjectionOperator(joint_vector_field, eta=1e-4)
D = parallelproj.operators.CompositeLinearOperator((P, G))

# the strictly positive contamination guarantees positive expected data in
# every bin, so the exact (unmodified) log-likelihood can be used
exact_mode = bool(xp.min(contamination) > 0)

# one data-fidelity function per subset
data_fid_subsets = [
    parallelproj.functions.NegPoissonLogL(d[sl], exact=exact_mode)
    for sl in subset_slices
]
nonneg = parallelproj.functions.NonNegativeIndicator()
reg = parallelproj.functions.MixedL21Norm(beta=beta)

# full data fidelity for PDHG and cost evaluation
data_fid_full = parallelproj.functions.NegPoissonLogL(d, exact=exact_mode)

# %%
# 11. PDHG
# --------
#
# The primal variable is warm-started from the MLEM result, the dual variable
# of the data term from the current ratio of data and expected data,
# :math:`y = 1 - d / (A x + s)`, and the dual variable of the regularizer with
# zeros.

x_pdhg = xp.asarray(x_mlem, copy=True)
y = 1 - d / (pet_lin_op(x_pdhg) + contamination)
w = xp.zeros(D.out_shape, dtype=xp.float32, device=dev)

z = pet_lin_op.adjoint(y) + D.adjoint(w)
zbar = xp.asarray(z, copy=True)

# %%
# The step sizes follow the rules from :footcite:p:`Schramm2022` (see the
# admonition in section 2, with :math:`p = 1` for PDHG).  Sinogram bins that
# no LOR hits (:math:`A \mathbf{1} = 0`) get the smallest non-zero value.

tmp = pet_lin_op(xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev))
tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
S_A = gamma * rho / tmp

T_A = (
    (1 / gamma)
    * rho
    / pet_lin_op.adjoint(xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev))
)

D_norm = D.norm(xp, dev, num_iter=100)
S_D = gamma * rho / D_norm
T_D = (1 / gamma) * rho / D_norm

T = xp.where(
    T_A < T_D, T_A, xp.full(pet_lin_op.in_shape, T_D, device=dev, dtype=xp.float32)
)

# %%
# Every PDHG iteration updates both dual blocks (data and regularizer) and
# uses one full forward and back projection, i.e. one epoch.

ys = [y, w]
fs = (data_fid_full, reg)
ops = (pet_lin_op, D)
cons = (contamination, None)

cost_pdhg = np.zeros(num_epochs_pdhg, dtype=np.float32)

for i in range(num_epochs_pdhg):
    x_pdhg, z, zbar = spdhg_update(
        x_pdhg,
        ys,
        z,
        zbar,
        fs,
        ops,
        cons,
        nonneg,
        (S_A, S_D),
        T,
        probs=None,  # full PDHG (all blocks updated every epoch)
    )

    if track_cost:
        cost = data_fid_full(pet_lin_op(x_pdhg) + contamination) + reg(D(x_pdhg))

        cost_pdhg[i] = cost
        print(
            f"PDHG epoch {(i+1):04} / {num_epochs_pdhg}, cost {cost_pdhg[i]:.7e}",
            end="\r",
        )

print("")

# %%
# 12. SPDHG
# ---------
#
# All blocks (data subsets and regularization) are collected into a single
# list.  Each mini-iteration, :func:`spdhg_update` draws one block at random
# according to ``probs_all`` and updates only that dual variable.

# blocks: n data subsets + 1 regularization
fs_all = list(data_fid_subsets) + [reg]
ops_all = list(pet_subset_linop_seq) + [D]
contams_all = [contamination[sl] for sl in subset_slices] + [None]
probs_all = [p_a] * num_subsets + [p_g]

# %%
# The dual step sizes are computed per subset; the primal step size is the
# elementwise minimum of the contributions of all blocks, which now include
# the selection probabilities.

# dual step sizes for data subsets: S^k = gamma * rho / (A^k * 1)
S_A = []
for op in pet_subset_linop_seq:
    tmp = op(xp.ones(op.in_shape, dtype=xp.float32, device=dev))
    tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
    S_A.append(gamma * rho / tmp)

# dual step size for regularization (reuse D_norm computed for PDHG above)
S_D = gamma * rho / D_norm

S_all = S_A + [S_D]

# primal step size contributions: T^k = rho * p_k / (gamma * (A^k)^T * 1)
T_candidates = xp.zeros(
    (num_subsets + 1,) + tuple(pet_lin_op.in_shape), dtype=xp.float32, device=dev
)
for k, op in enumerate(pet_subset_linop_seq):
    adj_ones_k = op.adjoint(xp.ones(op.out_shape, dtype=xp.float32, device=dev))
    T_candidates[k] = (rho * p_a / gamma) / adj_ones_k

T_candidates[-1] = xp.full(
    pet_lin_op.in_shape,
    (rho * p_g / gamma) / D_norm,
    device=dev,
    dtype=xp.float32,
)

# final primal step size: elementwise minimum over all blocks
T = xp.min(T_candidates, axis=0)

del T_candidates  # free memory

# %%
# As for PDHG, the primal variable starts from the MLEM result and the dual
# variables of the data subsets from the subset ratios; the dual variable of
# the regularizer starts at zero.

x_spdhg = xp.asarray(x_mlem, copy=True)

# warm-start dual variables for data subsets
ys = [
    1 - d[sl] / (pet_subset_linop_seq[k](x_spdhg) + contamination[sl])
    for k, sl in enumerate(subset_slices)
]

# zero-initialize the regularization dual variable
w = xp.zeros(D.out_shape, dtype=xp.float32, device=dev)

ys_all = ys + [w]

# initialize z = sum_k (A^k)^T y^k + D^T w  and  zbar = z
z = xp.zeros(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
for k, op in enumerate(pet_subset_linop_seq):
    z += op.adjoint(ys[k])
z += D.adjoint(w)
zbar = xp.asarray(z, copy=True)

# %%
# Each outer epoch consists of ``2 * num_subsets`` mini-iterations.
# In each mini-iteration :func:`spdhg_update` randomly draws one block
# according to ``probs_all`` (probability ``p_a`` per data subset,
# ``p_g`` for the regularization block) and updates only that dual variable.
# With ``p_g = 0.5`` and ``p_a = (1 - p_g) / num_subsets``, the expected
# number of updates per outer epoch is:
#
#   * ``p_a * 2 * num_subsets = 1`` update per data subset  (one full pass)
#   * ``p_g * 2 * num_subsets = num_subsets`` regularization updates
#
# so each outer SPDHG epoch consists, on average, of one pass over all data
# subsets plus ``num_subsets`` updates of the regularization dual variable,
# with an image update after every mini-iteration.

cost_spdhg = np.zeros(num_epochs_spdhg, dtype=np.float32)

for i in range(num_epochs_spdhg):
    for _ in range(2 * num_subsets):
        x_spdhg, z, zbar = spdhg_update(
            x_spdhg,
            ys_all,
            z,
            zbar,
            fs_all,
            ops_all,
            contams_all,
            nonneg,
            S_all,
            T,
            probs=probs_all,
        )

    if track_cost:
        cost = data_fid_full(pet_lin_op(x_spdhg) + contamination) + reg(D(x_spdhg))
        cost_spdhg[i] = cost
        print(
            f"SPDHG epoch {(i+1):04} / {num_epochs_spdhg}, cost {cost_spdhg[i]:.7e}",
            end="\r",
        )

print("")

# %%
# 13. Reconstructed images
# ------------------------
#
# The central transaxial slice of the ground truth, the MLEM initialization
# and the PDHG and SPDHG results, all with the same grey scale.

images = {
    "True image": x_true,
    f"MLEM, {num_epochs_mlem} epochs": x_mlem,
    f"DTV PDHG, {num_epochs_pdhg} epochs": x_pdhg,
    f"DTV SPDHG, {num_epochs_spdhg} epochs\n{num_subsets} subsets": x_spdhg,
}
vmax = 1.2 * float(xp.max(x_true))
sl = img_shape[2] // 2

fig, axes = plt.subplots(
    1, 4, figsize=(10, 3.2), sharex=True, sharey=True, layout="constrained"
)
for ax, (title, img) in zip(axes, images.items()):
    im = ax.imshow(to_numpy_array(img[:, :, sl]).T, cmap="Greys", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x0 voxel")
axes[0].set_ylabel("x1 voxel")
fig.colorbar(im, ax=axes, shrink=0.8, label="activity (a.u.)")
fig.suptitle(f"Transaxial slice x2 = {sl}")
fig.show()

# %%
# The MLEM initialization is blurry and noisy.  The DTV prior smooths the
# noise inside regions while keeping the edges that are present in the
# structural image sharp, e.g. those of the hot insert on the right and the
# cold inserts at the top.  On the CPU, where both algorithms run the same
# number of epochs, the SPDHG image is clearly closer to this solution than
# the PDHG image, which still resembles its MLEM starting point (on a GPU,
# PDHG runs ``num_subsets`` times more epochs than SPDHG).  The hot insert
# missing from the structural image is still reconstructed, but without the
# support of edges in the prior.

# %%
# 14. Convergence
# ---------------
#
# The cost function after every epoch.  One epoch of either algorithm uses
# (on average) one forward and back projection of the full data.

if track_cost:
    fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
    ax.semilogx(np.arange(1, num_epochs_pdhg + 1), cost_pdhg, ".-", label="PDHG")
    ax.semilogx(
        np.arange(1, num_epochs_spdhg + 1),
        cost_spdhg,
        ".-",
        label=f"SPDHG ({num_subsets} subsets)",
    )
    ax.grid(ls=":")
    ax.legend()
    ax.set_xlabel("epoch")
    ax.set_ylabel("cost")
    ax.set_title("Cost function")
    fig.show()

# %%
# At the same epoch, SPDHG reaches a lower cost than PDHG: it updates the
# image after every mini-iteration instead of once per pass over the data.

# %%
# Summary
# -------
#
# - PET reconstruction with a DTV prior is the sum of the Poisson negative
#   log-likelihood, a mixed L2-L1 norm of the projected gradient :math:`P_\xi
#   G x` and the non-negativity constraint; all have simple proximal
#   operators.
# - PDHG updates all dual variables in every iteration; SPDHG updates one
#   randomly selected block (data subset or regularizer) per mini-iteration
#   and scales the over-relaxation by :math:`1/p_i`.
# - The step sizes are set from :math:`A \mathbf{1}`, :math:`A^T \mathbf{1}`,
#   :math:`\|D\|`, the selection probabilities and the ratio :math:`\gamma`.
# - Per epoch, SPDHG decreases the cost faster than PDHG.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_03_run_fbp.py`
#    -- analytic reconstruction with filtered back projection.
#
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_number = 3
