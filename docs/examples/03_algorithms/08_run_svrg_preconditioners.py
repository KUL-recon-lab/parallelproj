"""
Preconditioners for SVRG with an edge-preserving prior
======================================================

:bdg-primary:`reconstruction` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

Preconditioned stochastic gradient methods such as SVRG take steps
:math:`x^+ = x - D\\, g` with a diagonal preconditioner :math:`D`.  For MAP
reconstruction with a prior, :math:`D` has to account for the curvature of
the data fidelity *and* of the prior.  This example compares three diagonal
preconditioners for a logcosh prior, for high and low count data and for
weak to strong, and more or less edge-preserving, regularisation, and uses a monotone MAP-EM algorithm
built from the separable majorizer of the prior as reference.

**You will learn**

- three ways to extend the EM preconditioner :math:`x / A^T 1` by the
  curvature of the prior: the **Hessian diagonal**
  (:meth:`.C2Function.hessian_diag`), **twice the Hessian diagonal**, and the
  **separable majorizer** (:meth:`.C2Function.separable_majorizer_curvature`),
- how the count level and the regularisation strength change their
  behaviour,
- how the separable majorizer turns De Pierro's MAP-EM algorithm into a
  monotone algorithm for an edge-preserving (non-quadratic) prior.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_01_run_sgd_svrg.py` (SVRG with a
logcosh prior) and :ref:`sphx_glr_auto_examples_03_algorithms_04_run_depierro.py`
(De Pierro's MAP-EM with a quadratic prior).

.. note::

    To keep the documentation build short, this example runs **fewer
    iterations on the CPU** than on a GPU (see the printed notes below).  The
    results shown on CPU-only builds (e.g. on Read the Docs) are therefore
    less converged.  Run the example on a GPU, or set the environment
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

from copy import copy

import matplotlib.pyplot as plt
import numpy as np

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    iterations_for_device,
    suggest_array_backend_and_device,
)
from parallelproj.functions import C2AffineObjective, LogCosh, NegPoissonLogL

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. The problem
# --------------
#
# We minimise the MAP objective
#
# .. math::
#     F(x) = \sum_i \left(\bar{y}_i(x) - y_i \log \bar{y}_i(x)\right)
#     + \beta R(x), \qquad \bar{y}(x) = A x + s, \qquad x \geq 0,
#
# with the edge-preserving logcosh prior
# :math:`R(x) = \sum_i \psi([Gx]_i)`,
# :math:`\psi(t) = \delta \log\cosh(t / \delta)`, and forward differences
# :math:`G` (:class:`.FiniteForwardDifference`).  All three preconditioners
# have the form
#
# .. math::
#     D(x) = \operatorname{diag}\left(\frac{x}{A^T 1 + \beta\, c_R(x)\, x}
#     \right),
#
# and differ in the curvature :math:`c_R` of the prior:
#
# - **Hessian diagonal**: :math:`c_R = \operatorname{diag}(\nabla^2 R(x)) =
#   |G|^T \psi''(Gx)`, the local curvature,
# - **2 x Hessian diagonal**: :math:`c_R = 2\, |G|^T \psi''(Gx)`, a
#   heuristic with the factor 2 of De Pierro's separable surrogate for
#   forward differences (:math:`|G| 1 = 2`, every difference involves two
#   voxels), but with the local curvature :math:`\psi''`,
# - **separable majorizer**: :math:`c_R = |G|^T (\omega(Gx) \odot |G| 1) =
#   2\, |G|^T \omega(Gx)` with Huber's curvature
#   :math:`\omega(t) = \psi'(t) / t \geq \psi''(t)`.  It is the curvature of
#   a separable quadratic that lies above :math:`R` everywhere (De Pierro's
#   surrogate).  2 x the Hessian diagonal is not: :math:`\psi''` of logcosh
#   is largest at :math:`t = 0`, so the quadratic built from it is too flat
#   for larger differences.

# %%
# 2. Scanner, projector and phantom
# ---------------------------------
#
# To compare 3 preconditioners in 6 scenarios in reasonable time, we use a
# single ring (a 2D problem): a 16-sided scanner, a non-TOF sinogram and a
# 55 x 55 x 1 image of 2 mm voxels.  The phantom is an elliptic cylinder with
# hot and cold inserts.

scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=16,
    num_lor_endpoints_per_side=12,
    lor_spacing=2.3,
    ring_positions=xp.asarray([0.0], device=dev),
    symmetry_axis=2,
)
lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(1, max_ring_difference=0, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)
img_shape, voxel_size = (55, 55, 1), (2.0, 2.0, 2.0)
proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

phantom = elliptic_cylinder_phantom(xp, dev, image_shape=img_shape, voxel_size=voxel_size)

# voxels outside the scanner's field of view are never measured
fov_mask = proj.fov_mask()
fov = xp.astype(fov_mask, xp.float32)

# %%
# 3. Forward model and subsets
# ----------------------------
#
# :math:`A` combines a Gaussian resolution model (4 mm FWHM), the projector
# and attenuation (water, :math:`\mu = 0.01` / mm, inside the phantom).  The
# views are distributed over 12 subsets for SVRG.

num_subsets = 12

att_sino = xp.exp(-proj(0.01 * xp.astype(phantom > 0, xp.float32)))
res_model = parallelproj.operators.GaussianFilterOperator(
    img_shape, sigma=[4.0 / (2.3548 * vs) for vs in voxel_size[:2]] + [0.0]
)
A = parallelproj.operators.CompositeLinearOperator(
    (parallelproj.operators.ElementwiseMultiplicationOperator(att_sino), proj, res_model)
)

subset_views, subset_slices = lor_desc.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)
proj.clear_cached_lor_endpoints()
subset_ops = []
for k in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[k]
    subset_ops.append(
        parallelproj.operators.CompositeLinearOperator(
            (
                parallelproj.operators.ElementwiseMultiplicationOperator(
                    att_sino[subset_slices[k]]
                ),
                subset_proj,
                res_model,
            )
        )
    )

sens = A.adjoint(xp.ones(A.out_shape, dtype=xp.float32, device=dev))
subset_sens = [
    op.adjoint(xp.ones(op.out_shape, dtype=xp.float32, device=dev)) for op in subset_ops
]

# %%
# 4. Eight scenarios
# ------------------
#
# We simulate Poisson data with about :math:`3 \cdot 10^6` (high) and
# :math:`3 \cdot 10^4` (low) true counts, with a constant contamination of
# half the mean of the noise-free trues, and combine both with four priors:
# a weak, a medium and a strong one with :math:`\delta` = 5 % of the maximum
# of the image, and a strong one with a small :math:`\delta` = 1 %, which is
# more edge-preserving (closer to total variation): more image differences
# lie in the range :math:`|t| \gg \delta`, where :math:`\psi` is almost
# linear.  :math:`\delta` scales with the image, so the prior scales like the
# data with the count level, and :math:`\beta` means the same for both count
# levels.
#
# The curvature of the prior adds to the denominator :math:`A^T 1` of the
# preconditioners (median over the FOV, Hessian diagonal / majorizer): about
# 3 % / 7 % for the weak prior (:math:`\beta = 3`), 9 % / 25 % for the
# medium prior (:math:`\beta = 10`), 26 % / 75 % for the strong prior
# (:math:`\beta = 30`, visibly over-smoothing) and 35 % / 170 % for the
# strong prior with small :math:`\delta`.  Twice the Hessian diagonal adds
# twice as much as the Hessian diagonal; with the small :math:`\delta`, that
# is still less than half of the majorizer.

# (beta, delta as a fraction of the maximum of the image)
priors = {
    "weak prior": (3.0, 0.05),
    "medium prior": (10.0, 0.05),
    "strong prior": (30.0, 0.05),
    "strong prior, small δ": (30.0, 0.01),
}
total_counts = {"high counts": 3e6, "low counts": 3e4}

G = parallelproj.operators.FiniteForwardDifference(img_shape)
trues_per_unit_activity = float(xp.sum(A(phantom)))

rng = np.random.default_rng(1)
scenarios = {}
for counts_name, counts in total_counts.items():
    x_true = (counts / trues_per_unit_activity) * phantom
    trues = A(x_true)
    contamination = xp.full(A.out_shape, 0.5 * float(xp.mean(trues)), device=dev)
    y = xp.asarray(
        rng.poisson(to_numpy_array(trues + contamination)), dtype=xp.float32, device=dev
    )
    for prior_name, (beta, delta_rel) in priors.items():
        scenarios[f"{counts_name}, {prior_name}"] = dict(
            x_true=x_true,
            y=y,
            s=contamination,
            delta=delta_rel * float(xp.max(x_true)),
            beta=beta,
        )
    print(f"{counts_name}: {float(xp.sum(y)):.3g} counts (trues + contamination)")

# %%
# 5. Objective functions and preconditioners
# ------------------------------------------
#
# For every scenario we build the full objective :math:`F`, the subset
# objectives :math:`f_k` (subset data fidelity plus :math:`\beta R / m`, so
# that :math:`F = \sum_k f_k`) and the three preconditioners.


def objectives(sc: dict):
    """prior, full objective and subset objectives of a scenario"""
    prior = C2AffineObjective(LogCosh(delta=sc["delta"], beta=sc["beta"]), G)
    full = C2AffineObjective(NegPoissonLogL(sc["y"], exact=True), A, sc["s"]) + prior
    subsets = [
        C2AffineObjective(NegPoissonLogL(sc["y"][sl], exact=True), op, sc["s"][sl])
        + C2AffineObjective(
            LogCosh(delta=sc["delta"], beta=sc["beta"] / num_subsets), G
        )
        for op, sl in zip(subset_ops, subset_slices)
    ]
    return prior, full, subsets


def preconditioner(kind: str, prior: C2AffineObjective, x: Array) -> Array:
    """x / (A^T 1 + c_R(x) x) inside the FOV, 0 outside"""
    if kind == "Hessian diagonal":
        denom = sens + prior.hessian_diag(x) * x
    elif kind == "2 x Hessian diagonal":
        denom = sens + 2 * prior.hessian_diag(x) * x
    else:  # separable majorizer
        denom = sens + prior.separable_majorizer_curvature(x) * x
    return xp.where(fov_mask, x / xp.where(fov_mask, denom, 1.0), 0.0)


precond_kinds = ("Hessian diagonal", "2 x Hessian diagonal", "separable majorizer")

# %%
# 6. SVRG
# -------
#
# SVRG as in :ref:`sphx_glr_auto_examples_03_algorithms_01_run_sgd_svrg.py`:
# an anchor (all subset gradients stored, one full gradient step) every other
# epoch, variance-reduced subset steps in between, step size 1, projection on
# :math:`x \geq 0`, and the preconditioner updated at the anchors of the
# first 5 epochs.  An epoch with an anchor costs two data passes, the others
# one.  All runs start from the same image, one OSEM epoch from a uniform
# image.

num_epochs = iterations_for_device(dev, cpu=10, gpu=30, what="SVRG epochs")


def osem_epoch(x: Array, sc: dict) -> Array:
    for op, sl, sk in zip(subset_ops, subset_slices, subset_sens):
        ratio = sc["y"][sl] / (op(x) + sc["s"][sl])
        x = xp.where(fov_mask, x * op.adjoint(ratio) / xp.where(fov_mask, sk, 1.0), 0.0)
    return x


def svrg(
    x: Array, kind: str, prior, full, subsets, epochs: int = num_epochs
) -> tuple[Array, list, list]:
    """returns the final image, F per epoch and the number of data passes"""
    m = len(subsets)
    cost, passes = [float(full(x))], [0.0]
    for epoch in range(epochs):
        n = 0.0
        if epoch % 2 == 0:
            if epoch <= 4:
                precond = preconditioner(kind, prior, x)
            anchor = [f.gradient(x) for f in subsets]
            full_grad = sum(anchor[1:], anchor[0])
            x = xp.clip(x - precond * full_grad, 0, None)
            n += 1.0
        for k in range(m):
            g = m * (subsets[k].gradient(x) - anchor[k]) + full_grad
            x = xp.clip(x - precond * g, 0, None)
        cost.append(float(full(x)))
        passes.append(passes[-1] + n + 1.0)
    return x, cost, passes


# %%
# 7. Reference: monotone MAP-EM with the separable majorizer
# ----------------------------------------------------------
#
# De Pierro's MAP-EM combines the EM surrogate of the data fidelity with a
# separable surrogate of the prior.  With the separable majorizer of the
# logcosh prior around the current image :math:`\bar{x}`,
#
# .. math::
#     \beta R(x) \leq \beta R(\bar{x}) + g^T (x - \bar{x})
#     + \tfrac{1}{2} \sum_j d_j (x_j - \bar{x}_j)^2, \qquad
#     g = \beta \nabla R(\bar{x}),\; d = \beta\, c_R(\bar{x}),
#
# the surrogate of :math:`F` separates into one-dimensional problems.  Setting
# their derivative to zero gives, per voxel, the positive root of
#
# .. math::
#     d_j x_j^2 + (s_j + g_j - d_j \bar{x}_j)\, x_j - s_j x^{EM}_j = 0,
#
# where :math:`s = A^T 1` and :math:`x^{EM}` is the MLEM update of
# :math:`\bar{x}`.  Each iteration costs one data pass and never increases
# :math:`F` (a majorize-minimize algorithm).  For a quadratic prior this is
# the algorithm of :ref:`sphx_glr_auto_examples_03_algorithms_04_run_depierro.py`.

num_iter_mm = iterations_for_device(dev, cpu=60, gpu=300, what="MAP-EM iterations")


def map_em(x: Array, sc: dict, prior, full) -> tuple[Array, list]:
    cost = [float(full(x))]
    s_safe = xp.where(fov_mask, sens, 1.0)
    for _ in range(num_iter_mm):
        x_em = x * A.adjoint(sc["y"] / (A(x) + sc["s"])) / s_safe
        g = prior.gradient(x)
        d = prior.separable_majorizer_curvature(x)
        b = sens + g - d * x
        x = xp.where(
            fov_mask,
            2 * sens * x_em / (b + xp.sqrt(b**2 + 4 * d * sens * x_em)),
            0.0,
        )
        cost.append(float(full(x)))
    return x, cost


# %%
# 8. Running all scenarios
# ------------------------
#
# For every scenario we run SVRG with the three preconditioners and MAP-EM
# from the same initial image, and a reference SVRG run with three times the
# number of epochs (separable majorizer) to estimate the minimum
# :math:`F^*`.

results = {}
for name, sc in scenarios.items():
    prior, full, subsets = objectives(sc)
    x0 = osem_epoch(fov * xp.ones(img_shape, dtype=xp.float32, device=dev), sc)
    res = {kind: svrg(x0, kind, prior, full, subsets) for kind in precond_kinds}
    res["MAP-EM"] = map_em(x0, sc, prior, full)
    reference_cost = svrg(
        x0, "separable majorizer", prior, full, subsets, epochs=3 * num_epochs
    )[1]
    res["F*"] = min(reference_cost + [min(res[k][1]) for k in res])
    results[name] = res
    print(f"{name}: done")

# %%
# The second-order root form ``2 c / (b + sqrt(b^2 + 4 d c))`` of the MAP-EM
# update avoids cancellation when :math:`d \to 0`; MAP-EM is indeed monotone:

for name, res in results.items():
    mm_cost = np.asarray(res["MAP-EM"][1])
    print(f"{name}: MAP-EM monotone: {bool(np.all(np.diff(mm_cost) <= 1e-6 * abs(mm_cost[0])))}")

# %%
# 9. Convergence
# --------------
#
# We plot the relative objective gap :math:`(F - F^*) / (F(x^0) - F^*)`
# against the number of data passes.

fig, axs = plt.subplots(
    2, 4, figsize=(10, 5.6), layout="constrained", sharex=True, sharey=True
)
for ax, (name, res) in zip(axs.ravel(), results.items()):
    f_star = res["F*"]
    f0 = res["MAP-EM"][1][0]
    for kind in precond_kinds:
        _, cost, passes = res[kind]
        gap = (np.asarray(cost) - f_star) / (f0 - f_star)
        ax.semilogy(passes, np.maximum(gap, 1e-8), "o-", ms=3, label=f"SVRG, {kind}")
    mm_gap = (np.asarray(res["MAP-EM"][1]) - f_star) / (f0 - f_star)
    ax.semilogy(np.maximum(mm_gap, 1e-8), "k--", label="MAP-EM (majorizer)")
    ax.set_title(name.replace(", ", ",\n", 1), fontsize="small")
    ax.set_ylim(1e-7, 30)
    ax.grid(alpha=0.3)
for ax in axs[1]:
    ax.set_xlabel("data passes")
for ax in axs[:, 0]:
    ax.set_ylabel(r"$(F - F^*) / (F(x^0) - F^*)$")
axs[0, 0].legend(fontsize="x-small")
fig.show()

# %%
# - **Weak and medium prior:** the curvature of the prior is small compared
#   with :math:`A^T 1`, and all three preconditioners behave almost
#   identically; the larger curvatures (2 x Hessian diagonal, majorizer) are
#   marginally more cautious.
# - **Strong prior:** SVRG with the Hessian diagonal stalls.  The local
#   curvature :math:`\psi''(t)` of logcosh decays exponentially for
#   differences :math:`|t| \gg \delta` (edges, noise), while the prior still
#   changes like :math:`\psi'(t) / t \sim 1 / |t|` there, so the steps of
#   these voxels are too large.  Twice the Hessian diagonal compensates for
#   this here, and SVRG converges about as fast as with the majorizer.
# - **Strong prior, small** :math:`\delta`: now most differences lie in the
#   range :math:`|t| \gg \delta`, where :math:`\psi'(t)/t` exceeds
#   :math:`\psi''(t)` by orders of magnitude, and the factor 2 is not
#   enough: SVRG with twice the Hessian diagonal does not improve on the
#   initial image (gap of about 1 or larger), with the Hessian diagonal it
#   gets worse.  Only the separable majorizer, an upper bound of the prior
#   by construction, converges; it is slower than for the larger
#   :math:`\delta`, as the prior is closer to non-smooth.
# - The count level hardly matters, because the prior was scaled with the
#   data.  MAP-EM is monotone in all scenarios, but needs many more data
#   passes than SVRG.

# %%
# 10. Reconstructed images
# ------------------------
#
# The images after SVRG with the separable majorizer preconditioner, with the
# true image of each count level for comparison (each row on its own grey
# scale).

fig, axs = plt.subplots(2, 5, figsize=(10, 4.4), layout="constrained")
for row, counts_name in enumerate(total_counts):
    x_true = scenarios[f"{counts_name}, weak prior"]["x_true"]
    vmax = 1.3 * float(xp.max(x_true))
    panels = [("true image", x_true)] + [
        (
            prior_name,
            results[f"{counts_name}, {prior_name}"]["separable majorizer"][0],
        )
        for prior_name in priors
    ]
    for ax, (title, img) in zip(axs[row], panels):
        im = ax.imshow(to_numpy_array(img)[:, :, 0].T, cmap="Greys", vmin=0, vmax=vmax)
        ax.set_title(f"{counts_name}:\n{title}", fontsize="small")
        ax.set_axis_off()
    fig.colorbar(im, ax=axs[row], shrink=0.8)
fig.show()

# %%
# The weak prior leaves the low count image noisy, the medium prior
# suppresses most of the noise and keeps the hot insert, and the strong prior
# over-smooths.  With the small :math:`\delta`, the strong prior keeps
# sharper edges (piecewise constant regions, as with total variation).  A preconditioner has to work for all of them, since the
# regularisation strength is usually tuned after the algorithm is chosen.

# %%
# Summary
# -------
#
# - The three preconditioners only differ in the curvature of the prior that
#   they add to the EM preconditioner :math:`x / A^T 1`.  (Ignoring the prior
#   altogether, i.e. the EM preconditioner itself, is not shown here: in the
#   same scenarios it did not converge already for the medium prior.)
# - The **separable majorizer** (:meth:`.C2Function.separable_majorizer_curvature`)
#   is an upper bound of the prior everywhere; with it, De Pierro's MAP-EM is
#   a monotone algorithm for the logcosh prior and a reliable reference.
# - The **Hessian diagonal** (:meth:`.C2Function.hessian_diag`) is the local
#   curvature: fine for weak and medium priors, but SVRG with step size 1
#   does not converge for a strong prior.
# - **Twice the Hessian diagonal** works for the strong prior with the larger
#   :math:`\delta`, but not for a more edge-preserving prior (small
#   :math:`\delta`): it is a heuristic without a guarantee.
# - With a fixed step size, use the separable majorizer when the
#   regularisation can be strong or strongly edge-preserving.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_04_listmode_algorithms_00_run_listmode_mlem_osem_svrg.py`
#    -- the same algorithms with listmode data.

# sphinx_gallery_thumbnail_number = 1
