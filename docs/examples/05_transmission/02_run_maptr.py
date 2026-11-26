"""
Penalised transmission reconstruction (MAPTR) with an edge-preserving prior
===========================================================================

:bdg-primary:`transmission` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

Maximum-likelihood transmission reconstruction amplifies the noise of the
data as the iterations proceed.  This example adds an edge-preserving
log-cosh prior to the ordered-subset transmission reconstruction of the
previous example (MAPTR -- maximum a posteriori transmission
reconstruction) and minimises the penalised objective with **OS-MLTR** and
**SVRG**, using a converged **L-BFGS-B** solution as reference.  The key
ingredient is a diagonal preconditioner that adds the curvature of a
separable majorizer of the prior to the MLTR curvature of the data.

**You will learn**

- how to add a log-cosh prior (:class:`.FiniteForwardDifference` and
  :class:`.LogCosh`) to the transmission Poisson log-likelihood,
- how to build a "harmonic-mean" preconditioner from the MLTR curvature and
  the separable majorizer curvature of the prior
  (:meth:`.C2Function.separable_majorizer_curvature`),
- how to split the prior over the subsets of OS-MLTR, and how to combine a
  variance-reduced data gradient with the full prior gradient in SVRG,
- how to compute a reference solution with L-BFGS-B from SciPy.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_05_transmission_01_os_mltr_svrg.py`
(OS-MLTR and SVRG for transmission data) and
:ref:`sphx_glr_auto_examples_05_transmission_00_mltr_sps.py` (the MLTR
update).

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

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    iterations_for_device,
    poisson_transmission_terms,
    suggest_array_backend_and_device,
)
from parallelproj.functions import C2AffineObjective, LogCosh

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Reconstruction settings
# --------------------------
#
# An **epoch** of the subset algorithms is one cycle through all
# ``num_subsets`` view subsets.  ``beta`` is the weight of the prior and
# ``delta`` its transition scale (see section 4): with
# :math:`\delta = \mu_{\text{water}}/2`, the jumps at the inserts (0.5 to
# 2 :math:`\mu_{\text{water}}` in the central slice, i.e. 1 to 4
# :math:`\delta`) lie in or close to the edge-preserving, linear regime of
# the prior, while the smaller background noise is smoothed.

# epochs of the subset algorithms (fewer on the CPU)
num_epochs = iterations_for_device(dev, cpu=10, gpu=30, what="epochs")
num_subsets = 28  # number of ordered view subsets (divides the 168 views evenly)
num_lbfgs = 80  # L-BFGS-B iterations for the reference solution
blank_counts = 500.0  # blank scan counts per LOR
scatter_fraction = 0.5  # scatter relative to mean unscattered transmission

mu_water = 0.0096  # 1/mm at 511 keV
beta = 2e2  # prior weight
# log-cosh transition scale: edges (inserts, up to 4 delta) are preserved,
# background noise < delta is smoothed
delta = mu_water / 2

# %%
# 2. Scanner, projector and attenuation image
# -------------------------------------------
#
# A large scanner (radius 300 mm, 56 sides, 3 rings) with a non-TOF span-1
# sinogram, and an image of 100 x 100 x 3 voxels of 4 x 4 x 5.3 mm (one
# slice per ring).  The ground truth attenuation image is the elliptic
# cylinder phantom with dense and less dense inserts, scaled to the
# attenuation coefficient of water at 511 keV.

num_rings = 3
ring_spacing = 5.3  # mm, axial distance between rings
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=300.0,
    num_sides=56,
    num_lor_endpoints_per_side=6,
    lor_spacing=5.3,
    # rings centred on the origin, spacing = ring_spacing -> [-5.3, 0, 5.3]
    ring_positions=(
        xp.arange(num_rings, dtype=xp.float32, device=dev) - (num_rings - 1) / 2
    )
    * ring_spacing,
    symmetry_axis=2,
)

# transaxial 100 x 100 @ 4 mm; axially one slice per ring (5.3 mm),
# so the image slices are aligned with the ring positions
img_shape = (100, 100, num_rings)
voxel_size = (4.0, 4.0, ring_spacing)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

mu_true = mu_water * elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# voxels never seen by any LOR must not be updated (their denominator is 0)
fov_mask = proj.fov_mask()

# %%
# 3. Simulated transmission data
# ------------------------------
#
# The expected transmission data are :math:`\bar{y} = b\, e^{-P\mu} + s` with
# a blank scan of :math:`b` = 500 counts per LOR and a constant scatter
# background :math:`s` of half the mean unscattered transmission.  The
# measured data :math:`y` are a Poisson realisation.

b = xp.full(proj.out_shape, blank_counts, device=dev, dtype=xp.float32)

psi_true = b * xp.exp(-proj(mu_true))
s = xp.full(
    proj.out_shape,
    scatter_fraction * float(xp.mean(psi_true)),
    device=dev,
    dtype=xp.float32,
)

np.random.seed(1)
y = xp.asarray(
    np.random.poisson(to_numpy_array(psi_true + s)),
    device=dev,
    dtype=xp.float32,
)

# %%
# 4. The penalised objective
# --------------------------
#
# We *minimise* the penalised objective
#
# .. math::
#     \Phi(\mu) = -L(\mu) + \beta R(\mu)
#
# .. math::
#     L(\mu) = \sum_i y_i \ln \bar{y}_i (\mu) - \bar{y}_i (\mu),
#     \qquad
#     \bar{y}_i (\mu) = \bar{z}_i (\mu) + s_i,
#     \qquad
#     \bar{z}_i (\mu) = b_i e^{-(P \mu)_i},
#
# with a **log-cosh** roughness penalty on the nearest-neighbour finite
# differences :math:`G\mu`,
#
# .. math::
#     R(\mu) = \delta \sum_d \sum_j
#     \log\cosh\!\left(\frac{(G\mu)_{d,j}}{\delta}\right),
#     \qquad
#     \nabla R = G^T \tanh(G\mu/\delta).
#
# The log-cosh penalty is quadratic for differences :math:`\ll \delta`
# (smooths noise) and linear for differences :math:`\gg \delta` (preserves
# edges).  ``reg(mu)`` and ``reg.gradient(mu)`` give :math:`\beta R` and
# :math:`\beta \nabla R`.  The negative log-likelihood is accumulated in
# float64.

G = parallelproj.operators.FiniteForwardDifference(proj.in_shape)
reg = C2AffineObjective(LogCosh(delta=delta, beta=beta), G)


def neg_logL(mu: Array) -> float:
    """Negative transmission Poisson log-likelihood (float64 accumulation)."""
    ybar = b * xp.exp(-proj(mu)) + s
    return float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64)))


def penalised_cost(mu: Array) -> float:
    """Penalised objective Phi = -L + beta R (to be minimised)."""
    return neg_logL(mu) + float(reg(mu))


# %%
# 5. Subsets and gradients
# ------------------------
#
# As in the previous example, the views are distributed over
# ``num_subsets`` subsets, each with its own copy of the projector and its
# part of the blank scan, scatter and data.  ``P1`` and ``Pk1`` are the
# forward projections :math:`P\mathbf 1` of a uniform image, needed by the
# MLTR curvature.  :func:`.poisson_transmission_terms` returns the expected
# data and the gradient and curvature sinograms of :math:`L`.

ones_img = xp.ones(proj.in_shape, dtype=xp.float32, device=dev)
P1 = proj(ones_img)

subset_views, subset_slices = lor_desc.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

subset_proj = []
for k in range(num_subsets):
    p = copy(proj)
    p.views = subset_views[k]
    subset_proj.append(p)

b_k = [b[subset_slices[k]] for k in range(num_subsets)]
s_k = [s[subset_slices[k]] for k in range(num_subsets)]
y_k = [y[subset_slices[k]] for k in range(num_subsets)]
Pk1 = [subset_proj[k](ones_img) for k in range(num_subsets)]


def _safe_ratio(num: Array, denom: Array) -> Array:
    """``num / denom`` where the denominator is positive, 0 else (FOV-safe)."""
    ok = fov_mask & (denom > 0)
    denom_safe = xp.where(ok, denom, xp.ones_like(denom))
    return xp.where(ok, num / denom_safe, xp.zeros_like(num))


def full_grad_and_curv(mu: Array) -> tuple[Array, Array]:
    """Full-data gradient of L and the MLTR curvature denominator."""
    _, grad_sino, curv_sino = poisson_transmission_terms(proj(mu), b, s, y)
    return proj.adjoint(grad_sino), proj.adjoint(P1 * curv_sino)


def subset_grad(mu: Array, k: int) -> Array:
    """Gradient of the subset log-likelihood L_k (no 1/m scaling)."""
    _, grad_sino, _ = poisson_transmission_terms(
        subset_proj[k](mu), b_k[k], s_k[k], y_k[k]
    )
    return subset_proj[k].adjoint(grad_sino)


# %%
# 6. The preconditioner
# ---------------------
#
# **The transmission "harmonic-mean" analogue.**  In emission MAP-EM one
# combines the EM step :math:`x/A^T\mathbf 1` with the prior curvature; that
# is the harmonic mean of the two step sizes, i.e. the reciprocal of the
# *sum of curvatures*.  Here the MLTR denominator
# :math:`P^T[(P\mathbf 1)\,\bar z^2/\bar y]` is exactly the separable
# diagonal majorant of the data Hessian (the analogue of
# :math:`A^T\mathbf 1 / x`), so the penalised preconditioner is
#
# .. math::
#     D(\mu) = \frac{1}{\;P^T\!\left[(P\mathbf 1)\,\bar z^2/\bar y\right]
#                        + \beta\,c_R(\mu)\;},
#
# where :math:`\beta\,c_R(\mu)` (``reg.separable_majorizer_curvature(mu)``)
# is the curvature of the separable quadratic majorizer of the log-cosh prior:
# De Pierro's surrogate :math:`c_R = |G|^T(\omega(G\mu) \odot |G|\mathbf 1)`
# with Huber's curvature :math:`\omega(t) = \psi'(t)/t` of
# :math:`\psi(t) = \delta \log\cosh(t/\delta)`, see
# :meth:`.C2Function.separable_majorizer_curvature`.  Like the MLTR term for
# the data, it majorizes the prior, and it adapts to the image: it is smaller
# for large differences (edges) than the global bound
# :math:`2\operatorname{diag}(G^TG)/\delta`.  (The maximal Hessian diagonal
# :math:`\operatorname{diag}(G^TG)/\delta` alone is *not* a separable
# majorizer: it lacks the factor :math:`|G|\mathbf 1 = 2` of every
# difference.)
#
# The preconditioner is computed inside the update loops below: per subset
# for OS-MLTR, once per anchor for SVRG.

# %%
# 7. OS-MLTR with the prior
# -------------------------
#
# OS-MLTR ascends :math:`L - \beta R` one subset at a time with the
# preconditioner ``D = 1 / (data curvature + prior curvature)``, both
# evaluated for the current subset.  The prior is split ``beta/m`` per
# subset (gradient and curvature), so the :math:`m` subset contributions sum
# to the full penalty.  All algorithms start from :math:`\mu = 0`.

cost: dict[str, np.ndarray] = {}
mu_final: dict[str, Array] = {}

mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
c = [penalised_cost(mu)]
for ep in range(num_epochs):
    print(f"OS-MLTR    epoch {ep + 1:03}/{num_epochs:03}", end="\r")
    for k in range(num_subsets):
        _, grad_sino, curv_sino = poisson_transmission_terms(
            subset_proj[k](mu), b_k[k], s_k[k], y_k[k]
        )
        num = subset_proj[k].adjoint(grad_sino)
        denom = subset_proj[k].adjoint(Pk1[k] * curv_sino)
        g_pen = num - reg.gradient(mu) / num_subsets
        prior_curv = reg.separable_majorizer_curvature(mu) / num_subsets
        mu = xp.clip(mu + _safe_ratio(g_pen, denom + prior_curv), 0, None)
    c.append(penalised_cost(mu))
print()
cost["OS-MLTR"] = np.asarray(c)
mu_final["OS-MLTR"] = mu

# %%
# 8. SVRG with the prior
# ----------------------
#
# Every other epoch, SVRG computes the full data gradient and the subset
# gradients at an anchor :math:`\tilde\mu`, and the preconditioner (full
# data curvature plus :math:`\beta c_R(\tilde\mu)`).  The subset steps, in a
# random order, use the variance-reduced data gradient; the cheap,
# deterministic prior gradient :math:`\beta\nabla R(\mu)` is added in full
# at every step.
#
# .. note::
#     Each OS-MLTR epoch is one full data pass; an SVRG epoch is roughly 1.5
#     (anchor + subset sweep), so the epoch axis understates SVRG's cost by
#     about a factor of 1.5.

rng = np.random.default_rng(1)
mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
c = [penalised_cost(mu)]
for ep in range(num_epochs):
    print(f"SVRG       epoch {ep + 1:03}/{num_epochs:03}", end="\r")

    if ep % 2 == 0:
        anchor = mu
        g_full, denom_full = full_grad_and_curv(anchor)
        gk_anchor = [subset_grad(anchor, k) for k in range(num_subsets)]
        precond = _safe_ratio(
            xp.ones_like(mu), denom_full + reg.separable_majorizer_curvature(anchor)
        )

    for k in rng.permutation(num_subsets):
        # variance-reduced data gradient + full (deterministic) prior gradient
        g_data = g_full + num_subsets * (subset_grad(mu, k) - gk_anchor[k])
        mu = xp.clip(mu + precond * (g_data - reg.gradient(mu)), 0, None)
    c.append(penalised_cost(mu))
print()
cost["SVRG"] = np.asarray(c)
mu_final["SVRG"] = mu

# %%
# 9. L-BFGS-B reference solution
# ------------------------------
#
# SciPy's L-BFGS-B minimises :math:`\Phi` on the full data (no subsets) with
# the bound :math:`\mu \geq 0`.  It works on flattened float64 NumPy
# arrays, so the function below converts between them and the array backend.
# :math:`\Phi` is recorded at every function evaluation.

n_vox = int(np.prod(proj.in_shape))
cost_lbfgs: list[float] = []  # Phi recorded at every function evaluation


def penalised_cost_and_grad(mu_flat: np.ndarray) -> tuple[float, np.ndarray]:
    m = xp.asarray(mu_flat.reshape(proj.in_shape), dtype=xp.float32, device=dev)
    ybar, grad_sino, _ = poisson_transmission_terms(proj(m), b, s, y)
    val = float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64))) + float(reg(m))
    # gradient of Phi = -L + beta R; grad_sino is the *ascent* gradient of L,
    # so the gradient of -L is its negative
    grad = -proj.adjoint(grad_sino) + reg.gradient(m)
    cost_lbfgs.append(val)
    return val, np.asarray(to_numpy_array(grad)).ravel().astype(np.float64)


res = minimize(
    penalised_cost_and_grad,
    np.zeros(n_vox),
    jac=True,
    method="L-BFGS-B",
    bounds=[(0.0, None)] * n_vox,
    options={"maxiter": num_lbfgs, "maxfun": num_lbfgs},
)
mu_final["L-BFGS-B"] = xp.asarray(
    res.x.reshape(proj.in_shape), dtype=xp.float32, device=dev
)
cost["L-BFGS-B"] = np.asarray(cost_lbfgs)
phi_ref = float(res.fun)  # converged reference penalised cost
print(f"L-BFGS-B reference: Phi = {phi_ref:.2f}")

for name in ("OS-MLTR", "SVRG"):
    print(f"{name:8}: Phi after {num_epochs} epochs = {cost[name][-1]:.2f}")

# %%
# 10. Convergence and profiles
# ----------------------------
#
# Left: the penalised objective :math:`\Phi` per epoch (subset algorithms)
# or per function evaluation (L-BFGS-B), zoomed in on the final values.
# Right: a profile along ``x0`` through the centre of the central slice.

c_min = min(float(curve.min()) for curve in cost.values())
c_max = float(cost["SVRG"][4])

fig, axs = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
for name in ("OS-MLTR", "SVRG", "L-BFGS-B"):
    axs[0].plot(cost[name], label=name)
axs[0].set_ylim(c_min, c_max)
axs[0].set_xlabel("epoch (subset methods) / function evaluation (L-BFGS-B)")
axs[0].set_ylabel(r"$\Phi(\mu) = -L(\mu) + \beta R(\mu)$")
axs[0].set_title("Convergence of the penalised objective")
axs[0].grid(ls=":")
axs[0].legend()

sl = img_shape[2] // 2
axs[1].plot(
    to_numpy_array(mu_true[:, img_shape[1] // 2, sl]), "k--", label=r"true $\mu$"
)
for name in ("OS-MLTR", "SVRG", "L-BFGS-B"):
    axs[1].plot(to_numpy_array(mu_final[name][:, img_shape[1] // 2, sl]), label=name)
axs[1].set_xlabel("x0 voxel (4 mm)")
axs[1].set_ylabel(r"$\mu$ (1/mm)")
axs[1].set_title(f"Profile along x0 (x1 = {img_shape[1] // 2}, slice {sl})")
axs[1].grid(ls=":")
axs[1].legend()
fig.show()

# %%
# All three algorithms minimise the same objective :math:`\Phi`.  OS-MLTR
# decreases it fastest in the first epochs, but then levels off: like OSEM,
# it does not converge to the minimiser (it approaches a subset-dependent
# limit cycle).  SVRG overtakes it after about five epochs (keep in mind
# that an SVRG epoch costs about 1.5 data passes) and ends closer to the
# L-BFGS-B value.  The differences are tiny compared with the total
# objective (note the offset of the y axis), and the three profiles are
# almost identical: they follow the jumps at the inserts (edges of up to
# 4 :math:`\delta`) and fluctuate only mildly in the uniform water
# background.

# %%
# 11. Reconstructed images
# ------------------------
#
# The central transaxial slice of the true attenuation image and of the
# three reconstructions, with the same grey scale (0 to 3.4
# :math:`\mu_{\text{water}}`).

titles = [r"true $\mu$"] + list(mu_final)
images = [mu_true] + list(mu_final.values())

fig, axs = plt.subplots(1, 4, figsize=(10, 3.0), layout="constrained")
for ax, title, img in zip(axs, titles, images):
    im = ax.imshow(
        to_numpy_array(img[:, :, sl]).T, cmap="Greys", vmin=0, vmax=3.4 * mu_water
    )
    ax.set_title(title)
    ax.set_xlabel("x0 voxel")
axs[0].set_ylabel("x1 voxel")
fig.colorbar(im, ax=axs, shrink=0.9, label=r"$\mu$ (1/mm)")
fig.show()

# %%
# The three reconstructions look almost the same.  The prior leaves only a
# fine-grained residual noise in the uniform background, while the large
# inserts keep sharp edges.  The smallest inserts are blurred and lose
# contrast.

# %%
# Summary
# -------
#
# - MAPTR minimises :math:`\Phi = -L + \beta R`: the exact transmission
#   Poisson likelihood plus an edge-preserving log-cosh prior that is
#   quadratic for small and linear for large differences.
# - The preconditioner adds the separable majorizer curvature of the prior,
#   :math:`\beta c_R(\mu)`, to the MLTR curvature of the data (the
#   transmission analogue of the harmonic-mean MAP-EM step); the maximal
#   Hessian diagonal :math:`\operatorname{diag}(G^TG)/\delta` would not be a
#   majorizer.
# - OS-MLTR splits the prior ``beta/m`` over the subsets; SVRG
#   variance-reduces only the data gradient and adds the full prior gradient
#   at every step.
# - L-BFGS-B on the full data gives a reference solution of the same
#   objective.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_05_transmission_03_mlaa.py`
#    -- joint estimation of activity and attenuation (MLAA) from TOF emission
#    data, with the same prior and preconditioner.

# sphinx_gallery_thumbnail_number = 2
