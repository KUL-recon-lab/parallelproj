"""
Accelerating MLTR with ordered subsets (OS-MLTR) and SVRG
=========================================================

:bdg-primary:`transmission` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

Full-data MLTR needs hundreds of iterations, each with a forward and back
projection of all data, to reach the maximum-likelihood attenuation image.
This example accelerates it with subset-based algorithms that update the
image after each subset of views: **OS-MLTR** (the transmission analogue of
OSEM: fast, but not convergent) and **SVRG** (stochastic variance-reduced
gradient: fast and convergent in practice).  Both are compared with
full-data MLTR and a converged **L-BFGS-B** reference.

**You will learn**

- how to split transmission data and the projector into view subsets,
- how to implement OS-MLTR, in which the :math:`1/m` factors of the subset
  gradient and curvature cancel exactly as in OSEM,
- how to combine SVRG with the MLTR diagonal preconditioner,
- how to compare the convergence of subset and full-data algorithms per
  epoch.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_05_transmission_00_mltr_sps.py`
(transmission model and MLTR) and
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(OSEM and SVRG for emission data).

.. note::

    Because of its run time (80 epochs each for MLTR, OS-MLTR and SVRG,
    and up to 500 L-BFGS-B iterations), this example is **not executed** in the
    documentation build: the figures are not rendered on this page.
    Download it and run it locally to see the results.
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

import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    poisson_transmission_terms,
    suggest_array_backend_and_device,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Simulation and reconstruction settings
# -----------------------------------------
#
# MLTR, OS-MLTR and SVRG run ``num_epochs`` epochs.  An **epoch** is one full
# MLTR iteration or one cycle through all ``num_subsets`` subsets.
# L-BFGS-B runs ``num_lbfgs`` iterations to provide the reference solution.
# The data are simulated as in the previous example.

num_epochs = 80  # epochs for MLTR and the subset algorithms
num_subsets = 28  # number of ordered view subsets (divides the 168 views evenly)
num_lbfgs = 500  # max. L-BFGS-B iterations for the reference solution
blank_counts = 500.0  # blank scan counts per LOR
scatter_fraction = 0.5  # scatter relative to mean unscattered transmission

# %%
# 2. Scanner, projector and ground-truth attenuation image
# --------------------------------------------------------
#
# The same non-TOF scanner (radius 300 mm, 3 rings), image grid
# (100 x 100 x 3 voxels) and water-equivalent elliptic cylinder phantom as
# in the previous example.

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

mu_water = 0.0096  # 1/mm at 511 keV
mu_true = mu_water * elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# voxels never seen by any LOR must not be updated (their denominator is 0)
fov_mask = proj.fov_mask()

# %%
# 3. Simulated transmission data
# ------------------------------
#
# The model is the exact transmission Poisson likelihood with a strictly
# positive, smooth scatter background :math:`s` of known mean,
#
# .. math::
#     L(\mu) = \sum_i y_i \ln \bar{y}_i (\mu) - \bar{y}_i (\mu),
#     \qquad
#     \bar{y}_i (\mu) = \bar{z}_i (\mu) + s_i,
#     \qquad
#     \bar{z}_i (\mu) = b_i e^{-(P \mu)_i},
#
# with the blank scan :math:`b`.  The measured counts :math:`y` are a
# Poisson realisation of :math:`\bar{y}(\mu_{true})`.

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
# 4. Full-data and subset ingredients
# -----------------------------------
#
# The sinogram is split into :math:`m` view subsets :math:`S_k`.  We build
# one subset projector :math:`P_k` per view subset (a copy of the full
# projector that projects only these views) and slice the blank, scatter and
# data sinograms accordingly.  ``P1`` and the subset intersection lengths
# ``Pk1`` (forward projections of an all-ones image) are precomputed once.
#
# ``poisson_transmission_terms`` returns the per-bin terms of the gradient
# :math:`\tfrac{\bar z}{\bar y}(\bar y - y)` and of the MLTR curvature
# :math:`\bar z^2/\bar y` (see the previous example).  From them we build
# the full-data gradient :math:`\nabla_\mu L`, the MLTR curvature denominator
# :math:`P^T[(P\mathbf 1)\,\bar z^2/\bar y]` and the subset gradients
# :math:`\nabla_\mu L_k`.

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


def neg_logL(mu: Array) -> float:
    """Negative transmission Poisson log-likelihood (float64 accumulation)."""
    ybar = b * xp.exp(-proj(mu)) + s
    return float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64)))


def _safe_ratio(num: Array, denom: Array) -> Array:
    """``num / denom`` where the (curvature) denominator is positive, 0 else.

    With many subsets a single subset may not see every FOV voxel, so its
    curvature denominator can be exactly 0 there (and the matching numerator
    too).  Such voxels simply receive no update from that subset; guarding
    ``denom > 0`` avoids the resulting 0/0.
    """
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
# 5. MLTR (full-data baseline)
# ----------------------------
#
# MLTR is preconditioned gradient ascent
# :math:`\mu \leftarrow [\mu + D(\mu)\odot\nabla_\mu L]_+` with the
# Newton-type diagonal preconditioner
# :math:`D_j = 1/P^T[(P\mathbf 1)\,\bar z^2/\bar y]_j`.  Each iteration
# (= epoch) is one full data pass.  All algorithms start from
# :math:`\mu = 0`.

mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
cost: dict[str, np.ndarray] = {}
mu_final: dict[str, Array] = {}

cost_epoch = [neg_logL(mu)]
for ep in range(num_epochs):
    print(f"MLTR       epoch {ep + 1:03}/{num_epochs:03}", end="\r")
    grad, denom = full_grad_and_curv(mu)
    mu = xp.clip(mu + _safe_ratio(grad, denom), 0, None)
    cost_epoch.append(neg_logL(mu))
print()
cost["MLTR"] = np.asarray(cost_epoch)
mu_final["MLTR"] = mu

# %%
# 6. OS-MLTR (one subset per update)
# ----------------------------------
#
# OS-MLTR evaluates the MLTR update on one subset at a time, using
# subset-local quantities throughout.  Exactly as in OSEM, the :math:`1/m`
# factors of the subset gradient and the subset curvature cancel in the
# ratio, so each subset update has roughly the full-data step size:
#
# .. math::
#     \mu \leftarrow \Bigl[\mu +
#       \frac{P_k^T\bigl[\tfrac{\bar z_k}{\bar y_k}(\bar y_k - y_k)\bigr]}
#            {P_k^T\bigl[(P_k\mathbf 1)\,\tfrac{\bar z_k^2}{\bar y_k}\bigr]}
#       \Bigr]_+ .
#
# One epoch (:math:`m` subset updates in a fixed order) is about one full
# data pass.  Like OSEM, OS-MLTR is fast per epoch but **not** convergent: it
# approaches a subset-dependent limit cycle near the solution.

mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
cost_epoch = [neg_logL(mu)]
for ep in range(num_epochs):
    print(f"OS-MLTR    epoch {ep + 1:03}/{num_epochs:03}", end="\r")
    for k in range(num_subsets):
        _, grad_sino, curv_sino = poisson_transmission_terms(
            subset_proj[k](mu), b_k[k], s_k[k], y_k[k]
        )
        num = subset_proj[k].adjoint(grad_sino)
        denom = subset_proj[k].adjoint(Pk1[k] * curv_sino)
        mu = xp.clip(mu + _safe_ratio(num, denom), 0, None)
    cost_epoch.append(neg_logL(mu))
print()
cost["OS-MLTR"] = np.asarray(cost_epoch)
mu_final["OS-MLTR"] = mu

# %%
# 7. SVRG (preconditioned by the MLTR diagonal at the anchor)
# -----------------------------------------------------------
#
# Every other epoch (epochs 1, 3, 5, ...), SVRG sets an anchor
# :math:`\tilde\mu` to the current image and computes the full gradient
# :math:`g = \nabla L(\tilde\mu)`, the subset gradients
# :math:`\nabla L_k(\tilde\mu)` and the full MLTR diagonal preconditioner
# at the anchor.  Each inner subset step (subsets in random order) uses the
# variance-reduced gradient estimate
#
# .. math::
#     g_{\mathrm{vr}} = g + m\bigl(\nabla L_k(\mu) - \nabla L_k(\tilde\mu)\bigr),
#
# followed by a preconditioned ascent step and clipping to
# :math:`\mu \geq 0`.  In contrast to OS-MLTR, the variance-reduced gradient
# vanishes at the solution, so SVRG does not get stuck in a limit cycle.
# (The variant used here, with a step size of 1, an anchor-dependent
# preconditioner and clipping, is not covered by the standard SVRG
# convergence proofs.)
#
# .. note::
#     An anchor epoch costs about three data passes (the :math:`m` subset
#     gradients at the anchor, the full gradient and MLTR curvature for the
#     preconditioner, and the :math:`m` subset steps), the other epochs one,
#     i.e. about two data passes per SVRG epoch on average.  The epoch axis
#     of the convergence plot below therefore understates the computational
#     cost of SVRG by about a factor of two compared to MLTR / OS-MLTR.

rng = np.random.default_rng(1)
mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
cost_epoch = [neg_logL(mu)]
for ep in range(num_epochs):
    print(f"SVRG       epoch {ep + 1:03}/{num_epochs:03}", end="\r")

    if ep % 2 == 0:
        anchor = mu
        g_full, denom_full = full_grad_and_curv(anchor)
        gk_anchor = [subset_grad(anchor, k) for k in range(num_subsets)]
        precond = _safe_ratio(xp.ones_like(mu), denom_full)  # MLTR diagonal

    for k in rng.permutation(num_subsets):
        g_vr = g_full + num_subsets * (subset_grad(mu, k) - gk_anchor[k])
        mu = xp.clip(mu + precond * g_vr, 0, None)
    cost_epoch.append(neg_logL(mu))
print()
cost["SVRG"] = np.asarray(cost_epoch)
mu_final["SVRG"] = mu

# %%
# 8. L-BFGS-B reference solution (no subsets)
# -------------------------------------------
#
# As in the previous example, SciPy's L-BFGS-B minimises :math:`-L(\mu)`
# with the bound :math:`\mu \geq 0` on the full data.  It provides the
# converged maximum-likelihood reference solution :math:`\hat\mu` and the
# reference value of :math:`-L`.  ``num_lbfgs`` is only an upper limit:
# L-BFGS-B stops at its default convergence tolerance, here after about 100
# function evaluations.

n_vox = int(np.prod(proj.in_shape))
cost_lbfgs: list[float] = []  # -L recorded at every function evaluation


def neg_logL_and_grad(mu_flat: np.ndarray) -> tuple[float, np.ndarray]:
    m = xp.asarray(mu_flat.reshape(proj.in_shape), dtype=xp.float32, device=dev)
    psi = b * xp.exp(-proj(m))
    ybar = psi + s
    val = float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64)))
    grad = proj.adjoint(psi / ybar * (y - ybar))  # gradient of -L
    cost_lbfgs.append(val)
    return val, np.asarray(to_numpy_array(grad)).ravel().astype(np.float64)


res = minimize(
    neg_logL_and_grad,
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
L_ref = float(res.fun)  # converged reference -L
print(f"L-BFGS-B reference: -L = {L_ref:.2f}")

for name in ("MLTR", "OS-MLTR", "SVRG"):
    print(f"{name:8}: -L after {num_epochs} epochs = {cost[name][-1]:.2f}")

# %%
# 9. Convergence and line profiles
# --------------------------------
#
# The left panel shows the cost :math:`-L(\mu)` per epoch (per function
# evaluation for L-BFGS-B), zoomed to the range below the MLTR value after
# ``num_epochs // 2`` epochs.  The right panel compares profiles along
# :math:`x_0` through the centre of the final images with the ground truth.

algo_names = ("MLTR", "OS-MLTR", "SVRG", "L-BFGS-B")
cost_min = float(min(c.min() for c in cost.values()))
cost_max = float(cost["MLTR"][num_epochs // 2])

sl = img_shape[2] // 2
x0_mm = (np.arange(img_shape[0]) - (img_shape[0] - 1) / 2) * voxel_size[0]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
for name in algo_names:
    ax[0].plot(cost[name], label=name)
ax[0].set_ylim(cost_min, cost_max)
ax[0].set_xlabel("epoch (MLTR, subset methods) / function evaluation (L-BFGS-B)")
ax[0].set_ylabel(r"$-L(\mu)$")
ax[0].set_title("Convergence")
ax[0].grid(ls=":")
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=4)

ax[1].plot(
    x0_mm,
    to_numpy_array(mu_true[:, img_shape[1] // 2, sl]),
    "k--",
    label=r"true $\mu$",
)
for name in algo_names:
    ax[1].plot(
        x0_mm, to_numpy_array(mu_final[name][:, img_shape[1] // 2, sl]), label=name
    )
ax[1].set_xlabel(r"$x_0$ [mm]")
ax[1].set_ylabel(r"$\mu$ [1/mm]")
ax[1].set_title(r"Central profile along $x_0$")
ax[1].grid(ls=":")
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=3)
fig.show()

# %%
# OS-MLTR and SVRG reach in a few epochs what full MLTR needs many more
# epochs for -- roughly a ``num_subsets``-fold per-epoch speed-up.  With many
# subsets, however, OS-MLTR has no convergence guarantee: it approaches a
# subset-dependent limit cycle and its cost stalls (or rises) above the
# optimum, whereas the variance-reduced SVRG remains stable and keeps
# decreasing towards the L-BFGS-B reference.  The profiles of the
# subset methods and of L-BFGS-B are similar and show the noise of the
# unregularised maximum-likelihood solution; full-data MLTR, still further
# from convergence, is smoother.

# %%
# 10. Reconstructed images
# ------------------------
#
# The central transaxial slice of the true :math:`\mu` image and of the four
# reconstructions, with the same colour scale.

images = {r"true $\mu$": mu_true} | mu_final
extent = (
    -img_shape[0] * voxel_size[0] / 2,
    img_shape[0] * voxel_size[0] / 2,
    img_shape[1] * voxel_size[1] / 2,
    -img_shape[1] * voxel_size[1] / 2,
)

fig2, ax2 = plt.subplots(
    1, len(images), figsize=(10, 2.8), sharey=True, layout="compressed"
)
for axi, (name, img) in zip(ax2, images.items()):
    im = axi.imshow(
        to_numpy_array(img[:, :, sl]).T,
        cmap="Greys",
        vmin=0,
        vmax=3.4 * mu_water,
        extent=extent,
    )
    axi.set_title(name)
    axi.set_xlabel(r"$x_0$ [mm]")
ax2[0].set_ylabel(r"$x_1$ [mm]")
fig2.colorbar(im, ax=ax2, shrink=0.8, label=r"$\mu$ [1/mm]")
fig2.show()

# %%
# The OS-MLTR, SVRG and L-BFGS-B images are visually almost identical and
# contain the noise of the unregularised maximum-likelihood solution; the
# MLTR image after the same number of epochs is less converged.  The next
# example adds a prior to suppress the noise.

# %%
# Summary
# -------
#
# - Splitting the views into :math:`m` subsets accelerates MLTR by roughly a
#   factor :math:`m` per epoch; in OS-MLTR the :math:`1/m` factors of the
#   subset gradient and curvature cancel, as in OSEM.
# - OS-MLTR does not converge but approaches a subset-dependent limit cycle.
# - SVRG with the MLTR diagonal preconditioner at the anchor is as fast per
#   epoch and keeps approaching the maximum-likelihood solution, at about
#   two data passes per epoch.
# - L-BFGS-B on the full data provides a converged reference solution.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py`
#    -- penalised (MAPTR) transmission reconstruction with an edge-preserving
#    prior.

# sphinx_gallery_thumbnail_path = "_static/thumbnails/01_os_mltr_svrg.png"
