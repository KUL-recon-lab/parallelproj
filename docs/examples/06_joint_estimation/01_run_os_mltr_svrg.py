"""
Accelerating MLTR with ordered subsets (OS-MLTR) and SVRG
=========================================================

This example accelerates the transmission MLTR reconstruction of
``00_run_mltr_sps.py`` with subset-based algorithms.  The model is the
exact transmission Poisson likelihood with a strictly positive, smooth
scatter background :math:`s` of known mean,

.. math::
    L(\\mu) = \\sum_i y_i \\ln \\bar{y}_i - \\bar{y}_i,
    \\qquad
    \\bar{y}_i = b_i e^{-(P \\mu)_i} + s_i,

reconstructed by preconditioned gradient ascent
:math:`\\mu \\leftarrow [\\mu + D(\\mu)\\odot\\nabla_\\mu L]_+` with the MLTR
(Newton-type) diagonal preconditioner
:math:`D_j = 1/P^T[(P\\mathbf 1)\\,\\bar\\psi^2/\\bar y]_j`.

The sinogram is split into :math:`m` view subsets :math:`S_k` with
subset projectors :math:`P_k`.  Three accelerations are compared against
**MLTR** (the full-data baseline) and a converged **L-BFGS-B** reference:

* **OS-MLTR** -- the MLTR update evaluated on one subset at a time, using
  subset-local quantities throughout.  Exactly as in OSEM, the
  :math:`1/m` factors of the subset gradient and the subset curvature
  cancel in the ratio, so each subset update has roughly the full-data
  step size:

  .. math::
      \\mu \\leftarrow \\Bigl[\\mu +
        \\frac{P_k^T\\bigl[\\tfrac{\\bar\\psi_k}{\\bar y_k}(\\bar y_k - y_k)\\bigr]}
             {P_k^T\\bigl[(P_k\\mathbf 1)\\,\\tfrac{\\bar\\psi_k^2}{\\bar y_k}\\bigr]}
        \\Bigr]_+ .

  One epoch = :math:`m` subset updates ≈ one full data pass.  Like OSEM it
  is fast per epoch but **not** convergent (it approaches a subset-dependent
  limit cycle near the solution).

* **SVRG** -- stochastic variance-reduced gradient.  Once per epoch the
  full gradient :math:`g = \\nabla L(\\tilde\\mu)` is computed at an anchor
  :math:`\\tilde\\mu`; each inner subset step uses the variance-reduced
  estimate

  .. math::
      g_{\\mathrm{vr}} = g + m\\bigl(\\nabla L_k(\\mu) - \\nabla L_k(\\tilde\\mu)\\bigr),

  preconditioned by the (fixed-per-epoch) full MLTR diagonal evaluated at
  the anchor.  One epoch costs **two** data passes (anchor full gradient +
  :math:`m` subset steps).  Unlike OS-MLTR it is provably convergent; with
  a moderate number of subsets the two behave very similarly per epoch, and
  the distinction matters mainly with many subsets or when iterating far
  past the point shown here.

* **L-BFGS-B** (≈ 100 iterations, no subsets) provides the converged
  maximum-likelihood reference solution :math:`\\hat\\mu` against which the
  per-epoch suboptimality :math:`-L(\\mu) + L(\\hat\\mu)` is plotted.

.. note::
    Each MLTR / OS-MLTR epoch is one full data pass; an SVRG epoch is
    roughly two (anchor + subset sweep), so the epoch axis understates
    SVRG's cost by about a factor of two.

.. note::

   To run this example locally, download
   `example_utils.py <https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/example_utils.py>`_
   into the **same folder** as this script. Make sure ``parallelproj`` is installed.
"""

# %%
from __future__ import annotations

from copy import copy

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
from parallelproj import Array, to_numpy_array

from example_utils import elliptic_cylinder_phantom, show_vol_cuts

# %%
from example_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")
xp, dev = suggest_array_backend_and_device(None, None)

# %%
num_epochs = 50  # epochs (full data passes) for the subset algorithms
num_subsets = 28  # number of ordered view subsets (divides the 168 views evenly)
num_lbfgs = 500  # L-BFGS-B iterations for the reference solution
blank_counts = 500.0  # blank scan counts per LOR
scatter_fraction = 0.5  # scatter relative to mean unscattered transmission

# %%
# Scanner, non-TOF projector, and ground-truth attenuation image
# ---------------------------------------------------------------

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
    parallelproj.pet_lors.Michelogram(
        scanner.num_rings, max_ring_difference=2, span=1
    ),
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
# Simulate transmission data
# ---------------------------

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
# Full-data and subset ingredients
# --------------------------------
#
# We build one subset projector :math:`P_k` per view subset and slice the
# blank / scatter / data sinograms accordingly.  ``P1`` and the subset
# sensitivities ``Pk1`` (forward projections of an all-ones image) are
# precomputed once.

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
    psi = b * xp.exp(-proj(mu))
    ybar = psi + s
    grad = proj.adjoint(psi / ybar * (ybar - y))
    denom = proj.adjoint(P1 * psi**2 / ybar)
    return grad, denom


def subset_grad(mu: Array, k: int) -> Array:
    """Gradient of the subset log-likelihood L_k (no 1/m scaling)."""
    psi = b_k[k] * xp.exp(-subset_proj[k](mu))
    ybar = psi + s_k[k]
    return subset_proj[k].adjoint(psi / ybar * (ybar - y_k[k]))


# %%
# MLTR (full-data baseline)
# -------------------------

mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
cost: dict[str, np.ndarray] = {}
mu_final: dict[str, Array] = {}

c = [neg_logL(mu)]
for ep in range(num_epochs):
    print(f"MLTR       epoch {ep + 1:03}/{num_epochs:03}", end="\r")
    grad, denom = full_grad_and_curv(mu)
    mu = xp.clip(mu + _safe_ratio(grad, denom), 0, None)
    c.append(neg_logL(mu))
print()
cost["MLTR"] = np.asarray(c)
mu_final["MLTR"] = mu

# %%
# OS-MLTR (one subset per update)
# -------------------------------

mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
c = [neg_logL(mu)]
for ep in range(num_epochs):
    print(f"OS-MLTR    epoch {ep + 1:03}/{num_epochs:03}", end="\r")
    for k in range(num_subsets):
        psi = b_k[k] * xp.exp(-subset_proj[k](mu))
        ybar = psi + s_k[k]
        num = subset_proj[k].adjoint(psi / ybar * (ybar - y_k[k]))
        denom = subset_proj[k].adjoint(Pk1[k] * psi**2 / ybar)
        mu = xp.clip(mu + _safe_ratio(num, denom), 0, None)
    c.append(neg_logL(mu))
print()
cost["OS-MLTR"] = np.asarray(c)
mu_final["OS-MLTR"] = mu

# %%
# SVRG (variance-reduced, preconditioned by the anchor MLTR diagonal)
# -------------------------------------------------------------------

rng = np.random.default_rng(1)
mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
c = [neg_logL(mu)]
for ep in range(num_epochs):
    print(f"SVRG       epoch {ep + 1:03}/{num_epochs:03}", end="\r")
    anchor = mu
    g_full, denom_full = full_grad_and_curv(anchor)
    precond = _safe_ratio(xp.ones_like(mu), denom_full)  # MLTR diagonal at anchor
    gk_anchor = [subset_grad(anchor, k) for k in range(num_subsets)]
    for k in rng.permutation(num_subsets):
        g_vr = g_full + num_subsets * (subset_grad(mu, k) - gk_anchor[k])
        mu = xp.clip(mu + precond * g_vr, 0, None)
    c.append(neg_logL(mu))
print()
cost["SVRG"] = np.asarray(c)
mu_final["SVRG"] = mu

# %%
# L-BFGS-B reference solution (no subsets)
# ----------------------------------------

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
# Convergence and reconstructions
# -------------------------------
#
# We plot the absolute cost :math:`-L(\mu)` per epoch (per function
# evaluation for L-BFGS-B), zoomed to the converged region.  OS-MLTR and
# SVRG reach in a few epochs what full MLTR needs many more for -- roughly
# an ``num_subsets``-fold per-epoch speed-up.  With many subsets, however,
# OS-MLTR has no convergence guarantee: it approaches a subset-dependent
# limit cycle and its cost stalls (or rises) above the optimum, whereas the
# variance-reduced SVRG remains stable and keeps decreasing towards the
# L-BFGS-B reference.

c_min = float(min(c.min() for c in cost.values()))
c_max = float(cost["MLTR"][num_epochs // 2])

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), tight_layout=True)
for name in ("MLTR", "OS-MLTR", "SVRG", "L-BFGS-B"):
    ax[0].plot(cost[name], label=name)
ax[0].set_ylim(c_min, c_max)
ax[0].set_xlabel("epoch (subset methods) / function evaluation (L-BFGS-B)")
ax[0].set_ylabel(r"$-L(\mu)$")
ax[0].grid(ls=":")
ax[0].legend()

sl = img_shape[2] // 2
ax[1].plot(
    to_numpy_array(mu_true[:, img_shape[1] // 2, sl]), "k--", label=r"true $\mu$"
)
for name in ("MLTR", "OS-MLTR", "SVRG", "L-BFGS-B"):
    ax[1].plot(to_numpy_array(mu_final[name][:, img_shape[1] // 2, sl]), label=name)
ax[1].set_xlabel("pixel")
ax[1].set_ylabel(r"$\mu$ [1/mm]")
ax[1].grid(ls=":")
ax[1].legend()
fig.show()

# %%
fig2 = show_vol_cuts(
    np.concatenate(
        [to_numpy_array(mu_true)[None]]
        + [to_numpy_array(mu_final[name])[None] for name in mu_final]
    ),
    voxel_size=voxel_size,
    fig_title=r"$\mu$: true / " + " / ".join(mu_final),
    vmin=0,
    vmax=2.5 * mu_water,
)

plt.show()
