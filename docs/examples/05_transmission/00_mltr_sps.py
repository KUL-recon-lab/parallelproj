"""
Transmission reconstruction with MLTR, SPS and L-BFGS-B
=======================================================

:bdg-primary:`transmission` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

A transmission scan measures how much an external photon source is
attenuated along each line of response; reconstructing the linear
attenuation image :math:`\\mu` from these counts is needed e.g. for the
attenuation correction of PET data.  This example maximises the **exact
Poisson log-likelihood** of transmission data with a known scatter background
(no log-linearisation into a weighted least-squares problem) with three
algorithms: **MLTR** and **SPS**, two preconditioned gradient ascent methods
that differ only in their diagonal preconditioner, and the general-purpose
quasi-Newton optimiser **L-BFGS-B** as a reference.

**You will learn**

- how to simulate transmission data :math:`y` with a blank scan and a scatter
  background,
- how to compute the gradient of the transmission log-likelihood with a
  forward and a back projection,
- how MLTR and SPS are derived as preconditioned gradient ascent with
  different separable curvatures,
- how to use SciPy's L-BFGS-B with a parallelproj projector and compare the
  convergence of all three algorithms.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_01_run_pet_non_tof_sinogram_projector.py`
(non-TOF projector) and
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(MLEM as preconditioned gradient descent for emission data).

.. note::

    Because of its run time (500 iterations per algorithm), this example is
    **not executed** in the documentation build: the figures are not
    rendered on this page.  Download it and run it locally to see the
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

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    suggest_array_backend_and_device,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Simulation and reconstruction settings
# -----------------------------------------
#
# ``blank_counts`` sets the noise level of the data (expected counts per LOR
# without an object), ``scatter_fraction`` the constant scatter background
# relative to the mean unscattered transmission.  All three algorithms run
# ``num_iter`` iterations (function evaluations for L-BFGS-B).

num_iter = 500  # iterations for all algorithms
blank_counts = 500.0  # blank scan counts per LOR
scatter_fraction = 0.5  # scatter relative to mean unscattered transmission

# %%
# 2. Scanner, projector and ground-truth attenuation image
# --------------------------------------------------------
#
# Transmission data have no TOF information, so we use a plain non-TOF
# projector of a clinical-size scanner (radius 300 mm, 3 rings) and an image
# of 100 x 100 x 3 voxels with one 5.3 mm slice per ring.  The ground-truth
# :math:`\mu` image is the elliptic cylinder phantom rescaled such that the
# cylinder background equals the linear attenuation coefficient of water at
# 511 keV (:math:`0.0096 \, \text{mm}^{-1}`); the hot / cold inserts become
# dense / air-like regions.

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
# The expected data are the noise-free unscattered transmission
# :math:`\bar{z}_i = b_i e^{-(P\mu)_i}`, with the blank scan :math:`b_i` and
# the line integrals :math:`(P\mu)_i`, plus a smooth (here: constant),
# strictly positive scatter background :math:`s_i` with known mean.  The
# measured counts :math:`y_i` are a Poisson realisation.

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
# 4. Transmission log-likelihood and its gradient
# -----------------------------------------------
#
# The Poisson log-likelihood of the transmission data is
#
# .. math::
#     L(\mu) = \sum_i y_i \ln \bar{y}_i (\mu) - \bar{y}_i (\mu),
#     \qquad
#     \bar{y}_i (\mu) = \bar{z}_i (\mu) + s_i,
#     \qquad
#     \bar{z}_i (\mu) = b_i e^{-(P \mu)_i}.
#
# With the background :math:`s_i > 0` all expressions below are free of
# divisions by zero (:math:`\bar{y}_i \geq s_i > 0`).  All algorithms use
# its gradient
#
# .. math::
#     \nabla_\mu L = P^T\left[\frac{\bar{z}}{\bar{y}}(\bar{y} - y)\right],
#
# which costs one forward and one back projection.  The forward projection
# of an all-ones image :math:`P\mathbf{1}` (the intersection lengths of the
# LORs with the image) is needed by the preconditioners below and is
# precomputed once.
#
# .. note::
#     With a scatter background the transmission log-likelihood is concave
#     only where :math:`y_i s_i \leq \bar{y}_i^2`.  Close to the solution
#     this holds (:math:`\bar{y}_i \to y_i` and :math:`s_i < y_i` for
#     reasonable scatter fractions), and experience shows convergence is not
#     a problem -- see the discussions in :footcite:p:`Nuyts1998` and
#     :footcite:p:`Erdogan1999`.

ones_img = xp.ones(proj.in_shape, dtype=xp.float32, device=dev)
P1 = proj(ones_img)  # sinogram of intersection-length sums


def neg_logL(mu: Array) -> float:
    """Negative transmission Poisson log-likelihood (to be minimised).

    The per-bin terms are accumulated in float64: near convergence the
    iteration-to-iteration changes of :math:`-L` drop below the float32
    rounding level of its absolute value.
    """
    ybar = b * xp.exp(-proj(mu)) + s
    return float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64)))


def grad_logL(mu: Array) -> tuple[Array, Array, Array]:
    """Gradient of the log-likelihood and the intermediates psi, ybar."""
    psi = b * xp.exp(-proj(mu))
    ybar = psi + s
    grad = proj.adjoint(psi / ybar * (ybar - y))
    return grad, psi, ybar


# %%
# 5. MLTR and SPS as preconditioned gradient ascent
# -------------------------------------------------
#
# MLTR and SPS are the *same* preconditioned gradient ascent on the
# log-likelihood, exactly analogous to MLEM for the emission problem
# (:math:`x \leftarrow x + \tfrac{x}{A^T\mathbf 1}\nabla_x L`):
#
# .. math::
#     \mu \leftarrow \bigl[\, \mu + D(\mu) \odot \nabla_\mu L \,\bigr]_+ .
#
# They share the gradient :math:`\nabla_\mu L` and differ **only** in the
# diagonal preconditioner :math:`D`, the inverse of a separable majorant of
# the curvature (the weight choice :math:`\alpha_j = 1` for MLTR):
#
# * **MLTR** (Nuyts et al. :footcite:p:`Nuyts1998`) uses the Newton-type
#   curvature :math:`\bar z^2/\bar y`:
#
#   .. math::
#       D_j = 1 \,/\, \left( P^T\!\left[(P\mathbf 1)\,
#         \tfrac{\bar z^2}{\bar y}\right] \right)_j .
#
#   It is derived from a quadratic *approximation* of :math:`L`, so a
#   monotone increase of :math:`L` is **not guaranteed**.
#
# * **SPS** with optimal curvature (Erdogan and Fessler
#   :footcite:p:`Erdogan1999`) replaces :math:`\bar z^2/\bar y` by the
#   optimal curvature :math:`c_i`, the smallest curvature whose parabola
#   *majorises* the per-ray negative log-likelihood on :math:`l \geq 0`:
#
#   .. math::
#       D_j = 1 \,/\, P^T\!\left[(P\mathbf 1)\, c\right]_j,
#       \qquad
#       c_i = \begin{cases}
#         \left[ 2 \, \frac{f_i(0) - f_i(l_i) + \dot{f}_i(l_i) l_i}
#           {l_i^2} \right]_+ & l_i > 0 \\
#         \left[ \ddot{f}_i(0) \right]_+ & l_i = 0
#       \end{cases}
#
#   with :math:`f_i(l) = (b_i e^{-l} + s_i) - y_i \ln(b_i e^{-l} + s_i)`
#   and :math:`l_i = (P\mu)_i`.  The optimal curvature is never smaller than
#   the Newton curvature, so SPS takes more conservative steps, but every
#   update is **monotone** (under the concavity condition of section 4).
#
# Each iteration costs one forward and two back projections (plus the
# precomputed :math:`P\mathbf 1`), updates all voxels simultaneously, and
# enforces non-negativity by clipping.  Voxels outside the FOV have a zero
# preconditioner denominator and are not updated.


def _precond(curv_sino: Array, mu: Array) -> Array:
    """Diagonal preconditioner ``1 / P^T[(P 1) * curv]`` (FOV-safe).

    Masked-out voxels have a zero denominator, so the denominator is set to
    1 there and the preconditioner to 0 (no update outside the FOV).
    """
    denom = proj.adjoint(P1 * curv_sino)
    denom_safe = xp.where(fov_mask, denom, xp.ones_like(denom))
    return xp.where(fov_mask, 1.0 / denom_safe, xp.zeros_like(mu))


def precond_mltr(mu: Array, psi: Array, ybar: Array) -> Array:
    """MLTR preconditioner: inverse Newton curvature psi^2 / ybar."""
    return _precond(psi**2 / ybar, mu)


def precond_sps(mu: Array, psi: Array, ybar: Array) -> Array:
    """SPS preconditioner: inverse Erdogan & Fessler optimal curvature."""
    line_int = proj(mu)
    # optimal curvature of f(l) = (b e^-l + s) - y log(b e^-l + s)
    f_l = ybar - y * xp.log(ybar)
    f_0 = (b + s) - y * xp.log(b + s)
    fdot_l = psi / ybar * (y - ybar)
    fddot_0 = xp.clip(b * (1 - y * s / (b + s) ** 2), 0, None)
    small = line_int < 1e-3  # fall back to f''(0) for tiny l (cancellation)
    line_int_safe = xp.where(small, xp.ones_like(line_int), line_int)
    curv = xp.where(
        small,
        fddot_0,
        xp.clip(2 * (f_0 - f_l + fdot_l * line_int) / line_int_safe**2, 0, None),
    )
    return _precond(curv, mu)


# %%
# 6. Running MLTR and SPS
# -----------------------
#
# Both algorithms use the identical update skeleton
# ``mu <- [mu + D * grad]_+`` from a zero initialisation; only the
# preconditioner ``D`` differs.  We record :math:`-L` after every iteration
# and count the iterations in which it increases by more than the float32
# rounding level (the updates themselves run in float32), i.e. the
# non-monotone steps.

algorithms = {"MLTR": precond_mltr, "SPS": precond_sps}

mu_final: dict[str, Array] = {}
cost: dict[str, np.ndarray] = {}

for name, precond in algorithms.items():
    mu = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
    cost_iter = np.zeros(num_iter + 1)
    cost_iter[0] = neg_logL(mu)
    for it in range(num_iter):
        print(f"{name:4} iteration {it + 1:03}/{num_iter:03}", end="\r")
        grad, psi, ybar = grad_logL(mu)
        mu = xp.clip(mu + precond(mu, psi, ybar) * grad, 0, None)
        cost_iter[it + 1] = neg_logL(mu)
    print()
    mu_final[name] = mu
    cost[name] = cost_iter

cost_min = min(c.min() for c in cost.values())
tol = float(np.finfo(np.float32).eps) * abs(cost_min)
for name in algorithms:
    viol = int(np.sum(np.diff(cost[name]) > tol))
    print(f"{name:4}: final -L = {cost[name][-1]:.2f}, non-monotone steps = {viol}")

# %%
# 7. L-BFGS-B on the exact objective
# ----------------------------------
#
# For comparison we run **L-BFGS-B**, a general-purpose bound-constrained
# quasi-Newton optimiser from SciPy, directly on the smooth objective
# :math:`-L(\mu)` with the box constraint :math:`\mu \geq 0`.  Because the
# transmission log-likelihood is smooth and (away from strong scatter)
# concave, no surrogate is needed: L-BFGS-B builds its own quasi-Newton
# metric from the gradient history.
#
# The objective works on a flat float64 vector (SciPy's convention); inside
# it we reshape, cast to the array-API backend, and return the value
# together with the gradient of :math:`-L`.  The objective records :math:`-L`
# at every function evaluation, which costs one forward and one back
# projection, roughly comparable to one MLTR / SPS iteration.

n_vox = int(np.prod(proj.in_shape))
cost_lbfgs: list[float] = []


def neg_logL_and_grad(mu_flat: np.ndarray) -> tuple[float, np.ndarray]:
    mu = xp.asarray(mu_flat.reshape(proj.in_shape), dtype=xp.float32, device=dev)
    psi = b * xp.exp(-proj(mu))
    ybar = psi + s
    val = float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64)))
    # gradient of -L (note the sign flip vs. grad_logL)
    grad = proj.adjoint(psi / ybar * (y - ybar))
    cost_lbfgs.append(val)
    return val, np.asarray(to_numpy_array(grad)).ravel().astype(np.float64)


res = minimize(
    neg_logL_and_grad,
    np.zeros(n_vox),
    jac=True,
    method="L-BFGS-B",
    bounds=[(0.0, None)] * n_vox,
    options={"maxiter": num_iter, "maxfun": num_iter},
)
mu_final["L-BFGS-B"] = xp.asarray(
    res.x.reshape(proj.in_shape), dtype=xp.float32, device=dev
)
cost["L-BFGS-B"] = np.asarray(cost_lbfgs)
print(
    f"L-BFGS-B: final -L = {cost['L-BFGS-B'][-1]:.2f}, "
    f"function evals = {len(cost_lbfgs)}"
)

# %%
# 8. Convergence and line profiles
# --------------------------------
#
# The left panel shows :math:`-L(\mu)` per iteration (per function
# evaluation for L-BFGS-B), zoomed to the range below the MLTR value after
# 50 iterations.  The right panel compares profiles along :math:`x_0`
# through the centre of the final images with the ground truth.

cost_max = float(cost["MLTR"][50])

sl = img_shape[2] // 2
x0_mm = (np.arange(img_shape[0]) - (img_shape[0] - 1) / 2) * voxel_size[0]

fig, ax = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
for name in cost:
    ax[0].plot(cost[name], label=name)
ax[0].set_ylim(cost_min, cost_max)
ax[0].set_xlabel("iteration (MLTR / SPS) or function evaluation (L-BFGS-B)")
ax[0].set_ylabel(r"$-L(\mu)$")
ax[0].set_title("Convergence")
ax[0].grid(ls=":")
ax[0].legend()

ax[1].plot(
    x0_mm,
    to_numpy_array(mu_true[:, img_shape[1] // 2, sl]),
    "k--",
    label=r"true $\mu$",
)
for name in mu_final:
    ax[1].plot(
        x0_mm, to_numpy_array(mu_final[name][:, img_shape[1] // 2, sl]), label=name
    )
ax[1].set_xlabel(r"$x_0$ [mm]")
ax[1].set_ylabel(r"$\mu$ [1/mm]")
ax[1].set_title(r"Central profile along $x_0$")
ax[1].grid(ls=":")
ax[1].legend()
fig.show()

# %%
# All three algorithms approach the same maximum-likelihood solution, and
# their profiles are almost indistinguishable: they follow the true
# :math:`\mu` with the noise of the unregularised maximum-likelihood
# estimate.  Of the two surrogate methods, MLTR decreases :math:`-L`
# faster; SPS additionally *guarantees* a monotone increase of :math:`L`.
#
# In practice, across the regimes tested for this example -- including
# low-count, high-attenuation data -- MLTR remained monotone (see the printed
# number of non-monotone steps) *and* reached a slightly **higher**
# likelihood than SPS at the same iteration: its Newton-type curvature is
# never larger than the SPS majorant, so it takes larger steps.
#
# On this unregularised problem L-BFGS-B converges faster than MLTR -- its
# quasi-Newton metric captures the curvature that the separable surrogates
# approximate with a fixed diagonal.  MLTR / SPS nevertheless remain
# attractive for their simplicity, guaranteed non-negativity without a
# constrained solver, trivial parallelism, and natural extension to ordered
# subsets (OS-MLTR, next example).

# %%
# 9. Reconstructed images
# -----------------------
#
# The central transaxial slice of the true :math:`\mu` image and of the
# three reconstructions, with the same colour scale.

images = {r"true $\mu$": mu_true} | mu_final
extent = (
    -img_shape[0] * voxel_size[0] / 2,
    img_shape[0] * voxel_size[0] / 2,
    img_shape[1] * voxel_size[1] / 2,
    -img_shape[1] * voxel_size[1] / 2,
)

fig2, ax2 = plt.subplots(
    1, len(images), figsize=(10, 3.2), sharey=True, layout="constrained"
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
# The three reconstructions look alike and show the elliptic cylinder with
# its dense and air-like inserts.  Without regularisation they all contain
# noticeable noise; a prior is added in
# :ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py`.

# %%
# Summary
# -------
#
# - The exact Poisson transmission log-likelihood with a positive scatter
#   background is smooth, and its gradient
#   :math:`P^T[\tfrac{\bar z}{\bar y}(\bar y - y)]` costs one forward and
#   one back projection.
# - MLTR and SPS are the same preconditioned gradient ascent and differ only
#   in the separable curvature of their diagonal preconditioner: Newton-type
#   :math:`\bar z^2/\bar y` (MLTR, faster) or the optimal majorising
#   curvature (SPS, provably monotone).
# - L-BFGS-B on :math:`-L` with the bound :math:`\mu \geq 0` converges
#   faster than both on this unregularised problem, but unlike MLTR it does
#   not extend naturally to ordered subsets.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_05_transmission_01_os_mltr_svrg.py`
#    -- accelerating MLTR with ordered subsets (OS-MLTR) and SVRG.
#
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_path = "_static/thumbnails/00_mltr_sps.png"
