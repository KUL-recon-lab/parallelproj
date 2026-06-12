"""
Transmission reconstruction: MLTR, SPS and the Convex algorithm
===============================================================

This example reconstructs a linear attenuation image :math:`\\mu` from
transmission data using the **exact Poisson model** (no log-linearisation
into a weighted least squares problem) including a strictly positive,
smooth scatter background :math:`s` with known mean:

.. math::
    L(\\mu) = \\sum_i y_i \\ln \\bar{y}_i - \\bar{y}_i,
    \\qquad
    \\bar{y}_i = \\bar{\\psi}_i + s_i,
    \\qquad
    \\bar{\\psi}_i = b_i e^{-(P \\mu)_i},

where :math:`b_i` is the blank scan, :math:`P\\mu` are line integrals of
:math:`\\mu`, and :math:`y_i` are the measured transmission counts.  Note
that with the background :math:`s_i > 0` all expressions below are free of
divisions by zero (:math:`\\bar{y}_i \\geq s_i > 0`).

**Preconditioned gradient ascent.**  All three algorithms below are the
*same* preconditioned gradient ascent on the log-likelihood, exactly
analogous to MLEM for the emission problem
(:math:`x \\leftarrow x + \\tfrac{x}{A^T\\mathbf 1}\\nabla_x L`):

.. math::
    \\mu \\leftarrow \\bigl[\\, \\mu + D(\\mu) \\odot \\nabla_\\mu L \\,\\bigr]_+,
    \\qquad
    \\nabla_\\mu L = P^T\\!\\left[\\tfrac{\\bar\\psi}{\\bar y}(\\bar y - y)\\right].

They share the gradient :math:`\\nabla_\\mu L` and differ **only** in the
diagonal preconditioner :math:`D`, which is the inverse of a separable
majorant of the curvature (the weight choice :math:`\\alpha_j` of Nuyts'
note):

* **MLTR** (Nuyts et al. [#f1]_, :math:`\\alpha_j = 1`) uses the Newton-type
  curvature :math:`\\bar\\psi^2/\\bar y`:

  .. math::
      D_j = 1 \\,/\\, P^T\\!\\left[(P\\mathbf 1)\\,
        \\tfrac{\\bar\\psi^2}{\\bar y}\\right]_j .

  Derived from a quadratic *approximation* of :math:`L`, so monotonic
  increase is **not guaranteed** (in practice it almost always holds).

* **SPS** with optimal curvature (Erdogan and Fessler [#f2]_,
  :math:`\\alpha_j = 1`) replaces :math:`\\bar\\psi^2/\\bar y` by the optimal
  curvature :math:`c_i`, the smallest curvature whose parabola *majorises*
  the per-ray negative log-likelihood on :math:`l \\geq 0`:

  .. math::
      D_j = 1 \\,/\\, P^T\\!\\left[(P\\mathbf 1)\\, c\\right]_j,
      \\qquad
      c_i = \\begin{cases}
        \\left[ 2 \\, \\frac{f_i(0) - f_i(l_i) + \\dot{f}_i(l_i) l_i}
          {l_i^2} \\right]_+ & l_i > 0 \\\\
        \\left[ \\ddot{f}_i(0) \\right]_+ & l_i = 0
      \\end{cases}

  with :math:`f_i(l) = (b_i e^{-l} + s_i) - y_i \\ln(b_i e^{-l} + s_i)`
  and :math:`l_i = (P\\mu)_i`.  The optimal curvature is never smaller than
  the Newton curvature, so SPS takes more conservative steps but every
  update is **monotone** (under the concavity condition below).

* **Convex** algorithm (Lange, :math:`\\alpha_j = \\mu_j`) uses a
  *multiplicative* preconditioner -- the direct transmission analogue of
  MLEM, updating each pixel in proportion to its current value:

  .. math::
      D_j = \\mu_j \\,/\\, P^T\\!\\left[(P\\mu)\\,
        \\tfrac{\\bar\\psi^2}{\\bar y}\\right]_j .

  Like MLEM, this requires a strictly **positive initial image** (a voxel
  at exactly zero can never become non-zero).

Each iteration costs two forward and one back projection (plus one
precomputable :math:`P\\mathbf 1`), updates all voxels simultaneously, and
enforces non-negativity by clipping.

.. note::
    With a scatter background the transmission log-likelihood is concave
    only where :math:`y_i s_i \\leq \\bar{y}_i^2`.  Close to the solution
    this holds (:math:`\\bar{y}_i \\to y_i` and :math:`s_i < y_i` for
    reasonable scatter fractions), and experience shows convergence is not
    a problem -- see the discussions in [#f1]_ and [#f2]_.

.. note::
    MLTR is usually the fastest and is well behaved in typical regimes.
    SPS trades a little speed for a guaranteed monotone increase of
    :math:`L`, which matters in low-count / high-scatter data, automated
    pipelines, ordered-subset schemes, or MAP with non-quadratic priors.
    Faster variants not shown: ordered-subsets SPS and momentum-accelerated
    OS-SQS, or -- since the objective is smooth -- generic solvers like
    L-BFGS-B.  A penalised (MAPTR) extension adds a prior surrogate to the
    same separable framework.

.. rubric:: References

.. [#f1] J. Nuyts, B. De Man, P. Dupont, M. Defrise, P. Suetens, L. Mortelmans,
   "Iterative reconstruction for helical CT: a simulation study",
   Phys. Med. Biol. 43 (1998) 729-737.

.. [#f2] H. Erdogan, J. A. Fessler,
   "Monotonic algorithms for transmission tomography",
   IEEE Trans. Med. Imaging 18 (1999) 801-814.

.. note::

   To run this example locally, download
   `example_utils.py <https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/example_utils.py>`_
   into the **same folder** as this script. Make sure ``parallelproj`` is installed.
"""

# %%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

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
num_iter = 300  # iterations for both algorithms
blank_counts = 5000.0  # blank scan counts per LOR
scatter_fraction = 0.9  # scatter relative to mean unscattered transmission

# %%
# Scanner, non-TOF projector, and ground-truth attenuation image
# ---------------------------------------------------------------
#
# Transmission data have no TOF information, so we use a plain non-TOF
# projector.  The ground-truth :math:`\mu` image is the elliptic cylinder
# phantom rescaled such that the cylinder background equals water at
# 511 keV (:math:`0.0096 \, \text{mm}^{-1}`); the hot / cold inserts
# become dense / air-like regions.

num_rings = 3
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=16,
    num_lor_endpoints_per_side=12,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-6, 6, num_rings, device=dev),
    symmetry_axis=2,
)

img_shape = (55, 55, 4)
voxel_size = (2.0, 2.0, 2.0)

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
#
# Noise-free unscattered transmission :math:`\bar{\psi} = b e^{-P\mu}`,
# plus a smooth (here: constant) strictly positive scatter background with
# known mean, then Poisson noise.

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
# Shared ingredients
# ------------------
#
# All algorithms use the same gradient of the log-likelihood
#
# .. math::
#     \nabla_\mu L = P^T\left[\frac{\bar{\psi}}{\bar{y}}(\bar{y} - y)\right]
#
# and the forward projection of an all-ones image :math:`P\mathbf{1}`
# (the De Pierro separability weights), precomputed once.

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


def _precond_from_denom(mu: Array, alpha_img: Array, curv_sino: Array) -> Array:
    """Diagonal preconditioner ``alpha / P^T[(P alpha) * curv]`` (FOV-safe)."""
    denom = proj.adjoint(proj(alpha_img) * curv_sino)
    denom_safe = xp.where(fov_mask, denom, xp.ones_like(denom))
    return xp.where(fov_mask, alpha_img / denom_safe, xp.zeros_like(mu))


def precond_mltr(mu: Array, psi: Array, ybar: Array) -> Array:
    """MLTR preconditioner: alpha = 1, Newton curvature psi^2 / ybar."""
    return _precond_from_denom(mu, ones_img, psi**2 / ybar)


def precond_convex(mu: Array, psi: Array, ybar: Array) -> Array:
    """Convex preconditioner: alpha = mu (multiplicative, MLEM-like)."""
    return _precond_from_denom(mu, mu, psi**2 / ybar)


def precond_sps(mu: Array, psi: Array, ybar: Array) -> Array:
    """SPS preconditioner: alpha = 1, Erdogan & Fessler optimal curvature."""
    l = proj(mu)
    # optimal curvature of f(l) = (b e^-l + s) - y log(b e^-l + s)
    f_l = ybar - y * xp.log(ybar)
    f_0 = (b + s) - y * xp.log(b + s)
    fdot_l = psi / ybar * (y - ybar)
    fddot_0 = xp.clip(b * (1 - y * s / (b + s) ** 2), 0, None)
    small = l < 1e-3  # fall back to f''(0) for tiny l (cancellation)
    l_safe = xp.where(small, xp.ones_like(l), l)
    curv = xp.where(
        small,
        fddot_0,
        xp.clip(2 * (f_0 - f_l + fdot_l * l) / l_safe**2, 0, None),
    )
    return _precond_from_denom(mu, ones_img, curv)


# %%
# Run all three algorithms as preconditioned gradient ascent
# ----------------------------------------------------------
#
# Identical update skeleton ``mu <- [mu + D * grad]_+``; only the
# preconditioner ``D`` differs.  MLTR / SPS start from zero, the Convex
# algorithm needs a strictly positive start (its preconditioner is
# proportional to the current image).

algorithms = {
    "MLTR": precond_mltr,
    "SPS": precond_sps,
    "Convex": precond_convex,
}

# Convex must start > 0; MLTR / SPS are happy at 0
mu0_zero = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
mu0_pos = xp.where(fov_mask, xp.full_like(mu0_zero, mu_water), mu0_zero)
init = {"MLTR": mu0_zero, "SPS": mu0_zero, "Convex": mu0_pos}

mu_final: dict[str, Array] = {}
cost: dict[str, np.ndarray] = {}

for name, precond in algorithms.items():
    mu = init[name]
    c = np.zeros(num_iter + 1)
    c[0] = neg_logL(mu)
    for it in range(num_iter):
        print(f"{name:6} iteration {it + 1:03}/{num_iter:03}", end="\r")
        grad, psi, ybar = grad_logL(mu)
        mu = xp.clip(mu + precond(mu, psi, ybar) * grad, 0, None)
        c[it + 1] = neg_logL(mu)
    print()
    mu_final[name] = mu
    cost[name] = c

# count monotonicity violations beyond the float32 rounding level of -L
# (the updates themselves run in float32)
c_min = min(c.min() for c in cost.values())
tol = float(np.finfo(np.float32).eps) * abs(c_min)
for name in algorithms:
    viol = int(np.sum(np.diff(cost[name]) > tol))
    print(f"{name:6}: final -L = {cost[name][-1]:.2f}, non-monotone steps = {viol}")

# %%
# Results
# -------
#
# All three algorithms converge to the same maximum-likelihood solution.
# MLTR is typically the fastest; SPS additionally *guarantees* a monotone
# increase of :math:`L` (its optimal curvature is never smaller than the
# Newton curvature, hence the slightly more conservative steps); the Convex
# algorithm is the multiplicative, MLEM-like variant.

fig, ax = plt.subplots(1, 2, figsize=(11, 4.5), tight_layout=True)
for name in algorithms:
    ax[0].plot(cost[name], label=name)
ax[0].set_xlabel("iteration")
ax[0].set_ylabel(r"$-L(\mu) - \min(-L) + 1$")
ax[0].set_ylim(cost["MLTR"].min(), cost["MLTR"][10:].max())
ax[0].grid(ls=":")
ax[0].legend()

sl = img_shape[2] // 2
ax[1].plot(
    to_numpy_array(mu_true[:, img_shape[1] // 2, sl]), "k--", label=r"true $\mu$"
)
for name in algorithms:
    ax[1].plot(
        to_numpy_array(mu_final[name][:, img_shape[1] // 2, sl]), label=name
    )
ax[1].set_xlabel("pixel")
ax[1].set_ylabel(r"$\mu$ [1/mm]")
ax[1].grid(ls=":")
ax[1].legend()
fig.show()

# %%
fig2 = show_vol_cuts(
    np.concatenate(
        [to_numpy_array(mu_true)[None]]
        + [to_numpy_array(mu_final[name])[None] for name in algorithms]
    ),
    voxel_size=voxel_size,
    fig_title=r"$\mu$: true / " + " / ".join(algorithms),
    vmin=0,
    vmax=2.5 * mu_water,
)

plt.show()