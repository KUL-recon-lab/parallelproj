"""
Joint activity and attenuation reconstruction (MLAA) for TOF PET
================================================================

Maximum Likelihood Activity and Attenuation (MLAA) jointly estimates the
activity image :math:`\\lambda` **and** the attenuation image :math:`\\mu`
from a single (TOF) emission scan, without a separate transmission/CT
measurement.  The TOF emission model is

.. math::
    \\bar{y}_{i,t} = a_i \\, (P \\lambda)_{i,t} + s_{i,t},
    \\qquad
    a_i = e^{-(P_\\text{nt}\\,\\mu)_i},

where :math:`P` is the **TOF** emission projector, :math:`P_\\text{nt}` the
**non-TOF** projector used for the attenuation line integrals (the
attenuation factor :math:`a_i` is the same for every TOF bin :math:`t` of a
given LOR :math:`i`), and :math:`s` is a known, smooth, strictly positive
contamination (scatter + randoms).

MLAA alternates two block updates of the penalised log-likelihood
:math:`L(\\lambda,\\mu) - \\beta_\\lambda R(\\lambda) - \\beta_\\mu R(\\mu)`:

* **activity** (fix :math:`\\mu`): penalised OSEM with the attenuation in
  the system matrix -- preconditioned gradient ascent with the
  EM/harmonic-mean preconditioner :math:`D_\\lambda = \\lambda /
  (A^T\\mathbf 1 + \\lambda\\,\\beta_\\lambda\\kappa/\\delta_\\lambda)`;
* **attenuation** (fix :math:`\\lambda`): penalised **OS-MLTR** -- this is
  exactly the transmission reconstruction of the ``05_tranmsmission``
  examples, with the *blank scan* replaced by the current activity forward
  projection :math:`b = P\\lambda` (and, with TOF, the per-TOF-bin MLTR
  weights summed over the TOF axis before back-projecting through the
  non-TOF projector).

**Why TOF is essential.**  Non-TOF MLAA is ill-posed: activity and
attenuation trade off against each other (crosstalk), and the joint problem
is non-unique.  TOF data determine :math:`\\lambda` and :math:`\\mu` up to a
single global scalar [Rezaei2012]_, which we fix by anchoring a region of
known (water) attenuation after each attenuation update.

**Two further practical ingredients used here:**

* the attenuation :math:`\\mu` is updated **only inside the object support**
  (obtained by thresholding the quick non-attenuation-corrected activity
  image); estimating :math:`\\mu` in the surrounding air/low-sensitivity
  region makes the joint problem unstable;
* both images carry an edge-preserving **log-cosh** prior with the
  harmonic-mean (data + prior curvature) preconditioner of
  ``05_tranmsmission/02_maptr.py``.

The activity and attenuation phantoms have **different** inserts on purpose,
so the reconstructions reveal how well MLAA separates the two (little
crosstalk = each image shows only its own structure).

.. note::
    This example uses the larger transmission-scanner geometry and is
    deliberately not part of the rendered gallery (no ``run_`` prefix): the
    TOF reconstruction with many subsets and outer iterations is slow on the
    CPU.  Run it locally, ideally on a GPU backend.

.. note::
    Keeping the contamination :math:`s` known/fixed is a simplification --
    in practice the scatter estimate depends on :math:`\\mu` and is
    re-estimated.  Likewise the NAC warm-start keeps the known :math:`s`
    while omitting attenuation, which is mild "cheating" for a clean start.

.. [Rezaei2012] A. Rezaei et al., "Simultaneous reconstruction of activity
   and attenuation in time-of-flight PET", IEEE TMI 31(12), 2012.

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
from scipy.ndimage import binary_fill_holes

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import Array, to_numpy_array
from parallelproj.functions import C2AffineObjective, LogCosh

from example_utils import elliptic_cylinder_phantom

# %%
from example_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")
xp, dev = suggest_array_backend_and_device(None, None)

# %%
num_subsets = 28  # ordered view subsets (divides the 168 views evenly)
num_outer = 20  # MLAA outer iterations
num_mltr_epochs = 3  # OS-MLTR epochs per outer iteration (MLTR is slower than MLEM)
scatter_fraction = 0.3  # contamination relative to mean true emission
count_factor = 5.0  # scales the activity (sets the count level / noise)

mu_water = 0.0096  # 1/mm at 511 keV

# edge-preserving log-cosh prior weights (harmonic-mean preconditioner as in
# 02_maptr)
beta_lam = 0.01  # activity prior weight
beta_mu = 5.0  # attenuation prior weight
delta_mu = mu_water / 2  # mu edges (inserts) >> delta are preserved
# delta_lam (the activity log-cosh scale) is derived from the warm-start below

# %%
# Scanner (large transmission geometry), TOF + non-TOF projectors, phantoms
# -------------------------------------------------------------------------

num_rings = 3
ring_spacing = 5.3
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=300.0,
    num_sides=56,
    num_lor_endpoints_per_side=6,
    lor_spacing=5.3,
    ring_positions=(
        xp.arange(num_rings, dtype=xp.float32, device=dev) - (num_rings - 1) / 2
    )
    * ring_spacing,
    symmetry_axis=2,
)

img_shape = (100, 100, num_rings)
voxel_size = (4.0, 4.0, ring_spacing)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

# non-TOF projector for the attenuation line integrals
proj_nt = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)
# TOF projector for the activity; 31 bins x 20 mm = 620 mm cover the ~600 mm
# LORs, FWHM 60 mm (sigma = FWHM / 2.355)
proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=31, tofbin_width=20.0, sigma_tof=60.0 / 2.355
)

fov_mask = proj_nt.fov_mask()

# %%
# Ground-truth activity and attenuation -- DIFFERENT insert patterns
# -------------------------------------------------------------------
#
# The activity uses the standard elliptic-cylinder phantom (its hot/cold
# inserts).  The attenuation is a water cylinder with its **own** dense and
# air-like inserts at different locations, so crosstalk between the two
# images is detectable.

activity_phantom = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)
act_true = count_factor * activity_phantom

cyl = to_numpy_array(activity_phantom) > 0  # body outline (shared support)
nx, ny, _ = img_shape
yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
dense = (xx - nx // 2) ** 2 + (yy - int(0.72 * ny)) ** 2 < 8**2  # bone-like
air = (xx - nx // 2) ** 2 + (yy - int(0.28 * ny)) ** 2 < 8**2  # lung-like
dense3 = np.repeat(dense[:, :, None], num_rings, axis=2) & cyl
air3 = np.repeat(air[:, :, None], num_rings, axis=2) & cyl

mu_np = np.where(cyl, mu_water, 0.0)
mu_np = np.where(dense3, 0.02, mu_np)  # dense insert
mu_np = np.where(air3, 0.002, mu_np)  # air-like insert
mu_true = xp.asarray(mu_np.astype(np.float32), device=dev)

# %%
# Simulate TOF emission data
# ---------------------------

att_true = xp.exp(-proj_nt(mu_true))  # (R, V, P) attenuation factors
emis_true = att_true[..., None] * proj(act_true)  # broadcast over TOF bins
s = xp.full(
    proj.out_shape,
    scatter_fraction * float(xp.mean(emis_true)),
    device=dev,
    dtype=xp.float32,
)
ybar_true = emis_true + s

np.random.seed(1)
y = xp.asarray(
    np.random.poisson(to_numpy_array(ybar_true)), device=dev, dtype=xp.float32
)
print(f"mean emission counts / (LOR, TOF bin) = {float(xp.mean(y)):.2f}")

# %%
# Subsets, priors and shared helpers
# ----------------------------------

subset_views, subset_slices = lor_desc.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)  # 4D (TOF) slices
)

proj_k = []  # TOF subset projectors (activity)
proj_nt_k = []  # non-TOF subset projectors (attenuation)
for k in range(num_subsets):
    p = copy(proj)
    p.views = subset_views[k]
    proj_k.append(p)
    q = copy(proj_nt)
    q.views = subset_views[k]
    proj_nt_k.append(q)

y_k = [y[subset_slices[k]] for k in range(num_subsets)]
s_k = [s[subset_slices[k]] for k in range(num_subsets)]

ones_img = xp.ones(img_shape, dtype=xp.float32, device=dev)
Pnt1_k = [proj_nt_k[k](ones_img) for k in range(num_subsets)]  # subset att sensitivity

G = parallelproj.operators.FiniteForwardDifference(img_shape)
kappa = 2.0 * len(img_shape)  # diag(G^T G) for forward differences


def emission_neg_logL(lam: Array, mu: Array) -> float:
    """Negative TOF emission Poisson log-likelihood (float64 accumulation)."""
    ybar = xp.exp(-proj_nt(mu))[..., None] * proj(lam) + s
    return float(xp.sum(xp.astype(ybar - y * xp.log(ybar), xp.float64)))


def _safe(num: Array, denom: Array, mask: Array) -> Array:
    """``num / denom`` where ``mask`` holds and ``denom > 0``, else 0."""
    ok = mask & (denom > 0)
    denom_safe = xp.where(ok, denom, xp.ones_like(denom))
    return xp.where(ok, num / denom_safe, xp.zeros_like(num))


# %%
# Non-attenuation-corrected (NAC) OSEM warm-start
# -----------------------------------------------
#
# One OSEM epoch without an attenuation model (but with the known
# contamination) gives a fast, attenuation-biased activity image used to
# (a) initialise the activity and (b) define the object support

lam = xp.where(fov_mask, ones_img, xp.zeros_like(ones_img))
for k in range(num_subsets):
    ybar = proj_k[k](lam) + s_k[k]
    sens = proj_k[k].adjoint(xp.ones_like(ybar))
    update = proj_k[k].adjoint(y_k[k] / ybar)
    lam = _safe(lam * update, sens, fov_mask)
print(f"NAC OSEM done (lam max = {float(xp.max(lam)):.1f})")

# 0th-order attenuation: water inside the thresholded support, air outside
support = lam > 0.1 * float(xp.mean(lam[lam > 0]))
# Fill interior holes (low-/no-activity regions inside the body, e.g. the
# activity's cold inserts) per transaxial slice, so the 0th-order
# attenuation is a *solid* water blob.  Holes would leave attenuation-free
# pockets inside the body that bias and slow the MLAA attenuation update.
support_np = to_numpy_array(support)
support_np = np.stack(
    [binary_fill_holes(support_np[:, :, z]) for z in range(support_np.shape[2])],
    axis=2,
)
support = xp.asarray(support_np, device=dev) & fov_mask

# 0th-order attenuation image: uniform water inside the filled support
mu0 = xp.where(support, xp.asarray(mu_water, dtype=xp.float32), xp.zeros_like(lam))

# small central water-attenuation calibration region (away from the inserts)
water_roi_np = np.zeros(img_shape, dtype=bool)
water_roi_np[nx // 2 - 6 : nx // 2 + 6, ny // 2 - 6 : ny // 2 + 6, :] = True
water_roi = xp.asarray(water_roi_np, device=dev) & support

# %%
# Log-cosh priors
# ---------------
#
# The log-cosh transition scale for the activity is tied to the warm-start
# activity level; the attenuation scale ``delta_mu`` was set above.  The
# (log-cosh max) prior curvature :math:`\beta\,\kappa/\delta` enters the
# harmonic-mean preconditioner.

delta_lam = 0.3 * float(xp.mean(lam[lam > 0]))
reg_lam = C2AffineObjective(LogCosh(delta=delta_lam, beta=beta_lam), G)
reg_mu = C2AffineObjective(LogCosh(delta=delta_mu, beta=beta_mu), G)
prior_curv_lam = beta_lam * kappa / delta_lam
prior_curv_mu = beta_mu * kappa / delta_mu

# %%
# Baseline: OS-MLEM activity with the fixed 0th-order attenuation image
# ---------------------------------------------------------------------
#
# Reconstruct the activity with attenuation correction based on the crude
# uniform-water :math:`\mu_0` (held fixed, no joint estimation).  Wherever
# the true attenuation differs from water (the dense / air inserts), this
# baseline shows attenuation-correction artefacts that MLAA removes.

a0_k = [xp.exp(-proj_nt_k[k](mu0))[..., None] for k in range(num_subsets)]
lam_ac = lam  # start from the NAC activity
for it in range(num_outer):
    print(f"OSEM (mu0) epoch {it + 1:03}/{num_outer:03}", end="\r")
    for k in range(num_subsets):
        ybar = a0_k[k] * proj_k[k](lam_ac) + s_k[k]
        grad = proj_k[k].adjoint(a0_k[k] * (y_k[k] / ybar - 1.0))
        sens = proj_k[k].adjoint(a0_k[k] * xp.ones_like(ybar))
        g_pen = grad - reg_lam.gradient(lam_ac) / num_subsets
        D = _safe(lam_ac, sens + lam_ac * prior_curv_lam / num_subsets, fov_mask)
        lam_ac = xp.clip(lam_ac + D * g_pen, 0, None)
print()

# %%
# MLAA: interleaved penalised OS-MLEM (activity) and OS-MLTR (attenuation)
# ------------------------------------------------------------------------

mu = mu0  # attenuation initialised at the 0th-order water blob
cost = np.zeros(num_outer + 1)
cost[0] = emission_neg_logL(lam, mu) + float(reg_lam(lam)) + float(reg_mu(mu))

for it in range(num_outer):
    print(f"MLAA outer {it + 1:03}/{num_outer:03}", end="\r")

    # --- activity update: 1 penalised OSEM epoch (attenuation fixed) ---
    for k in range(num_subsets):
        a_k = xp.exp(-proj_nt_k[k](mu))[..., None]  # subset attenuation factors
        ybar = a_k * proj_k[k](lam) + s_k[k]
        grad = proj_k[k].adjoint(a_k * (y_k[k] / ybar - 1.0))  # emission gradient
        sens = proj_k[k].adjoint(a_k * xp.ones_like(ybar))  # A^T 1 (attenuated)
        g_pen = grad - reg_lam.gradient(lam) / num_subsets
        # harmonic-mean preconditioner: 1 / (sens/lam + prior curvature)
        D = _safe(lam, sens + lam * prior_curv_lam / num_subsets, fov_mask)
        lam = xp.clip(lam + D * g_pen, 0, None)

    # --- attenuation update: OS-MLTR with blank = P lam (lam fixed) ---
    # the activity forward projection is the transmission "blank scan"
    blank_k = [proj_k[k](lam) for k in range(num_subsets)]
    for _ in range(num_mltr_epochs):
        for k in range(num_subsets):
            a_k = xp.exp(-proj_nt_k[k](mu))[..., None]
            psi = a_k * blank_k[k]  # attenuated activity projection
            ybar = psi + s_k[k]
            # per-TOF-bin MLTR weights, summed over the TOF axis
            w_num = xp.sum(psi / ybar * (ybar - y_k[k]), axis=-1)
            w_den = xp.sum(psi**2 / ybar, axis=-1)
            grad = proj_nt_k[k].adjoint(w_num) - reg_mu.gradient(mu) / num_subsets
            denom = (
                proj_nt_k[k].adjoint(Pnt1_k[k] * w_den) + prior_curv_mu / num_subsets
            )
            # mu is estimated only inside the object support
            mu = xp.clip(mu + _safe(grad, denom, support), 0, None)
        # fix the global scale ambiguity: anchor the known-water region
        mu = mu * (mu_water / float(xp.mean(mu[water_roi])))

    cost[it + 1] = emission_neg_logL(lam, mu) + float(reg_lam(lam)) + float(reg_mu(mu))
print()
print(f"final penalised cost = {cost[-1]:.2f}")

# %%
# Convergence of the joint objective
# ----------------------------------

fig0, ax0 = plt.subplots(figsize=(5, 4), tight_layout=True)
ax0.plot(cost)
ax0.set_xlabel("outer iteration")
ax0.set_ylabel(r"$-L + \beta_\lambda R(\lambda) + \beta_\mu R(\mu)$")
ax0.set_title("MLAA joint objective")
ax0.grid(ls=":")
fig0.show()

# %%
# Comparison: ground truth vs. 0th-order / baseline vs. MLAA
# ----------------------------------------------------------
#
# Top row -- attenuation: truth, the 0th-order water blob, and the MLAA
# estimate (which recovers the dense / air inserts).  Bottom row --
# activity: truth, OS-MLEM with the fixed 0th-order attenuation (artefacts
# where the true attenuation differs from water), and the MLAA activity.
# Because the activity and attenuation phantoms have different inserts,
# little crosstalk means each MLAA image shows only its own structure.

sl = img_shape[2] // 2
vmax_mu = 2.5 * mu_water
vmax_lam = float(xp.max(act_true))


def _show(ax, vol, vmax, title):
    h = ax.imshow(
        to_numpy_array(vol[:, :, sl]).T,
        origin="lower",
        cmap="Greys",
        vmin=0,
        vmax=vmax,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return h


fig, ax = plt.subplots(2, 3, figsize=(11, 7.5), layout="constrained")
_show(ax[0, 0], mu_true, vmax_mu, r"true $\mu$")
_show(ax[0, 1], mu0, vmax_mu, r"0th-order $\mu$ (water blob)")
h_mu = _show(ax[0, 2], mu, vmax_mu, r"MLAA $\mu$")
fig.colorbar(h_mu, ax=ax[0, :], fraction=0.04, location="right")

_show(ax[1, 0], act_true, vmax_lam, r"true activity")
_show(ax[1, 1], lam_ac, vmax_lam, r"OS-MLEM (0th-order $\mu$)")
h_lam = _show(ax[1, 2], lam, vmax_lam, r"MLAA activity")
fig.colorbar(h_lam, ax=ax[1, :], fraction=0.04, location="right")
fig.show()

plt.show()
