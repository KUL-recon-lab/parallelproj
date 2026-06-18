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

where :math:`P` is the **TOF** emission projector -- here composed with an
image-based Gaussian **resolution model** (PSF), so :math:`P\\lambda` really
means (TOF projector) :math:`\\circ` (PSF) applied to :math:`\\lambda`.
:math:`P_\\text{nt}` is the **non-TOF** projector used for the attenuation
line integrals: the attenuation factor :math:`a_i` is the same for every TOF
bin :math:`t` of a given LOR :math:`i` and carries **no** resolution model --
the PET resolution loss (positron range, non-collinearity, detector
response) blurs the apparent *activity*, not the bulk attenuation of the
medium.  Finally :math:`s` is a strictly positive contamination (scatter +
randoms); here it is assumed known and fixed (see the warning below).

MLAA alternates two block updates of the penalised log-likelihood
:math:`L(\\lambda,\\mu) - \\beta_\\lambda R(\\lambda) - \\beta_\\mu R(\\mu)`:

* **activity** (fix :math:`\\mu`): penalised OSEM with the attenuation in
  the system matrix -- preconditioned gradient ascent with the
  EM/harmonic-mean preconditioner :math:`D_\\lambda = \\lambda /
  (A^T\\mathbf 1 + \\lambda\\,\\beta_\\lambda\\kappa/\\delta_\\lambda)`;
* **attenuation** (fix :math:`\\lambda`): penalised **OS-MAPTR** -- this is
  exactly the transmission reconstruction of the ``05_transmission``
  examples, with the *blank scan* replaced by the current activity forward
  projection :math:`b = P\\lambda` (and, with TOF, the per-TOF-bin MLTR
  weights summed over the TOF axis before back-projecting through the
  non-TOF projector).

The two blocks are interleaved at the **subset** level: every activity
subset update is immediately followed by ``num_mltr_epochs`` attenuation
subset updates, so the two images improve together rather than in separate
full passes (the total number of activity and attenuation updates is
unchanged).

**Why TOF is essential.**  Non-TOF MLAA is ill-posed: activity and
attenuation trade off against each other (crosstalk), and the joint problem
is non-unique.  TOF data determine :math:`\\lambda` and :math:`\\mu` up to a
single global scalar :footcite:p:`Rezaei2012`, which we fix by anchoring a region of
known (water) attenuation after each attenuation update.

**Two further practical ingredients used here:**

* the attenuation :math:`\\mu` is updated **only inside the object support**
  (obtained by thresholding the quick non-attenuation-corrected activity
  image); estimating :math:`\\mu` in the surrounding air/low-sensitivity
  region makes the joint problem unstable;
* both images carry an edge-preserving **log-cosh** prior with the
  harmonic-mean (data + prior curvature) preconditioner of
  ``05_transmission/02_run_maptr.py``.

The activity and attenuation phantoms have **different** inserts on purpose,
so the reconstructions reveal how well MLAA separates the two (little
crosstalk = each image shows only its own structure).

.. note::
    This example uses the larger transmission-scanner geometry and is
    deliberately not part of the rendered gallery (no ``run_`` prefix): the
    TOF reconstruction with many subsets and outer iterations is slow on the
    CPU.  Run it locally, ideally on a GPU backend.

.. warning::
    For simplicity the contamination :math:`s` (scatter + randoms) is treated
    as **known and fixed** throughout.  This is not realistic: the scatter
    distribution depends on *both* the activity :math:`\\lambda` **and** the
    attenuation :math:`\\mu`, so a real MLAA pipeline must **re-estimate it
    iteratively** (e.g. a single-scatter simulation refreshed as
    :math:`\\lambda` and :math:`\\mu` evolve).  Holding it fixed here -- and
    reusing the known :math:`s` in the non-attenuation-corrected warm-start
    while omitting attenuation -- is a deliberate idealisation that keeps the
    example focused on the joint activity/attenuation update itself.

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

from scipy.ndimage import binary_fill_holes, gaussian_filter, label

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import Array, to_numpy_array
from parallelproj.functions import C2AffineObjective, LogCosh

from example_utils import (
    elliptic_cylinder_phantom,
    poisson_transmission_terms,
    show_vol_cuts,
)

# %%
from example_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")
xp, dev = suggest_array_backend_and_device(None, None)

# %%
num_subsets = 28  # ordered view subsets (divides the 168 views evenly)
num_outer = 10  # MLAA outer iterations
num_mltr_epochs = 5  # OS-MAPTR updates per OS-MAPEM updates (MLTR is slower than MLEM)
scatter_fraction = 0.6  # contamination relative to mean true emission
count_factor = 5.0  # scales the activity (sets the count level / noise)
support_threshold = 0.5  # body segmentation: fraction of the smoothed-NAC mean
psf_fwhm = 6.0  # mm, emission image-based resolution model (Gaussian PSF)

mu_water = 0.0096  # 1/mm at 511 keV

# edge-preserving log-cosh prior weights (harmonic-mean preconditioner as in
# 02_maptr)
beta_lam = 0.01  # activity prior weight
beta_mu = 10.0  # attenuation prior weight
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
# TOF projector for the activity; 51 bins x 10 mm = 510 mm cover the ~510 mm
# LORs, FWHM 30 mm (200ps)
proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=51, tofbin_width=10.0, sigma_tof=30.0 / 2.355
)

fov_mask = proj_nt.fov_mask()

# Image-based Gaussian resolution model (PSF) for the *emission* path only.
# Composing it with the TOF projector into a single operator means the
# adjoint (used in every activity update) is assembled automatically in the
# right order -- no chance of forgetting R^T.  The attenuation path keeps the
# bare geometric non-TOF projector (no PSF; see the module docstring).
psf_sigma = tuple(psf_fwhm / 2.355 / vs for vs in voxel_size)  # voxels
res_model = parallelproj.operators.GaussianFilterOperator(img_shape, sigma=psf_sigma)
A = parallelproj.operators.CompositeLinearOperator([proj, res_model])

# %%
# Ground-truth activity and attenuation -- DIFFERENT insert patterns
# ------------------------------------------------------------------
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
# --------------------------

att_true = xp.exp(-proj_nt(mu_true))  # (R, V, P) attenuation factors
emis_true = att_true[..., None] * A(act_true)  # PSF-blurred, broadcast over TOF
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

proj_nt_k = []  # non-TOF subset projectors (attenuation)
A_k = []  # subset emission operators: TOF projector composed with the PSF
for k in range(num_subsets):
    p = copy(proj)
    p.views = subset_views[k]
    A_k.append(parallelproj.operators.CompositeLinearOperator([p, res_model]))
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
    ybar = xp.exp(-proj_nt(mu))[..., None] * A(lam) + s
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
# One OSEM epoch without an attenuation model and without scatter modelling
# (scatter is typically not available before the attenuation is known) gives
# a fast, high-contrast, attenuation-biased activity image used only to
# segment the object support.  A tiny constant keeps the ratios finite.

lam = xp.where(fov_mask, ones_img, xp.zeros_like(ones_img))
for k in range(num_subsets):
    # ybar = A_k[k](lam) + s_k[k]
    ybar = A_k[k](lam) + 1e-2
    sens = A_k[k].adjoint(xp.ones_like(ybar))
    # update = A_k[k].adjoint(y_k[k] / ybar)
    update = A_k[k].adjoint((y_k[k] + 1e-2) / ybar)
    lam = _safe(lam * update, sens, fov_mask)
print(f"NAC OSEM done (lam max = {float(xp.max(lam)):.1f})")


# Segment the body support from the NAC image.  Because the NAC
# reconstruction omits the scatter contamination it has high contrast and
# noisy background "junk", so a plain threshold is not robust.  We instead:
#   1. smooth in-plane to suppress background noise,
#   2. threshold relative to the mean object activity,
#   3. keep only the largest connected component (drops background islands),
#   4. fill interior holes per slice -> a solid water blob.
nac_smooth = gaussian_filter(to_numpy_array(lam), sigma=(1.0, 1.0, 0.0))
mask = nac_smooth > support_threshold * float(nac_smooth[nac_smooth > 0].mean())

labels, n_labels = label(mask)  # connected components (background = 0)
if n_labels > 0:
    largest = 1 + int(np.argmax(np.bincount(labels.ravel())[1:]))
    mask = labels == largest

support_np = np.stack(
    [binary_fill_holes(mask[:, :, z]) for z in range(mask.shape[2])], axis=2
)
support = xp.asarray(support_np, device=dev) & fov_mask

# 0th-order attenuation image: uniform water inside the filled support
mu0 = xp.where(support, xp.asarray(mu_water, dtype=xp.float32), xp.zeros_like(lam))

# small central water-attenuation calibration region (away from the inserts)
water_roi_np = np.zeros(img_shape, dtype=bool)
water_roi_np[nx // 2 - 6 : nx // 2 + 6, ny // 2 - 6 : ny // 2 + 6, :] = True
water_roi = xp.asarray(water_roi_np, device=dev) & support

# %%
# Warm-start activity (with the 0th-order attenuation) and log-cosh priors
# ------------------------------------------------------------------------
#
# One OS-MLEM epoch *with* the 0th-order attenuation produces a correctly
# scaled (attenuation-corrected) activity image.  Its level sets the
# activity log-cosh scale ``delta_lam`` -- a far better basis than the
# mis-scaled NAC image -- and serves as the common warm start for both the
# OSEM baseline and MLAA.

a0_k = [xp.exp(-proj_nt_k[k](mu0))[..., None] for k in range(num_subsets)]
lam_warm = lam  # NAC activity
for k in range(num_subsets):
    ybar = a0_k[k] * A_k[k](lam_warm) + s_k[k]
    sens = A_k[k].adjoint(a0_k[k] * xp.ones_like(ybar))
    update = A_k[k].adjoint(a0_k[k] * y_k[k] / ybar)
    lam_warm = _safe(lam_warm * update, sens, fov_mask)

delta_lam = 0.3 * float(xp.mean(lam_warm[lam_warm > 0]))
reg_lam = C2AffineObjective(LogCosh(delta=delta_lam, beta=beta_lam), G)
reg_mu = C2AffineObjective(LogCosh(delta=delta_mu, beta=beta_mu), G)
prior_curv_lam = beta_lam * kappa / delta_lam
prior_curv_mu = beta_mu * kappa / delta_mu


def penalised_cost(lam: Array, mu: Array) -> float:
    """Penalised joint objective Phi = -L + beta_lam R(lam) + beta_mu R(mu)."""
    return emission_neg_logL(lam, mu) + float(reg_lam(lam)) + float(reg_mu(mu))


# %%
# Baseline: OS-MAPEM activity with the fixed 0th-order attenuation image
# ----------------------------------------------------------------------
#
# Reconstruct the activity with attenuation correction based on the crude
# uniform-water :math:`\mu_0` (held fixed, no joint estimation).  Wherever
# the true attenuation differs from water (the dense / air inserts), this
# baseline shows attenuation-correction artefacts that MLAA removes.

lam_ac = lam_warm  # start from the warm (attenuation-corrected) activity
for it in range(num_outer):
    print(f"OSEM (mu0) epoch {it + 1:03}/{num_outer:03}", end="\r")
    for k in range(num_subsets):
        ybar = a0_k[k] * A_k[k](lam_ac) + s_k[k]
        grad = A_k[k].adjoint(a0_k[k] * (y_k[k] / ybar - 1.0))
        sens = A_k[k].adjoint(a0_k[k] * xp.ones_like(ybar))
        g_pen = grad - reg_lam.gradient(lam_ac) / num_subsets
        D = _safe(lam_ac, sens + lam_ac * prior_curv_lam / num_subsets, fov_mask)
        lam_ac = xp.clip(lam_ac + D * g_pen, 0, None)
print()

# %%
# Reference: OS-MAPEM activity with the TRUE attenuation image
# ------------------------------------------------------------
#
# The activity we would reconstruct if the attenuation were known exactly --
# the gold standard against which MLAA is judged.

aT_k = [xp.exp(-proj_nt_k[k](mu_true))[..., None] for k in range(num_subsets)]
lam_ref = lam_warm  # start from the warm (attenuation-corrected) activity
for it in range(num_outer):
    print(f"OSEM (true mu) epoch {it + 1:03}/{num_outer:03}", end="\r")
    for k in range(num_subsets):
        ybar = aT_k[k] * A_k[k](lam_ref) + s_k[k]
        grad = A_k[k].adjoint(aT_k[k] * (y_k[k] / ybar - 1.0))
        sens = A_k[k].adjoint(aT_k[k] * xp.ones_like(ybar))
        g_pen = grad - reg_lam.gradient(lam_ref) / num_subsets
        D = _safe(lam_ref, sens + lam_ref * prior_curv_lam / num_subsets, fov_mask)
        lam_ref = xp.clip(lam_ref + D * g_pen, 0, None)
print()

# %%
# MLAA: interleaved penalised OS-MAPEM (activity) and OS-MAPTR (attenuation)
# --------------------------------------------------------------------------

lam = lam_warm  # activity initialised at the warm-start
mu = mu0  # attenuation initialised at the 0th-order water blob

# keep every intermediate estimate to visualise the convergence
lam_hist = [lam]
mu_hist = [mu]

# Updates are interleaved at the *subset* level: each activity (OS-MAPEM)
# subset update is immediately followed by ``num_mltr_epochs`` attenuation
# (OS-MAPTR) subset updates.  Over one outer iteration this still amounts to
# one activity pass (``num_subsets`` updates) and ``num_mltr_epochs``
# attenuation passes, but the two images now improve in lock-step.  The
# attenuation "blank scan" is the activity forward projection ``P lam``,
# recomputed from the just-updated activity for every attenuation update.

att_k = 0  # persistent attenuation subset pointer (cycles through subsets)
for it in range(num_outer):
    print(f"MLAA outer {it + 1:03}/{num_outer:03}", end="\r")

    for ka in range(num_subsets):
        # --- 1 activity (OS-MAPEM) subset update (attenuation fixed) ---
        a_k = xp.exp(-proj_nt_k[ka](mu))[..., None]
        ybar = a_k * A_k[ka](lam) + s_k[ka]
        grad = A_k[ka].adjoint(a_k * (y_k[ka] / ybar - 1.0))
        sens = A_k[ka].adjoint(a_k * xp.ones_like(ybar))  # A^T 1 (attenuated)
        g_pen = grad - reg_lam.gradient(lam) / num_subsets
        # harmonic-mean preconditioner: 1 / (sens/lam + prior curvature)
        D = _safe(lam, sens + lam * prior_curv_lam / num_subsets, fov_mask)
        lam = xp.clip(lam + D * g_pen, 0, None)

        # --- num_mltr_epochs attenuation (OS-MAPTR) subset updates ---
        # the transmission update with the blank scan replaced by the current
        # activity forward projection P lam (TOF terms summed over TOF bins)
        for _ in range(num_mltr_epochs):
            kt = att_k % num_subsets
            att_k += 1
            _, grad_sino, curv_sino = poisson_transmission_terms(
                proj_nt_k[kt](mu),
                blank=A_k[kt](lam),
                contamination=s_k[kt],
                data=y_k[kt],
                tof_sum=True,
            )
            grad = proj_nt_k[kt].adjoint(grad_sino) - reg_mu.gradient(mu) / num_subsets
            denom = (
                proj_nt_k[kt].adjoint(Pnt1_k[kt] * curv_sino)
                + prior_curv_mu / num_subsets
            )
            # mu is estimated only inside the object support
            mu = xp.clip(mu + _safe(grad, denom, support), 0, None)

        # fix the global scale ambiguity: anchor the known-water region
        mu = mu * (mu_water / float(xp.mean(mu[water_roi])))

    lam_hist.append(lam)
    mu_hist.append(mu)
print()

# %%
# Final penalised objective: MLAA vs. the true-attenuation reference
# ------------------------------------------------------------------
#
# Compare the *full* penalised cost
# :math:`\Phi = -L + \beta_\lambda R(\lambda) + \beta_\mu R(\mu)` of the MLAA
# solution with that of the OS-MAPEM reference that used the true attenuation.
# MLAA estimates :math:`\mu` jointly, so on noisy data it can even reach a
# slightly *lower* :math:`\Phi` -- the meaningful question is whether the
# images themselves are correct (see the comparison figure).

print(f"penalised cost  OS-MAPEM (true mu): {penalised_cost(lam_ref, mu_true):.2f}")
print(f"penalised cost  MLAA              : {penalised_cost(lam, mu):.2f}")

# %%
# Comparison: ground truth vs. 0th-order / baseline vs. MLAA
# ----------------------------------------------------------
#
# Each column pairs the attenuation used/estimated (top) with the resulting
# activity (bottom): ground truth; the 0th-order water blob; the MLAA joint
# estimate; and the true-attenuation reference (gold standard).  Because the
# activity and attenuation phantoms have different inserts, little crosstalk
# means each MLAA image shows only its own structure.

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


fig, ax = plt.subplots(2, 4, figsize=(14, 7.5), layout="constrained")
_show(ax[0, 0], mu_true, vmax_mu, r"true $\mu$")
_show(ax[0, 1], mu0, vmax_mu, r"0th-order $\mu$ (water blob)")
_show(ax[0, 2], mu, vmax_mu, r"MLAA $\mu$")
h_mu = _show(ax[0, 3], mu_true, vmax_mu, r"true $\mu$ (reference)")
fig.colorbar(h_mu, ax=ax[0, :], fraction=0.04, location="right")

_show(ax[1, 0], act_true, vmax_lam, r"true activity")
_show(ax[1, 1], lam_ac, vmax_lam, r"OS-MAPEM (0th-order $\mu$)")
_show(ax[1, 2], lam, vmax_lam, r"MLAA activity")
h_lam = _show(ax[1, 3], lam_ref, vmax_lam, r"OS-MAPEM (true $\mu$)")
fig.colorbar(h_lam, ax=ax[1, :], fraction=0.04, location="right")
fig.show()

# %%
# Convergence of the MLAA estimates over the outer iterations
# -----------------------------------------------------------
#
# The intermediate attenuation and activity estimates are stacked into 4D
# arrays (leading axis = outer iteration); ``show_vol_cuts`` adds a slider
# over that axis so the convergence can be stepped through.

mu_hist_4d = np.stack([to_numpy_array(m) for m in mu_hist])
lam_hist_4d = np.stack([to_numpy_array(li) for li in lam_hist])

fig_mu = show_vol_cuts(
    mu_hist_4d,
    voxel_size=voxel_size,
    fig_title=r"MLAA $\mu$ vs. outer iteration",
    vmin=0,
    vmax=vmax_mu,
)
fig_lam = show_vol_cuts(
    lam_hist_4d,
    voxel_size=voxel_size,
    fig_title=r"MLAA activity vs. outer iteration",
    vmin=0,
    vmax=vmax_lam,
)

plt.show()

# %%
# .. rubric:: References
#
# .. footbibliography::
