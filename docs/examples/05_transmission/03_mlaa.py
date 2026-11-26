"""
Joint activity and attenuation reconstruction (MLAA) for TOF PET
================================================================

:bdg-primary:`transmission` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

Maximum Likelihood Activity and Attenuation (MLAA) jointly estimates the
activity image :math:`\\lambda` **and** the attenuation image :math:`\\mu`
from a single TOF emission scan, without a separate transmission or CT
measurement.  It alternates a penalised ordered-subset **emission** update
of the activity (OS-MAPEM) with a penalised ordered-subset **transmission**
update of the attenuation (OS-MAPTR), in which the activity forward
projection plays the role of the blank scan.  Without TOF, activity and
attenuation cannot be separated; with TOF they are determined up to a single
global scale factor :footcite:p:`Rezaei2012`, which we fix with a region of
known attenuation.

**You will learn**

- how to model TOF emission data with an attenuation factor computed by a
  separate non-TOF projector, and a resolution model for the activity only,
- how to segment the object support from a quick non-attenuation-corrected
  reconstruction,
- how to interleave penalised OS-MAPEM (activity) and OS-MAPTR (attenuation)
  subset updates with log-cosh priors and separable majorizer
  preconditioners,
- how to fix the global scale ambiguity of MLAA,
- how MLAA compares with attenuation correction based on a uniform water
  image and with the true attenuation image.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py` (penalised
transmission reconstruction, the attenuation block of MLAA),
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py` and
:ref:`sphx_glr_auto_examples_03_algorithms_01_run_sgd_svrg.py` (ordered-subset
emission updates, the activity block of MLAA), and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF projectors).  The two MLAA blocks are essentially these two algorithms
applied in alternation.

.. note::

    This example is **not executed** in the documentation build because of
    its run time (several minutes on the CPU: TOF reconstructions with many
    subsets and outer iterations on a large scanner geometry).  Run it
    locally, ideally on a GPU backend.
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
from scipy.ndimage import binary_fill_holes, gaussian_filter, label

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    poisson_transmission_terms,
    show_vol_cuts,
    suggest_array_backend_and_device,
)
from parallelproj.functions import C2AffineObjective, LogCosh

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Reconstruction settings
# --------------------------
#
# ``num_outer`` outer iterations each sweep once through all activity subsets;
# every activity subset update is followed by
# ``num_att_updates_per_act_update`` attenuation subset updates, because MLTR
# converges more slowly than MLEM.  Both images get an edge-preserving
# log-cosh prior (section 7).

num_subsets = 28  # ordered view subsets (divides the 168 views evenly)
num_outer = 10  # MLAA outer iterations
# attenuation (OS-MAPTR) subset updates per activity (OS-MAPEM) subset update
# (MLTR converges slower than MLEM)
num_att_updates_per_act_update = 5
scatter_fraction = 0.6  # contamination relative to mean true emission
count_factor = 5.0  # scales the activity (sets the count level / noise)
support_threshold = 0.5  # body segmentation: fraction of the smoothed-NAC mean
psf_fwhm = 6.0  # mm, emission image-based resolution model (Gaussian PSF)

mu_water = 0.0096  # 1/mm at 511 keV

# edge-preserving log-cosh prior weights (preconditioner as in 02_run_maptr)
beta_lam = 0.01  # activity prior weight
beta_mu = 10.0  # attenuation prior weight
delta_mu = mu_water / 2  # mu edges (inserts) >> delta are preserved
# delta_lam (the activity log-cosh scale) is derived from the warm-start below

# %%
# 2. TOF emission model, scanner and projectors
# ---------------------------------------------
#
# The TOF emission model is
#
# .. math::
#     \bar{y}_{i,t}(\lambda, \mu) = \bar z_{i,t}(\lambda, \mu) + \bar s_{i,t},
#     \qquad
#     \bar z_{i,t}(\lambda, \mu) = a_i(\mu) \, (P_\text{tof} B \lambda)_{i,t},
#     \qquad
#     a_i(\mu) = e^{-(P_\text{nt}\,\mu)_i},
#
# where :math:`\bar z_{i,t}` is the expected (attenuated, resolution-blurred)
# emission contribution to TOF bin :math:`t` of LOR :math:`i`,
# :math:`\bar y_{i,t}` the expected data after adding the **expected**
# contamination :math:`\bar s_{i,t}`, :math:`P_\text{tof}` is the **TOF
# emission projector**, :math:`B` is an image-based Gaussian **resolution
# model** (PSF) applied to the activity :math:`\lambda`, and
# :math:`P_\text{nt}` is the **non-TOF** projector used for the attenuation
# line integrals.  The attenuation factor :math:`a_i` is the same for every
# TOF bin :math:`t` of a given LOR :math:`i` and carries **no** resolution
# model -- the PET resolution loss (positron range, non-collinearity,
# detector response) blurs the apparent *activity*, not the bulk attenuation
# of the medium.  Finally :math:`\bar s` is the strictly positive
# **expected** contamination (mean scatter + randoms) -- an expectation, not
# a noisy realisation; here it is assumed known and fixed (see the warning in
# section 4).
#
# For implementation convenience the TOF projector :math:`P_\text{tof}` and
# the resolution model :math:`B` are composed into a single linear operator
# :math:`A = P_\text{tof} B` (a :class:`.CompositeLinearOperator`); its
# transpose :math:`B^T P_\text{tof}^T` is then assembled automatically, so
# the code uses ``A`` and ``A.adjoint`` wherever the equations write
# :math:`P_\text{tof} B` and :math:`B^T P_\text{tof}^T`.
#
# We use the large scanner of the previous example (radius 300 mm, 56 sides,
# 3 rings) and an image of 100 x 100 x 3 voxels of 4 x 4 x 5.3 mm.  The TOF
# projector has 51 TOF bins of 10 mm and a TOF resolution of 30 mm FWHM
# (200 ps).

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
    num_tofbins=51, tofbin_width=10.0, sigma_tof=30.0 / 2.3548
)

fov_mask = proj_nt.fov_mask()

# Image-based Gaussian resolution model (PSF) -- the operator B above -- for
# the *emission* path only.  Composing it with the TOF projector into a
# single operator means the transpose (used in every activity update) is
# assembled automatically in the right order -- no chance of forgetting B^T.
# The attenuation path keeps the bare geometric non-TOF projector (no PSF).
psf_sigma = tuple(psf_fwhm / 2.3548 / vs for vs in voxel_size)  # voxels
B = parallelproj.operators.GaussianFilterOperator(img_shape, sigma=psf_sigma)
A = parallelproj.operators.CompositeLinearOperator([proj, B])

# %%
# 3. Activity and attenuation phantoms with different inserts
# -----------------------------------------------------------
#
# The activity uses the standard elliptic cylinder phantom (its hot and cold
# inserts).  The attenuation is a water cylinder with the same outline and
# its **own** inserts at other locations: a dense, bone-like insert
# (:math:`\mu` = 0.02 / mm) and an air-like insert (:math:`\mu` = 0.002 / mm)
# on the ``x0`` axis.  Because the two images have different inserts, the
# reconstructions reveal how well MLAA separates them (little crosstalk =
# each image shows only its own structure).

activity_phantom = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)
act_true = count_factor * activity_phantom

cyl = to_numpy_array(activity_phantom) > 0  # body outline (shared support)
nx, ny, _ = img_shape
i0, i1 = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
dense = (i0 - int(0.72 * nx)) ** 2 + (i1 - ny // 2) ** 2 < 8**2  # bone-like
air = (i0 - int(0.28 * nx)) ** 2 + (i1 - ny // 2) ** 2 < 8**2  # lung-like
dense3 = np.repeat(dense[:, :, None], num_rings, axis=2) & cyl
air3 = np.repeat(air[:, :, None], num_rings, axis=2) & cyl

mu_np = np.where(cyl, mu_water, 0.0)
mu_np = np.where(dense3, 0.02, mu_np)  # dense insert
mu_np = np.where(air3, 0.002, mu_np)  # air-like insert
mu_true = xp.asarray(mu_np.astype(np.float32), device=dev)

# %%
# 4. Simulated TOF emission data
# ------------------------------
#
# The noise-free data are the attenuated, PSF-blurred TOF forward projection
# of the activity plus a constant contamination of 0.6 times its mean.  The
# attenuation factors :math:`a = e^{-P_\text{nt}\mu}` have no TOF axis and are
# broadcast over the TOF bins.
#
# .. warning::
#     For simplicity the expected contamination :math:`\bar s` (scatter +
#     randoms) is treated as **known and fixed** throughout.  This is not
#     realistic: the scatter distribution depends on *both* the activity
#     :math:`\lambda` **and** the attenuation :math:`\mu`, so a real MLAA
#     pipeline must **re-estimate it iteratively** (e.g. a single-scatter
#     simulation refreshed as :math:`\lambda` and :math:`\mu` evolve).
#     Holding it fixed here -- and using the known :math:`\bar s` already in
#     the warm start with the 0th-order attenuation image (section 7) -- is a
#     deliberate idealisation that keeps the example focused on the joint
#     activity/attenuation update itself.

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
# 5. Subsets and the likelihood
# -----------------------------
#
# MLAA maximises the penalised emission Poisson log-likelihood
#
# .. math::
#
#     \Phi(\lambda,\mu) = L(\lambda,\mu)
#     - \beta_\lambda R(\lambda) - \beta_\mu R(\mu),
#     \qquad
#     L(\lambda,\mu) = \sum_{i,t}\big(
#     y_{i,t}\,\log \bar y_{i,t}(\lambda,\mu) - \bar y_{i,t}(\lambda,\mu)
#     \big)
#
# (:math:`L` is the Poisson log-likelihood up to a constant independent of
# :math:`\lambda,\mu`; the priors :math:`R` follow in section 7).  The code
# evaluates the penalised **cost** :math:`-\Phi` (to be minimised).
#
# The ordered-subset updates use subset operators: the views are distributed
# over ``num_subsets`` subsets; ``A_k`` (TOF projector composed with the PSF)
# and ``proj_nt_k`` (non-TOF) project only the views of subset :math:`k`.
# ``Pnt1_k`` are the non-TOF subset projections of a uniform image (for the
# MLTR curvature), and :math:`G` is the finite-difference operator of the
# priors (not to be confused with the PSF :math:`B`).

subset_views, subset_slices = lor_desc.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)  # 4D (TOF) slices
)

proj_nt_k = []  # non-TOF subset projectors (attenuation)
A_k = []  # subset emission operators: TOF projector composed with the PSF
for k in range(num_subsets):
    p = copy(proj)
    p.views = subset_views[k]
    A_k.append(parallelproj.operators.CompositeLinearOperator([p, B]))
    q = copy(proj_nt)
    q.views = subset_views[k]
    proj_nt_k.append(q)

y_k = [y[subset_slices[k]] for k in range(num_subsets)]
s_k = [s[subset_slices[k]] for k in range(num_subsets)]

ones_img = xp.ones(img_shape, dtype=xp.float32, device=dev)
# subset attenuation sensitivity
Pnt1_k = [proj_nt_k[k](ones_img) for k in range(num_subsets)]

# finite-difference operator G of the edge-preserving prior (NOT the PSF B)
G = parallelproj.operators.FiniteForwardDifference(img_shape)


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
# 6. Object support from a non-attenuation-corrected OSEM image
# -------------------------------------------------------------
#
# One OSEM epoch without an attenuation model and without scatter modelling
# (scatter is typically not available before the attenuation is known) gives
# a fast, high-contrast, attenuation-biased (NAC) activity image used only to
# segment the object support.  A tiny constant (0.01), added to the expected
# and the measured data, keeps the ratios finite.

lam = xp.where(fov_mask, ones_img, xp.zeros_like(ones_img))
for k in range(num_subsets):
    ybar = A_k[k](lam) + 1e-2
    sens = A_k[k].adjoint(xp.ones_like(ybar))
    update = A_k[k].adjoint((y_k[k] + 1e-2) / ybar)
    lam = _safe(lam * update, sens, fov_mask)
print(f"NAC OSEM done (lam max = {float(xp.max(lam)):.1f})")

# %%
# Because the NAC reconstruction omits the contamination, it has high
# contrast and noisy background "junk", so a plain threshold is not robust.
# We instead:
#
# 1. smooth in-plane to suppress background noise,
# 2. threshold relative to the mean object activity,
# 3. keep only the largest connected component (drops background islands),
# 4. fill interior holes per slice, giving a solid water blob.
#
# The support defines the 0th-order attenuation image :math:`\mu_0` (uniform
# water inside the support).  A small central region, away from the inserts,
# is the water calibration region that fixes the global scale of MLAA.

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
# 7. Warm start and log-cosh priors
# ---------------------------------
#
# One OS-MLEM epoch *with* the 0th-order attenuation produces a correctly
# scaled (attenuation-corrected) activity image.  Its level sets the
# activity log-cosh scale ``delta_lam`` -- a far better basis than the
# mis-scaled NAC image -- and it serves as the common warm start of the
# OS-MAPEM baseline, the reference and MLAA.
#
# Each penalty :math:`R` is an edge-preserving **log-cosh** roughness prior
# on the nearest-neighbour finite differences :math:`G x` of the image (the
# same prior and preconditioner as in
# :ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py`):
#
# .. math::
#
#     R(x) = \delta \sum_d \sum_j
#     \log\cosh\!\Big(\frac{(G x)_{d,j}}{\delta}\Big),
#     \qquad
#     \nabla R(x) = G^T \tanh(G x / \delta),
#
# where :math:`x` is :math:`\lambda` or :math:`\mu`, the sum runs over the
# difference directions :math:`d` and voxels :math:`j`, and :math:`\delta`
# -- :math:`\delta_\lambda` or :math:`\delta_\mu` -- is the
# edge-preservation scale: differences :math:`\gg \delta` are penalised
# roughly linearly (edges preserved), :math:`\ll \delta` quadratically (noise
# smoothed).
#
# The prior enters the **harmonic-mean preconditioners** of section 10
# through the curvature :math:`c_R(x)` of its **separable quadratic
# majorizer** (``reg.separable_majorizer_curvature(x)``, De Pierro's
# surrogate :math:`c_R = |G|^T(\omega(Gx) \odot |G|\mathbf 1)` with Huber's
# curvature :math:`\omega(t) = \psi'(t)/t` of
# :math:`\psi(t) = \delta \log\cosh(t/\delta)`, see
# :meth:`.C2Function.separable_majorizer_curvature`), which combines with
# the data (sensitivity) curvature.  It adapts to the image and, unlike the
# maximal Hessian diagonal :math:`\operatorname{diag}(G^TG)/\delta`, it is a
# majorizer of the prior.  ``reg_lam`` and ``reg_mu`` include the weights
# :math:`\beta`; the helper functions return the curvature per subset
# (divided by :math:`m`).

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


def prior_curv_lam(lam: Array) -> Array:
    """separable majorizer curvature of beta_lam R at lam, per subset"""
    return reg_lam.separable_majorizer_curvature(lam) / num_subsets


def prior_curv_mu(mu: Array) -> Array:
    """separable majorizer curvature of beta_mu R at mu, per subset"""
    return reg_mu.separable_majorizer_curvature(mu) / num_subsets


def penalised_cost(lam: Array, mu: Array) -> float:
    """Penalised cost -Phi = -L + beta_lam R(lam) + beta_mu R(mu)."""
    return emission_neg_logL(lam, mu) + float(reg_lam(lam)) + float(reg_mu(mu))


# %%
# 8. Baseline: OS-MAPEM with the 0th-order attenuation image
# ----------------------------------------------------------
#
# We reconstruct the activity with attenuation correction based on the crude
# uniform-water :math:`\mu_0` (held fixed, no joint estimation), with the
# same penalised OS-MAPEM update as the activity block of MLAA (section 10)
# and as many epochs as MLAA has outer iterations.  Wherever the true
# attenuation differs from water (the dense and air-like inserts), this
# baseline shows attenuation correction artefacts that MLAA should remove.

lam_ac = lam_warm  # start from the warm (attenuation-corrected) activity
for it in range(num_outer):
    print(f"OSEM (mu0) epoch {it + 1:03}/{num_outer:03}", end="\r")
    for k in range(num_subsets):
        ybar = a0_k[k] * A_k[k](lam_ac) + s_k[k]
        grad = A_k[k].adjoint(a0_k[k] * (y_k[k] / ybar - 1.0))
        sens = A_k[k].adjoint(a0_k[k] * xp.ones_like(ybar))
        g_pen = grad - reg_lam.gradient(lam_ac) / num_subsets
        D = _safe(lam_ac, sens + lam_ac * prior_curv_lam(lam_ac), fov_mask)
        lam_ac = xp.clip(lam_ac + D * g_pen, 0, None)
print()

# %%
# 9. Reference: OS-MAPEM with the true attenuation image
# ------------------------------------------------------
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
        D = _safe(lam_ref, sens + lam_ref * prior_curv_lam(lam_ref), fov_mask)
        lam_ref = xp.clip(lam_ref + D * g_pen, 0, None)
print()

# %%
# 10. MLAA: interleaved OS-MAPEM and OS-MAPTR updates
# ---------------------------------------------------
#
# MLAA maximises :math:`\Phi` by alternating two preconditioned
# gradient-ascent **block** updates -- one for :math:`\lambda` (activity,
# :math:`\mu` held fixed) and one for :math:`\mu` (attenuation,
# :math:`\lambda` held fixed).  In the equations below, operators
# (:math:`P_\text{tof}`, :math:`B`, :math:`P_\text{nt}` and their transposes)
# act on whole arrays; :math:`\odot` and :math:`\oslash` denote elementwise
# (Hadamard) product and division; :math:`a = e^{-P_\text{nt}\mu}` is per-LOR
# and broadcasts over the TOF axis; :math:`\bar z = a \odot
# (P_\text{tof} B \lambda)` and :math:`\bar y = \bar z + \bar s` are the array
# (elementwise) forms of :math:`\bar z_{i,t}` and :math:`\bar y_{i,t}`;
# :math:`\Sigma_t` sums over the TOF axis; and :math:`m` is the number of
# (ordered-view) subsets.  Each update operates on a **single subset**
# :math:`k` (one of :math:`m`): :math:`P_\text{tof}^{(k)}` and
# :math:`P_\text{nt}^{(k)}` are the emission and attenuation projectors
# restricted to the LORs of subset :math:`k`, and :math:`y^{(k)}`,
# :math:`\bar s^{(k)}`, :math:`a^{(k)}`,
# :math:`\bar z^{(k)} = a^{(k)} \odot (P_\text{tof}^{(k)} B \lambda)` and
# :math:`\bar y^{(k)} = \bar z^{(k)} + \bar s^{(k)}` the corresponding
# subset sinograms.  The :math:`1/m` factor distributes the penalty gradient
# evenly across the :math:`m` subsets, so one full sweep applies it once.
#
# Differentiating :math:`L` restricted to the LORs of subset :math:`k` gives
# the two **subset log-likelihood gradients** that drive the updates:
#
# .. math::
#
#     \nabla_\lambda L^{(k)} = B^T (P_\text{tof}^{(k)})^T\big[
#     a^{(k)} \odot (y^{(k)} \oslash \bar y^{(k)} - \mathbf 1)\big],
#
# .. math::
#
#     \nabla_\mu L^{(k)} = (P_\text{nt}^{(k)})^T g^{(k)},
#     \qquad
#     g^{(k)}_i = \Sigma_t\, \frac{\bar z^{(k)}_{i,t}}{\bar y^{(k)}_{i,t}}
#     (\bar y^{(k)}_{i,t} - y^{(k)}_{i,t}) .
#
# The activity back projection :math:`B^T (P_\text{tof}^{(k)})^T` already
# sums over the TOF axis, so :math:`\nabla_\lambda L^{(k)}` needs no
# intermediate sinogram; the attenuation gradient first forms the TOF-summed
# per-LOR residual :math:`g^{(k)}` and then back-projects it through the
# non-TOF projector.
#
# * **activity** (:math:`\mu` fixed) -- a penalised OSEM step driven by
#   :math:`\nabla_\lambda L^{(k)}`:
#
#   .. math::
#
#       \lambda \leftarrow \Big[\lambda + D_\lambda^{(k)} \odot \big(
#       \nabla_\lambda L^{(k)}
#       - \tfrac{\beta_\lambda}{m}\nabla R(\lambda)\big)\Big]_+,
#
#   .. math::
#
#       D_\lambda^{(k)} = \lambda \oslash \big(
#       B^T (P_\text{tof}^{(k)})^T a^{(k)}
#       + \tfrac{\beta_\lambda}{m}\,\lambda \odot c_R(\lambda)\big) .
#
# * **attenuation** (:math:`\lambda` fixed) -- a penalised **OS-MAPTR** step
#   driven by :math:`\nabla_\mu L^{(k)}`: the transmission update of
#   :ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py` with the
#   *blank scan* replaced by the activity forward projection
#   :math:`P_\text{tof}^{(k)} B \lambda`, and restricted to the object
#   support:
#
#   .. math::
#
#       \mu \leftarrow \Big[\mu + D_\mu^{(k)} \odot \big(
#       \nabla_\mu L^{(k)} - \tfrac{\beta_\mu}{m}\nabla R(\mu)\big)\Big]_+,
#
#   .. math::
#
#       D_\mu^{(k)} = \mathbf 1 \oslash \big(
#       (P_\text{nt}^{(k)})^T\big[(P_\text{nt}^{(k)}\mathbf 1) \odot c^{(k)}\big]
#       + \tfrac{\beta_\mu}{m}\,c_R(\mu)\big),
#       \qquad
#       c^{(k)}_i = \Sigma_t\, \frac{(\bar z^{(k)}_{i,t})^2}{\bar y^{(k)}_{i,t}},
#
#   where :math:`c^{(k)}` is the MLTR / SPS separable curvature (as in
#   :ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py`).
#
# .. note::
#     **Exact TOF gradient vs. the TOF-summed approximation.**  The gradient
#     sinogram :math:`g^{(k)}_i` above forms the per-TOF-bin residual
#     :math:`\tfrac{\bar z^{(k)}_{i,t}}{\bar y^{(k)}_{i,t}}
#     (\bar y^{(k)}_{i,t} - y^{(k)}_{i,t})` and only **then** sums over the
#     TOF axis (:math:`\Sigma_t`) -- i.e. :math:`(P_\text{nt}^{(k)})^T g^{(k)}`
#     is the *exact* gradient of the subset-:math:`k` TOF log-likelihood with
#     respect to :math:`\mu`.  The MLTR update in :footcite:t:`Rezaei2012`
#     (their Eq. 6) instead sums :math:`\bar z`, :math:`\bar s` and :math:`y`
#     over TOF **first** and runs a non-TOF transmission step on the
#     resulting TOF-summed sinogram.  Because the attenuation factor
#     :math:`a_i` is identical for every TOF bin, that TOF-summed sinogram is
#     a valid non-TOF transmission measurement; the two gradients coincide
#     for a single TOF bin or when the contamination vanishes
#     (:math:`\bar s = 0`), and differ slightly otherwise (a sum of ratios
#     :math:`\sum_t \bar z_{i,t} y_{i,t}/\bar y_{i,t}` vs. a ratio of sums).
#     Summing first is cheaper -- it operates on the much smaller non-TOF
#     sinogram -- whereas the exact per-TOF form used here uses the full TOF
#     information at slightly higher cost.
#
# The two blocks are interleaved at the **subset** level: every activity
# subset update is immediately followed by ``num_att_updates_per_act_update``
# attenuation subset updates, so the two images improve together rather than
# in separate full passes.  Over one outer iteration this amounts to one
# activity pass (``num_subsets`` updates) and
# ``num_att_updates_per_act_update`` attenuation passes.  The attenuation
# "blank scan" is recomputed from the just-updated activity for every
# attenuation update.
#
# **Why TOF is essential.**  Non-TOF MLAA is ill-posed: activity and
# attenuation trade off against each other (crosstalk), and the joint problem
# is non-unique.  TOF data determine :math:`\lambda` and :math:`\mu` up to a
# single global scalar :footcite:p:`Rezaei2012`, which we fix by anchoring
# the water calibration region to :math:`\mu_{\text{water}}` after each
# block of attenuation updates.
#
# **Two further practical ingredients used here:**
#
# * the attenuation :math:`\mu` is updated **only inside the object support**
#   (section 6); estimating :math:`\mu` in the surrounding air /
#   low-sensitivity region makes the joint problem unstable;
# * both images carry an edge-preserving **log-cosh** prior with the
#   harmonic-mean (data + prior curvature) preconditioner of
#   :ref:`sphx_glr_auto_examples_05_transmission_02_run_maptr.py`.
#
# The activity starts at the warm start, the attenuation at the 0th-order
# water blob :math:`\mu_0`.  Every intermediate estimate is kept to
# visualise the convergence.

lam = lam_warm  # activity initialised at the warm-start
mu = mu0  # attenuation initialised at the 0th-order water blob

lam_hist = [lam]
mu_hist = [mu]

att_k = 0  # persistent attenuation subset pointer (cycles through subsets)
for it in range(num_outer):
    print(f"MLAA outer {it + 1:03}/{num_outer:03}", end="\r")

    for ka in range(num_subsets):
        # --- 1 activity (OS-MAPEM) subset update (attenuation fixed) ---
        a_k = xp.exp(-proj_nt_k[ka](mu))[..., None]
        ybar = a_k * A_k[ka](lam) + s_k[ka]
        grad = A_k[ka].adjoint(a_k * (y_k[ka] / ybar - 1.0))
        # B^T P_tof^T (a * 1), attenuated sensitivity
        sens = A_k[ka].adjoint(a_k * xp.ones_like(ybar))
        g_pen = grad - reg_lam.gradient(lam) / num_subsets
        # harmonic-mean preconditioner: 1 / (sens/lam + prior curvature)
        D = _safe(lam, sens + lam * prior_curv_lam(lam), fov_mask)
        lam = xp.clip(lam + D * g_pen, 0, None)

        # --- num_att_updates_per_act_update attenuation (OS-MAPTR) updates ---
        # the transmission update with the blank scan replaced by the current
        # activity forward projection P lam (TOF terms summed over TOF bins)
        for _ in range(num_att_updates_per_act_update):
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
            denom = proj_nt_k[kt].adjoint(Pnt1_k[kt] * curv_sino) + prior_curv_mu(mu)
            # mu is estimated only inside the object support
            mu = xp.clip(mu + _safe(grad, denom, support), 0, None)

        # fix the global scale ambiguity: anchor the known-water region
        mu = mu * (mu_water / float(xp.mean(mu[water_roi])))

    lam_hist.append(lam)
    mu_hist.append(mu)
print()

# %%
# 11. Penalised cost: MLAA vs. the true-attenuation reference
# -----------------------------------------------------------
#
# We compare the *full* penalised cost
# :math:`-\Phi = -L + \beta_\lambda R(\lambda) + \beta_\mu R(\mu)` (lower is
# better) of the MLAA solution with that of the OS-MAPEM reference that used
# the true attenuation.  MLAA estimates :math:`\mu` jointly, so on noisy data
# it can even reach a slightly *lower* cost -- the meaningful question is
# whether the images themselves are correct (see the next figure).

print(f"penalised cost  OS-MAPEM (true mu): {penalised_cost(lam_ref, mu_true):.2f}")
print(f"penalised cost  MLAA              : {penalised_cost(lam, mu):.2f}")

# %%
# 12. Comparison of the images
# ----------------------------
#
# Each column pairs the attenuation image used or estimated (top) with the
# resulting activity image (bottom), in the central transaxial slice: ground
# truth; the 0th-order water blob with the OS-MAPEM baseline; the MLAA joint
# estimate; and the true attenuation with the OS-MAPEM reference (gold
# standard).  Each row uses one grey scale.

sl = img_shape[2] // 2
vmax_mu = 2.5 * mu_water
vmax_lam = float(xp.max(act_true))


def _show(ax, vol, vmax, title):
    """central transaxial slice of vol on a grey scale from 0 to vmax"""
    h = ax.imshow(to_numpy_array(vol[:, :, sl]).T, cmap="Greys", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    return h


fig, ax = plt.subplots(2, 4, figsize=(10, 5.4), layout="constrained")
_show(ax[0, 0], mu_true, vmax_mu, r"true $\mu$")
_show(ax[0, 1], mu0, vmax_mu, r"0th-order $\mu$ (water blob)")
_show(ax[0, 2], mu, vmax_mu, r"MLAA $\mu$")
h_mu = _show(ax[0, 3], mu_true, vmax_mu, r"true $\mu$ (reference)")
fig.colorbar(h_mu, ax=ax[0, :], fraction=0.04, location="right", label="1/mm")

_show(ax[1, 0], act_true, vmax_lam, r"true activity")
_show(ax[1, 1], lam_ac, vmax_lam, r"OS-MAPEM (0th-order $\mu$)")
_show(ax[1, 2], lam, vmax_lam, r"MLAA activity")
h_lam = _show(ax[1, 3], lam_ref, vmax_lam, r"OS-MAPEM (true $\mu$)")
fig.colorbar(h_lam, ax=ax[1, :], fraction=0.04, location="right")
fig.show()

# %%
# INTERPRETATION_IMAGES

# %%
# 13. Convergence of the MLAA estimates
# -------------------------------------
#
# The intermediate attenuation and activity estimates are stacked into 4D
# arrays (leading axis = outer iteration, 0 = initial images);
# ``show_vol_cuts`` adds a slider over that axis so the convergence can be
# stepped through.  With a non-interactive matplotlib backend, only the
# middle outer iteration is shown.

mu_hist_4d = np.stack([to_numpy_array(m) for m in mu_hist])
lam_hist_4d = np.stack([to_numpy_array(li) for li in lam_hist])
hist_axis_labels = ("outer iteration", "x0", "x1", "x2")

fig_mu, _, mu_widgets = show_vol_cuts(
    mu_hist_4d,
    voxel_size=voxel_size,
    axis_labels=hist_axis_labels,
    fig_title=r"MLAA $\mu$ vs. outer iteration",
    vmin=0,
    vmax=vmax_mu,
)
fig_lam, _, lam_widgets = show_vol_cuts(
    lam_hist_4d,
    voxel_size=voxel_size,
    axis_labels=hist_axis_labels,
    fig_title=r"MLAA activity vs. outer iteration",
    vmin=0,
    vmax=vmax_lam,
)
plt.show()

# %%
# Summary
# -------
#
# - MLAA estimates activity and attenuation from TOF emission data alone by
#   alternating a penalised OS-MAPEM update of :math:`\lambda` and a
#   penalised OS-MAPTR update of :math:`\mu`, in which the activity forward
#   projection replaces the blank scan.
# - The attenuation factors come from a non-TOF projector without
#   resolution model and are broadcast over the TOF bins; the attenuation
#   gradient sums the exact per-TOF-bin residuals.
# - TOF determines both images up to a global scale, fixed here by a region
#   of known (water) attenuation; :math:`\mu` is only estimated inside an
#   object support segmented from a non-attenuation-corrected image.
# - Both blocks use log-cosh priors and preconditioners that add the
#   separable majorizer curvature of the prior to the data curvature.
# - The contamination was assumed known and fixed; a real MLAA pipeline must
#   re-estimate the scatter as :math:`\lambda` and :math:`\mu` evolve.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_06_torch_01_run_projection_layer.py`
#    -- parallelproj projectors as layers of PyTorch models.
#
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_path = "_static/thumbnails/03_mlaa.png"
