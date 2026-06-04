"""
TOF vs non-TOF: variance reduction in a uniform cylinder
=========================================================

Why TOF reduces image noise
----------------------------

In a standard (non-TOF) PET scan each detected coincidence event tells us
that an annihilation occurred *somewhere* along a line of response (LOR),
but gives no information about *where* along it.

Time-of-flight (TOF) PET additionally measures the small difference in
arrival times of the two 511 keV photons and uses that difference to
localise the annihilation along the LOR to a Gaussian probability kernel:

.. math::

    h(\\ell) = \\frac{1}{\\sqrt{2\\pi}\\,\\sigma_\\text{TOF}}
               \\exp\\!\\left(-\\frac{\\ell^2}{2\\sigma_\\text{TOF}^2}\\right),
    \\qquad
    \\sigma_\\text{TOF} = \\frac{c}{2} \\cdot \\frac{\\Delta t_\\text{FWHM}}{2.355}

where :math:`\\ell` is the distance from the LOR midpoint and
:math:`\\Delta t_\\text{FWHM}` is the scanner's coincidence timing
resolution (CTR).  A CTR of 200 ps corresponds to a spatial FWHM of
≈ 30 mm.

It is known that TOF reduces the variance in the center of a 2D cylinder
with diameter :math:`D`, where the SNR gain is approximately

.. math::

    G_\\text{TOF} \\approx \\sqrt{0.66 \\frac{D}{\\text{FWHM}_\\text{TOF}}}.

For :math:`D = 240` mm and :math:`\\text{FWHM} = 30` mm this gives
:math:`G \\approx 2.0`, i.e. a roughly two-fold noise reduction at the
centre.

The convergence-speed trap
---------------------------

TOF reconstruction also **converges faster** than non-TOF reconstruction.
This creates a common pitfall: if both reconstructions are stopped at the
*same* (small) number of epochs, the TOF image may appear *noisier*
than the non-TOF image — not because TOF is worse, but because TOF has
already converged past its low-noise plateau while non-TOF is still
climbing.  Conversely, at very early epochs non-TOF may look
smoother simply because it has not yet amplified the noise.

To observe the *true* asymptotic advantage of TOF one must run **both**
algorithms long enough to reach (approximate) convergence.

What this example shows
------------------------

* A single-ring 2-D scanner with a uniform circular phantom.
* **SVRG** (:func:`00_run_mlem_osem_svrg`) with ``num_subsets=28``
  subsets run for ``num_epochs=10`` epochs (warm-started by a single
  OSEM epoch), applied independently to the non-TOF and TOF forward
  models.  10 SVRG epochs are sufficient for both to reach their
  respective noise plateaux.
* The standard deviation inside a small (25 mm-radius) central ROI is
  tracked after every epoch — this is a fast single-realisation proxy
  for the true noise level that avoids the need for Monte Carlo repeats.
* Eight-panel figure:

  - **Top-left**: std.dev curves vs. SVRG epoch (epoch 0 = OSEM warm
    start) — shows faster TOF convergence *and* the lower asymptotic
    noise level.
  - **Bottom-left**: ratio non-TOF / TOF std.dev — values > 1 confirm
    that non-TOF is noisier; the ratio stabilises above 1 once both
    algorithms have converged.
  - **2nd column**: smoothed images after the OSEM warm start (epoch 0).
  - **3rd column**: smoothed images after 1 SVRG epoch, illustrating
    that at very early iterations TOF may appear noisier (it has already
    amplified noise while non-TOF is still initialisation-smooth).
  - **Right column**: final smoothed images after ``num_epochs`` SVRG
    epochs for visual comparison of the asymptotic noise levels.

.. note::

    The standard deviation in a single noise realisation is used here as
    a proxy for the true noise standard deviation.  For a uniform phantom
    spatial variability inside the ROI equals the noise variability, so
    the single realisation is sufficient.
"""

# %%
from __future__ import annotations
from collections.abc import Sequence
import matplotlib.pyplot as plt
import numpy as np

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array
from parallelproj.functions import NegPoissonLogL, C2AffineObjective, C1Function

from copy import copy

# %%
from array_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)

# %%
# Key simulation parameters
# -------------------------
#
# ``num_epochs`` controls how many MLEM iterations are stored.  700 is
# enough for both non-TOF and TOF to be well past their respective
# convergence knees, so the asymptotic noise levels are clearly visible.
#
# ``fwhm_tof_mm = 30 mm`` corresponds to a coincidence timing resolution
# of approximately 200 ps — representative of state-of-the-art clinical
# scanners as of 2025.
#
# ``sm_fwhm_mm`` is the FWHM of the Gaussian post-filter applied after
# every iteration.  A mild 9 mm filter is applied to suppress
# high-frequency "salt-and-pepper" noise while preserving the
# convergence-related noise trend.
#
# ``count_factor`` scales the phantom activity to control the total number
# of detected events.  Moderate counts (0.3) give a clearly visible noise
# difference between TOF and non-TOF.
#
# ``cylinder_radius`` (in voxels) defines the uniform phantom disk.

num_subsets = 14
num_epochs = 10
fwhm_tof_mm = 30.0
fwhm_res_model_mm = 4.0
sm_fwhm_mm = 9.0
cylinder_radius_mm = 120
count_factor = 0.3
step_size = 2.0

# %%
# Scanner and image geometry
# --------------------------
#
# We use a **single-ring scanner** (``num_rings=1``) so that the
# reconstruction is effectively 2-D.  This keeps computation fast and
# isolates the transaxial TOF effect without axial compression artefacts.
#
# The scanner radius of 300 mm and 28 × 16 = 448 detector elements give a
# realistic clinical-scale geometry.  The single image plane has
# 151 × 151 × 1 voxels of 2 mm side length, yielding a 302 mm transaxial
# field of view.

num_rings = 1
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=300.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.asarray([0], dtype=xp.float32, device=dev),
    symmetry_axis=2,
)

# %%
# LOR descriptor and projectors
# -----------------------------
#
# The :class:`.RegularPolygonPETLORDescriptor` maps detector pairs to
# sinogram bins.  ``max_ring_difference=2`` is harmless here (single ring)
# and ``radial_trim=150`` discards the outermost radial bins that fall
# outside the cylinder FOV.

img_shape = (151, 151, 1)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=150,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj_non_tof = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# %%
# Uniform cylinder phantom
# ------------------------
#
# A disk of radius ``cylinder_radius`` voxels centred in the
# FOV with uniform activity ``count_factor``.

x_pos = voxel_size[0] * (
    xp.arange(img_shape[0], device=dev, dtype=xp.float32) - img_shape[0] / 2 + 0.5
)
X, Y = xp.meshgrid(x_pos, x_pos, indexing="ij")
RHO = xp.sqrt(X**2 + Y**2)

x_true = xp.ones(img_shape, device=dev, dtype=xp.float32)
x_true[..., 0] = count_factor * (RHO <= cylinder_radius_mm)

# %%
# Attenuation model
# -----------------
#
# A uniform water-equivalent attenuation coefficient of
# :math:`\mu = 0.01\,\text{mm}^{-1}` is used inside the cylinder.
# Attenuation is the same for TOF and non-TOF; it is included here for
# realism and has no bearing on the variance comparison.

x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
att_sino = xp.exp(-proj_non_tof(x_att))

# %%
# TOF projector setup
# -------------------
#
# The TOF projector is a copy of the non-TOF projector with
# :class:`.TOFParameters` attached.  The bin width is set to
# :math:`\text{FWHM}/4` so that each TOF kernel spans approximately
# 4 bins (good Gaussian sampling).  The total number of bins is chosen to
# cover an FOV of (300 mm) with some margin.
#
# Both forward operators also include the same image-based resolution model
# (:class:`.GaussianFilterOperator`, FWHM = 4 mm) to model finite detector
# resolution.

proj_tof = copy(proj_non_tof)

proj_tof.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=int(300 / (fwhm_tof_mm / 4.0)) + 1,
    tofbin_width=fwhm_tof_mm / 4.0,
    sigma_tof=fwhm_tof_mm / 2.35,
)

# For TOF, att_sino has no TOF-bins dimension while the projector output does.
# broadcast_to adds a trailing singleton via expand_dims and broadcasts it over
# the TOF-bins axis without copying data (zero-stride view).
att_values_tof = xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj_tof.out_shape)
att_op_tof = parallelproj.operators.ElementwiseMultiplicationOperator(att_values_tof)
att_op_non_tof = parallelproj.operators.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.operators.GaussianFilterOperator(
    img_shape,
    sigma=[fwhm_res_model_mm / (2.35 * float(vs)) for vs in proj_tof.voxel_size],
)

# compose all 3 operators into a single linear operator
pet_lin_op_tof = parallelproj.operators.CompositeLinearOperator(
    (att_op_tof, proj_tof, res_model)
)

# setup non-TOF fwd model
pet_lin_op_non_tof = parallelproj.operators.CompositeLinearOperator(
    (att_op_non_tof, proj_non_tof, res_model)
)

# %%
# Data simulation
# ---------------
#
# The TOF sinogram is simulated once and a constant scatter/randoms
# contamination (50 % of the mean prompt rate) is added before Poisson
# sampling.
#
# The non-TOF sinogram is obtained by **summing the noisy TOF sinogram over
# its TOF-bin axis**.  This marginalisation is mathematically equivalent to
# discarding the timing information in a real scanner, and it ensures that
# both reconstructions see exactly the same Poisson noise realisation —
# they differ only in how much of the timing information they exploit.

noise_free_data_tof = pet_lin_op_tof(x_true)

contamination_tof = xp.full(
    noise_free_data_tof.shape,
    0.5 * float(xp.mean(noise_free_data_tof)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data_tof += contamination_tof

np.random.seed(1)
y_tof = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data_tof)),
    device=dev,
    dtype=xp.float32,
)

# marginalise: sum over TOF bins gives the non-TOF sinogram
y_non_tof = xp.sum(y_tof, axis=-1)
contamination_non_tof = xp.sum(contamination_tof, axis=-1)

# %%
# Post-filter and subset splitting
# ---------------------------------
#
# A mild Gaussian post-filter is applied after each SVRG epoch so that the
# stored image matches the typical clinical workflow.
#
# The sinogram views are split into ``num_subsets`` disjoint groups.
# Non-TOF data and the attenuation sinogram are 3-D (R × V × P); TOF data
# adds a fourth TOF-bin axis.  We therefore request 3-D slices for
# attenuation / non-TOF data indexing and 4-D slices for TOF data indexing.

sm_op = parallelproj.operators.GaussianFilterOperator(
    in_shape=img_shape, sigma=sm_fwhm_mm / (2.35 * voxel_size[0])
)

# 3-D slices: used for non-TOF data *and* to index att_sino
subset_views, subset_slices_nt = lor_desc.get_distributed_views_and_slices(
    num_subsets, 3
)
# 4-D slices: used to index TOF data and contamination
_, subset_slices_tof = lor_desc.get_distributed_views_and_slices(num_subsets, 4)

# %%
# SVRG helper functions
# ---------------------
#
# These two functions implement SVRG exactly as in
# :ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`.
# ``svrg_calc_snapshot_gradients`` computes and stores all per-subset
# gradients at the current anchor point; ``svrg_update`` performs a single
# variance-reduced subset step.


def em_update(
    x_cur: Array,
    data_fidelity: C1Function,
    adj_ones: Array,
) -> Array:
    """EM update (preconditioned gradient step) used for the warm start."""
    return x_cur - (x_cur / adj_ones) * data_fidelity.gradient(x_cur)


def svrg_calc_snapshot_gradients(
    x_cur: Array,
    subset_obj_functions: Sequence[C1Function],
) -> tuple[Array, Array]:
    """Compute and store per-subset gradients at the anchor point."""
    m = len(subset_obj_functions)
    stored = xp.zeros((m,) + x_cur.shape, dtype=x_cur.dtype, device=dev)
    for k, df in enumerate(subset_obj_functions):
        stored[k] = df.gradient(x_cur)
    return stored, xp.sum(stored, axis=0)


def svrg_update(
    x_cur: Array,
    subset_idx: int,
    subset_obj_functions: Sequence[C1Function],
    stored_grads: Array,
    full_grad: Array,
    precond: Array,
    step_size: float = 1.0,
) -> Array:
    """Single variance-reduced subset update."""
    m = len(subset_obj_functions)
    grad_k = subset_obj_functions[subset_idx].gradient(x_cur)
    approx_grad = m * (grad_k - stored_grads[subset_idx]) + full_grad
    return xp.clip(x_cur - step_size * precond * approx_grad, 0, None)


# %%
# Non-TOF: subset operators, warm start, and SVRG
# ------------------------------------------------
#
# One :class:`.CompositeLinearOperator` is built per subset, combining
# the subset projector, the attenuation diagonal, and the resolution model.
# Sensitivity images :math:`(A^k)^H \mathbf{1}` are pre-computed once and
# summed to obtain the full :math:`A^H \mathbf{1}`.
#
# The warm start runs a single OSEM epoch, which moves the initial flat
# image close enough to the solution for the SVRG preconditioner to be
# meaningful from the very first epoch.

proj_non_tof.clear_cached_lor_endpoints()
subset_linops_nt = []
for i in range(num_subsets):
    sp = copy(proj_non_tof)
    sp.views = subset_views[i]
    att_op_k = parallelproj.operators.ElementwiseMultiplicationOperator(
        att_sino[subset_slices_nt[i]]
    )
    subset_linops_nt.append(
        parallelproj.operators.CompositeLinearOperator([att_op_k, sp, res_model])
    )

subset_adj_ones_nt = xp.zeros((num_subsets,) + img_shape, dtype=xp.float32, device=dev)
for k, op in enumerate(subset_linops_nt):
    subset_adj_ones_nt[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )
adjoint_ones_nt = xp.sum(subset_adj_ones_nt, axis=0)

subset_fidelities_nt = [
    C2AffineObjective(
        NegPoissonLogL(y_non_tof[subset_slices_nt[k]]),
        subset_linops_nt[k],
        contamination_non_tof[subset_slices_nt[k]],
    )
    for k in range(num_subsets)
]

# --- warm start: 1 OSEM epoch ---
x_nt = count_factor * xp.ones(img_shape, device=dev, dtype=xp.float32)
for k in range(num_subsets):
    print(f"  non-TOF warm-start subset {k + 1:03}/{num_subsets:03}", end="\r")
    x_nt = em_update(x_nt, subset_fidelities_nt[k], subset_adj_ones_nt[k])
print()
x_nt_warmstart_sm = sm_op(x_nt)

# --- SVRG loop ---
recons_non_tof = xp.ones((num_epochs,) + img_shape, device=dev, dtype=xp.float32)
svrg_precond_nt = x_nt / adjoint_ones_nt
stored_grads_nt, full_grad_nt = None, None

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond_nt = x_nt / adjoint_ones_nt
        stored_grads_nt, full_grad_nt = svrg_calc_snapshot_gradients(
            x_nt, subset_fidelities_nt
        )
        x_nt = xp.clip(x_nt - svrg_precond_nt * full_grad_nt, 0, None)

    for k in range(num_subsets):
        print(
            f"  non-TOF SVRG epoch {epoch + 1:02}/{num_epochs:02},"
            f" subset {k + 1:03}/{num_subsets:03}",
            end="\r",
        )
        x_nt = svrg_update(
            x_nt,
            k,
            subset_fidelities_nt,
            stored_grads_nt,
            full_grad_nt,
            svrg_precond_nt,
            step_size=step_size,
        )
    recons_non_tof[epoch, ...] = sm_op(x_nt)
print()

# %%
# TOF: subset operators, warm start, and SVRG
# --------------------------------------------
#
# Identical structure to the non-TOF case.  The only differences are:
#
# * Each subset attenuation operator must broadcast ``att_sino`` over the
#   TOF-bin axis (zero-copy via :func:`xp.broadcast_to`).
# * Data and contamination are sliced with 4-D ``subset_slices_tof``.
#
# Because the TOF forward model localises each event to ≈ 30 mm along the
# LOR rather than the full ≈ 600 mm chord, every gradient step is more
# informative and the algorithm reaches its noise floor in fewer epochs.

proj_tof.clear_cached_lor_endpoints()
subset_linops_tof = []
for i in range(num_subsets):
    sp = copy(proj_tof)
    sp.views = subset_views[i]
    att_values_k = xp.broadcast_to(
        xp.expand_dims(att_sino[subset_slices_nt[i]], axis=-1), sp.out_shape
    )
    att_op_k = parallelproj.operators.ElementwiseMultiplicationOperator(att_values_k)
    subset_linops_tof.append(
        parallelproj.operators.CompositeLinearOperator([att_op_k, sp, res_model])
    )

subset_adj_ones_tof = xp.zeros((num_subsets,) + img_shape, dtype=xp.float32, device=dev)
for k, op in enumerate(subset_linops_tof):
    subset_adj_ones_tof[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )
adjoint_ones_tof = xp.sum(subset_adj_ones_tof, axis=0)

subset_fidelities_tof = [
    C2AffineObjective(
        NegPoissonLogL(y_tof[subset_slices_tof[k]]),
        subset_linops_tof[k],
        contamination_tof[subset_slices_tof[k]],
    )
    for k in range(num_subsets)
]

# --- warm start: 1 OSEM epoch ---
x_tof = count_factor * xp.ones(img_shape, device=dev, dtype=xp.float32)
for k in range(num_subsets):
    print(f"  TOF warm-start subset {k + 1:03}/{num_subsets:03}", end="\r")
    x_tof = em_update(x_tof, subset_fidelities_tof[k], subset_adj_ones_tof[k])
print()
x_tof_warmstart_sm = sm_op(x_tof)

# --- SVRG loop ---
recons_tof = xp.ones((num_epochs,) + img_shape, device=dev, dtype=xp.float32)
svrg_precond_tof = x_tof / adjoint_ones_tof
stored_grads_tof, full_grad_tof = None, None

for epoch in range(num_epochs):
    if epoch % 2 == 0:
        if epoch <= 4:
            svrg_precond_tof = x_tof / adjoint_ones_tof
        stored_grads_tof, full_grad_tof = svrg_calc_snapshot_gradients(
            x_tof, subset_fidelities_tof
        )
        x_tof = xp.clip(x_tof - svrg_precond_tof * full_grad_tof, 0, None)

    for k in range(num_subsets):
        print(
            f"  TOF SVRG epoch {epoch + 1:02}/{num_epochs:02},"
            f" subset {k + 1:03}/{num_subsets:03}",
            end="\r",
        )
        x_tof = svrg_update(
            x_tof,
            k,
            subset_fidelities_tof,
            stored_grads_tof,
            full_grad_tof,
            svrg_precond_tof,
            step_size=step_size,
        )
    recons_tof[epoch, ...] = sm_op(x_tof)
print()


# %%
# Noise vs. iteration in the central ROI
# ---------------------------------------
#
# The standard deviation of voxel values inside a small 25 mm-radius
# central ROI is used as a single-realisation proxy for the true noise
# standard deviation.  Because the phantom is uniform, every voxel inside
# the ROI has the same expected value, so spatial variability equals noise
# variability.
#
# Two effects are visible in the plot:
#
# 1. **Faster convergence of TOF**: the TOF std.dev curve rises steeply and
#    then falls to its asymptote in far fewer iterations than non-TOF.
#    At early iteration counts TOF can therefore appear *noisier* — not
#    because TOF is worse, but because it has already amplified noise while
#    non-TOF is still initialisation-smooth.
#
# 2. **Lower asymptotic noise for TOF**: once both curves have stabilised,
#    the TOF std.dev is clearly below the non-TOF std.dev.  The ratio plot
#    (bottom-left) shows this: values > 1 confirm the TOF advantage, and
#    the ratio continues to grow as both algorithms converge.

roi_std_non_tof = np.array([float(x[:, :, 0][RHO < 25].std()) for x in recons_non_tof])
roi_std_tof = np.array([float(x[:, :, 0][RHO < 25].std()) for x in recons_tof])
# prepend warm-start (epoch 0) so the x-axis starts at 0
roi_std_non_tof = np.concatenate(
    [[float(x_nt_warmstart_sm[:, :, 0][RHO < 25].std())], roi_std_non_tof]
)
roi_std_tof = np.concatenate(
    [[float(x_tof_warmstart_sm[:, :, 0][RHO < 25].std())], roi_std_tof]
)
epochs = np.arange(0, num_epochs + 1)  # 0 = OSEM warm start

# %%
# Visualisation
# -------------
#
# The eight-panel figure summarises the comparison:
#
# * **Top-left**: std.dev in the central 25 mm ROI vs. SVRG epoch (epoch 0
#   is the OSEM warm start) for non-TOF (orange) and TOF (blue).  Note how
#   TOF rises *and falls* faster; comparing at a fixed early epoch can give
#   the wrong conclusion.
# * **Bottom-left**: ratio of std.devs (non-TOF / TOF).  The ratio
#   increases with epoch count and stabilises above 1, quantifying the
#   asymptotic noise advantage of TOF.
# * **2nd column**: smoothed warm-start images (epoch 0).
# * **3rd column**: smoothed images after 1 SVRG epoch.  At this early
#   stage TOF may look noisier than non-TOF because it has converged further.
# * **Right column**: final smoothed images after ``num_epochs`` SVRG
#   epochs.  Visual noise in the uniform disk is lower for TOF.

ims = dict(vmin=0, vmax=xp.max(recons_non_tof), cmap="Greys")

fig, ax = plt.subplots(2, 4, figsize=(12, 6), layout="constrained", sharex="col")
ax[0, 0].plot(epochs, roi_std_non_tof, label="non-TOF", color="tab:orange")
ax[0, 0].plot(
    epochs, roi_std_tof, label=f"TOF ({fwhm_tof_mm:.0f} mm FWHM)", color="tab:blue"
)
ax[0, 0].legend(fontsize=8)
ax[1, 0].plot(epochs, roi_std_non_tof / roi_std_tof, color="tab:green")
ax[1, 0].axhline(1.0, color="gray", ls=":", lw=0.8)
ax[1, 0].set_xlabel(f"SVRG epoch ({num_subsets} subsets)")
ax[0, 0].set_ylabel("std.dev in central ROI")
ax[1, 0].set_ylabel("std.dev ratio  (non-TOF / TOF)")
ax[0, 0].set_title(
    f"(central 25 mm ROI, {sm_fwhm_mm:.0f} mm post-filter)",
    fontsize=8,
)
ax[0, 0].grid(ls=":")
ax[1, 0].grid(ls=":")

# warm-start images (epoch 0)
ax[0, 1].imshow(to_numpy_array(x_nt_warmstart_sm[:, :, 0]), **ims)
ax[1, 1].imshow(to_numpy_array(x_tof_warmstart_sm[:, :, 0]), **ims)
ax[0, 1].set_title(
    f"non-TOF  (epoch 0)\nstd.dev = {roi_std_non_tof[0]:.4f}",
    fontsize=8,
)
ax[1, 1].set_title(
    f"TOF {fwhm_tof_mm:.0f} mm  (epoch 0)\nstd.dev = {roi_std_tof[0]:.4f}",
    fontsize=8,
)

# epoch 1 images (index 0 in recons arrays)
ax[0, 2].imshow(to_numpy_array(recons_non_tof[0, :, :, 0]), **ims)
ax[1, 2].imshow(to_numpy_array(recons_tof[0, :, :, 0]), **ims)
ax[0, 2].set_title(
    f"non-TOF  (epoch 1)\nstd.dev = {roi_std_non_tof[1]:.4f}",
    fontsize=8,
)
ax[1, 2].set_title(
    f"TOF {fwhm_tof_mm:.0f} mm  (epoch 1)\nstd.dev = {roi_std_tof[1]:.4f}",
    fontsize=8,
)

# final-epoch images
ax[0, 3].imshow(to_numpy_array(recons_non_tof[-1, :, :, 0]), **ims)
ax[1, 3].imshow(to_numpy_array(recons_tof[-1, :, :, 0]), **ims)
ax[0, 3].set_title(
    f"non-TOF  (epoch {num_epochs})\nstd.dev = {roi_std_non_tof[-1]:.4f}",
    fontsize=8,
)
ax[1, 3].set_title(
    f"TOF {fwhm_tof_mm:.0f} mm  (epoch {num_epochs})\nstd.dev = {roi_std_tof[-1]:.4f}",
    fontsize=8,
)

for a in ax[:, 1:].flat:
    a.set_axis_off()

fig.suptitle(
    f"TOF variance reduction - uniform cylinder Ø {2*cylinder_radius_mm:.0f} mm - 9mm post filter",
    fontsize=9,
)
fig.show()
