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

It is know that TOF reduces the variance in the center of a 2D cylinder with
diameter :math:`D`, where the SNR gain is given by

.. math::

    G_\\text{TOF} \\approx \\sqrt{0.66 \\frac{D}{\\text{FWHM}_\\text{TOF}}}.

For :math:`D = 240` mm and :math:`\\text{FWHM} = 30` mm this gives
:math:`G \\approx 2.0`, i.e. a roughly three-fold noise reduction at the
centre.

The convergence-speed trap
---------------------------

TOF reconstruction also **converges faster** than non-TOF MLEM.
This creates a common pitfall: if both reconstructions are stopped at the
*same* (small) number of iterations, the TOF image may appear *noisier*
than the non-TOF image — not because TOF is worse, but because TOF has
already converged past its low-noise plateau while non-TOF is still
climbing.  Conversely, at very early iterations non-TOF may look
smoother simply because it has not yet amplified the noise.

To observe the *true* asymptotic advantage of TOF one must run **both**
algorithms long enough to reach (approximate) convergence.

What this example shows
------------------------

* A single-ring 2-D scanner with a uniform circular phantom.
* **Full MLEM** run for 700 iterations, with a mild post-filter applied
  after each iteration so that the smoothed image is stored.
* The standard deviation inside a small (25 mm-radius) central ROI is
  tracked vs. iteration number — this is a fast single-realisation proxy
  for the true noise level that avoids the need for Monte Carlo repeats.
* Four-panel figure:

  - **Top-left**: raw std.dev curves vs. iteration — shows faster TOF
    convergence *and* the lower asymptotic noise level.
  - **Bottom-left**: ratio non-TOF / TOF std.dev — values > 1 confirm
    that non-TOF is noisier; the ratio rises as iterations increase and
    both methods converge.
  - **Top/bottom-right**: final smoothed images for visual comparison.

.. note::

    The standard deviation in a single noise realisation is used here as
    a proxy for the true noise standard deviation.
"""

# %%
from __future__ import annotations
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

num_epochs = 300
fwhm_tof_mm = 30.0
fwhm_res_model_mm = 4.0
sm_fwhm_mm = 9.0
cylinder_radius_mm = 120
count_factor = 0.3

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
# EM update rule
# --------------
#
# The standard MLEM update :cite:p:`Shepp1982` :cite:p:`Lange1984` can be
# written as a preconditioned gradient-descent step:
#
# .. math::
#     x^+ = x - D\,\nabla_x f(x),
#     \qquad
#     D = \operatorname{diag}\!\left(\frac{x}{A^H \mathbf{1}}\right)
#
# where :math:`f(x) = \sum_i [\bar{y}_i - y_i \log \bar{y}_i]` is the
# negative Poisson log-likelihood and :math:`A^H \mathbf{1}` is the
# sensitivity image.  The same update is used for both non-TOF and TOF;
# the only difference is the forward operator :math:`A`.


def em_update(
    x_cur: Array,
    negpoissonlogl: C1Function,
    adj_ones: Array,
) -> Array:
    """EM update re-written as preconditioned GD step"""
    em_diag_precond = x_cur / adj_ones
    return x_cur - em_diag_precond * negpoissonlogl.gradient(x_cur)


# %%
# Non-TOF MLEM
# ------------
#
# We run ``num_epochs`` full-data MLEM iterations and store the
# **post-filtered** image after every iteration.  Applying the same
# post-filter (:class:`.GaussianFilterOperator`, FWHM = ``sm_fwhm_mm``)
# at each iteration mirrors the typical clinical workflow where a
# reconstruction is post-smoothed before evaluation.  Storing all
# intermediate images lets us plot noise vs. iteration and observe both
# the convergence speed and the asymptotic noise level.

sm_op = parallelproj.operators.GaussianFilterOperator(
    in_shape=img_shape, sigma=sm_fwhm_mm / (2.35 * voxel_size[0])
)

full_data_fidelity_non_tof = C2AffineObjective(
    NegPoissonLogL(y_non_tof), pet_lin_op_non_tof, contamination_non_tof
)

adjoint_ones_non_tof = pet_lin_op_non_tof.adjoint(
    xp.ones(pet_lin_op_non_tof.out_shape, dtype=xp.float32, device=dev)
)

x_mlem_non_tof = count_factor * xp.ones(img_shape, device=dev, dtype=xp.float32)
recons_non_tof = xp.ones((num_epochs,) + img_shape, device=dev, dtype=xp.float32)

for i in range(num_epochs):
    print(f"NON-TOF MLEM epoch {(i + 1):04} / {num_epochs:04}", end="\r")
    x_mlem_non_tof = em_update(
        x_mlem_non_tof, full_data_fidelity_non_tof, adjoint_ones_non_tof
    )
    recons_non_tof[i, ...] = sm_op(x_mlem_non_tof)
print()


# %%
# TOF MLEM
# --------
#
# Identical loop but using the TOF forward operator and TOF sinogram.
# Because the TOF update is more informative (each LOR contributes noise
# over the kernel width ≈ 30 mm rather than the full chord ≈ 600 mm), the
# image converges to its maximum-likelihood solution in fewer iterations.

full_data_fidelity_tof = C2AffineObjective(
    NegPoissonLogL(y_tof), pet_lin_op_tof, contamination_tof
)

adjoint_ones_tof = pet_lin_op_tof.adjoint(
    xp.ones(pet_lin_op_tof.out_shape, dtype=xp.float32, device=dev)
)

x_mlem_tof = count_factor * xp.ones(img_shape, device=dev, dtype=xp.float32)
recons_tof = xp.ones((num_epochs,) + img_shape, device=dev, dtype=xp.float32)

for i in range(num_epochs):
    print(f"TOF MLEM epoch {(i + 1):04} / {num_epochs:04}", end="\r")
    x_mlem_tof = em_update(x_mlem_tof, full_data_fidelity_tof, adjoint_ones_tof)
    recons_tof[i, ...] = sm_op(x_mlem_tof)


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
epochs = np.arange(1, 1 + num_epochs)

# %%
# Visualisation
# -------------
#
# The four-panel figure summarises the comparison:
#
# * **Top-left**: std.dev in the central 25 mm ROI vs. MLEM iteration for
#   non-TOF (orange) and TOF (blue).  Note how TOF rises *and falls* faster;
#   comparing at a fixed early iteration can give the wrong conclusion.
# * **Bottom-left**: ratio of std.devs (non-TOF / TOF).  The ratio
#   increases with iteration count and stabilises above 1, quantifying the
#   asymptotic noise advantage of TOF.
# * **Top-right / bottom-right**: final smoothed images after ``num_epochs``
#   iterations.  Visual noise in the uniform disk is lower for TOF.

ims = dict(vmin=0, vmax=xp.max(recons_non_tof), cmap="Greys")

fig, ax = plt.subplots(2, 2, figsize=(6, 6), layout="constrained", sharex="col")
ax[0, 0].plot(epochs, roi_std_non_tof, label="non-TOF", color="tab:orange")
ax[0, 0].plot(
    epochs, roi_std_tof, label=f"TOF ({fwhm_tof_mm:.0f} mm FWHM)", color="tab:blue"
)
ax[0, 0].legend(fontsize=8)
ax[1, 0].plot(epochs, roi_std_non_tof / roi_std_tof, color="tab:green")
ax[1, 0].axhline(1.0, color="gray", ls=":", lw=0.8)
ax[1, 0].set_xlabel("MLEM iteration")
ax[0, 0].set_ylabel("std.dev in central ROI")
ax[1, 0].set_ylabel("std.dev ratio  (non-TOF / TOF)")
ax[0, 0].set_title(
    f"(central 25 mm ROI, {sm_fwhm_mm:.0f} mm post-filter)",
    fontsize=8,
)
ax[0, 0].grid(ls=":")
ax[1, 0].grid(ls=":")

ax[0, 1].imshow(to_numpy_array(recons_non_tof[-1, :, :, 0]), **ims)
ax[1, 1].imshow(to_numpy_array(recons_tof[-1, :, :, 0]), **ims)
ax[0, 1].set_title(
    f"non-TOF  ({num_epochs} iter)\nstd.dev = {roi_std_non_tof[-1]:.4f}",
    fontsize=8,
)
ax[1, 1].set_title(
    f"TOF {fwhm_tof_mm:.0f} mm  ({num_epochs} iter)\nstd.dev = {roi_std_tof[-1]:.4f}",
    fontsize=8,
)
for a in ax[:, 1]:
    a.set_axis_off()

fig.suptitle(
    f"TOF variance reduction — uniform cylinder Ø {2*cylinder_radius_mm:.0f} mm",
    fontsize=9,
)
fig.show()
