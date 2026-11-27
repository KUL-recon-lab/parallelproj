"""
TOF vs non-TOF: variance reduction in a uniform cylinder
========================================================

:bdg-primary:`reconstruction` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Time-of-flight (TOF) information localises each detected annihilation along
its line of response (LOR), which lowers the noise in reconstructed images.
In this example we reconstruct the same Poisson data of a uniform cylinder
with and without TOF information (OSEM warm start followed by SVRG) and
compare the noise in the image centre after every epoch.  Because TOF
reconstructions also converge faster, a fair comparison needs both
reconstructions to be run long enough.

**You will learn**

- how TOF data relate to non-TOF data (summing over the TOF bins),
- how to set up matching non-TOF and TOF forward models with attenuation and
  a resolution model,
- how much TOF reduces the noise in the centre of a uniform cylinder, and how
  this compares to the approximate gain :math:`\\sqrt{0.66 \\, D / \\text{FWHM}}`,
- why comparing TOF and non-TOF images after the same small number of
  iterations can be misleading.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(OSEM and SVRG) and
:ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`
(TOF sinogram projector).

.. note::

    The standard deviation in a single noise realisation is used here as
    a proxy for the true noise standard deviation.  For a uniform phantom
    spatial variability inside the ROI equals the noise variability, so
    the single realisation is sufficient.
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

from collections.abc import Sequence
from copy import copy

import matplotlib.pyplot as plt
import numpy as np

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import Array, to_numpy_array
from parallelproj._examples_utils import suggest_array_backend_and_device
from parallelproj.functions import C1Function, C2AffineObjective, NegPoissonLogL

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Simulation parameters
# ------------------------
#
# - ``num_epochs`` SVRG epochs (with ``num_subsets`` subsets) are run after a
#   one-epoch OSEM warm start.
# - ``fwhm_tof_mm = 30 mm`` corresponds to a coincidence timing resolution of
#   about 200 ps, representative of state-of-the-art clinical scanners.
# - ``sm_fwhm_mm`` is the FWHM of a mild Gaussian post-filter that suppresses
#   high-frequency "salt-and-pepper" noise in the images we analyse.
# - ``count_factor`` is the activity of the phantom and controls the number
#   of detected events.  Moderate counts (0.3) give a clearly visible noise
#   difference between TOF and non-TOF.
# - ``cylinder_radius_mm`` is the radius of the uniform phantom disk (in mm).

num_subsets = 14
num_epochs = 10
fwhm_tof_mm = 30.0
fwhm_res_model_mm = 4.0
sm_fwhm_mm = 9.0
cylinder_radius_mm = 120
count_factor = 0.3
step_size = 2.0

# %%
# 2. Scanner, sinogram and image geometry
# ---------------------------------------
#
# We use a **single-ring scanner** so that the reconstruction is effectively
# 2D.  This keeps the computation fast and isolates the transaxial TOF effect.
# The scanner radius of 300 mm and 28 x 16 = 448 LOR endpoints per ring give a
# clinical-scale geometry.  The single image plane has 151 x 151 x 1 voxels of
# 2 mm, i.e. a 302 mm transaxial field of view (FOV).
#
# In the :class:`.RegularPolygonPETLORDescriptor`, ``max_ring_difference=2``
# is harmless (there is only one ring) and ``radial_trim=150`` discards the
# outermost radial bins whose LORs do not pass through the image.

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
# 3. Uniform cylinder phantom
# ---------------------------
#
# A disk of radius ``cylinder_radius_mm`` (240 mm diameter) centred in the FOV
# with uniform activity ``count_factor``.  ``RHO`` (the distance of each voxel
# from the centre in mm) is also used later to define the central ROI.

x_pos = voxel_size[0] * (
    xp.arange(img_shape[0], device=dev, dtype=xp.float32) - img_shape[0] / 2 + 0.5
)
X0, X1 = xp.meshgrid(x_pos, x_pos, indexing="ij")
RHO = xp.sqrt(X0**2 + X1**2)

x_true = xp.ones(img_shape, device=dev, dtype=xp.float32)
x_true[..., 0] = count_factor * (RHO <= cylinder_radius_mm)

# %%
# 4. Non-TOF and TOF forward models
# ---------------------------------
#
# **Attenuation.**  A uniform water-equivalent attenuation coefficient of
# :math:`\mu = 0.01\,\text{mm}^{-1}` is used inside the cylinder.  Attenuation
# is the same for TOF and non-TOF; it is included for realism and has no
# bearing on the variance comparison.
#
# **TOF.**  TOF PET measures the difference in arrival times of the two
# 511 keV photons and uses it to localise the annihilation along the LOR with
# a Gaussian kernel
#
# .. math::
#
#     h(\ell) = \frac{1}{\sqrt{2\pi}\,\sigma_\text{TOF}}
#               \exp\!\left(-\frac{\ell^2}{2\sigma_\text{TOF}^2}\right),
#     \qquad
#     \sigma_\text{TOF} = \frac{c}{2} \cdot \frac{\Delta t_\text{FWHM}}{2.3548}
#
# where :math:`\ell` is the distance from the emission point along the LOR and
# :math:`\Delta t_\text{FWHM}` is the coincidence timing resolution (CTR) of
# the scanner.  A CTR of 200 ps corresponds to a spatial FWHM of about 30 mm.
#
# The TOF projector is a copy of the non-TOF projector with
# :class:`.TOFParameters` attached.  The TOF bin width is FWHM / 4, so that
# the kernel is sampled by about 4 bins per FWHM, and the 41 TOF bins cover
# about 300 mm along each LOR.  The attenuation sinogram has no TOF axis: it
# is broadcast over the TOF bins without copying data.
#
# Both forward models also contain the same image-based resolution model
# (:class:`.GaussianFilterOperator`, FWHM = 4 mm).

x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
att_sino = xp.exp(-proj_non_tof(x_att))

proj_tof = copy(proj_non_tof)

proj_tof.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=int(300 / (fwhm_tof_mm / 4.0)) + 1,
    tofbin_width=fwhm_tof_mm / 4.0,
    sigma_tof=fwhm_tof_mm / 2.3548,
)

# add a trailing TOF axis to att_sino and broadcast it (zero-stride view)
att_values_tof = xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj_tof.out_shape)
att_op_tof = parallelproj.operators.ElementwiseMultiplicationOperator(att_values_tof)
att_op_non_tof = parallelproj.operators.ElementwiseMultiplicationOperator(att_sino)

res_model = parallelproj.operators.GaussianFilterOperator(
    img_shape,
    sigma=[fwhm_res_model_mm / (2.3548 * float(vs)) for vs in proj_tof.voxel_size],
)

pet_lin_op_tof = parallelproj.operators.CompositeLinearOperator(
    (att_op_tof, proj_tof, res_model)
)
pet_lin_op_non_tof = parallelproj.operators.CompositeLinearOperator(
    (att_op_non_tof, proj_non_tof, res_model)
)

# %%
# 5. Data simulation
# ------------------
#
# The TOF sinogram is simulated once, and a constant scatter / randoms
# contamination (50 % of the mean of the noise-free TOF data) is added before
# Poisson sampling.
#
# The non-TOF sinogram is obtained by **summing the noisy TOF sinogram over
# its TOF bins**.  This is equivalent to discarding the timing information
# in a real scanner, and it ensures that both reconstructions use exactly the
# same Poisson noise realisation: they differ only in whether they exploit
# the timing information.

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

# sum over the TOF bins gives the non-TOF sinogram
y_non_tof = xp.sum(y_tof, axis=-1)
contamination_non_tof = xp.sum(contamination_tof, axis=-1)

# %%
# 6. Post-filter, subsets and FOV mask
# ------------------------------------
#
# The Gaussian post-filter ``sm_op`` is applied to a copy of the image after
# every epoch for the analysis; it is not part of the iterations.
#
# The sinogram views are split into ``num_subsets`` disjoint groups.  Non-TOF
# data and the attenuation sinogram are 3D (radial x view x plane), TOF data
# have a fourth (TOF bin) axis.  We therefore request 3D slices to index the
# attenuation and non-TOF data and 4D slices to index the TOF data.
#
# The cylindrical FOV of the scanner does not cover every voxel of the image
# grid.  Voxels outside the FOV are not intersected by any LOR, so their
# sensitivity :math:`(A^T 1)_j` is 0, and dividing by it in the EM
# preconditioner would produce NaN / Inf values.
# :meth:`.RegularPolygonPETProjector.fov_mask` returns a boolean array that
# is ``True`` inside the FOV; ``fov_mask`` is ``None`` if every voxel is
# inside the FOV (no masking needed).

sm_op = parallelproj.operators.GaussianFilterOperator(
    in_shape=img_shape, sigma=sm_fwhm_mm / (2.3548 * voxel_size[0])
)

# 3D slices: used for non-TOF data *and* to index att_sino
subset_views, subset_slices_nt = lor_desc.get_distributed_views_and_slices(
    num_subsets, 3
)
# 4D slices: used to index TOF data and contamination
_, subset_slices_tof = lor_desc.get_distributed_views_and_slices(num_subsets, 4)

cyl_mask = proj_non_tof.fov_mask()
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask
del cyl_mask

# %%
# 7. EM and SVRG updates
# ----------------------
#
# These functions implement the OSEM update (written as EM-preconditioned
# gradient descent) and SVRG exactly as in
# :ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`.
# ``svrg_calc_snapshot_gradients`` computes and stores all subset gradients at
# the current anchor point; ``svrg_update`` performs a single
# variance-reduced subset step.


def em_update(
    x_cur: Array,
    data_fidelity: C1Function,
    adj_ones: Array,
    img_mask: Array | None = None,
) -> Array:
    """One EM update rewritten as a preconditioned gradient descent step.

    Computes :math:`x^+ = x - D \\nabla f(x)` where the diagonal
    preconditioner is :math:`D = \\operatorname{diag}(x / (A^T 1))`.
    Voxels outside the FOV are excluded via ``img_mask`` to avoid
    division by the zero sensitivity values in ``adj_ones``.

    Parameters
    ----------
    x_cur : Array
        Current image estimate.
    data_fidelity : C1Function
        Differentiable data-fidelity term whose gradient is evaluated at
        ``x_cur``.
    adj_ones : Array
        Sensitivity image :math:`A^T 1` (or subset variant
        :math:`(A^k)^T 1`).
    img_mask : Array or None, optional
        Boolean FOV mask (``True`` inside the FOV).  Preconditioner is
        zeroed outside the FOV so that zero-sensitivity voxels do not
        produce NaN / Inf.  Pass ``None`` when every voxel is in the FOV.

    Returns
    -------
    Array
        Updated image :math:`x^+`, same shape as ``x_cur``.
    """
    if img_mask is None:
        d = x_cur / adj_ones
    else:
        d = xp.where(img_mask, x_cur / adj_ones, xp.zeros_like(x_cur))
    return x_cur - d * data_fidelity.gradient(x_cur)


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
# 8. Non-TOF reconstruction
# -------------------------
#
# One :class:`.CompositeLinearOperator` is built per subset, combining the
# subset attenuation factors, the subset projector and the resolution model.
# The subset sensitivity images :math:`(A^k)^T \mathbf{1}` are computed once
# and summed to obtain the full :math:`A^T \mathbf{1}`.
#
# The warm start runs a single OSEM epoch, which moves the initial flat image
# close enough to the solution for the SVRG preconditioner to be meaningful
# from the first epoch on.  SVRG then recomputes the snapshot gradients every
# second epoch (and the preconditioner up to epoch 5).  After every epoch, the
# post-filtered image is stored.
#
# .. note::
#     By default :class:`.NegPoissonLogL` evaluates a "safe epsilon"
#     (shifted Poisson) surrogate: a tiny ``eps = rel_eps * mean(y)`` is
#     added to the measured and the expected data.  This is finite for any
#     non-negative expectation (never ``nan`` / ``inf``), at the price of a
#     tiny (~``rel_eps``) bias that vanishes at the fit.  Since our
#     contamination is strictly positive (in TOF and non-TOF form), the
#     expected data ``A x + s`` are positive in every bin and we can use
#     ``exact=True`` to evaluate the unmodified log-likelihood instead
#     (bins with ``y = 0`` are still handled exactly).  Keep the default
#     whenever the expected data can reach zero in bins with counts.

# the strictly positive contamination guarantees A x + s > 0 in every bin,
# so the exact (unmodified) log-likelihood can be used
exact_mode_nt = bool(xp.min(contamination_non_tof) > 0)
exact_mode_tof = bool(xp.min(contamination_tof) > 0)

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
        NegPoissonLogL(y_non_tof[subset_slices_nt[k]], exact=exact_mode_nt),
        subset_linops_nt[k],
        contamination_non_tof[subset_slices_nt[k]],
    )
    for k in range(num_subsets)
]

# --- warm start: 1 OSEM epoch ---
x_nt = count_factor * xp.ones(img_shape, device=dev, dtype=xp.float32)
if fov_mask is not None:
    x_nt = xp.where(fov_mask, x_nt, xp.zeros_like(x_nt))
for k in range(num_subsets):
    print(f"  non-TOF warm-start subset {k + 1:03}/{num_subsets:03}", end="\r")
    x_nt = em_update(x_nt, subset_fidelities_nt[k], subset_adj_ones_nt[k], fov_mask)
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
# 9. TOF reconstruction
# ---------------------
#
# The structure is identical to the non-TOF case.  The only differences are:
#
# * each subset attenuation operator broadcasts ``att_sino`` over the TOF
#   bins (zero-copy via ``xp.broadcast_to``),
# * data and contamination are indexed with the 4D ``subset_slices_tof``.
#
# Because the TOF forward model localises each event to about 30 mm along the
# LOR instead of anywhere along the LOR, every gradient step is more
# informative and the algorithm reaches its noise level in fewer epochs.

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
        NegPoissonLogL(y_tof[subset_slices_tof[k]], exact=exact_mode_tof),
        subset_linops_tof[k],
        contamination_tof[subset_slices_tof[k]],
    )
    for k in range(num_subsets)
]

# --- warm start: 1 OSEM epoch ---
x_tof = count_factor * xp.ones(img_shape, device=dev, dtype=xp.float32)
if fov_mask is not None:
    x_tof = xp.where(fov_mask, x_tof, xp.zeros_like(x_tof))
for k in range(num_subsets):
    print(f"  TOF warm-start subset {k + 1:03}/{num_subsets:03}", end="\r")
    x_tof = em_update(x_tof, subset_fidelities_tof[k], subset_adj_ones_tof[k], fov_mask)
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
# 10. Noise vs. epoch in the central ROI
# --------------------------------------
#
# The standard deviation of the voxel values inside a central ROI of 25 mm
# radius is a single-realisation proxy for the noise standard deviation:
# because the phantom is uniform, every voxel inside the ROI has the same
# expected value.  Epoch 0 is the OSEM warm start.
#
# TOF reduces the noise in the centre of a uniform cylinder with diameter
# :math:`D`; the SNR gain derived by Tomitani :footcite:p:`Tomitani1981` is
# approximately
#
# .. math::
#
#     G_\text{TOF} \approx \sqrt{0.66 \frac{D}{\text{FWHM}_\text{TOF}}}.
#
# For :math:`D = 240` mm and :math:`\text{FWHM} = 30` mm this gives
# :math:`G \approx 2.3`, i.e. more than a two-fold noise reduction in the
# centre.  The right panel compares the measured ratio of the standard
# deviations (non-TOF / TOF) with this value.


def roi_std(img: Array) -> float:
    """Standard deviation in the central ROI (radius 25 mm)"""
    return float(img[:, :, 0][RHO < 25].std())


roi_std_non_tof = np.array(
    [roi_std(x_nt_warmstart_sm)] + [roi_std(x) for x in recons_non_tof]
)
roi_std_tof = np.array([roi_std(x_tof_warmstart_sm)] + [roi_std(x) for x in recons_tof])
epochs = np.arange(0, num_epochs + 1)  # 0 = OSEM warm start
gain_tof = np.sqrt(0.66 * 2 * cylinder_radius_mm / fwhm_tof_mm)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained", sharex=True)
ax[0].plot(epochs, roi_std_non_tof, marker="o", label="non-TOF")
ax[0].plot(epochs, roi_std_tof, marker="o", label=f"TOF ({fwhm_tof_mm:.0f} mm FWHM)")
ax[0].set_ylabel("std. dev. in central ROI [a.u.]")
ax[0].set_title(f"Noise in the central ROI ({sm_fwhm_mm:.0f} mm post-filter)")
ax[0].legend()
ax[1].plot(epochs, roi_std_non_tof / roi_std_tof, marker="o", label="measured")
ax[1].axhline(
    gain_tof, color="0.4", ls="--", label=f"predicted $G_{{TOF}}$ = {gain_tof:.2f}"
)
ax[1].axhline(1.0, color="0.4", ls=":", label="no gain")
ax[1].set_ylabel("std. dev. ratio (non-TOF / TOF)")
ax[1].set_title("Noise reduction by TOF")
ax[1].legend()
for a in ax:
    a.set_xlabel(f"SVRG epoch ({num_subsets} subsets, 0 = OSEM warm start)")
    a.grid(ls=":")
fig.show()

# %%
# The TOF noise reaches its level after about 2 epochs and then stays
# constant, while the non-TOF noise is still increasing after 10 epochs.
# After the warm start (epoch 0), TOF is *noisier* than non-TOF (ratio 0.84):
# it has already amplified the noise, while non-TOF is still smooth from the
# flat initialisation.  From epoch 1 on the ratio is above 1 and reaches 2.3
# after 10 epochs, close to the predicted gain.  Since the non-TOF
# reconstruction is not yet fully converged, the asymptotic ratio is
# probably slightly higher.
#
# This illustrates a common pitfall: if TOF and non-TOF reconstructions are
# stopped after the *same* small number of iterations, the TOF image may look
# noisier, not because TOF is worse but because it converges faster.  To see
# the true advantage of TOF, both reconstructions must be run long enough to
# (approximately) converge.

# %%
# 11. Post-filtered images
# ------------------------
#
# The images after the warm start (epoch 0), after 1 SVRG epoch and after
# ``num_epochs`` SVRG epochs, all with the same grey scale.  The panel titles
# give the standard deviation in the central ROI.

shown_epochs = (0, 1, num_epochs)
rows = [
    ("non-TOF", (x_nt_warmstart_sm, recons_non_tof[0], recons_non_tof[-1])),
    (f"TOF {fwhm_tof_mm:.0f} mm", (x_tof_warmstart_sm, recons_tof[0], recons_tof[-1])),
]
row_stds = (roi_std_non_tof, roi_std_tof)
vmax = float(xp.max(recons_non_tof))

fig, ax = plt.subplots(2, 3, figsize=(8.5, 6.2), layout="constrained")
for row, ((label, imgs), stds) in enumerate(zip(rows, row_stds)):
    for col, (img, ep) in enumerate(zip(imgs, shown_epochs)):
        a = ax[row, col]
        im = a.imshow(to_numpy_array(img[:, :, 0]).T, cmap="Greys", vmin=0, vmax=vmax)
        a.set_title(f"{label}, epoch {ep}\nstd. dev. = {stds[ep]:.4f}")
        a.set_axis_off()
fig.colorbar(im, ax=ax, shrink=0.6, label="activity [a.u.]")
fig.suptitle(
    f"Uniform cylinder ({2 * cylinder_radius_mm:.0f} mm diameter),"
    f" {sm_fwhm_mm:.0f} mm post-filter"
)
fig.show()

# %%
# After the warm start the TOF image shows slightly more texture than the
# non-TOF image.  After 10 SVRG epochs the non-TOF image is much noisier
# than the TOF image, whose noise has hardly changed since epoch 1.

# %%
# Summary
# -------
#
# - Summing TOF data over the TOF bins gives non-TOF data; reconstructing both
#   from the same noise realisation isolates the effect of the timing
#   information.
# - In the centre of a 240 mm uniform cylinder, 30 mm FWHM TOF reduces the
#   noise standard deviation by a factor of about 2.3 after 10 SVRG epochs,
#   close to the approximate gain :math:`\sqrt{0.66 \, D / \text{FWHM}}`.
# - TOF reconstructions converge faster: comparing TOF and non-TOF images
#   after the same small number of iterations can wrongly suggest that TOF
#   is noisier.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_06_run_osem_memmap.py`
#    -- OSEM with disk-backed TOF sinograms to save memory.


# %%
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_number = 2
