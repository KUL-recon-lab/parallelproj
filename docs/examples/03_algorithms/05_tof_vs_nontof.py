"""
TOF vs NON-TOF RECONSTRUCTIONS in a 2D uniform cylinder
=======================================================

This example compares variance reduction due to the presence of TOF information.
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

num_epochs = 700
fwhm_tof_mm = 30.0
sm_fwhm_mm = 9.0
cylinder_radius = 140
count_factor = 0.3

# %%
# Setup of the forward model :math:`\bar{y}(x) = A x + s`
# --------------------------------------------------------
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

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
# setup the LOR descriptor that defines the sinogram

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

# setup a uniform circle
x_pos = voxel_size[0] * (
    xp.arange(img_shape[0], device=dev, dtype=xp.float32) - img_shape[0] / 2 + 0.5
)
X, Y = xp.meshgrid(x_pos, x_pos, indexing="ij")
RHO = xp.sqrt(X**2 + Y**2)

x_true = xp.ones(img_shape, device=dev, dtype=xp.float32)
x_true[..., 0] = count_factor * (RHO <= cylinder_radius)

# %%
# Attenuation image and sinogram setup
# ------------------------------------

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj_non_tof(x_att))

# %%
# Complete PET forward model setup
# --------------------------------
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

proj_tof = copy(proj_non_tof)

proj_tof.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=int(300 / (fwhm_tof_mm / 5.0)) + 1,
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
    img_shape, sigma=[4.0 / (2.35 * float(vs)) for vs in proj_tof.voxel_size]
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
# Simulation of projection data
# -----------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data_tof = pet_lin_op_tof(x_true)

# generate a contant contamination sinogram
contamination_tof = xp.full(
    noise_free_data_tof.shape,
    0.5 * float(xp.mean(noise_free_data_tof)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data_tof += contamination_tof

# add Poisson noise
np.random.seed(1)
y_tof = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data_tof)),
    device=dev,
    dtype=xp.float32,
)

y_non_tof = xp.sum(y_tof, axis=-1)
contamination_non_tof = xp.sum(contamination_tof, axis=-1)

# %%
# EM update
# ---------
#
# The EM update used in MLEM and OSEM is :cite:p:`Dempster1977`
# :cite:p:`Shepp1982` :cite:p:`Lange1984` :cite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{A^H 1} A^H \frac{y}{A x + s}
#
# which can be rewritten as a preconditioned gradient descent step with
# diagonal preconditioner :math:`D = \operatorname{diag}(x / (A^H 1))`:
#
# .. math::
#     x^+ = x - D \, \nabla_x f(x).
#
# We implement this as a single function used by both MLEM and OSEM.


def em_update(
    x_cur: Array,
    negpoissonlogl: C1Function,
    adj_ones: Array,
) -> Array:
    """EM update re-written as preconditioned GD step"""
    em_diag_precond = x_cur / adj_ones
    return x_cur - em_diag_precond * negpoissonlogl.gradient(x_cur)


# %%
# NON-TOF EM reconstruction
# -------------------------

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
# TOF EM reconstruction
# ---------------------

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
# Visualize (smoothed) reconstructions
# ------------------------------------

roi_std_non_tof = np.array([float(x[:, :, 0][RHO < 25].std()) for x in recons_non_tof])
roi_std_tof = np.array([float(x[:, :, 0][RHO < 25].std()) for x in recons_tof])
epochs = np.arange(1, 1 + num_epochs)

ims = dict(vmin=0, vmax=xp.max(recons_non_tof), cmap="Greys")

fig, ax = plt.subplots(2, 2, figsize=(6, 6), layout="constrained", sharex="col")
ax[0, 0].plot(epochs, roi_std_non_tof, label="non-TOF")
ax[0, 0].plot(epochs, roi_std_tof, label="TOF")
ax[0, 0].legend()
ax[1, 0].plot(roi_std_non_tof / roi_std_tof)
ax[1, 0].set_xlabel("epoch")
ax[0, 0].set_ylabel("std.dev in central ROI")
ax[1, 0].set_ylabel("std.dev(non-TOF) / std.dev(TOF)")
ax[0, 0].grid(ls=":")
ax[1, 0].grid(ls=":")

ax[0, 1].imshow(to_numpy_array(recons_non_tof[-1, :, :, 0]), **ims)
ax[1, 1].imshow(to_numpy_array(recons_tof[-1, :, :, 0]), **ims)
ax[0, 1].set_title(f"Non-TOF {num_epochs} epochs", fontsize="medium")
ax[1, 1].set_title(f"TOF ({fwhm_tof_mm}mm FWHM) {num_epochs} epochs", fontsize="medium")

fig.show()
