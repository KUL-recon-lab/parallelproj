"""
Listmode MLEM, OSEM and SVRG
============================

This example demonstrates how to run the MLEM reconstruction algorithm
using listmode (event-by-event) data, and compares it with the equivalent
sinogram-based reconstruction.

Two objective functions from :mod:`parallelproj.functions` are used:

- :class:`.NegPoissonLogL` — operates in **prediction space** :math:`\\bar{y}`,
  wrapped with :class:`.C2AffineObjective` to compose with the sinogram
  forward model :math:`\\bar{y}(x) = A x + s`:

  .. math::

      f_{\\text{sino}}(x) = \\sum_i \\bigl[(Ax+s)_i - y_i \\log(Ax+s)_i\\bigr]

- :class:`.NegPoissonLogLListmode` — operates directly in **image space**
  :math:`x`, with the listmode forward model built in internally:

  .. math::

      f_{\\text{LM}}(x) = \\langle A^T \\mathbf{1},\\, x \\rangle + c_{\\text{sino}}
                          - \\sum_{e=1}^{N_{\\text{ev}}} \\log\\bigl((A_{\\text{LM}}\\,x)_e + s_e\\bigr)

  where :math:`c_{\\text{sino}} = \\sum_i s_i` is the total contamination summed
  over all sinogram bins.

Both functions are mathematically equivalent (since
:math:`\\sum_e \\log \\bar{y}_{j_e} = \\sum_i y_i \\log \\bar{y}_i`) and share
the same :class:`.C1Function` interface, exposing a :meth:`~.C1Function.gradient`
method that returns the gradient w.r.t. the image :math:`x`.

**Key learning goal:** The MLEM update, written as a preconditioned gradient
descent step, is *agnostic of the data representation* — the same
:func:`em_update` function works identically for both sinogram and listmode
objectives because both implement the :class:`.C1Function` interface with the
correct gradient w.r.t. :math:`x`.
"""

# %%
from __future__ import annotations
from array_api_compat import size
from vis import show_vol_cuts
from img import elliptic_cylinder_phantom
import numpy as np

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array
from parallelproj.functions import (
    NegPoissonLogL,
    NegPoissonLogLListmode,
    C2AffineObjective,
    C1Function,
)

# %%
from importlib import import_module, util
import parallelproj_core as ppc


# choose array backend and a device (CPU or CUDA GPU)
if util.find_spec("torch") is not None:
    xp = import_module("array_api_compat.torch")
    dev = "cuda" if xp.cuda.is_available() and ppc.cuda_enabled == 1 else "cpu"
elif util.find_spec("cupy") is not None and ppc.cupy_enabled == 1:
    xp = import_module("array_api_compat.cupy")
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
else:
    xp = import_module("array_api_compat.numpy")
    # using numpy, device must be cpu
    dev = "cpu"

print(f"Using array API: {xp.__name__}, device: {dev}")


# %%
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated listmode data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram that can be converted to listmode data.

# %%
# Sinogram forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_epochs = 50
sens_factor = 0.1

num_rings = 5
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=10,
    max_ring_difference=2,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

x_true = elliptic_cylinder_phantom(xp, dev)


# %%
# Attenuation image and sinogram setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# %%
# Complete PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

# For TOF, att_sino has no TOF-bins dimension while the projector output does.
# broadcast_to adds a trailing singleton via expand_dims and broadcasts it over
# the TOF-bins axis without copying data (zero-stride view).
att_values = (
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
    if proj.tof
    else att_sino
)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    sens_factor * att_values
)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=4.5 / (2.35 * proj.voxel_size)
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# Simulation of sinogram projection data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a contant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.int16,
)

# %%
# Conversion of the emission sinogram to listmode
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Using :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` we can convert an
# integer non-TOF or TOF sinogram to an event list for listmode processing.
#
# **Note:** The create event list is sorted and should be shuffled running LM-MLEM.

event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    y
)

# %%
# Setup of the LM projector and LM forward model
# ----------------------------------------------

lm_proj = parallelproj.projectors.ListmodePETProjector(
    event_start_coords,
    event_end_coords,
    img_shape,
    voxel_size,
    proj.img_origin,
)

# recalculate the attenuation factor for all LM events
# this needs to be a non-TOF projection
att_list = xp.exp(-lm_proj(x_att))
lm_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
    sens_factor * att_list
)

# enable TOF in the LM projector
lm_proj.tof_parameters = proj.tof_parameters
if proj.tof:
    lm_proj.event_tofbins = event_tofbins
    lm_proj.tof = proj.tof

# create the contamination list
contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=xp.float32,
)

lm_pet_lin_op = parallelproj.operators.CompositeLinearOperator(
    (lm_att_op, lm_proj, res_model)
)

# %%
# Setup of sinogram and listmode objective functions
# --------------------------------------------------
#
# Both objective functions expose the same :class:`.C1Function` interface
# (``__call__`` for the function value, ``gradient`` for the gradient w.r.t.
# the image :math:`x`).
#
# The sinogram objective wraps :class:`.NegPoissonLogL` — which operates on
# predicted counts :math:`\bar{y}` — with :class:`.C2AffineObjective` to
# pull the gradient back through the forward model :math:`\bar{y}(x) = Ax + s`:
#
# .. math::
#
#     f_\text{sino}(x) = \sum_i \bigl[(Ax+s)_i - y_i\log(Ax+s)_i\bigr],
#     \quad \nabla_x f_\text{sino} = A^T\!\left(1 - \tfrac{y}{Ax+s}\right)
#
# The listmode objective :class:`.NegPoissonLogLListmode` takes :math:`x`
# directly and has the forward model built in.  Its gradient is:
#
# .. math::
#
#     \nabla_x f_\text{LM}(x) = A^T\mathbf{1}
#         - A_\text{LM}^T\!\left(\frac{1}{A_\text{LM} x + s_\text{LM}}\right)
#
# Both gradients are mathematically equivalent because each detected event
# from sinogram bin :math:`i` contributes the same :math:`\log\bar{y}_i`
# term, so :math:`\sum_e \log\bar{y}_{j_e} = \sum_i y_i\log\bar{y}_i`.

sinogram_neg_logL = C2AffineObjective(NegPoissonLogL(y), pet_lin_op, contamination)

# The sensitivity image A^T 1 is required by NegPoissonLogLListmode.
# It equals the adjoint of the *sinogram* (full) forward model applied to
# all-ones, and serves as the diagonal of the EM preconditioner.
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)
lm_neg_logL = NegPoissonLogLListmode(
    lm_pet_lin_op, adjoint_ones, contamination_list, float(xp.sum(contamination))
)
# %%
# Verify gradient equivalence at a test image
# -------------------------------------------
#
# We evaluate both objectives and their gradients at a flat all-ones image
# :math:`x_\text{init}` to confirm that sinogram and listmode formulations
# produce the same (up to floating-point) function values and gradients.
#
# The preconditioned gradient :math:`(x / A^T\mathbf{1}) \odot \nabla_x f`
# is displayed: it equals :math:`x - x_\text{new}`, i.e. the negative of
# a single EM update step, and highlights where the current iterate would
# increase or decrease after one MLEM step.

x_init = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)

df_sino = sinogram_neg_logL(x_init)
df_lm = lm_neg_logL(x_init)

print(
    f"Negative Poisson log-likelihood at x_init: {df_sino:.4E} (sinogram), {df_lm:.4E} (listmode)"
)

grad_sinogram = sinogram_neg_logL.gradient(x_init)
grad_lm = lm_neg_logL.gradient(x_init)

# %%
# Preconditioned gradient of the sinogram objective
vmax = float(xp.max(xp.abs(x_init / adjoint_ones) * grad_sinogram))

fig_gs, _, widgets_gs = show_vol_cuts(
    to_numpy_array((x_init / adjoint_ones) * grad_sinogram),
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
)
fig_gs.show()

# %%
# Preconditioned gradient of the listmode objective (should match the sinogram plot above)
fig_glm, _, widgets_glm = show_vol_cuts(
    to_numpy_array((x_init / adjoint_ones) * grad_lm),
    vmin=-vmax,
    vmax=vmax,
    cmap="seismic",
)
fig_glm.show()


# %%
# MLEM reconstruction (sinogram and listmode)
# -------------------------------------------
#
# The classical MLEM update
#
# .. math::
#
#     x^{(k+1)} = \frac{x^{(k)}}{A^T \mathbf{1}} \odot A^T\!\left(\frac{y}{A x^{(k)} + s}\right)
#
# can equivalently be written as a preconditioned gradient descent step:
#
# .. math::
#
#     x^{(k+1)} = x^{(k)} - \underbrace{\frac{x^{(k)}}{A^T \mathbf{1}}}_{\text{preconditioner}} \odot \nabla_x f(x^{(k)})
#
# This formulation is **data-representation agnostic**: the same
# :func:`em_update` below works for any objective :math:`f` that provides
# a gradient :math:`\nabla_x f` via the :class:`.C1Function` interface —
# whether that objective is :class:`.C2AffineObjective` wrapping
# :class:`.NegPoissonLogL` (sinogram data) or
# :class:`.NegPoissonLogLListmode` (listmode data).


def em_update(
    x_cur: Array,
    negpoissonlogl: C1Function,
    adj_ones: Array,
) -> Array:
    """One MLEM iteration as a preconditioned gradient descent step.

    Implements

    .. math::

        x^{(k+1)} = x^{(k)} - \\frac{x^{(k)}}{A^T\\mathbf{1}}
                    \\odot \\nabla_x f(x^{(k)})

    This is equivalent to the standard MLEM multiplicative update and works
    for *any* objective ``negpoissonlogl`` that implements the
    :class:`.C1Function` interface — in particular for both
    :class:`.C2AffineObjective` (sinogram) and
    :class:`.NegPoissonLogLListmode` (listmode).

    Parameters
    ----------
    x_cur:
        Current image estimate :math:`x^{(k)}`.
    negpoissonlogl:
        Negative Poisson log-likelihood objective.  Must provide a
        ``gradient(x)`` method returning the gradient w.r.t. :math:`x`.
    adj_ones:
        Sensitivity image :math:`A^T\\mathbf{1}` used as the diagonal
        preconditioner.

    Returns
    -------
    Array
        Updated image estimate :math:`x^{(k+1)}`.
    """
    em_diag_precond = x_cur / adj_ones

    return x_cur - em_diag_precond * negpoissonlogl.gradient(x_cur)


# run MLEM with sinogram data
x_mlem_sino = xp.asarray(x_init, copy=True)
for i in range(num_epochs):
    print(f"MLEM epoch {(i + 1):04} / {num_epochs:04}", end="\r")
    x_mlem_sino = em_update(x_mlem_sino, sinogram_neg_logL, adjoint_ones)
print()

# run MLEM with listmode data — identical loop, only the objective changes
x_mlem_lm = xp.asarray(x_init, copy=True)
for i in range(num_epochs):
    print(f"LM-MLEM epoch {(i + 1):04} / {num_epochs:04}", end="\r")
    x_mlem_lm = em_update(x_mlem_lm, lm_neg_logL, adjoint_ones)
print()

# %%

vmax = float(xp.max(x_mlem_sino))

fig_xsino, _, widgets_xsino = show_vol_cuts(
    to_numpy_array(x_mlem_sino),
    vmin=0,
    vmax=vmax,
)
fig_xsino.show()

# %%
fig_xlm, _, widgets_xlm = show_vol_cuts(
    to_numpy_array(x_mlem_lm),
    vmin=0,
    vmax=vmax,
)
fig_xlm.show()
