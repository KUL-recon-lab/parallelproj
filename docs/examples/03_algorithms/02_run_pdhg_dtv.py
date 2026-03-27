"""
PDHG to optimize the Poisson logL and directional TV (structural prior)
=======================================================================

This example demonstrates the primal-dual hybrid gradient (PDHG) algorithm
applied to the problem

.. math::
    \\min_x \\; f_\\text{data}(Ax + s) + \\beta \\, f_\\text{reg}(Dx) + g(x)

where

- :math:`f_\\text{data} = \\text{NegPoissonLogL}` -- the negative Poisson log-likelihood,
- :math:`f_\\text{reg} = \\text{MixedL21Norm}` -- the isotropic mixed L2-L1 norm (TV semi-norm),
- :math:`g = \\iota_{\\geq 0}` -- the indicator function of the non-negative orthant,
- :math:`D = P_{\\xi} G` -- the projected finite-difference gradient operator
  implementing a directional total variation (DTV) structural prior,
- :math:`A` -- the composite PET forward operator (resolution model, TOF projector,
  attenuation), and
- :math:`s` -- the contamination sinogram.

The example uses simulated TOF sinogram data with a synthetic elliptic-cylinder
phantom and a structural prior image derived from the ground-truth activity.
MLEM is run for a small number of iterations to obtain a warm start for PDHG.

See :cite:p:`Ehrhardt2016` and :cite:p:`Ehrhardt2019` for details on the DTV prior,
and :cite:p:`Schramm2022` for the step-size rules used here.
"""

# %%
from __future__ import annotations
from collections.abc import Sequence
import matplotlib.pyplot as plt
from vis import show_vol_cuts
from img import elliptic_cylinder_phantom
import numpy as np

import parallelproj.operators
import parallelproj.functions
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, Array

# %%
from array_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)


# %%
# PDHG update
# -----------
#
# .. admonition:: PDHG algorithm
#
#   Minimize :math:`f_\text{data}(Ax + s) + \beta \, f_\text{reg}(Dx) + g(x)`, where
#   :math:`f_\text{data}` and :math:`f_\text{reg}` have known proximal operators of
#   their convex conjugates and :math:`g = \iota_{\geq 0}` has a known proximal operator.
#
#   | **Input** data :math:`d`, operators :math:`A`, :math:`D`
#   | **Initialize** primal :math:`x`, dual :math:`y`, :math:`w`; step sizes :math:`S_A`, :math:`S_D`, :math:`T`
#   | **Preprocessing** :math:`z = \bar{z} = A^T y + D^T w`
#   | **Repeat** until stopping criterion is met
#   |     :math:`x \;\gets\; \operatorname{prox}_{T g}(x - T\bar{z})`
#   |     :math:`y^+ \gets \operatorname{prox}_{S_A f_\text{data}^*}(y + S_A (Ax + s))`
#   |     :math:`w^+ \gets \operatorname{prox}_{S_D (\beta f_\text{reg})^*}(w + S_D D x)`
#   |     :math:`\Delta z \gets A^T(y^+ - y) + D^T(w^+ - w)`
#   |     :math:`y \gets y^+, \quad w \gets w^+`
#   |     :math:`z \gets z + \Delta z`
#   |     :math:`\bar{z} \gets z + \Delta z`
#   | **Return** :math:`x`
#
# .. admonition:: Step sizes
#
#  :math:`S_A = \gamma \, \text{diag}\!\left(\frac{\rho}{A \mathbf{1}}\right)`
#
#  :math:`S_D = \gamma \, \frac{\rho}{\|D\|}`
#
#  :math:`T_A = \gamma^{-1} \text{diag}\!\left(\frac{\rho}{A^T \mathbf{1}}\right)`
#
#  :math:`T_D = \gamma^{-1} \frac{\rho}{\|D\|}`
#
#  :math:`T = \min(T_A, T_D)` elementwise
#
# See :cite:p:`Ehrhardt2019` and :cite:p:`Schramm2022` for more details.


def pdhg_update(
    x: Array,
    dual_vars: list[Array],  # modified in-place
    z_array: Array,
    zbar_array: Array,
    f_functions: Sequence[parallelproj.functions.FunctionWithConjProx],
    ops: Sequence[parallelproj.operators.LinearOperator],
    contams: Sequence[Array | None],
    g_function: parallelproj.functions.FunctionWithProx,
    dual_step_sizes: Sequence[float | Array],
    primal_step_size: float | Array,
) -> tuple[Array, Array, Array]:
    """Perform one PDHG iteration for problems with multiple linear operators.

    Executes one update step of the general PDHG scheme for problems of the form

    .. math::

        \\min_x \\; \\sum_{i} f_i(K_i x + c_i) + g(x)

    where each :math:`f_i` has a known proximal operator of its convex conjugate
    and :math:`g` has a known proximal operator.  For this example the sum has
    two terms: :math:`f_\\text{data}(Ax + s)` and :math:`\\beta f_\\text{reg}(Dx)`.

    The primal variable is updated via

    .. math::

        x \\gets \\operatorname{prox}_{T g}(x - T \\bar{z})

    Each dual variable :math:`y_i` is updated via

    .. math::

        y_i^+ \\gets \\operatorname{prox}_{S_i f_i^*}(y_i + S_i (K_i x + c_i))

    and the auxiliary variables :math:`z`, :math:`\\bar{z}` are then updated as

    .. math::

        \\Delta z = \\sum_i K_i^T (y_i^+ - y_i), \\quad
        z \\gets z + \\Delta z, \\quad
        \\bar{z} \\gets z + \\Delta z.

    Parameters
    ----------
    x :
        Current primal variable.  Modified in-place during the primal update.
    dual_vars :
        List of current dual variables :math:`y_i`, one per term in
        ``f_functions``.  Updated in-place.
    z_array :
        Auxiliary variable :math:`z = \\sum_i K_i^T y_i` accumulating the
        adjoints of the dual variables.
    zbar_array :
        Over-relaxed auxiliary variable :math:`\\bar{z}` used in the primal
        gradient step.
    f_functions :
        Functions :math:`f_i`, each exposing a proximal operator of their
        convex conjugate via :meth:`~parallelproj.functions.FunctionWithConjProx.prox_convex_conj`.
    ops :
        Linear operators :math:`K_i`, one per function in ``f_functions``.
    contams :
        Additive offsets :math:`c_i` applied after each forward projection
        :math:`K_i x`.  Pass ``None`` for terms without an offset.
    g_function :
        Proximal-friendly constraint or regularization function :math:`g`
        applied to the primal variable, exposing
        :meth:`~parallelproj.functions.FunctionWithProx.prox`.
    dual_step_sizes :
        Dual step sizes :math:`S_i`, one per function/operator pair.
    primal_step_size :
        Primal step size :math:`T`.

    Returns
    -------
    x :
        Updated primal variable.
    z_array :
        Updated auxiliary variable :math:`z`.
    zbar_array :
        Updated over-relaxed auxiliary variable :math:`\\bar{z}`.
    """
    # prox of g function (e.g. non-negativity indicator)
    x -= primal_step_size * zbar_array
    x = g_function.prox(x, primal_step_size)

    y_plus = []

    for i_f, f in enumerate(f_functions):
        # dual update for the i-th term
        fwd = ops[i_f](x)
        if contams[i_f] is not None:
            fwd += contams[i_f]

        y_plus.append(
            f.prox_convex_conj(
                dual_vars[i_f] + dual_step_sizes[i_f] * fwd, dual_step_sizes[i_f]
            )
        )

    delta_z = xp.zeros_like(z_array)

    for i_o, op in enumerate(ops):
        delta_z += op.adjoint(y_plus[i_o] - dual_vars[i_o])
        dual_vars[i_o] = y_plus[i_o]

    z_array += delta_z
    zbar_array = z_array + delta_z

    return x, z_array, zbar_array


# %%
# **Input Parameters**

# image scale (can be used to simulate more or less counts)
img_scale = 0.1
# number of MLEM iterations used to initialize PDHG
num_iter_mlem = 20
# number of PDHG iterations
num_iter_pdhg = 200 if dev == "cpu" else 1000
# regularization weight
beta = 6.0
# step size ratio for PDHG
gamma = 10.0 / img_scale
# rho value for PDHG
rho = 0.9999
# contamination in every sinogram bin relative to mean of trues sinogram
contam = 1.0

track_cost = True

# %%
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated sinogram data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram.

# %%
# Setup of the sinogram forward model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model.

num_rings = 2
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=350.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-2.5, 2.5, num_rings, device=dev),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 4)
voxel_size = (4.0, 4.0, 2.5)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=170,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

x_true = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# setup a structural prior image
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_struct = -1.0 * xp.sqrt(x_true)
x_struct[x_true == 3] = -1.0

# scale image to get more counts
x_true *= img_scale


# %%
# Attenuation image and sinogram setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# %%
# Complete sinogram PET forward model setup
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We combine an image-based resolution model,
# a non-TOF or TOF PET projector and an attenuation model
# into a single linear operator.

# enable TOF - uncomment if you want to run TOF recons
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=10, tofbin_width=24.0, sigma_tof=24.0
)

# For TOF, att_sino has no TOF-bins dimension while the projector output does.
# broadcast_to adds a trailing singleton via expand_dims and broadcasts it over
# the TOF-bins axis without copying data (zero-stride view).
att_values = (
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
    if proj.tof
    else att_sino
)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape, sigma=[4.0 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# Simulation of sinogram projection data
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`d` by adding Poisson noise.

# simulated noise-free data
noise_free_data = pet_lin_op(x_true)

# generate a constant contamination sinogram
contamination = xp.full(
    noise_free_data.shape,
    contam * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

# add Poisson noise
np.random.seed(1)
d = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

# %%
# Run quick MLEM as initialization
# --------------------------------

x_mlem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^T 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter_mlem):
    print(f"MLEM iteration {(i + 1):03} / {num_iter_mlem:03}", end="\r")
    dbar = pet_lin_op(x_mlem) + contamination
    x_mlem *= pet_lin_op.adjoint(d / dbar) / adjoint_ones

# %%
# Setup the regularization operator and function objects
# ------------------------------------------------------
#
# The finite-difference gradient operator :math:`G` is projected by
# :math:`P_{\xi}` to obtain the DTV operator :math:`D = P_{\xi} G`.
# Three function objects handle all prox evaluations during PDHG:
#
# - ``data_fid`` -- :class:`.NegPoissonLogL`, implements :math:`f_\text{data}`,
# - ``nonneg``   -- :class:`.NonNegativeIndicator`, implements :math:`g = \iota_{\geq 0}`,
# - ``reg``      -- :class:`.MixedL21Norm` (weighted by ``beta``), implements :math:`\beta f_\text{reg}`.
#
# To use a different regularizer, replace ``op_D`` and ``reg`` accordingly.

# setup the finite-difference gradient operator
G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)
# calculate the joint vector field from the structural prior image
joint_vector_field = G(x_struct)
# setup the projected gradient (DTV) operator
P = parallelproj.operators.GradientFieldProjectionOperator(joint_vector_field, eta=1e-4)
D = parallelproj.operators.CompositeLinearOperator((P, G))

# function objects used by PDHG
data_fid = parallelproj.functions.NegPoissonLogL(d)
nonneg = parallelproj.functions.NonNegativeIndicator()
reg = parallelproj.functions.MixedL21Norm(beta=beta)

# %%
# Setup the cost function
# ^^^^^^^^^^^^^^^^^^^^^^^
#
# The total cost is :math:`f_\text{data}(Ax + s) + \beta \, f_\text{reg}(Dx) + g(x)`
# (the indicator :math:`g(x)` is zero for feasible :math:`x \geq 0`).


# %%
# initialize primal and dual variables
x_pdhg = 1.0 * x_mlem
y = 1 - d / (pet_lin_op(x_pdhg) + contamination)
w = xp.zeros(D.out_shape, dtype=xp.float32, device=dev)

z = pet_lin_op.adjoint(y) + D.adjoint(w)
zbar = 1.0 * z

# %%
# calculate PDHG step sizes
tmp = pet_lin_op(xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev))
tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
S_A = gamma * rho / tmp

T_A = (
    (1 / gamma)
    * rho
    / pet_lin_op.adjoint(xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev))
)

D_norm = D.norm(xp, dev, num_iter=100)
S_D = gamma * rho / D_norm
T_D = (1 / gamma) * rho / D_norm

T = xp.where(
    T_A < T_D, T_A, xp.full(pet_lin_op.in_shape, T_D, device=dev, dtype=xp.float32)
)


# %%
# Run PDHG
# --------


ys = [y, w]
fs = (data_fid, reg)
ops = (pet_lin_op, D)
cons = (contamination, None)

cost_pdhg = np.zeros(num_iter_pdhg, dtype=np.float32)

for i in range(num_iter_pdhg):
    x_pdhg, z, zbar = pdhg_update(
        x_pdhg,
        ys,
        z,
        zbar,
        fs,
        ops,
        cons,
        nonneg,
        (S_A, S_D),
        T,
    )

    if track_cost:
        cost = 0
        for k, f in enumerate(fs):
            x_fwd = ops[k](x_pdhg)
            if cons[k] is not None:
                x_fwd += cons[k]
            cost += f(x_fwd)

        cost += nonneg(x_pdhg)

        cost_pdhg[i] = cost
        print(
            f"PDHG iter {(i+1):04} / {num_iter_pdhg}, cost {cost_pdhg[i]:.7e}", end="\r"
        )

print("")

# %%
# Visualizations
# --------------

vmax = 1.2 * float(xp.max(x_true))

# %%
fig_true, _, widgets_true = show_vol_cuts(
    to_numpy_array(x_true),
    voxel_size=voxel_size,
    vmin=0,
    vmax=vmax,
    fig_title="true image",
)
fig_true.show()

# %%
fig_mlem, _, widgets_mlem = show_vol_cuts(
    to_numpy_array(x_mlem),
    voxel_size=voxel_size,
    vmin=0,
    vmax=vmax,
    fig_title=f"MLEM {num_iter_mlem} iterations",
)
fig_mlem.show()

# %%
fig_pdhg, _, widgets_pdhg = show_vol_cuts(
    to_numpy_array(x_pdhg),
    voxel_size=voxel_size,
    vmin=0,
    vmax=vmax,
    fig_title=f"DTV PDHG {num_iter_pdhg} iterations",
)
fig_pdhg.show()

# %%
fig_struct, _, widgets_struct = show_vol_cuts(
    to_numpy_array(x_struct), voxel_size=voxel_size, fig_title="structural image"
)
fig_struct.show()

# %%
if track_cost:
    epochs = np.arange(1, num_iter_pdhg + 1)
    fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
    ax2.semilogx(epochs, cost_pdhg, ".-", label="PDHG")
    ax2.grid(ls=":")
    ax2.legend()
    ax2.set_xlabel("iteration")
    ax2.set_title("cost", fontsize="medium")
    fig2.show()
