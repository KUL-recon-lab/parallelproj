"""
PDHG and LM-SPHG to optimize the Poisson logL and total variation
=================================================================

This example demonstrates the use of the primal dual hybrid gradient (PDHG) algorithm,
the listmode stochastic PDHG (LM-SPDHG) to minimize the negative
Poisson log-likelihood function combined with a total variation regularizer:

.. math::
    f(x) = \\sum_{i=1}^m \\bar{d}_i (x) - d_i \\log(\\bar{d}_i (x)) + \\beta \\|\\nabla x \\|_{1,2}

subject to

.. math::
    x \\geq 0

using the linear forward model

.. math::
    \\bar{d}(x) = A x + s

.. warning::
    Running this example using GPU arrays (e.g. using cupy as array backend)
    is highly recommended due to "longer" execution times with CPU arrays
"""

# %%
from __future__ import annotations
import matplotlib.pyplot as plt
from vis import show_vol_cuts
from img import elliptic_cylinder_phantom
import numpy as np
import math
from array_api_compat import size

import parallelproj.operators
import parallelproj.tof
import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array, count_event_multiplicity

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
# **Input Parameters**

# image scale (can be used to simulated more or less counts)
img_scale = 0.1
# number of MLEM iterations to init. PDHG and LM-SPDHG
num_iter_mlem = 10
# number of PDHG iterations
num_iter_pdhg = 2000
# number of subsets for SPDHG and LM-SPDHG
num_subsets = 10
# number of iterations for stochastic PDHGs
num_iter_spdhg = 200
# prior weight
beta = 10.0
# step size ratio for LM-SPDHG
gamma = 1.0 / img_scale
# rho value for LM-SPHDHG
rho = 0.9999
# contaminaton in every sinogram bin relative to mean of trues sinogram
contam = 1.0


# subset probabilities for SPDHG
p_g = 0.5  # gradient update
p_a = (1 - p_g) / num_subsets  # data subset update

track_cost = True

# %%
# Simulation of PET data in sinogram space
# ----------------------------------------
#
# In this example, we use simulated listmode data for which we first
# need to setup a sinogram forward model to create a noise-free and noisy
# emission sinogram that can be converted to listmode data.

# %%
# Setup of the sinogram forward model
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# We setup a linear forward operator :math:`A` consisting of an
# image-based resolution model, a non-TOF PET projector and an attenuation model
#

num_rings = 5
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=350.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 5)
voxel_size = (4.0, 4.0, 4.0)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=170,
    max_ring_difference=num_rings - 1,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

x_true = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

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

# enable TOF - comment if you want to run non-TOF
proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=17, tofbin_width=12.0, sigma_tof=12.0
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
    dtype=xp.int16,
)

# %%
# Run quick MLEM as initialization
# --------------------------------

x_mlem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
# calculate A^H 1
adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

for i in range(num_iter_mlem):
    print(f"MLEM iteration {(i + 1):03} / {num_iter_mlem:03}", end="\r")
    dbar = pet_lin_op(x_mlem) + contamination
    x_mlem *= pet_lin_op.adjoint(d / dbar) / adjoint_ones

# %%
# Setup the cost function
# ^^^^^^^^^^^^^^^^^^^^^^^


def cost_function(img):
    exp = pet_lin_op(img) + contamination
    res = float(xp.sum(exp - d * xp.log(exp)))
    res += beta * float(xp.sum(xp.linalg.vector_norm(op_G(img), axis=0)))
    return res


# %%
# PDHG
# ----
#
# .. admonition:: PDHG algorithm to minimize negative Poisson log-likelihood + regularization
#
#   | **Input** Poisson data :math:`d`
#   | **Initialize** :math:`x,y,w,S_A,S_G,T`
#   | **Preprocessing** :math:`\overline{z} = z = A^T y + \nabla^T w`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Update** :math:`y^+ \gets \text{prox}_{D^*}^{S_A} ( y + S_A  ( A x + s))`
#   |     **Update** :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta)`
#   |     **Update** :math:`\Delta z \gets A^T (y^+ - y) + \nabla^T (w^+ - w)`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + \Delta z`
#   |     **Update** :math:`y \gets y^+`
#   |     **Update** :math:`w \gets w^+`
#   | **Return** :math:`x`
#
# See :cite:p:`Ehrhardt2019` :cite:p:`Schramm2022` for more details.
#
# .. admonition:: Proximal operator of the convex dual of the negative Poisson log-likelihood
#
#  :math:`(\text{prox}_{D^*}^{S}(y))_i = \text{prox}_{D^*}^{S}(y_i) = \frac{1}{2} \left(y_i + 1 - \sqrt{ (y_i-1)^2 + 4 S d_i} \right)`
#
# .. admonition:: Step sizes
#
#  :math:`S_A = \gamma \, \text{diag}(\frac{\rho}{A 1})`
#
#  :math:`S_G = \gamma \, \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T_A = \gamma^{-1} \text{diag}(\frac{\rho}{A^T 1})`
#
#  :math:`T_G = \gamma^{-1} \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T = \min T_A, T_G` pointwise
#

op_G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)

# initialize primal and dual variables
x_pdhg = 1.0 * x_mlem
y = 1 - d / (pet_lin_op(x_pdhg) + contamination)

# initialize dual variable for the gradient
w = xp.zeros(op_G.out_shape, dtype=xp.float32, device=dev)

z = pet_lin_op.adjoint(y) + op_G.adjoint(w)
zbar = 1.0 * z

# %%

# calculate PHDG step sizes
tmp = pet_lin_op(xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev))
tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
S_A = gamma * rho / tmp

T_A = (
    (1 / gamma)
    * rho
    / pet_lin_op.adjoint(xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev))
)

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm
T_G = (1 / gamma) * rho / op_G_norm

T = xp.where(
    T_A < T_G, T_A, xp.full(pet_lin_op.in_shape, T_G, device=dev, dtype=xp.float32)
)


# %%
# Run PDHG
# ^^^^^^^^

print("")
cost_pdhg = np.zeros(num_iter_pdhg, dtype=np.float32)

for i in range(num_iter_pdhg):
    x_pdhg -= T * zbar
    x_pdhg = xp.where(x_pdhg < 0, xp.zeros_like(x_pdhg), x_pdhg)

    if track_cost:
        cost_pdhg[i] = cost_function(x_pdhg)

    if i == (num_iter_spdhg - 1):
        x_pdhg_early = 1.0 * x_pdhg

    y_plus = y + S_A * (pet_lin_op(x_pdhg) + contamination)
    # prox of convex conjugate of negative Poisson logL
    y_plus = 0.5 * (y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_A * d))

    w_plus = (w + S_G * op_G(x_pdhg)) / beta
    # prox of convex conjugate of TV
    denom = xp.linalg.vector_norm(w_plus, axis=0)
    w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
    w_plus *= beta

    delta_z = pet_lin_op.adjoint(y_plus - y) + op_G.adjoint(w_plus - w)
    y = 1.0 * y_plus
    w = 1.0 * w_plus

    z = z + delta_z
    zbar = z + delta_z

    print(f"PDHG iter {(i+1):04} / {num_iter_pdhg}, cost {cost_pdhg[i]:.7e}", end="\r")

# %%
# Conversion of the emission sinogram to listmode
# -----------------------------------------------
#
# Using :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` we can convert an
# integer non-TOF or TOF sinogram to an event list for listmode processing.
#
# .. warning::
#     **Note:** The created event list is "ordered" and should be shuffled depending on the
#     strategy to define subsets in LM-OSEM.

print(f"\nGenerating LM events ({float(xp.sum(d)):.2e})")
event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    d
)

# %%
# Shuffle the simulated "ordered" LM events
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

random_inds = np.random.permutation(event_start_coords.shape[0])
event_start_coords = event_start_coords[random_inds, :]
event_end_coords = event_end_coords[random_inds, :]
event_tofbins = event_tofbins[random_inds]

# %%
# Setup of the LM subset projectors and LM subset forward models
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

# slices that define which elements of the event list belong to each subset
# here every "num_subset-th element" is used
subset_slices_lm = [slice(i, None, num_subsets) for i in range(num_subsets)]

lm_pet_subset_linop_seq = []

for i, sl in enumerate(subset_slices_lm):
    subset_lm_proj = parallelproj.projectors.ListmodePETProjector(
        1 * event_start_coords[sl, :],
        1 * event_end_coords[sl, :],
        proj.in_shape,
        proj.voxel_size,
        proj.img_origin,
    )

    # recalculate the attenuation factor for all LM events
    # this needs to be a non-TOF projection
    subset_att_list = xp.exp(-subset_lm_proj(x_att))

    # enable TOF in the LM projector
    subset_lm_proj.tof_parameters = proj.tof_parameters
    if proj.tof:
        subset_lm_proj.event_tofbins = xp.asarray(event_tofbins[sl], copy=True)
        subset_lm_proj.tof = proj.tof

    subset_lm_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
        subset_att_list
    )

    lm_pet_subset_linop_seq.append(
        parallelproj.operators.CompositeLinearOperator(
            (subset_lm_att_op, subset_lm_proj, res_model)
        )
    )

lm_pet_subset_linop_seq = parallelproj.operators.LinearOperatorSequence(
    lm_pet_subset_linop_seq
)

# create the contamination list
contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=xp.float32,
)

# %%
# Calculate event multiplicity :math:`\mu` for each event in the list
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
events = xp.concat(
    [event_start_coords, event_end_coords, xp.expand_dims(event_tofbins, -1)], axis=1
)
mu = count_event_multiplicity(events)

# %%
# Listmode SPDHG
# --------------
#
# .. admonition:: Listmode SPDHG algorithm to minimize negative Poisson log-likelihood
#
#   | **Input** event list :math:`N`, contamination list :math:`s_N`
#   | **Calculate** event counts :math:`\mu_e` for each :math:`e` in :math:`N`
#   | **Initialize** :math:`x,(S_i)_i,S_G,T,(p_i)_i`
#   | **Initialize list** :math:`y_{N} = 1 - (\mu_N /(A^{LM}_{N} x + s_{N}))`
#   | **Preprocessing** :math:`\overline{z} = z = {A^T} 1 - {A^{LM}_N}^T (y_N-1)/\mu_N`
#   | **Split lists** :math:`N`, :math:`s_N` and :math:`y_N` into :math:`n` sublists :math:`N_i`, :math:`y_{N_i}` and :math:`s_{N_i}`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Select** :math:`i \in \{ 1,\ldots,n+1\}` randomly according to :math:`(p_i)_i`
#   |     **if** :math:`i \leq n`:
#   |         **Update** :math:`y_{N_i}^+ \gets \text{prox}_{D^*}^{S_i} \left( y_{N_i} + S_i \left(A^{LM}_{N_i} x + s^{LM}_{N_i} \right) \right)`
#   |         **Update** :math:`\Delta z \gets {A^{LM}_{N_i}}^T \left(\frac{y_{N_i}^+ - y_{N_i}}{\mu_{N_i}}\right)`
#   |         **Update** :math:`y_{N_i} \gets y_{N_i}^+`
#   |     **else:**
#   |         **Update** :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta)`
#   |         **Update** :math:`\Delta z \gets \nabla^T (w^+ - w)`
#   |         **Update** :math:`w \gets w+`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + (\Delta z/p_i)`
#   | **Return** :math:`x`
#
# .. admonition:: Step sizes
#
#  :math:`S_i = \gamma \, \text{diag}(\frac{\rho}{A^{LM}_{N_i} 1})`
#
#  :math:`T_i = \gamma^{-1} \text{diag}(\frac{\rho p_i}{{A^{LM}_{N_i}}^T 1/\mu_{N_i}})`
#
#  :math:`T = \min_{i=1,\ldots,n+1} T_i` pointwise
#

# %%
# Initialize variables
# ^^^^^^^^^^^^^^^^^^^^

# Intialize image x with solution from quick LM OSEM
x_lmspdhg = 1.0 * x_mlem

# setup dual variable for data subsets
ys = []
for k, sl in enumerate(subset_slices_lm):
    ys.append(
        1 - (mu[sl] / (lm_pet_subset_linop_seq[k](x_lmspdhg) + contamination_list[sl]))
    )

# initialize dual variable for the gradient
w_lm = xp.zeros(op_G.out_shape, dtype=xp.float32, device=dev)

z = 1.0 * adjoint_ones
for k, sl in enumerate(subset_slices_lm):
    z += lm_pet_subset_linop_seq[k].adjoint((ys[k] - 1) / mu[sl])
    # tmp = lm_pet_subset_linop_seq[k].adjoint(1 / mu[sl])
z += op_G.adjoint(w_lm)
zbar = 1.0 * z

# %%
# Calculate the step sizes
# ^^^^^^^^^^^^^^^^^^^^^^^^

S_A_lm = []
ones_img = xp.ones(img_shape, dtype=xp.float32, device=dev)

for lm_op in lm_pet_subset_linop_seq:
    tmp = lm_op(ones_img)
    tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
    S_A_lm.append(gamma * rho / tmp)


T_A_lm = xp.zeros(
    (num_subsets + 1,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, sl in enumerate(subset_slices_lm):
    tmp = lm_pet_subset_linop_seq[k].adjoint(1 / mu[sl])
    T_A_lm[k] = (rho * p_a / gamma) / tmp
T_A_lm[-1] = T_G
T_lm = xp.min(T_A_lm, axis=0)

# %%
# Run LM-SPDHG
# ^^^^^^^^^^^^

print("")
cost_lmspdhg = np.zeros(num_iter_spdhg, dtype=np.float32)
psnr_lmspdhg = np.zeros(num_iter_spdhg, dtype=np.float32)

psnr_scale = float(xp.max(x_true))

for i in range(num_iter_spdhg):
    subset_sequence = np.random.permutation(2 * num_subsets)

    psnr_lmspdhg[i] = 10 * math.log10(
        (psnr_scale**2) / float(xp.mean((x_lmspdhg - x_pdhg) ** 2))
    )

    if track_cost:
        cost_lmspdhg[i] = cost_function(x_lmspdhg)
    print(
        f"LM-SPDHG iter {(i+1):04} / {num_iter_spdhg}, cost {cost_lmspdhg[i]:.7e}",
        end="\r",
    )

    for k in subset_sequence:
        x_lmspdhg -= T_lm * zbar
        x_lmspdhg = xp.where(x_lmspdhg < 0, xp.zeros_like(x_lmspdhg), x_lmspdhg)

        if k < num_subsets:
            sl = subset_slices_lm[k]
            y_plus = ys[k] + S_A_lm[k] * (
                lm_pet_subset_linop_seq[k](x_lmspdhg) + contamination_list[sl]
            )
            y_plus = 0.5 * (
                y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_A_lm[k] * mu[sl])
            )
            dz = lm_pet_subset_linop_seq[k].adjoint((y_plus - ys[k]) / mu[sl])
            ys[k] = y_plus
            p = p_a
        else:
            w_plus = (w_lm + S_G * op_G(x_lmspdhg)) / beta
            # prox of convex conjugate of TV
            denom = xp.linalg.vector_norm(w_plus, axis=0)
            w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
            w_plus *= beta
            dz = op_G.adjoint(w_plus - w_lm)
            w_lm = 1.0 * w_plus
            p = p_g

        z = z + dz
        zbar = z + (dz / p)


# %%
# Visualizations
# --------------

x_true_np = to_numpy_array(x_true)
x_mlem_np = to_numpy_array(x_mlem)
x_pdhg_np = to_numpy_array(x_pdhg)
x_pdhg_early_np = to_numpy_array(x_pdhg_early)
x_lmspdhg_np = to_numpy_array(x_lmspdhg)

vmax = 1.2 * x_true_np.max()

for vol_np, title in [
    (x_true_np, "true image"),
    (x_mlem_np, "init image (MLEM)"),
    (x_pdhg_np, f"PDHG {num_iter_pdhg} it. (reference)"),
    (x_lmspdhg_np, f"LM-SPDHG {num_iter_spdhg} it. / {num_subsets} subsets"),
    (x_pdhg_early_np, f"PDHG {num_iter_spdhg} it."),
]:
    fig_i, _, widgets_i = show_vol_cuts(
        vol_np, voxel_size=voxel_size, vmin=0, vmax=vmax, fig_title=title
    )
    fig_i.show()

# %%

if track_cost:
    fig2, ax2 = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
    for i in range(2):
        ax2[i].plot(cost_pdhg, ".-", label="PDHG")
        ax2[i].plot(cost_lmspdhg, ".-", label="LM-SPDHG")
        ax2[i].grid(ls=":")
        ax2[i].legend()
        ax2[i].set_ylim(None, cost_pdhg[10:].max())
    ax2[1].set_xlim(0, num_iter_spdhg)
    ax2[2].plot(psnr_lmspdhg, ".-")
    ax2[2].grid(ls=":")
    for axx in ax2.ravel():
        axx.set_xlabel("iteration")
    ax2[0].set_title("cost", fontsize="medium")
    ax2[1].set_title("cost (zoom)", fontsize="medium")
    ax2[2].set_title("PSNR LM-SPDHG vs ref", fontsize="medium")
    fig2.show()
