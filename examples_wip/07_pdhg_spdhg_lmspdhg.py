"""
PDHG, SPDHG and LM-SPHG to optimize the Poisson logL and total variation
========================================================================

This example demonstrates the use of the primal dual hybrid gradient (PDHG) algorithm,
the stochastic PDHG (SPDHG) and the listmode SPDHG (LM-SPDHG) to minimize the negative
Poisson log-likelihood function combined with a total variation regularizer:

.. math::
    f(x) = \sum_{i=1}^m \\bar{d}_i (x) - d_i \log(\\bar{d}_i (x)) + \\beta \\|\\nabla x \\|_{1,2}

subject to

.. math::
    x \geq 0

using the linear forward model

.. math::
    \\bar{d}(x) = A x + s

.. tip::
    parallelproj is python array API compatible meaning it supports different
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
from __future__ import annotations
from copy import copy

import array_api_compat.numpy as xp

# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp

import parallelproj
from array_api_compat import to_device
import array_api_compat.numpy as np
import matplotlib.pyplot as plt

# choose a device (CPU or CUDA GPU)
if "numpy" in xp.__name__:
    # using numpy, device must be cpu
    dev = "cpu"
elif "cupy" in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif "torch" in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    if parallelproj.cuda_present:
        dev = "cuda"
    else:
        dev = "cpu"

# %%
# **Define the number of iterations and subsets**

num_iter_mlem = 10
# number of PDHG iterations
num_iter_pdhg = 2000
# number of subsets for SPDHG and LM-SPDHG
num_subsets = 10
# prior weight
beta = 3e-1
# step size ratio for LM-SPDHG
gamma = 3.0 / 5.0
# rho value for LM-SPHDHG
rho = 0.9999

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

num_rings = 3
scanner = parallelproj.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=15,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-5, 5, num_rings),
    symmetry_axis=2,
)

# setup the LOR descriptor that defines the sinogram

img_shape = (40, 40, 5)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=40,
    max_ring_difference=2,
    sinogram_order=parallelproj.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

# setup a simple test image containing a few "hot rods"
x_true = xp.ones(proj.in_shape, device=dev, dtype=xp.float32)
c0 = proj.in_shape[0] // 2
c1 = proj.in_shape[1] // 2
x_true[(c0 - 2) : (c0 + 2), (c1 - 2) : (c1 + 2), :] = 5.0
x_true[4, c1, 2:] = 5.0
x_true[c0, 4, :-2] = 5.0

tmp_n = proj.in_shape[0] // 4
x_true[:tmp_n, :, :] = 0
x_true[-tmp_n:, :, :] = 0
x_true[:, :2, :] = 0
x_true[:, -2:, :] = 0

# scale image to get more counts
x_true *= 1.0

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
proj.tof_parameters = parallelproj.TOFParameters(
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
att_op = parallelproj.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.GaussianFilterOperator(
    proj.in_shape, sigma=[4.5 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.CompositeLinearOperator((att_op, proj, res_model))

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
d = xp.asarray(
    np.random.poisson(np.asarray(to_device(noise_free_data, "cpu"))),
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
# -----------------------


def cost_function(img):
    exp = pet_lin_op(img) + contamination
    res = float(xp.sum(exp - d * xp.log(exp)))
    res += beta * float(xp.sum(xp.linalg.vector_norm(op_G(img), axis=0)))
    return res


# %%
# PDHG update to minimize :math:`f(x)`
# ------------------------------------
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

op_G = parallelproj.FiniteForwardDifference(pet_lin_op.in_shape)

# initialize primal and dual variables
x_pdhg = 1.0 * x_mlem
y = 1 - d / (pet_lin_op(x_pdhg) + contamination)
w = beta * xp.sign(op_G(x_pdhg))

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
    / pet_lin_op.adjoint(xp.ones(pet_lin_op.out_shape, dtype=xp.float64, device=dev))
)

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm
T_G = (1 / gamma) * rho / op_G_norm

T = xp.where(T_A < T_G, T_A, xp.full(pet_lin_op.in_shape, T_G))


# %%
# Run PDHG
# --------
cost_pdhg = np.zeros(num_iter_pdhg, dtype=xp.float32)

for i in range(num_iter_pdhg):
    x_pdhg -= T * zbar
    x_pdhg = xp.where(x_pdhg < 0, xp.zeros_like(x_pdhg), x_pdhg)

    cost_pdhg[i] = cost_function(x_pdhg)

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

    print(f"PDHG iter {i:04} / {num_iter_pdhg}, cost {cost_pdhg[i]:.7e}", end="\r")

# %%
# Splitting of the forward model into subsets :math:`A_i`
# -------------------------------------------------------
#
# Calculate the view numbers and slices for each subset.
# We will use the subset views to setup a sequence of projectors projecting only
# a subset of views. The slices can be used to extract the corresponding subsets
# from full data or corrections sinograms.

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear the cached LOR endpoints since we will create many copies of the projector
proj.clear_cached_lor_endpoints()
pet_subset_linop_seq = []

# we setup a sequence of subset forward operators each constisting of
# (1) image-based resolution model
# (2) subset projector
# (3) multiplication with the corresponding subset of the attenuation sinogram
for i in range(num_subsets):
    # make a copy of the full projector and reset the views to project
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    # same TOF/non-TOF broadcasting as for the full operator above
    att_values_k = (
        xp.broadcast_to(
            xp.expand_dims(att_sino[subset_slices_non_tof[i]], axis=-1),
            subset_proj.out_shape,
        )
        if subset_proj.tof
        else att_sino[subset_slices_non_tof[i]]
    )
    subset_att_op = parallelproj.ElementwiseMultiplicationOperator(att_values_k)

    # add the resolution model and multiplication with a subset of the attenuation sinogram
    pet_subset_linop_seq.append(
        parallelproj.CompositeLinearOperator(
            [
                subset_att_op,
                subset_proj,
                res_model,
            ]
        )
    )

pet_subset_linop_seq = parallelproj.LinearOperatorSequence(pet_subset_linop_seq)

# %%
# SPDHG updates to minimize :math:`f(x)`
# --------------------------------------
#
# .. admonition:: SPDHG algorithm to minimize negative Poisson log-likelihood + regularization
#
#   | **Input** Poisson data :math:`d`
#   | **Initialize** :math:`x,y_i,w,S_{A_i},S_G,T,p_i`
#   | **Preprocessing** :math:`\overline{z} = z = \sum_i A_i^T y + \nabla^T w`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **select a random data subset number i or do prior update according to** :math:`p_i`
#   |       **Update** :math:`y_i^+ \gets \text{prox}_{D^*}^{S_{A_i}} ( y_i + S_{A_i}  ( {A_i} x + s))`
#   |       **Update** :math:`\Delta z \gets A_i^T (y_i^+ - y_i)`
#   |       **Update** :math:`y_i \gets y_i^+`
#   |     **or**
#   |       **Update** :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w + S_G  \nabla x)/\beta)`
#   |       **Update** :math:`\Delta z \gets \nabla^T (w^+ - w)`
#   |       **Update** :math:`w \gets w^+`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + (\Delta z \ p_i)`
#   | **Return** :math:`x`
#
# See :cite:p:`Ehrhardt2019` :cite:p:`Schramm2022` for more details.
#
# .. admonition:: Step sizes
#
#  :math:`S_{A_i} = \gamma \, \text{diag}(\frac{\rho}{A_i 1})`
#
#  :math:`S_G = \gamma \, \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T_{A_i} = \gamma^{-1} \text{diag}(\frac{\rho}{A_i^T 1})`
#
#  :math:`T_G = \gamma^{-1} \text{diag}(\frac{\rho}{|\nabla|})`
#
#  :math:`T = \min T_{A_i}, T_G` pointwise
#


# %%
# Initialize SPDHG primal and dual variables
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

p_g = 0.5
p_a = (1 - p_g) / len(pet_subset_linop_seq)

x_spdhg = 1.0 * x_mlem

ys = [
    (1 - d[sl] / (pet_subset_linop_seq[k](x_spdhg) + contamination[sl]))
    for k, sl in enumerate(subset_slices)
]

w = beta * xp.sign(op_G(x_spdhg))

z = pet_subset_linop_seq.adjoint(ys) + op_G.adjoint(w)
zbar = 1.0 * z

# %%
# calculate SPHDG step sizes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^

S_A = []
for op in pet_subset_linop_seq:
    tmp = op(xp.ones(op.in_shape, dtype=xp.float32, device=dev))
    tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
    S_A.append(gamma * rho / tmp)

subset_T = [
    (
        (1 / gamma)
        * rho
        * p_a
        / op.adjoint(xp.ones(op.out_shape, dtype=xp.float32, device=dev))
    )
    for op in pet_subset_linop_seq
]

# calculate the element wise min over all subsets

op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm

T_G = (1 / gamma) * rho * p_g / op_G_norm
subset_T.append(xp.full(pet_lin_op.in_shape, T_G))

T = xp.min(xp.asarray(subset_T), axis=0)

# %%
# Run SPDHG
# ^^^^^^^^^

num_iter_spdhg = 3 * (num_iter_pdhg // num_subsets)
cost_spdhg = np.zeros(num_iter_spdhg, dtype=xp.float32)

for i in range(num_iter_spdhg):
    x_spdhg -= T * zbar
    x_spdhg = xp.where(x_spdhg < 0, xp.zeros_like(x_spdhg), x_spdhg)

    cost_spdhg[i] = cost_function(x_spdhg)
    print(
        f"SPDHG iter {i:04} / {num_iter_spdhg}, cost {cost_spdhg[i]:.7e}",
        end="\r",
    )

    # select a random subset
    # in 50% of the cases we select a subset of the forward operator
    # in the other 50% select the gradient operator
    for i_ss in np.random.permutation(2 * num_subsets):
        if i_ss < num_subsets:
            sl = subset_slices[i_ss]
            y_plus = ys[i_ss] + S_A[i_ss] * (
                pet_subset_linop_seq[i_ss](x_spdhg) + contamination[sl]
            )
            # prox of convex conjugate of negative Poisson logL
            y_plus = 0.5 * (
                y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_A[i_ss] * d[sl])
            )

            delta_z = pet_subset_linop_seq[i_ss].adjoint(y_plus - ys[i_ss])
            ys[i_ss] = y_plus
            p = p_a
        else:
            w_plus = (w + S_G * op_G(x_spdhg)) / beta
            # prox of convex conjugate of TV
            denom = xp.linalg.vector_norm(w_plus, axis=0)
            w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
            w_plus *= beta

            delta_z = op_G.adjoint(w_plus - w)
            w = 1.0 * w_plus
            p = p_g

        z = z + delta_z
        zbar = z + delta_z / p

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

print("\nGenerating LM events")
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
# --------------------------------------------------------------

# slices that define which elements of the event list belong to each subset
# here every "num_subset-th element" is used
subset_slices_lm = [slice(i, None, num_subsets) for i in range(num_subsets)]

lm_pet_subset_linop_seq = []

for i, sl in enumerate(subset_slices_lm):
    subset_lm_proj = parallelproj.ListmodePETProjector(
        event_start_coords[sl, :],
        event_end_coords[sl, :],
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

    subset_lm_att_op = parallelproj.ElementwiseMultiplicationOperator(subset_att_list)

    lm_pet_subset_linop_seq.append(
        parallelproj.CompositeLinearOperator(
            (subset_lm_att_op, subset_lm_proj, res_model)
        )
    )

lm_pet_subset_linop_seq = parallelproj.LinearOperatorSequence(lm_pet_subset_linop_seq)

# create the contamination list
contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, -1)[0]),
    device=dev,
    dtype=xp.float32,
)

# %%
# Calculate event multiplicity :math:`\mu` for each event in the list
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
events = xp.concat(
    [event_start_coords, event_end_coords, xp.expand_dims(event_tofbins, -1)], axis=1
)
mu = parallelproj.count_event_multiplicity(events)

# %%
# Listmode SPDHG
# --------------
#
# .. admonition:: Listmode SPDHG algorithm to minimize negative Poisson log-likelihood
#
#   | **Input** event list :math:`N`, contamination list :math:`s_N`
#   | **Calculate** event counts :math:`\mu_e` for each :math:`e` in :math:`N`
#   | **Initialize** :math:`x,(S_i)_i,T,(p_i)_i`
#   | **Initialize list** :math:`y_{N} = 1 - (\mu_N /(A^{LM}_{N} x + s_{N}))`
#   | **Preprocessing** :math:`\overline{z} = z = {A^T} 1 - {A^{LM}_N}^T (y_N-1)/\mu_N`
#   | **Split lists** :math:`N`, :math:`s_N` and :math:`y_N` into :math:`n` sublists :math:`N_i`, :math:`y_{N_i}` and :math:`s_{N_i}`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update** :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Select** :math:`i \in \{ 1,\ldots,n+1\}` randomly according to :math:`(p_i)_i`
#   |     **Update** :math:`y_{N_i}^+ \gets \text{prox}_{D^*}^{S_i} \left( y_{N_i} + S_i \left(A^{LM}_{N_i} x + s^{LM}_{N_i} \right) \right)`
#   |     **Update** :math:`\Delta z \gets {A^{LM}_{N_i}}^T \left(\frac{y_{N_i}^+ - y_{N_i}}{\mu_{N_i}}\right)`
#   |     **Update** :math:`y_{N_i} \gets y_{N_i}^+`
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

# setup gradient operator
op_G = parallelproj.FiniteForwardDifference(pet_lin_op.in_shape)
# initialize dual variable for the gradient
w = beta * xp.sign(op_G(x_lmspdhg))

z = 1.0 * adjoint_ones
for k, sl in enumerate(subset_slices_lm):
    z += lm_pet_subset_linop_seq[k].adjoint((ys[k] - 1) / mu[sl])
    tmp = lm_pet_subset_linop_seq[k].adjoint(1 / mu[sl])
z += op_G.adjoint(w)
zbar = 1.0 * z

# %%
# Calculate the step sizes
# ^^^^^^^^^^^^^^^^^^^^^^^^

S_A = []
ones_img = xp.ones(img_shape, dtype=xp.float32, device=dev)

for lm_op in lm_pet_subset_linop_seq:
    tmp = lm_op(ones_img)
    tmp = xp.where(tmp == 0, xp.min(tmp[tmp > 0]), tmp)
    S_A.append(gamma * rho / tmp)


# step size for the gradient operator
op_G_norm = op_G.norm(xp, dev, num_iter=100)
S_G = gamma * rho / op_G_norm

T_A = xp.zeros((num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32)
for k, sl in enumerate(subset_slices_lm):
    tmp = lm_pet_subset_linop_seq[k].adjoint(1 / mu[sl])
    T_A[k] = (rho * p_a / gamma) / tmp
T_A = xp.min(T_A, axis=0)

T_G = (rho * p_g / gamma) / op_G_norm
T = xp.where(T_A < T_G, T_A, xp.full(T_A.shape, T_G))

# %%
# Run LM-SPDHG
# ^^^^^^^^^^^^

cost_lmspdhg = np.zeros(num_iter_spdhg, dtype=xp.float32)

for i in range(num_iter_spdhg):
    subset_sequence = np.random.permutation(2 * num_subsets)

    cost_lmspdhg[i] = cost_function(x_lmspdhg)
    print(
        f"LM-SPDHG iter {i:04} / {num_iter_spdhg}, cost {cost_lmspdhg[i]:.7e}",
        end="\r",
    )

    for k in subset_sequence:
        x_lmspdhg -= T * zbar
        x_lmspdhg = xp.where(x_lmspdhg < 0, xp.zeros_like(x_lmspdhg), x_lmspdhg)

        if k < num_subsets:
            sl = subset_slices_lm[k]
            y_plus = ys[k] + S_A[k] * (
                lm_pet_subset_linop_seq[k](x_lmspdhg) + contamination_list[sl]
            )
            y_plus = 0.5 * (
                y_plus + 1 - xp.sqrt((y_plus - 1) ** 2 + 4 * S_A[k] * mu[sl])
            )
            dz = lm_pet_subset_linop_seq[k].adjoint((y_plus - ys[k]) / mu[sl])
            ys[k] = y_plus
            p = p_a
        else:
            w_plus = (w + S_G * op_G(x_lmspdhg)) / beta
            # prox of convex conjugate of TV
            denom = xp.linalg.vector_norm(w_plus, axis=0)
            w_plus /= xp.where(denom < 1, xp.ones_like(denom), denom)
            w_plus *= beta
            dz = op_G.adjoint(w_plus - w)
            w = 1.0 * w_plus
            p = p_g

        z = z + dz
        zbar = z + (dz / p)


# %%
# Show the results
# ^^^^^^^^^^^^^^^^

x_true_np = parallelproj.to_numpy_array(x_true)
x_mlem_np = parallelproj.to_numpy_array(x_mlem)
x_pdhg_np = parallelproj.to_numpy_array(x_pdhg)
x_spdhg_np = parallelproj.to_numpy_array(x_spdhg)
x_lmspdhg_np = parallelproj.to_numpy_array(x_lmspdhg)

pl2 = x_true_np.shape[2] // 2
pl1 = x_true_np.shape[1] // 2
pl0 = x_true_np.shape[0] // 2

fig, ax = plt.subplots(2, 5, figsize=(12, 4), tight_layout=True)
vmax = 1.2 * x_true_np.max()
ax[0, 0].imshow(x_true_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)
ax[0, 1].imshow(x_mlem_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)
ax[0, 2].imshow(x_pdhg_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)
ax[0, 3].imshow(x_spdhg_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)
ax[0, 4].imshow(x_lmspdhg_np[:, :, pl2], cmap="Greys", vmin=0, vmax=vmax)

ax[1, 0].imshow(x_true_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)
ax[1, 1].imshow(x_mlem_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)
ax[1, 2].imshow(x_pdhg_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)
ax[1, 3].imshow(x_spdhg_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)
ax[1, 4].imshow(x_lmspdhg_np[pl0, :, :].T, cmap="Greys", vmin=0, vmax=vmax)

ax[0, 0].set_title("true img")
ax[0, 1].set_title("init img")
ax[0, 2].set_title("PDHG")
ax[0, 3].set_title("SPDHG")
ax[0, 4].set_title("LM-SPDHG")
fig.show()

# %%
fig2, ax2 = plt.subplots(1, 1, figsize=(6, 4), tight_layout=True)
ax2.plot(cost_pdhg, ".-", label="PDHG")
ax2.plot(cost_spdhg, ".-", label="SPDHG")
ax2.plot(cost_lmspdhg, ".-", label="LM-SPDHG")
ax2.grid(ls=":")
ax2.legend()
ax2.set_ylim((cost_pdhg[10:].min(), cost_pdhg.max()))
fig2.show()
