"""
PDHG and listmode SPDHG with a total variation prior
====================================================

:bdg-primary:`reconstruction` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

The stochastic primal-dual hybrid gradient algorithm (SPDHG) can be run
directly on listmode data (LM-SPDHG): the dual variables then live on the
events instead of the sinogram bins, which saves memory and time for sparse
(e.g. TOF) data.  In this example we minimize the negative Poisson
log-likelihood plus a total variation (TV) prior with LM-SPDHG and compare it
with a converged sinogram PDHG reconstruction.

**You will learn**

- how the dual update of SPDHG is written for an event list, using the
  multiplicity of every event,
- how to build listmode subset forward models (TOF, attenuation and resolution
  model) from a simulated event list,
- how to choose the step sizes of PDHG and LM-SPDHG,
- how fast LM-SPDHG converges compared with sinogram PDHG.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_02_run_pdhg_spdhg_dtv.py`
(PDHG and SPDHG with sinograms) and
:ref:`sphx_glr_auto_examples_04_listmode_algorithms_00_run_listmode_mlem_osem_svrg.py`
(listmode data and projectors).

.. note::

    Because of its run time (2000 PDHG iterations for the reference
    reconstruction), this example is **not executed** in the documentation
    build: the gallery shows the code only, and the thumbnail is a
    pre-rendered image.  Run it locally, preferably on a GPU (e.g. with CuPy
    or PyTorch as array backend); on the CPU it takes a long time.
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

import math

import matplotlib.pyplot as plt
import numpy as np
from array_api_compat import size

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    show_vol_cuts,
    suggest_array_backend_and_device,
)
from parallelproj.data import count_event_multiplicity

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. The reconstruction problem
# -----------------------------
#
# We minimize the negative Poisson log-likelihood combined with a total
# variation regularizer
#
# .. math::
#     f(x) = \sum_{i=1}^m \bar{d}_i (x) - d_i \log(\bar{d}_i (x))
#     + \beta \|\nabla x \|_{1,2}
#
# subject to
#
# .. math::
#     x \geq 0
#
# using the linear forward model
#
# .. math::
#     \bar{d}(x) = A x + s ,
#
# where :math:`d` are the measured data, :math:`A` is the forward operator
# (resolution model, TOF projector and attenuation), :math:`s` is the
# contamination, :math:`\nabla` the finite-difference gradient operator and
# :math:`\|\cdot\|_{1,2}` the isotropic mixed L2-L1 norm (TV semi-norm).
#
# We first compute a reference solution with **sinogram PDHG** (many
# iterations), then solve the same problem with **listmode SPDHG**
# :footcite:p:`Schramm2022` on the event list of the same data.

# %%
# 2. Input parameters
# -------------------

# image scale (can be used to simulate more or less counts)
img_scale = 0.1
# number of MLEM iterations used to initialize PDHG and LM-SPDHG
num_iter_mlem = 10
# number of PDHG iterations (reference reconstruction)
num_iter_pdhg = 2000
# number of subsets for LM-SPDHG
num_subsets = 10
# number of LM-SPDHG iterations (each = 2 * num_subsets mini-iterations)
num_iter_spdhg = 200
# prior weight
beta = 10.0
# step size ratio (used by both PDHG and LM-SPDHG)
gamma = 1.0 / img_scale
# rho parameter controlling the step size margin (PDHG and LM-SPDHG)
rho = 0.9999
# contamination in every sinogram bin relative to mean of trues sinogram
contam = 1.0

# selection probabilities for LM-SPDHG
p_g = 0.5  # gradient (regularizer) update
p_a = (1 - p_g) / num_subsets  # data subset update

# evaluate the cost function in every iteration
track_cost = True

# %%
# 3. Scanner, sinogram projector and phantom
# ------------------------------------------
#
# We use simulated listmode data, which we obtain from a simulated sinogram.
# We therefore first set up a sinogram projector: a scanner with 28 sides and
# 5 rings, all ring differences (span 1), and an image of 40 x 40 x 5 voxels
# of 4 mm.  The ground truth :math:`x_{true}` is an elliptic cylinder with hot
# and cold spherical inserts.

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

img_shape = (40, 40, 5)
voxel_size = (4.0, 4.0, 4.0)

# the LOR descriptor defines the sinogram
lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(
        scanner.num_rings, max_ring_difference=num_rings - 1, span=1
    ),
    radial_trim=170,
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

fig_true, _, _ = show_vol_cuts(
    x_true, voxel_size=voxel_size, vmin=0, fig_title="True activity image"
)
fig_true.show()

# %%
# The transaxial slice shows the uniform cylinder with its hot and cold
# inserts; along ``x2`` the image has only 5 slices.

# %%
# 4. Complete TOF forward model
# -----------------------------
#
# The attenuation sinogram (water, 0.01 / mm, inside the cylinder) is computed
# with the non-TOF projector.  We then enable TOF and combine an image-based
# resolution model (4.5 mm FWHM), the TOF projector and the attenuation into a
# single linear operator :math:`A`.

# setup an attenuation image
x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
# calculate the (non-TOF) attenuation sinogram
att_sino = xp.exp(-proj(x_att))

# enable TOF - remove these lines to run non-TOF reconstructions
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
    proj.in_shape, sigma=[4.5 / (2.3548 * float(vs)) for vs in proj.voxel_size]
)

# compose all 3 operators into a single linear operator
pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# 5. Simulation of noisy data
# ---------------------------
#
# We forward project the ground truth, add a constant contamination
# :math:`s` (equal to the mean of the noise-free true sinogram) and draw
# Poisson noise to obtain the integer data sinogram :math:`d`.

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
    dtype=xp.int16,
)

# %%
# 6. MLEM initialization
# ----------------------
#
# A few (sinogram) MLEM iterations provide a warm start for both PDHG and
# LM-SPDHG.

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
# 7. Gradient operator and cost function
# --------------------------------------
#
# The finite-difference gradient operator :math:`\nabla` and the cost
# function :math:`f(x)` from section 1 (up to a constant), evaluated on the
# sinogram data for both algorithms.

op_G = parallelproj.operators.FiniteForwardDifference(pet_lin_op.in_shape)


def cost_function(img):
    exp = pet_lin_op(img) + contamination
    res = float(xp.sum(exp - d * xp.log(exp)))
    res += beta * float(xp.sum(xp.linalg.vector_norm(op_G(img), axis=0)))
    return res


# %%
# 8. PDHG reference reconstruction
# --------------------------------
#
# .. admonition:: PDHG for the Poisson log-likelihood with a TV prior
#
#   | **Input** Poisson data :math:`d`
#   | **Initialize** :math:`x,y,w,S_A,S_G,T`
#   | **Preprocessing** :math:`\overline{z} = z = A^T y + \nabla^T w`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update**
#     :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Update** :math:`y^+ \gets \text{prox}_{D^*}^{S_A} ( y + S_A  ( A x + s))`
#   |     **Update**
#     :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w+S_G \nabla x)/\beta)`
#   |     **Update** :math:`\Delta z \gets A^T (y^+ - y) + \nabla^T (w^+ - w)`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + \Delta z`
#   |     **Update** :math:`y \gets y^+`
#   |     **Update** :math:`w \gets w^+`
#   | **Return** :math:`x`
#
# See :footcite:p:`Ehrhardt2019` and :footcite:p:`Schramm2022` for more
# details.  :math:`D^*` and :math:`R^*` are the convex conjugates of the
# Poisson data term and of the mixed L2-L1 norm.  The proximal operator of
# :math:`R^*` is a pointwise projection of the gradient field onto the unit
# ball; the one of :math:`D^*` has a closed form:
#
# .. admonition:: Proximal operator of the convex dual of the Poisson term
#
#  :math:`(\text{prox}_{D^*}^{S}(y))_i = \text{prox}_{D^*}^{S}(y_i) =
#  \frac{1}{2} \left(y_i + 1 - \sqrt{ (y_i-1)^2 + 4 S d_i} \right)`
#
# .. admonition:: Step sizes
#
#  :math:`S_A = \gamma \, \text{diag}(\frac{\rho}{A 1})`
#
#  :math:`S_G = \gamma \, \frac{\rho}{\|\nabla\|}`
#
#  :math:`T_A = \gamma^{-1} \text{diag}(\frac{\rho}{A^T 1})`
#
#  :math:`T_G = \gamma^{-1} \frac{\rho}{\|\nabla\|}`
#
#  :math:`T = \min(T_A, T_G)` pointwise
#
# The primal variable is warm-started from the MLEM result, the dual variable
# of the data term from the current ratio of data and expected data,
# :math:`y = 1 - d / (A x + s)`, and the dual variable of the gradient with
# zeros.

x_pdhg = 1.0 * x_mlem
y = 1 - d / (pet_lin_op(x_pdhg) + contamination)

w = xp.zeros(op_G.out_shape, dtype=xp.float32, device=dev)

z = pet_lin_op.adjoint(y) + op_G.adjoint(w)
zbar = 1.0 * z

# %%
# The step sizes follow the rules above.  Sinogram bins that no LOR hits
# (:math:`A 1 = 0`) get the smallest non-zero value, and :math:`\|\nabla\|`
# is estimated with 100 power iterations.

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
# Every PDHG iteration uses one full forward and back projection of the
# sinogram data.  The cost is evaluated after each primal update.  We also
# keep the PDHG image after ``num_iter_spdhg`` iterations, to compare it with
# LM-SPDHG after the same number of iterations.

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
# 9. Conversion of the emission sinogram to listmode
# --------------------------------------------------
#
# :meth:`.RegularPolygonPETProjector.convert_sinogram_to_listmode` converts an
# integer non-TOF or TOF sinogram to an event list: the world coordinates of
# the two LOR endpoints of every event and its TOF bin.
#
# .. warning::
#     The created event list is "ordered" (sorted by sinogram bin).  Since the
#     listmode subsets below take every ``num_subsets``-th event, we shuffle
#     the events first.

print(f"\nGenerating LM events ({float(xp.sum(d)):.2e})")
event_start_coords, event_end_coords, event_tofbins = proj.convert_sinogram_to_listmode(
    d
)

random_inds = np.random.permutation(event_start_coords.shape[0])
event_start_coords = event_start_coords[random_inds, :]
event_end_coords = event_end_coords[random_inds, :]
event_tofbins = event_tofbins[random_inds]

# %%
# 10. Listmode subset forward models
# ----------------------------------
#
# Subset :math:`i` contains every ``num_subsets``-th event, starting at event
# :math:`i`.  For each subset we create a :class:`.ListmodePETProjector`, compute
# the attenuation factor of every event (with a non-TOF projection of the
# attenuation image), enable TOF, and combine attenuation, projector and
# resolution model into the listmode forward operator :math:`A^{LM}_{N_i}`.

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

# %%
# The contamination is constant, so the contamination of every event
# (:math:`s_N`) is the same value.

contamination_list = xp.full(
    event_start_coords.shape[0],
    float(xp.reshape(contamination, (size(contamination),))[0]),
    device=dev,
    dtype=xp.float32,
)

# %%
# 11. Event multiplicity
# ----------------------
#
# A sinogram bin with :math:`d_i > 1` counts appears as :math:`d_i` identical
# events in the list.  LM-SPDHG stores one dual variable per event, and needs
# the multiplicity :math:`\mu_e` of every event :math:`e` (the number of
# events in the list with the same LOR endpoints and TOF bin, i.e. :math:`d_i`
# of its bin) to give such bins the correct weight.

events = xp.concat(
    [event_start_coords, event_end_coords, xp.expand_dims(event_tofbins, -1)], axis=1
)
mu = count_event_multiplicity(events)

# %%
# 12. Listmode SPDHG
# ------------------
#
# .. admonition:: Listmode SPDHG for the Poisson log-likelihood with a TV prior
#
#   | **Input** event list :math:`N`, contamination list :math:`s_N`
#   | **Calculate** event counts :math:`\mu_e` for each :math:`e` in :math:`N`
#   | **Initialize** :math:`x,(S_i)_i,S_G,T,(p_i)_i`
#   | **Initialize list** :math:`y_{N} = 1 - (\mu_N /(A^{LM}_{N} x + s_{N}))`
#   | **Preprocessing**
#     :math:`\overline{z} = z = {A^T} 1 - {A^{LM}_N}^T (y_N-1)/\mu_N`
#   | **Split lists** :math:`N`, :math:`s_N` and :math:`y_N` into :math:`n`
#     sublists :math:`N_i`, :math:`y_{N_i}` and :math:`s_{N_i}`
#   | **Repeat**, until stopping criterion fulfilled
#   |     **Update**
#     :math:`x \gets \text{proj}_{\geq 0} \left( x - T \overline{z} \right)`
#   |     **Select** :math:`i \in \{ 1,\ldots,n+1\}` randomly according to
#     :math:`(p_i)_i`
#   |     **if** :math:`i \leq n`:
#   |         **Update** :math:`y_{N_i}^+ \gets \text{prox}_{D^*}^{S_i}
#     \left( y_{N_i} + S_i \left(A^{LM}_{N_i} x + s^{LM}_{N_i} \right) \right)`
#   |         **Update** :math:`\Delta z \gets {A^{LM}_{N_i}}^T
#     \left(\frac{y_{N_i}^+ - y_{N_i}}{\mu_{N_i}}\right)`
#   |         **Update** :math:`y_{N_i} \gets y_{N_i}^+`
#   |     **else:**
#   |         **Update**
#     :math:`w^+ \gets \beta \, \text{prox}_{R^*}^{S_G/\beta} ((w+S_G \nabla x)/\beta)`
#   |         **Update** :math:`\Delta z \gets \nabla^T (w^+ - w)`
#   |         **Update** :math:`w \gets w^+`
#   |     **Update** :math:`z \gets z + \Delta z`
#   |     **Update** :math:`\bar{z} \gets z + (\Delta z/p_i)`
#   | **Return** :math:`x`
#
# Here the data :math:`d_i` in :math:`\text{prox}_{D^*}` are replaced by the
# multiplicities :math:`\mu_{N_i}` of the events.  Sinogram bins without
# counts do not appear in the event list: their dual variables stay at the
# value 1, which is accounted for by the term :math:`A^T 1` in :math:`z`.
#
# .. admonition:: Step sizes
#
#  :math:`S_i = \gamma \, \text{diag}(\frac{\rho}{A^{LM}_{N_i} 1})`
#
#  :math:`T_i = \gamma^{-1} \text{diag}(\frac{\rho p_i}{{A^{LM}_{N_i}}^T 1/\mu_{N_i}})`
#  for the data subsets :math:`i \leq n`
#
#  :math:`T_{n+1} = T_G = \gamma^{-1} \frac{\rho}{\|\nabla\|}` for the gradient
#  (the same value as for PDHG)
#
#  :math:`T = \min_{i=1,\ldots,n+1} T_i` pointwise
#
# The primal variable starts from the MLEM result, the dual variables of the
# events from the ratio of the multiplicities and the expected data, and the
# dual variable of the gradient with zeros.

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
z += op_G.adjoint(w_lm)
zbar = 1.0 * z

# %%
# The step sizes follow the rules above (the gradient step size ``S_G`` and
# ``T_G`` are the ones computed for PDHG).

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
# One LM-SPDHG iteration consists of ``2 * num_subsets`` mini-iterations, each
# with an image update.  Instead of drawing every block independently, we
# visit the blocks in a random permutation of ``num_subsets`` data subsets and
# ``num_subsets`` gradient updates, which realizes the selection probabilities
# ``p_a`` (per data subset) and ``p_g`` exactly within each iteration.  One
# iteration therefore uses every event once, i.e. one pass over the data, like
# one PDHG iteration.
#
# At the start of every iteration we record the cost and the PSNR of the
# current image with respect to the PDHG reference (with the maximum of the
# true image as peak value).

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

print("")

# %%
# 13. Reconstructed images
# ------------------------
#
# The central transaxial slice of the ground truth, the MLEM initialization,
# the PDHG reference, and LM-SPDHG and PDHG after the same number of
# iterations, all with the same grey scale.

images = {
    "True image": x_true,
    f"MLEM init.\n{num_iter_mlem} it.": x_mlem,
    f"PDHG reference\n{num_iter_pdhg} it.": x_pdhg,
    f"LM-SPDHG, {num_iter_spdhg} it.\n{num_subsets} subsets": x_lmspdhg,
    f"PDHG\n{num_iter_spdhg} it.": x_pdhg_early,
}
vmax = 1.2 * float(xp.max(x_true))
sl = img_shape[2] // 2

fig, axes = plt.subplots(
    1, 5, figsize=(10, 2.8), sharex=True, sharey=True, layout="constrained"
)
for ax, (title, img) in zip(axes, images.items()):
    im = ax.imshow(to_numpy_array(img[:, :, sl]).T, cmap="Greys", vmin=0, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel("x0 voxel")
axes[0].set_ylabel("x1 voxel")
fig.colorbar(im, ax=axes, shrink=0.8, label="activity (a.u.)")
fig.suptitle(f"Transaxial slice x2 = {sl}")
fig.show()

# %%
# The MLEM initialization is noisy.  The TV prior removes the noise and gives
# piecewise constant images with sharp edges.  After the same number of
# iterations (i.e. passes over the data), the LM-SPDHG image is visually
# indistinguishable from the PDHG reference, while PDHG itself is not yet
# converged.

# %%
# 14. Convergence
# ---------------
#
# The cost function per iteration (one pass over the data for both
# algorithms), a zoom to the first ``num_iter_spdhg`` iterations, and the PSNR
# of LM-SPDHG with respect to the PDHG reference.  The LM-SPDHG values were
# recorded at the start of each iteration, i.e. after 0, 1, 2, ... completed
# iterations.

if track_cost:
    it_pdhg = np.arange(1, num_iter_pdhg + 1)
    it_lmspdhg = np.arange(num_iter_spdhg)

    fig2, ax2 = plt.subplots(1, 3, figsize=(10, 3.5), layout="constrained")
    for i in range(2):
        ax2[i].plot(it_pdhg, cost_pdhg, ".-", label="PDHG")
        ax2[i].plot(it_lmspdhg, cost_lmspdhg, ".-", label="LM-SPDHG")
        ax2[i].grid(ls=":")
        ax2[i].legend()
        ax2[i].set_ylim(None, cost_pdhg[10:].max())
        ax2[i].set_ylabel("cost")
    ax2[1].set_xlim(0, num_iter_spdhg)
    ax2[2].plot(it_lmspdhg, psnr_lmspdhg, ".-")
    ax2[2].grid(ls=":")
    ax2[2].set_ylabel("PSNR (dB)")
    for axx in ax2.ravel():
        axx.set_xlabel("iteration")
    ax2[0].set_title("Cost")
    ax2[1].set_title("Cost (zoom)")
    ax2[2].set_title("PSNR of LM-SPDHG vs. reference")
    fig2.show()

# %%
# LM-SPDHG decreases the cost much faster per pass over the data than PDHG,
# and its PSNR with respect to the PDHG reference increases quickly in the
# first iterations.

# %%
# Summary
# -------
#
# - LM-SPDHG solves the Poisson log-likelihood + TV problem directly on an
#   event list: it stores one dual variable per event and uses the event
#   multiplicities :math:`\mu` in place of the sinogram counts.
# - The listmode subset forward models combine a :class:`.ListmodePETProjector`
#   (with TOF bins of the events), the attenuation factors of the events and the
#   image-based resolution model.
# - The step sizes are set from :math:`A^{LM}_{N_i} 1`,
#   :math:`{A^{LM}_{N_i}}^T 1/\mu_{N_i}`, :math:`\|\nabla\|`, the selection
#   probabilities and the ratio :math:`\gamma`.
# - Per pass over the data, LM-SPDHG converges much faster than PDHG.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_05_transmission_00_mltr_sps.py`
#    -- reconstruction of attenuation images from transmission data.
#
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_path = "_static/thumbnails/02_lm_spdhg.png"
