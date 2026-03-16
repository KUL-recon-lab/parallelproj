"""
Basic OSEM
==========

This example demonstrates the use of the MLEM algorithm with ordered subsets (OS-MLEM or OSEM)
to minimize the negative Poisson log-likelihood function.

.. math::
    f(x) = \\sum_{i=1}^m \\bar{y}_i (x) - y_i \\log(\\bar{y}_i (x))

subject to

.. math::
    x \\geq 0

using the linear forward model

.. math::
    \\bar{y}(x) = A x + s

The idea is to split the complete forward operator into a sequence :math:`n` disjoint
subset operators

.. math::
    A = \\{ A^1, \\ldots, A^n \\}

which is can be used to evaluate a subset of the forward model

.. math::
    \\bar{y}^k(x) = A^k x + s^k

.. tip::
    parallelproj is python array API compatible meaning it supports different
    array backends (e.g. numpy, cupy, torch, ...) and devices (CPU or GPU).
    Choose your preferred array API ``xp`` and device ``dev`` below.

.. image:: https://mybinder.org/badge_logo.svg
 :target: https://mybinder.org/v2/gh/gschramm/parallelproj/master?labpath=examples
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import parallelproj.operators as ppo
from parallelproj import to_numpy_array

# %%
from importlib import import_module, util


# choose array backend and a device (CPU or CUDA GPU)
if util.find_spec("torch") is not None:
    xp = import_module("array_api_compat.torch")
    dev = "cuda" if xp.cuda.is_available() else "cpu"
elif util.find_spec("cupy") is not None:
    xp = import_module("array_api_compat.cupy")
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
else:
    xp = import_module("array_api_compat.numpy")
    # using numpy, device must be cpu
    dev = "cpu"

print(f"Using array API: {xp.__name__}, device: {dev}")


# %%
# Setup of the complete forward model :math:`\bar{y}(x) = A x + s`
# ----------------------------------------------------------------
#
# We setup a minimal linear forward operator :math:`A` respresented by a 4x4 matrix
# and an arbritrary contamination vector :math:`s` of length 4.

# setup an arbitrary 4x4 matrix
mat = xp.asarray(
    [
        [2.5, 1.2, 0.3, 0.1],
        [0.4, 3.1, 0.7, 0.2],
        [0.1, 0.3, 4.1, 2.5],
        [0.2, 0.5, 0.2, 0.9],
    ],
    dtype=xp.float64,
    device=dev,
)

op_A = ppo.MatrixOperator(mat)
# setup an arbitrary contamination vector that has shape op_A.out_shape
contamination = xp.asarray([0.3, 0.2, 0.1, 0.4], dtype=xp.float64, device=dev)

print(op_A.A)

# %%
# Setup of ground truth and data simulation
# -----------------------------------------
#
# We setup an arbitrary ground truth :math:`x_{true}` and simulate
# noise-free and noisy data :math:`y` by adding Poisson noise.

# ground truth
x_true = xp.asarray([5.5, 10.7, 8.2, 7.9], dtype=xp.float64, device=dev)

# simulated noise-free data
noise_free_data = op_A(x_true) + contamination

# add Poisson noise
np.random.seed(1)
y = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float64,
)

# %%
# Analytic calculation of the optimal point (as reference)
# --------------------------------------------------------
#
# Since our linear forward operator :math:`A` is invertible
# (*which is usually not the case in practice*),
# we can calculate the optimal point :math:`x^* = A^{-1} (y - s)`
# and the corresponding optimal value of :math:`f(x^*)`.

# calculate the reference solution by inverting A
mat_inv = xp.linalg.inv(mat)
x_ref = mat_inv @ (y - contamination)

# also calculate the cost of the reference solution
exp_ref = op_A(x_ref) + contamination
cost_ref = float(xp.sum(exp_ref - y * xp.log(exp_ref)))

# %%
# Splitting of the forward model into subsets :math:`A^k`
# -------------------------------------------------------
#
# We split the matrix operator into two disjoint subsets
# using slicing along the first dimension of the matrix.
#
# .. note::
#     The OSEM implementation below works with all linear operators that
#     subclass :class:`.LinearOperatorSequence`

# define two subsets (they don't need to have equal size)
subset_slices = (slice(0, 2), slice(2, None))

# setup two subsets operators each containing 1 and 3 rows of the matrix A
op_seq = ppo.LinearOperatorSequence(
    [ppo.MatrixOperator(mat[sl, :]) for sl in subset_slices]
)

for op in op_seq:
    print(op.A)

# %%
# OSEM iterations to minimize :math:`f(x)`
# ----------------------------------------
#
# We apply multiple OSEM updates :cite:p:`Hudson1994`
#
# .. math::
#     x^+ = \frac{x}{(A^k)^H 1} (A^k)^H \frac{y^k}{A^k x + s^k}
#
# to calculate the minimizer of :math:`f(x)` iteratively.
#
# To monitor the convergence we calculate the relative cost
#
# .. math::
#    \frac{f(x) - f(x^*)}{|f(x^*)|}
#
# and the distance to the optimal point
#
# .. math::
#    \frac{\|x - x^*\|}{\|x^*\|}.

# number MLEM iterations
num_iter = 1000 // len(op_seq)

# initialize x
x = xp.ones(op_A.in_shape, dtype=xp.float64, device=dev)

# calculate A_k^H 1 for all subsets k
subset_adjoint_ones = [
    x.adjoint(xp.ones(x.out_shape, dtype=xp.float64, device=dev)) for x in op_seq
]

# allocate arrays for the relative cost and the relative distance to the
# optimal point
rel_cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)
rel_dist = xp.zeros(num_iter, dtype=xp.float64, device=dev)

for i in range(num_iter):
    for k, sl in enumerate(subset_slices):
        # evaluate the forward model
        subset_exp = op_seq[k](x) + contamination[sl]
        # calculate the relative cost and distance to the optimal point
        # to do this, we need the full expectation
        exp = op_A(x) + contamination
        rel_cost[i] = (xp.sum(exp - y * xp.log(exp)) - cost_ref) / abs(cost_ref)
        rel_dist[i] = xp.linalg.vector_norm(x - x_ref) / xp.linalg.vector_norm(x_ref)
        # OSEM update
        ratio = y[sl] / subset_exp
        x *= op_seq[k].adjoint(ratio) / subset_adjoint_ones[k]


# %%
# Convergences plots
# ------------------
#
# ..note::
#       The basic OSEM does not converge to the optimal point
#       (but can come close using a few number of fast updates).

fig, ax = plt.subplots(1, 2, figsize=(8, 4), sharex=True)
ax[0].semilogx(to_numpy_array(rel_cost))
ax[1].loglog(to_numpy_array(rel_dist))
ax[0].set_ylim(-float(rel_cost[2]), float(rel_cost[2]))
ax[0].set_ylabel(r"( f($x$) - f($x^*$) )   /   | f($x^*$) |")
ax[1].set_ylabel(r"rel. distance to optimum $\|x - x^*\| / \|x^*\|$")
ax[0].set_xlabel("iteration")
ax[1].set_xlabel("iteration")
ax[0].grid(ls=":")
ax[1].grid(ls=":")
fig.tight_layout()
fig.show()
