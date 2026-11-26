"""
DePierro's algorithm for the Poisson log-likelihood with a quadratic prior
==========================================================================

:bdg-primary:`reconstruction` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Adding a quadratic prior to the Poisson log-likelihood regularizes the
reconstruction, but the resulting MAP problem cannot be solved with MLEM.
DePierro's algorithm keeps the simplicity of MLEM: every iteration minimizes a
separable surrogate of the cost function, which has a closed-form solution
for every voxel.  In this example we apply it to a tiny problem with a 5 x 4
system matrix and check the result against a general-purpose optimizer.

**You will learn**

- how to write the MAP cost function with a quadratic intensity prior,
- how to implement DePierro's update for any :class:`.LinearOperator`,
- how to check the solution with a reference optimizer (Powell's method from
  SciPy),
- how to monitor the convergence with the cost function.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(MLEM and the Poisson log-likelihood).
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import fmin_powell

import parallelproj.operators
from parallelproj import to_numpy_array
from parallelproj._examples_utils import suggest_array_backend_and_device

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. The optimization problem
# ---------------------------
#
# We minimize the negative Poisson log-likelihood combined with a quadratic
# intensity prior that pulls the image towards a prior image :math:`z`:
#
# .. math::
#     f(x) = \sum_{i=1}^m \bar{y}_i(x) - y_i \log(\bar{y}_i (x))
#     + \frac{\beta}{2} \|x - z \|^2
#
# subject to
#
# .. math::
#     x \geq 0
#
# using the linear forward model
#
# .. math::
#     \bar{y}(x) = A x + s

# %%
# 2. Forward model :math:`\bar{y}(x) = A x + s`
# ---------------------------------------------
#
# We set up a minimal linear forward operator :math:`A` represented by a 5x4
# matrix and an arbitrary contamination vector :math:`s` of length 5.
#
# .. note::
#     The DePierro implementation below works with all linear operators that
#     subclass :class:`.LinearOperator` (e.g. the high-level projectors).

# setup an arbitrary 5x4 matrix
mat = xp.asarray(
    [
        [2.5, 1.2, 0.3, 0.1],
        [0.4, 3.1, 0.7, 0.2],
        [0.1, 0.3, 4.1, 2.5],
        [0.2, 0.5, 0.2, 0.9],
        [0.3, 0.1, 0.7, 0.2],
    ],
    dtype=xp.float64,
    device=dev,
)

op_A = parallelproj.operators.MatrixOperator(mat)
# setup an arbitrary contamination vector that has shape op_A.out_shape
contamination = xp.asarray([0.3, 0.2, 0.1, 0.4, 0.1], dtype=xp.float64, device=dev)

# %%
# 3. Ground truth and data simulation
# -----------------------------------
#
# We set up an arbitrary ground truth :math:`x_{true}` and simulate
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
# 4. Prior and cost function
# --------------------------
#
# The prior image :math:`z` (``x_prior``) is constant and equal to the
# smallest value of the ground truth; :math:`\beta` sets the strength of the
# prior.  The cost function :math:`f(x)` returns the largest float64 value
# for infeasible :math:`x` (negative values or negative expected data).

x_prior = xp.full(op_A.in_shape, xp.min(x_true), dtype=xp.float64, device=dev)
beta = 0.3
num_iter = 50


def cost_function(x):
    exp = op_A(x) + contamination
    if (xp.min(exp) < 0) or (xp.min(x) < 0):
        res = xp.finfo(xp.float64).max
    else:
        res = (xp.sum(exp - y * xp.log(exp))) + 0.5 * beta * xp.sum((x - x_prior) ** 2)
    return res


# %%
# 5. DePierro iterations
# ----------------------
#
# Starting from a uniform image, we apply the DePierro update
#
# .. math::
#     b = A^T 1 - \beta z
# .. math::
#     t = x A^T \frac{y}{A x + s}
# .. math::
#     x^+ = \frac{2 t}{\sqrt{b^2 + 4 \beta t} + b}
#
# (all operations element-wise) to calculate the minimizer of :math:`f(x)`
# iteratively.  :math:`x^+` is the positive root of
# :math:`\beta x^2 + b x - t = 0`, the minimizer of the separable surrogate.
# For :math:`\beta = 0` it reduces to the MLEM update :math:`x^+ = t / A^T 1`.
#
# See :footcite:p:`DePierro1995` for more details.

# initialize x
x = xp.ones(op_A.in_shape, dtype=xp.float64, device=dev)

cost = xp.zeros(num_iter, dtype=xp.float64, device=dev)

# "b" - modified sensitivity image
mod_adjoint_ones = (
    op_A.adjoint(xp.ones(op_A.out_shape, dtype=xp.float64, device=dev)) - beta * x_prior
)

for i in range(num_iter):
    # evaluate the forward model
    exp = op_A(x) + contamination
    ratio = y / exp
    t = x * op_A.adjoint(ratio)
    x = 2 * t / (xp.sqrt(mod_adjoint_ones**2 + 4 * beta * t) + mod_adjoint_ones)
    cost[i] = cost_function(x)

print(f"Solution after {num_iter} DePierro iterations:")
print(x)
print(f"cost: {cost[-1]:.6e}")

# %%
# 6. Reference solution with Powell's method
# ------------------------------------------
#
# To check the result, we minimize the same cost function with Powell's
# derivative-free method from SciPy, started at the DePierro solution.
# Since :math:`f` is convex, Powell's method finding no better point supports
# that the DePierro solution is the minimizer.  SciPy works with NumPy arrays,
# so the cost function is wrapped.


def cost_function_numpy(x_np):
    return float(cost_function(xp.asarray(x_np, dtype=xp.float64, device=dev)))


x_ref = xp.asarray(
    fmin_powell(cost_function_numpy, to_numpy_array(x), xtol=1e-6, ftol=1e-6),
    dtype=xp.float64,
    device=dev,
)
rel_dist = float(xp.sum((x - x_ref) ** 2)) / float(xp.sum(x_ref**2))
cost_ref = float(cost_function(x_ref))

print("\nReference solution using Powell optimizer:")
print(x_ref)
print(f"rel. distance to DePierro solution: {rel_dist:.2e}")
print(f"cost: {cost_ref:.6e}")

# %%
# 7. Convergence
# --------------
#
# The left panel shows the cost function after every DePierro iteration and
# the cost of the reference solution, the right panel the difference between
# the two on a logarithmic scale.

iterations = np.arange(1, num_iter + 1)
cost_np = to_numpy_array(cost)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
axes[0].plot(iterations, cost_np, label="DePierro")
axes[0].axhline(cost_ref, color="k", linestyle="--", label="Powell reference")
axes[0].set_ylabel("cost function f(x)")
axes[0].set_title("Cost function")
axes[0].legend()
axes[1].semilogy(iterations, np.abs(cost_np - cost_ref))
axes[1].set_ylabel(r"$|f(x) - f(x_\mathrm{ref})|$")
axes[1].set_title("Distance to the reference cost")
for ax in axes:
    ax.set_xlabel("iteration")
    ax.grid(ls=":")
fig.show()

# %%
# The cost decreases monotonically, as guaranteed by the surrogate
# construction.  The distance to the reference cost decreases linearly on the
# logarithmic scale (linear convergence) until it reaches the float64
# precision of the cost (about :math:`10^{-13}`) after roughly 30 iterations;
# gaps in the curve are iterations where the difference is exactly zero.

# %%
# Summary
# -------
#
# - DePierro's algorithm minimizes the Poisson negative log-likelihood plus a
#   quadratic prior :math:`\frac{\beta}{2}\|x - z\|^2` with an MLEM-like,
#   closed-form update that keeps the image non-negative.
# - It only needs forward and back projections, so it works with any
#   :class:`.LinearOperator`.
# - A general-purpose optimizer (Powell's method) confirms the solution on this
#   small problem.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_05_run_tof_variance.py`
#    -- how TOF information reduces the noise of reconstructed images.
#
# .. rubric:: References
#
# .. footbibliography::

# sphinx_gallery_thumbnail_number = 1
