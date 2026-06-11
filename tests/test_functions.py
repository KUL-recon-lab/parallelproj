from __future__ import annotations

import math
import warnings

import numpy as _np
import pytest
import array_api_compat
from typing import cast
from types import ModuleType

import parallelproj.functions as ppf
import parallelproj.operators as ppo

from .config import pytestmark


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def allclose(x, y, atol: float = 1e-5, rtol: float = 1e-5) -> bool:
    xp = array_api_compat.array_namespace(x, y)
    return bool(xp.all(xp.abs(x - y) <= atol + rtol * xp.abs(y)))


def finite_diff_gradient(f, x_np, xp, dev, eps: float = 1e-4):
    """Central-difference numerical gradient.

    Parameters
    ----------
    f : callable
        Scalar-valued function that accepts an array in namespace *xp*.
    x_np : numpy array
        1-D evaluation point (plain numpy, used to build perturbations).
    xp, dev :
        Target array namespace and device.
    eps : float
        Finite-difference step size.
    """
    n = x_np.size
    grad = _np.zeros(n)
    for i in range(n):
        dx = _np.zeros(n)
        dx[i] = eps
        fp = float(f(xp.asarray(x_np + dx, device=dev)))
        fm = float(f(xp.asarray(x_np - dx, device=dev)))
        grad[i] = (fp - fm) / (2.0 * eps)
    return xp.asarray(grad, device=dev)


# ---------------------------------------------------------------------------
# Abstract classes
# ---------------------------------------------------------------------------


def test_c1function_not_instantiable(xp: ModuleType, dev: str):
    with pytest.raises(TypeError):
        ppf.C1Function()


def test_c2function_not_instantiable(xp: ModuleType, dev: str):
    with pytest.raises(TypeError):
        ppf.C2Function()


def test_beta_property_get_set(xp: ModuleType, dev: str):
    """Covers the beta getter (line 35) and setter (line 39)."""
    d = xp.asarray(_Y_NP, device=dev)
    f = ppf.NegPoissonLogL(d)
    assert f.beta == 1.0  # getter
    f.beta = 3.0  # setter
    assert f.beta == 3.0


# ---------------------------------------------------------------------------
# NegPoissonLogL
# ---------------------------------------------------------------------------

_Y_NP = _np.asarray([3.0, 1.0, 2.0, 4.0])
_YBAR_NP = _np.asarray([2.5, 1.5, 2.0, 3.5])


def test_neg_poisson_logl_call(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    expected = float(_np.sum(_YBAR_NP - _Y_NP * _np.log(_YBAR_NP)))
    assert abs(f(ybar) - expected) < 1e-5

    # default (eps) mode evaluates the shifted-Poisson surrogate
    f_eps = ppf.NegPoissonLogL(y)
    e = f_eps.eps
    expected_eps = float(
        _np.sum((_YBAR_NP + e) - (_Y_NP + e) * _np.log(_YBAR_NP + e))
    )
    assert abs(f_eps(ybar) - expected_eps) < 1e-5


def test_neg_poisson_logl_gradient(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)

    f = ppf.NegPoissonLogL(y)
    grad = f.gradient(ybar)
    fd_grad = finite_diff_gradient(f, _YBAR_NP, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-4, rtol=1e-4)


def test_neg_poisson_logl_call_and_gradient(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)

    f = ppf.NegPoissonLogL(y)
    val, grad = f.call_and_gradient(ybar)
    assert abs(val - f(ybar)) < 1e-8
    assert allclose(grad, f.gradient(ybar))


def test_neg_poisson_logl_hessian_diag_vec_prod(xp: ModuleType, dev: str):
    v_np = _np.asarray([1.0, -1.0, 2.0, 0.5])
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)
    v = xp.asarray(v_np, device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    hv = f.hessian_diag_vec_prod(ybar, v)
    expected = xp.asarray(_Y_NP / _YBAR_NP**2 * v_np, device=dev)
    assert allclose(hv, expected)

    # default (eps) mode
    f_eps = ppf.NegPoissonLogL(y)
    e = f_eps.eps
    hv_eps = f_eps.hessian_diag_vec_prod(ybar, v)
    expected_eps = xp.asarray((_Y_NP + e) / (_YBAR_NP + e) ** 2 * v_np, device=dev)
    assert allclose(hv_eps, expected_eps)


def test_neg_poisson_logl_beta_scaling(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)

    f1 = ppf.NegPoissonLogL(y)
    f2 = ppf.NegPoissonLogL(y)
    f2.beta = 2.0
    assert abs(f2(ybar) - 2.0 * f1(ybar)) < 1e-8
    assert allclose(f2.gradient(ybar), 2.0 * f1.gradient(ybar))


# ---------------------------------------------------------------------------
# NegPoissonLogL exact mode (exact=True)
# ---------------------------------------------------------------------------

# bins 2 & 3: y = 0 AND ybar = 0 (virtual bins or noisy zeros)
_Y_SAFE_NP = _np.asarray([3.0, 1.0, 0.0, 0.0])
_YBAR_SAFE_NP = _np.asarray([2.5, 1.5, 0.0, 0.0])


def test_neg_poisson_logl_exact_call_zero_bins(xp: ModuleType, dev: str):
    """Bins with y = 0 and ybar = 0 must contribute 0 to the function value
    (f_i(ybar) = ybar is exact when y = 0)."""
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    assert f.exact and f.eps == 0.0
    val = f(ybar)

    expected = float(
        _np.sum(_YBAR_SAFE_NP[:2] - _Y_SAFE_NP[:2] * _np.log(_YBAR_SAFE_NP[:2]))
    )
    assert abs(val - expected) < 1e-5


def test_neg_poisson_logl_exact_gradient_zero_bins(xp: ModuleType, dev: str):
    """Bins with y = 0 must have gradient value = 1 (exact), for ybar = 0
    and ybar > 0 alike."""
    y = xp.asarray(_np.asarray([3.0, 1.0, 0.0, 0.0]), device=dev)
    ybar_np = _np.asarray([2.5, 1.5, 0.0, 0.5])  # y=0 bins: ybar 0 and > 0
    ybar = xp.asarray(ybar_np, device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    grad = f.gradient(ybar)

    expected_np = _np.asarray([1.0 - 3.0 / 2.5, 1.0 - 1.0 / 1.5, 1.0, 1.0])
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(grad, expected)


def test_neg_poisson_logl_exact_hessian_diag_zero_bins(xp: ModuleType, dev: str):
    """Bins with y = 0 and ybar = 0 must be zero in the diag Hessian-vector product."""
    v_np = _np.asarray([1.0, 1.0, 1.0, 1.0])
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)
    v = xp.asarray(v_np, device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    hv = f.hessian_diag_vec_prod(ybar, v)

    expected_np = _np.concatenate(
        [
            _Y_SAFE_NP[:2] / _YBAR_SAFE_NP[:2] ** 2 * v_np[:2],
            _np.asarray([0.0, 0.0]),
        ]
    )
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(hv, expected)


def test_neg_poisson_logl_exact_call_and_gradient(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    val, grad = f.call_and_gradient(ybar)
    assert abs(val - f(ybar)) < 1e-8
    assert allclose(grad, f.gradient(ybar))


def test_neg_poisson_logl_exact_model_violation_yields_inf(xp: ModuleType, dev: str):
    """y > 0 with ybar = 0 is a genuine model violation: f = +inf, grad = -inf
    in exact mode (must not be masked)."""
    y = xp.asarray(_np.asarray([3.0, 1.0]), device=dev)
    ybar = xp.asarray(_np.asarray([2.5, 0.0]), device=dev)

    f = ppf.NegPoissonLogL(y, exact=True)
    with warnings.catch_warnings():
        # numpy warns on log(0) / division by zero; inf is the intended result
        warnings.simplefilter("ignore", RuntimeWarning)
        val = f(ybar)
        grad = f.gradient(ybar)
    assert math.isinf(val) and val > 0
    assert math.isinf(float(grad[1])) and float(grad[1]) < 0


# ---------------------------------------------------------------------------
# NegPoissonLogL default ("safe epsilon" / shifted Poisson) mode
# ---------------------------------------------------------------------------


def test_neg_poisson_logl_eps_finite_on_zero_bins(xp: ModuleType, dev: str):
    """The default mode is finite for all ybar >= 0: gradient 0 at bins with
    y = ybar = 0, gradient -y/eps at model-violation bins (y > 0, ybar = 0)."""
    y_np = _np.asarray([3.0, 1.0, 0.0, 0.0])
    ybar_np = _np.asarray([2.5, 0.0, 0.0, 0.5])  # bin 1: y > 0, ybar = 0!
    y = xp.asarray(y_np, device=dev)
    ybar = xp.asarray(ybar_np, device=dev)

    f = ppf.NegPoissonLogL(y)
    e = f.eps
    assert not f.exact and e > 0

    val = f(ybar)
    grad = f.gradient(ybar)
    hv = f.hessian_diag_vec_prod(ybar, xp.ones_like(ybar))
    assert math.isfinite(val)
    assert bool(xp.all(xp.isfinite(grad))) and bool(xp.all(xp.isfinite(hv)))

    expected_np = (ybar_np - y_np) / (ybar_np + e)
    assert allclose(grad, xp.asarray(expected_np, device=dev))
    assert abs(float(grad[2]) - 0.0) < 1e-12  # y = ybar = 0 -> gradient 0
    assert abs(float(grad[1]) + 1.0 / e) < 1e-6 / e  # violation -> -y/eps


def test_neg_poisson_logl_eps_minimizer_at_data(xp: ModuleType, dev: str):
    """The shifted-Poisson surrogate keeps the per-bin minimiser at ybar = y:
    the gradient at ybar = y is exactly 0, including bins with y = 0."""
    y = xp.asarray(_Y_SAFE_NP, device=dev)

    f = ppf.NegPoissonLogL(y)
    grad = f.gradient(xp.asarray(_Y_SAFE_NP, device=dev))
    assert allclose(grad, xp.zeros_like(grad), atol=1e-12)


def test_neg_poisson_logl_eps_shifted_poisson_identity(xp: ModuleType, dev: str):
    """The default mode equals NegPoissonLogL(y + eps, exact=True) evaluated
    at ybar + eps, in value, gradient, and Hessian."""
    e = 0.125  # exactly representable
    y_np = _np.asarray([3.0, 1.0, 0.0, 2.0])
    ybar_np = _np.asarray([2.5, 1.5, 0.5, 3.0])
    v_np = _np.asarray([1.0, -1.0, 2.0, 0.5])

    y = xp.asarray(y_np, device=dev)
    ybar = xp.asarray(ybar_np, device=dev)
    v = xp.asarray(v_np, device=dev)

    f_eps = ppf.NegPoissonLogL(y, eps=e)
    assert f_eps.eps == e
    f_ref = ppf.NegPoissonLogL(xp.asarray(y_np + e, device=dev), exact=True)
    ybar_shift = xp.asarray(ybar_np + e, device=dev)

    assert abs(f_eps(ybar) - f_ref(ybar_shift)) < 1e-8
    assert allclose(f_eps.gradient(ybar), f_ref.gradient(ybar_shift))
    assert allclose(
        f_eps.hessian_diag_vec_prod(ybar, v),
        f_ref.hessian_diag_vec_prod(ybar_shift, v),
    )


def test_neg_poisson_logl_eps_derivation_and_override(xp: ModuleType, dev: str):
    """eps defaults to rel_eps * mean(y); an absolute eps overrides it."""
    y = xp.asarray(_Y_NP, device=dev)
    mean_y = float(_np.mean(_Y_NP))

    f = ppf.NegPoissonLogL(y)
    assert abs(f.eps - 1e-6 * mean_y) < 1e-15

    f2 = ppf.NegPoissonLogL(y, rel_eps=1e-3)
    assert abs(f2.eps - 1e-3 * mean_y) < 1e-12

    f3 = ppf.NegPoissonLogL(y, rel_eps=1e-3, eps=0.5)
    assert f3.eps == 0.5


def test_neg_poisson_logl_eps_validation(xp: ModuleType, dev: str):
    """Invalid eps / rel_eps combinations raise ValueError."""
    y = xp.asarray(_Y_NP, device=dev)
    y_zero = xp.asarray(_np.zeros(4), device=dev)

    with pytest.raises(ValueError, match="rel_eps"):
        ppf.NegPoissonLogL(y, rel_eps=0.0)
    with pytest.raises(ValueError, match="rel_eps"):
        ppf.NegPoissonLogL(y, rel_eps=-1e-6)
    with pytest.raises(ValueError, match="eps"):
        ppf.NegPoissonLogL(y, eps=0.0)
    with pytest.raises(ValueError, match="eps"):
        ppf.NegPoissonLogL(y, eps=-0.1)
    # all-zero data -> derived eps would be 0
    with pytest.raises(ValueError, match="all-zero"):
        ppf.NegPoissonLogL(y_zero)
    # ... but exact mode and explicit eps work
    ppf.NegPoissonLogL(y_zero, exact=True)
    ppf.NegPoissonLogL(y_zero, eps=1e-6)


# ---------------------------------------------------------------------------
# NegPoissonLogL extra checks
# ---------------------------------------------------------------------------


def test_neg_poisson_logl_extra_checks_warn_negative(xp: ModuleType, dev: str):
    """enable_extra_checks warns on negative values in x (default mode)."""
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_np.asarray([2.5, -1.5, 4.0, 0.5]), device=dev)

    f = ppf.NegPoissonLogL(y, enable_extra_checks=True)
    with pytest.warns(RuntimeWarning, match="negative"):
        f.gradient(ybar)


def test_neg_poisson_logl_extra_checks_exact_mode(xp: ModuleType, dev: str):
    """In exact mode, zeros at bins with y == 0 are fine (no warning); zeros
    at bins with y > 0 warn about the model violation."""
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar_ok = xp.asarray(_YBAR_SAFE_NP, device=dev)  # zeros only where y == 0

    f = ppf.NegPoissonLogL(y, exact=True, enable_extra_checks=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        f.gradient(ybar_ok)  # must not warn

    ybar_bad = xp.asarray(_np.asarray([2.5, 0.0, 0.0, 0.0]), device=dev)  # y[1] > 0
    with pytest.warns(RuntimeWarning, match="positive measured data"), _np.errstate(
        divide="ignore", invalid="ignore"
    ):
        f.gradient(ybar_bad)


def test_neg_poisson_logl_extra_checks_off_by_default(xp: ModuleType, dev: str):
    """Without enable_extra_checks, no parallelproj warning is emitted on positive x."""
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)

    f = ppf.NegPoissonLogL(y)
    assert not f.enable_extra_checks
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        f.gradient(ybar)  # must not warn
    # property setter
    f.enable_extra_checks = True
    assert f.enable_extra_checks


# ---------------------------------------------------------------------------
# NegPoissonLogL + C2AffineObjective: analytic ML solution
# ---------------------------------------------------------------------------

# A is invertible, so the ML solution saturates the likelihood (ybar = y):
#   x_ml = inv(A) y, with inv(A) = [[0, 1], [1, -1]]  ->  x_ml = [2, 0]
#   ybar(x_ml) = y = [2, 2] > 0 everywhere -> exact mode is valid
#   f(x_ml)    = sum(ybar - y log(ybar)) = 4 - 4 log(2)
#   grad(x_ml) = A^T (1 - y / ybar) = A^T 0 = 0
_A_ML_NP = _np.asarray([[1.0, 1.0], [1.0, 0.0]])
_Y_ML_NP = _np.asarray([2.0, 2.0])
_X_ML_NP = _np.asarray([2.0, 0.0])
_F_ML = 4.0 - 4.0 * math.log(2.0)

# same model with an extra all-zero row (third, virtual detector) measuring 0:
#   ybar(x_ml) = [2, 2, 0] -> bin 3 has y = 0 and ybar = 0
#   the virtual bin contributes 0 to f and (via the zero column of A^T)
#   nothing to the gradient, so f(x_ml) and grad(x_ml) are unchanged
_A_ML3_NP = _np.asarray([[1.0, 1.0], [1.0, 0.0], [0.0, 0.0]])
_Y_ML3_NP = _np.asarray([2.0, 2.0, 0.0])


def test_neg_poisson_logl_affine_ml_solution_exact(xp: ModuleType, dev: str):
    """Function value and gradient at the analytic ML solution (no virtual
    bins, exact mode): f(x_ml) = 4 - 4 log(2), grad(x_ml) = 0."""
    A = xp.asarray(_A_ML_NP, device=dev)
    y = xp.asarray(_Y_ML_NP, device=dev)
    x_ml = xp.asarray(_X_ML_NP, device=dev)

    obj = ppf.C2AffineObjective(
        ppf.NegPoissonLogL(y, exact=True), ppo.MatrixOperator(A)
    )

    assert abs(obj(x_ml) - _F_ML) < 1e-6
    assert allclose(obj.gradient(x_ml), xp.zeros_like(x_ml), atol=1e-6)


def test_neg_poisson_logl_affine_ml_solution_exact_virtual_bin(
    xp: ModuleType, dev: str
):
    """With an extra virtual bin (zero row in A, y = 0), exact mode gives a
    function value and gradient at the ML solution identical to the
    virtual-bin-free model."""
    A3 = xp.asarray(_A_ML3_NP, device=dev)
    y3 = xp.asarray(_Y_ML3_NP, device=dev)
    x_ml = xp.asarray(_X_ML_NP, device=dev)

    obj3 = ppf.C2AffineObjective(
        ppf.NegPoissonLogL(y3, exact=True), ppo.MatrixOperator(A3)
    )

    assert abs(obj3(x_ml) - _F_ML) < 1e-6
    assert allclose(obj3.gradient(x_ml), xp.zeros_like(x_ml), atol=1e-6)

    # explicit cross-check against the 2x2 model without virtual bins
    A = xp.asarray(_A_ML_NP, device=dev)
    y = xp.asarray(_Y_ML_NP, device=dev)
    obj = ppf.C2AffineObjective(
        ppf.NegPoissonLogL(y, exact=True), ppo.MatrixOperator(A)
    )
    assert abs(obj3(x_ml) - obj(x_ml)) < 1e-8
    assert allclose(obj3.gradient(x_ml), obj.gradient(x_ml), atol=1e-8)


def test_neg_poisson_logl_affine_ml_solution_default_virtual_bin(
    xp: ModuleType, dev: str
):
    """The default (eps) mode on the virtual-bin model: everything finite,
    gradient at the ML solution exactly 0 (ybar = y in every bin), and the
    function value within ~eps of the exact one."""
    A3 = xp.asarray(_A_ML3_NP, device=dev)
    y3 = xp.asarray(_Y_ML3_NP, device=dev)
    x_ml = xp.asarray(_X_ML_NP, device=dev)

    loss = ppf.NegPoissonLogL(y3)
    obj3 = ppf.C2AffineObjective(loss, ppo.MatrixOperator(A3))

    val = obj3(x_ml)
    grad = obj3.gradient(x_ml)
    assert math.isfinite(val)
    assert bool(xp.all(xp.isfinite(grad)))
    # ybar(x_ml) = y -> surrogate gradient is exactly 0 (minimiser preserved)
    assert allclose(grad, xp.zeros_like(grad), atol=1e-12)
    # value deviates from the exact one only by O(eps * |log eps|)
    assert abs(val - _F_ML) < 100 * loss.eps * abs(math.log(loss.eps))


# ---------------------------------------------------------------------------
# NegPoissonLogL: forward-model mismatch (point source, mismatched Gaussian)
# ---------------------------------------------------------------------------


def test_neg_poisson_logl_model_mismatch_point_source(xp: ModuleType, dev: str):
    """y ~ G x_true with a wide Gaussian G, reconstruction model G' narrower
    and truncated -> at a point-source-like iterate there are bins with
    y > 0 but ybar = 0.  In exact mode the EM-preconditioned gradient is
    poisoned by nan/inf; the default (eps) mode stays finite and the
    objective does not blow up over a few EM-preconditioned steps."""
    n = 41
    c = n // 2

    def conv_matrix(sig: float, trunc: float) -> _np.ndarray:
        i = _np.arange(n, dtype=_np.float64)
        dist = _np.abs(i[:, None] - i[None, :])
        kern = _np.exp(-0.5 * (dist / sig) ** 2)
        kern[dist > trunc * sig] = 0.0  # compact support -> exact zeros
        return (kern / kern.sum(axis=0, keepdims=True)).astype(_np.float32)

    G_true_np = conv_matrix(3.0, trunc=1e9)  # effectively untruncated
    G_rec_np = conv_matrix(1.5, trunc=3.0)  # narrower AND truncated

    x_true_np = _np.zeros(n, dtype=_np.float32)
    x_true_np[c] = 100.0
    y_np = (G_true_np @ x_true_np).astype(_np.float32)  # > 0 in every bin

    y = xp.asarray(y_np, device=dev, dtype=xp.float32)
    x0 = xp.asarray(x_true_np, device=dev, dtype=xp.float32)
    op = ppo.MatrixOperator(xp.asarray(G_rec_np, device=dev, dtype=xp.float32))
    sens = op.adjoint(xp.ones(op.out_shape, dtype=xp.float32, device=dev))

    # sanity: the mismatch scenario is actually present
    ybar0 = op(x0)
    assert bool(xp.any((ybar0 == 0) & (y > 0)))

    # exact mode: gradient at the point source is poisoned by nan/inf
    obj_exact = ppf.C2AffineObjective(ppf.NegPoissonLogL(y, exact=True), op)
    with _np.errstate(divide="ignore", invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        g_exact = obj_exact.gradient(x0)
    assert not bool(xp.all(xp.isfinite(g_exact)))

    # default (eps) mode: finite gradient, finite and non-exploding iterates
    obj_eps = ppf.C2AffineObjective(ppf.NegPoissonLogL(y), op)
    assert bool(xp.all(xp.isfinite(obj_eps.gradient(x0))))

    x = x0
    val0 = obj_eps(x)
    for _ in range(3):
        x = x - x / sens * obj_eps.gradient(x)  # EM-preconditioned step
        assert bool(xp.all(xp.isfinite(x)))
    val3 = obj_eps(x)
    assert math.isfinite(val3) and val3 <= val0 + 1e-3


# ---------------------------------------------------------------------------
# HalfSquaredL2Deviation
# ---------------------------------------------------------------------------

_X_NP = _np.asarray([1.0, -2.0, 3.0])
_D_NP = _np.asarray([0.5, -1.0, 2.5])
_W_NP = _np.asarray([2.0, 1.0, 3.0])
_V_NP = _np.asarray([1.0, -1.0, 2.0])


def test_half_sq_l2_no_data_no_weights(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation()
    expected = float(0.5 * _np.sum(_X_NP**2))
    assert abs(f(x) - expected) < 1e-8


def test_half_sq_l2_with_data(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation(data=d)
    expected = float(0.5 * _np.sum((_X_NP - _D_NP) ** 2))
    assert abs(f(x) - expected) < 1e-8


def test_half_sq_l2_with_weights(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    expected = float(0.5 * _np.sum(_W_NP * (_X_NP - _D_NP) ** 2))
    assert abs(f(x) - expected) < 1e-8


def test_half_sq_l2_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    grad = f.gradient(x)
    fd_grad = finite_diff_gradient(f, _X_NP, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-4, rtol=1e-4)


def test_half_sq_l2_call_and_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation(data=d)
    val, grad = f.call_and_gradient(x)
    assert abs(val - f(x)) < 1e-8
    assert allclose(grad, f.gradient(x))


def test_half_sq_l2_call_and_gradient_with_weights(xp: ModuleType, dev: str):
    """Covers the weighted branch of _call_and_gradient (lines 379-380)."""
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    val, grad = f.call_and_gradient(x)
    assert abs(val - f(x)) < 1e-8
    assert allclose(grad, f.gradient(x))


def test_half_sq_l2_hessian_diag_vec_prod(xp: ModuleType, dev: str):
    """Hessian diagonal is w, so hessian_diag_vec_prod = w * v."""
    x = xp.asarray(_X_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)
    v = xp.asarray(_V_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation(weights=w)
    hv = f.hessian_diag_vec_prod(x, v)
    expected = xp.asarray(_W_NP * _V_NP, device=dev)
    assert allclose(hv, expected)


def test_half_sq_l2_hessian_diag_no_weights(xp: ModuleType, dev: str):
    """Without weights the Hessian diagonal is all-ones, so hessian_diag_vec_prod = v."""
    x = xp.asarray(_X_NP, device=dev)
    v = xp.asarray(_V_NP, device=dev)

    f = ppf.HalfSquaredL2Deviation()
    hv = f.hessian_diag_vec_prod(x, v)
    assert allclose(hv, v)


def test_half_sq_l2_beta_scaling(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d, beta=1.0)
    f3 = ppf.HalfSquaredL2Deviation(data=d, beta=3.0)
    assert abs(f3(x) - 3.0 * f1(x)) < 1e-8
    assert allclose(f3.gradient(x), 3.0 * f1.gradient(x))


# ---------------------------------------------------------------------------
# LogCosh
# ---------------------------------------------------------------------------

_X_LC_NP = _np.asarray([1.0, -2.0, 0.5])
_V_LC_NP = _np.asarray([1.0, -1.0, 2.0])
_DELTA = 2.0


def test_log_cosh_delta_property(xp: ModuleType, dev: str):
    assert ppf.LogCosh().delta is None
    assert ppf.LogCosh(delta=_DELTA).delta == _DELTA


def test_log_cosh_call_at_zero(xp: ModuleType, dev: str):
    """f(0) must be exactly (up to float rounding) zero for any delta."""
    x = xp.asarray(_np.zeros(3), device=dev)
    for f in (ppf.LogCosh(), ppf.LogCosh(delta=_DELTA)):
        assert abs(f(x)) < 1e-6


def test_log_cosh_call(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LC_NP, device=dev)

    f = ppf.LogCosh()
    expected = float(_np.sum(_np.log(_np.cosh(_X_LC_NP))))
    assert abs(f(x) - expected) < 1e-5

    f_d = ppf.LogCosh(delta=_DELTA)
    expected_d = float(_DELTA * _np.sum(_np.log(_np.cosh(_X_LC_NP / _DELTA))))
    assert abs(f_d(x) - expected_d) < 1e-5


def test_log_cosh_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LC_NP, device=dev)

    f = ppf.LogCosh()
    assert allclose(f.gradient(x), finite_diff_gradient(f, _X_LC_NP, xp, dev), atol=1e-4, rtol=1e-4)

    f_d = ppf.LogCosh(delta=_DELTA)
    assert allclose(f_d.gradient(x), finite_diff_gradient(f_d, _X_LC_NP, xp, dev), atol=1e-4, rtol=1e-4)


def test_log_cosh_call_and_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LC_NP, device=dev)

    for f in (ppf.LogCosh(), ppf.LogCosh(delta=_DELTA)):
        val, grad = f.call_and_gradient(x)
        assert abs(val - f(x)) < 1e-8
        assert allclose(grad, f.gradient(x))


def test_log_cosh_hessian_diag_vec_prod(xp: ModuleType, dev: str):
    """Hessian diagonal is sech^2(x/delta)/delta^2 = (1 - tanh^2(x/delta))/delta^2."""
    x = xp.asarray(_X_LC_NP, device=dev)
    v = xp.asarray(_V_LC_NP, device=dev)

    f = ppf.LogCosh()
    expected = xp.asarray((1 - _np.tanh(_X_LC_NP) ** 2) * _V_LC_NP, device=dev)
    assert allclose(f.hessian_diag_vec_prod(x, v), expected, atol=1e-5)

    f_d = ppf.LogCosh(delta=_DELTA)
    expected_d = xp.asarray(
        (1 - _np.tanh(_X_LC_NP / _DELTA) ** 2) / _DELTA * _V_LC_NP, device=dev
    )
    assert allclose(f_d.hessian_diag_vec_prod(x, v), expected_d, atol=1e-5)


def test_log_cosh_beta_scaling(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LC_NP, device=dev)
    f1 = ppf.LogCosh(delta=_DELTA, beta=1.0)
    f3 = ppf.LogCosh(delta=_DELTA, beta=3.0)
    assert abs(f3(x) - 3.0 * f1(x)) < 1e-8
    assert allclose(f3.gradient(x), 3.0 * f1.gradient(x))


def test_log_cosh_overflow_safe(xp: ModuleType, dev: str):
    """Naive cosh(100) overflows float32; our stable form must not."""
    import math

    x = xp.asarray(_np.asarray([100.0, -100.0]), dtype=xp.float32, device=dev)

    f = ppf.LogCosh()
    expected = 2 * (100.0 - math.log(2.0))
    assert abs(f(x) - expected) < 1e-3
    assert all(xp.isfinite(f.gradient(x)))

    f_d = ppf.LogCosh(delta=0.5)
    expected_d = 2 * (100.0 - 0.5 * math.log(2.0))
    assert abs(f_d(x) - expected_d) < 1e-1
    assert all(xp.isfinite(f_d.gradient(x)))


# ---------------------------------------------------------------------------
# SumC1Function / SumC2Function  (via __add__)
# ---------------------------------------------------------------------------


def test_add_two_c2_functions_returns_sumc2(xp: ModuleType, dev: str):
    d1 = xp.asarray(_np.asarray([1.0, 2.0, 3.0]), device=dev)
    d2 = xp.asarray(_np.asarray([0.5, 1.5, 2.5]), device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d1)
    f2 = ppf.HalfSquaredL2Deviation(data=d2)
    s = f1 + f2
    assert isinstance(s, ppf.SumC2Function)


def test_add_c1_and_c2_returns_sumc1(xp: ModuleType, dev: str):
    """Covers the SumC1Function branch of __add__ (line 132) when one operand is C1-only."""

    # Minimal concrete C1Function (not C2) to force the else-branch in __add__
    class _SimpleC1(ppf.C1Function):
        def _call(self, x):
            xp_ = array_api_compat.array_namespace(x)
            return float(xp_.sum(x))

        def _gradient(self, x):
            xp_ = array_api_compat.array_namespace(x)
            return xp_.ones_like(x)

    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f_c2 = ppf.HalfSquaredL2Deviation(data=d)
    f_c1 = _SimpleC1()
    s = f_c2 + f_c1
    assert isinstance(s, ppf.SumC1Function)
    assert not isinstance(s, ppf.SumC2Function)
    assert abs(s(x) - (f_c2(x) + f_c1(x))) < 1e-8


def test_sum_c1_call(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d, beta=1.0)
    f2 = ppf.HalfSquaredL2Deviation(beta=2.0)
    s = f1 + f2
    assert abs(s(x) - (f1(x) + f2(x))) < 1e-8


def test_sum_c1_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d)
    f2 = ppf.HalfSquaredL2Deviation(beta=2.0)
    s = f1 + f2
    assert allclose(s.gradient(x), f1.gradient(x) + f2.gradient(x))


def test_sum_c1_call_and_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d)
    f2 = ppf.HalfSquaredL2Deviation(beta=2.0)
    s = f1 + f2

    val, grad = s.call_and_gradient(x)
    assert abs(val - s(x)) < 1e-8
    assert allclose(grad, s.gradient(x))


def test_sum_c2_hessian_diag_vec_prod(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    v = xp.asarray(_V_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d)
    f2 = ppf.HalfSquaredL2Deviation(weights=w)
    s = cast(ppf.SumC2Function, f1 + f2)

    hv = s.hessian_diag_vec_prod(x, v)
    expected = f1.hessian_diag_vec_prod(x, v) + f2.hessian_diag_vec_prod(x, v)
    assert allclose(hv, expected)


def test_add_chain_three_functions(xp: ModuleType, dev: str):
    x = xp.asarray(_X_NP, device=dev)
    d1 = xp.asarray(_np.asarray([0.0, 0.0, 0.0]), device=dev)
    d2 = xp.asarray(_np.asarray([1.0, 1.0, 1.0]), device=dev)

    f1 = ppf.HalfSquaredL2Deviation(data=d1)
    f2 = ppf.HalfSquaredL2Deviation(data=d2, beta=2.0)
    f3 = ppf.HalfSquaredL2Deviation(beta=0.5)

    s = f1 + f2 + f3
    assert abs(s(x) - (f1(x) + f2(x) + f3(x))) < 1e-8
    assert allclose(s.gradient(x), f1.gradient(x) + f2.gradient(x) + f3.gradient(x))


# ---------------------------------------------------------------------------
# Shared matrix / contamination fixtures (numpy arrays reused across sections)
# ---------------------------------------------------------------------------

_A_NP = _np.asarray([[1.0, 2.0], [3.0, 1.0], [0.5, 2.0]])
_X2_NP = _np.asarray([1.0, 0.5])
_S_NP = _np.asarray([0.1, 0.2, 0.3])
_Y3_NP = _np.asarray([2.0, 2.0, 2.0])
_D3_NP = _np.asarray([2.0, 2.5, 3.0])
_V2_NP = _np.asarray([1.0, -1.0])
_W3_NP = _np.asarray([2.0, 1.0, 3.0])


# ---------------------------------------------------------------------------
# C1AffineObjective
# ---------------------------------------------------------------------------


def test_c1_affine_objective_call(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    s = xp.asarray(_S_NP, device=dev)
    y = xp.asarray(_Y3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.NegPoissonLogL(y, exact=True)
    obj = ppf.C1AffineObjective(loss, op, s)

    pred_np = _A_NP @ _X2_NP + _S_NP
    expected = float(_np.sum(pred_np - _Y3_NP * _np.log(pred_np)))
    assert abs(obj(x) - expected) < 1e-5


def test_c1_affine_objective_gradient(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    s = xp.asarray(_S_NP, device=dev)
    y = xp.asarray(_Y3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.NegPoissonLogL(y)
    obj = ppf.C1AffineObjective(loss, op, s)

    grad = obj.gradient(x)
    fd_grad = finite_diff_gradient(obj, _X2_NP, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-4, rtol=1e-4)


def test_c1_affine_objective_call_and_gradient(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    s = xp.asarray(_S_NP, device=dev)
    y = xp.asarray(_Y3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.NegPoissonLogL(y)
    obj = ppf.C1AffineObjective(loss, op, s)

    val, grad = obj.call_and_gradient(x)
    assert abs(val - obj(x)) < 1e-8
    assert allclose(grad, obj.gradient(x))


def test_c1_affine_objective_no_contamination(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    y = xp.asarray(_Y3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.NegPoissonLogL(y, exact=True)
    obj = ppf.C1AffineObjective(loss, op)  # no s

    pred_np = _A_NP @ _X2_NP
    expected = float(_np.sum(pred_np - _Y3_NP * _np.log(pred_np)))
    assert abs(obj(x) - expected) < 1e-5


# ---------------------------------------------------------------------------
# C2AffineObjective
# ---------------------------------------------------------------------------


def test_c2_affine_objective_call_gradient(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    d = xp.asarray(_D3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.HalfSquaredL2Deviation(data=d)
    obj = ppf.C2AffineObjective(loss, op)

    pred_np = _A_NP @ _X2_NP
    expected_val = float(0.5 * _np.sum((pred_np - _D3_NP) ** 2))
    assert abs(obj(x) - expected_val) < 1e-8

    grad = obj.gradient(x)
    fd_grad = finite_diff_gradient(obj, _X2_NP, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-4, rtol=1e-4)


def test_c2_affine_objective_call_and_gradient(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    d = xp.asarray(_D3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.HalfSquaredL2Deviation(data=d)
    obj = ppf.C2AffineObjective(loss, op)

    val, grad = obj.call_and_gradient(x)
    assert abs(val - obj(x)) < 1e-8
    assert allclose(grad, obj.gradient(x))


def test_c2_affine_objective_hessian_diag_vec_prod(xp: ModuleType, dev: str):
    """H_f(x) v = A^T (diag_H_g(Ax) * (Av)) = A^T (w * (Av)) for HalfSquaredL2."""
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    v = xp.asarray(_V2_NP, device=dev)
    d = xp.asarray(_D3_NP, device=dev)
    w = xp.asarray(_W3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    obj = ppf.C2AffineObjective(loss, op)

    hv = obj.hessian_diag_vec_prod(x, v)

    # manual: A^T (w * (A v))
    Av_np = _A_NP @ _V2_NP
    expected_np = _A_NP.T @ (_W3_NP * Av_np)
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(hv, expected)


def test_c2_affine_objective_hessian_with_contamination(xp: ModuleType, dev: str):
    """Covers the contamination branch inside _hessian_diag_vec_prod (line 606)."""
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    v = xp.asarray(_V2_NP, device=dev)
    s = xp.asarray(_S_NP, device=dev)
    d = xp.asarray(_D3_NP, device=dev)
    w = xp.asarray(_W3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    obj = ppf.C2AffineObjective(loss, op, s)

    hv = obj.hessian_diag_vec_prod(x, v)

    # manual: A^T (w * (A v)),  pred = Ax + s (s cancels in hessian for this loss)
    Av_np = _A_NP @ _V2_NP
    expected_np = _A_NP.T @ (_W3_NP * Av_np)
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(hv, expected)


def test_c2_affine_objective_beta_scaling(xp: ModuleType, dev: str):
    A = xp.asarray(_A_NP, device=dev)
    x = xp.asarray(_X2_NP, device=dev)
    v = xp.asarray(_V2_NP, device=dev)
    d = xp.asarray(_D3_NP, device=dev)

    op = ppo.MatrixOperator(A)
    loss1 = ppf.HalfSquaredL2Deviation(data=d, beta=1.0)
    loss2 = ppf.HalfSquaredL2Deviation(data=d, beta=2.0)
    obj1 = ppf.C2AffineObjective(loss1, op)
    obj2 = ppf.C2AffineObjective(loss2, op)

    assert abs(obj2(x) - 2.0 * obj1(x)) < 1e-8
    assert allclose(obj2.gradient(x), 2.0 * obj1.gradient(x))
    assert allclose(
        obj2.hessian_diag_vec_prod(x, v), 2.0 * obj1.hessian_diag_vec_prod(x, v)
    )


# ---------------------------------------------------------------------------
# NegPoissonLogLListmode
# ---------------------------------------------------------------------------
#
# We build a minimal (2 image voxels, 6 listmode events) system that is the
# exact listmode equivalent of a 3-bin sinogram with integer counts [2, 1, 3].
#
# Sinogram layout:  A_sino (3x2), y_sino = [2, 1, 3], s_sino = [0.1, 0.2, 0.1]
# Listmode layout:  row e of A_lm  == A_sino[bin_e, :]
#                   s_lm[e]        == s_sino[bin_e]
#                   sens           == A_sino^T 1  (full sensitivity image)
#                   c_sino_sum     == sum(s_sino)
#
# Because sum_e log(pred_e) == sum_i y_i log(pred_i) the two formulations are
# mathematically identical; the tests below verify this numerically.

_A_SINO_LM_NP = _np.asarray([[1.0, 2.0], [3.0, 1.0], [0.5, 2.0]], dtype=_np.float64)
_X_LM_NP = _np.asarray([0.5, 1.0])
_V_LM_NP = _np.asarray([1.0, -0.5])
_S_SINO_LM_NP = _np.asarray([0.1, 0.2, 0.1])
_Y_LM_SINO_NP = _np.asarray([2, 1, 3])  # integer counts per sinogram bin

_EVENT_BINS_NP = _np.repeat(_np.arange(3), _Y_LM_SINO_NP)  # [0,0,1,2,2,2]
_A_LM_NP = _A_SINO_LM_NP[_EVENT_BINS_NP]         # (6, 2)
_S_LM_NP = _S_SINO_LM_NP[_EVENT_BINS_NP]          # (6,)
_SENS_LM_NP = _A_SINO_LM_NP.T @ _np.ones(3)       # (2,)
_CONT_SINO_SUM = float(_np.sum(_S_SINO_LM_NP))


def _make_lm_obj(xp, dev) -> ppf.NegPoissonLogLListmode:
    """Construct a NegPoissonLogLListmode on the minimal test system."""
    A_lm = xp.asarray(_A_LM_NP, device=dev, dtype=xp.float32)
    s_lm = xp.asarray(_S_LM_NP, device=dev, dtype=xp.float32)
    sens = xp.asarray(_SENS_LM_NP, device=dev, dtype=xp.float32)
    op = ppo.MatrixOperator(A_lm)
    return ppf.NegPoissonLogLListmode(op, sens, s_lm, _CONT_SINO_SUM)


def test_neg_poisson_logl_lm_call(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)
    f = _make_lm_obj(xp, dev)

    pred_sino = _A_SINO_LM_NP @ _X_LM_NP + _S_SINO_LM_NP
    expected = float(
        _np.dot(_SENS_LM_NP, _X_LM_NP)
        + _CONT_SINO_SUM
        - _np.sum(_Y_LM_SINO_NP * _np.log(pred_sino))
    )
    assert abs(f(x) - expected) < 1e-4


def test_neg_poisson_logl_lm_pred_raises_on_nonpositive(xp: ModuleType, dev: str):
    """_pred must raise ValueError when any predicted count is <= 0."""
    f = _make_lm_obj(xp, dev)
    x_neg = xp.asarray(_np.asarray([-10.0, -10.0]), device=dev, dtype=xp.float32)
    with pytest.raises(ValueError):
        f(x_neg)


def test_neg_poisson_logl_lm_gradient_fd(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)
    f = _make_lm_obj(xp, dev)

    grad = f.gradient(x)
    fd_grad = finite_diff_gradient(f, _X_LM_NP, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-3, rtol=1e-3)


def test_neg_poisson_logl_lm_call_and_gradient(xp: ModuleType, dev: str):
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)
    f = _make_lm_obj(xp, dev)

    val, grad = f.call_and_gradient(x)
    assert abs(val - f(x)) < 1e-6
    assert allclose(grad, f.gradient(x))


def test_neg_poisson_logl_lm_hessian_diag_vec_prod(xp: ModuleType, dev: str):
    """H_lm(x) v = A_lm^T (A_lm v / pred^2); verify against finite differences of gradient."""
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)
    v = xp.asarray(_V_LM_NP, device=dev, dtype=xp.float32)
    f = _make_lm_obj(xp, dev)

    hv = f.hessian_diag_vec_prod(x, v)

    # finite-difference check: d/dt gradient(x + t*v) |_{t=0}
    eps = 1e-4
    xp_v = _X_LM_NP + eps * _V_LM_NP
    xm_v = _X_LM_NP - eps * _V_LM_NP
    fd_hv = (
        f.gradient(xp.asarray(xp_v, device=dev, dtype=xp.float32))
        - f.gradient(xp.asarray(xm_v, device=dev, dtype=xp.float32))
    ) / (2 * eps)
    assert allclose(hv, fd_hv, atol=5e-3, rtol=5e-3)


def test_neg_poisson_logl_lm_matches_sinogram_value(xp: ModuleType, dev: str):
    """NegPoissonLogLListmode and C2AffineObjective(NegPoissonLogL) must agree in value."""
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)

    f_lm = _make_lm_obj(xp, dev)

    A_sino = xp.asarray(_A_SINO_LM_NP, device=dev, dtype=xp.float32)
    y_sino = xp.asarray(_Y_LM_SINO_NP, device=dev, dtype=xp.float32)
    s_sino = xp.asarray(_S_SINO_LM_NP, device=dev, dtype=xp.float32)
    f_sino = ppf.C2AffineObjective(
        ppf.NegPoissonLogL(y_sino, exact=True), ppo.MatrixOperator(A_sino), s_sino
    )

    assert abs(f_lm(x) - f_sino(x)) < 1e-4


def test_neg_poisson_logl_lm_matches_sinogram_gradient(xp: ModuleType, dev: str):
    """Gradient of NegPoissonLogLListmode must match that of the sinogram objective."""
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)

    f_lm = _make_lm_obj(xp, dev)

    A_sino = xp.asarray(_A_SINO_LM_NP, device=dev, dtype=xp.float32)
    y_sino = xp.asarray(_Y_LM_SINO_NP, device=dev, dtype=xp.float32)
    s_sino = xp.asarray(_S_SINO_LM_NP, device=dev, dtype=xp.float32)
    f_sino = ppf.C2AffineObjective(
        ppf.NegPoissonLogL(y_sino, exact=True), ppo.MatrixOperator(A_sino), s_sino
    )

    assert allclose(f_lm.gradient(x), f_sino.gradient(x), atol=1e-4, rtol=1e-4)


def test_neg_poisson_logl_lm_matches_sinogram_hessian(xp: ModuleType, dev: str):
    """Hessian-vector product of NegPoissonLogLListmode must match sinogram objective."""
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)
    v = xp.asarray(_V_LM_NP, device=dev, dtype=xp.float32)

    f_lm = _make_lm_obj(xp, dev)

    A_sino = xp.asarray(_A_SINO_LM_NP, device=dev, dtype=xp.float32)
    y_sino = xp.asarray(_Y_LM_SINO_NP, device=dev, dtype=xp.float32)
    s_sino = xp.asarray(_S_SINO_LM_NP, device=dev, dtype=xp.float32)
    f_sino = ppf.C2AffineObjective(
        ppf.NegPoissonLogL(y_sino, exact=True), ppo.MatrixOperator(A_sino), s_sino
    )

    assert allclose(
        f_lm.hessian_diag_vec_prod(x, v),
        f_sino.hessian_diag_vec_prod(x, v),
        atol=1e-4,
        rtol=1e-4,
    )


# ---------------------------------------------------------------------------
# NegPoissonLogLListmode: optional eps smoothing
# ---------------------------------------------------------------------------


def test_neg_poisson_logl_lm_eps_matches_manual_surrogate(xp: ModuleType, dev: str):
    """With eps > 0 the LM objective equals the expectation-only-shift
    surrogate sum(ybar) - sum(y * log(ybar + eps)) in sinogram space, and
    value / gradient stay mutually consistent (finite differences)."""
    e = 0.25
    x = xp.asarray(_X_LM_NP, device=dev, dtype=xp.float32)

    A_lm = xp.asarray(_A_LM_NP, device=dev, dtype=xp.float32)
    s_lm = xp.asarray(_S_LM_NP, device=dev, dtype=xp.float32)
    sens = xp.asarray(_SENS_LM_NP, device=dev, dtype=xp.float32)
    f = ppf.NegPoissonLogLListmode(
        ppo.MatrixOperator(A_lm), sens, s_lm, _CONT_SINO_SUM, eps=e
    )
    assert f.eps == e

    pred_sino = _A_SINO_LM_NP @ _X_LM_NP + _S_SINO_LM_NP
    expected = float(
        _np.dot(_SENS_LM_NP, _X_LM_NP)
        + _CONT_SINO_SUM
        - _np.sum(_Y_LM_SINO_NP * _np.log(pred_sino + e))
    )
    assert abs(f(x) - expected) < 1e-4

    grad_expected_np = _SENS_LM_NP - _A_LM_NP.T @ (
        1.0 / (_A_LM_NP @ _X_LM_NP + _S_LM_NP + e)
    )
    assert allclose(
        f.gradient(x), xp.asarray(grad_expected_np, device=dev), atol=1e-4, rtol=1e-4
    )
    fd_grad = finite_diff_gradient(f, _X_LM_NP, xp, dev)
    assert allclose(f.gradient(x), fd_grad, atol=1e-3, rtol=1e-3)


def test_neg_poisson_logl_lm_eps_finite_on_zero_pred(xp: ModuleType, dev: str):
    """With zero contamination and x = 0 every per-event prediction is 0:
    eps = 0 raises ValueError, eps > 0 stays finite."""
    e = 0.5
    A_lm = xp.asarray(_A_LM_NP, device=dev, dtype=xp.float32)
    s_zero = xp.zeros(_A_LM_NP.shape[0], device=dev, dtype=xp.float32)
    sens = xp.asarray(_SENS_LM_NP, device=dev, dtype=xp.float32)
    x0 = xp.zeros(2, device=dev, dtype=xp.float32)
    v = xp.ones(2, device=dev, dtype=xp.float32)

    f_exact = ppf.NegPoissonLogLListmode(ppo.MatrixOperator(A_lm), sens, s_zero, 0.0)
    with pytest.raises(ValueError, match="strictly positive"):
        f_exact(x0)

    f_eps = ppf.NegPoissonLogLListmode(
        ppo.MatrixOperator(A_lm), sens, s_zero, 0.0, eps=e
    )
    val = f_eps(x0)
    grad = f_eps.gradient(x0)
    hv = f_eps.hessian_diag_vec_prod(x0, v)
    assert math.isfinite(val)
    assert bool(xp.all(xp.isfinite(grad))) and bool(xp.all(xp.isfinite(hv)))
    # n_events * (-log(eps)) is the exact value at x = 0
    assert abs(val - (-_A_LM_NP.shape[0] * math.log(e))) < 1e-5


def test_neg_poisson_logl_lm_eps_validation(xp: ModuleType, dev: str):
    """Negative eps raises; negative predictions raise even with eps > 0."""
    A_lm = xp.asarray(_A_LM_NP, device=dev, dtype=xp.float32)
    s_lm = xp.asarray(_S_LM_NP, device=dev, dtype=xp.float32)
    sens = xp.asarray(_SENS_LM_NP, device=dev, dtype=xp.float32)

    with pytest.raises(ValueError, match="eps"):
        ppf.NegPoissonLogLListmode(
            ppo.MatrixOperator(A_lm), sens, s_lm, _CONT_SINO_SUM, eps=-0.1
        )

    f = ppf.NegPoissonLogLListmode(
        ppo.MatrixOperator(A_lm), sens, s_lm, _CONT_SINO_SUM, eps=0.5
    )
    x_neg = xp.asarray(_np.asarray([-10.0, -10.0]), device=dev, dtype=xp.float32)
    with pytest.raises(ValueError, match="non-negative"):
        f(x_neg)


# ---------------------------------------------------------------------------
# Gradient finite-difference consistency (random inputs)
# ---------------------------------------------------------------------------


def test_gradient_fd_consistency_neg_poisson(xp: ModuleType, dev: str):
    _np.random.seed(42)
    y_np = _np.asarray(_np.random.uniform(0.5, 5.0, size=8))
    ybar_np = _np.asarray(_np.random.uniform(0.5, 5.0, size=8))

    y = xp.asarray(y_np, device=dev)
    ybar = xp.asarray(ybar_np, device=dev)

    f = ppf.NegPoissonLogL(y)
    grad = f.gradient(ybar)
    fd_grad = finite_diff_gradient(f, ybar_np, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-4, rtol=1e-4)


def test_gradient_fd_consistency_half_sq_l2(xp: ModuleType, dev: str):
    _np.random.seed(123)
    x_np = _np.asarray(_np.random.randn(8))
    d_np = _np.asarray(_np.random.randn(8))
    w_np = _np.asarray(_np.random.uniform(0.5, 2.0, size=8))

    x = xp.asarray(x_np, device=dev)
    d = xp.asarray(d_np, device=dev)
    w = xp.asarray(w_np, device=dev)

    f = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    grad = f.gradient(x)
    fd_grad = finite_diff_gradient(f, x_np, xp, dev)
    assert allclose(grad, fd_grad, atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------
# FunctionWithConjProx base-class methods (lines 37, 42, 46, 65-66, 112-114, 138)
# Exercised via MixedL21Norm, which is a concrete FunctionWithConjProx subclass.
# ---------------------------------------------------------------------------

_GRAD_NP = _np.asarray([[3.0, 0.5, 0.0], [4.0, 0.5, 0.0]], dtype=_np.float64)


def test_function_with_conj_prox_beta_property(xp: ModuleType, dev: str):
    """Covers FunctionWithConjProx.__init__ (37), beta getter (42), beta setter (46)."""
    f = ppf.MixedL21Norm()
    assert f.beta == 1.0          # getter
    f.beta = 2.5                  # setter
    assert f.beta == 2.5


def test_function_with_conj_prox_call(xp: ModuleType, dev: str):
    """Covers FunctionWithConjProx.__call__ (65-66) with beta == 1 and beta != 1."""
    g = xp.asarray(_GRAD_NP, device=dev)

    f1 = ppf.MixedL21Norm(beta=1.0)
    f2 = ppf.MixedL21Norm(beta=3.0)

    v1 = f1(g)
    v2 = f2(g)
    assert abs(v2 - 3.0 * v1) < 1e-8


def test_function_with_conj_prox_prox_convex_conj_beta_one(xp: ModuleType, dev: str):
    """Covers the beta == 1 fast path of prox_convex_conj (lines 112-113)."""
    g = xp.asarray(_GRAD_NP, device=dev)
    f = ppf.MixedL21Norm(beta=1.0)
    result = f.prox_convex_conj(g, sigma=1.0)
    # each column norm should be <= 1 after projection
    xp_ = array_api_compat.array_namespace(result)
    norms = xp_.linalg.vector_norm(result, axis=0)
    assert bool(xp_.all(norms <= 1.0 + 1e-8))


def test_function_with_conj_prox_prox_convex_conj_beta_scaled(xp: ModuleType, dev: str):
    """Covers the beta != 1 rescaling branch of prox_convex_conj (line 114)."""
    g = xp.asarray(_GRAD_NP, device=dev)
    beta = 2.0
    f = ppf.MixedL21Norm(beta=beta)
    result = f.prox_convex_conj(g, sigma=1.0)
    # result should equal beta * prox_{sigma/beta * f*}(g / beta)
    f_unscaled = ppf.MixedL21Norm(beta=1.0)
    xp_ = array_api_compat.array_namespace(result)
    expected = beta * f_unscaled.prox_convex_conj(g / beta, sigma=1.0 / beta)
    assert allclose(result, expected)


def test_function_with_conj_prox_prox_moreau(xp: ModuleType, dev: str):
    """Covers FunctionWithConjProx.prox via Moreau's identity (line 138)."""
    g = xp.asarray(_GRAD_NP, device=dev)
    sigma = 0.5
    f = ppf.MixedL21Norm(beta=1.0)
    # Moreau: prox_f(x) + sigma * prox_{f*/sigma}(x/sigma) == x
    p = f.prox(g, sigma)
    q = f.prox_convex_conj(g / sigma, 1.0 / sigma)
    xp_ = array_api_compat.array_namespace(g)
    assert allclose(p + sigma * q, g)


# ---------------------------------------------------------------------------
# FunctionWithProx base-class methods (lines 164, 169, 173, 192-193, 238-240, 265)
# Exercised via NonNegativeIndicator.
# ---------------------------------------------------------------------------


def test_function_with_prox_beta_property(xp: ModuleType, dev: str):
    """Covers FunctionWithProx.__init__ (164), beta getter (169), beta setter (173)."""
    f = ppf.NonNegativeIndicator()
    assert f.beta == 1.0
    f.beta = 4.0
    assert f.beta == 4.0


def test_function_with_prox_call(xp: ModuleType, dev: str):
    """Covers FunctionWithProx.__call__ (192-193) with beta == 1 and beta != 1."""
    x_nn = xp.asarray(_np.asarray([1.0, 0.0, 2.0]), device=dev)

    f1 = ppf.NonNegativeIndicator(beta=1.0)
    f2 = ppf.NonNegativeIndicator(beta=5.0)

    assert f1(x_nn) == 0.0
    # indicator value is 0; scaling 0 by beta is still 0
    assert f2(x_nn) == 0.0


def test_function_with_prox_prox_beta_one(xp: ModuleType, dev: str):
    """Covers the beta == 1 fast path of FunctionWithProx.prox (lines 238-239)."""
    x = xp.asarray(_np.asarray([1.0, -2.0, 0.5, -0.1]), device=dev)
    f = ppf.NonNegativeIndicator(beta=1.0)
    result = f.prox(x, sigma=1.0)
    expected = xp.asarray(_np.asarray([1.0, 0.0, 0.5, 0.0]), device=dev)
    assert allclose(result, expected)


def test_function_with_prox_prox_beta_scaled(xp: ModuleType, dev: str):
    """Covers the beta != 1 rescaling branch of FunctionWithProx.prox (line 240)."""
    x = xp.asarray(_np.asarray([1.0, -2.0, 0.5]), device=dev)
    # For NonNegativeIndicator, prox is max(x, 0) regardless of effective step size
    f = ppf.NonNegativeIndicator(beta=3.0)
    result = f.prox(x, sigma=1.0)
    expected = xp.asarray(_np.asarray([1.0, 0.0, 0.5]), device=dev)
    assert allclose(result, expected)


def test_function_with_prox_prox_convex_conj_moreau(xp: ModuleType, dev: str):
    """Covers FunctionWithProx.prox_convex_conj via Moreau's identity (line 265)."""
    y = xp.asarray(_np.asarray([1.0, -2.0, 0.5, -0.1]), device=dev)
    sigma = 0.5
    f = ppf.NonNegativeIndicator(beta=1.0)
    # prox_{sigma f*}(y) = min(y, 0)  (projection onto non-positive orthant)
    result = f.prox_convex_conj(y, sigma)
    expected = xp.asarray(_np.asarray([0.0, -2.0, 0.0, -0.1]), device=dev)
    assert allclose(result, expected)


# ---------------------------------------------------------------------------
# NonNegativeIndicator: _call and _prox correctness (lines 1131-1132, 1135-1136)
# ---------------------------------------------------------------------------


def test_non_negative_indicator_call_non_negative(xp: ModuleType, dev: str):
    """_call returns 0.0 when all entries are non-negative (line 1131-1132)."""
    x = xp.asarray(_np.asarray([0.0, 1.0, 5.0]), device=dev)
    f = ppf.NonNegativeIndicator()
    assert f(x) == 0.0


def test_non_negative_indicator_call_with_negative(xp: ModuleType, dev: str):
    """_call returns +inf when any entry is negative (line 1131-1132)."""
    x = xp.asarray(_np.asarray([1.0, -0.5, 2.0]), device=dev)
    f = ppf.NonNegativeIndicator()
    assert f(x) == float("inf")


def test_non_negative_indicator_prox_clips_negatives(xp: ModuleType, dev: str):
    """prox is max(x, 0): clips negatives, preserves positives (lines 1135-1136)."""
    x_np = _np.asarray([3.0, -1.0, 0.0, -0.01, 2.5])
    x = xp.asarray(x_np, device=dev)
    f = ppf.NonNegativeIndicator()

    for sigma in (0.1, 1.0, 10.0):
        result = f.prox(x, sigma)
        expected = xp.asarray(_np.maximum(x_np, 0.0), device=dev)
        assert allclose(result, expected)


def test_non_negative_indicator_prox_dual_is_non_positive_projection(
    xp: ModuleType, dev: str
):
    """prox_convex_conj(y, sigma) = min(y, 0) for any sigma > 0."""
    y_np = _np.asarray([2.0, -3.0, 0.0, 1.5, -0.2])
    y = xp.asarray(y_np, device=dev)
    f = ppf.NonNegativeIndicator()

    result = f.prox_convex_conj(y, sigma=1.0)
    expected = xp.asarray(_np.minimum(y_np, 0.0), device=dev)
    assert allclose(result, expected)


# ---------------------------------------------------------------------------
# MixedL21Norm: _call and _prox_convex_conj correctness (lines 1182-1183, 1186-1189)
# ---------------------------------------------------------------------------

# Gradient field: shape (2, 3) - 2 gradient directions, 3 spatial positions.
# Column norms: [5.0, sqrt(0.5), 0.0]
_MIXED_GRAD_NP = _np.asarray([[3.0, 0.5, 0.0], [4.0, 0.5, 0.0]], dtype=_np.float64)


def test_mixed_l21_norm_call(xp: ModuleType, dev: str):
    """_call computes sum of per-voxel L2 norms (lines 1182-1183)."""
    g = xp.asarray(_MIXED_GRAD_NP, device=dev)
    f = ppf.MixedL21Norm()

    expected = float(5.0 + _np.sqrt(0.5) + 0.0)
    assert abs(f(g) - expected) < 1e-8


def test_mixed_l21_norm_prox_convex_conj_clips_to_unit_ball(xp: ModuleType, dev: str):
    """prox_convex_conj projects each column onto the L2 unit ball (lines 1186-1189)."""
    g = xp.asarray(_MIXED_GRAD_NP, device=dev)
    f = ppf.MixedL21Norm()
    xp_ = array_api_compat.array_namespace(g)

    result = f.prox_convex_conj(g, sigma=1.0)
    norms_out = xp_.linalg.vector_norm(result, axis=0)

    # every column norm must be <= 1
    assert bool(xp_.all(norms_out <= 1.0 + 1e-8))

    # column 0 had norm 5 -> should be normalised to unit vector [3/5, 4/5]
    expected_col0 = xp.asarray(_np.asarray([3.0 / 5.0, 4.0 / 5.0]), device=dev)
    assert allclose(result[:, 0], expected_col0)

    # column 1 had norm < 1 -> unchanged
    assert allclose(result[:, 1], g[:, 1])

    # column 2 was zero -> still zero
    assert allclose(result[:, 2], g[:, 2])


def test_mixed_l21_norm_beta_scaling(xp: ModuleType, dev: str):
    """MixedL21Norm(beta) scales the function value and prox_convex_conj correctly."""
    g = xp.asarray(_MIXED_GRAD_NP, device=dev)
    beta = 2.0
    f1 = ppf.MixedL21Norm(beta=1.0)
    f2 = ppf.MixedL21Norm(beta=beta)

    assert abs(f2(g) - beta * f1(g)) < 1e-8

    # prox_convex_conj with beta should satisfy Moreau decomposition
    sigma = 0.5
    p = f2.prox(g, sigma)
    q = f2.prox_convex_conj(g / sigma, 1.0 / sigma)
    assert allclose(p + sigma * q, g)


# ---------------------------------------------------------------------------
# NegPoissonLogL: prox_convex_conj correctness (lines 518-519)
# ---------------------------------------------------------------------------


def test_neg_poisson_logl_prox_convex_conj_formula(xp: ModuleType, dev: str):
    """Exact mode: prox_{sigma f*}(y) = 0.5*(y+1 - sqrt((y-1)^2 + 4*sigma*data))."""
    y_np = _np.asarray([0.5, 0.8, -0.2, 1.5])
    sigma = 0.7
    y = xp.asarray(y_np, device=dev)
    d = xp.asarray(_Y_NP, device=dev)

    f = ppf.NegPoissonLogL(d, exact=True)
    result = f.prox_convex_conj(y, sigma)

    expected_np = 0.5 * (y_np + 1 - _np.sqrt((y_np - 1) ** 2 + 4 * sigma * _Y_NP))
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(result, expected)


def test_neg_poisson_logl_prox_moreau_identity(xp: ModuleType, dev: str):
    """Moreau: prox_f(x) + sigma * prox_{f*/sigma}(x/sigma) == x (both modes)."""
    x_np = _np.asarray([2.5, 1.5, 2.0, 3.5])
    sigma = 1.2
    x = xp.asarray(x_np, device=dev)
    d = xp.asarray(_Y_NP, device=dev)

    for f in (ppf.NegPoissonLogL(d, exact=True), ppf.NegPoissonLogL(d)):
        p = f.prox(x, sigma)
        q = f.prox_convex_conj(x / sigma, 1.0 / sigma)
        assert allclose(p + sigma * q, x, atol=1e-6, rtol=1e-6)


def test_neg_poisson_logl_prox_convex_conj_beta_scaling(xp: ModuleType, dev: str):
    """With beta != 1, prox_convex_conj applies the scaling identity correctly."""
    y_np = _np.asarray([0.5, 0.8, -0.2, 1.5])
    sigma = 0.5
    beta = 2.0
    y = xp.asarray(y_np, device=dev)
    d = xp.asarray(_Y_NP, device=dev)

    f = ppf.NegPoissonLogL(d)
    f.beta = beta
    result = f.prox_convex_conj(y, sigma)

    # manually: beta * prox_{sigma/beta * f*}(y/beta)
    f_unit = ppf.NegPoissonLogL(d)
    expected = beta * f_unit.prox_convex_conj(y / beta, sigma / beta)
    assert allclose(result, expected)


# ---------------------------------------------------------------------------
# NegPoissonLogL: prox_convex_conj with zero-count bins and in eps mode
# ---------------------------------------------------------------------------


def test_neg_poisson_logl_exact_prox_convex_conj_zero_bins(xp: ModuleType, dev: str):
    """Exact mode prox needs no masking: with d_i = 0 the closed form already
    yields min(y_i, 1), the prox of the conjugate of the linear loss."""
    y_np = _np.asarray([0.5, 0.8, 1.2, -0.3])
    sigma = 0.7
    y = xp.asarray(y_np, device=dev)
    d = xp.asarray(_Y_SAFE_NP, device=dev)

    f = ppf.NegPoissonLogL(d, exact=True)
    result = f.prox_convex_conj(y, sigma)

    expected_np = 0.5 * (
        y_np + 1 - _np.sqrt((y_np - 1) ** 2 + 4 * sigma * _Y_SAFE_NP)
    )
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(result, expected)
    # bins with d = 0 (indices 2, 3) reduce to min(y, 1)
    assert allclose(result[2:], xp.asarray(_np.minimum(y_np[2:], 1.0), device=dev))


def test_neg_poisson_logl_eps_prox_convex_conj_shifted(xp: ModuleType, dev: str):
    """In the default mode the prox belongs to the shifted-Poisson surrogate:
    same closed form with d -> d + eps evaluated at y + sigma*eps."""
    e = 0.25
    y_np = _np.asarray([0.5, 0.8, 1.2, -0.3])
    sigma = 0.7
    y = xp.asarray(y_np, device=dev)
    d = xp.asarray(_Y_SAFE_NP, device=dev)

    f = ppf.NegPoissonLogL(d, eps=e)
    result = f.prox_convex_conj(y, sigma)

    w_np = y_np + sigma * e
    expected_np = 0.5 * (
        w_np + 1 - _np.sqrt((w_np - 1) ** 2 + 4 * sigma * (_Y_SAFE_NP + e))
    )
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(result, expected)


def test_neg_poisson_logl_eps_prox_gradient_fixed_point(xp: ModuleType, dev: str):
    """prox and gradient of one instance refer to the same (surrogate)
    objective: p = prox_f(z, sigma) satisfies p - z + sigma * grad f(p) = 0."""
    z_np = _np.asarray([2.5, 1.5, 0.8, 3.5])
    sigma = 0.6
    z = xp.asarray(z_np, device=dev)
    d = xp.asarray(_Y_NP, device=dev)

    for f in (ppf.NegPoissonLogL(d, eps=0.3), ppf.NegPoissonLogL(d, exact=True)):
        p = f.prox(z, sigma)
        residual = p - z + sigma * f.gradient(p)
        assert allclose(residual, xp.zeros_like(residual), atol=1e-6)


# ---------------------------------------------------------------------------
# HalfSquaredL2Deviation: prox_convex_conj all four branches (lines 756-759)
# ---------------------------------------------------------------------------

_SIGMA_PCC = 2.0
_Y_PCC_NP = _np.asarray([1.0, -0.5, 2.0])


def test_half_sq_l2_prox_convex_conj_no_data_no_weights(xp: ModuleType, dev: str):
    """No data, no weights: result = y / (1 + sigma) (line 758)."""
    y = xp.asarray(_Y_PCC_NP, device=dev)
    f = ppf.HalfSquaredL2Deviation()
    result = f.prox_convex_conj(y, _SIGMA_PCC)
    expected = xp.asarray(_Y_PCC_NP / (1 + _SIGMA_PCC), device=dev)
    assert allclose(result, expected)


def test_half_sq_l2_prox_convex_conj_with_data_no_weights(xp: ModuleType, dev: str):
    """With data, no weights: result = (y - sigma*d) / (1 + sigma) (line 758)."""
    y = xp.asarray(_Y_PCC_NP, device=dev)
    d = xp.asarray(_X_NP, device=dev)
    f = ppf.HalfSquaredL2Deviation(data=d)
    result = f.prox_convex_conj(y, _SIGMA_PCC)
    expected = xp.asarray(
        (_Y_PCC_NP - _SIGMA_PCC * _X_NP) / (1 + _SIGMA_PCC), device=dev
    )
    assert allclose(result, expected)


def test_half_sq_l2_prox_convex_conj_no_data_with_weights(xp: ModuleType, dev: str):
    """No data, with weights: result = w * y / (w + sigma) (line 759)."""
    y = xp.asarray(_Y_PCC_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)
    f = ppf.HalfSquaredL2Deviation(weights=w)
    result = f.prox_convex_conj(y, _SIGMA_PCC)
    expected = xp.asarray(_W_NP * _Y_PCC_NP / (_W_NP + _SIGMA_PCC), device=dev)
    assert allclose(result, expected)


def test_half_sq_l2_prox_convex_conj_with_data_and_weights(xp: ModuleType, dev: str):
    """With data and weights: result = w*(y - sigma*d) / (w + sigma) (line 759)."""
    y = xp.asarray(_Y_PCC_NP, device=dev)
    d = xp.asarray(_X_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)
    f = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    result = f.prox_convex_conj(y, _SIGMA_PCC)
    expected = xp.asarray(
        _W_NP * (_Y_PCC_NP - _SIGMA_PCC * _X_NP) / (_W_NP + _SIGMA_PCC), device=dev
    )
    assert allclose(result, expected)


def test_half_sq_l2_prox_moreau_identity(xp: ModuleType, dev: str):
    """Moreau identity: prox_f(x) + sigma * prox_{f*/sigma}(x/sigma) == x."""
    x = xp.asarray(_X_NP, device=dev)
    d = xp.asarray(_D_NP, device=dev)
    w = xp.asarray(_W_NP, device=dev)
    sigma = 1.5

    f = ppf.HalfSquaredL2Deviation(data=d, weights=w)
    p = f.prox(x, sigma)
    q = f.prox_convex_conj(x / sigma, 1.0 / sigma)
    assert allclose(p + sigma * q, x, atol=1e-6, rtol=1e-6)
