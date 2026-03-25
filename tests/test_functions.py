from __future__ import annotations

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

    f = ppf.NegPoissonLogL(y)
    expected = float(_np.sum(_YBAR_NP - _Y_NP * _np.log(_YBAR_NP)))
    assert abs(f(ybar) - expected) < 1e-5


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

    f = ppf.NegPoissonLogL(y)
    hv = f.hessian_diag_vec_prod(ybar, v)
    expected = xp.asarray(_Y_NP / _YBAR_NP**2 * v_np, device=dev)
    assert allclose(hv, expected)


def test_neg_poisson_logl_beta_scaling(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)

    f1 = ppf.NegPoissonLogL(y)
    f2 = ppf.NegPoissonLogL(y)
    f2.beta = 2.0
    assert abs(f2(ybar) - 2.0 * f1(ybar)) < 1e-8
    assert allclose(f2.gradient(ybar), 2.0 * f1.gradient(ybar))


# ---------------------------------------------------------------------------
# NegPoissonLogLSafe
# ---------------------------------------------------------------------------

_Y_SAFE_NP = _np.asarray([3.0, 1.0, 0.0, 0.0])
_YBAR_SAFE_NP = _np.asarray([2.5, 1.5, 0.0, 0.0])
_MASK_NP = _np.asarray([True, True, False, False])


def test_neg_poisson_logl_safe_call_no_virtual_bins(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)
    mask = xp.asarray(_np.asarray([True, True, True, True]), device=dev)

    f_safe = ppf.NegPoissonLogLSafe(y, mask)
    f_ref = ppf.NegPoissonLogL(y)
    assert abs(f_safe(ybar) - f_ref(ybar)) < 1e-8


def test_neg_poisson_logl_safe_gradient_no_virtual_bins(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_NP, device=dev)
    ybar = xp.asarray(_YBAR_NP, device=dev)
    mask = xp.asarray(_np.asarray([True, True, True, True]), device=dev)

    f_safe = ppf.NegPoissonLogLSafe(y, mask)
    f_ref = ppf.NegPoissonLogL(y)
    assert allclose(f_safe.gradient(ybar), f_ref.gradient(ybar))


def test_neg_poisson_logl_safe_virtual_bins_zero_out(xp: ModuleType, dev: str):
    """Virtual bins (mask=False, ybar=0) must contribute 0 to the function value."""
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)
    mask = xp.asarray(_MASK_NP, device=dev)

    f = ppf.NegPoissonLogLSafe(y, mask)
    val = f(ybar)

    # only active bins contribute
    expected = float(
        _np.sum(_YBAR_SAFE_NP[:2] - _Y_SAFE_NP[:2] * _np.log(_YBAR_SAFE_NP[:2]))
    )
    assert abs(val - expected) < 1e-5


def test_neg_poisson_logl_safe_gradient_virtual_bins(xp: ModuleType, dev: str):
    """Virtual bins must have gradient value = 1 (the correct limiting value)."""
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)
    mask = xp.asarray(_MASK_NP, device=dev)

    f = ppf.NegPoissonLogLSafe(y, mask)
    grad = f.gradient(ybar)

    expected_np = _np.concatenate(
        [1.0 - _Y_SAFE_NP[:2] / _YBAR_SAFE_NP[:2], _np.asarray([1.0, 1.0])]
    )
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(grad, expected)


def test_neg_poisson_logl_safe_hessian_diag_virtual_bins(xp: ModuleType, dev: str):
    """Virtual bins must be zero in the diagonal Hessian-vector product."""
    v_np = _np.asarray([1.0, 1.0, 1.0, 1.0])
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)
    mask = xp.asarray(_MASK_NP, device=dev)
    v = xp.asarray(v_np, device=dev)

    f = ppf.NegPoissonLogLSafe(y, mask)
    hv = f.hessian_diag_vec_prod(ybar, v)

    expected_np = _np.concatenate(
        [
            _Y_SAFE_NP[:2] / _YBAR_SAFE_NP[:2] ** 2 * v_np[:2],
            _np.asarray([0.0, 0.0]),
        ]
    )
    expected = xp.asarray(expected_np, device=dev)
    assert allclose(hv, expected)


def test_neg_poisson_logl_safe_call_and_gradient(xp: ModuleType, dev: str):
    y = xp.asarray(_Y_SAFE_NP, device=dev)
    ybar = xp.asarray(_YBAR_SAFE_NP, device=dev)
    mask = xp.asarray(_MASK_NP, device=dev)

    f = ppf.NegPoissonLogLSafe(y, mask)
    val, grad = f.call_and_gradient(ybar)
    assert abs(val - f(ybar)) < 1e-8
    assert allclose(grad, f.gradient(ybar))


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
    loss = ppf.NegPoissonLogL(y)
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
    loss = ppf.NegPoissonLogL(y)
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
