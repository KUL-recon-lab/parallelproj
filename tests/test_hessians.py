"""Hessian-vector products, Hessian diagonals and ``abs_sq_adjoint`` (#185).

* ``hessian_vec_prod`` of every C2 function agrees with a central finite
  difference of its gradient,
* ``hessian_diag`` agrees with :math:`e_j^T H e_j` computed from
  ``hessian_vec_prod`` for unit vectors :math:`e_j`,
* ``LinearOperator.abs_sq_adjoint`` agrees with the transpose of the dense
  element-wise squared matrix, including the operator scale,
* the deprecated ``hessian_diag_vec_prod`` (alias of ``hessian_vec_prod``) and
  subclasses implementing the old ``_hessian_diag_vec_prod`` still work.
"""

from __future__ import annotations

from types import ModuleType

import array_api_compat
import numpy as np
import pytest

import parallelproj.functions as ppf
import parallelproj.operators as ppo

from .config import pytestmark

RNG = np.random.default_rng(42)
A_NP = RNG.uniform(0.1, 1.0, size=(6, 4))  # non-negative system matrix
S_NP = RNG.uniform(0.1, 0.5, size=6)  # contamination
Y_NP = np.asarray([3.0, 0.0, 5.0, 2.0, 1.0, 4.0])  # counts, one zero bin
X_NP = RNG.uniform(0.5, 2.0, size=4)  # evaluation point
IMG_SHAPE = (3, 4)
IMG_NP = RNG.uniform(0.0, 2.0, size=IMG_SHAPE)
IMG_NP[1, 2] = 0.0  # a cold voxel: H x is negative there, diag(H) is not


def _arr(xp, dev, a):
    return xp.asarray(np.asarray(a, dtype=np.float64), device=dev, dtype=xp.float64)


def _to_numpy(x) -> np.ndarray:
    from parallelproj import to_numpy_array

    return np.asarray(to_numpy_array(x), dtype=np.float64)


def _functions(xp, dev) -> dict[str, tuple[ppf.C2Function, np.ndarray]]:
    """(function, evaluation point) of every C2 function type"""
    A = ppo.MatrixOperator(_arr(xp, dev, A_NP))
    y = _arr(xp, dev, Y_NP)
    s = _arr(xp, dev, S_NP)
    G = ppo.FiniteForwardDifference(IMG_SHAPE)
    lm_rows = A_NP[[0, 0, 2, 2, 3, 5]]
    lm = ppf.NegPoissonLogLListmode(
        ppo.MatrixOperator(_arr(xp, dev, lm_rows)),
        _arr(xp, dev, A_NP.sum(axis=0)),
        _arr(xp, dev, S_NP[[0, 0, 2, 2, 3, 5]]),
        float(S_NP.sum()),
    )
    data_fid = ppf.C2AffineObjective(ppf.NegPoissonLogL(y), A, s)
    data_fid.beta = 0.7
    prior = ppf.C2AffineObjective(ppf.LogCosh(delta=0.8, beta=1.3), G)
    weighted = ppf.HalfSquaredL2Deviation(
        _arr(xp, dev, Y_NP), weights=_arr(xp, dev, np.linspace(0.5, 2, 6))
    )
    return {
        "poisson": (ppf.NegPoissonLogL(y), A_NP @ X_NP + S_NP),
        "poisson_exact": (ppf.NegPoissonLogL(y, exact=True), A_NP @ X_NP + S_NP),
        "half_sq_l2": (ppf.HalfSquaredL2Deviation(_arr(xp, dev, Y_NP)), Y_NP + 0.3),
        "half_sq_l2_weighted": (weighted, Y_NP + 0.3),
        "log_cosh": (ppf.LogCosh(delta=0.5, beta=2.0), X_NP - 1.0),
        "affine_poisson": (data_fid, X_NP),
        "affine_logcosh_fd": (prior, IMG_NP),
        "listmode": (lm, X_NP),
        "sum": (
            ppf.C2AffineObjective(ppf.NegPoissonLogL(y), A, s)
            + ppf.C2AffineObjective(
                ppf.HalfSquaredL2Deviation(_arr(xp, dev, np.zeros(4)), beta=0.4),
                ppo.MatrixOperator(_arr(xp, dev, np.eye(4))),
            ),
            X_NP,
        ),
    }


def _fd_hessian_vec_prod(f, x_np, v_np, xp, dev, eps=1e-5) -> np.ndarray:
    """central finite difference of the gradient along v"""
    gp = f.gradient(_arr(xp, dev, x_np + eps * v_np))
    gm = f.gradient(_arr(xp, dev, x_np - eps * v_np))
    return (_to_numpy(gp) - _to_numpy(gm)) / (2 * eps)


def test_hessian_vec_prod_vs_finite_differences(xp: ModuleType, dev) -> None:
    for name, (f, x_np) in _functions(xp, dev).items():
        v_np = np.random.default_rng(7).standard_normal(x_np.shape)
        hv = _to_numpy(f.hessian_vec_prod(_arr(xp, dev, x_np), _arr(xp, dev, v_np)))
        ref = _fd_hessian_vec_prod(f, x_np, v_np, xp, dev)
        assert np.allclose(hv, ref, rtol=1e-5, atol=1e-6), name


def test_hessian_diag_vs_unit_vectors(xp: ModuleType, dev) -> None:
    for name, (f, x_np) in _functions(xp, dev).items():
        if name == "listmode":
            continue  # MatrixOperator lm_op: tested separately below
        x = _arr(xp, dev, x_np)
        diag = _to_numpy(f.hessian_diag(x)).ravel()
        n = x_np.size
        ref = np.zeros(n)
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            e = e.reshape(x_np.shape)
            ref[j] = np.sum(e * _to_numpy(f.hessian_vec_prod(x, _arr(xp, dev, e))))
        assert np.allclose(diag, ref, rtol=1e-6, atol=1e-9), name


def test_hessian_diag_listmode(xp: ModuleType, dev) -> None:
    f, x_np = _functions(xp, dev)["listmode"]
    x = _arr(xp, dev, x_np)
    diag = _to_numpy(f.hessian_diag(x))
    ref = [
        np.sum(np.eye(4)[j] * _to_numpy(f.hessian_vec_prod(x, _arr(xp, dev, np.eye(4)[j]))))
        for j in range(4)
    ]
    assert np.allclose(diag, ref, rtol=1e-6)


def test_hessian_diag_non_negative_but_hvp_negative(xp: ModuleType, dev) -> None:
    """The motivating case of #185: H x < 0 at a cold voxel, diag(H) >= 0."""
    f, x_np = _functions(xp, dev)["affine_logcosh_fd"]
    x = _arr(xp, dev, x_np)
    hx = _to_numpy(f.hessian_vec_prod(x, x))
    diag = _to_numpy(f.hessian_diag(x))
    assert hx[1, 2] < 0
    assert np.all(diag >= 0)


def test_hessian_diag_not_available(xp: ModuleType, dev) -> None:
    """operators without abs_sq_adjoint raise NotImplementedError"""
    G = ppo.FiniteForwardDifference(IMG_SHAPE)
    op = ppo.CompositeLinearOperator((G, ppo.ElementwiseMultiplicationOperator(
        _arr(xp, dev, np.ones(IMG_SHAPE))
    )))
    f = ppf.C2AffineObjective(ppf.LogCosh(), op)
    x = _arr(xp, dev, IMG_NP)
    f.hessian_vec_prod(x, x)  # always available
    with pytest.raises(NotImplementedError, match="abs_sq_adjoint"):
        f.hessian_diag(x)


# ---------------------------------------------------------------------------
# LinearOperator.abs_sq_adjoint
# ---------------------------------------------------------------------------


def _dense(op: ppo.LinearOperator, xp, dev) -> np.ndarray:
    """dense matrix of the (scaled) operator, columns = images of unit vectors"""
    n = int(np.prod(op.in_shape))
    cols = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        cols.append(_to_numpy(op(_arr(xp, dev, e.reshape(op.in_shape)))).ravel())
    return np.stack(cols, axis=1)


@pytest.mark.parametrize("shape", [(5,), (4, 3), (3, 4, 2), (2, 3, 2, 3)])
def test_abs_sq_adjoint_finite_difference(xp: ModuleType, dev, shape) -> None:
    op = ppo.FiniteForwardDifference(shape)
    op.scale = -1.5
    y_np = np.random.default_rng(3).standard_normal(op.out_shape)
    res = _to_numpy(op.abs_sq_adjoint(_arr(xp, dev, y_np))).ravel()
    ref = (np.abs(_dense(op, xp, dev)) ** 2).T @ y_np.ravel()
    assert np.allclose(res, ref)


def test_abs_sq_adjoint_matrix_elementwise_vstack(xp: ModuleType, dev) -> None:
    A = ppo.MatrixOperator(_arr(xp, dev, A_NP - 0.5))
    A.scale = 2.0
    E = ppo.ElementwiseMultiplicationOperator(_arr(xp, dev, X_NP - 1.0))
    V = ppo.VstackOperator((A, E))
    for op in (A, E, V):
        y_np = np.random.default_rng(5).standard_normal(op.out_shape)
        res = _to_numpy(op.abs_sq_adjoint(_arr(xp, dev, y_np)))
        ref = (np.abs(_dense(op, xp, dev)) ** 2).T @ y_np.ravel()
        assert np.allclose(res, ref)


def test_abs_sq_adjoint_complex(xp: ModuleType, dev) -> None:
    """complex matrix / elementwise operators use |a|^2 (NumPy only)"""
    if not array_api_compat.is_numpy_namespace(xp):
        pytest.skip("complex test with NumPy only")
    A_np = np.asarray([[1 + 2j, -1j], [0.5, 3 - 1j], [2j, 1.0]])
    A = ppo.MatrixOperator(A_np)
    E = ppo.ElementwiseMultiplicationOperator(np.asarray([1 + 1j, -2j]))
    y = np.asarray([1.0, 2.0, 3.0])
    assert np.allclose(A.abs_sq_adjoint(y), (np.abs(A_np) ** 2).T @ y)
    assert np.allclose(E.abs_sq_adjoint(y[:2]), np.asarray([2.0, 4.0]) * y[:2])


def test_abs_sq_adjoint_not_implemented(xp: ModuleType, dev) -> None:
    op = ppo.GaussianFilterOperator(IMG_SHAPE, sigma=1.0)
    with pytest.raises(NotImplementedError, match="GaussianFilterOperator"):
        op.abs_sq_adjoint(_arr(xp, dev, IMG_NP))


# ---------------------------------------------------------------------------
# deprecated API
# ---------------------------------------------------------------------------


def test_hessian_diag_vec_prod_deprecated(xp: ModuleType, dev) -> None:
    f, x_np = _functions(xp, dev)["affine_logcosh_fd"]
    x = _arr(xp, dev, x_np)
    with pytest.warns(DeprecationWarning, match="hessian_vec_prod"):
        old = f.hessian_diag_vec_prod(x, x)
    assert np.allclose(_to_numpy(old), _to_numpy(f.hessian_vec_prod(x, x)))


class _LegacyQuadratic(ppf.C2Function):
    """subclass written against the old API (only _hessian_diag_vec_prod)"""

    def _call(self, x):
        return float(0.5 * array_api_compat.array_namespace(x).sum(x**2))

    def _gradient(self, x):
        return x

    def _hessian_diag_vec_prod(self, x, v):
        return v


class _NoHessian(ppf.C2Function):
    def _call(self, x):
        return 0.0

    def _gradient(self, x):
        return x


def test_legacy_subclass(xp: ModuleType, dev) -> None:
    x = _arr(xp, dev, X_NP)
    f = _LegacyQuadratic(beta=2.0)
    with pytest.warns(DeprecationWarning, match="_hessian_vec_prod"):
        hv = f.hessian_vec_prod(x, x)
    assert np.allclose(_to_numpy(hv), 2 * X_NP)
    with pytest.raises(NotImplementedError, match="_hessian_diag"):
        f.hessian_diag(x)
    g = _NoHessian()
    with pytest.raises(NotImplementedError, match="_hessian_vec_prod"):
        g.hessian_vec_prod(x, x)
