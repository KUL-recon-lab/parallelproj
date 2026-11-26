"""Separable quadratic majorizers (#187).

* ``LinearOperator.abs_apply`` / ``abs_adjoint`` agree with the dense
  :math:`|A|` (including the scale), non-negative operators fall back to
  ``apply`` / ``adjoint``,
* ``separable_majorizer_curvature`` majorizes: for random points and steps
  :math:`f(x + z) \\le f(x) + \\nabla f(x)^T z + \\tfrac12 \\sum_j c_j z_j^2`,
* it dominates the Hessian diagonal, the majorize-minimize step decreases the
  objective monotonically, and :math:`2\\,\\mathrm{diag}(H)` (local curvature)
  is *not* a majorizer for logcosh.
"""

from __future__ import annotations

from types import ModuleType

import array_api_compat
import numpy as np
import pytest

import parallelproj.functions as ppf
import parallelproj.operators as ppo
import parallelproj.projectors as ppp
from parallelproj import to_numpy_array

from .config import pytestmark

IMG_SHAPE = (5, 4)


def _arr(xp, dev, a):
    return xp.asarray(np.asarray(a, dtype=np.float64), device=dev, dtype=xp.float64)


def _np(x) -> np.ndarray:
    return np.asarray(to_numpy_array(x), dtype=np.float64)


def _dense(op: ppo.LinearOperator, xp, dev) -> np.ndarray:
    n = int(np.prod(op.in_shape))
    cols = []
    for j in range(n):
        e = np.zeros(n)
        e[j] = 1.0
        cols.append(_np(op(_arr(xp, dev, e.reshape(op.in_shape)))).ravel())
    return np.stack(cols, axis=1)


def _check_abs(op, xp, dev, seed=0):
    rng = np.random.default_rng(seed)
    M = np.abs(_dense(op, xp, dev))
    x_np = rng.standard_normal(op.in_shape)
    y_np = rng.standard_normal(op.out_shape)
    assert np.allclose(_np(op.abs_apply(_arr(xp, dev, x_np))).ravel(), M @ x_np.ravel())
    assert np.allclose(
        _np(op.abs_adjoint(_arr(xp, dev, y_np))).ravel(), M.T @ y_np.ravel()
    )


# ---------------------------------------------------------------------------
# abs_apply / abs_adjoint
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(5,), (4, 3), (3, 4, 2), (2, 3, 2, 3)])
def test_abs_finite_difference(xp: ModuleType, dev, shape) -> None:
    op = ppo.FiniteForwardDifference(shape)
    op.scale = -1.5
    _check_abs(op, xp, dev)


def test_abs_matrix_elementwise_vstack_adjoint(xp: ModuleType, dev) -> None:
    rng = np.random.default_rng(1)
    A = ppo.MatrixOperator(_arr(xp, dev, rng.standard_normal((6, 4))))
    A.scale = -2.0
    E = ppo.ElementwiseMultiplicationOperator(_arr(xp, dev, rng.standard_normal(4)))
    for op in (A, E, ppo.VstackOperator((A, E)), A.H):
        _check_abs(op, xp, dev)
    assert not A.nonnegative and not A.H.nonnegative


def test_abs_complex(xp: ModuleType, dev) -> None:
    if not array_api_compat.is_numpy_namespace(xp):
        pytest.skip("complex test with NumPy only")
    A_np = np.asarray([[1 + 2j, -1j], [0.5, 3 - 1j], [2j, 1.0]])
    E_np = np.asarray([1 + 1j, -2j])
    x = np.asarray([1.0, -2.0])
    y = np.asarray([1.0, 2.0, 3.0])
    A = ppo.MatrixOperator(A_np)
    E = ppo.ElementwiseMultiplicationOperator(E_np)
    assert np.allclose(A.abs_apply(x), np.abs(A_np) @ x)
    assert np.allclose(A.abs_adjoint(y), np.abs(A_np).T @ y)
    assert np.allclose(E.abs_apply(x), np.abs(E_np) * x)
    assert np.allclose(E.abs_adjoint(x), np.abs(E_np) * x)


def test_abs_nonnegative_operators(xp: ModuleType, dev) -> None:
    """non-negative operators: |A| = A, with |scale|"""
    proj = ppp.ParallelViewProjector2D(
        (4, 4),
        xp.asarray([-1.0, 0.0, 1.0], dtype=xp.float32, device=dev),
        xp.asarray([0.0, 1.0], dtype=xp.float32, device=dev),
        6.0,
        (-1.5, -1.5),
        (1.0, 1.0),
    )
    proj.scale = -2.0
    assert proj.nonnegative
    x = xp.asarray(np.random.default_rng(2).random((4, 4)), dtype=xp.float32, device=dev)
    y = xp.ones(proj.out_shape, dtype=xp.float32, device=dev)
    assert np.allclose(_np(proj.abs_apply(x)), -_np(proj(x)))
    assert np.allclose(_np(proj.abs_adjoint(y)), -_np(proj.adjoint(y)))

    gauss = ppo.GaussianFilterOperator(IMG_SHAPE, sigma=1.0)
    comp = ppo.CompositeLinearOperator((gauss, gauss))
    assert gauss.nonnegative and comp.nonnegative
    _check_abs(comp, xp, dev)

    gauss2 = ppo.GaussianFilterOperator(IMG_SHAPE, sigma=1.0)
    gauss2.scale = -1.0
    assert not ppo.CompositeLinearOperator((gauss, gauss2)).nonnegative


def test_abs_not_implemented(xp: ModuleType, dev) -> None:
    G = ppo.FiniteForwardDifference(IMG_SHAPE)
    comp = ppo.CompositeLinearOperator(
        (G, ppo.ElementwiseMultiplicationOperator(_arr(xp, dev, np.ones(IMG_SHAPE))))
    )
    assert not comp.nonnegative
    with pytest.raises(NotImplementedError, match="abs_apply"):
        comp.abs_apply(_arr(xp, dev, np.ones(IMG_SHAPE)))
    with pytest.raises(NotImplementedError, match="abs_adjoint"):
        comp.abs_adjoint(_arr(xp, dev, np.ones(comp.out_shape)))


# ---------------------------------------------------------------------------
# separable majorizer curvature
# ---------------------------------------------------------------------------


def _objectives(xp, dev) -> dict[str, tuple[ppf.C2Function, np.ndarray]]:
    rng = np.random.default_rng(3)
    img = 3 * rng.random(IMG_SHAPE)
    G = ppo.FiniteForwardDifference(IMG_SHAPE)
    G_scaled = ppo.FiniteForwardDifference(IMG_SHAPE)
    G_scaled.scale = -0.7
    A_np = rng.standard_normal((7, 4))
    A = ppo.MatrixOperator(_arr(xp, dev, A_np))
    s = _arr(xp, dev, rng.random(7))
    data = ppf.C2AffineObjective(
        ppf.HalfSquaredL2Deviation(
            _arr(xp, dev, rng.random(7)), weights=_arr(xp, dev, rng.random(7) + 0.5)
        ),
        A,
        s,
    )
    prior = ppf.C2AffineObjective(ppf.LogCosh(delta=0.3, beta=1.5), G)
    return {
        "logcosh": (ppf.LogCosh(delta=0.5, beta=2.0), 3 * rng.standard_normal(9)),
        "logcosh_no_delta": (ppf.LogCosh(), 3 * rng.standard_normal(9)),
        "half_sq_l2": (
            ppf.HalfSquaredL2Deviation(_arr(xp, dev, np.ones(9)), beta=0.3),
            rng.standard_normal(9),
        ),
        "prior_fd": (prior, img),
        "prior_fd_scaled": (
            ppf.C2AffineObjective(ppf.LogCosh(delta=0.3), G_scaled),
            img,
        ),
        "data_matrix": (data, rng.standard_normal(4)),
        "sum": (
            prior
            + ppf.HalfSquaredL2Deviation(_arr(xp, dev, np.ones(IMG_SHAPE)), beta=0.1),
            img,
        ),
    }


def _gaps(f, x_np, c, xp, dev, n=300, seed=4) -> np.ndarray:
    """majorization gaps (must be >= 0) for random steps of different sizes"""
    rng = np.random.default_rng(seed)
    x = _arr(xp, dev, x_np)
    fx = float(f(x))
    g = _np(f.gradient(x))
    gaps = []
    for _ in range(n):
        z = rng.standard_normal(x_np.shape) * rng.choice([0.01, 0.1, 1.0, 10.0])
        bound = fx + np.sum(g * z) + 0.5 * np.sum(c * z**2)
        gaps.append(bound - float(f(_arr(xp, dev, x_np + z))))
    return np.asarray(gaps)


def test_majorization(xp: ModuleType, dev) -> None:
    for name, (f, x_np) in _objectives(xp, dev).items():
        c = _np(f.separable_majorizer_curvature(_arr(xp, dev, x_np)))
        assert np.all(c >= 0), name
        gaps = _gaps(f, x_np, c, xp, dev)
        assert gaps.min() >= -1e-9 * max(1.0, abs(float(f(_arr(xp, dev, x_np))))), name


def test_majorizer_dominates_hessian_diag(xp: ModuleType, dev) -> None:
    for name, (f, x_np) in _objectives(xp, dev).items():
        x = _arr(xp, dev, x_np)
        c = _np(f.separable_majorizer_curvature(x))
        assert np.all(c >= _np(f.hessian_diag(x)) - 1e-12), name


def test_local_curvature_is_not_a_majorizer(xp: ModuleType, dev) -> None:
    """2 diag(H) (with psi'') violates the majorization for logcosh"""
    f, x_np = _objectives(xp, dev)["prior_fd"]
    c = 2 * _np(f.hessian_diag(_arr(xp, dev, x_np)))
    assert _gaps(f, x_np, c, xp, dev, n=1000).min() < 0


def test_majorize_minimize_monotone(xp: ModuleType, dev) -> None:
    """x+ = x - grad / c never increases the objective"""
    f, x_np = _objectives(xp, dev)["sum"]
    x = _arr(xp, dev, x_np)
    values = [float(f(x))]
    for _ in range(30):
        x = x - f.gradient(x) / f.separable_majorizer_curvature(x)
        values.append(float(f(x)))
    assert np.all(np.diff(values) <= 1e-9 * abs(values[0]))
    assert values[-1] < values[0]


def test_logcosh_curvature_near_zero(xp: ModuleType, dev) -> None:
    """Huber's curvature tanh(t/d)/t is continuous at 0 with limit 1/d"""
    delta = 0.5
    f = ppf.LogCosh(delta=delta)
    t = np.asarray([-1e-7, 0.0, 1e-7, 4.9e-5, 5.1e-5, 0.3, -2.0])
    c = _np(f.separable_majorizer_curvature(_arr(xp, dev, t)))
    safe_t = np.where(t == 0, 1.0, t)
    ref = np.where(t == 0, 1 / delta, np.tanh(t / delta) / safe_t)
    assert np.allclose(c, ref, rtol=1e-9)


def test_affine_curvature_follows_operator_scale(xp: ModuleType, dev) -> None:
    """|A| 1 is cached without the scale: a later scale change is respected"""
    G = ppo.FiniteForwardDifference(IMG_SHAPE)
    f = ppf.C2AffineObjective(ppf.HalfSquaredL2Deviation(_arr(xp, dev, np.zeros(G.out_shape))), G)
    x = _arr(xp, dev, np.ones(IMG_SHAPE))
    c1 = _np(f.separable_majorizer_curvature(x))
    G.scale = 3.0
    c3 = _np(f.separable_majorizer_curvature(x))
    assert np.allclose(c3, 9 * c1)


def test_majorizer_not_implemented(xp: ModuleType, dev) -> None:
    y = _arr(xp, dev, np.ones(3))
    with pytest.raises(NotImplementedError, match="NegPoissonLogL"):
        ppf.NegPoissonLogL(y).separable_majorizer_curvature(y)


def test_projectors_are_nonnegative(xp: ModuleType, dev) -> None:
    """all PET projectors report non-negative system matrices (|A| = A)"""
    from .test_projector_streams import _projectors

    for name, proj in _projectors(xp, dev).items():
        assert proj.nonnegative, name
        x = xp.ones(proj.in_shape, dtype=xp.float32, device=dev)
        assert np.allclose(_np(proj.abs_apply(x)), _np(proj(x))), name
