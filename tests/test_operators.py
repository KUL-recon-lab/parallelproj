from __future__ import annotations

import importlib
import pytest
import math
import numpy.random as nprnd
import parallelproj.operators as ppo
import array_api_compat
import array_api_compat.numpy as np

from types import ModuleType
from math import prod
from unittest.mock import patch

from .config import pytestmark


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
    inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


# ---------------------------------------------------------------------------------------


def test_matrix(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray([[1.0, 2.0], [-3.0, 2.0], [-1.0, -1.0]], device=dev)
    x = xp.asarray([-2.0, 1.0], device=dev)

    op = ppo.MatrixOperator(A)

    # set scale that is not 1
    scale_fac = -2.5
    op.scale = scale_fac

    assert op.scale == scale_fac
    assert allclose(op.A, A)

    with pytest.raises(ValueError):
        op.scale = xp.ones(1, device=dev)

    # test call to norm
    op_norm = op.norm(xp, dev)

    assert op.adjointness_test(xp, dev)
    assert allclose(scale_fac * (A @ x), op(x))


def test_adjoint_operator(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray([[1.0, 2.0], [-3.0, 2.0], [-1.0, -1.0]], device=dev)
    x = xp.asarray([-2.0, 1.0], device=dev)
    y = xp.asarray([1.0, 0.0, -1.0], device=dev)

    op = ppo.MatrixOperator(A)
    op_H = op.H

    # shapes are swapped
    assert op_H.in_shape == op.out_shape
    assert op_H.out_shape == op.in_shape

    # forward of A.H equals adjoint of A
    assert allclose(op_H(y), op.adjoint(y))

    # adjoint of A.H equals forward of A
    assert allclose(op_H.adjoint(x), op(x))

    # A.H is itself a valid linear operator
    assert op_H.adjointness_test(xp, dev)

    # scale propagation: setting A.scale updates A.H.scale to its conjugate
    op.scale = 2.0
    assert op_H.scale == 2.0  # real scale: conjugate of 2.0 is 2.0

    # setting A.H.scale propagates back to A as conjugate
    op_H.scale = 3.0
    assert op.scale == 3.0


def test_adjoint_operator_complex_scale(xp: ModuleType, dev: str):
    A = xp.asarray([[1.0, 2.0], [-3.0, 2.0], [-1.0, -1.0]], device=dev)

    op = ppo.MatrixOperator(A)
    op_H = op.H

    # complex scale: A.H.scale == conj(A.scale)
    op.scale = 1.0 + 2.0j
    assert op_H.scale == (1.0 - 2.0j)

    # setting A.H.scale = α sets A.scale = conj(α)
    op_H.scale = 3.0 + 1.0j
    assert op.scale == (3.0 - 1.0j)


def test_complex_matrix(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray(
        [[1.0, 2j], [-3.0, 2.0], [-1.0, -1.0]], device=dev, dtype=xp.complex128
    )
    x = xp.asarray([-2.0, 1.0], device=dev, dtype=xp.complex128)

    op = ppo.MatrixOperator(A)
    assert op.adjointness_test(xp, dev, iscomplex=True)

    n = op.norm(xp, dev, iscomplex=True)

    assert allclose((A @ x), op(x))


def test_elementwise(xp: ModuleType, dev: str):
    np.random.seed(0)

    v = xp.asarray([3.0, -1.0], device=dev)
    x = xp.asarray([-2.0, 1.0], device=dev)

    op = ppo.ElementwiseMultiplicationOperator(v)
    # test call to norm
    op_norm = op.norm(xp, dev)

    assert xp.all(op.values == v)

    assert op.adjointness_test(xp, dev)
    assert allclose(v * x, op(x))


def test_tofnontofelemenwise(xp: ModuleType, dev: str):
    np.random.seed(0)

    x = xp.reshape(xp.arange(3 * 3 * 2, device=dev, dtype=xp.float32), (3, 3, 2))
    v = xp.reshape(xp.arange(3 * 3, device=dev, dtype=xp.float32), (3, 3))

    op = ppo.TOFNonTOFElementwiseMultiplicationOperator(x.shape, v)
    # test call to norm

    assert xp.all(op.values == v)

    assert op.adjointness_test(xp, dev)
    assert allclose(v * x[..., 0], op(x)[..., 0])
    assert allclose(v * x[..., 1], op(x)[..., 1])


def test_elemenwise_complex(xp: ModuleType, dev: str):
    np.random.seed(0)

    v = xp.asarray([3j, -1.0], device=dev, dtype=xp.complex128)
    x = xp.asarray([-2.0, 1j], device=dev, dtype=xp.complex128)

    op = ppo.ElementwiseMultiplicationOperator(v)
    # test call to norm
    op_norm = op.norm(xp, dev, iscomplex=True)

    assert op.adjointness_test(xp, dev, iscomplex=True)
    assert allclose(v * x, op(x))


def test_tofnontofelemenwise_complex(xp: ModuleType, dev: str):
    np.random.seed(0)

    x = xp.ones((3, 3, 2), device=dev, dtype=xp.complex128)
    x[0, 0, 1] = 2.0 + 1j
    x[1, 1, 0] = 1.0 - 2j
    v = xp.ones((3, 3), device=dev, dtype=xp.complex128)
    v[2, 2] = 3.0 + 2j
    v[1, 2] = -4.0 + 1j

    op = ppo.TOFNonTOFElementwiseMultiplicationOperator(x.shape, v)
    # test call to norm

    assert op.adjointness_test(xp, dev, iscomplex=True)
    assert allclose(v * x[..., 0], op(x)[..., 0])
    assert allclose(v * x[..., 1], op(x)[..., 1])


def test_gaussian(xp: ModuleType, dev: str):
    np.random.seed(0)
    in_shape = (32, 32)
    sigma1 = 2.3

    op = ppo.GaussianFilterOperator(in_shape, sigma=sigma1)
    assert op.adjointness_test(xp, dev)

    sigma2 = xp.asarray([2.3, 1.2], device=dev)
    op = ppo.GaussianFilterOperator(in_shape, sigma=sigma2)
    assert op.adjointness_test(xp, dev)


def test_composite(xp: ModuleType, dev: str):
    np.random.seed(0)

    A = xp.asarray([[1.0, 2.0], [-3.0, 2.0], [-1.0, -1.0]], device=dev)
    x = xp.asarray([-2.0, 1.0], device=dev)
    v = xp.asarray([3.0, -1.0, 2.0], device=dev)

    op1 = ppo.ElementwiseMultiplicationOperator(v)
    op2 = ppo.MatrixOperator(A)

    op = ppo.CompositeLinearOperator([op1, op2])

    assert op.operators == [op1, op2]

    # test call to norm
    op_norm = op.norm(xp, dev)

    assert op.adjointness_test(xp, dev)
    assert allclose(v * (A @ x), op(x))
    assert [x.out_shape for x in op] == [op1.out_shape, op2.out_shape]


def test_vstack_mismatched_in_shape_raises(xp: ModuleType, dev: str):
    A1 = ppo.GaussianFilterOperator((16, 11), sigma=1.0)
    A2 = ppo.GaussianFilterOperator((8, 11), sigma=1.0)  # different in_shape

    with pytest.raises(ValueError, match="in_shape"):
        ppo.VstackOperator((A1, A2))


def test_vstack(xp: ModuleType, dev: str):
    np.random.seed(0)
    in_shape = (16, 11)

    A1 = ppo.GaussianFilterOperator(in_shape, sigma=1.0)
    A2 = ppo.ElementwiseMultiplicationOperator(
        xp.asarray(np.random.rand(*in_shape), device=dev)
    )
    A3 = ppo.GaussianFilterOperator(in_shape, sigma=2.0)

    A = ppo.VstackOperator((A1, A2, A3))
    # test call to norm
    A_norm = A.norm(xp, dev)

    assert A.adjointness_test(xp, dev)

    x = xp.asarray(np.random.rand(*in_shape), device=dev)
    x_fwd = A(x)

    assert allclose(
        x_fwd,
        xp.concat(
            (
                xp.reshape(A1(x), (-1,)),
                xp.reshape(A2(x), (-1,)),
                xp.reshape(A3(x), (-1,)),
            )
        ),
    )


def test_operator_sequence(xp: ModuleType, dev: str):
    np.random.seed(0)
    in_shape = (3,)

    A1 = ppo.MatrixOperator(xp.asarray(np.random.randn(4, 3), device=dev))
    A2 = ppo.MatrixOperator(xp.asarray(np.random.randn(5, 3), device=dev))
    A3 = ppo.MatrixOperator(xp.asarray(np.random.randn(2, 3), device=dev))

    A = ppo.LinearOperatorSequence([A1, A2, A3])

    assert len(A) == 3

    # test call to norms
    ns = A.norms(xp, dev)
    assert np.isclose(ns[0], A1.norm(xp, dev))
    assert np.isclose(ns[1], A2.norm(xp, dev))
    assert np.isclose(ns[2], A3.norm(xp, dev))

    x = xp.asarray(np.random.rand(*in_shape), device=dev)
    x_fwd = A(x)

    assert allclose(x_fwd[0], A1(x))
    assert allclose(x_fwd[1], A2(x))
    assert allclose(x_fwd[2], A3(x))

    y = A.adjoint(x_fwd)
    tmp = sum([Ak.adjoint(x_fwd[k]) for k, Ak in enumerate(A)])

    assert allclose(y, tmp)

    assert A.in_shape == A3.in_shape
    assert A.out_shapes == [A1.out_shape, A2.out_shape, A3.out_shape]
    assert A.operators == [A1, A2, A3]


def test_finite_difference(xp: ModuleType, dev: str):
    # 1D tests
    A = ppo.FiniteForwardDifference((3,))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    n = A.norm(xp, dev)
    # test adjointness
    assert A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1] == 1)

    # 2D tests
    A = ppo.FiniteForwardDifference((5, 3))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    # test call to norm
    n = A.norm(xp, dev)
    # test adjointness
    assert A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1, :] == 3)
    assert xp.all(y[1, :, :-1] == 1)

    # 3D tests
    A = ppo.FiniteForwardDifference((5, 3, 4))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    # test adjointness
    assert A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1, :, :] == 12)
    assert xp.all(y[1, :, :-1, :] == 4)
    assert xp.all(y[2, :, :, :-1] == 1)

    # 4D tests
    A = ppo.FiniteForwardDifference((5, 3, 4, 6))
    x = xp.reshape(xp.arange(prod(A.in_shape), device=dev), A.in_shape)

    # test adjointness
    assert A.adjointness_test(xp, dev)

    # test simple forward
    y = A(x)
    assert xp.all(y[0, :-1, :, :, :] == 72)
    assert xp.all(y[1, :, :-1, :, :] == 24)
    assert xp.all(y[2, :, :, :-1, :] == 6)
    assert xp.all(y[3, :, :, :, :-1] == 1)

    with pytest.raises(ValueError):
        A = ppo.FiniteForwardDifference((3, 3, 3, 3, 3))


def test_gradient_projection(xp: ModuleType, dev: str):
    np.random.seed(0)

    # g = xp.asarray([[[1.0, 0.0, 1.0]], [[0.0, 1.0, 1.0]]], device=dev)
    g = xp.asarray([[[1, 0, 1]], [[0, 1, 1]]], device=dev)

    op = ppo.GradientFieldProjectionOperator(g, eta=0.0)

    assert op.eta == 0.0
    assert op.xp == xp
    d = op.dev

    ngf = xp.asarray(
        [[[1.0, 0.0, 1 / math.sqrt(2)]], [[0.0, 1.0, 1 / math.sqrt(2)]]], device=dev
    )

    assert allclose(ngf, op.normalized_gradient_field)

    x = xp.asarray([[[1.0, 1.0, 1.0]], [[0.0, 0.0, 0.0]]], device=dev)

    px = op(x)
    px2 = (
        x
        - xp.sum(x * op.normalized_gradient_field, axis=0)
        * op.normalized_gradient_field
    )

    assert allclose(px, px2)

    # test call to norm
    op_norm = op.norm(xp, dev)

    with pytest.raises(ValueError):
        gc = xp.asarray([[[1j, 0, 1]], [[0, 1, 1]]], device=dev)
        A = ppo.GradientFieldProjectionOperator(gc)


def test_abstract_raises(xp: ModuleType, dev: str):
    """Covers raise NotImplementedError in abstract method bodies (lines 24, 30, 46, 51)."""

    class _Op(ppo.LinearOperator):
        @property
        def in_shape(self):
            return super().in_shape

        @property
        def out_shape(self):
            return super().out_shape

        def _apply(self, x):
            return super()._apply(x)

        def _adjoint(self, y):
            return super()._adjoint(y)

    op = _Op()
    with pytest.raises(NotImplementedError):
        _ = op.in_shape
    with pytest.raises(NotImplementedError):
        _ = op.out_shape
    with pytest.raises(NotImplementedError):
        op._apply(None)
    with pytest.raises(NotImplementedError):
        op._adjoint(None)


def test_adjointness_verbose(xp: ModuleType, dev: str):
    """Covers the verbose print in adjointness_test (line 153)."""
    A = ppo.MatrixOperator(xp.asarray([[1.0, 2.0], [-3.0, 2.0]], device=dev))
    assert A.adjointness_test(xp, dev, verbose=True)


def test_norm_verbose(xp: ModuleType, dev: str):
    """Covers the verbose print in norm (line 204)."""
    A = ppo.MatrixOperator(xp.asarray([[1.0, 2.0], [-3.0, 2.0]], device=dev))
    A.norm(xp, dev, verbose=True, num_iter=2)


def test_gaussian_array_api_strict(xp: ModuleType, dev: str):
    """Covers the array_api_strict branch in GaussianFilterOperator._apply (line 502)."""
    if importlib.util.find_spec("array_api_strict") is None:
        pytest.skip("array_api_strict not available")
    import array_api_strict as xp_strict

    op = ppo.GaussianFilterOperator((8, 8), sigma=1.0)
    x = xp_strict.asarray(nprnd.rand(8, 8))
    y = op._apply(x)
    assert y.shape == (8, 8)


def test_gaussian_unsupported_type(xp: ModuleType, dev: str):
    """Covers the raise TypeError fallback in GaussianFilterOperator._apply (line 543)."""
    import numpy as _np

    op = ppo.GaussianFilterOperator((4,), sigma=1.0)
    x = _np.array([1.0, 2.0, 3.0, 4.0])

    with (
        patch("array_api_compat.is_numpy_array", return_value=False),
        patch("array_api_compat.is_array_api_strict_namespace", return_value=False),
        patch("array_api_compat.is_cupy_array", return_value=False),
        patch("array_api_compat.is_torch_array", return_value=False),
    ):
        with pytest.raises(TypeError, match="Unsupported input type"):
            op._apply(x)
