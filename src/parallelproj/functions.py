from abc import ABC, abstractmethod
from ._backend import Array
from .operators import LinearOperator

from array_api_compat import get_namespace


class C1Function(ABC):
    """Abstract base class for continuously differentiable (C¹) scalar functions.

    Subclasses must implement :meth:`__call__` and :meth:`gradient`.
    The default :meth:`call_and_gradient` calls both separately; subclasses
    may override it to share intermediate computations for efficiency.
    """

    @abstractmethod
    def __call__(self, x: Array) -> float:
        """Evaluate the function at x.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the function.

        Returns
        -------
        float
            Scalar function value.
        """
        ...

    @abstractmethod
    def gradient(self, x: Array) -> Array:
        """Gradient of the function evaluated at x.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the gradient.

        Returns
        -------
        Array
            Array of the same shape as x containing the gradient.
        """
        ...

    def call_and_gradient(self, x: Array) -> tuple[float, Array]:
        """Evaluate the function and its gradient at x.

        May be overridden by subclasses to share intermediate computations.

        Parameters
        ----------
        x : Array
            Point at which to evaluate.

        Returns
        -------
        tuple[float, Array]
            Function value and gradient.
        """
        return self.__call__(x), self.gradient(x)


class C2Function(C1Function):
    """Abstract base class for twice continuously differentiable (C2) scalar functions.

    Extends :class:`C1Function` with access to curvature information via the
    diagonal of the Hessian, useful for diagonal pre-conditioning in
    second-order or quasi-Newton optimization methods.
    """

    @abstractmethod
    def hessian_diag(self, x: Array) -> Array:
        """Diagonal of the Hessian of the function evaluated at x.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the Hessian diagonal.

        Returns
        -------
        Array
            Array of the same shape as x containing the diagonal entries
            of the Hessian matrix.
        """
        ...


class NegPoissonLogL(C1Function):
    """Negative Poisson log-likelihood for a linear forward model.

    Implements the objective function

    .. math::

        f(x) = \\sum_i \\bar{y}_i(x) - y_i \\log \\bar{y}_i(x)

    with the linear forward model :math:`\\bar{y}(x) = A x + s`, where
    :math:`A` is a :class:`.LinearOperator`, :math:`y` is the measured data,
    and :math:`s` is an additive contamination term (e.g. scatter/randoms).

    The gradient is

    .. math::

        \\nabla f(x) = A^H \\left(1 - \\frac{y}{\\bar{y}(x)}\\right).

    Parameters
    ----------
    data : Array
        Measured data :math:`y`.
    op : LinearOperator
        Forward operator :math:`A`.
    s : Array
        Additive contamination sinogram :math:`s`.
    """

    def __init__(self, data: Array, op: LinearOperator, s: Array):
        self._data, self._op, self._s = data, op, s

    def __call__(self, x: Array) -> float:
        pred = self._op(x) + self._s
        xp = get_namespace(pred)
        return float(xp.sum(pred - self._data * xp.log(pred)))

    def gradient(self, x: Array) -> Array:
        pred = self._op(x) + self._s
        return self._op.adjoint(1 - self._data / pred)

    def call_and_gradient(self, x: Array) -> tuple[float, Array]:
        pred = self._op(x) + self._s
        xp = get_namespace(pred)
        value = float(xp.sum(pred - self._data * xp.log(pred)))
        grad = self._op.adjoint(1 - self._data / pred)
        return value, grad
