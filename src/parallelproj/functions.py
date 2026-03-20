from abc import ABC, abstractmethod
from ._backend import Array
from .operators import LinearOperator

from array_api_compat import get_namespace


class C1Function(ABC):
    """Abstract base class for continuously differentiable (C1) scalar functions
    with an optional scalar scale factor :math:`\\beta`.

    The public interface (:meth:`__call__`, :meth:`gradient`,
    :meth:`call_and_gradient`) evaluates :math:`\\beta f(x)`.  Subclasses only
    implement the *unscaled* private methods :meth:`_call` and
    :meth:`_gradient`; :math:`\\beta` is applied automatically.

    Parameters
    ----------
    beta : float, optional
        Multiplicative scale factor :math:`\\beta` applied to the function
        value and all derivatives.  Defaults to ``1.0``.
    """

    def __init__(self, beta: float = 1.0):
        self._beta = beta

    # ------------------------------------------------------------------
    # beta property
    # ------------------------------------------------------------------

    @property
    def beta(self) -> float:
        """Multiplicative scale factor :math:`\\beta`."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value

    # ------------------------------------------------------------------
    # Abstract interface — subclasses implement the *unscaled* versions
    # ------------------------------------------------------------------

    @abstractmethod
    def _call(self, x: Array) -> float:
        """Unscaled function value at x (implemented by subclasses)."""
        ...

    @abstractmethod
    def _gradient(self, x: Array) -> Array:
        """Unscaled gradient at x (implemented by subclasses)."""
        ...

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        """Unscaled function value and gradient at x.

        Subclasses may override this to share intermediate computations.
        """
        return self._call(x), self._gradient(x)

    # ------------------------------------------------------------------
    # Public interface — applies self._beta
    # ------------------------------------------------------------------

    def __call__(self, x: Array) -> float:
        """Evaluate :math:`\\beta f(x)`.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the function.

        Returns
        -------
        float
            Scaled scalar function value.
        """
        v = self._call(x)
        return v if self._beta == 1.0 else self._beta * v

    def gradient(self, x: Array) -> Array:
        """Gradient of :math:`\\beta f(x)`.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the gradient.

        Returns
        -------
        Array
            Array of the same shape as x containing
            :math:`\\beta \\nabla f(x)`.
        """
        g = self._gradient(x)
        return g if self._beta == 1.0 else self._beta * g

    def call_and_gradient(self, x: Array) -> tuple[float, Array]:
        """Evaluate :math:`\\beta f(x)` and its gradient simultaneously.

        Parameters
        ----------
        x : Array
            Point at which to evaluate.

        Returns
        -------
        tuple[float, Array]
            Scaled function value and gradient.
        """
        v, g = self._call_and_gradient(x)
        return (v, g) if self._beta == 1.0 else (self._beta * v, self._beta * g)

    def __add__(self, other: "C1Function") -> "SumC1Function":
        """Return a :class:`SumC2Function` or :class:`SumC1Function` for ``self + other``.

        Parameters
        ----------
        other : C1Function
            Right-hand operand.

        Returns
        -------
        SumC2Function
            If both operands are :class:`C2Function` instances.
        SumC1Function
            Otherwise.
        """
        if isinstance(self, C2Function) and isinstance(other, C2Function):
            return SumC2Function([self, other])
        return SumC1Function([self, other])


class C2Function(C1Function):
    """Abstract base class for twice continuously differentiable (C2) scalar
    functions with an optional scalar scale factor :math:`\\beta`.

    Extends :class:`C1Function` with curvature information.  The public
    :meth:`hessian_diag_vec_prod` returns
    :math:`\\beta \\operatorname{diag}(H_f(x))\\, v`.
    Subclasses implement the unscaled :meth:`_hessian_diag_vec_prod`.
    """

    @abstractmethod
    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        """Unscaled diagonal Hessian-vector product (implemented by subclasses)."""
        ...

    def hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        """Scaled diagonal Hessian-vector product:
        :math:`\\beta \\operatorname{diag}(H_f(x))\\, v`.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the Hessian diagonal.
        v : Array
            Vector to multiply with the Hessian diagonal, same shape as x.

        Returns
        -------
        Array
            Array of the same shape as x containing
            :math:`\\beta \\operatorname{diag}(H_f(x)) \\odot v`
            (elementwise).
        """
        h = self._hessian_diag_vec_prod(x, v)
        return h if self._beta == 1.0 else self._beta * h


class NegPoissonLogL(C2Function):
    """Negative Poisson log-likelihood as a function of expected counts.

    Implements

    .. math::

        f(\\bar{y}) = \\sum_i \\bar{y}_i - y_i \\log \\bar{y}_i

    and its gradient w.r.t. the expected counts :math:`\\bar{y}`:

    .. math::

        \\nabla_{\\bar{y}} f = 1 - \\frac{y}{\\bar{y}}.

    This class operates directly on the predicted counts :math:`\\bar{y}`,
    independently of how they were computed. Use :class:`C2AffineObjective` to
    compose with a forward model :math:`\\bar{y}(x) = A x + s`.

    Parameters
    ----------
    data : Array
        Measured data :math:`y`.
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
    """

    def __init__(self, data: Array, beta: float = 1.0):
        super().__init__(beta)
        self._data = data

    def _call(self, pred: Array) -> float:
        xp = get_namespace(pred)
        return float(xp.sum(pred - self._data * xp.log(pred)))

    def _gradient(self, pred: Array) -> Array:
        return 1 - self._data / pred

    def _call_and_gradient(self, pred: Array) -> tuple[float, Array]:
        xp = get_namespace(pred)
        return float(xp.sum(pred - self._data * xp.log(pred))), 1 - self._data / pred

    def _hessian_diag_vec_prod(self, pred: Array, v: Array) -> Array:
        return self._data / (pred**2) * v


class HalfSquaredL2Deviation(C2Function):
    """Half squared L2 deviation from reference data.

    Implements

    .. math::

        f(x) = \\frac{1}{2} \\|x - d\\|_2^2
             = \\frac{1}{2} \\sum_i (x_i - d_i)^2

    with gradient

    .. math::

        \\nabla f(x) = x - d

    and diagonal Hessian-vector product

    .. math::

        \\operatorname{diag}(H_f(x)) \\odot v = v

    (the Hessian is the identity).  The :math:`\\tfrac{1}{2}` prefactor is
    chosen so that the gradient contains no factor of 2, keeping expressions
    clean when :math:`\\beta = 1`.

    Parameters
    ----------
    data : Array
        Reference array :math:`d`.
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
    """

    def __init__(self, data: Array, beta: float = 1.0):
        super().__init__(beta)
        self._data = data

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        diff = x - self._data
        return float(0.5 * xp.sum(diff**2))

    def _gradient(self, x: Array) -> Array:
        return x - self._data

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        xp = get_namespace(x)
        diff = x - self._data
        return float(0.5 * xp.sum(diff**2)), diff

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        return v


class SumC1Function(C1Function):
    """Sum of an arbitrary number of :class:`C1Function` objects.

    Represents

    .. math::

        h(x) = \\sum_{k} f_k(x)

    where each :math:`f_k` may itself carry its own :math:`\\beta_k`.
    The gradients add accordingly:

    .. math::

        \\nabla h(x) = \\sum_{k} \\nabla f_k(x).

    Instances are most conveniently created via the ``+`` operator on any
    two :class:`C1Function` objects, but can also be constructed directly
    to sum more than two terms at once.

    Parameters
    ----------
    functions : list[C1Function]
        The functions :math:`f_k` to sum.  Must contain at least one element.
    """

    def __init__(self, functions: list[C1Function]):
        super().__init__()
        self._functions = functions

    def _call(self, x: Array) -> float:
        return sum(f(x) for f in self._functions)

    def _gradient(self, x: Array) -> Array:
        result = self._functions[0].gradient(x)
        for f in self._functions[1:]:
            result = result + f.gradient(x)
        return result

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        val, grad = self._functions[0].call_and_gradient(x)
        for f in self._functions[1:]:
            v, g = f.call_and_gradient(x)
            val += v
            grad = grad + g
        return val, grad


class SumC2Function(C2Function, SumC1Function):
    """Sum of an arbitrary number of :class:`C2Function` objects.

    Extends :class:`SumC1Function` with second-order information.  The
    diagonal Hessian-vector product is:

    .. math::

        \\operatorname{diag}\\!\\left(H_h(x)\\right) \\odot v
        = \\sum_{k} \\operatorname{diag}\\!\\left(H_{f_k}(x)\\right) \\odot v.

    Parameters
    ----------
    functions : list[C2Function]
        The functions :math:`f_k` to sum.  Must contain at least one element.
    """

    def __init__(self, functions: list[C2Function]):
        self._functions: list[C2Function]
        SumC1Function.__init__(self, functions)

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        result = self._functions[0].hessian_diag_vec_prod(x, v)
        for f in self._functions[1:]:
            result = result + f.hessian_diag_vec_prod(x, v)
        return result


class C1AffineObjective(C1Function):
    """Composes a prediction-space :class:`C1Function` with an affine forward model.

    Turns :math:`g(\\bar{y})` into
    :math:`f(x) = \\beta \\cdot g(A x + s)` using the chain rule:

    .. math::

        \\nabla_x f(x) = \\beta \\cdot A^H \\nabla_{\\bar{y}} g(A x + s).

    Parameters
    ----------
    loss : C1Function
        A :class:`C1Function` operating on the prediction space.
    op : LinearOperator
        The linear part of the forward model :math:`A`.
    s : Array
        Additive contamination term :math:`s` (e.g. scatter/randoms).
    beta : float, optional
        Multiplicative scale factor :math:`\\beta` applied on top of any
        :math:`\\beta` already carried by ``loss``.  Defaults to ``1.0``.
    """

    def __init__(
        self, loss: C1Function, op: LinearOperator, s: Array, beta: float = 1.0
    ):
        super().__init__(beta)
        self._loss, self._op, self._s = loss, op, s

    def _call(self, x: Array) -> float:
        return self._loss(self._op(x) + self._s)

    def _gradient(self, x: Array) -> Array:
        pred = self._op(x) + self._s
        return self._op.adjoint(self._loss.gradient(pred))

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        pred = self._op(x) + self._s
        value, grad_pred = self._loss.call_and_gradient(pred)
        return value, self._op.adjoint(grad_pred)


class C2AffineObjective(C2Function, C1AffineObjective):
    """Composes a prediction-space :class:`C2Function` with an affine forward model.

    Extends :class:`C1AffineObjective` with second-order information. For
    :math:`f(x) = \\beta \\cdot g(A x + s)` the Hessian-vector product is:

    .. math::

        H_f(x)\\, v = \\beta \\cdot A^H \\bigl(\\operatorname{diag}(H_g(\\bar{y})) \\odot (A v)\\bigr)

    where :math:`\\bar{y} = A x + s` and :math:`\\odot` denotes elementwise
    multiplication.

    Parameters
    ----------
    loss : C2Function
        A :class:`C2Function` operating on the prediction space.
    op : LinearOperator
        The linear part of the forward model :math:`A`.
    s : Array
        Additive contamination term :math:`s` (e.g. scatter/randoms).
    beta : float, optional
        Multiplicative scale factor :math:`\\beta` applied on top of any
        :math:`\\beta` already carried by ``loss``.  Defaults to ``1.0``.

    Examples
    --------
    >>> import numpy as np
    >>> from parallelproj.operators import MatrixOperator
    >>> from parallelproj.functions import NegPoissonLogL, C2AffineObjective
    >>> # 4-element image space, 6-element sinogram space
    >>> A = np.random.rand(6, 4)
    >>> op = MatrixOperator(A)
    >>> s = 0.1 * np.ones(6)          # scatter/randoms contamination
    >>> data = np.ones(6)             # measured counts
    >>> x = np.ones(4)                # image estimate
    >>> v = np.ones(4)                # direction vector
    >>>
    >>> loss = NegPoissonLogL(data)
    >>> f = C2AffineObjective(loss, op, s, beta=0.5)
    >>>
    >>> grad = f.gradient(x)                          # shape (4,), scaled by beta=0.5
    >>> hv   = f.hessian_diag_vec_prod(x, v)          # shape (4,), scaled by beta=0.5
    """

    def __init__(
        self, loss: C2Function, op: LinearOperator, s: Array, beta: float = 1.0
    ):
        self._loss: C2Function
        C1AffineObjective.__init__(self, loss, op, s, beta)

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        pred = self._op(x) + self._s
        return self._op.adjoint(self._loss.hessian_diag_vec_prod(pred, self._op(v)))
