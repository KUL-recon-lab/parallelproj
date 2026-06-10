"""Functions and proximal operators.

Provides a hierarchy of differentiable and non-differentiable scalar functions
-- including the negative Poisson log-likelihood, quadratic penalties, and the
mixed L2-L1 norm -- together with their gradients, Hessian-diagonal-vector
products, and closed-form proximal operators where available.  All classes are
array-API compatible and work with NumPy, CuPy, and PyTorch.
"""

import math
from abc import ABC, abstractmethod
from collections.abc import Sequence

from array_api_compat import get_namespace

from ._backend import Array
from .operators import LinearOperator


class FunctionWithConjProx(ABC):
    """Abstract base class for functions with a closed-form proximal operator
    of their convex conjugate.

    This class is a standalone root -- it does **not** require the function to
    be differentiable.  Non-smooth functions (e.g. total variation, indicator
    functions) that admit a closed-form :math:`\\text{prox}_{\\sigma f^*}`
    should inherit directly from this class.

    Differentiable functions that additionally have a closed-form dual prox
    should use :class:`C1FunctionWithConjProx` or :class:`C2FunctionWithConjProx`
    instead.

    The public :meth:`prox_convex_conj` handles the :math:`\\beta` scaling
    automatically.  Subclasses implement only the *unscaled* private methods
    :meth:`_call` and :meth:`_prox_convex_conj`.

    A default :meth:`prox` is provided via Moreau's identity for convenience;
    subclasses may override it with a more efficient closed-form if available.

    Parameters
    ----------
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
    """

    def __init__(self, beta: float = 1.0):
        self._beta = beta

    @property
    def beta(self) -> float:
        """Multiplicative scale factor :math:`\\beta` (should be > 0)."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value

    @abstractmethod
    def _call(self, x: Array) -> float:
        """Unscaled function value f(x) (implemented by subclasses)."""

    def __call__(self, x: Array) -> float:
        """Evaluate :math:`\\beta f(x)`.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the function.

        Returns
        -------
        float
            Scaled scalar function value :math:`\\beta f(x)`.
        """
        v = self._call(x)
        return v if self._beta == 1.0 else self._beta * v

    @abstractmethod
    def _prox_convex_conj(self, y: Array, sigma: float | Array) -> Array:
        """Unscaled proximal operator of the convex conjugate.

        Computes :math:`\\text{prox}_{\\sigma f^*}(y)` for the *unscaled*
        function :math:`f` (i.e. with :math:`\\beta = 1`).

        Parameters
        ----------
        y : Array
            Input array.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\text{prox}_{\\sigma f^*}(y)`.
        """

    def prox_convex_conj(self, y: Array, sigma: float | Array) -> Array:
        """Proximal operator of the convex conjugate of :math:`\\beta f`.

        Uses the identity

        .. math::

            \\text{prox}_{\\sigma (\\beta f)^*}(y)
            = \\beta \\, \\text{prox}_{(\\sigma / \\beta)\\, f^*}(y / \\beta)

        to reduce to the unscaled :meth:`_prox_convex_conj`.

        Parameters
        ----------
        y : Array
            Input array.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\text{prox}_{\\sigma (\\beta f)^*}(y)`.
        """
        if self._beta == 1.0:
            return self._prox_convex_conj(y, sigma)
        return self._beta * self._prox_convex_conj(y / self._beta, sigma / self._beta)

    def prox(self, x: Array, sigma: float | Array) -> Array:
        """Proximal operator of :math:`\\beta f` via Moreau's identity.

        .. math::

            \\text{prox}_{\\sigma (\\beta f)}(x)
            = x - \\sigma \\, \\text{prox}_{(1/\\sigma)(\\beta f)^*}(x / \\sigma)

        where the inner prox uses step-size :math:`1/\\sigma`.

        Subclasses with a cheaper closed-form direct prox may override this.

        Parameters
        ----------
        x : Array
            Input array.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\text{prox}_{\\sigma (\\beta f)}(x)`.
        """
        return x - sigma * self.prox_convex_conj(x / sigma, 1.0 / sigma)


class FunctionWithProx(ABC):
    """Abstract base class for functions with a closed-form proximal operator.

    This class is a standalone root -- it does **not** require the function to
    be differentiable.  Functions (e.g. indicator functions, L1 norm) that
    admit a closed-form :math:`\\text{prox}_{\\sigma f}` should inherit
    directly from this class.

    The public :meth:`prox` handles the :math:`\\beta` scaling automatically.
    Subclasses implement only the *unscaled* private methods :meth:`_call` and
    :meth:`_prox`.

    A default :meth:`prox_convex_conj` is provided via Moreau's identity for
    convenience; subclasses may override it with a more efficient closed-form
    if available.

    Parameters
    ----------
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
    """

    def __init__(self, beta: float = 1.0):
        self._beta = beta

    @property
    def beta(self) -> float:
        """Multiplicative scale factor :math:`\\beta` (should be > 0)."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value

    @abstractmethod
    def _call(self, x: Array) -> float:
        """Unscaled function value f(x) (implemented by subclasses)."""

    def __call__(self, x: Array) -> float:
        """Evaluate :math:`\\beta f(x)`.

        Parameters
        ----------
        x : Array
            Point at which to evaluate the function.

        Returns
        -------
        float
            Scaled scalar function value :math:`\\beta f(x)`.
        """
        v = self._call(x)
        return v if self._beta == 1.0 else self._beta * v

    @abstractmethod
    def _prox(self, x: Array, sigma: float | Array) -> Array:
        """Unscaled proximal operator.

        Computes :math:`\\text{prox}_{\\sigma f}(x)` for the *unscaled*
        function :math:`f` (i.e. with :math:`\\beta = 1`).

        Parameters
        ----------
        x : Array
            Input array.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\text{prox}_{\\sigma f}(x)`.
        """

    def prox(self, x: Array, sigma: float | Array) -> Array:
        """Proximal operator of :math:`\\beta f`.

        .. math::

            \\text{prox}_{\\sigma (\\beta f)}(x)
            = \\text{prox}_{(\\sigma \\beta) f}(x)

        so the effective step size passed to the unscaled :meth:`_prox` is
        :math:`\\sigma \\beta`.

        Parameters
        ----------
        x : Array
            Input array.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\text{prox}_{\\sigma (\\beta f)}(x)`.
        """
        if self._beta == 1.0:
            return self._prox(x, sigma)
        return self._prox(x, sigma * self._beta)

    def prox_convex_conj(self, y: Array, sigma: float | Array) -> Array:
        """Proximal operator of the convex conjugate of :math:`\\beta f`
        via Moreau's identity.

        .. math::

            \\text{prox}_{\\sigma (\\beta f)^*}(y)
            = y - \\sigma \\, \\text{prox}_{(1/\\sigma)(\\beta f)}(y / \\sigma)

        where the inner prox uses step-size :math:`1/\\sigma`.

        Subclasses with a cheaper closed-form dual prox may override this.

        Parameters
        ----------
        y : Array
            Input array.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\text{prox}_{\\sigma (\\beta f)^*}(y)`.
        """
        return y - sigma * self.prox(y / sigma, 1.0 / sigma)


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
        """Multiplicative scale factor :math:`\\beta` (should be > 0)."""
        return self._beta

    @beta.setter
    def beta(self, value: float) -> None:
        self._beta = value

    # ------------------------------------------------------------------
    # Abstract interface -- subclasses implement the *unscaled* versions
    # ------------------------------------------------------------------

    @abstractmethod
    def _call(self, x: Array) -> float:
        """Unscaled function value at x (implemented by subclasses)."""

    @abstractmethod
    def _gradient(self, x: Array) -> Array:
        """Unscaled gradient at x (implemented by subclasses)."""

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        """Unscaled function value and gradient at x.

        Subclasses may override this to share intermediate computations.
        """
        return self._call(x), self._gradient(x)

    # ------------------------------------------------------------------
    # Public interface -- applies self._beta
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
        """Return the sum ``self + other`` as a new function object.

        Parameters
        ----------
        other : C1Function
            Right-hand operand.

        Returns
        -------
        SumC1Function
            A :class:`SumC2Function` (subclass of :class:`SumC1Function`) when
            both operands are :class:`C2Function` instances; a plain
            :class:`SumC1Function` otherwise.
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


class C1FunctionWithConjProx(C1Function, FunctionWithConjProx):
    """Abstract base class for C1 functions that also admit a closed-form
    proximal operator of their convex conjugate.

    Use this as the base class when your function is differentiable **and**
    has a cheap closed-form :math:`\\operatorname{prox}_{\\sigma f^*}`.
    Subclasses must implement :meth:`_call`, :meth:`_gradient`, and
    :meth:`_prox_convex_conj`.  The :math:`\\beta` scaling and the public
    :meth:`prox_convex_conj` wrapper are inherited and require no override.

    See :class:`NegPoissonLogL` and :class:`HalfSquaredL2Deviation` for
    concrete examples.
    """


class C2FunctionWithConjProx(C2Function, C1FunctionWithConjProx):
    """Abstract base class for C2 functions that also admit a closed-form
    proximal operator of their convex conjugate.

    Combines :class:`C2Function` (Hessian diagonal) with
    :class:`C1FunctionWithConjProx` (gradient, call, dual prox).
    Subclasses must implement :meth:`_call`, :meth:`_gradient`,
    :meth:`_hessian_diag_vec_prod`, and :meth:`_prox_convex_conj`.

    MRO: ``C2FunctionWithConjProx -> C2Function -> C1FunctionWithConjProx -> C1Function -> FunctionWithConjProx -> ABC``
    """


class NegPoissonLogL(C2FunctionWithConjProx):
    """Negative Poisson log-likelihood as a function of expected counts.

    Implements

    .. math::

        f(\\bar{y}) = \\sum_i \\bar{y}_i - y_i \\log \\bar{y}_i

    and its gradient w.r.t. the expected counts :math:`\\bar{y}`:

    .. math::

        \\nabla_{\\bar{y}} f = 1 - \\frac{y}{\\bar{y}}.

    This class operates directly on the predicted counts :math:`\\bar{y}`
    (passed as the argument ``x`` to the callable interface),
    independently of how they were computed. Use :class:`C2AffineObjective` to
    compose with a forward model :math:`\\bar{y}(x) = A x + s`.

    **Safe mode** (``safe=True``) makes the evaluation robust for bins where
    both :math:`y_i = 0` and :math:`\\bar{y}_i = 0`, which would otherwise
    produce ``nan`` from :math:`0 \\cdot \\log 0`, :math:`0/0`, or
    :math:`0/0^2`.  Such bins arise either as *virtual bins* (no image voxel
    contributes and the contamination is zero, so :math:`\\bar{y}_i = 0` by
    construction) or as active bins that observed zero counts due to Poisson
    noise while :math:`\\bar{y}_i \\to 0` during optimization.  In both cases
    the per-bin function is exactly linear, :math:`f_i(\\bar{y}_i) =
    \\bar{y}_i`, so the exact values

    ============================================  ===============
    Expression                                    Value
    ============================================  ===============
    :math:`\\bar{y}_i - y_i \\log \\bar{y}_i`        :math:`\\bar{y}_i`
    :math:`1 - y_i / \\bar{y}_i`                   :math:`1`
    :math:`y_i / \\bar{y}_i^2`                     :math:`0`
    ============================================  ===============

    are substituted on **all** bins with :math:`y_i = 0`.  This is exact (not
    a limit approximation) for any :math:`\\bar{y}_i \\geq 0`, so no
    user-supplied mask is needed -- the condition :math:`y_i = 0` is derived
    from the data at init time.

    .. note::

        Bins with :math:`y_i > 0` and :math:`\\bar{y}_i = 0` deliberately
        yield :math:`f = +\\infty` (and gradient :math:`-\\infty`) in both
        modes: observing counts where the model predicts strictly zero counts
        is a genuine model violation that should surface, not be masked.

    Parameters
    ----------
    data : Array
        Measured data :math:`y` (non-negative).
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
    safe : bool, optional
        If ``True``, handle bins with :math:`y_i = 0` and
        :math:`\\bar{y}_i = 0` exactly as described above, at the cost of
        one extra ``where`` per evaluation.  Defaults to ``False``.
    """

    def __init__(self, data: Array, beta: float = 1.0, safe: bool = False):
        super().__init__(beta)
        self._data = data
        self._safe = safe
        self._data_is_zero = (data == 0) if safe else None

    @property
    def safe(self) -> bool:
        """Whether safe handling of bins with :math:`y_i = \\bar{y}_i = 0` is enabled."""
        return self._safe

    def _safe_x(self, x: Array) -> Array:
        """Return ``x`` with entries replaced by 1 where ``data == 0`` (safe mode only).

        On those bins the numerator :math:`y_i = 0` already nullifies the
        log / division terms, so substituting 1 in the denominator (or log
        argument) yields the exact values while leaving all other bins
        untouched.
        """
        if not self._safe:
            return x
        xp = get_namespace(x)
        return xp.where(self._data_is_zero, xp.ones_like(x), x)

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        return float(xp.sum(x - self._data * xp.log(self._safe_x(x))))

    def _gradient(self, x: Array) -> Array:
        return 1 - self._data / self._safe_x(x)

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        return self._data / (self._safe_x(x) ** 2) * v

    def _prox_convex_conj(self, y: Array, sigma: float | Array) -> Array:
        """Proximal operator of the convex conjugate of the negative Poisson log-likelihood.

        .. math::

            \\left(\\operatorname{prox}_{\\sigma f^*}(y)\\right)_i
            = \\frac{1}{2}\\left(y_i + 1 - \\sqrt{(y_i - 1)^2 + 4 \\sigma d_i}\\right)

        No special handling is needed in safe mode: the formula contains no
        log or division.  For bins with :math:`d_i = 0` the loss is linear,
        :math:`f_i(\\bar{y}_i) = \\bar{y}_i`, whose conjugate is the indicator
        of :math:`(-\\infty, 1]` with prox :math:`\\min(y_i, 1)` -- which the
        formula above already yields exactly when :math:`d_i = 0`.

        Parameters
        ----------
        y : Array
            Input array (dual variable), same shape as the data.
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\operatorname{prox}_{\\sigma f^*}(y)`.
        """
        xp = get_namespace(y)
        return 0.5 * (y + 1 - xp.sqrt((y - 1) ** 2 + 4 * sigma * self._data))


class HalfSquaredL2Deviation(C2FunctionWithConjProx):
    """Half squared L2 deviation from reference data, with optional weights.

    Implements

    .. math::

        f(x) = \\frac{1}{2} \\sum_i w_i (x_i - d_i)^2

    where :math:`w_i = 1` when no weights are supplied (reducing to the
    standard :math:`\\tfrac{1}{2}\\|x - d\\|_2^2`).

    Gradient:

    .. math::

        \\nabla f(x) = w \\odot (x - d)

    Diagonal Hessian-vector product:

    .. math::

        \\operatorname{diag}(H_f(x)) \\odot v = w \\odot v

    The :math:`\\tfrac{1}{2}` prefactor is chosen so that the gradient
    contains no factor of 2, keeping expressions clean when :math:`\\beta = 1`.

    Parameters
    ----------
    data : Array or None, optional
        Reference array :math:`d`.  ``None`` (default) is equivalent to
        :math:`d = 0` but avoids the subtraction entirely.
    weights : Array or None, optional
        Non-negative weight array :math:`w` of the same shape as ``data``
        (or ``x`` when ``data`` is ``None``).  ``None`` (default) is
        equivalent to unit weights but avoids the multiplication entirely.
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
    """

    def __init__(
        self,
        data: Array | None = None,
        weights: Array | None = None,
        beta: float = 1.0,
    ):
        super().__init__(beta)
        self._data = data
        self._weights = weights

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        diff = x if self._data is None else x - self._data
        if self._weights is None:
            return float(0.5 * xp.sum(diff**2))
        return float(0.5 * xp.sum(self._weights * diff**2))

    def _gradient(self, x: Array) -> Array:
        xp = get_namespace(x)
        diff = xp.asarray(x, copy=True) if self._data is None else x - self._data
        if self._weights is None:
            return diff
        return self._weights * diff

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        xp = get_namespace(x)
        diff = xp.asarray(x, copy=True) if self._data is None else x - self._data
        if self._weights is None:
            return float(0.5 * xp.sum(diff**2)), diff
        wdiff = self._weights * diff
        return float(0.5 * xp.sum(self._weights * diff**2)), wdiff

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        if self._weights is None:
            return v
        return self._weights * v

    def _prox_convex_conj(self, y: Array, sigma: float | Array) -> Array:
        """Proximal operator of the convex conjugate of the half squared L2 deviation.

        For :math:`f(x) = \\frac{1}{2} \\sum_i w_i (x_i - d_i)^2` the
        convex conjugate is

        .. math::

            f^*(p) = \\langle p, d \\rangle + \\frac{1}{2} \\sum_i \\frac{p_i^2}{w_i}

        and its proximal operator is

        .. math::

            \\left(\\operatorname{prox}_{\\sigma f^*}(y)\\right)_i
            = \\frac{w_i (y_i - \\sigma d_i)}{w_i + \\sigma}

        which simplifies to :math:`(y_i - \\sigma d_i)/(1 + \\sigma)` for unit
        weights and to :math:`y_i / (1 + \\sigma)` when :math:`d = 0`.

        Parameters
        ----------
        y : Array
            Input array (dual variable).
        sigma : float or Array
            Step-size parameter :math:`\\sigma > 0`.

        Returns
        -------
        Array
            :math:`\\operatorname{prox}_{\\sigma f^*}(y)`.
        """
        numerator = y if self._data is None else y - sigma * self._data
        if self._weights is None:
            return numerator / (1 + sigma)
        return self._weights * numerator / (self._weights + sigma)


class LogCosh(C2Function):
    """Sum of scaled log-cosh values, a smooth approximation to the L1 norm.

    Implements

    .. math::

        f(x) = \\delta \\sum_i \\log\\!\\left(\\cosh\\!\\left(\\frac{x_i}{\\delta}\\right)\\right)

    where :math:`\\delta > 0` is a transition scale parameter (default 1).
    The function satisfies :math:`f(0) = 0` and has two limiting regimes:

    * **Quadratic** for :math:`|x_i| \\ll \\delta`:
      :math:`\\delta\\log(\\cosh(u)) \\approx u^2/2`, so
      :math:`f(x) \\approx \\tfrac{1}{2\\delta}\\sum_i x_i^2`.
    * **Linear** for :math:`|x_i| \\gg \\delta`:
      :math:`f(x) \\approx \\sum_i |x_i| - n\\,\\delta\\log 2 \\approx \\sum_i |x_i|`.

    The :math:`\\delta` prefactor ensures the asymptotic slope equals 1
    regardless of :math:`\\delta`, so the transition scale and the gradient
    magnitude at saturation are decoupled.

    Gradient:

    .. math::

        \\nabla f(x)_i = \\tanh\\!\\left(\\frac{x_i}{\\delta}\\right)

    Diagonal Hessian-vector product:

    .. math::

        \\operatorname{diag}(H_f(x))_i \\cdot v_i
        = \\frac{1}{\\delta}\\,\\operatorname{sech}^2\\!\\left(\\frac{x_i}{\\delta}\\right) v_i
        = \\frac{1 - \\tanh^2(x_i/\\delta)}{\\delta}\\, v_i

    The function value is computed via the numerically stable identity

    .. math::

        \\delta\\log(\\cosh(z)) = \\delta\\bigl(|z| + \\log(1 + e^{-2|z|}) - \\log 2\\bigr),
        \\quad z = x/\\delta

    which avoids the overflow that :math:`\\cosh(z) = (e^z + e^{-z})/2`
    would cause for large :math:`|z|`.

    Parameters
    ----------
    delta : float or None, optional
        Transition scale :math:`\\delta > 0`.  ``None`` (default) is
        equivalent to :math:`\\delta = 1` but skips the division entirely.
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.

    Notes
    -----
    The function value computation uses ``math.prod(x.shape)`` to subtract
    the normalisation constant :math:`n \\log 2`.  This requires that
    ``x.shape`` is a concrete tuple of integers at call time, which holds
    for all supported backends (NumPy, CuPy, PyTorch) but would fail for
    frameworks with symbolic or lazy shapes.
    """

    def __init__(self, delta: float | None = None, beta: float = 1.0):
        self._delta = delta
        self._log2 = math.log(2)
        super().__init__(beta)

    @property
    def delta(self) -> float | None:
        """Transition scale :math:`\\delta`."""
        return self._delta

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        z = x if self._delta is None else x / self._delta
        az = xp.abs(z)
        raw = (
            float(xp.sum(az + xp.log(1 + xp.exp(-2 * az))))
            - math.prod(x.shape) * self._log2
        )
        return raw if self._delta is None else self._delta * raw

    def _gradient(self, x: Array) -> Array:
        xp = get_namespace(x)
        z = x if self._delta is None else x / self._delta
        return xp.tanh(z)

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        xp = get_namespace(x)
        z = x if self._delta is None else x / self._delta
        az = xp.abs(z)
        raw = (
            float(xp.sum(az + xp.log(1 + xp.exp(-2 * az))))
            - math.prod(x.shape) * self._log2
        )
        return (raw if self._delta is None else self._delta * raw), xp.tanh(z)

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        xp = get_namespace(x)
        z = x if self._delta is None else x / self._delta
        t = xp.tanh(z)
        h = 1 - t**2
        return h * v if self._delta is None else h * v / self._delta


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

    .. note::

        This class inherits a ``beta`` attribute (default ``1.0``) that acts
        as an **outer** scale factor applied on top of the individual
        ``beta_k`` of each component.  Setting ``h.beta = c`` evaluates the
        sum as :math:`c \\sum_k \\beta_k f_k(x)`.  In most cases you should
        leave ``h.beta = 1.0`` and control scaling through the components.

    Parameters
    ----------
    functions : Sequence[C1Function]
        The functions :math:`f_k` to sum.  Must contain at least one element;
        an empty sequence will raise an ``IndexError`` when evaluated.
    """

    def __init__(self, functions: Sequence[C1Function]):
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
    functions : Sequence[C2Function]
        The functions :math:`f_k` to sum.  Must contain at least one element.
    """

    def __init__(self, functions: Sequence[C2Function]):
        self._functions: Sequence[C2Function]
        super().__init__(functions)

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        result = self._functions[0].hessian_diag_vec_prod(x, v)
        for f in self._functions[1:]:
            result = result + f.hessian_diag_vec_prod(x, v)
        return result


class NegPoissonLogLListmode(C2Function):
    """Negative Poisson log-likelihood for listmode (event-by-event) data.

    Implements the listmode equivalent of :class:`NegPoissonLogL` with an
    affine forward model :math:`\\bar{y}_e = (A_{\\text{LM}}\\, x)_e + s_e`.
    The function value is

    .. math::

        f(x) = \\langle \\text{sens},\\, x \\rangle + c_{\\text{sino}}
                - \\sum_{e=1}^{N_{\\text{ev}}} \\log\\bigl((A_{\\text{LM}}\\,x)_e + s_e\\bigr)

    where

    - :math:`\\text{sens} = A_{\\text{full}}^T \\mathbf{1}` is the sensitivity
      image (backprojection of all-ones from the full sinogram),
    - :math:`c_{\\text{sino}} = \\sum_i s_i^{\\text{sino}}` is the scalar sum
      of the contamination over all sinogram bins, and
    - the sum runs over all :math:`N_{\\text{ev}}` detected events.

    This is mathematically equivalent to the sinogram :class:`NegPoissonLogL`
    because :math:`\\sum_e \\log \\bar{y}_{j_e} = \\sum_i y_i \\log \\bar{y}_i`
    (each event from bin :math:`i` contributes :math:`\\log \\bar{y}_i` and
    there are :math:`y_i` such events).

    The gradient with respect to the image :math:`x` is

    .. math::

        \\nabla_x f(x) = \\text{sens}
            - A_{\\text{LM}}^T \\!\\left(\\frac{1}{A_{\\text{LM}}\\,x + s_{\\text{LM}}}\\right)

    and the diagonal Hessian-vector product is

    .. math::

        \\operatorname{diag}(H_f(x)) \\odot v
            = A_{\\text{LM}}^T \\!\\left(
                \\frac{A_{\\text{LM}}\\, v}{(A_{\\text{LM}}\\,x + s_{\\text{LM}})^2}
              \\right).

    .. note::

        Unlike :class:`NegPoissonLogL`, the forward model
        :math:`A_{\\text{LM}}` is built into this class rather than being
        supplied externally.  Wrapping it with :class:`C2AffineObjective`
        would double-compose the forward model and is therefore not supported.

    .. note::

        Every call to :meth:`__call__`, :meth:`gradient`, or
        :meth:`hessian_diag_vec_prod` requires that all per-event predicted
        counts :math:`(A_{\\text{LM}}\\,x)_e + s_e` are **strictly positive**.
        A :exc:`ValueError` is raised otherwise, so ensure that
        ``contamination_list`` is positive whenever the image :math:`x` may
        have zero-valued voxels.

    Parameters
    ----------
    lm_op : LinearOperator
        Listmode forward projector :math:`A_{\\text{LM}}` mapping an image
        (shape: ``n_voxels``) to per-event predicted counts
        (shape: ``n_events``).
    sensitivity_image : Array
        Sensitivity image :math:`A_{\\text{full}}^T \\mathbf{1}` (same shape
        as the image :math:`x`).
    contamination_list : Array
        Per-event additive contamination :math:`s_{\\text{LM}}` (same shape
        as the listmode output, i.e. ``n_events``).
    contamination_sinogram_sum : float
        Scalar sum of the contamination over all sinogram bins,
        :math:`c_{\\text{sino}} = \\sum_i s_i^{\\text{sino}}`.
    """

    def __init__(
        self,
        lm_op: LinearOperator,
        sensitivity_image: Array,
        contamination_list: Array,
        contamination_sinogram_sum: float,
    ):
        super().__init__()
        self._lm_op = lm_op
        self._sensitivity_image = sensitivity_image
        self._contamination_sinogram_sum = contamination_sinogram_sum
        self._contamination_list = contamination_list

    def _pred(self, x: Array) -> Array:
        """Compute per-event predicted counts :math:`A_{\\text{LM}} x + s_{\\text{LM}}`."""
        pred = self._lm_op(x) + self._contamination_list
        xp = get_namespace(pred)
        if float(xp.min(pred)) <= 0:
            raise ValueError("Per-event predicted counts must be strictly positive. ")
        return pred

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        pred = self._pred(x)
        norm = (
            float(xp.sum(self._sensitivity_image * x))
            + self._contamination_sinogram_sum
        )
        return norm - float(xp.sum(xp.log(pred)))

    def _gradient(self, x: Array) -> Array:
        return self._sensitivity_image - self._lm_op.adjoint(1.0 / self._pred(x))

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        pred = self._pred(x)
        xp = get_namespace(x)
        norm = (
            float(xp.sum(self._sensitivity_image * x))
            + self._contamination_sinogram_sum
        )
        value = norm - float(xp.sum(xp.log(pred)))
        grad = self._sensitivity_image - self._lm_op.adjoint(1.0 / pred)
        return value, grad

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        pred = self._pred(x)
        return self._lm_op.adjoint(self._lm_op(v) / pred**2)


class C1AffineObjective(C1Function):
    """Composes a prediction-space :class:`C1Function` with an affine forward model.

    Turns :math:`g(\\bar{y})` into :math:`f(x) = g(A x + s)` using the
    chain rule:

    .. math::

        \\nabla_x f(x) = A^H \\nabla_{\\bar{y}} g(A x + s).

    Any scaling is carried exclusively by the ``beta`` attribute of ``loss``.
    When :math:`s` is ``None`` the pure linear model :math:`\\bar{y} = A x`
    is used, avoiding the addition entirely.

    Parameters
    ----------
    loss : C1Function
        A :class:`C1Function` operating on the prediction space.
    op : LinearOperator
        The linear part of the forward model :math:`A`.
    s : Array or None, optional
        Additive contamination term :math:`s` (e.g. scatter/randoms).
        ``None`` (default) selects the pure linear model :math:`\\bar{y} = A x`.
    """

    def __init__(self, loss: C1Function, op: LinearOperator, s: Array | None = None):
        super().__init__()
        self._loss, self._op, self._s = loss, op, s

    def _call(self, x: Array) -> float:
        pred = self._op(x)
        if self._s is not None:
            pred = pred + self._s
        return self._loss(pred)

    def _gradient(self, x: Array) -> Array:
        pred = self._op(x)
        if self._s is not None:
            pred = pred + self._s
        return self._op.adjoint(self._loss.gradient(pred))

    def _call_and_gradient(self, x: Array) -> tuple[float, Array]:
        pred = self._op(x)
        if self._s is not None:
            pred = pred + self._s
        value, grad_pred = self._loss.call_and_gradient(pred)
        return value, self._op.adjoint(grad_pred)


class C2AffineObjective(C2Function, C1AffineObjective):
    """Composes a prediction-space :class:`C2Function` with an affine forward model.

    Extends :class:`C1AffineObjective` with second-order information. For
    :math:`f(x) = g(A x + s)` the Hessian-vector product is:

    .. math::

        H_f(x)\\, v = A^H \\bigl(\\operatorname{diag}(H_g(\\bar{y})) \\odot (A v)\\bigr)

    where :math:`\\bar{y} = A x + s` (or :math:`\\bar{y} = A x` when
    :math:`s` is ``None``) and :math:`\\odot` denotes elementwise
    multiplication.  Any scaling is carried exclusively by the ``beta``
    attribute of ``loss``.

    Parameters
    ----------
    loss : C2Function
        A :class:`C2Function` operating on the prediction space.
    op : LinearOperator
        The linear part of the forward model :math:`A`.
    s : Array or None, optional
        Additive contamination term :math:`s` (e.g. scatter/randoms).
        ``None`` (default) selects the pure linear model :math:`\\bar{y} = A x`.

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
    >>> loss.beta = 0.5
    >>> aff_obj = C2AffineObjective(loss, op, s)
    >>>
    >>> fx = aff_obj(x)                           # scalar value, scaled by beta=0.5
    >>> grad = aff_obj.gradient(x)                          # shape (4,), scaled by beta=0.5
    >>> hv   = aff_obj.hessian_diag_vec_prod(x, v)          # shape (4,), scaled by beta=0.5

    Pure linear forward model with virtual bins (zero rows in :math:`A`)
    handled via :class:`NegPoissonLogL` in safe mode (no mask needed --
    bins with :math:`y_i = 0` are derived from the data):

    >>> A2 = np.zeros((6, 4))
    >>> A2[:4, :] = np.random.rand(4, 4)   # last 2 rows are virtual (all zero)
    >>> op2   = MatrixOperator(A2)
    >>> data2 = np.array([2., 1., 3., 0., 0., 0.])  # virtual bins measure 0
    >>>
    >>> loss2 = NegPoissonLogL(data2, safe=True)
    >>> aff_obj2 = C2AffineObjective(loss2, op2)            # no contamination s
    >>>
    >>> fx2   = aff_obj2(x)                                 # scalar function value, no nan
    >>> grad2 = aff_obj2.gradient(x)                        # shape (4,), no nan
    >>> hv2   = aff_obj2.hessian_diag_vec_prod(x, v)        # shape (4,), no nan

    A regularised objective combining a data fidelity term and a roughness
    penalty via :class:`SumC2Function`:

    .. math::

        h(x) = \\underbrace{f_{\\text{PL}}(Ax + s)}_{\\text{data fidelity}}
               + \\underbrace{\\beta_{\\text{reg}} \\cdot
               \\tfrac{1}{2}\\|Dx\\|_2^2}_{\\text{roughness penalty}}

    where :math:`D` is a finite forward difference operator.

    >>> from parallelproj.operators import FiniteForwardDifference
    >>> from parallelproj.functions import HalfSquaredL2Deviation
    >>> beta_reg = 0.1
    >>> D   = FiniteForwardDifference(x.shape)            # finite differences in image space
    >>> reg = C2AffineObjective(HalfSquaredL2Deviation(beta=beta_reg), D)
    >>> data_fidelity = C2AffineObjective(NegPoissonLogL(data), op, s)
    >>>
    >>> obj_fun = data_fidelity + reg                     # SumC2Function
    >>> obj_val = obj_fun(x)                              # scalar function value
    >>> grad = obj_fun.gradient(x)                        # shape (4,)
    >>> hv   = obj_fun.hessian_diag_vec_prod(x, v)        # shape (4,)
    """

    def __init__(self, loss: C2Function, op: LinearOperator, s: Array | None = None):
        self._loss: C2Function
        super().__init__(loss, op, s)

    def _hessian_diag_vec_prod(self, x: Array, v: Array) -> Array:
        pred = self._op(x)
        if self._s is not None:
            pred = pred + self._s
        return self._op.adjoint(self._loss.hessian_diag_vec_prod(pred, self._op(v)))


class NonNegativeIndicator(FunctionWithProx):
    """Indicator function of the non-negative orthant.

    .. math::

        f(x) = \\begin{cases} 0 & x \\geq 0 \\\\ +\\infty & \\text{otherwise} \\end{cases}

    The proximal operator is the projection onto the non-negative orthant:

    .. math::

        \\text{prox}_{\\sigma f}(x) = \\max(x, 0)

    independent of :math:`\\sigma` and :math:`\\beta`.

    The dual prox (via Moreau's identity) is the projection onto the
    non-positive orthant:

    .. math::

        \\text{prox}_{\\sigma f^*}(y) = \\min(y, 0)

    Parameters
    ----------
    beta : float, optional
        Multiplicative scale factor :math:`\\beta`.  Defaults to ``1.0``.
        For an indicator function the value is 0 or :math:`+\\infty`
        regardless of :math:`\\beta > 0`.  The :meth:`prox` output is also
        unaffected by :math:`\\beta`, because the projection onto the
        non-negative orthant (:math:`\\max(x, 0)`) is independent of the
        step size.
    """

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        return 0.0 if bool(xp.all(x >= 0)) else float("inf")

    def _prox(self, x: Array, sigma: float | Array) -> Array:
        xp = get_namespace(x)
        return xp.where(x < 0, xp.zeros_like(x), x)


class MixedL21Norm(FunctionWithConjProx):
    """Mixed L2-L1 norm (isotropic TV semi-norm on a gradient field).

    For an array :math:`g` whose **first axis** enumerates the gradient
    directions (as produced by :class:`~parallelproj.operators.FiniteForwardDifference`),
    the norm is

    .. math::

        f(g) = \\sum_{\\mathbf{i}} \\|g_{:, \\mathbf{i}}\\|_2

    where the sum runs over all spatial multi-indices :math:`\\mathbf{i}` and
    the L2 norm is taken along axis 0.

    The convex conjugate is the indicator of the mixed :math:`L_{\\infty,2}`
    unit ball:

    .. math::

        f^*(p) = \\begin{cases}
            0 & \\|p_{:, \\mathbf{i}}\\|_2 \\leq 1 \\; \\forall \\mathbf{i} \\\\
            +\\infty & \\text{otherwise}
        \\end{cases}

    so its proximal operator is a pointwise projection onto the L2 unit ball
    along axis 0 (independent of :math:`\\sigma`):

    .. math::

        \\left(\\text{prox}_{\\sigma f^*}(y)\\right)_{:, \\mathbf{i}}
        = \\frac{y_{:, \\mathbf{i}}}{\\max\\!\\left(\\|y_{:, \\mathbf{i}}\\|_2,\\, 1\\right)}

    The direct :meth:`prox` (block soft-thresholding) is available via
    Moreau's identity.

    Parameters
    ----------
    beta : float, optional
        Multiplicative scale factor :math:`\\beta` (regularization weight).
        Defaults to ``1.0``.
    """

    def _call(self, x: Array) -> float:
        xp = get_namespace(x)
        return float(xp.sum(xp.linalg.vector_norm(x, axis=0)))

    def _prox_convex_conj(self, y: Array, sigma: float | Array) -> Array:
        xp = get_namespace(y)
        norm = xp.linalg.vector_norm(y, axis=0)  # shape: spatial_shape
        denom = xp.where(norm < 1, xp.ones_like(norm), norm)
        return y / xp.expand_dims(denom, axis=0)
