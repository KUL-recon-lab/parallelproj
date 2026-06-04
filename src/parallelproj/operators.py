"""Array-API-compatible linear operator abstractions and concrete implementations.

Provides the :class:`LinearOperator` abstract base class — with forward and
adjoint application, norm estimation via power iteration, and an adjointness
test — together with concrete operators: dense matrix multiplication,
element-wise multiplication, Gaussian filtering, forward finite differences,
operator composition, and vertical stacking.  All implementations dispatch
correctly across NumPy, CuPy, and PyTorch.
"""

from __future__ import annotations

from collections.abc import Sequence
from types import ModuleType
import abc
import os

import numpy as np
import array_api_compat

# Enable scipy's experimental array API support so that scipy.ndimage functions
# dispatch correctly for NumPy, CuPy, and PyTorch CPU arrays uniformly.
os.environ.setdefault("SCIPY_ARRAY_API", "1")
import scipy.ndimage as ndimage
from array_api_compat import device, get_namespace

try:
    import cupy as cp
except Exception:
    cp = None

from parallelproj import Array


class LinearOperator(abc.ABC):
    """Abstract base class for array-API-compatible linear operators.

    Subclasses implement :meth:`_apply` (:math:`y = Ax`) and
    :meth:`_adjoint` (:math:`x = A^H y`).  The public :meth:`apply` and
    :meth:`adjoint` methods apply an optional scalar :attr:`scale` factor
    (:math:`\\alpha A` and :math:`\\overline{\\alpha} A^H`).  Utility methods
    :meth:`adjointness_test` and :meth:`norm` are provided for validation and
    step-size estimation.
    """

    def __init__(self) -> None:
        self._scale: float | complex = 1.0

    @property
    @abc.abstractmethod
    def in_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def out_shape(self) -> tuple[int, ...]:
        """shape of the output array"""
        raise NotImplementedError

    @property
    def scale(self) -> float | complex:
        """scalar factor applied to the linear operator"""
        return self._scale

    @scale.setter
    def scale(self, value: float | complex):
        if not np.isscalar(value):
            raise ValueError("scale must be a scalar value")
        self._scale = complex(value) if isinstance(value, complex) else float(value)

    @abc.abstractmethod
    def _apply(self, x: Array) -> Array:
        """forward step :math:`y = Ax`"""
        raise NotImplementedError

    @abc.abstractmethod
    def _adjoint(self, y: Array) -> Array:
        """adjoint step :math:`x = A^H y`"""
        raise NotImplementedError

    def apply(self, x: Array) -> Array:
        """(scaled) forward step :math:`y = \\alpha A x`

        Parameters
        ----------
        x : Array

        Returns
        -------
        Array
        """
        if self._scale == 1.0:
            return self._apply(x)
        else:
            return self._scale * self._apply(x)

    def __call__(self, x: Array) -> Array:
        """alias to apply(x)"""
        return self.apply(x)

    def adjoint(self, y: Array) -> Array:
        """(scaled) adjoint step :math:`x = \\overline{\\alpha} A^H y`

        Parameters
        ----------
        y : Array

        Returns
        -------
        Array
        """

        if self._scale == 1.0:
            return self._adjoint(y)
        else:
            return self._scale.conjugate() * self._adjoint(y)

    def adjointness_test(
        self,
        xp: ModuleType,
        dev: str,
        verbose: bool = False,
        iscomplex: bool = False,
        dtype: type | None = None,
        **kwargs,
    ) -> bool:
        """test whether the adjoint is correctly implemented

        Parameters
        ----------
        xp : ModuleType
            array module to use
        dev : str
            device (cpu or cuda)
        verbose : bool, optional
            verbose output
        iscomplex : bool, optional
            use complex arrays
        dtype : type | None, optional
            data type of the arrays
        **kwargs : dict
            passed to np.isclose

        Returns
        -------
        bool
            whether the adjoint is correctly implemented
        """

        if dtype is None:
            if iscomplex:
                dtype = xp.complex128
            else:
                dtype = xp.float64

        x = xp.asarray(np.random.rand(*self.in_shape), device=dev, dtype=dtype)
        y = xp.asarray(np.random.rand(*self.out_shape), device=dev, dtype=dtype)

        if iscomplex:
            x = x + 1j * xp.asarray(
                np.random.rand(*self.in_shape), device=dev, dtype=dtype
            )

        if iscomplex:
            y = y + 1j * xp.asarray(
                np.random.rand(*self.out_shape), device=dev, dtype=dtype
            )

        x_fwd = self.apply(x)

        y_adj = self.adjoint(y)

        if iscomplex:
            ip1 = complex(xp.sum(xp.conj(x_fwd) * y))
            ip2 = complex(xp.sum(xp.conj(x) * y_adj))
        else:
            ip1 = float(xp.sum(x_fwd * y))
            ip2 = float(xp.sum(x * y_adj))

        if verbose:
            print(ip1, ip2)

        return bool(np.isclose(ip1, ip2, **kwargs))

    def norm(
        self,
        xp: ModuleType,
        dev: str,
        num_iter: int = 30,
        iscomplex: bool = False,
        verbose: bool = False,
    ) -> float:
        """estimate norm of the linear operator using power iterations

        Parameters
        ----------
        xp : ModuleType
            array module to use
        dev : str
            device (cpu or cuda)
        num_iter : int, optional
            number of power iterations
        iscomplex : bool, optional
            use complex arrays
        verbose : bool, optional
            verbose output

        Returns
        -------
        float
            the norm of the linear operator
        """

        if iscomplex:
            dtype = xp.complex128
        else:
            dtype = xp.float64

        x = xp.asarray(np.random.rand(*self.in_shape), device=dev, dtype=dtype)

        if iscomplex:
            x = x + 1j * xp.asarray(
                np.random.rand(*self.in_shape), device=dev, dtype=dtype
            )

        for i in range(num_iter):
            x = self.adjoint(self.apply(x))
            norm_squared = xp.sqrt(xp.sum(xp.abs(x) ** 2))
            x /= float(norm_squared)

            if verbose:
                print(f"{(i+1):03} {float(xp.sqrt(norm_squared)):.2E}")

        return float(xp.sqrt(norm_squared))

    @property
    def H(self) -> AdjointLinearOperator:
        """adjoint operator :math:`A^H`"""
        return AdjointLinearOperator(self)


class AdjointLinearOperator(LinearOperator):
    """Adjoint of a linear operator

    Wraps an existing :class:`LinearOperator` so that ``__call__`` applies
    :math:`A^H` and ``adjoint`` applies :math:`A`.  The scale of this operator
    is always the complex conjugate of the wrapped operator's scale; setting
    the scale on either one propagates to the other.

    Use the :attr:`LinearOperator.H` property rather than constructing this
    class directly.
    """

    def __init__(self, operator: LinearOperator) -> None:
        """init method

        Parameters
        ----------
        operator : LinearOperator
            the operator whose adjoint is to be represented
        """
        super().__init__()
        self._operator = operator

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._operator.out_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._operator.in_shape

    @property
    def scale(self) -> float | complex:
        """conjugate of the wrapped operator's scale"""
        return self._operator.scale.conjugate()

    @scale.setter
    def scale(self, value: float | complex):
        # setting A.H.scale = a sets A.scale = conj(a)
        self._operator.scale = value.conjugate()

    def _apply(self, x: Array) -> Array:
        return self._operator._adjoint(x)

    def _adjoint(self, y: Array) -> Array:
        return self._operator._apply(y)


class MatrixOperator(LinearOperator):
    """Linear Operator defined by dense matrix multiplication"""

    def __init__(self, A: Array) -> None:
        """init method

        Parameters
        ----------
        A : Array
            2D real or complex array representing the matrix
        """
        super().__init__()
        self._A = A

    @property
    def in_shape(self) -> tuple[int, ...]:
        return (self._A.shape[1],)

    @property
    def out_shape(self) -> tuple[int, ...]:
        return (self._A.shape[0],)

    @property
    def A(self) -> Array:
        """matrix of the operator"""
        return self._A

    @property
    def xp(self) -> ModuleType:
        """array module of the operator"""
        return array_api_compat.get_namespace(self._A)

    @property
    def iscomplex(self) -> bool:
        """bool whether the operator is complex"""
        return self.xp.isdtype(self._A.dtype, self.xp.complex64) or self.xp.isdtype(
            self._A.dtype, self.xp.complex128
        )

    def _apply(self, x: Array) -> Array:
        return self.xp.matmul(self._A, x)

    def _adjoint(self, y: Array) -> Array:
        if self.iscomplex:
            return self.xp.matmul(self.xp.conj(self._A).T, y)
        else:
            return self.xp.matmul(self._A.T, y)


class CompositeLinearOperator(LinearOperator):
    """Composite Linear Operator defined by a sequence of Linear Operators

    Given a Sequence of operators

    .. math::
            A^0, A^1, \\ldots, A^{n-1}

    the composite linear operator is defined as

    .. math::
        A(x) = A^0( A^1( ... ( A^{n-1}(x) ) ) )
    """

    def __init__(self, operators: Sequence[LinearOperator]):
        """init method

        Parameters
        ----------
        operators : Sequence[LinearOperator]
            Sequence of linear operators
        """
        super().__init__()
        self._operators = operators

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._operators[-1].in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._operators[0].out_shape

    @property
    def operators(self) -> Sequence[LinearOperator]:
        """tuple of linear operators"""
        return self._operators

    def _apply(self, x: Array) -> Array:
        y = x
        for op in reversed(self._operators):
            y = op(y)
        return y

    def _adjoint(self, y: Array) -> Array:
        x = y
        for op in self:
            x = op.adjoint(x)
        return x

    def __getitem__(self, i: int) -> LinearOperator:
        """get the i-th operator :math:`A_i`"""
        return self._operators[i]


class ElementwiseMultiplicationOperator(LinearOperator):
    """Element-wise multiplication operator (multiplication with a diagonal matrix)"""

    def __init__(self, values: Array):
        """init method

        Parameters
        ----------
        values : Array
            values of the diagonal matrix
        """
        super().__init__()
        self._values = values

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._values.shape

    @property
    def xp(self) -> ModuleType:
        """array module of the operator"""
        return array_api_compat.get_namespace(self._values)

    @property
    def values(self) -> Array:
        """values that get multiplied"""
        return self._values

    def _apply(self, x: Array) -> Array:
        return self._values * x

    def _adjoint(self, y: Array) -> Array:
        if self.iscomplex:
            return self.xp.conj(self._values) * y
        else:
            return self._values * y

    @property
    def iscomplex(self) -> bool:
        """bool whether the operator is complex"""
        return self.xp.isdtype(
            self._values.dtype, self.xp.complex64
        ) or self.xp.isdtype(self._values.dtype, self.xp.complex128)


class GaussianFilterOperator(LinearOperator):
    """Isotropic Gaussian smoothing operator (self-adjoint).

    Wraps ``scipy.ndimage.gaussian_filter`` and dispatches via the array API
    so it works with NumPy, CuPy, and PyTorch CPU arrays.  PyTorch CUDA
    tensors are round-tripped through CuPy via DLPack.  All keyword arguments
    accepted by ``scipy.ndimage.gaussian_filter`` (e.g. ``sigma``, ``mode``,
    ``truncate``) are forwarded through ``**kwargs``.  Because the Gaussian
    kernel is symmetric, the adjoint equals the forward application.
    """

    def __init__(self, in_shape: tuple[int, ...], **kwargs):
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        **kwargs : dict
            passed to scipy.ndimage.gaussian_filter (e.g. ``mode``, ``truncate``)
        """
        super().__init__()
        self._in_shape = in_shape
        self._kwargs = kwargs

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._in_shape

    def _apply(self, x: Array) -> Array:
        xp = array_api_compat.get_namespace(x)
        dev = array_api_compat.device(x)

        # PyTorch CUDA: scipy array API dispatch does not support CUDA tensors
        # yet, so round-trip via CuPy using DLPack.
        if array_api_compat.is_torch_array(x) and x.device.type != "cpu":
            assert (
                cp is not None
            ), "cupy must be installed to use GaussianFilterOperator with PyTorch CUDA tensors"
            x_cp = cp.from_dlpack(x.detach())
            y_cp = ndimage.gaussian_filter(x_cp, **self._kwargs)
            return xp.asarray(xp.from_dlpack(y_cp))
        else:
            # All other cases (NumPy, CuPy, PyTorch CPU, array-api-strict) are
            # handled uniformly via scipy's array API dispatch (SCIPY_ARRAY_API=1).
            # scipy may return a plain numpy array even for non-numpy inputs, so
            # convert the result back to the input's array namespace/device.
            result = ndimage.gaussian_filter(x, **self._kwargs)
            return xp.asarray(result, device=dev, dtype=x.dtype)

    def _adjoint(self, y: Array) -> Array:
        # A Gaussian filter with a symmetric kernel is self-adjoint, so the
        # adjoint equals the forward application.
        return self._apply(y)


class VstackOperator(LinearOperator):
    """Stacking operator for stacking multiple linear operators vertically"""

    def __init__(self, operators: tuple[LinearOperator, ...]) -> None:
        """init method

        Parameters
        ----------
        operators : tuple[LinearOperator, ...]
            tuple of linear operators
        """
        super().__init__()
        self._operators = operators
        self._in_shape = self._operators[0].in_shape
        if any(op.in_shape != self._in_shape for op in self._operators[1:]):
            raise ValueError(
                "all operators in VstackOperator must have the same in_shape"
            )
        self._out_shapes = tuple([x.out_shape for x in operators])
        self._raveled_out_shapes = tuple([int(np.prod(x)) for x in self._out_shapes])
        self._out_shape = (sum(self._raveled_out_shapes),)

        # setup the slices for slicing the raveled output array
        self._slices = []
        start = 0
        for length in self._raveled_out_shapes:
            end = start + length
            self._slices.append(slice(start, end, None))
            start = end
        self._slices = tuple(self._slices)

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._out_shape

    def _apply(self, x: Array) -> Array:
        xp = array_api_compat.get_namespace(x)
        y = xp.zeros(self._out_shape, dtype=x.dtype, device=device(x))

        for i, op in enumerate(self._operators):
            y[self._slices[i]] = xp.reshape(op(x), (-1,))

        return y

    def _adjoint(self, y: Array) -> Array:
        xp = array_api_compat.get_namespace(y)
        x = xp.zeros(self._in_shape, dtype=y.dtype, device=device(y))

        for i, op in enumerate(self._operators):
            x += op.adjoint(xp.reshape(y[self._slices[i]], self._out_shapes[i]))

        return x


class LinearOperatorSequence(Sequence[LinearOperator]):
    """Sequence of linear operators

    .. math::
       A^0, A^1 \\ldots, A^{n-1}

    that can be evaluated independently.
    """

    def __init__(self, operators: Sequence[LinearOperator]) -> None:
        """init method

        Parameters
        ----------
        operators : Sequence[LinearOperator]
            Sequence of linear operators
        """
        self._operators = operators
        self._in_shape = self._operators[0].in_shape
        self._out_shapes = [x.out_shape for x in operators]
        self._len = len(operators)

    @property
    def in_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        return self._in_shape

    @property
    def out_shapes(self) -> list[tuple[int, ...]]:
        """shapes of the output array of all subset operators"""
        return self._out_shapes

    @property
    def operators(self) -> Sequence[LinearOperator]:
        """all subset operators"""
        return self._operators

    def __len__(self) -> int:
        """length of operator sequence"""
        return self._len

    def __getitem__(self, i: int) -> LinearOperator:
        """get the i-th linear operator :math:`A^i`"""
        return self._operators[i]

    def apply(self, x: Array) -> list[Array]:
        """:math:`(A^0(x), A^1(x), \\ldots, A^{n-1}(x))`"""

        y = [op(x) for op in self]

        return y

    def __call__(self, x: Array) -> list[Array]:
        return self.apply(x)

    def adjoint(self, y: list[Array]) -> Array:
        """:math:`\\sum_i (A^i)^H y^i` for all :math:`i`"""

        result = self._operators[0].adjoint(y[0])
        for i, op in enumerate(self._operators[1:], start=1):
            result = result + op.adjoint(y[i])
        return result

    def norms(self, xp: ModuleType, dev: str) -> list[float]:
        """:math:`\\text{norm}(A^i)` for all :math:`i`

        Parameters
        ----------
        xp : ModuleType
            array module to use
        dev : str
            device (cpu or cuda)

        Returns
        -------
        list[float]
            norm of each operator in the sequence
        """
        return [op.norm(xp, dev) for op in self]


class FiniteForwardDifference(LinearOperator):
    """Forward finite-difference gradient operator for 1-D to 4-D images.

    Maps an image of shape ``in_shape`` to a gradient field of shape
    ``(ndim, *in_shape)`` where axis 0 enumerates the spatial directions.
    Boundary conditions are Neumann (zero-padding): the last slice along each
    axis is set to zero in the forward pass.  The adjoint is the negative
    discrete divergence, consistent with the standard TV regularisation
    convention.  Self-adjointness is verified by :meth:`adjointness_test`.
    """

    def __init__(self, in_shape: tuple[int, ...]) -> None:
        if len(in_shape) > 4:
            raise ValueError("only up to 4 dimensions supported")

        self._ndim = len(in_shape)
        self._in_shape = in_shape
        self._out_shape = (self.ndim,) + in_shape
        super().__init__()

    @property
    def in_shape(self) -> tuple[int, ...]:
        """shape of the input array"""
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        """shape of the output array"""
        return self._out_shape

    @property
    def ndim(self) -> int:
        """number of dimensions of the input array"""
        return self._ndim

    def _apply(self, x: Array) -> Array:
        xp = array_api_compat.get_namespace(x)

        g = xp.zeros(self.out_shape, dtype=x.dtype, device=device(x))

        if self.ndim == 1:
            g[0, :-1] = x[1:] - x[:-1]
        elif self.ndim == 2:
            g[0, :-1, :] = x[1:, :] - x[:-1, :]
            g[1, :, :-1] = x[:, 1:] - x[:, :-1]
        elif self.ndim == 3:
            g[0, :-1, :, :] = x[1:, :, :] - x[:-1, :, :]
            g[1, :, :-1, :] = x[:, 1:, :] - x[:, :-1, :]
            g[2, :, :, :-1] = x[:, :, 1:] - x[:, :, :-1]
        elif self.ndim == 4:
            g[0, :-1, :, :, :] = x[1:, :, :, :] - x[:-1, :, :, :]
            g[1, :, :-1, :, :] = x[:, 1:, :, :] - x[:, :-1, :, :]
            g[2, :, :, :-1, :] = x[:, :, 1:, :] - x[:, :, :-1, :]
            g[3, :, :, :, :-1] = x[:, :, :, 1:] - x[:, :, :, :-1]

        return g

    def _adjoint(self, y: Array) -> Array:
        xp = array_api_compat.get_namespace(y)

        if self.ndim == 1:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp0[-1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div0[1:] = -tmp0[1:] + tmp0[:-1]
            div0[0] = -tmp0[0]

            res = div0

        elif self.ndim == 2:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp1 = xp.asarray(y[1, ...], copy=True)
            tmp0[-1, :] = 0
            tmp1[:, -1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))

            div0[1:, :] = -tmp0[1:, :] + tmp0[:-1, :]
            div1[:, 1:] = -tmp1[:, 1:] + tmp1[:, :-1]

            div0[0, :] = -tmp0[0, :]
            div1[:, 0] = -tmp1[:, 0]

            res = div0 + div1

        elif self.ndim == 3:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp1 = xp.asarray(y[1, ...], copy=True)
            tmp2 = xp.asarray(y[2, ...], copy=True)
            tmp0[-1, :, :] = 0
            tmp1[:, -1, :] = 0
            tmp2[:, :, -1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div2 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))

            div0[1:, :, :] = -tmp0[1:, :, :] + tmp0[:-1, :, :]
            div1[:, 1:, :] = -tmp1[:, 1:, :] + tmp1[:, :-1, :]
            div2[:, :, 1:] = -tmp2[:, :, 1:] + tmp2[:, :, :-1]

            div0[0, :, :] = -tmp0[0, :, :]
            div1[:, 0, :] = -tmp1[:, 0, :]
            div2[:, :, 0] = -tmp2[:, :, 0]

            res = div0 + div1 + div2

        elif self.ndim == 4:
            tmp0 = xp.asarray(y[0, ...], copy=True)
            tmp1 = xp.asarray(y[1, ...], copy=True)
            tmp2 = xp.asarray(y[2, ...], copy=True)
            tmp3 = xp.asarray(y[3, ...], copy=True)
            tmp0[-1, :, :, :] = 0
            tmp1[:, -1, :, :] = 0
            tmp2[:, :, -1, :] = 0
            tmp3[:, :, :, -1] = 0

            div0 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div1 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div2 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))
            div3 = xp.zeros(self.in_shape, dtype=y.dtype, device=device(y))

            div0[1:, :, :, :] = -tmp0[1:, :, :, :] + tmp0[:-1, :, :, :]
            div1[:, 1:, :, :] = -tmp1[:, 1:, :, :] + tmp1[:, :-1, :, :]
            div2[:, :, 1:, :] = -tmp2[:, :, 1:, :] + tmp2[:, :, :-1, :]
            div3[:, :, :, 1:] = -tmp3[:, :, :, 1:] + tmp3[:, :, :, :-1]

            div0[0, :, :, :] = -tmp0[0, :, :, :]
            div1[:, 0, :, :] = -tmp1[:, 0, :, :]
            div2[:, :, 0, :] = -tmp2[:, :, 0, :]
            div3[:, :, :, 0] = -tmp3[:, :, :, 0]

            res = div0 + div1 + div2 + div3
        else:
            raise ValueError("only up to 4 dimensions supported")

        return res


class GradientFieldProjectionOperator(LinearOperator):
    """Gradient Field Projection Operator

    See Ehrhardt and Betcke: "Multicontrast MRI Reconstruction with Structure-Guided Total Variation"
    https://doi.org/10.1137/15M1047325

    .. math::
       P_{\\xi_n}x = x - \\langle \\xi_n, x \\rangle \\xi_n

    .. math::
       \\xi_n = g_n / \\| g_n \\|_{\\eta}

    for the joint gradient field :math:`g_n`
    """

    def __init__(self, gradient_field: Array, eta: float = 0.0):
        """
        Parameters
        ----------
        gradient_field : Array
            a real gradient field. In 3D, the shape would be [3,n0,n1,n2].
            In 2D, the shape would be [2,n0,n1].
            This can be e.g. the output of the FiniteForwardDifference operator
            applied to a structural prior image.
        eta : float, optional
            smoothing parameter used in the pointwise gradient norm
            default 0.0
        """

        self._xp = get_namespace(gradient_field)
        self._dev = device(gradient_field)

        if self._xp.isdtype(gradient_field.dtype, "complex floating"):
            raise ValueError("complex gradient fields not supported")

        self._eta = eta

        self._in_shape = gradient_field.shape
        self._out_shape = gradient_field.shape

        gradient_field_float = self._xp.astype(gradient_field, self._xp.float64)

        norm = self._xp.sqrt(
            self._xp.sum(gradient_field_float**2, axis=0) + self._eta**2
        )
        inds = norm > 0
        self._normalized_gradient_field = self._xp.zeros(
            gradient_field.shape,
            dtype=self._xp.float64,
            device=self._dev,
        )

        for i in range(self.out_shape[0]):
            self._normalized_gradient_field[i, ...][inds] = (
                gradient_field_float[i, ...][inds] / norm[inds]
            )

        super().__init__()

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._out_shape

    @property
    def xp(self) -> ModuleType:
        """array module of the operator"""
        return self._xp

    @property
    def dev(self) -> str:
        """device of the operator"""
        return self._dev

    @property
    def eta(self) -> float:
        """smoothing parameter"""
        return self._eta

    @property
    def normalized_gradient_field(self) -> Array:
        """normalized gradient field"""
        return self._normalized_gradient_field

    def _apply(self, x: Array) -> Array:
        return (
            x
            - self._xp.sum(x * self._normalized_gradient_field, axis=0)
            * self._normalized_gradient_field
        )

    def _adjoint(self, y: Array) -> Array:
        return self._apply(y)
