"""basic linear operators"""

from __future__ import annotations

from types import ModuleType
import abc
import numpy as np
import array_api_compat
from array_api_compat import device, get_namespace
from parallelproj import Array
from collections.abc import Sequence


class LinearOperator(abc.ABC):
    """abstract base class for linear operators"""

    def __init__(self) -> None:
        self._scale = 1

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
    def scale(self) -> int | float | complex:
        """scalar factor applied to the linear operator"""
        return self._scale

    @scale.setter
    def scale(self, value: int | float | complex):
        if not np.isscalar(value):
            raise ValueError
        self._scale = value

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
        if self._scale == 1:
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

        if self._scale == 1:
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

        return np.isclose(ip1, ip2, **kwargs)

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
                print(f"{(i+1):03} {xp.sqrt(norm_squared):.2E}")

        return float(xp.sqrt(norm_squared))


class MatrixOperator(LinearOperator):
    """Linear Operator defined by dense matrix multiplication

    Examples
    --------

    .. minigallery:: parallelproj.MatrixOperator
    """

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
    def in_shape(self) -> tuple[int]:
        return (self._A.shape[1],)

    @property
    def out_shape(self) -> tuple[int]:
        return (self._A.shape[0],)

    @property
    def A(self) -> Array:
        """matrix of the operator"""
        return self._A

    @property
    def xp(self) -> ModuleType:
        """array module of the operator"""
        return array_api_compat.get_namespace(self._A)

    def iscomplex(self) -> bool:
        """bool whether the operator is complex"""
        return self.xp.isdtype(self._A.dtype, self.xp.complex64) or self.xp.isdtype(
            self._A.dtype, self.xp.complex128
        )

    def _apply(self, x: Array) -> Array:
        return self.xp.matmul(self._A, x)

    def _adjoint(self, y: Array) -> Array:
        if self.iscomplex():
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

    Examples
    --------
    .. minigallery:: parallelproj.CompositeLinearOperator
    """

    def __init__(self, operators: Sequence[LinearOperator]):
        """init method

        Parameters
        ----------
        operators : Sequence[LinearOperator, ...]
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
        for op in self[::-1]:
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
    """Element-wise multiplication operator (multiplication with a diagonal matrix)

    Examples
    --------
    .. minigallery:: parallelproj.ElementwiseMultiplicationOperator
    """

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
        if self.iscomplex():
            return self.xp.conj(self._values) * y
        else:
            return self._values * y

    def iscomplex(self) -> bool:
        """bool whether the operator is complex"""
        return self.xp.isdtype(
            self._values.dtype, self.xp.complex64
        ) or self.xp.isdtype(self._values.dtype, self.xp.complex128)


class TOFNonTOFElementwiseMultiplicationOperator(LinearOperator):
    """Element-wise multiplication operator between a non-TOF and TOF sinogram

    Examples
    --------
    .. minigallery:: parallelproj.TOFNonTOFElementwiseMultiplicationOperator
    """

    def __init__(self, in_shape: tuple[int, ...], values: Array):
        """init method

        Parameters
        ----------
        in_shape: tuple[int, ...]
            shape of the TOF sinogram
        values : Array
            a non-TOF sinogram
        """
        super().__init__()
        self._in_shape = in_shape
        self._values = values

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def xp(self) -> ModuleType:
        """array module of the operator"""
        return array_api_compat.get_namespace(self._values)

    @property
    def values(self) -> Array:
        """values that get multiplied"""
        return self._values

    def _apply(self, x: Array) -> Array:
        y = 1 * x  # suboptimal to copy array, but bug in torch asarray with copy = True
        for i in range(x.shape[-1]):
            y[..., i] *= self._values
        return y

    def _adjoint(self, y: Array) -> Array:
        x = 1 * y  # suboptimal to copy array
        if self.iscomplex():
            tmp = self.xp.conj(self._values)
        else:
            tmp = self._values

        for i in range(x.shape[-1]):
            x[..., i] *= tmp

        return x

    def iscomplex(self) -> bool:
        """bool whether the operator is complex"""
        return self.xp.isdtype(
            self._values.dtype, self.xp.complex64
        ) or self.xp.isdtype(self._values.dtype, self.xp.complex128)


def _sigma_for_numpy(sigma):
    if array_api_compat.is_array_api_obj(sigma):
        return np.asarray(sigma)
    return sigma


def _sigma_for_cupy(sigma):
    if array_api_compat.is_array_api_obj(sigma):
        import cupy as cp

        return cp.asarray(sigma)
    return sigma


class GaussianFilterOperator(LinearOperator):
    """Gaussian filter operator

    Examples
    --------
    .. minigallery:: parallelproj.GaussianFilterOperator
    """

    def __init__(self, in_shape: tuple[int, ...], sigma: float | Array, **kwargs):
        """init method

        Parameters
        ----------
        in_shape : tuple[int, ...]
            shape of the input array
        sigma: float | array
            standard deviation of the gaussian filter
        **kwargs : sometype
            passed to the ndimage gaussian_filter function
        """
        super().__init__()
        self._in_shape = in_shape
        self._sigma = sigma
        self._kwargs = kwargs

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._in_shape

    def _apply(self, x: Array) -> Array:
        xp = array_api_compat.get_namespace(x)
        # NumPy
        if array_api_compat.is_numpy_array(
            x
        ) or array_api_compat.is_array_api_strict_namespace(xp):
            import scipy.ndimage as ndimage

            tmp = ndimage.gaussian_filter(
                np.asarray(x),
                sigma=_sigma_for_numpy(self._sigma),
                **self._kwargs,
            )

            if array_api_compat.is_array_api_strict_namespace(xp):
                return xp.asarray(tmp)
            else:
                return tmp

        # CuPy
        if array_api_compat.is_cupy_array(x):
            import cupy as cp
            import cupyx.scipy.ndimage as cndimage

            return cndimage.gaussian_filter(
                cp.asarray(x),
                sigma=_sigma_for_cupy(self._sigma),
                **self._kwargs,
            )

        # Torch
        if array_api_compat.is_torch_array(x):
            import torch

            if x.device.type == "cpu":
                import scipy.ndimage as ndimage

                y_np = ndimage.gaussian_filter(
                    x.detach().numpy(),
                    sigma=_sigma_for_numpy(self._sigma),
                    **self._kwargs,
                )
                return torch.from_numpy(np.asarray(y_np)).to(device=x.device)

            # Torch GPU -> CuPy via DLPack -> filter -> Torch via DLPack
            import cupy as cp
            import cupyx.scipy.ndimage as cndimage

            x_cp = cp.from_dlpack(x.detach())
            y_cp = cndimage.gaussian_filter(
                x_cp,
                sigma=_sigma_for_cupy(self._sigma),
                **self._kwargs,
            )
            return torch.from_dlpack(y_cp)

        raise TypeError(
            "Unsupported input type. Expected NumPy array, CuPy array, or Torch tensor."
        )

    def _adjoint(self, y: Array) -> Array:
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
        self._out_shapes = tuple([x.out_shape for x in operators])
        self._raveled_out_shapes = tuple([np.prod(x) for x in self._out_shapes])
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

    Examples
    --------

    .. minigallery:: parallelproj.LinearOperatorSequence
    """

    def __init__(self, operators: Sequence[LinearOperator]) -> None:
        """init method

        Parameters
        ----------
        operators : Sequence[LinearOperator, ...]
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

        return sum([op.adjoint(y[i]) for (i, op) in enumerate(self)])

    def norms(self, xp: ModuleType, dev: str) -> list[float]:
        """:math:`\\text{norm}(A^i)` for all :math:`i`"""
        return [op.norm(xp, dev) for op in self]


class FiniteForwardDifference(LinearOperator):
    """finite difference gradient operator"""

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

        if (
            gradient_field.dtype == self._xp.complex64
            or gradient_field.dtype == self._xp.complex128
        ):
            raise ValueError("complex gradient fields not supported")

        self._eta = eta

        self._in_shape = gradient_field.shape
        self._out_shape = gradient_field.shape

        gradient_field_float = self._xp.astype(gradient_field, self._xp.float64)

        norm = self._xp.sqrt(
            self._xp.sum(gradient_field_float**2 + self._eta**2, axis=0)
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
    def in_shape(self) -> tuple[int]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int]:
        return self._out_shape

    @property
    def xp(self):
        """array module of the operator"""
        return self._xp

    @property
    def dev(self):
        """device of the operator"""
        return self._dev

    @property
    def eta(self) -> float:
        """smoothing parameter"""
        return self._eta

    @property
    def normalized_gradient_field(self):
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
