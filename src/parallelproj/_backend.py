"""Internal array-API utilities, the Array protocol type, and backend helpers."""

from __future__ import annotations

from types import ModuleType
from typing import Any, Iterator, Protocol

import numpy as np
import array_api_compat


class Array(Protocol):
    """Protocol for Python Array API-compliant arrays.

    Any array satisfying this protocol (numpy ndarray, CuPy ndarray,
    PyTorch Tensor, ...) can be used wherever :class:`Array` is expected.
    The declared operations correspond to the Python Array API standard
    (https://data-apis.org/array-api/latest/API_specification/array_object.html).
    """

    # ------------------------------------------------------------------
    # Attributes
    # ------------------------------------------------------------------

    @property
    def dtype(self) -> Any: ...
    @property
    def device(self) -> Any: ...
    @property
    def ndim(self) -> int: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def size(self) -> int: ...
    @property
    def T(self) -> Any: ...
    @property
    def mT(self) -> Any: ...

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def __bool__(self) -> bool: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __index__(self) -> int: ...
    def __len__(self) -> int: ...
    def __iter__(self) -> Iterator[Any]: ...

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def __getitem__(self, key: Any) -> Any: ...
    def __setitem__(self, key: Any, value: Any) -> None: ...

    # ------------------------------------------------------------------
    # Arithmetic operators
    # ------------------------------------------------------------------

    def __pos__(self) -> Any: ...
    def __neg__(self) -> Any: ...
    def __add__(self, other: Any) -> Any: ...
    def __radd__(self, other: Any) -> Any: ...
    def __iadd__(self, other: Any) -> Any: ...
    def __sub__(self, other: Any) -> Any: ...
    def __rsub__(self, other: Any) -> Any: ...
    def __isub__(self, other: Any) -> Any: ...
    def __mul__(self, other: Any) -> Any: ...
    def __rmul__(self, other: Any) -> Any: ...
    def __imul__(self, other: Any) -> Any: ...
    def __truediv__(self, other: Any) -> Any: ...
    def __rtruediv__(self, other: Any) -> Any: ...
    def __itruediv__(self, other: Any) -> Any: ...
    def __floordiv__(self, other: Any) -> Any: ...
    def __rfloordiv__(self, other: Any) -> Any: ...
    def __ifloordiv__(self, other: Any) -> Any: ...
    def __mod__(self, other: Any) -> Any: ...
    def __rmod__(self, other: Any) -> Any: ...
    def __imod__(self, other: Any) -> Any: ...
    def __pow__(self, other: Any) -> Any: ...
    def __rpow__(self, other: Any) -> Any: ...
    def __ipow__(self, other: Any) -> Any: ...

    # ------------------------------------------------------------------
    # Matrix multiplication
    # ------------------------------------------------------------------

    def __matmul__(self, other: Any) -> Any: ...
    def __rmatmul__(self, other: Any) -> Any: ...
    def __imatmul__(self, other: Any) -> Any: ...

    # ------------------------------------------------------------------
    # Bitwise operators
    # ------------------------------------------------------------------

    def __invert__(self) -> Any: ...
    def __and__(self, other: Any) -> Any: ...
    def __rand__(self, other: Any) -> Any: ...
    def __iand__(self, other: Any) -> Any: ...
    def __or__(self, other: Any) -> Any: ...
    def __ror__(self, other: Any) -> Any: ...
    def __ior__(self, other: Any) -> Any: ...
    def __xor__(self, other: Any) -> Any: ...
    def __rxor__(self, other: Any) -> Any: ...
    def __ixor__(self, other: Any) -> Any: ...
    def __lshift__(self, other: Any) -> Any: ...
    def __rlshift__(self, other: Any) -> Any: ...
    def __ilshift__(self, other: Any) -> Any: ...
    def __rshift__(self, other: Any) -> Any: ...
    def __rrshift__(self, other: Any) -> Any: ...
    def __irshift__(self, other: Any) -> Any: ...

    # ------------------------------------------------------------------
    # Comparison operators
    # ------------------------------------------------------------------

    def __lt__(self, other: Any) -> Any: ...
    def __le__(self, other: Any) -> Any: ...
    def __gt__(self, other: Any) -> Any: ...
    def __ge__(self, other: Any) -> Any: ...
    def __eq__(self, other: Any) -> Any: ...  # type: ignore[override]
    def __ne__(self, other: Any) -> Any: ...  # type: ignore[override]

    # ------------------------------------------------------------------
    # Interoperability
    # ------------------------------------------------------------------

    def __array_namespace__(self, *, api_version: str | None = None) -> Any: ...
    def __dlpack__(self, *, stream: Any = None) -> Any: ...
    def __dlpack_device__(self) -> tuple[int, int]: ...
    def to_device(self, device: Any, /, *, stream: Any = None) -> Any: ...


def to_numpy_array(x: Array) -> np.ndarray:
    """Convert any Array-API-compatible array to a NumPy ndarray.

    Handles CuPy, PyTorch (CPU and CUDA), and anything ``np.asarray`` can
    digest.  For CUDA arrays a device-to-host copy is performed.

    Parameters
    ----------
    x : Array
        Any Array-API-compatible array.

    Returns
    -------
    np.ndarray
        NumPy ndarray with the same data and dtype as *x*.
    """
    if array_api_compat.is_cupy_array(x):  # pragma: no cover
        cp = array_api_compat.get_namespace(x)

        return cp.asnumpy(x)
    elif array_api_compat.is_torch_array(x):
        return (
            x.detach().numpy() if x.device.type == "cpu" else x.detach().cpu().numpy()
        )
    else:
        return np.asarray(x)


def empty_cuda_cache(xp: ModuleType) -> None:
    """Empty cached CUDA memory for supported backends.

    For CuPy, frees all blocks in both the default memory pool and the pinned
    memory pool.  For PyTorch, calls ``torch.cuda.empty_cache()`` when a CUDA
    device is available.  For all other namespaces (e.g. NumPy) this is a
    no-op.

    Parameters
    ----------
    xp : ModuleType
        Array namespace as returned by ``array_api_compat.get_namespace()``.
    """
    if array_api_compat.is_cupy_namespace(xp):  # pragma: no cover
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    elif array_api_compat.is_torch_namespace(xp):
        torch_mod = getattr(xp, "torch", xp)
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()


def _current_cuda_stream(x: Array) -> int | None:
    """Handle of the current CUDA stream of the GPU of *x* (``None`` for CPU arrays).

    Passed as ``stream`` to the projectors of ``parallelproj_core``, so that a
    projection of GPU arrays behaves like any other operation of the array
    library: it is queued on the current stream of the GPU of the arrays, runs
    in order with all other operations on that stream (also inside
    ``with torch.cuda.stream(s):`` / ``with cupy.cuda.Stream():``) and returns
    without waiting for the GPU.  Projections on several GPUs therefore run at
    the same time, even when called one after another from a single thread.
    The stream allocators of PyTorch and CuPy are stream-ordered, so temporary
    arrays freed after the call are not reused before the projection is done.

    Parameters
    ----------
    x : Array
        Array on the device of the projection.

    Returns
    -------
    int or None
        ``torch.cuda.current_stream(x.device).cuda_stream`` for PyTorch CUDA
        tensors, ``cupy.cuda.get_current_stream().ptr`` (of the GPU of *x*) for
        CuPy arrays (``0`` is the default stream of both libraries), ``None``
        otherwise (synchronous call; CPU arrays always give complete results).
    """
    if array_api_compat.is_torch_array(x):
        if x.device.type == "cuda":  # pragma: no cover (GPU only)
            import torch

            return torch.cuda.current_stream(x.device).cuda_stream
        return None
    if array_api_compat.is_cupy_array(x):  # pragma: no cover
        import cupy

        with x.device:
            return cupy.cuda.get_current_stream().ptr
    return None
