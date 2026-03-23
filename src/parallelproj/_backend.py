from __future__ import annotations

import numpy as np
import array_api_compat

from types import ModuleType
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    from cupy import ndarray as CupyArray
    from torch import Tensor as TorchTensor

    Array = Union[np.ndarray, CupyArray, TorchTensor]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime


def to_numpy_array(x: Array) -> np.ndarray:
    if array_api_compat.is_cupy_array(x):
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

    For unsupported or non-CUDA namespaces (e.g. NumPy), do nothing.
    """
    if array_api_compat.is_cupy_namespace(xp):
        xp.get_default_memory_pool().free_all_blocks()
        xp.get_default_pinned_memory_pool().free_all_blocks()
    elif array_api_compat.is_torch_namespace(xp):
        torch_mod = getattr(xp, "torch", xp)
        if torch_mod.cuda.is_available():
            torch_mod.cuda.empty_cache()


def count_event_multiplicity(events: Array) -> Array:
    """Count the multiplicity of each event row."""
    xp = array_api_compat.get_namespace(events)

    if events.ndim != 2:
        raise ValueError("events must be a 2D array")

    if array_api_compat.is_torch_namespace(xp):
        return _count_event_multiplicity_torch(events, xp)
    elif array_api_compat.is_cupy_namespace(xp):
        return _count_event_multiplicity_cupy(events, xp)
    else:
        return _count_event_multiplicity_numpy_fallback(events, xp)


def _count_event_multiplicity_torch(events: Array, xp) -> Array:
    torch_mod = _native_torch_module(xp)
    _, inverse, counts = torch_mod.unique(
        events,
        dim=0,
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse].reshape(-1)


def _count_event_multiplicity_cupy(events: Array, xp) -> Array:
    cupy_mod = xp
    _, inverse, counts = cupy_mod.unique(
        events,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return counts[inverse].reshape(-1)


def _count_event_multiplicity_numpy_fallback(events: Array, xp) -> Array:
    x_np = to_numpy_array(events)
    _, inverse, counts = np.unique(
        x_np,
        axis=0,
        return_inverse=True,
        return_counts=True,
    )
    return xp.asarray(counts[inverse].reshape(-1))


def _native_torch_module(xp):
    return xp if xp.__name__ == "torch" else xp.torch
