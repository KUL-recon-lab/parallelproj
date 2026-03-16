from __future__ import annotations

import numpy as np
import array_api_compat

from types import ModuleType
from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    import array_api_compat.cupy as cp
    import array_api_compat.torch as torch

    Array = Union[np.ndarray, cp.ndarray, torch.Tensor]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime


def to_numpy_array(x) -> np.ndarray:
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
