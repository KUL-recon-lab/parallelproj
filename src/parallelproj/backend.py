from __future__ import annotations

import numpy as np
import array_api_compat

from typing import Union, TYPE_CHECKING


if TYPE_CHECKING:
    import array_api_compat.cupy as cp
    import array_api_compat.torch as torch

    Array = Union[np.ndarray, cp.ndarray, torch.Tensor]  # Used for type checking
else:
    Array = np.ndarray  # Default at runtime


def to_numpy_array(x) -> np.ndarray:
    if array_api_compat.is_cupy_array(x):
        import array_api_compat.cupy as cp

        return cp.asnumpy(x)
    elif array_api_compat.is_torch_array(x):
        return (
            x.detach().numpy() if x.device.type == "cpu" else x.detach().cpu().numpy()
        )
    else:
        return np.asarray(x)
