from __future__ import annotations

import array_api_compat.numpy as np
from types import ModuleType
from unittest.mock import patch

from parallelproj import to_numpy_array
from parallelproj._backend import empty_cuda_cache

from .config import pytestmark


def test_version_fallback(xp: ModuleType, dev: str) -> None:
    """Lines 5-6 of __init__.py: PackageNotFoundError causes __version__ = 'unknown'."""
    import importlib
    import importlib.metadata
    import parallelproj
    from importlib.metadata import PackageNotFoundError

    with patch.object(importlib.metadata, "version", side_effect=PackageNotFoundError):
        importlib.reload(parallelproj)

    assert parallelproj.__version__ == "unknown"

    # Restore normal state
    importlib.reload(parallelproj)


def test_to_numpy_array(xp: ModuleType, dev: str) -> None:
    arr = xp.asarray([1, 2, 3, 4, 5], device=dev)
    np_arr = np.asarray([1, 2, 3, 4, 5])

    arr_to_np = to_numpy_array(arr)

    assert np.all(arr_to_np == np_arr)


def test_empty_cuda_cache(xp: ModuleType, dev: str) -> None:
    empty_cuda_cache(xp)
