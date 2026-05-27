from __future__ import annotations

import pytest
import array_api_compat.numpy as np
from types import ModuleType
from unittest.mock import patch

from parallelproj import to_numpy_array, count_event_multiplicity
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


def test_event_multiplicity(xp: ModuleType, dev: str) -> None:

    events = xp.asarray(
        [
            [2, 1, 1, 1, 1],
            [1, -1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [2, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
        ],
        device=dev,
    )

    mu = count_event_multiplicity(events)

    assert xp.all(mu == xp.asarray([2, 1, 4, 4, 4, 2, 4], device=dev))


def test_to_numpy_array(xp: ModuleType, dev: str) -> None:
    arr = xp.asarray([1, 2, 3, 4, 5], device=dev)
    np_arr = np.asarray([1, 2, 3, 4, 5])

    arr_to_np = to_numpy_array(arr)

    assert np.all(arr_to_np == np_arr)


def test_empty_cuda_cache(xp: ModuleType, dev: str) -> None:
    empty_cuda_cache(xp)


def test_count_event_multiplicity_1d_raises(xp: ModuleType, dev: str) -> None:
    events_1d = xp.asarray([1, 2, 3], device=dev)
    with pytest.raises(ValueError, match="events must be a 2D array"):
        count_event_multiplicity(events_1d)
