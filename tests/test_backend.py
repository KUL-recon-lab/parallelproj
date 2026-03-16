from __future__ import annotations

import array_api_compat.numpy as np
from types import ModuleType

from parallelproj import to_numpy_array, count_event_multiplicity

from .config import pytestmark


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
