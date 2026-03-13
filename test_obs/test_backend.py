from __future__ import annotations

import parallelproj
import array_api_compat
import array_api_compat.numpy as np
from types import ModuleType

# import the global pytestmark variable containing the xp/dev matrix we
# want to test
from .config import pytestmark


def test_calc_chunks(xp: ModuleType, dev: str) -> None:
    assert parallelproj.backend.calc_chunks(10, 3) == [0, 4, 7, 10]
    assert parallelproj.backend.calc_chunks(10, 2) == [0, 5, 10]
    assert parallelproj.backend.calc_chunks(10, 1) == [0, 10]

    assert parallelproj.backend.calc_chunks(1, 1) == [0, 1]


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

    mu = parallelproj.count_event_multiplicity(events)

    assert xp.all(mu == xp.asarray([2, 1, 4, 4, 4, 2, 4], device=dev))


def test_to_numpy_array(xp: ModuleType, dev: str) -> None:
    arr = xp.asarray([1, 2, 3, 4, 5], device=dev)
    np_arr = np.array([1, 2, 3, 4, 5])

    arr_to_np = parallelproj.to_numpy_array(arr)

    assert np.all(arr_to_np == np_arr)
