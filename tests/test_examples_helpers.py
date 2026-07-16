"""Smoke tests for the private examples-only helper module.

``parallelproj._examples_utils`` is not part of the public API and is excluded
from the coverage target (see ``[tool.coverage.run]`` in ``pyproject.toml`` and
``ignore`` in ``codecov.yml``).  These tests exist only to guarantee that the
module stays importable and that its main entry points run headlessly, so a
broken helper is caught in CI instead of shipping to users.

The tests run on numpy/CPU only and are deliberately not parametrized over
array backends/devices.  They create figures but never call ``show()`` (each
figure is closed immediately), so they are headless-safe and -- importantly --
do not switch the global matplotlib backend, which would make the rest of the
suite's ``fig.show()`` calls warn under a non-interactive backend.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

import array_api_compat.numpy as np

from parallelproj import _examples_utils as ex


def test_module_exposes_expected_names() -> None:
    for name in (
        "suggest_array_backend_and_device",
        "elliptic_cylinder_phantom",
        "show_vol_cuts",
        "show_3d_cuts",
        "poisson_transmission_terms",
        "RadonObject",
        "RadonObjectSequence",
        "RadonDisk",
        "RadonSquare",
        "neighbor_offsets",
        "neighbor_difference_and_sum",
        "neighbor_product",
        "SmoothFunction",
        "SmoothFunctionWithDiagonalHessian",
        "RDP",
    ):
        assert hasattr(ex, name), f"missing {name}"

    # documented backward-compatible alias
    assert ex.show_3d_cuts is ex.show_vol_cuts


def test_suggest_array_backend_and_device_numpy() -> None:
    xp, dev = ex.suggest_array_backend_and_device("numpy", "cpu")
    assert hasattr(xp, "asarray")
    assert dev == "cpu"


def test_elliptic_cylinder_phantom_default() -> None:
    img = ex.elliptic_cylinder_phantom(np, "cpu")
    assert img.ndim == 3
    assert float(np.max(img)) > 0.0


def test_show_vol_cuts_headless() -> None:
    vol = np.reshape(np.arange(8 * 8 * 4, dtype=np.float32), (8, 8, 4))
    fig, ax_im, widgets = ex.show_vol_cuts(vol, fig_title="smoke test")
    try:
        assert fig is not None
        assert ax_im is not None
        assert isinstance(widgets, dict)
    finally:
        plt.close(fig)


def test_radon_objects_construct() -> None:
    disk = ex.RadonDisk(np, "cpu", 1.0)
    seq = ex.RadonObjectSequence([disk])
    assert len(seq) == 1
    assert seq[0] is disk


def test_neighbor_helpers() -> None:
    x = np.reshape(np.arange(3 * 3 * 3, dtype=np.float32), (3, 3, 3))
    offsets = ex.neighbor_offsets(x.ndim)
    assert len(offsets) > 0
    diff, ssum = ex.neighbor_difference_and_sum(x, np)
    assert diff.shape[-3:] == x.shape
    assert ssum.shape[-3:] == x.shape


def test_rdp_prior_construct_and_call() -> None:
    in_shape = (4, 4, 2)
    x = np.ones(in_shape, dtype=np.float32)
    rdp = ex.RDP(in_shape, np, "cpu", voxel_size=np.asarray([1.0, 1.0, 1.0]))
    val = rdp(x)
    assert np.isfinite(float(val))
