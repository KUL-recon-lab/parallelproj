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
import pytest

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
    """Interactive viewer (forced), built headlessly."""
    vol = np.reshape(np.arange(8 * 8 * 4, dtype=np.float32), (8, 8, 4))
    fig, ax_im, widgets = ex.show_vol_cuts(vol, fig_title="smoke test", interactive=True)
    try:
        assert fig is not None
        assert ax_im is not None
        assert isinstance(widgets, dict) and len(widgets) > 0
    finally:
        plt.close(fig)


@pytest.mark.parametrize("ndim", [3, 4])
def test_show_vol_cuts_static(ndim, monkeypatch) -> None:
    """Non-interactive backend (as in the docs build): static cuts, no widgets."""
    shape = (8, 8, 4) if ndim == 3 else (3, 8, 8, 4)
    vol = np.reshape(np.arange(int(np.prod(np.asarray(shape))), dtype=np.float32), shape)
    # the default backend of the test runner can be interactive (e.g. macosx on
    # macOS, tkagg on Windows): emulate a non-interactive one for interactive=None
    monkeypatch.setattr(ex, "_interactive_backend", lambda: False)
    fig, ax_im, widgets = ex.show_vol_cuts(
        vol, voxel_size=(1.0, 1.0, 2.0), fig_title="static", origin="lower"
    )
    try:
        assert widgets == {}
        assert len(ax_im) == 3
        assert ax_im[2].get_title() == "x2 = 2"
        assert fig.get_suptitle().startswith("static")
        assert ("t = 1" in fig.get_suptitle()) == (ndim == 4)
        assert len(fig.axes) == 4  # three panels + colorbar
    finally:
        plt.close(fig)
    with pytest.raises(ValueError):
        ex.show_vol_cuts(vol, origin=("lower", "upper"), interactive=False)


def test_interactive_backend_detection(monkeypatch) -> None:
    for backend, expected in [
        ("agg", False),
        ("module://matplotlib_inline.backend_inline", False),
        ("QtAgg", True),
        ("tkagg", True),
        ("module://ipympl.backend_nbagg", True),
        ("widget", True),
    ]:
        monkeypatch.setattr(ex.plt, "get_backend", lambda b=backend: b)
        assert ex._interactive_backend() == expected, backend


def test_set_style() -> None:
    import matplotlib as mpl

    with mpl.rc_context():
        ex._reset_example_style(None, None)
        colors = mpl.rcParams["axes.prop_cycle"].by_key()["color"]
        assert tuple(colors) == ex.EXAMPLE_COLORS
        assert mpl.rcParams["legend.frameon"] is False


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


def test_iterations_for_device(capsys, monkeypatch) -> None:
    """CPU: reduced number + printed note; GPU: full number, no note."""
    monkeypatch.delenv("PARALLELPROJ_EXAMPLES_FULL_ITERATIONS", raising=False)
    assert ex.iterations_for_device("cpu", cpu=3, gpu=10, what="epochs") == 3
    out = capsys.readouterr().out
    assert "NOTE" in out and "3 epochs instead of 10" in out
    assert ex.iterations_for_device("cuda", cpu=3, gpu=10) == 10
    assert ex.iterations_for_device("cuda:1", cpu=3, gpu=10) == 10
    assert capsys.readouterr().out == ""
    # no note if the CPU number is not reduced
    assert ex.iterations_for_device("cpu", cpu=10, gpu=10) == 10
    assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
    "value, expected", [("1", 10), ("true", 10), ("ON", 10), ("0", 3), ("", 3)]
)
def test_iterations_for_device_env(capsys, monkeypatch, value, expected) -> None:
    """PARALLELPROJ_EXAMPLES_FULL_ITERATIONS=1 gives the full number on the CPU."""
    monkeypatch.setenv("PARALLELPROJ_EXAMPLES_FULL_ITERATIONS", value)
    assert ex.iterations_for_device("cpu", cpu=3, gpu=10) == expected
    assert ("NOTE" in capsys.readouterr().out) == (expected == 3)
