"""Private helper utilities backing the parallelproj documentation examples.

.. warning::

   **Dirty, examples-only module — NOT part of the public parallelproj API.**

   The names in this module exist purely so that the Sphinx-Gallery examples
   (and the notebooks / scripts downloaded from the rendered documentation)
   run with nothing more than ``parallelproj`` installed -- no ``PYTHONPATH``
   setup and no extra file downloads.  It is intentionally excluded from the
   test-coverage target and its contents may change or be removed at any time
   without notice.  Do not rely on it in your own code.

   Importing this module requires ``matplotlib`` (a runtime dependency of the
   examples); it is deliberately not imported by ``parallelproj/__init__`` so
   that ``import parallelproj`` stays matplotlib-free.

Contents
--------
* :func:`suggest_array_backend_and_device` -- pick the best available array
  backend and compute device.
* :func:`elliptic_cylinder_phantom` -- 3-D test phantom with spherical inserts.
* :func:`show_vol_cuts` -- interactive 2-D slice viewer for 3-D/4-D arrays
  (aliased as :func:`show_3d_cuts`).
* :func:`poisson_transmission_terms` -- per-bin terms of the exact-Poisson
  transmission log-likelihood.
* :class:`RadonObject`, :class:`RadonObjectSequence`, :class:`RadonDisk`,
  :class:`RadonSquare` -- analytic 2-D phantoms with known Radon transforms
  (used by the FBP example).
* :func:`neighbor_offsets`, :func:`neighbor_difference_and_sum`,
  :func:`neighbor_product` and the :class:`SmoothFunction` / :class:`RDP`
  priors used by the reconstruction-algorithm examples.
"""

from __future__ import annotations

import abc
import itertools
import math
from collections.abc import Sequence
from importlib import import_module, util as iutil
from types import ModuleType
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

import array_api_compat
from array_api_compat import device, to_device
import parallelproj_core as ppc
from parallelproj import Array, to_numpy_array

# ---------------------------------------------------------------------------
# array_utils
# ---------------------------------------------------------------------------


def suggest_array_backend_and_device(
    backend: str | None = None,
    dev: str | None = None,
) -> tuple[ModuleType, Any]:
    """Select an array API-compatible module and compute device.

    When called without arguments the function probes the environment and
    returns the most capable available backend in the priority order
    ``"torch"`` > ``"cupy"`` > ``"numpy"``, paired with a CUDA device when
    one is available and enabled in ``parallelproj_core``.

    To force a specific backend or device -- for example to benchmark on CPU
    or to reproduce a result without a GPU -- pass explicit ``backend``
    and/or ``dev`` arguments::

        # auto-detect (default)
        xp, dev = suggest_array_backend_and_device()

        # force NumPy on CPU regardless of available hardware
        xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")

        # force PyTorch on CPU
        xp, dev = suggest_array_backend_and_device(backend="torch", dev="cpu")

    Parameters
    ----------
    backend : {"torch", "cupy", "numpy"} or None, optional
        Name of the desired array backend.  ``None`` (default) triggers
        automatic selection.  Raises :class:`ValueError` for an
        unrecognised string and :class:`ImportError` if the requested
        backend is not installed.
    dev : str or None, optional
        Compute device string, e.g. ``"cpu"`` or ``"cuda"``.  ``None``
        (default) lets the function choose the best available device for
        the selected backend.  Note that ``"cupy"`` only supports CUDA
        devices; combining ``backend="cupy"`` with ``dev="cpu"`` raises
        :class:`ValueError`.  To select a specific CUDA device for
        cupy pass ``"cuda:N"`` (e.g. ``"cuda:1"``); plain ``"cuda"``
        selects device 0.

    Returns
    -------
    xp : module
        Array API-compatible namespace (``array_api_compat.torch``,
        ``array_api_compat.cupy``, or ``array_api_compat.numpy``).
    dev : str or cupy.cuda.Device
        Compute device compatible with the returned ``xp``.  For the
        ``"cupy"`` backend this is a :class:`cupy.cuda.Device` object;
        for ``"torch"`` and ``"numpy"`` it is a plain string
        (``"cuda"`` or ``"cpu"``).

    Raises
    ------
    ValueError
        If ``backend`` is not one of ``"torch"``, ``"cupy"``,
        ``"numpy"``, or if ``backend="cupy"`` is combined with
        ``dev="cpu"``.
    ImportError
        If the explicitly requested ``backend`` is not installed.
    """
    _valid = ("torch", "cupy", "numpy")
    if backend is not None and backend not in _valid:
        raise ValueError(f"Unknown backend {backend!r}. Choose from {_valid}.")

    # ------------------------------------------------------------------ torch
    if backend == "torch" or (backend is None and iutil.find_spec("torch") is not None):
        if iutil.find_spec("torch") is None:
            raise ImportError("Backend 'torch' was requested but is not installed.")
        xp = import_module("array_api_compat.torch")
        if dev is None:
            dev = "cuda" if xp.cuda.is_available() and ppc.cuda_enabled == 1 else "cpu"

    # ------------------------------------------------------------------ cupy
    elif backend == "cupy" or (
        backend is None
        and iutil.find_spec("cupy") is not None
        and ppc.cupy_enabled == 1
    ):
        if iutil.find_spec("cupy") is None:
            raise ImportError("Backend 'cupy' was requested but is not installed.")
        if dev == "cpu":
            raise ValueError("cupy only supports CUDA devices; dev='cpu' is not valid.")
        xp = import_module("array_api_compat.cupy")
        if dev is None:
            dev = xp.cuda.Device(0)
        elif isinstance(dev, str):
            # accept "cuda" -> Device(0) or "cuda:N" -> Device(N)
            device_id = int(dev.split(":")[-1]) if ":" in dev else 0
            dev = xp.cuda.Device(device_id)

    # ----------------------------------------------------------------- numpy
    else:
        xp = import_module("array_api_compat.numpy")
        if dev is None:
            dev = "cpu"

    print(f"Using array API: {xp.__name__}, device: {dev}")
    return xp, dev


# ---------------------------------------------------------------------------
# img
# ---------------------------------------------------------------------------


def elliptic_cylinder_phantom(
    xp: ModuleType,
    dev: Any,
    image_shape: tuple[int, int, int] = (40, 40, 8),
    voxel_size: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> Array:
    """3D elliptic cylinder phantom with hot and cold spherical inserts.

    The cylinder has an elliptical cross-section and is oriented along the
    third axis (axis 2).  Six spherical inserts of varying radii and
    contrasts are embedded inside the uniform cylinder background.

    The coordinate origin is at the centre of the image volume.  All
    geometric parameters (cylinder semi-axes, insert positions and radii)
    scale automatically with ``image_shape`` and ``voxel_size``, so the
    phantom looks the same regardless of resolution.

    Parameters
    ----------
    xp:
        Array API-compatible namespace (e.g. ``array_api_compat.torch`` or
        ``array_api_compat.numpy``).
    dev:
        Device passed to array-creation functions (e.g. ``"cpu"`` or
        ``"cuda"``).
    image_shape:
        Number of voxels along each axis ``(n0, n1, n2)``.
        Default: ``(40, 40, 8)``.
    voxel_size:
        Physical voxel size along each axis.
        Default: ``(2.0, 2.0, 2.0)``.

    Returns
    -------
    Array
        ``float32`` array of shape ``image_shape`` on device ``dev``.

        * Outside the cylinder: ``0.0``
        * Cylinder background: ``1.0``
        * Hot inserts: ``> 1.0`` (contrasts 2x, 4x, 8x)
        * Cold inserts: ``< 1.0`` (contrasts 0x, 0.25x, 0.5x)
    """
    n0, n1, n2 = image_shape
    v0, v1, v2 = voxel_size

    fov0 = n0 * v0
    fov1 = n1 * v1
    fov2 = n2 * v2

    x0 = (xp.arange(n0, device=dev, dtype=xp.float32) - (n0 - 1) / 2.0) * v0
    x1 = (xp.arange(n1, device=dev, dtype=xp.float32) - (n1 - 1) / 2.0) * v1
    x2 = (xp.arange(n2, device=dev, dtype=xp.float32) - (n2 - 1) / 2.0) * v2

    X0, X1, X2 = xp.meshgrid(x0, x1, x2, indexing="ij")

    a0 = 0.45 * fov0
    a1 = 0.30 * fov1
    hz = 0.50 * fov2

    inside_cyl = ((X0 / a0) ** 2 + (X1 / a1) ** 2 <= 1.0) & (xp.abs(X2) <= hz)
    img = xp.astype(inside_cyl, xp.float32)

    r_ref = min(a0, a1)

    inserts: list[tuple[float, float, float, float, float]] = [
        (0.50, 0.00, 0.00, 0.08, 2.0),
        (-0.50, 0.00, 0.30, 0.12, 1.5),
        (0.00, 0.50, -0.30, 0.18, 3.0),
        (-0.45, 0.00, -0.30, 0.08, 0.5),
        (0.30, -0.50, 0.00, 0.18, 0.0),
        (0.00, -0.45, 0.30, 0.12, 0.25),
    ]

    for c0f, c1f, c2f, r_frac, value in inserts:
        cx0 = c0f * a0
        cx1 = c1f * a1
        cx2 = c2f * hz
        radius = r_frac * r_ref
        dist2 = (X0 - cx0) ** 2 + (X1 - cx1) ** 2 + (X2 - cx2) ** 2
        fill = xp.full(image_shape, value, dtype=xp.float32, device=dev)
        img = xp.where(dist2 <= radius**2, fill, img)

    return img


# ---------------------------------------------------------------------------
# vis
# ---------------------------------------------------------------------------


def show_vol_cuts(
    vol: "np.ndarray | Array",
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis_labels: tuple[str, ...] | None = None,
    fig_title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greys",
    origin: str | tuple[str, str, str] = ("lower", "lower", "upper"),
    figsize: tuple[int, int] = (12, 6),
) -> tuple:
    """Display interactive 2D cuts through a 3D or 4D array.

    For a 3D array the three panels show cuts perpendicular to each of the
    three axes.  For a 4D array the leading axis (e.g. time or TOF bin) gets
    an additional full-width slider; the three panels still show spatial cuts
    at the currently selected leading-axis index.

    Parameters
    ----------
    vol : array, shape (nx, ny, nz) or (n0, nx, ny, nz)
        3-D or 4-D array to visualise.  May be a NumPy array or any array-API
        array (NumPy, CuPy, PyTorch, ...), including one on a GPU -- it is
        converted to a host NumPy array via ``parallelproj.to_numpy_array``.
    voxel_size : tuple of three floats, optional
        Physical size ``(dx, dy, dz)`` of one voxel along the last three axes.
        Default ``(1, 1, 1)``.
    axis_labels : tuple of str or None, optional
        Label for each axis, e.g. ``("x", "y", "z")`` (3-D) or
        ``("t", "x", "y", "z")`` (4-D).
    fig_title : str or None, optional
        Overall figure title.
    vmin, vmax : float or None, optional
        Initial colour scale limits.  Defaults to the global min / max of
        ``vol``.
    cmap : str, optional
        Initial matplotlib colormap name.  Default ``"Greys"``.
    origin : str or tuple of three str, optional
        ``imshow`` origin for each of the three panels.  A single string is
        applied to all panels; a 3-tuple sets them individually.  Default
        ``("lower", "lower", "upper")`` -- the right-most (transaxial ``x``-``y``)
        panel uses ``"upper"`` so the vertical axis ``x1``/``y`` runs top to
        bottom, matching the scanner coordinate convention.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of three Axes (image panels)
    widgets : dict
        All interactive widgets keyed by name.  Keep a reference to prevent
        garbage collection of the callbacks.
    """
    # accept any array-API array (numpy / cupy / torch, incl. on GPU): move it
    # to a host numpy array once, up front, so all downstream slicing / imshow
    # works regardless of the input backend or device.
    if not isinstance(vol, np.ndarray):
        vol = np.asarray(to_numpy_array(vol))

    if vol.ndim not in (3, 4):
        raise ValueError("vol must be a 3-D or 4-D array")

    is_4d = vol.ndim == 4

    if axis_labels is None:
        axis_labels = ("t", "x", "y", "z") if is_4d else ("x", "y", "z")
    elif len(axis_labels) != vol.ndim:
        raise ValueError(
            f"axis_labels must have {vol.ndim} entries for a {vol.ndim}-D array"
        )

    if is_4d:
        n0, nx, ny, nz = vol.shape
        l0, lx, ly, lz = axis_labels
        i00 = n0 // 2
    else:
        nx, ny, nz = vol.shape
        lx, ly, lz = axis_labels
        n0, l0, i00 = None, None, None

    dx, dy, dz = voxel_size

    if vmin is None:
        vmin = float(vol.min())
    if vmax is None:
        vmax = float(vol.max())

    ix0, iy0, iz0 = nx // 2, ny // 2, nz // 2

    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)

    # Some matplotlib versions ship a ``TextBox`` ``resize_event`` handler that
    # assumes a mouse event and reads ``event.inaxes`` -- which a ``ResizeEvent``
    # does not have.  This raises a harmless ``AttributeError`` (printed once per
    # TextBox) on every figure resize.  Filter just that error so it does not
    # spam the console; everything else propagates normally.
    _cb = fig.canvas.callbacks
    _orig_exc_handler = getattr(_cb, "exception_handler", None)

    def _filter_resize_bug(exc, _orig=_orig_exc_handler):
        if isinstance(exc, AttributeError) and "inaxes" in str(exc):
            return
        if _orig is not None:
            return _orig(exc)
        raise exc

    if hasattr(_cb, "exception_handler"):
        _cb.exception_handler = _filter_resize_bug

    if is_4d:
        # rows: images | leading slider (+ its slice-entry) | 3 sliders |
        #       3 slice-entry boxes | vmin/vmax/cmap | colorbar
        gs = fig.add_gridspec(
            6, 3, height_ratios=[8, 1, 1, 1, 1, 0.5], hspace=0.6, wspace=0.35
        )
        ax_s0 = fig.add_subplot(gs[1, 0:2])
        ax_ntb0 = fig.add_subplot(gs[1, 2])  # leading-axis slice entry
        ax_sl = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_ntb = [fig.add_subplot(gs[3, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[4, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[5, :])
    else:
        # rows: images | 3 sliders | 3 slice-entry boxes | vmin/vmax/cmap | colorbar
        gs = fig.add_gridspec(
            5, 3, height_ratios=[8, 1, 1, 1, 0.5], hspace=0.6, wspace=0.35
        )
        ax_s0 = None
        ax_ntb0 = None
        ax_sl = [fig.add_subplot(gs[1, c]) for c in range(3)]
        ax_ntb = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[3, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[4, :])

    ax_im = [fig.add_subplot(gs[0, c]) for c in range(3)]

    if isinstance(origin, str):
        origins = [origin, origin, origin]
    else:
        origins = list(origin)
    if len(origins) != 3:
        raise ValueError("origin must be a string or a sequence of three strings")

    aspects = [dz / dy, dz / dx, dy / dx]
    panel_xlabels = [ly, lx, lx]
    panel_ylabels = [lz, lz, ly]

    # per-panel cut extractor: panel 0 depends on ix, panel 1 on iy, panel 2 on iz
    def _panel_cut(c, i0, ix, iy, iz):
        if is_4d:
            return (vol[i0, ix, :, :], vol[i0, :, iy, :], vol[i0, :, :, iz])[c].T
        return (vol[ix, :, :], vol[:, iy, :], vol[:, :, iz])[c].T

    ims = [
        ax_im[c].imshow(
            _panel_cut(c, i00, ix0, iy0, iz0),
            aspect=aspects[c],
            origin=origins[c],
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        for c in range(3)
    ]

    for c, ax in enumerate(ax_im):
        ax.set_xlabel(panel_xlabels[c])
        ax.set_ylabel(panel_ylabels[c])

    sx = Slider(ax_sl[0], lx, 0, nx - 1, valinit=ix0, valstep=1)
    sy = Slider(ax_sl[1], ly, 0, ny - 1, valinit=iy0, valstep=1)
    sz = Slider(ax_sl[2], lz, 0, nz - 1, valinit=iz0, valstep=1)
    s0 = Slider(ax_s0, l0, 0, n0 - 1, valinit=i00, valstep=1) if is_4d else None

    def _current_indices():
        i0 = int(s0.val) if s0 is not None else None
        return i0, int(sx.val), int(sy.val), int(sz.val)

    # --- fast (blitted) rendering -------------------------------------------
    # Moving a slider re-renders only the affected panel(s), and (when the
    # backend supports it) blits just that panel instead of redrawing the whole
    # figure.  Backgrounds are captured on every full draw (covers resize and
    # colour-scale / colormap changes).
    _use_blit = bool(getattr(fig.canvas, "supports_blit", False))
    _bg = [None, None, None]

    def _capture_backgrounds(_event=None):
        if not _use_blit:
            return
        for c in range(3):
            _bg[c] = fig.canvas.copy_from_bbox(ax_im[c].bbox)

    fig.canvas.mpl_connect("draw_event", _capture_backgrounds)

    def _render(panels):
        i0, ix, iy, iz = _current_indices()
        for c in panels:
            ims[c].set_data(_panel_cut(c, i0, ix, iy, iz))
        if _use_blit and all(_bg[c] is not None for c in panels):
            for c in panels:
                fig.canvas.restore_region(_bg[c])
                ax_im[c].draw_artist(ims[c])
                fig.canvas.blit(ax_im[c].bbox)
        else:
            fig.canvas.draw_idle()

    # axis -> (entry box, slider); filled in when the entry boxes are built
    # below.  Lets slider drags / clicks push the current index into the box.
    _entry_for_axis = {}

    def _sync_entry(axis):
        pair = _entry_for_axis.get(axis)
        if pair is None:
            return
        entry, slider = pair
        txt = str(int(slider.val))
        if entry.text_disp.get_text() != txt:
            # update the text artist directly (no submit, no extra draw); it
            # refreshes on the redraw the slider already schedules
            entry.text_disp.set_text(txt)

    # each spatial slider affects only its own panel (and its own entry box);
    # the leading-axis slider (4-D only) affects all three panels
    def _slider_cb(axis, panels):
        def _cb(_v):
            _render(panels)
            _sync_entry(axis)

        return _cb

    sx.on_changed(_slider_cb(0, [0]))
    sy.on_changed(_slider_cb(1, [1]))
    sz.on_changed(_slider_cb(2, [2]))
    if s0 is not None:
        s0.on_changed(_slider_cb("t", [0, 1, 2]))

    # --- click-to-locate: clicking a panel moves the two orthogonal panels ---
    # index-axis a (0=x, 1=y, 2=z) is the slice axis of panel a, so re-rendering
    # panel a exactly follows a change of index-axis a.
    _slider_for_axis = {0: sx, 1: sy, 2: sz}
    _axis_extent = {0: nx, 1: ny, 2: nz}
    # per panel: (horizontal index-axis, vertical index-axis) of the displayed cut
    _panel_click_axes = {0: (1, 2), 1: (0, 2), 2: (0, 1)}

    def _on_click(event):
        if event.button != 1 or event.inaxes not in ax_im:
            return
        if event.xdata is None or event.ydata is None:
            return
        toolbar = getattr(fig.canvas, "toolbar", None)
        if toolbar is not None and getattr(toolbar, "mode", ""):
            return  # ignore clicks while the zoom / pan tool is active
        c = ax_im.index(event.inaxes)
        col_axis, row_axis = _panel_click_axes[c]
        panels = []
        for axis, coord in ((col_axis, event.xdata), (row_axis, event.ydata)):
            value = max(0, min(_axis_extent[axis] - 1, int(round(coord))))
            slider = _slider_for_axis[axis]
            slider.eventson = False  # set both, then render once
            slider.set_val(value)
            slider.eventson = True
            panels.append(axis)  # index-axis a <-> panel a
        _render(sorted(panels))
        for axis in panels:
            _sync_entry(axis)  # click used eventson=False, so sync boxes here

    fig.canvas.mpl_connect("button_press_event", _on_click)

    # --- direct slice-number entry: type an index and press Enter to jump ----
    def _attach_slice_entry(ax_box, label, slider, n, axis):
        entry = TextBox(ax_box, label, initial=str(int(slider.val)))

        def _submit(text, _entry=entry, _slider=slider, _n=n):
            try:
                v = int(round(float(text)))
            except (ValueError, TypeError):
                v = int(_slider.val)
            v = max(0, min(_n - 1, v))
            _entry.text_disp.set_text(str(v))  # show the clamped value
            # set_val fires the slider callback -> _render + _sync_entry; no
            # entry.set_val() here, so there is no submit-recursion
            _slider.set_val(v)

        entry.on_submit(_submit)
        _entry_for_axis[axis] = (entry, slider)
        return entry

    slice_entries = {
        lx: _attach_slice_entry(ax_ntb[0], lx, sx, nx, 0),
        ly: _attach_slice_entry(ax_ntb[1], ly, sy, ny, 1),
        lz: _attach_slice_entry(ax_ntb[2], lz, sz, nz, 2),
    }
    if is_4d:
        slice_entries[l0] = _attach_slice_entry(ax_ntb0, l0, s0, n0, "t")

    cb = fig.colorbar(ims[0], cax=ax_cb, orientation="horizontal")

    tb_vmin = TextBox(ax_tb[0], "vmin", initial=f"{vmin:.4g}")
    tb_vmax = TextBox(ax_tb[1], "vmax", initial=f"{vmax:.4g}")

    def update_clim(_):
        try:
            lo = float(tb_vmin.text)
            hi = float(tb_vmax.text)
        except ValueError:
            return
        for im in ims:
            im.set_clim(lo, hi)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_vmin.on_submit(update_clim)
    tb_vmax.on_submit(update_clim)

    tb_cmap = TextBox(ax_tb[2], "cmap", initial=cmap)

    def update_cmap(name):
        try:
            cm = plt.get_cmap(name.strip())
        except (ValueError, KeyError):
            tb_cmap.set_val(ims[0].cmap.name)
            return
        for im in ims:
            im.set_cmap(cm)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_cmap.on_submit(update_cmap)

    widgets = {
        f"s_{lx}": sx,
        f"s_{ly}": sy,
        f"s_{lz}": sz,
        "tb_vmin": tb_vmin,
        "tb_vmax": tb_vmax,
        "tb_cmap": tb_cmap,
        # keep a strong ref to the click callback (also handy for tests)
        "on_click": _on_click,
    }
    for _lab, _entry in slice_entries.items():
        widgets[f"e_{_lab}"] = _entry
    if s0 is not None:
        widgets[f"s_{l0}"] = s0

    return fig, ax_im, widgets


#: Backward-compatible alias.
show_3d_cuts = show_vol_cuts


# ---------------------------------------------------------------------------
# transmission (exact Poisson model) helper
# ---------------------------------------------------------------------------


def poisson_transmission_terms(
    att_line_integral, blank, contamination, data, *, tof_sum=False
):
    r"""Per-bin terms of the *exact Poisson* transmission log-likelihood.

    For the transmission model

    .. math::
        \bar{y} = \bar z + s, \qquad \bar z = b\,e^{-\ell},

    with line integral :math:`\ell`, blank :math:`b`, contamination
    :math:`s` and measured counts :math:`y`, this returns the ingredients
    shared by the MLTR / SPS / MLAA updates (see ``00_mltr_sps.py`` for the
    explicit math).  It is the *exact Poisson* model -- no log-linearisation
    into a weighted-least-squares problem.

    In MLAA the "blank" is the (TOF) activity forward projection
    :math:`P\lambda`; pass ``tof_sum=True`` so the per-TOF-bin gradient and
    curvature are summed over the trailing TOF axis before back-projecting
    through the non-TOF attenuation projector (``att_line_integral`` is then
    the geometric line integral, broadcast over the TOF axis).

    Parameters
    ----------
    att_line_integral : Array
        Attenuation line integral :math:`\ell = P\mu` (geometric).
    blank : Array
        Blank scan :math:`b` (transmission) or activity forward projection
        :math:`P\lambda` (MLAA).  May carry a trailing TOF axis.
    contamination : Array
        Additive contamination :math:`s` (scatter + randoms), > 0.
    data : Array
        Measured counts :math:`y`.
    tof_sum : bool, optional
        If ``True``, sum the gradient and curvature over the trailing (TOF)
        axis.  Defaults to ``False``.

    Returns
    -------
    expected_counts : Array
        :math:`\bar{y} = b\,e^{-\ell} + s` (full per-bin shape).
    gradient_sino : Array
        :math:`\tfrac{\bar z}{\bar y}(\bar y - y)` -- back-project this to
        get the ascent gradient :math:`\nabla_\mu L`.  TOF-summed if
        ``tof_sum``.
    curvature_sino : Array
        MLTR / SPS separable curvature :math:`\bar z^2/\bar y`; the
        preconditioner denominator is ``P^T[(P 1) * curvature_sino]``.
        TOF-summed if ``tof_sum``.
    """
    xp = array_api_compat.get_namespace(att_line_integral)
    att = xp.exp(-att_line_integral)
    if tof_sum:
        att = att[..., None]  # broadcast the geometric attenuation over TOF
    psi = blank * att
    ybar = psi + contamination
    gradient_sino = psi / ybar * (ybar - data)
    curvature_sino = psi**2 / ybar
    if tof_sum:
        gradient_sino = xp.sum(gradient_sino, axis=-1)
        curvature_sino = xp.sum(curvature_sino, axis=-1)
    return ybar, gradient_sino, curvature_sino


class RadonObject(abc.ABC):
    """abstract base class for objects with known radon transform"""

    def __init__(self, xp: ModuleType, dev: str) -> None:
        self._xp = xp
        self._dev = dev

        self._x0_offset: float = 0.0
        self._x1_offset: float = 0.0
        self._s0: float = 1.0
        self._s1: float = 1.0
        self._amplitude: float = 1.0
        self._rotation: float = 0.0

    @abc.abstractmethod
    def _centered_radon_transform(self, r: Array, phi: Array) -> Array:
        pass

    @abc.abstractmethod
    def _centered_values(self, x0: Array, x1: Array) -> Array:
        pass

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @property
    def dev(self) -> str:
        return self._dev

    @property
    def x0_offset(self) -> float:
        return self._x0_offset

    @x0_offset.setter
    def x0_offset(self, value: float) -> None:
        self._x0_offset = value

    @property
    def x1_offset(self) -> float:
        return self._x1_offset

    @x1_offset.setter
    def x1_offset(self, value: float) -> None:
        self._x1_offset = value

    @property
    def s0(self) -> float:
        return self._s0

    @s0.setter
    def s0(self, value: float) -> None:
        self._s0 = value

    @property
    def s1(self) -> float:
        return self._s1

    @s1.setter
    def s1(self, value: float) -> None:
        self._s1 = value

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude = value

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        self._rotation = value

    def radon_transform(self, s, phi) -> Array:

        phi_rotated = (phi + self.rotation) % (2 * self.xp.pi)

        s_prime = s / self.xp.sqrt(
            self._s0**2 * self.xp.cos(phi_rotated) ** 2
            + self._s1**2 * self.xp.sin(phi_rotated) ** 2
        )
        phi_prime = self.xp.atan2(
            self._s0 * self.xp.cos(phi_rotated),
            self._s1 * self.xp.sin(phi_rotated),
        )

        fac = (
            self._s0
            * self._s1
            / self.xp.sqrt(
                self._s0**2 * self.xp.cos(phi_rotated) ** 2
                + self._s1**2 * self.xp.sin(phi_rotated) ** 2
            )
        )

        return (
            self._amplitude
            * fac
            * self._centered_radon_transform(
                s_prime
                - self._x0_offset * self.xp.sin(phi_prime)
                - self._x1_offset * self.xp.cos(phi_prime),
                phi_prime,
            )
        )

    def values(self, x0: Array, x1: Array) -> Array:
        x0_p = x0 * math.cos(self._rotation) - x1 * math.sin(self._rotation)
        x1_p = x0 * math.sin(self._rotation) + x1 * math.cos(self._rotation)

        x0_pp = x0_p / self._s0 - self._x0_offset
        x1_pp = x1_p / self._s1 - self._x1_offset

        return self._amplitude * self._centered_values(x0_pp, x1_pp)


class RadonObjectSequence(Sequence[RadonObject]):
    def __init__(self, objects: Sequence[RadonObject]) -> None:
        super().__init__()
        self._objects: Sequence[RadonObject] = objects

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, i: int) -> RadonObject:
        return self._objects[i]

    def radon_transform(self, r, phi) -> float:
        return sum([x.radon_transform(r, phi) for x in self])

    def values(self, x0: Array, x1: Array) -> Array:
        return sum([x.values(x0, x1) for x in self])


class RadonDisk(RadonObject):
    """2D disk with known radon transform"""

    def __init__(self, xp: ModuleType, dev: str, radius: float) -> None:
        super().__init__(xp, dev)
        self._radius: float = radius

    def _centered_radon_transform(self, r: Array, phi: Array) -> Array:
        mask = self.xp.abs(r) <= self._radius
        safe = self.xp.where(mask, self._radius**2 - r**2, self.xp.zeros_like(r))
        return self.xp.where(mask, 2 * self.xp.sqrt(safe), self.xp.zeros_like(r))

    def _centered_values(self, x0: Array, x1: Array) -> Array:
        return self.xp.where(
            x0**2 + x1**2 <= self._radius**2,
            self.xp.ones_like(x0),
            self.xp.zeros_like(x0),
        )

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = value


class RadonSquare(RadonObject):
    """2D square with known radon transform"""

    def __init__(self, xp: ModuleType, dev: str, width: float) -> None:
        super().__init__(xp, dev)
        self._width: float = width
        self._eps = 1e-6

    def _centered_radon_transform(self, r: Array, phi: Array) -> Array:
        # calculate the angle alpha which is "distance to pi/4"
        f1 = phi // (self.xp.pi / 4)
        f2 = f1 % 2
        m = -(2 * f2 - 1)
        alpha = m * (phi - f1 * self.xp.pi / 4) + f2 * self.xp.pi / 4

        abs_r = self.xp.abs(r)

        # height of the plateau
        h = 2 * self.a / self.xp.cos(alpha)

        sqrt2 = float(self.xp.sqrt(self.xp.asarray(2.0)))

        # r2 is the distance until we have the plateau
        r2 = sqrt2 * self.a * self.xp.cos(self.xp.pi / 4 + alpha)
        # between r2 and r1 we have the triangle part
        r1 = sqrt2 * self.a * self.xp.cos(self.xp.pi / 4 - alpha)

        mask1 = self.xp.where(
            (r1 - r2) > self.eps, self.xp.ones_like(r), self.xp.zeros_like(r)
        )
        mask2 = self.xp.where(abs_r >= r2, self.xp.ones_like(r), self.xp.zeros_like(r))
        mask3 = self.xp.where(abs_r < r1, self.xp.ones_like(r), self.xp.zeros_like(r))

        # triangle mask
        t_mask = mask1 * mask2 * mask3

        # plateau apart
        p1 = self.xp.where(abs_r < r2, h * self.xp.ones_like(r), self.xp.zeros_like(r))

        # triangle part avoiding division by zero
        denom = self.xp.where(
            r1 - r2 > self.eps, r1 - r2, self.eps * self.xp.ones_like(r)
        )
        tmp = -self.xp.sign(r) * h * self.xp.divide(r - self.xp.sign(r) * r2, denom) + h
        p2 = self.xp.where(t_mask == 1, tmp, self.xp.zeros_like(r))

        return p1 + p2

    def _centered_values(self, x0: Array, x1: Array) -> Array:
        return self.xp.where(
            self.xp.logical_and(self.xp.abs(x0) <= self.a, self.xp.abs(x1) <= self.a),
            self.xp.ones_like(x0),
            self.xp.zeros_like(x0),
        )

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = value

    @property
    def a(self) -> float:
        return self._width / 2

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        self._eps = value


def neighbor_offsets(ndim):
    # all offsets in {-1,0,1}^ndim except (0,...,0), as Python int tuples
    return [
        tuple(v - 1 for v in ind)
        for ind in itertools.product(range(3), repeat=ndim)
        if any(v != 1 for v in ind)
    ]


def neighbor_difference_and_sum(x, xp):
    """
    x: array_api-compatible array, shape (n1, n2, ..., nD)
    returns:
        d, s: shape (num_neigh, n1, n2, ..., nD)
              differences and sums with all nearest neighbors.
              If a neighbor is outside the image, the corresponding
              entry is 0 by definition.
    """
    ndim = x.ndim
    shape = x.shape
    offsets = neighbor_offsets(ndim)  # list of (num_neigh,) int tuples
    num_neigh = len(offsets)

    dev = device(x)

    d = xp.zeros((num_neigh,) + shape, dtype=x.dtype, device=dev)
    s = xp.zeros((num_neigh,) + shape, dtype=x.dtype, device=dev)

    for k in range(num_neigh):
        off = offsets[k]  # e.g. [-1, 0, 1, ...], shape (ndim,)

        # build slices for "center" and "neighbor" so that
        # center[...] and neighbor[...] have exactly the same shape
        center_sl = []
        neigh_sl = []
        for _ax, (n, o) in enumerate(zip(shape, off)):
            if o == 0:
                # full axis in both
                center_sl.append(slice(0, n))
                neigh_sl.append(slice(0, n))
            elif o > 0:
                # neighbor is shifted +o -> valid center is [0 : n-o]
                center_sl.append(slice(0, n - o))
                neigh_sl.append(slice(o, n))
            else:  # o < 0
                # neighbor is shifted -|o| -> valid center is [|o| : n]
                o_abs = -o
                center_sl.append(slice(o_abs, n))
                neigh_sl.append(slice(0, n - o_abs))

        center_sl = tuple(center_sl)
        neigh_sl = tuple(neigh_sl)

        # view with aligned shapes
        xc = x[center_sl]
        xn = x[neigh_sl]

        # place results into the corresponding region of d[k], s[k]
        # outside that region, d and s remain zero
        d_view = d[(k,) + center_sl]
        s_view = s[(k,) + center_sl]

        d_view[...] = xc - xn
        s_view[...] = xc + xn

    return d, s


def neighbor_product(x, xp):
    """
    Same neighbor definition as neighbor_difference_and_sum, but returns
    products x * neighbor. Products with neighbors outside the image are 0.
    """
    ndim = x.ndim
    shape = x.shape
    offsets = neighbor_offsets(ndim)
    num_neigh = len(offsets)
    dev = device(x)

    p = xp.zeros((num_neigh,) + shape, dtype=x.dtype, device=dev)

    for k in range(num_neigh):
        off = offsets[k]

        center_sl = []
        neigh_sl = []
        for _ax, (n, o) in enumerate(zip(shape, off)):
            if o == 0:
                center_sl.append(slice(0, n))
                neigh_sl.append(slice(0, n))
            elif o > 0:
                center_sl.append(slice(0, n - o))
                neigh_sl.append(slice(o, n))
            else:  # o < 0
                o_abs = -o
                center_sl.append(slice(o_abs, n))
                neigh_sl.append(slice(0, n - o_abs))

        center_sl = tuple(center_sl)
        neigh_sl = tuple(neigh_sl)

        xc = x[center_sl]
        xn = x[neigh_sl]

        p_view = p[(k,) + center_sl]
        p_view[...] = xc * xn

    return p


class SmoothFunction(abc.ABC):

    def __init__(self, in_shape, xp, dev, scale: float = 1.0) -> None:

        self._in_shape = in_shape
        self._scale = scale
        self._xp = xp
        self._dev = dev

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def xp(self):
        return self._xp

    @property
    def dev(self):
        return self._dev

    @abc.abstractmethod
    def _call(self, x: Array) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _gradient(self, x: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._call(x)
        else:
            res = self._scale * self._call(x)

        return res

    def gradient(self, x: Array) -> Array:
        dev_input = device(x)

        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._gradient(x)
        else:
            res = self._scale * self._gradient(x)

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        res = to_device(res, dev_input)

        return res

    def prox_function(self, z: Array, x: Array, T: Array) -> float:
        r"""returns the function h(z) = \sum_i f_i(z) + 0.5 * \sum_i (z_i - x_i)^2 / T_i
        which when minimized over z is the proximal operator of the function at x
        """
        return self.__call__(z) + 0.5 * float((((z - x) ** 2) / T).sum())

    def prox_gradient(self, z: Array, x: Array, T: Array) -> Array:
        """return the gradient of the prox function h(z), needed for numeric evaluation of the proximal operator"""
        return self.gradient(z) + (z - x) / T


class SmoothFunctionWithDiagonalHessian(SmoothFunction):
    @abc.abstractmethod
    def _diag_hessian(self, x: Array) -> Array:
        """(approximation) of the diagonal of the Hessian"""
        raise NotImplementedError

    def diag_hessian(self, x: Array) -> Array:
        dev_input = device(x)

        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._diag_hessian(x)
        else:
            res = self._scale * self._diag_hessian(x)

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        res = to_device(res, dev_input)

        return res


class RDP(SmoothFunctionWithDiagonalHessian):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        xp: ModuleType,
        dev: str,
        voxel_size: Array,
        eps: float | None = None,
        gamma: float = 2.0,
    ) -> None:
        self._gamma = gamma

        if eps is None:
            self._eps = xp.finfo(xp.float32).eps
        else:
            self._eps = eps

        self._ndim = len(in_shape)

        if self._ndim != 3:
            raise ValueError("RDP only implemented for 3D images")

        super().__init__(in_shape=in_shape, xp=xp, dev=dev)

        # number of nearest neighbors
        self._num_neigh = 3**self._ndim - 1

        self._voxel_size = voxel_size

        # array for differences and sums with nearest neighbors
        self._voxel_size_weights = xp.zeros(
            (self._num_neigh,) + in_shape, dtype=xp.float32, device=dev
        )

        for i, ind in enumerate(itertools.product(range(3), repeat=self._ndim)):
            if i != (self._num_neigh // 2):
                offset = xp.asarray(ind, device=dev) - 1

                # this only works for 3D
                vw = voxel_size[2] / xp.linalg.norm(offset * voxel_size)

                if i < self._num_neigh // 2:
                    self._voxel_size_weights[i, ...] = vw
                else:
                    self._voxel_size_weights[i - 1, ...] = vw

        self._weights = self._voxel_size_weights
        self._kappa = None

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def weights(self) -> Array:
        return self._weights

    @property
    def kappa(self) -> Array | None:
        return self._kappa

    @kappa.setter
    def kappa(self, image: Array) -> None:
        self._kappa = image
        self._weights = (
            neighbor_product(self._kappa, self._xp) * self._voxel_size_weights
        )

    def _call(self, x: Array) -> float:

        if float(self.xp.min(x)) < 0:
            return self.xp.inf

        d, s = neighbor_difference_and_sum(x, self.xp)
        # phi = s + self.gamma * self.xp.abs(d) + self.eps
        phi = self.xp.abs(d)
        phi *= self.gamma
        phi += s
        phi += self.eps

        # tmp = (d**2) / phi
        # reuse d to save memory
        tmp = d
        tmp *= tmp
        tmp /= phi

        if self._weights is not None:
            tmp *= self._weights

        return 0.5 * float(self.xp.sum(tmp, dtype=self.xp.float64))

    def _gradient(self, x: Array) -> Array:
        d, s = neighbor_difference_and_sum(x, self.xp)
        # phi = s + self.gamma * self.xp.abs(d) + self.eps
        tmp = self.xp.abs(d)
        tmp *= self.gamma
        phi = s + tmp
        phi += self.eps

        # tmp = d * (2 * phi - (d + self.gamma * self.xp.abs(d))) / (phi**2)
        tmp += d
        tmp -= phi
        tmp -= phi
        tmp *= -1
        tmp *= d
        tmp /= phi
        tmp /= phi

        if self._weights is not None:
            tmp *= self._weights

        return tmp.sum(axis=0)

    def _diag_hessian(self, x: Array) -> Array:
        d, s = neighbor_difference_and_sum(x, self.xp)
        # phi = s + self.gamma * self.xp.abs(d) + self.eps
        phi = self.xp.abs(d)
        phi *= self.gamma
        phi += s
        phi += self.eps

        # tmp = ((s - d + self.eps) ** 2) / (phi**3)
        tmp = s - d
        tmp += self.eps
        tmp *= tmp
        tmp /= phi
        tmp /= phi
        tmp /= phi

        if self._weights is not None:
            tmp *= self._weights

        return 2 * tmp.sum(axis=0)
