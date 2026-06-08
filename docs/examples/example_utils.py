"""Example helper utilities for the parallelproj gallery.

This module is **not** part of the parallelproj package.  It is a
convenience helper for the Sphinx-Gallery examples only and lives under
``docs/examples/``.  It is excluded from the gallery by the
``ignore_pattern`` in ``docs/conf.py``.

Contents
--------
* :func:`suggest_array_backend_and_device` -- select the best available
  array backend and compute device.
* :func:`elliptic_cylinder_phantom` -- 3-D test phantom with spherical inserts.
* :func:`show_vol_cuts` -- interactive 2-D slice viewer for 3-D/4-D arrays.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# array_utils
# ---------------------------------------------------------------------------

from importlib import import_module, util as iutil
from types import ModuleType
from typing import Any

import parallelproj_core as ppc


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

from parallelproj import Array  # noqa: E402


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

import numpy as np  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.widgets import Slider, TextBox  # noqa: E402


def show_vol_cuts(
    vol: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    axis_labels: tuple[str, ...] | None = None,
    fig_title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greys",
    figsize: tuple[int, int] = (12, 6),
) -> tuple:
    """Display interactive 2D cuts through a 3D or 4D array.

    For a 3D array the three panels show cuts perpendicular to each of the
    three axes.  For a 4D array the leading axis (e.g. time or TOF bin) gets
    an additional full-width slider; the three panels still show spatial cuts
    at the currently selected leading-axis index.

    Parameters
    ----------
    vol : np.ndarray, shape (nx, ny, nz) or (n0, nx, ny, nz)
        3-D or 4-D array to visualise.
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

    if is_4d:
        gs = fig.add_gridspec(5, 3, height_ratios=[8, 1, 1, 1, 0.5], hspace=0.6, wspace=0.35)
        ax_s0 = fig.add_subplot(gs[1, :])
        ax_sl = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[3, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[4, :])
    else:
        gs = fig.add_gridspec(4, 3, height_ratios=[8, 1, 1, 0.5], hspace=0.6, wspace=0.35)
        ax_s0 = None
        ax_sl = [fig.add_subplot(gs[1, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[3, :])

    ax_im = [fig.add_subplot(gs[0, c]) for c in range(3)]

    aspects = [dz / dy, dz / dx, dy / dx]
    panel_xlabels = [ly, lx, lx]
    panel_ylabels = [lz, lz, ly]

    def _get_cuts(i0, ix, iy, iz):
        if is_4d:
            return vol[i0, ix, :, :].T, vol[i0, :, iy, :].T, vol[i0, :, :, iz].T
        return vol[ix, :, :].T, vol[:, iy, :].T, vol[:, :, iz].T

    c0, c1, c2 = _get_cuts(i00, ix0, iy0, iz0)
    imkw = dict(vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    ims = [
        ax_im[0].imshow(c0, aspect=aspects[0], **imkw),
        ax_im[1].imshow(c1, aspect=aspects[1], **imkw),
        ax_im[2].imshow(c2, aspect=aspects[2], **imkw),
    ]

    for i, ax in enumerate(ax_im):
        ax.set_xlabel(panel_xlabels[i])
        ax.set_ylabel(panel_ylabels[i])
    ax_im[0].set_title(f"{lx} = {ix0}")
    ax_im[1].set_title(f"{ly} = {iy0}")
    ax_im[2].set_title(f"{lz} = {iz0}")

    sx = Slider(ax_sl[0], lx, 0, nx - 1, valinit=ix0, valstep=1)
    sy = Slider(ax_sl[1], ly, 0, ny - 1, valinit=iy0, valstep=1)
    sz = Slider(ax_sl[2], lz, 0, nz - 1, valinit=iz0, valstep=1)
    s0 = Slider(ax_s0, l0, 0, n0 - 1, valinit=i00, valstep=1) if is_4d else None

    def update_slices(_):
        i0 = int(s0.val) if s0 is not None else None
        ix, iy, iz = int(sx.val), int(sy.val), int(sz.val)
        cuts = _get_cuts(i0, ix, iy, iz)
        for j, im in enumerate(ims):
            im.set_data(cuts[j])
        ax_im[0].set_title(f"{lx} = {ix}")
        ax_im[1].set_title(f"{ly} = {iy}")
        ax_im[2].set_title(f"{lz} = {iz}")
        fig.canvas.draw_idle()

    sx.on_changed(update_slices)
    sy.on_changed(update_slices)
    sz.on_changed(update_slices)
    if s0 is not None:
        s0.on_changed(update_slices)

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
        f"s_{lx}": sx, f"s_{ly}": sy, f"s_{lz}": sz,
        "tb_vmin": tb_vmin, "tb_vmax": tb_vmax, "tb_cmap": tb_cmap,
    }
    if s0 is not None:
        widgets[f"s_{l0}"] = s0

    return fig, ax_im, widgets


#: Backward-compatible alias.
show_3d_cuts = show_vol_cuts
