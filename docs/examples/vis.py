"""Shared visualisation utilities for parallelproj examples."""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox


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

    Controls below the images:

    * One slider per axis to scroll through cuts.
    * Two text boxes to update the colour scale limits (vmin / vmax).
    * One text box to change the colormap (any valid matplotlib name).
    * A horizontal colorbar that reflects the current clim and colormap.

    Non-uniform voxel sizes are respected via the ``voxel_size`` parameter.

    Parameters
    ----------
    vol : np.ndarray, shape (nx, ny, nz) or (n0, nx, ny, nz)
        3-D or 4-D array to visualise.
    voxel_size : tuple of three floats, optional
        Physical size ``(dx, dy, dz)`` of one voxel along the last three axes.
        Default ``(1, 1, 1)``.
    axis_labels : tuple of str or None, optional
        Label for each axis, e.g. ``("x", "y", "z")`` (3-D) or
        ``("t", "x", "y", "z")`` (4-D).  Defaults to ``("x", "y", "z")``
        for 3-D and ``("t", "x", "y", "z")`` for 4-D.
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
        All interactive widgets keyed by name.  Slider keys follow the
        pattern ``"s_{label}"`` (e.g. ``"s_x"``); text-box keys are
        ``"tb_vmin"``, ``"tb_vmax"``, ``"tb_cmap"``.  Keep a reference to
        this dict to prevent garbage collection of the callbacks.
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

    # --- figure layout via GridSpec ---
    # Always:   row 0      = images (3 panels)
    #           row -3     = spatial slice sliders (3 columns)
    #           row -2     = vmin / vmax / cmap text boxes (3 columns)
    #           row -1     = horizontal colorbar (spans all columns)
    # 4-D only: row 1      = leading-axis slider (spans all columns)
    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)

    if is_4d:
        gs = fig.add_gridspec(
            5,
            3,
            height_ratios=[8, 1, 1, 1, 0.5],
            hspace=0.6,
            wspace=0.35,
        )
        ax_s0 = fig.add_subplot(gs[1, :])  # leading-axis slider spans all cols
        ax_sl = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[3, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[4, :])
    else:
        gs = fig.add_gridspec(
            4,
            3,
            height_ratios=[8, 1, 1, 0.5],
            hspace=0.6,
            wspace=0.35,
        )
        ax_s0 = None
        ax_sl = [fig.add_subplot(gs[1, c]) for c in range(3)]
        ax_tb = [fig.add_subplot(gs[2, c]) for c in range(3)]
        ax_cb = fig.add_subplot(gs[3, :])

    ax_im = [fig.add_subplot(gs[0, c]) for c in range(3)]

    # --- images ---
    # The 3 panels always show cuts through the last 3 (spatial) axes:
    #   panel 0: cut along lx -> shows (ly, lz) plane, aspect = dz / dy
    #   panel 1: cut along ly -> shows (lx, lz) plane, aspect = dz / dx
    #   panel 2: cut along lz -> shows (lx, ly) plane, aspect = dy / dx
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

    # --- sliders ---
    sx = Slider(ax_sl[0], lx, 0, nx - 1, valinit=ix0, valstep=1)
    sy = Slider(ax_sl[1], ly, 0, ny - 1, valinit=iy0, valstep=1)
    sz = Slider(ax_sl[2], lz, 0, nz - 1, valinit=iz0, valstep=1)
    # for 4D: one extra full-width slider for the leading axis
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

    # --- colorbar ---
    cb = fig.colorbar(ims[0], cax=ax_cb, orientation="horizontal")

    # --- vmin / vmax text boxes ---
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

    # --- colormap text box ---
    tb_cmap = TextBox(ax_tb[2], "cmap", initial=cmap)

    def update_cmap(name):
        try:
            cm = plt.get_cmap(name.strip())
        except (ValueError, KeyError):
            tb_cmap.set_val(ims[0].cmap.name)  # revert to current
            return
        for im in ims:
            im.set_cmap(cm)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_cmap.on_submit(update_cmap)

    # widget keys use the axis labels so callers can look up by name
    widgets = {
        f"s_{lx}": sx,
        f"s_{ly}": sy,
        f"s_{lz}": sz,
        "tb_vmin": tb_vmin,
        "tb_vmax": tb_vmax,
        "tb_cmap": tb_cmap,
    }
    if s0 is not None:
        widgets[f"s_{l0}"] = s0

    return fig, ax_im, widgets


#: Backward-compatible alias.
show_3d_cuts = show_vol_cuts
