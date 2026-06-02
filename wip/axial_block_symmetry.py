"""Visualise axial sinogram plane symmetries for a PET scanner with cylindrical symmetry."""

import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button, TextBox

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinogram_plane_symmetries import (
    is_interior_ring,
    plane_orbit,
    compute_sinogram_plane_symmetries,
)

# ── Cache (recomputed only when block_size / num_blocks / max_ring_diff / n_edge change) ──

_cache: dict = {"key": None, "data": ({}, {}, 0)}


def _cached_symmetry_classes(block_size, num_blocks, max_ring_diff, n_edge=0):
    """Return (plane_to_class, class_to_planes, num_classes), reusing cached data when possible."""
    key = (block_size, num_blocks, max_ring_diff, n_edge)
    if _cache["key"] != key:
        _cache["key"] = key
        _cache["data"] = compute_sinogram_plane_symmetries(
            block_size, num_blocks, max_ring_diff, n_edge=n_edge
        )
    return _cache["data"]


# ── Drawing helpers ───────────────────────────────────────────────────────────


def draw_panel(ax, B, num_blocks, r1_base, r2_base, class_idx=None, n_edge=0):
    """Draw scanner cross-section with all equivalent LORs."""
    ax.cla()
    N = B * num_blocks
    GAP, D, cw, ch = 0.4, 3.0, 0.82, 1.0

    k_arr = np.arange(B)
    if B > 1:
        sens = 0.35 + 0.65 * np.cos(np.pi * (k_arr - (B - 1) / 2) / (B - 1)) ** 2
        sens /= sens.max()
    else:
        sens = np.ones(1)  # single crystal per block: uniform sensitivity

    z = np.array(
        [blk * (B + GAP) + pos for blk, pos in (divmod(r, B) for r in range(N))]
    )
    z -= z.mean()

    all_equiv = plane_orbit(r1_base, r2_base, B, N, n_edge)
    p1_base, p2_base = r1_base % B, r2_base % B
    prod = float(sens[p1_base] * sens[p2_base])

    for r in range(N):
        is_edge = not is_interior_ring(r, N, n_edge)
        fc = (plt.cm.Greys(0.30 + 0.45 * sens[r % B]) if is_edge
              else plt.cm.magma(0.01 + 0.99 * sens[r % B]))
        ec = "dimgray" if is_edge else "k"
        lw = 0.8 if is_edge else 0.4
        for y0 in (-D - ch, D):
            ax.add_patch(
                mpatches.Rectangle(
                    (z[r] - cw / 2, y0), cw, ch,
                    facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2,
                )
            )

    for b in range(1, num_blocks):
        zg = (z[b * B - 1] + z[b * B]) / 2
        for y0 in (-D - ch, D):
            ax.add_patch(
                mpatches.Rectangle(
                    (zg - GAP / 2, y0),
                    GAP,
                    ch,
                    facecolor="white",
                    edgecolor="none",
                    zorder=3,
                )
            )

    first_pos = first_neg = True
    for ra, rb in all_equiv:
        if ra < rb:
            col, ls, lw, alpha = "tab:blue", "-", 2.2, 0.90
            lbl = rf"$\Delta>0$: k=({ra%B},{rb%B})" if first_pos else "_nolegend_"
            first_pos = False
        else:
            col, ls, lw, alpha = "tab:orange", "--", 2.0, 0.80
            lbl = (
                rf"$\Delta<0$ (flip): k=({ra%B},{rb%B})" if first_neg else "_nolegend_"
            )
            first_neg = False
        ax.plot(
            [z[ra], z[rb]],
            [-D, D],
            color=col,
            lw=lw,
            ls=ls,
            label=lbl,
            zorder=5,
            alpha=alpha,
        )
        ax.plot(z[ra], -D, "o", ms=7, color=col, zorder=6, alpha=alpha)
        ax.plot(z[rb], D, "o", ms=7, color=col, zorder=6, alpha=alpha)

    ax.axvline(0, color="k", ls=":", lw=1.2, alpha=0.40, zorder=4)

    for b in range(num_blocks):
        ax.text(
            z[b * B : (b + 1) * B].mean(),
            D + ch + 0.65,
            f"block {b}",
            ha="center",
            fontsize=7.5,
            color="gray",
        )

    ax.text(
        z[0] + 0.5,
        -D - ch - 0.35,
        rf"k=({p1_base},{p2_base}), $\varepsilon\cdot\varepsilon={prod:.3f}$",
        ha="left",
        va="top",
        fontsize=8,
        bbox={
            "facecolor": "lightyellow",
            "edgecolor": "gray",
            "alpha": 0.88,
            "boxstyle": "round,pad=0.3",
        },
    )

    ax.set_xlim(z[0] - 1.2, z[-1] + 1.2)
    ax.set_ylim(-D - ch - 2.0, D + ch + 1.4)
    ax.set_yticks([-D - ch / 2, D + ch / 2])
    ax.set_yticklabels(["det. A", "det. B"])
    ax.set_xlabel("Axial position z")

    cls_str = (
        f"  (class #{class_idx})" if class_idx is not None and class_idx >= 0 else ""
    )
    ec_str = f"  [edge n={n_edge}]" if n_edge > 0 else ""
    ax.set_title(
        f"Base ({r1_base},{r2_base}),  "
        + rf"$\Delta={r2_base - r1_base}$,  "
        + f"{len(all_equiv)} equivalent LORs{cls_str}{ec_str}"
    )
    ax.legend(fontsize=8, loc="upper left", framealpha=0.9)


def draw_michelogram(
    ax,
    B,
    num_blocks,
    max_ring_diff,
    class_map,
    class_members,
    n_classes,
    highlight_pair=None,
):
    """Plot michelogram coloured by equivalence class.

    Cells carry their class index as a label.  Members of the equivalence
    class of ``highlight_pair`` are outlined in red.
    """
    ax.cla()
    N = B * num_blocks

    grid = np.full((N, N), -1, dtype=int)
    for (r1, r2), cls in class_map.items():
        grid[r2, r1] = cls  # row = r2 (end ring), col = r1 (start ring)

    base_colours = plt.cm.tab20.colors
    colours = [base_colours[i % len(base_colours)] for i in range(n_classes)]
    cmap = plt.matplotlib.colors.ListedColormap(colours)
    cmap.set_bad("whitesmoke")

    masked = np.ma.array(grid, mask=(grid < 0))
    ax.imshow(
        masked,
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=n_classes - 0.5,
        interpolation="nearest",
        aspect="equal",
    )

    # Class-index labels (skip for large grids)
    if N <= 50:
        fs = max(3, min(6, int(180 / N)))
        for (r1, r2), cls in class_map.items():
            ax.text(
                r1,
                r2,
                str(cls),
                ha="center",
                va="center",
                fontsize=fs,
                color="k",
                zorder=3,
            )

    # Block boundary lines
    for b in range(1, num_blocks):
        ax.axvline(b * B - 0.5, color="k", lw=0.8, alpha=0.55)
        ax.axhline(b * B - 0.5, color="k", lw=0.8, alpha=0.55)

    # Highlight selected equivalence class with thick red borders
    if highlight_pair is not None:
        r1h, r2h = highlight_pair
        if (r1h, r2h) in class_map:
            for r1, r2 in class_members[class_map[(r1h, r2h)]]:
                ax.add_patch(
                    mpatches.Rectangle(
                        (r1 - 0.5, r2 - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="red",
                        linewidth=3.0,
                        zorder=5,
                    )
                )

    for b in range(num_blocks):
        mid = b * B + (B - 1) / 2
        ax.text(mid, -2.5, f"b{b}", ha="center", va="top", fontsize=7, color="gray")
        ax.text(-2.5, mid, f"b{b}", ha="right", va="center", fontsize=7, color="gray")

    ax.set_xlabel("Start ring $r_1$")
    ax.set_ylabel("End ring $r_2$")
    highlight_note = (
        "  —  red = selected class  (click to select)" if highlight_pair else ""
    )
    ax.set_title(
        f"Michelogram  (B={B}, {num_blocks} blocks, "
        + r"$|\Delta|\leq$"
        + f"{max_ring_diff})\n"
        + f"{n_classes} equivalence classes{highlight_note}"
    )


def draw_class_sizes(ax, class_members, n_classes, highlight_cls=None):
    """Bar chart: number of sinogram planes per equivalence class."""
    ax.cla()

    sizes = [len(class_members[i]) for i in range(n_classes)]
    base_colours = plt.cm.tab20.colors
    bar_colours = [base_colours[i % len(base_colours)] for i in range(n_classes)]

    bars = ax.bar(
        range(n_classes), sizes, color=bar_colours, edgecolor="none", width=0.8
    )

    # Highlight selected class with a thick red outline
    if highlight_cls is not None and 0 <= highlight_cls < n_classes:
        bars[highlight_cls].set_edgecolor("red")
        bars[highlight_cls].set_linewidth(2.5)

    # Count labels on top of each bar (only when there are few enough classes)
    if n_classes <= 40:
        fs = max(4, min(7, int(200 / n_classes)))
        for bar, sz in zip(bars, sizes):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                str(sz),
                ha="center",
                va="bottom",
                fontsize=fs,
                color="k",
            )

    # x-ticks: show every tick if few classes, otherwise every 5th
    step = 1 if n_classes <= 20 else 5
    ax.set_xticks(range(0, n_classes, step))
    ax.set_xlabel("Equivalence class index")
    ax.set_ylabel("Number of sinogram planes")
    ax.set_title(
        f"Class sizes  ({n_classes} classes)"
        + (f"\n↑ class #{highlight_cls} selected" if highlight_cls is not None else "")
    )
    ax.set_xlim(-0.5, n_classes - 0.5)
    ax.set_ylim(0, max(sizes) * 1.18)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _parse(tb, default, lo, hi):
    """Parse a TextBox value as a clamped integer."""
    try:
        return max(lo, min(hi, int(tb.text)))
    except ValueError:
        return default


# ── Figure and initial state ──────────────────────────────────────────────────

INIT_B, INIT_NB, INIT_R1, INIT_R2, INIT_DIFF = 5, 4, 10, 14, 19

fig, (ax_lor, ax_mich, ax_bar) = plt.subplots(
    1,
    3,
    figsize=(20, 8),
    gridspec_kw={"wspace": 0.32, "width_ratios": [1.6, 1.4, 1.0]},
)
plt.subplots_adjust(bottom=0.18, left=0.05, right=0.98, top=0.95)

# Pre-compute and draw initial state  (n_edge=0: no edge correction by default)
_cm0, _mb0, _nc0 = _cached_symmetry_classes(INIT_B, INIT_NB, INIT_DIFF, n_edge=0)
_cls0 = _cm0.get((INIT_R1, INIT_R2))
draw_panel(ax_lor, INIT_B, INIT_NB, INIT_R1, INIT_R2, class_idx=_cls0, n_edge=0)
draw_michelogram(
    ax_mich, INIT_B, INIT_NB, INIT_DIFF, _cm0, _mb0, _nc0,
    highlight_pair=(INIT_R1, INIT_R2),
)
draw_class_sizes(ax_bar, _mb0, _nc0, highlight_cls=_cls0)

# ── Text-box controls ─────────────────────────────────────────────────────────
#
# Layout (figure fractions):
#   Left column  (x ≈ 0.07 … 0.43): B, num_blocks, max|Δ|, n_edge
#   Right column (x ≈ 0.55 … 0.92): r1, r2
#   Centre:                           Recalculate button

sh, sg = 0.032, 0.010  # text-box height, gap


def _sy(row):
    """Bottom edge of row (0 = lowest)."""
    return 0.03 + row * (sh + sg)


# axes widths are the INPUT FIELD only; matplotlib places the label to the left
ax_tB    = fig.add_axes([0.22, _sy(3), 0.20, sh])
ax_tnb   = fig.add_axes([0.22, _sy(2), 0.20, sh])
ax_tdiff = fig.add_axes([0.22, _sy(1), 0.20, sh])
ax_tne   = fig.add_axes([0.22, _sy(0), 0.20, sh])
ax_tr1   = fig.add_axes([0.72, _sy(2), 0.20, sh])
ax_tr2   = fig.add_axes([0.72, _sy(1), 0.20, sh])
ax_btn   = fig.add_axes([0.42, _sy(0), 0.16, sh * 1.3])

tb_B    = TextBox(ax_tB,    "B (cryst/blk)   ", initial=str(INIT_B))
tb_nb   = TextBox(ax_tnb,   "Num blocks      ", initial=str(INIT_NB))
tb_diff = TextBox(ax_tdiff, "Max |Δ|         ", initial=str(INIT_DIFF))
tb_ne   = TextBox(ax_tne,   "Edge n          ", initial="0")
tb_r1   = TextBox(ax_tr1,   "r1   ", initial=str(INIT_R1))
tb_r2   = TextBox(ax_tr2,   "r2   ", initial=str(INIT_R2))
btn_recalc = Button(ax_btn, "↻  Recalculate", color="0.85", hovercolor="0.70")


def _update(_event=None):
    """Recompute and redraw all three panels from current text-box values."""
    B        = _parse(tb_B,    INIT_B,    1,  20)
    nb       = _parse(tb_nb,   INIT_NB,   2,  10)
    N        = B * nb
    r1       = _parse(tb_r1,   0,          0,  N - 1)
    r2       = _parse(tb_r2,   0,          0,  N - 1)
    max_diff = _parse(tb_diff, min(B, N-1), 1,  N - 1)
    n_edge   = _parse(tb_ne,   0,          0,  N // 2)

    cm, members, nc = _cached_symmetry_classes(B, nb, max_diff, n_edge=n_edge)
    cls_idx = cm.get((r1, r2))
    draw_panel(ax_lor, B, nb, r1, r2, class_idx=cls_idx, n_edge=n_edge)
    draw_michelogram(ax_mich, B, nb, max_diff, cm, members, nc, highlight_pair=(r1, r2))
    draw_class_sizes(ax_bar, members, nc, highlight_cls=cls_idx)
    fig.canvas.draw_idle()


btn_recalc.on_clicked(_update)

# ── Click on michelogram to select plane ──────────────────────────────────────


def _on_mich_click(event):
    """Select (r1, r2) by clicking a cell in the michelogram."""
    if event.inaxes is not ax_mich or event.button != 1:
        return
    if event.xdata is None or event.ydata is None:
        return
    B = _parse(tb_B, INIT_B, 1, 20)
    nb = _parse(tb_nb, INIT_NB, 2, 10)
    N = B * nb
    r1 = int(round(event.xdata))
    r2 = int(round(event.ydata))
    if not (0 <= r1 < N and 0 <= r2 < N):
        return
    max_diff = _parse(tb_diff, min(B, N - 1), 1, N - 1)
    if abs(r1 - r2) > max_diff:
        return
    # Update text boxes (set_val does not trigger on_submit)
    tb_r1.set_val(str(r1))
    tb_r2.set_val(str(r2))
    _update()


fig.canvas.mpl_connect("button_press_event", _on_mich_click)

plt.show()
