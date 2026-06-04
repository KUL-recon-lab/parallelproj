"""Visualise axial sinogram plane symmetries for a PET scanner with cylindrical symmetry."""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sinogram_symmetries import (
    is_interior_ring,
    plane_orbit,
    compute_sinogram_plane_symmetries,
)

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
        fc = (
            plt.cm.Greys(0.30 + 0.45 * sens[r % B])
            if is_edge
            else plt.cm.magma(0.005 + 0.995 * sens[r % B])
        )
        ec = "dimgray" if is_edge else "k"
        lw = 0.8 if is_edge else 0.4
        for y0 in (-D - ch, D):
            ax.add_patch(
                mpatches.Rectangle(
                    (z[r] - cw / 2, y0),
                    cw,
                    ch,
                    facecolor=fc,
                    edgecolor=ec,
                    linewidth=lw,
                    zorder=2,
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


# ── Command-line interface ────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Visualise axial sinogram plane symmetries for a "
            "cylindrically-symmetric PET scanner."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "-B",
        "--block-size",
        type=int,
        default=7,
        metavar="B",
        help="Axial crystals per detector block",
    )
    p.add_argument(
        "-n",
        "--num-blocks",
        type=int,
        default=5,
        metavar="N",
        help="Number of axial detector blocks",
    )
    p.add_argument(
        "--r1",
        type=int,
        default=28,
        metavar="R1",
        help="Start ring of the highlighted plane",
    )
    p.add_argument(
        "--r2",
        type=int,
        default=29,
        metavar="R2",
        help="End ring of the highlighted plane",
    )
    p.add_argument(
        "-d",
        "--max-ring-diff",
        type=int,
        default=None,
        metavar="D",
        help="Maximum |r1-r2| in the sinogram (default: block_size)",
    )
    p.add_argument(
        "-e",
        "--n-edge",
        type=int,
        default=0,
        metavar="E",
        help="Edge rings at each scanner end with different sensitivity",
    )
    p.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        metavar="FILE",
        help="Save figure to FILE instead of (or in addition to) showing it",
    )
    return p


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = _build_parser().parse_args()

    block_size = args.block_size
    num_blocks = args.num_blocks
    num_rings = block_size * num_blocks
    r1 = max(0, min(args.r1, num_rings - 1))
    r2 = max(0, min(args.r2, num_rings - 1))
    max_ring_diff = (
        args.max_ring_diff if args.max_ring_diff is not None else num_rings - 1
    )
    max_ring_diff = max(1, min(max_ring_diff, num_rings - 1))
    n_edge = max(0, args.n_edge)

    plane_to_class, class_to_planes, num_classes = compute_sinogram_plane_symmetries(
        block_size, num_blocks, max_ring_diff, n_edge=n_edge
    )
    cls_idx = plane_to_class.get((r1, r2))

    fig, (ax_lor, ax_mich, ax_bar) = plt.subplots(
        1,
        3,
        figsize=(20, 7),
        layout="constrained",
    )
    fig.suptitle(
        f"Axial sinogram plane symmetries  "
        f"(B={block_size}, {num_blocks} blocks, "
        rf"$|\Delta|\leq${max_ring_diff}, n_edge={n_edge})",
        fontsize=12,
    )

    draw_panel(ax_lor, block_size, num_blocks, r1, r2, class_idx=cls_idx, n_edge=n_edge)
    draw_michelogram(
        ax_mich,
        block_size,
        num_blocks,
        max_ring_diff,
        plane_to_class,
        class_to_planes,
        num_classes,
        highlight_pair=(r1, r2),
    )
    draw_class_sizes(ax_bar, class_to_planes, num_classes, highlight_cls=cls_idx)

    if args.output:
        fig.savefig(args.output, dpi=300)
        print(f"Saved to {args.output}")
    plt.show()
