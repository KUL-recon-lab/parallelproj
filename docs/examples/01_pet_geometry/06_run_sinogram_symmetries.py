"""
Sinogram symmetries
===================

:bdg-primary:`geometry` :bdg-secondary:`advanced` :bdg-light:`CPU or GPU`

In a cylindrically symmetric PET scanner many sinogram bins are geometrically
equivalent: they are mapped onto each other by a shift, reflection or rotation
of the scanner.  Quantities that depend only on the geometry (e.g. the
geometric sensitivity of a symmetric object) therefore need to be computed for
one representative per equivalence class only.  This example groups the
sinogram planes into classes under three *axial* symmetries, adds the two
*in-plane* symmetries of the view and radial axes, and reduces and expands a
sinogram over these classes.

**You will learn**

- which five symmetries act on the plane, view and radial axes of a span-1
  sinogram,
- how to compute axial plane equivalence classes with
  :func:`.compute_sinogram_plane_symmetries` and inspect them in the
  Michelogram,
- how the edge rings of a block scanner (``n_edge``) split classes,
- how to build view and radial classes with :func:`.build_view_class_indices`
  and :func:`.build_radial_class_indices`,
- how to reduce a sinogram to its classes and expand it back with
  :func:`.reduce_sinogram_by_symmetry_class` and
  :func:`.expand_sinogram_by_symmetry_class`.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_02_run_regular_polygon_pet_sino.py`
and
:ref:`sphx_glr_auto_examples_01_pet_geometry_04_run_michelogram_and_compression.py`.
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import parallelproj.pet_lors
import parallelproj.pet_scanners
from parallelproj._examples_utils import suggest_array_backend_and_device
from parallelproj.sinogram_symmetries import (
    build_plane_class_indices,
    build_radial_class_indices,
    build_view_class_indices,
    compute_sinogram_plane_symmetries,
    expand_sinogram_by_symmetry_class,
    is_interior_ring,
    plane_orbit,
    reduce_sinogram_by_symmetry_class,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# The three plotting helpers below draw the axial cross-section of the
# scanner with all LORs of one equivalence class, the Michelogram coloured by
# class, and the bar chart of class sizes.  They are pure matplotlib and take
# the class dictionaries returned by :func:`.compute_sinogram_plane_symmetries`.
# You can skip them on a first reading.


def draw_panel(ax, B, num_blocks, r1_base, r2_base, class_idx=None, n_edge=0):
    """Axial cross-section of the scanner with all LORs equivalent to a plane."""
    N = B * num_blocks
    GAP, D, cw, ch = 0.4, 3.0, 0.82, 1.0

    # illustrative intra-block crystal sensitivity (cos^2 profile)
    k_arr = np.arange(B)
    if B > 1:
        sens = 0.35 + 0.65 * np.cos(np.pi * (k_arr - (B - 1) / 2) / (B - 1)) ** 2
        sens /= sens.max()
    else:
        sens = np.ones(1)

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

    first_pos = first_neg = True
    for ra, rb in all_equiv:
        if ra < rb:
            col, ls, lw, alpha = "tab:blue", "-", 2.2, 0.90
            lbl = rf"$\Delta>0$: k=({ra % B},{rb % B})" if first_pos else "_nolegend_"
            first_pos = False
        else:
            col, ls, lw, alpha = "tab:orange", "--", 2.0, 0.80
            lbl = (
                rf"$\Delta<0$ (swap): k=({ra % B},{rb % B})"
                if first_neg
                else "_nolegend_"
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
            fontsize="small",
            color="gray",
        )

    ax.text(
        z[0] + 0.5,
        -D - ch - 0.35,
        rf"k=({p1_base},{p2_base}), $\varepsilon\cdot\varepsilon={prod:.3f}$",
        ha="left",
        va="top",
        fontsize="small",
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
    ax.set_xlabel("Axial position z (crystal pitch units)")

    has_cls = class_idx is not None and class_idx >= 0
    cls_str = f", class #{class_idx}" if has_cls else ""
    ec_str = f", n_edge={n_edge}" if n_edge > 0 else ""
    ax.set_title(
        f"Plane ({r1_base},{r2_base}), "
        + rf"$\Delta={r2_base - r1_base}$: "
        + f"{len(all_equiv)} equivalent planes{cls_str}{ec_str}"
    )
    ax.legend(fontsize="small", loc="lower right", framealpha=0.9)


def class_colours(n_classes):
    """One tab20 colour per class (repeating), shared by the two class plots."""
    base = plt.cm.tab20.colors
    return [base[i % len(base)] for i in range(n_classes)]


def draw_michelogram(
    ax, B, num_blocks, max_ring_diff, class_map, class_members, n_classes, highlight
):
    """Michelogram coloured by equivalence class, class of ``highlight`` in red."""
    N = B * num_blocks

    grid = np.full((N, N), -1, dtype=int)
    for (r1, r2), cls in class_map.items():
        grid[r2, r1] = cls  # row = r2 (end ring), col = r1 (start ring)

    cmap = plt.matplotlib.colors.ListedColormap(class_colours(n_classes))
    cmap = cmap.with_extremes(bad="whitesmoke")

    ax.imshow(
        np.ma.array(grid, mask=(grid < 0)),
        origin="lower",
        cmap=cmap,
        vmin=-0.5,
        vmax=n_classes - 0.5,
        interpolation="nearest",
        aspect="equal",
    )

    # class-index labels (skipped for large grids)
    if N <= 50:
        fs = max(3, min(6, int(180 / N)))
        for (r1, r2), cls in class_map.items():
            ax.text(r1, r2, str(cls), ha="center", va="center", fontsize=fs)

    # block boundaries
    for b in range(1, num_blocks):
        ax.axvline(b * B - 0.5, color="k", lw=0.8, alpha=0.55)
        ax.axhline(b * B - 0.5, color="k", lw=0.8, alpha=0.55)

    # all members of the class of the highlighted plane
    if highlight in class_map:
        for r1, r2 in class_members[class_map[highlight]]:
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

    # ring indices on the bottom / left, block indices on the top / right
    step = 1 if N <= 30 else 5
    ax.set_xticks(range(0, N, step))
    ax.set_yticks(range(0, N, step))
    ax.tick_params(labelsize="small")
    centres = [b * B + (B - 1) / 2 for b in range(num_blocks)]
    labels = [f"block {b}" for b in range(num_blocks)]
    top = ax.secondary_xaxis("top")
    top.set_xticks(centres, labels)
    right = ax.secondary_yaxis("right")
    right.set_yticks(centres, labels, rotation=90, va="center")
    for sec in (top, right):
        sec.tick_params(length=0, labelcolor="gray", labelsize="small")
    ax.set_xlabel("Start ring $r_1$")
    ax.set_ylabel("End ring $r_2$")
    ax.set_title(
        f"B={B}, {num_blocks} blocks, "
        + r"$|\Delta|\leq$"
        + f"{max_ring_diff}: {n_classes} classes\n"
        + f"red: class of plane {highlight}"
    )


def draw_class_sizes(ax, class_members, n_classes, highlight_cls=None):
    """Bar chart: number of sinogram planes per equivalence class."""
    sizes = [len(class_members[i]) for i in range(n_classes)]
    bars = ax.bar(range(n_classes), sizes, color=class_colours(n_classes), width=0.8)

    if highlight_cls is not None and 0 <= highlight_cls < n_classes:
        bars[highlight_cls].set_edgecolor("red")
        bars[highlight_cls].set_linewidth(2.5)

    # every tick if few classes, otherwise every 5th
    step = 1 if n_classes <= 20 else 5
    ax.set_xticks(range(0, n_classes, step))
    ax.set_xlabel("Equivalence class index")
    ax.set_ylabel("Number of sinogram planes")
    note = f", red: class #{highlight_cls}" if highlight_cls is not None else ""
    ax.set_title(f"Class sizes ({n_classes} classes{note})")
    ax.set_xlim(-0.5, n_classes - 0.5)


# %%
# 1. Five sinogram symmetries
# ---------------------------
#
# A span-1 sinogram has a plane axis (ring pairs ``(r1, r2)``), a view axis
# and a radial axis.  Five families of symmetries act on them.
#
# **Axial symmetries (plane axis, ring pairs)**
#
# 1. **Axial block shift**: shifting both ring indices by one full block
#    width maps every intra-block crystal position to the same position in
#    the adjacent block.  The geometry is preserved.
# 2. **Scanner midplane reflection**: reflecting about the axial centre maps
#    ring ``r`` to ring ``N - 1 - r``.  For an object that is symmetric in
#    ``z`` this maps each plane ``(r1, r2)`` to an equivalent plane.
# 3. **Endpoint swap**: ``(r1, r2)`` and ``(r2, r1)`` describe the same
#    physical LOR traversed in the opposite direction, so their expected
#    counts are equal.
#
# **In-plane symmetries (view and radial axes)**
#
# 4. **Scanner rotational symmetry**: a regular polygon with ``num_sides``
#    sides has ``num_sides``-fold rotational symmetry, which reduces the number
#    of distinct views by a factor of ``num_sides / 2``.
# 5. **Radial mirror symmetry**: radial bins ``r`` and ``num_rad - 1 - r``
#    have the same perpendicular distance from the FOV centre and carry equal
#    expected counts for a centred object.
#
# Sections 3 to 6 deal with the axial symmetries, section 7 with the in-plane
# ones, and sections 8 and 9 apply all five to a sinogram.

# %%
# 2. Scanner and sinogram descriptor
# ----------------------------------
#
# The scanner has ``num_blocks = 4`` axial detector blocks of ``B = 5``
# crystals each, i.e. ``N = 20`` rings, and 28 polygon sides with 16
# crystals each.  The two outermost rings at each end (``n_edge = 2``) are
# treated as edge rings (section 3).  The span-1
# :class:`.RegularPolygonPETLORDescriptor` with ``max_ring_difference = 19``
# contains all ring pairs.  We follow one plane, ``(3, 5)``, through the
# example.

B = 5  # crystals per axial block
num_blocks = 4  # axial blocks
max_ring_diff = 19  # maximum |r1 - r2| in the sinogram
n_edge = 2  # edge rings at each scanner end
r1_sel = 3  # start ring of the highlighted plane
r2_sel = 5  # end ring of the highlighted plane

num_rings = B * num_blocks

scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=300.0,
    num_sides=28,
    num_lor_endpoints_per_side=16,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-95.0, 95.0, num_rings, device=dev),
    symmetry_axis=2,
)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(
        num_rings, max_ring_difference=max_ring_diff, span=1
    ),
    radial_trim=3,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

print(f"Scanner  : {num_rings} rings ({num_blocks} blocks x B={B}), n_edge={n_edge}")
print(
    f"Sinogram : shape {lor_desc.spatial_sinogram_shape} "
    f"(num_rad={lor_desc.num_rad}, num_views={lor_desc.num_views}, "
    f"num_planes={lor_desc.num_planes})"
)
print(f"Highlighted plane : ({r1_sel}, {r2_sel})")

# %%
# 3. Axial plane equivalence classes
# ----------------------------------
#
# :func:`.compute_sinogram_plane_symmetries` iterates over all valid ring
# pairs and groups them into orbits under the three axial symmetries.  Each
# orbit is one *equivalence class* with an integer index:
#
# * ``plane_to_class`` maps every ``(r1, r2)`` pair to its class index,
# * ``class_to_planes`` maps a class index to the list of its member planes,
# * ``num_classes`` is the number of classes.
#
# To estimate the geometric sensitivity of a cylindrically symmetric object,
# only one representative plane per class needs to be forward-projected; the
# result is then copied to all members of the class.
#
# .. note::
#
#    ``n_edge`` restricts the block-shift equivalence for the outermost
#    rings.  These rings have no neighbouring block on one side, so their
#    crystal sensitivity differs from that of the same intra-block position
#    in an interior block.  With ``n_edge > 0`` edge and interior planes are
#    kept in separate classes, so different sensitivities are not mixed.

plane_to_class, class_to_planes, num_classes = compute_sinogram_plane_symmetries(
    B, num_blocks, max_ring_diff, n_edge=n_edge
)

cls_sel = plane_to_class.get((r1_sel, r2_sel))

print(f"Total sinogram planes : {len(plane_to_class)}")
print(f"Equivalence classes   : {num_classes}")
print(f"Average class size    : {len(plane_to_class) / num_classes:.1f} planes")
print(
    f"Class of plane ({r1_sel},{r2_sel}) : class #{cls_sel} "
    f"({len(class_to_planes[cls_sel])} members)"
)

# %%
# 4. Michelogram coloured by equivalence class
# --------------------------------------------
#
# In the Michelogram, the cell in column ``r1`` and row ``r2`` is the sinogram
# plane ``(r1, r2)``.  Cells outside ``|r1 - r2| <= max_ring_diff`` would be
# masked in light grey (here all ring pairs are included).  Each colour is
# one equivalence class, and each cell is labelled with its class index.
# Thin black lines mark the boundaries between axial detector blocks.  The
# red outlines mark all planes in the class of the selected plane ``(3, 5)``.

fig, ax = plt.subplots(figsize=(6.5, 6.5), layout="constrained")
draw_michelogram(
    ax,
    B,
    num_blocks,
    max_ring_diff,
    plane_to_class,
    class_to_planes,
    num_classes,
    highlight=(r1_sel, r2_sel),
)
fig.show()

# %%
# The 12 red cells are spread over the whole Michelogram: block shifts move a
# plane along the diagonal, the midplane reflection mirrors it through the
# centre, and the endpoint swap mirrors it about the main diagonal.  Cells of
# the same colour always carry the same label; since the 20 colours repeat,
# the label identifies the class.

# %%
# 5. Equivalent LORs of the selected plane
# ----------------------------------------
#
# :func:`.plane_orbit` returns all ring pairs in the class of the seed pair
# ``(r1_sel, r2_sel)``.  The figure shows an axial cross-section of the
# scanner: the two rows of boxes are the crystals of all rings on two opposite
# sides (detector A and B), and each line is one equivalent plane.
#
# Crystals are coloured from dark (low) to bright (high) sensitivity
# according to an illustrative cosine-squared axial sensitivity profile
# within each block.  Crystals in the outermost ``n_edge`` rings are grey:
# they are edge rings and belong to separate classes.  Blue solid lines are
# ring pairs with ``r2 > r1`` (positive ring difference ``Delta``), orange
# dashed lines the swapped pairs ``(r2, r1)``.
#
# .. note::
#
#    The annotation box shows the intra-block crystal positions
#    ``k = (r1 % B, r2 % B)`` and the product of their sensitivities
#    ``epsilon * epsilon``.  This product is the same for every plane in the
#    class: it is the quantity that the block-shift symmetry preserves.

fig, ax = plt.subplots(figsize=(10, 4.5), layout="constrained")
draw_panel(ax, B, num_blocks, r1_sel, r2_sel, class_idx=cls_sel, n_edge=n_edge)
fig.show()

# %%
# All 12 LORs connect crystals at the same intra-block positions (3 and 0 in
# one direction, 0 and 3 in the other).  None of them touches a grey edge
# ring, and the pattern is mirror symmetric about the axial centre (dotted).

# %%
# 6. Class-size distribution
# --------------------------
#
# The bar chart shows the number of sinogram planes in each equivalence
# class, with the same colours as the Michelogram.  Class sizes differ: a
# plane with a large ring difference has fewer block-shifted copies that fit
# into the scanner, and planes that are mapped onto themselves by the
# reflection or the swap have smaller orbits.  With ``n_edge > 0`` planes
# that touch an edge ring are split off into further, smaller classes.
#
# .. note::
#
#    The class sizes give the **compression ratio** of the symmetry
#    reduction.  A sensitivity sinogram of ``N_planes`` planes needs only
#    ``num_classes`` unique computations, i.e. the speed-up is the average
#    class size (``m``-fold if all classes have size ``m``).

fig, ax = plt.subplots(figsize=(8, 3.5), layout="constrained")
draw_class_sizes(ax, class_to_planes, num_classes, highlight_cls=cls_sel)
fig.show()

# %%
# Most classes have 4 members.  The selected class #56 is among the largest
# (12 planes); the 400 planes reduce to 80 classes, a 5-fold reduction.

# %%
# 7. In-plane symmetries
# ----------------------
#
# Two further symmetries act on the *view* and *radial* axes.
#
# * :func:`.build_view_class_indices` groups views by the rotational symmetry
#   of the scanner: views ``v, v + n, v + 2n, ...`` (with
#   ``n = num_lor_endpoints_per_side``) are related by a rotation of the
#   polygon by one side.  There are ``n`` view classes with
#   ``num_views // n`` views each.
# * :func:`.build_radial_class_indices` groups radial bins by the mirror
#   symmetry of the FOV: bins ``r`` and ``num_rad - 1 - r`` have the same
#   distance from the scanner axis.  ``num_rad`` is odd for regular polygon
#   scanners, so the central bin forms a class of its own.
#
# The combined reduction factor is the product of the factors of the three
# axes.

view_period = scanner.num_lor_endpoints_per_side
view_classes = build_view_class_indices(lor_desc.num_views, view_period)
rad_classes = build_radial_class_indices(lor_desc.num_rad)

n_view_classes = len(view_classes)
n_rad_classes = len(rad_classes)
views_per_class = lor_desc.num_views // n_view_classes
rads_per_class_max = max(len(c) for c in rad_classes)

reduction_planes = len(plane_to_class) / num_classes
reduction_views = lor_desc.num_views / n_view_classes
reduction_rad = lor_desc.num_rad / n_rad_classes
reduction_total = reduction_planes * reduction_views * reduction_rad

n_bins = len(plane_to_class) * lor_desc.num_views * lor_desc.num_rad
n_unique = num_classes * n_view_classes * n_rad_classes

print("In-plane symmetries")
print(
    f"  View axis  : {lor_desc.num_views} views -> {n_view_classes} classes "
    f"({views_per_class} views each, reduction factor {reduction_views:.1f}x)"
)
print(
    f"  Radial axis: {lor_desc.num_rad} bins  -> {n_rad_classes} classes "
    f"(up to {rads_per_class_max} bins each, reduction factor {reduction_rad:.2f}x)"
)
print(f"Axial plane symmetry reduction factor : {reduction_planes:.1f}x")
print(
    f"Combined reduction (planes x views x radial) : {reduction_total:.1f}x "
    f"({n_bins} -> {n_unique} unique bins)"
)

# %%
# 8. Reducing a sinogram to its equivalence classes
# -------------------------------------------------
#
# Any sinogram (e.g. a Monte Carlo emission scan of a uniform cylinder, or a
# forward projection of a sensitivity phantom) can be contracted to its
# equivalence classes with :func:`.reduce_sinogram_by_symmetry_class`, one
# axis at a time, using the three class index lists built above.  The
# reductions can be chained in any order.  For the plane axis,
# :func:`.build_plane_class_indices` converts the ring-pair classes into
# plane indices (this requires a span-1 descriptor).
#
# The optional ``reduction`` argument selects how bins of a class are
# combined:
#
# .. note::
#
#    * ``reduction=xp.sum`` (**default**) adds all counts of a class into one
#      bin, so the total count of the sinogram is preserved.  Use it to reduce
#      noisy Monte Carlo data before dividing by a forward projection to get a
#      per-class sensitivity estimate.
#    * ``reduction=xp.mean`` divides by the class size, so each reduced bin
#      holds the *average* count per original bin and can be compared
#      directly with a single unreduced bin.
#
# We use a Poisson sinogram with an expected value of 10 counts in every
# bin.  The reduction shrinks its shape from ``(num_rad, num_views,
# num_planes)`` to ``(n_rad_classes, n_view_classes, num_classes)`` and
# preserves the total count.

np.random.seed(42)
sino = xp.asarray(
    np.random.poisson(10, lor_desc.spatial_sinogram_shape).astype(np.float64),
    device=dev,
)

plane_class_idx = build_plane_class_indices(
    lor_desc.michelogram.plane_for_ring_pair_table, class_to_planes, num_classes
)

print(f"Sinogram shape before reduction : {tuple(sino.shape)}")

# reduce one axis after the other: view -> radial -> plane
sino_red = reduce_sinogram_by_symmetry_class(
    sino, view_classes, lor_desc.view_axis_num, xp.sum
)
sino_red = reduce_sinogram_by_symmetry_class(
    sino_red, rad_classes, lor_desc.radial_axis_num, xp.sum
)
sino_red = reduce_sinogram_by_symmetry_class(
    sino_red, plane_class_idx, lor_desc.plane_axis_num, xp.sum
)

print(f"Sinogram shape after  reduction : {tuple(sino_red.shape)}")
print(f"Total counts before : {float(xp.sum(sino)):.0f}")
print(f"Total counts after  : {float(xp.sum(sino_red)):.0f} (preserved by xp.sum)")

# %%
# 9. Expanding the reduced sinogram to the original shape
# -------------------------------------------------------
#
# After a reduction with ``xp.mean`` each reduced bin holds the average count
# of the original bins of its class.  :func:`.expand_sinogram_by_symmetry_class`
# broadcasts these class values back to the original sinogram shape, so that
# every bin gets the mean value of its class.  The result is a *denoised*
# sinogram in which all symmetry-equivalent LORs have identical values.

sino_mean = reduce_sinogram_by_symmetry_class(
    sino, view_classes, lor_desc.view_axis_num, xp.mean
)
sino_mean = reduce_sinogram_by_symmetry_class(
    sino_mean, rad_classes, lor_desc.radial_axis_num, xp.mean
)
sino_mean = reduce_sinogram_by_symmetry_class(
    sino_mean, plane_class_idx, lor_desc.plane_axis_num, xp.mean
)
print(f"Reduced (mean) shape : {tuple(sino_mean.shape)}")

sino_expanded = expand_sinogram_by_symmetry_class(
    sino_mean, plane_class_idx, lor_desc.num_planes, lor_desc.plane_axis_num
)
sino_expanded = expand_sinogram_by_symmetry_class(
    sino_expanded, rad_classes, lor_desc.num_rad, lor_desc.radial_axis_num
)
sino_expanded = expand_sinogram_by_symmetry_class(
    sino_expanded, view_classes, lor_desc.num_views, lor_desc.view_axis_num
)
print(f"Expanded shape       : {tuple(sino_expanded.shape)} (== original)")

# %%
# To check, we print the expanded values of all members of one class per
# axis, with the indices of the two other axes fixed: view class 3, radial
# class 0 (the outermost pair of radial bins) and plane class 4.

sample_class_view = view_classes[3]
sample_class_rad = rad_classes[0]
sample_class_planes = xp.asarray(
    [lor_desc.michelogram.plane_for_ring_pair(*x) for x in class_to_planes[4]],
    device=dev,
)

r_idx, v_idx, p_idx = 0, 0, 0  # fixed radial, view and plane bin

vals_view = sino_expanded[r_idx, sample_class_view, p_idx]
vals_rad = sino_expanded[sample_class_rad, v_idx, p_idx]
vals_planes = sino_expanded[r_idx, v_idx, sample_class_planes]

print(f"View class 3   -- values at (rad={r_idx}, plane={p_idx}) : {vals_view}")
print(f"Radial class 0 -- values at (view={v_idx}, plane={p_idx}): {vals_rad}")
print(f"Plane class 4  -- values at (rad={r_idx}, view={v_idx})  : {vals_planes}")

# %%
# Within each class all values are identical: the expanded sinogram is
# constant on every equivalence class.  The values differ slightly from the
# expected 10 because each class mean is still a noisy average.

# %%
# Summary
# -------
#
# - Three axial symmetries (block shift, midplane reflection, endpoint swap)
#   group span-1 sinogram planes into classes:
#   ``compute_sinogram_plane_symmetries(B, num_blocks, max_ring_diff, n_edge)``
#   returns ``plane_to_class``, ``class_to_planes`` and ``num_classes``;
#   ``plane_orbit`` gives the class of a single plane.
# - ``n_edge`` keeps planes that touch the outermost rings of a block scanner
#   in separate classes.
# - ``build_view_class_indices`` (rotational symmetry) and
#   ``build_radial_class_indices`` (radial mirror symmetry) add the in-plane
#   symmetries; the reduction factors of the three axes multiply.
# - ``reduce_sinogram_by_symmetry_class`` contracts one sinogram axis to its
#   classes (``xp.sum`` preserves counts, ``xp.mean`` averages), and
#   ``expand_sinogram_by_symmetry_class`` broadcasts class values back.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_01_pet_geometry_07_run_detector_mashing.py`
#    -- reducing the sinogram size by grouping neighbouring detectors.
#    Michelograms and span compression are introduced in
#    :ref:`sphx_glr_auto_examples_01_pet_geometry_04_run_michelogram_and_compression.py`.

# sphinx_gallery_thumbnail_number = 1
