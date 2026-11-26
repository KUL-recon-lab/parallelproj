"""
Detector mashing: fewer, bigger virtual detectors
=================================================

:bdg-primary:`geometry` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

*Mashing* groups neighbouring detectors into larger **virtual** detectors
located at the **average** position of the detectors they replace.  Fewer
detectors mean far fewer lines of response (LORs), so the sinogram shrinks
and reconstruction gets faster, at the cost of spatial resolution.
:class:`.SinogramMashingOperator` mashes a span-1 regular polygon sinogram
transaxially (crystals within each polygon side) and axially (rings).

**You will learn**

- how to mash a span-1 sinogram with :class:`.SinogramMashingOperator` and
  where the virtual detectors of the mashed scanner sit,
- the difference between the exact mashed model ``mash(P_fine(x))`` and the
  fast projector built on the coarse LOR descriptor,
- when to use ``mode="sum"`` and ``mode="average"``, and how the adjoint
  upsamples a coarse sinogram back to the fine grid,
- how to mash GE-layout sinograms by composing mashing with
  :class:`.SinogramAxialCompressionOperator`.

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
#
# The sinograms in this example are shown with ``show_vol_cuts`` as
# ``(radial, view, plane)`` arrays; ``as_rvp`` brings a sinogram into this
# axis order regardless of the sinogram order of its descriptor.

import matplotlib.pyplot as plt
import numpy as np

import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
from parallelproj import to_numpy_array
from parallelproj._examples_utils import show_vol_cuts, suggest_array_backend_and_device
from parallelproj.operators import CompositeLinearOperator

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

SINO_AXES = ("radial", "view", "plane")


def view_from_front(ax) -> None:
    """3D view from in front of the scanner (looking along +x2, +x1 down)"""
    ax.view_init(elev=-30, azim=160, roll=180, vertical_axis="y")


def as_rvp(sino, desc):
    """Return the sinogram as a ``(radial, view, plane)`` numpy array."""
    axes = (desc.radial_axis_num, desc.view_axis_num, desc.plane_axis_num)
    return np.moveaxis(to_numpy_array(sino), axes, (0, 1, 2))


# keep references to the viewers so that interactive slider callbacks are
# not garbage collected (static figures in the documentation)
viewers = []

# %%
# 1. Mashing factors and the two models
# -------------------------------------
#
# :class:`.SinogramMashingOperator` mashes a span-1 regular polygon sinogram
# by
#
# * ``transaxial_factor`` (:math:`N`): group :math:`N` neighbouring crystals
#   **within each polygon side** (around the ring), and
# * ``axial_factor`` (:math:`M`): group :math:`M` neighbouring **rings**
#   (along the symmetry axis).
#
# Averaging uniformly spaced crystals within a side (and uniformly spaced
# ring positions) again gives a regular polygon.  The mashed geometry is
# therefore itself a :class:`.RegularPolygonPETScannerGeometry`
# (``mash.coarse_scanner``) with a matching
# :class:`.RegularPolygonPETLORDescriptor` (``mash.coarse_lor_descriptor``).
# This gives two ways to model mashed data:
#
# * the **exact** model mashes the fine forward projection:
#   ``mash(P_fine(x))``;
# * the **fast** model projects directly along the averaged LORs with a
#   :class:`.RegularPolygonPETProjector` built on
#   ``mash.coarse_lor_descriptor``.
#
# ``mode="sum"`` preserves counts (use it for emission or other measured
# data); ``mode="average"`` averages and matches the fast coarse projector
# (use it for multiplicative factors such as attenuation or normalisation).

# %%
# 2. A fine scanner and its sinogram descriptor
# ---------------------------------------------
#
# The fine ("true") scanner has 14 sides with 8 crystals each (112 crystals
# per ring) and 12 rings.  Its span-1 sinogram includes ring differences up
# to 5.

num_rings = 12
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=95.0,
    num_sides=14,
    num_lor_endpoints_per_side=8,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-14.0, 14.0, num_rings, device=dev),
    symmetry_axis=2,
)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=5, span=1),
    radial_trim=11,
)

# %%
# 3. The mashing operator
# -----------------------
#
# ``transaxial_factor`` (``N``) must divide the number of crystals per side
# and ``axial_factor`` (``M``) must divide the number of rings.  We mash
# ``N = 4`` crystals and ``M = 2`` rings.  All mashing operators in this
# example use the same factors, so their coarse grids match.

transaxial_factor = 4  # mash this many neighbouring within-side crystals (N)
axial_factor = 2  # mash this many neighbouring rings (M)

mash = parallelproj.pet_lors.SinogramMashingOperator(
    lor_desc,
    transaxial_factor=transaxial_factor,
    axial_factor=axial_factor,
    mode="sum",
)
coarse_desc = mash.coarse_lor_descriptor

n_fine = int(np.prod(mash.in_shape))
n_coarse = int(np.prod(mash.out_shape))
print(mash)
print(f"fine   sinogram shape : {mash.in_shape}  ({n_fine} LORs)")
print(f"mashed sinogram shape : {mash.out_shape}  ({n_coarse} LORs)")
print(f"LOR reduction factor  : {n_fine / n_coarse:.1f}x")

# %%
# Mashing 4 crystals at each end of an LOR and 2 rings reduces the number of
# LORs by roughly ``N**2 * M**2 = 64``; the reduction factor is somewhat
# lower because the coarse sinogram keeps a few partially filled bins at the
# radial and axial edges.

# %%
# 4. Where the virtual detectors sit
# ----------------------------------
#
# The virtual crystals of the mashed scanner lie at the **average** position
# of the ``N`` crystals within a side that they replace.  The figure shows one
# ring in the transaxial plane (front view, ``+x1`` pointing down): small dots
# are the fine crystals, crosses the virtual crystals.

fine_pts = to_numpy_array(scanner.all_lor_endpoints).reshape(
    scanner.num_rings, scanner.num_lor_endpoints_per_ring, 3
)[0]
coarse_pts = to_numpy_array(mash.coarse_scanner.all_lor_endpoints).reshape(
    mash.coarse_scanner.num_rings,
    mash.coarse_scanner.num_lor_endpoints_per_ring,
    3,
)[0]
# transaxial plane = the two axes orthogonal to the symmetry axis
tax = [a for a in range(3) if a != scanner.symmetry_axis]

fig, ax = plt.subplots(figsize=(5.5, 5.5), layout="constrained")
ax.scatter(
    fine_pts[:, tax[0]],
    fine_pts[:, tax[1]],
    s=12,
    label=f"fine crystals ({fine_pts.shape[0]} per ring)",
)
ax.scatter(
    coarse_pts[:, tax[0]],
    coarse_pts[:, tax[1]],
    s=90,
    marker="x",
    label=f"mashed virtual crystals ({coarse_pts.shape[0]} per ring)",
)
ax.set_aspect("equal")
ax.invert_yaxis()  # +x1 points down, as in the 3D front views
ax.set_xlabel(f"x{tax[0]} (mm)")
ax.set_ylabel(f"x{tax[1]} (mm)")
ax.set_title(f"One ring, transaxial plane (N={transaxial_factor})")
ax.legend(loc="center", fontsize="small")
fig.show()

# %%
# Each side of 8 fine crystals is replaced by 2 virtual crystals at the
# centres of its two groups of 4.  The virtual crystals stay on the polygon,
# so the mashed scanner is again a regular polygon with 14 sides.

# %%
# 5. One view of the central plane, fine vs. mashed
# -------------------------------------------------
#
# :meth:`.RegularPolygonPETLORDescriptor.show_views` draws the LORs of a set
# of views and planes.  Below, each panel shows all LOR endpoints of a
# scanner in 3D together with the LORs of one view in the central plane.
# The mashed descriptor has far fewer LORs for (approximately) the same
# projection angle, each running between two averaged endpoints.

central_fine = xp.asarray([num_rings // 2], device=dev)
central_coarse = xp.asarray([num_rings // (2 * axial_factor)], device=dev)
view_fine = xp.asarray([lor_desc.num_views // 4], device=dev)
view_coarse = xp.asarray([coarse_desc.num_views // 4], device=dev)

fig, axes = plt.subplots(
    1, 2, figsize=(10, 5), subplot_kw={"projection": "3d"}, layout="constrained"
)
panels = (
    (scanner, lor_desc, view_fine, central_fine, "Fine scanner"),
    (mash.coarse_scanner, coarse_desc, view_coarse, central_coarse, "Mashed scanner"),
)
for ax, (sc, desc, view, plane, title) in zip(axes, panels):
    view_from_front(ax)
    sc.show_lor_endpoints(ax, show_linear_index=False)
    desc.show_views(ax, views=view, planes=plane, lw=0.5)
    ax.set_title(f"{title}: one view, central plane")
fig.show()

# %%
# The fine scanner has 8 crystals per side and 12 rings, the mashed one 2
# virtual crystals per side and 6 rings.  Its LORs of one view are 4 times
# sparser in the radial direction.

# %%
# 6. The two Michelograms
# -----------------------
#
# Axial mashing (``M``) merges neighbouring rings, so the mashed scanner has
# fewer rings and therefore a smaller Michelogram, i.e. fewer sinogram planes.

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
lor_desc.show_michelogram(axes[0])
axes[0].set_title(
    f"Fine: {scanner.num_rings} rings, {lor_desc.num_planes} planes"
)
coarse_desc.show_michelogram(axes[1])
axes[1].set_title(
    f"Mashed: {mash.coarse_scanner.num_rings} rings, {coarse_desc.num_planes} planes"
)
fig.show()

# %%
# The maximum ring difference shrinks with the number of rings: 5 fine ring
# differences become 2 coarse ring differences (``|Delta| <= 2``).

# %%
# 7. Mashing a simulated emission sinogram
# ----------------------------------------
#
# We forward-project a simple phantom with the fine projector to get a fine
# emission sinogram, and mash it with ``mode="sum"`` (counts add).
#
# By default ``coarse_radial_trim`` is ``lor_desc.radial_trim //
# transaxial_factor``, so the radial extent of the coarse sinogram matches
# that of the fine data.  With ``coarse_radial_trim=0`` the coarse sinogram
# would keep extra peripheral radial bins without any fine contributor; they
# would stay empty and the mashed sinogram would seem to lose counts at the
# largest radial offsets.

img_shape = (100, 100, num_rings)
voxel_size = (3.5, 3.5, 3.5)
proj_fine = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape, voxel_size
)

img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
img[25:55, 40:55, 2:] = 1.0
img[45:55, 45:75, :-2] = 2.0
img[:, :, : num_rings // 2] *= 1.5
img[:, :, ::2] *= 1.5

fine_sino = proj_fine(img)
mashed_sino = mash(fine_sino)

viewers.append(show_vol_cuts(img, fig_title="Simulated image"))

# %%
# The fine and mashed sinograms are 3D arrays; ``show_vol_cuts`` shows
# orthogonal cuts through them (with a slider per axis in an interactive
# matplotlib session).

viewers.append(
    show_vol_cuts(
        as_rvp(fine_sino, lor_desc),
        axis_labels=SINO_AXES,
        fig_title=f"Fine sinogram {mash.in_shape}",
        cmap="Greys",
    )
)
viewers.append(
    show_vol_cuts(
        as_rvp(mashed_sino, coarse_desc),
        axis_labels=SINO_AXES,
        fig_title=f"Mashed sinogram {mash.out_shape}",
        cmap="Greys",
    )
)

# %%
# The mashed sinogram shows the same structures at a 4 times coarser radial
# and view sampling and with half the number of rings.  Its values are larger
# because each bin sums the counts of many fine LORs.

# %%
# 8. How many fine LORs does each mashed LOR combine?
# ---------------------------------------------------
#
# Mashing a sinogram of ones with ``mode="sum"`` gives, for each mashed bin,
# the number of fine LORs that fold into it: its **multiplicity**.  It is
# largest in the interior (about ``transaxial_factor**2 * axial_factor**2``)
# and smaller toward the radial and axial edges, where fewer fine LORs
# contribute.

ones_fine = xp.ones(lor_desc.spatial_sinogram_shape, dtype=xp.float32, device=dev)
multiplicity_sino = mash(ones_fine)  # sum mode -> per-mashed-bin count

viewers.append(
    show_vol_cuts(
        as_rvp(multiplicity_sino, coarse_desc),
        axis_labels=SINO_AXES,
        fig_title="Multiplicity: number of fine LORs per mashed LOR",
        cmap="viridis",
    )
)

# %%
# 9. Fast coarse projector vs. the exact mashed model
# ---------------------------------------------------
#
# A projector built on ``mash.coarse_lor_descriptor`` projects directly along
# the averaged LORs, which is cheap.  With ``mode="average"`` it approximates
# the average over the bundle of fine LORs, i.e.
# ``mash_avg(P_fine(x)) ~ P_coarse(x)``.  The two are not identical: a single
# averaged LOR has a slightly narrower sensitivity profile than the bundle it
# replaces, so they differ by a resolution-dependent amount.  Use the exact
# ``mash(P_fine)`` model when that difference matters, and the fast coarse
# projector otherwise.

mash_avg = parallelproj.pet_lors.SinogramMashingOperator(
    lor_desc,
    transaxial_factor=transaxial_factor,
    axial_factor=axial_factor,
    mode="average",
)
proj_coarse = parallelproj.projectors.RegularPolygonPETProjector(
    mash_avg.coarse_lor_descriptor, img_shape, voxel_size
)

exact = mash_avg(proj_fine(img))  # mash the fine forward projection
fast = proj_coarse(img)  # project directly along the averaged LORs

rel = float(
    np.linalg.norm(to_numpy_array(exact) - to_numpy_array(fast))
    / np.linalg.norm(to_numpy_array(fast))
)
print(
    "relative difference  ||mash_avg(P_fine x) - P_coarse x|| / ||P_coarse x|| = "
    f"{rel:.3f}"
)

exact_rvp = as_rvp(exact, mash_avg.coarse_lor_descriptor)
fast_rvp = as_rvp(fast, mash_avg.coarse_lor_descriptor)
vmax = float(max(exact_rvp.max(), fast_rvp.max()))
diff_rvp = exact_rvp - fast_rvp
dmax = float(np.abs(diff_rvp).max())

viewers.append(
    show_vol_cuts(
        exact_rvp,
        axis_labels=SINO_AXES,
        vmin=0,
        vmax=vmax,
        fig_title="Exact: mash_avg(P_fine x)",
        cmap="Greys",
    )
)
viewers.append(
    show_vol_cuts(
        fast_rvp,
        axis_labels=SINO_AXES,
        vmin=0,
        vmax=vmax,
        fig_title="Fast: P_coarse x",
        cmap="Greys",
    )
)
viewers.append(
    show_vol_cuts(
        diff_rvp,
        axis_labels=SINO_AXES,
        vmin=-dmax,
        vmax=dmax,
        fig_title="Difference (exact - fast)",
        cmap="RdBu",
    )
)

# %%
# The exact and fast sinograms look almost the same (relative difference of
# about 10 %).  The differences are largest at the edges of the projected
# objects, where the different widths of the sensitivity profiles matter
# most.

# %%
# 10. Upsampling a coarse sinogram to the fine grid
# -------------------------------------------------
#
# Sometimes a quantity is estimated cheaply on the coarse grid (a typical
# example is the **scatter expectation**) while the reconstruction runs on
# the fine grid, so it must be upsampled.  The **adjoint** of the mashing
# operator does this without extra dependencies.  The adjoint is **not** the
# inverse: mashing discards information, so ``mash.adjoint(mash(x)) != x``.
# The mode determines the normalisation:
#
# * ``mash.adjoint`` (``mode="sum"``) **replicates**: every fine bin gets the
#   value of its coarse bin unchanged.  This preserves the per-LOR
#   **value / rate** (each fine LOR inherits the coarse value) but does
#   **not** conserve the total: the sum grows by roughly the mashing factor.
#   Use it for rate-like quantities such as a scatter estimate or a forward
#   model.
# * ``mash_avg.adjoint`` (``mode="average"``) **spreads**: every fine bin
#   gets ``coarse / multiplicity``.  This **conserves the total counts** (the
#   fine bins of a group sum back to the coarse value) but lowers the
#   per-bin value.  Use it for counts that you want to redistribute.
#
# .. note::
#
#    Interpolation is another option for smooth quantities like scatter, but
#    the array API standard has no interpolation primitive: you would have
#    to implement linear interpolation yourself or use NumPy / SciPy.  The
#    two adjoints are array API compliant and need nothing extra.

coarse = mash(fine_sino)  # a coarse (counts) sinogram to upsample
up_replicate = mash.adjoint(coarse)  # sum-mode adjoint: copy (rate-preserving)
up_spread = mash_avg.adjoint(coarse)  # average-mode adjoint: count-preserving

print(f"sum(coarse)                  = {float(xp.sum(coarse)):.1f}")
print(
    f"sum(replicate, sum-adjoint)  = {float(xp.sum(up_replicate)):.1f}"
    "  (total NOT preserved)"
)
print(
    f"sum(spread, average-adjoint) = {float(xp.sum(up_spread)):.1f}  (total preserved)"
)

viewers.append(
    show_vol_cuts(
        as_rvp(up_replicate, lor_desc),
        axis_labels=SINO_AXES,
        cmap="Greys",
        fig_title="Upsampled with mash.adjoint (replicate, per-bin value preserved)",
    )
)
viewers.append(
    show_vol_cuts(
        as_rvp(up_spread, lor_desc),
        axis_labels=SINO_AXES,
        cmap="Greys",
        fig_title="Upsampled with mash_avg.adjoint (spread, total counts preserved)",
    )
)

# %%
# Both upsampled sinograms have the fine shape and the blocky structure of
# the coarse grid; they differ only by the per-bin factor
# ``1 / multiplicity``, which is visible in the colour scales.

# %%
# 11. Mashing GE-layout sinograms by composition
# ----------------------------------------------
#
# GE sinograms use the "span 3 in the centre" staircase segmentation:
# segment 0 pools ring differences ``{-1, 0, +1}`` into direct and cross
# planes, and the oblique segments pool ``{+/-2k, +/-(2k+1)}``.
# :class:`.SinogramMashingOperator` itself works on span-1 sinograms only,
# but GE mashing follows from **composition**, and the result is a plain
# span-1 coarse sinogram (without GE cross planes):
#
# 1. a span-1 <-> GE :class:`.SinogramAxialCompressionOperator` with
#    ``mode="average"``, whose **adjoint** distributes a GE sinogram back
#    onto the span-1 grid while preserving counts (each cross-plane count is
#    split evenly between its two ring differences);
# 2. the span-1 :class:`.SinogramMashingOperator`, which then mashes that
#    span-1 sinogram to a coarse span-1 sinogram.

ge_span1 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=3, span=1),
    radial_trim=11,
)

# span-1 -> GE (average mode); its adjoint distributes GE -> span-1
span1_to_ge_avg = parallelproj.pet_lors.SinogramAxialCompressionOperator(
    ge_span1, target_layout=parallelproj.pet_lors.MichelogramLayout.GE, mode="average"
)
ge_desc = span1_to_ge_avg.out_lor_descriptor  # the GE data layout

# span-1 detector mashing (same factors as above)
mash_span1 = parallelproj.pet_lors.SinogramMashingOperator(
    ge_span1, transaxial_factor=transaxial_factor, axial_factor=axial_factor, mode="sum"
)
span1_coarse_desc = mash_span1.coarse_lor_descriptor

# GE data --(distribute)--> fine span-1 --(mash)--> coarse span-1
ge_mash = CompositeLinearOperator([mash_span1, span1_to_ge_avg.H])

print(
    f"fine   GE sinogram   : {ge_mash.in_shape}  "
    f"(layout={ge_desc.michelogram.layout.name})"
)
print(
    f"mashed span-1 sinogram : {ge_mash.out_shape}  "
    f"(coarse rings={mash_span1.coarse_scanner.num_rings}, "
    f"layout={span1_coarse_desc.michelogram.layout.name}, "
    f"span={span1_coarse_desc.span})"
)

# %%
# The GE input Michelogram and the coarse span-1 output Michelogram:

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), layout="constrained")
ge_desc.show_michelogram(axes[0])
axes[0].set_title(
    f"Fine GE layout: {scanner.num_rings} rings, {ge_desc.num_planes} planes"
)
span1_coarse_desc.show_michelogram(axes[1])
axes[1].set_title(
    f"Mashed span 1: {mash_span1.coarse_scanner.num_rings} rings, "
    f"{span1_coarse_desc.num_planes} planes"
)
fig.show()

# %%
# In the GE Michelogram, neighbouring ring pairs are merged into one plane
# (connected by lines); the mashed span-1 Michelogram has one plane per pair
# of the 6 coarse rings.

# %%
# 12. GE detector sensitivity
# ---------------------------
#
# The projector traces a **single representative LOR per plane**, so the
# segment-0 **cross** planes, which physically collect two ring differences,
# are under-weighted.  We multiply the projection by the plane multiplicity
# (2 for cross planes, 1 otherwise) to give the cross planes their true
# double sensitivity before mashing.

plane_mult = to_numpy_array(ge_desc.michelogram.plane_multiplicity).astype("float64")
bshape = [1, 1, 1]
bshape[ge_desc.plane_axis_num] = ge_desc.spatial_sinogram_shape[ge_desc.plane_axis_num]
fine_phys_mult = xp.asarray(
    np.broadcast_to(plane_mult.reshape(bshape), ge_desc.spatial_sinogram_shape).copy(),
    dtype=xp.float64,
    device=dev,
)

viewers.append(
    show_vol_cuts(
        as_rvp(fine_phys_mult, ge_desc),
        axis_labels=SINO_AXES,
        fig_title="GE detector pairs per bin (1 direct / oblique, 2 cross)",
        cmap="viridis",
    )
)

# %%
# The multiplicity is constant within a plane, so the radial-view cut is
# uniform and the plane cuts show which planes are cross planes.

# %%
# 13. Mashing a simulated GE emission sinogram
# --------------------------------------------
#
# We forward-project the same phantom with the GE projector (weighted for the
# cross-plane sensitivity) and apply the composite operator.  It first
# distributes the GE sinogram onto the fine span-1 grid (the "un-GE" step)
# and then mashes it to a coarse span-1 sinogram.

proj_ge = parallelproj.projectors.RegularPolygonPETProjector(
    ge_desc, img_shape, voxel_size
)
ge_fine_sino = proj_ge(img) * fine_phys_mult  # GE data with cross planes at 2x
span1_fine_sino = span1_to_ge_avg.H(ge_fine_sino)  # distribute GE -> fine span-1
span1_coarse_sino = ge_mash(ge_fine_sino)  # = mash_span1(span1_fine_sino)

viewers.append(
    show_vol_cuts(
        as_rvp(ge_fine_sino, ge_desc),
        axis_labels=SINO_AXES,
        fig_title=f"Fine GE emission sinogram {tuple(ge_mash.in_shape)}",
    )
)
viewers.append(
    show_vol_cuts(
        as_rvp(span1_fine_sino, ge_span1),
        axis_labels=SINO_AXES,
        fig_title="Distributed to fine span 1 "
        f"{tuple(ge_span1.spatial_sinogram_shape)}",
    )
)
viewers.append(
    show_vol_cuts(
        as_rvp(span1_coarse_sino, span1_coarse_desc),
        axis_labels=SINO_AXES,
        fig_title=f"Mashed coarse span-1 sinogram {tuple(ge_mash.out_shape)}",
    )
)

# counts are preserved end to end (distribute preserves, sum-mash preserves)
print(
    f"sum(fine GE)          = {float(xp.sum(ge_fine_sino)):.1f}\n"
    f"sum(mashed span-1)    = {float(xp.sum(span1_coarse_sino)):.1f}"
    "  (counts preserved)"
)

# %%
# The three sinograms show the same object at fine GE, fine span-1 and
# coarse span-1 sampling, and the total count is the same in all of them.

# %%
# 14. Upsampling the coarse span-1 sinogram to the fine span-1 grid
# -----------------------------------------------------------------
#
# As in section 10, the adjoint of the span-1 mashing operator upsamples the
# coarse sinogram.  Since the sinogram was pooled with ``sum``, the
# count-preserving upsampling is the adjoint of the **average**-mode
# operator, which spreads each coarse bin over its fine bins.

mash_span1_avg = parallelproj.pet_lors.SinogramMashingOperator(
    ge_span1,
    transaxial_factor=transaxial_factor,
    axial_factor=axial_factor,
    mode="average",
)
span1_upsampled = mash_span1_avg.adjoint(span1_coarse_sino)  # counts preserved

print(
    f"sum(upsampled span-1) = {float(xp.sum(span1_upsampled)):.1f}  "
    f"(matches sum(mashed span-1) = {float(xp.sum(span1_coarse_sino)):.1f})"
)

viewers.append(
    show_vol_cuts(
        as_rvp(span1_upsampled, ge_span1),
        axis_labels=SINO_AXES,
        fig_title="Upsampled span-1 sinogram (average-mode adjoint) "
        f"{tuple(ge_span1.spatial_sinogram_shape)}",
    )
)

# %%
# 15. Round trip back to the fine GE sinogram
# -------------------------------------------
#
# Finally, we return to the original GE layout.  Recombining span-1 -> GE is
# the forward of the **sum**-mode span-1 <-> GE operator: each GE cross plane
# sums its two span-1 planes, which exactly undoes the earlier
# ``mode="average"`` distribution.  We combine upsampling and recombination
# into one operator that maps the coarse span-1 sinogram back to fine GE:
#
# .. code-block:: text
#
#    coarse span-1 --(average-mode adjoint of the mash)--> fine span-1
#                  --(sum span-1 -> GE)--> fine GE
#
# Counts are preserved; the result approximates the original fine GE
# sinogram up to the axial and transaxial resolution lost in mashing.

span1_to_ge_sum = parallelproj.pet_lors.SinogramAxialCompressionOperator(
    ge_span1, target_layout=parallelproj.pet_lors.MichelogramLayout.GE, mode="sum"
)
coarse_span1_to_ge = CompositeLinearOperator([span1_to_ge_sum, mash_span1_avg.H])

ge_reconstructed = coarse_span1_to_ge(span1_coarse_sino)  # fine GE (approx.)

rel_ge = float(
    np.linalg.norm(to_numpy_array(ge_reconstructed) - to_numpy_array(ge_fine_sino))
    / np.linalg.norm(to_numpy_array(ge_fine_sino))
)
print(
    f"sum(upsampled sino) = {float(xp.sum(ge_reconstructed)):.1f}"
    "  (counts preserved)\n"
    f"rel. difference to original fine GE = {rel_ge:.2f}"
    "  (resolution lost in mashing)"
)

viewers.append(
    show_vol_cuts(
        as_rvp(ge_reconstructed, ge_desc),
        axis_labels=SINO_AXES,
        fig_title="Upsampled sinogram in GE plane layout "
        f"{tuple(ge_desc.spatial_sinogram_shape)}",
    )
)

# %%
# Compared with the fine GE emission sinogram of section 13, the round-trip
# sinogram is blockier: the fine detail removed by mashing cannot be
# recovered by the adjoint.

# %%
# Summary
# -------
#
# - :class:`.SinogramMashingOperator` groups ``transaxial_factor`` crystals
#   within each side and ``axial_factor`` rings into virtual detectors at
#   the average position; ``mash.coarse_scanner`` and
#   ``mash.coarse_lor_descriptor`` describe the mashed geometry.
# - ``mode="sum"`` preserves counts (emission data), ``mode="average"``
#   averages (multiplicative factors) and approximates a projector on the
#   coarse descriptor.
# - The exact model ``mash(P_fine(x))`` and the fast coarse projector differ
#   by a resolution-dependent amount.
# - The adjoint upsamples: the sum-mode adjoint replicates (per-bin value
#   preserved), the average-mode adjoint spreads (total counts preserved).
# - GE-layout sinograms are mashed by composing the adjoint of an
#   average-mode :class:`.SinogramAxialCompressionOperator` with span-1
#   mashing.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_01_pet_geometry_08_run_tof_bin_mashing.py`
#    -- mashing of time-of-flight bins.  Sinogram symmetries, another way to
#    reduce computations, are shown in
#    :ref:`sphx_glr_auto_examples_01_pet_geometry_06_run_sinogram_symmetries.py`.

# sphinx_gallery_thumbnail_number = 1
