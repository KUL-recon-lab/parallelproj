"""
TOF-bin mashing: fewer, wider time-of-flight bins
=================================================

:bdg-primary:`geometry` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

TOF *mashing* groups neighbouring time-of-flight (TOF) bins into fewer, wider
bins.  This shrinks the TOF axis of a sinogram (less memory, faster
reconstruction) at the cost of TOF resolution, and is independent of any
detector / LOR mashing along the spatial axes.
:class:`.TOFBinMashingOperator` groups every ``mashing_factor`` (:math:`G`)
neighbouring TOF bins.  It only acts on the **trailing** (TOF) axis; all
leading (spatial sinogram) axes pass through unchanged.

**You will learn**

- how to mash the TOF axis of a sinogram with :class:`.TOFBinMashingOperator`
  in ``mode="sum"`` (for counts-like data) and ``mode="average"`` (for
  multiplicative factors),
- why sum-mashing a fine TOF projection equals projecting directly onto the
  coarse TOF grid given by ``.coarse_tof_parameters``,
- how the adjoint upsamples a mashed sinogram back to the fine TOF grid, and
  how the two modes differ there,
- how to combine TOF mashing with spatial :class:`.SinogramMashingOperator`.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_07_run_detector_mashing.py`
(spatial sinogram mashing).
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
# TOF sinograms are 4D arrays ``(radial, view, plane, TOF)``.  To display them
# with ``show_vol_cuts`` we move the TOF axis to the front, so that the viewer
# gets a slider for the TOF bin and shows spatial cuts for the selected bin.

import matplotlib.pyplot as plt
import numpy as np

import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import to_numpy_array
from parallelproj._examples_utils import show_vol_cuts, suggest_array_backend_and_device
from parallelproj.operators import CompositeLinearOperator

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# ``show_vol_cuts`` returns interactive widgets (when run with an interactive
# matplotlib backend); keep references so their callbacks are not
# garbage-collected.
keep_alive = []

sino_axis_labels = ("TOF bin", "radial", "view", "plane")


def tof_first(sino):
    """Move the trailing TOF axis to the front:
    (radial, view, plane, TOF) -> (TOF, radial, view, plane)"""
    return np.moveaxis(to_numpy_array(sino), -1, 0)


# %%
# 1. A TOF scanner, sinogram descriptor and projector
# ---------------------------------------------------
#
# A small cylindrical scanner with 4 rings (span 1, maximum ring difference 1)
# is enough to illustrate the TOF axis.

num_rings = 4
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=100.0,
    num_sides=12,
    num_lor_endpoints_per_side=4,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-6.0, 6.0, num_rings, device=dev),
    symmetry_axis=2,
)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=1, span=1),
    radial_trim=3,
)

# %%
# The fine TOF grid has 27 bins.  ``tofbin_width`` and ``sigma_tof`` are in mm
# (see :class:`.TOFParameters` for the conversion from ns to mm).

fine_tof = parallelproj.tof.TOFParameters(
    num_tofbins=27,
    tofbin_width=20.0,
    sigma_tof=30.0,
    num_sigmas=3.0,
)

# %%
# We forward-project a simple emission image with two rectangular hot regions
# onto the fine TOF grid.  Setting ``tof_parameters`` on the projector enables
# TOF.

img_shape = (40, 40, num_rings)
voxel_size = (4.0, 4.0, 4.0)

img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
img[10:18, 18:30, :] = 1.0
img[24:30, 12:22, :] = 2.0

proj_fine = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape, voxel_size
)
proj_fine.tof_parameters = fine_tof

fine_sino = proj_fine(img)  # shape: spatial_sinogram_shape + (27,)
print(f"fine TOF sinogram shape : {tuple(fine_sino.shape)}")

# %%
# 2. Mashing the TOF bins
# -----------------------
#
# We group every ``G = 3`` neighbouring TOF bins (27 -> 9 bins);
# ``mashing_factor`` must divide ``num_tofbins``.  The two modes are:
#
# * ``mode="sum"`` (default): a coarse bin is the **sum** of its :math:`G` fine
#   bins.  Use it for counts-like data.  A TOF-bin weight is a Gaussian
#   integrated over the bin, and integrals over adjacent bins add up exactly,
#   so sum-mashing a TOF forward projection is (up to the ``num_sigmas``
#   truncation) **identical** to projecting directly onto the coarse TOF grid.
# * ``mode="average"``: a coarse bin is the **mean** of its :math:`G` fine
#   bins.  Use it for multiplicative factors.
#
# The matching coarse TOF parameters (``num_tofbins`` divided by :math:`G`,
# ``tofbin_width`` multiplied by :math:`G`) are available as
# ``.coarse_tof_parameters``, so that a projector can be set up directly on the
# mashed grid.  Only the trailing TOF axis of the sinogram changes.

G = 3
tof_mash = parallelproj.pet_lors.TOFBinMashingOperator(
    fine_tof,
    lor_desc.spatial_sinogram_shape,
    mashing_factor=G,
    mode="sum",
)
print(tof_mash)
print(f"norm (sum mode)         : {tof_mash.norm():.4f}  (== sqrt({G}))")

coarse_tof = tof_mash.coarse_tof_parameters
print(
    f"coarse TOF parameters   : num_tofbins={coarse_tof.num_tofbins}, "
    f"tofbin_width={coarse_tof.tofbin_width} mm, sigma_tof={coarse_tof.sigma_tof} mm"
)

mashed_sino = tof_mash(fine_sino)  # shape: spatial_sinogram_shape + (9,)
print(f"mashed TOF sinogram     : {tuple(mashed_sino.shape)}")

# %%
# The operator norm of the sum-mode operator is :math:`\sqrt{G}`, since every
# coarse bin sums :math:`G` fine bins.  The TOF resolution ``sigma_tof`` is a
# property of the scanner and is not changed by mashing.

# %%
# 3. The TOF sinograms before and after mashing
# ---------------------------------------------
#
# ``show_vol_cuts`` takes the TOF bin as the leading (slider) axis and shows
# the spatial cuts (radial, view, plane) of the selected TOF bin.  In the
# documentation, a static figure of the central TOF bin is shown; in an
# interactive session you can scroll through the TOF bins.

keep_alive.append(
    show_vol_cuts(
        tof_first(fine_sino),
        axis_labels=sino_axis_labels,
        fig_title=f"fine TOF sinogram ({fine_tof.num_tofbins} TOF bins)",
    )
)
keep_alive.append(
    show_vol_cuts(
        tof_first(mashed_sino),
        axis_labels=sino_axis_labels,
        fig_title=f"mashed TOF sinogram ({coarse_tof.num_tofbins} TOF bins, G={G})",
    )
)

# %%
# The central TOF bins of the two sinograms show a similar spatial pattern.
# The mashed sinogram has 9 instead of 27 TOF bins, and its values are larger
# because each coarse bin collects the counts of three fine bins.

# %%
# 4. TOF profile before and after mashing
# ---------------------------------------
#
# We pick the spatial sinogram bin with the most counts and plot its TOF
# profile on both grids.  The TOF-bin centres are given in mm along the LOR,
# symmetric about 0.

fine_np = to_numpy_array(fine_sino)
mashed_np = to_numpy_array(mashed_sino)

tof_axis = fine_sino.ndim - 1
spatial_counts = fine_np.sum(axis=tof_axis)
peak = np.unravel_index(np.argmax(spatial_counts), spatial_counts.shape)

fine_profile = fine_np[peak]  # (27,)
mashed_profile = mashed_np[peak]  # (9,)

fine_centers = (np.arange(fine_tof.num_tofbins) - (fine_tof.num_tofbins - 1) / 2) * (
    fine_tof.tofbin_width
)
coarse_centers = (
    np.arange(coarse_tof.num_tofbins) - (coarse_tof.num_tofbins - 1) / 2
) * coarse_tof.tofbin_width

fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
ax.bar(
    fine_centers,
    fine_profile,
    width=fine_tof.tofbin_width * 0.9,
    alpha=0.6,
    label=f"fine ({fine_tof.num_tofbins} bins)",
)
ax.bar(
    coarse_centers,
    mashed_profile,
    width=coarse_tof.tofbin_width * 0.9,
    facecolor="none",
    edgecolor="C1",
    linewidth=2.0,
    label=f"mashed ({coarse_tof.num_tofbins} bins, G={G})",
)
ax.set_xlabel("TOF position along the LOR [mm]")
ax.set_ylabel("counts")
ax.set_title(f"TOF profile of spatial bin {tuple(int(i) for i in peak)}")
ax.legend()
fig.show()

# %%
# The 27 fine bins collapse into 9 bins that are three times wider.  The
# height of each coarse bin is the sum of the three fine bins it covers, so
# the total number of counts is preserved.

# %%
# 5. Exactness of the sum-mashed forward model
# --------------------------------------------
#
# For ``mode="sum"``, mashing the fine TOF projection is equivalent to
# projecting directly onto the coarse TOF grid (``coarse_tof_parameters``).
# We verify this with a second projector on the same geometry but with the
# coarse TOF parameters.

proj_coarse = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape, voxel_size
)
proj_coarse.tof_parameters = coarse_tof
coarse_sino_direct = proj_coarse(img)

rel_err = float(
    np.linalg.norm(mashed_np - to_numpy_array(coarse_sino_direct))
    / np.linalg.norm(to_numpy_array(coarse_sino_direct))
)
print(f"|| mash(P_fine x) - P_coarse x || / || P_coarse x || = {rel_err:.2e}")

keep_alive.append(
    show_vol_cuts(
        tof_first(coarse_sino_direct),
        axis_labels=sino_axis_labels,
        fig_title=f"direct coarse-TOF projection ({coarse_tof.num_tofbins} TOF bins)",
    )
)

# %%
# The relative difference is at the level of single-precision round-off, and
# the directly projected coarse TOF sinogram is visually identical to the
# mashed one of section 3.

# %%
# 6. Upsampling to the fine TOF grid
# ----------------------------------
#
# The adjoint maps a coarse TOF sinogram back onto the fine TOF grid.  The two
# modes give two different upsamplings:
#
# * the **sum**-mode adjoint *replicates* each coarse bin into its :math:`G`
#   fine bins.  The summed counts are copied unchanged, so the values are about
#   :math:`G` times larger than in the original fine sinogram (this is the
#   exact transpose used inside reconstruction, not a rescaled view of the
#   data);
# * the **average**-mode adjoint additionally divides by :math:`G`, *spreading*
#   each coarse bin's counts evenly over its fine bins.  Each fine bin then
#   holds the group **mean**, so the result is on the same scale as, and
#   directly comparable to, the original fine TOF sinogram (a
#   piecewise-constant approximation along the TOF axis).
#
# The average-mode adjoint is therefore the one to use when the upsampled
# sinogram is to be compared with the fine one.
#
# .. note::
#
#    To **upsample** a coarse (mashed) sinogram back to the fine TOF grid, use
#    the operator's ``adjoint``.  The adjoint is **not** the inverse: mashing
#    discards information, so ``tof_mash.adjoint(tof_mash(x)) != x``.  Choose
#    the mode by what you want to keep: the ``mode="sum"`` adjoint
#    **replicates** the coarse value into every fine bin (per-bin value
#    preserved, total grows by about G), while the ``mode="average"`` adjoint
#    **spreads** it (divides by G; total counts preserved, per-bin value lower
#    and comparable to the fine data).
#
# Both adjoints pass the adjointness test.

tof_mash_avg = parallelproj.pet_lors.TOFBinMashingOperator(
    fine_tof,
    lor_desc.spatial_sinogram_shape,
    mashing_factor=G,
    mode="average",
)

replicated = tof_mash.adjoint(mashed_sino)  # sum-mode adjoint: ~G x too large
upsampled = tof_mash_avg.adjoint(mashed_sino)  # average-mode adjoint: fine scale
assert tof_mash.adjointness_test(xp, dev, dtype=xp.float64)
assert tof_mash_avg.adjointness_test(xp, dev, dtype=xp.float64)

rel_up = float(
    np.linalg.norm(to_numpy_array(upsampled) - fine_np) / np.linalg.norm(fine_np)
)
print(
    f"upsampled shape         : {tuple(upsampled.shape)}  (back on the fine TOF grid)"
)
print(f"upsampled (avg-adjoint) vs fine, rel. difference = {rel_up:.2f}")
total_ratio = float(xp.sum(replicated) / xp.sum(fine_sino))
print(f"total of sum-adjoint / total of fine sinogram = {total_ratio:.2f}  (~ G)")

keep_alive.append(
    show_vol_cuts(
        tof_first(upsampled),
        axis_labels=sino_axis_labels,
        fig_title=f"upsampled sinogram (avg-adjoint, {fine_tof.num_tofbins} TOF bins)",
    )
)

# %%
# The upsampled sinogram lives on the fine 27-bin TOF grid and matches the
# original fine sinogram up to the averaging within each group of three TOF
# bins, which causes the relative difference of about 0.3.

# %%
# 7. Combination with detector mashing
# ------------------------------------
#
# TOF-bin mashing composes with :class:`.SinogramMashingOperator`, which
# mashes the spatial LOR axes.  We compress the TOF axis first and then mash
# the geometry; :class:`.CompositeLinearOperator` gives the combined operator
# (and its adjoint).  The spatial operator must be told how many TOF bins
# remain after TOF mashing.

spatial_mash = parallelproj.pet_lors.SinogramMashingOperator(
    lor_desc,
    transaxial_factor=2,
    axial_factor=2,
    mode="sum",
    num_tof_bins=coarse_tof.num_tofbins,
)

# operators are applied from right to left: TOF first, then spatial
full_mash = CompositeLinearOperator([spatial_mash, tof_mash])
full_sino = full_mash(fine_sino)

n_in = int(np.prod(full_mash.in_shape))
n_out = int(np.prod(full_mash.out_shape))
print(f"combined mashing        : {full_mash.in_shape} -> {full_mash.out_shape}")
print(f"total data reduction    : {n_in / n_out:.1f}x")

# %%
# Summary
# -------
#
# - :class:`.TOFBinMashingOperator` groups ``mashing_factor`` neighbouring TOF
#   bins and only acts on the trailing TOF axis.
# - ``mode="sum"`` is for counts; sum-mashing a TOF projection equals a direct
#   projection with ``.coarse_tof_parameters``.  ``mode="average"`` is for
#   multiplicative factors.
# - The adjoint upsamples to the fine TOF grid: it replicates (sum mode) or
#   spreads (average mode) each coarse bin; it is not an inverse.
# - :class:`.CompositeLinearOperator` combines TOF mashing with spatial
#   :class:`.SinogramMashingOperator` (pass ``num_tof_bins``).
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_01_run_pet_non_tof_sinogram_projector.py`
#    -- the first projector example: non-TOF sinogram projections.
#    TOF projections are covered in
#    :ref:`sphx_glr_auto_examples_02_pet_sinogram_projections_02_run_pet_tof_sinogram_projector.py`.

# sphinx_gallery_thumbnail_number = 3
