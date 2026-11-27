"""
Zig-zag sampling of LORs in a sinogram view
===========================================

:bdg-primary:`geometry` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

Within one sinogram view, a regular polygon PET scanner connects pairs of
in-ring detector endpoints in a zig-zag pattern: moving from the central LOR
toward the sinogram edge, the start and the end detector take turns stepping
to their next position.  Two conventions exist for which of the two steps
first, and they lead to different (start, end) pairs in each radial bin, which
matters when sinograms are exchanged with other software.

**You will learn**

- what the zig-zag sampling of LORs within a sinogram view is,
- the two conventions of :class:`.SinogramZigZagOrder`: ``END_FIRST``
  (default) and ``START_FIRST``,
- how to select a convention with the ``zig_zag_order`` parameter of
  :class:`.RegularPolygonPETLORDescriptor`,
- how to read the (start, end) detector pairs of each view from
  ``start_in_ring_index`` and ``end_in_ring_index``.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_01_pet_geometry_02_run_regular_polygon_pet_sino.py`
(how LORs are sorted into sinogram views and radial bins).
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import matplotlib.pyplot as plt

import parallelproj.pet_lors
import parallelproj.pet_scanners
from parallelproj import to_numpy_array
from parallelproj._examples_utils import suggest_array_backend_and_device

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. A minimal scanner
# --------------------
#
# To keep the pattern readable we use a single ring with 8 detector endpoints
# (one per side of an octagon) and no radial trimming, so that every LOR of a
# view appears in the sinogram.

n_endpoints = 8
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=100.0,
    num_sides=n_endpoints,
    num_lor_endpoints_per_side=1,
    lor_spacing=1.0,
    ring_positions=xp.asarray([0.0], device=dev),
    symmetry_axis=2,
)

# %%
# 2. LOR descriptors for both zig-zag conventions
# -----------------------------------------------
#
# The ``zig_zag_order`` parameter of :class:`.RegularPolygonPETLORDescriptor`
# selects the convention:
#
# * ``SinogramZigZagOrder.END_FIRST`` (default): moving outward from the
#   central LOR in the direction of increasing radial bin, the *end* detector
#   steps to its next position before the start detector does.  The pairs of
#   view 0 are (6,5), (7,5), (7,4), **(0,4)**, (0,3), (1,3), (1,2).
# * ``SinogramZigZagOrder.START_FIRST``: the *start* detector steps first.
#   The pairs of view 0 are (7,6), (7,5), (0,5), **(0,4)**, (1,4), (1,3), (2,3).
#
# In both cases the central radial bin (bold) is the LOR (0,4).

ZigZag = parallelproj.pet_lors.SinogramZigZagOrder

lor_end_first = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=0,
    zig_zag_order=ZigZag.END_FIRST,
)

lor_start_first = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    radial_trim=0,
    zig_zag_order=ZigZag.START_FIRST,
)

# %%
# 3. Detector pairs per view
# --------------------------
#
# ``start_in_ring_index[view, radial_bin]`` and
# ``end_in_ring_index[view, radial_bin]`` hold the in-ring indices of the two
# detectors of every LOR.  We print the (start, end) pairs of all views for
# both conventions.

for name, lor_desc in [("END_FIRST", lor_end_first), ("START_FIRST", lor_start_first)]:
    print(f"{name} -- (start, end) detector pairs per view:")
    for view in range(lor_desc.num_views):
        s = lor_desc.start_in_ring_index[view, :].tolist()
        e = lor_desc.end_in_ring_index[view, :].tolist()
        print(f"  view {view}: {list(zip(s, e))}")
    if name == "END_FIRST":
        print()

# %%
# In every view, radial bins at an even distance from the central bin (bins 1,
# 3 and 5 here) contain the same LOR in both conventions, while bins at an odd
# distance (0, 2, 4 and 6) connect different detector pairs.  A sinogram stored
# with one convention therefore cannot be read with the other one without
# reordering.

# %%
# 4. LORs of view 0
# -----------------
#
# The figure draws the LORs of view 0 at the true detector positions in the
# front view of the scanner (``x0`` to the right, ``x1`` down, see
# :ref:`sphx_glr_auto_examples_01_pet_geometry_01_run_regular_polygon_pet_scanner.py`),
# coloured by radial bin.  The legend lists the (start, end) pair of each bin.

endpoints = to_numpy_array(
    scanner.get_lor_endpoints(
        xp.zeros(n_endpoints, dtype=xp.int32, device=dev),
        xp.arange(n_endpoints, device=dev),
    )
)
x0_det, x1_det = endpoints[:, 0], endpoints[:, 1]

cmap = plt.colormaps["tab10"].resampled(lor_end_first.num_rad)
view = 0

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), layout="constrained")

for ax, lor_desc, title in zip(
    axes,
    [lor_end_first, lor_start_first],
    ["END_FIRST (default)", "START_FIRST"],
):
    # detector endpoints with their in-ring index
    ax.scatter(x0_det, x1_det, color="k", zorder=5, s=40)
    for idx in range(n_endpoints):
        ax.text(
            1.14 * x0_det[idx],
            1.14 * x1_det[idx],
            str(idx),
            ha="center",
            va="center",
        )

    # LORs of view 0, coloured by radial bin
    s_idx = lor_desc.start_in_ring_index[view, :].tolist()
    e_idx = lor_desc.end_in_ring_index[view, :].tolist()
    for rad_bin, (si, ei) in enumerate(zip(s_idx, e_idx)):
        ax.plot(
            [x0_det[si], x0_det[ei]],
            [x1_det[si], x1_det[ei]],
            color=cmap(rad_bin),
            lw=2,
            label=f"bin {rad_bin}: ({si},{ei})",
        )

    ax.set_xlim(-135, 135)
    ax.set_ylim(135, -135)  # x1 points down in the front view
    ax.set_aspect("equal")
    ax.set_xlabel("x0 [mm]")
    ax.set_ylabel("x1 [mm]")
    ax.legend(fontsize="x-small", loc="center left", bbox_to_anchor=(1.0, 0.5))
    ax.set_title(f"view {view}, {title}")

fig.suptitle(f"Zig-zag LOR sampling, {n_endpoints} detectors, radial_trim=0")
fig.show()

# %%
# Bins 1, 3 and 5 (including the central LOR (0,4)) are identical in both
# panels.  In the other bins, END_FIRST has moved the end detector one step
# further and START_FIRST the start detector, so the same radial bin connects
# different detectors and the LORs are tilted in opposite directions.

# %%
# Summary
# -------
#
# - Within a sinogram view, the radial bins sample LORs in a zig-zag pattern
#   in which start and end detector alternately step to their next position.
# - :class:`.SinogramZigZagOrder` selects which detector steps first:
#   ``END_FIRST`` (default) or ``START_FIRST``; pass it as ``zig_zag_order``
#   to :class:`.RegularPolygonPETLORDescriptor`.
# - ``start_in_ring_index`` and ``end_in_ring_index`` give the detector pair of
#   every (view, radial bin).
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_01_pet_geometry_06_run_sinogram_symmetries.py`
#    -- symmetries of the sinogram that relate views, radial bins and planes.

# sphinx_gallery_thumbnail_number = 1
