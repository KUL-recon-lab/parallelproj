"""
The Michelogram and axial sinogram compression
==============================================

A *Michelogram* is a diagram of which ring pairs ``(s, e)`` form valid
coincidences in a cylindrical PET scanner, and how they are grouped into
sinogram planes under Siemens / STIR axial compression conventions.  Ring
pairs are sorted by *segment* (a function of the ring difference
``rd = e - s``) and, within each segment, by *axial midpoint* ``s + e``.

This example introduces

* :class:`.Michelogram` -- captures the segment / axial-position layout
  in pure integer space, independently of any scanner geometry;
* :class:`.SinogramAxialCompressionOperator` -- the linear operator that
  uses a Michelogram to compress a span-1 sinogram into a higher-span
  sinogram by summing the ring-pair sinograms that fold into the same
  compressed plane.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array

# %%
from array_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)

# %%
# A Michelogram, standalone
# -------------------------
#
# A :class:`.Michelogram` is built from three integers:
#
# * ``num_rings`` -- the number of detector rings,
# * ``max_ring_difference`` -- the maximum ``|e - s|`` considered,
# * ``span`` -- an odd axial compression factor (``1`` = no compression).
#
# It knows nothing about ring z-positions or scanner radius -- it is a
# combinatorial object describing the ``(segment, axial midpoint)`` layout
# of sinogram planes.  Each point in the Michelogram plot is one valid
# ring pair ``(start_ring, end_ring)``, coloured by ``|segment|``;
# numerals annotate the resulting sinogram plane index.

num_rings = 13

m_span1 = parallelproj.pet_lors.Michelogram(
    num_rings=num_rings,
    max_ring_difference=num_rings - 1,
    span=1,
)

print(repr(m_span1))
print(f"num_planes       = {m_span1.num_planes}")
print(f"max_multiplicity = {m_span1.max_multiplicity}")

fig, ax = plt.subplots(figsize=(6, 6), tight_layout=True)
m_span1.show(ax)
fig.show()

# %%
# Effect of ``span`` and ``max_ring_difference``
# ----------------------------------------------
#
# Increasing the **span** merges ring pairs that share both a segment and
# an axial midpoint into one sinogram plane.  In the Michelogram plot,
# merged ring pairs are connected by thin grey *merge lines* — visually,
# each grey line collapses into one plane index.
#
# Restricting the **max_ring_difference** removes the outer segments
# entirely, shrinking the diagonal band of valid ring pairs.

configs = [
    (1, num_rings - 1),  # span=1, all ring differences
    (3, num_rings - 1),  # span=3, all ring differences
    (5, num_rings - 1),  # span=5, all ring differences
    (5, 5),              # span=5, max_ring_difference restricted
]

fig, axes = plt.subplots(2, 2, figsize=(11, 11), tight_layout=True)
for ax, (span, mrd) in zip(axes.flat, configs):
    m = parallelproj.pet_lors.Michelogram(
        num_rings=num_rings,
        max_ring_difference=mrd,
        span=span,
    )
    m.show(ax)
    ax.set_title(
        f"span={span}, max_ring_difference={mrd}\n"
        f"-> num_planes = {m.num_planes}, "
        f"max_multiplicity = {m.max_multiplicity}",
        fontsize="small",
    )
fig.show()

# %%
# Compressing a forward-projected sinogram
# ----------------------------------------
#
# :class:`.SinogramAxialCompressionOperator` takes a span-1 LOR descriptor
# and a target odd span, and produces a linear operator that maps a span-1
# sinogram to a span-``target_span`` sinogram via
#
# .. math::
#
#     y_n = \\sum_{p_1 \\,\\in\\, \\mathcal{G}(n)} x_{p_1}\\,,
#
# where :math:`\\mathcal{G}(n)` is the set of span-1 plane indices that
# fold into target plane :math:`n`.  Its adjoint replicates each output
# value back to every input plane that contributed to it.
#
# To see the operator in action, we set up a small 5-ring scanner, build
# a span-1 projector, forward-project a small Gaussian phantom, then
# compress the resulting span-1 sinogram to span 5.

num_rings_small = 5
target_span = 5

scanner_small = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=12,
    num_lor_endpoints_per_side=4,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-4, 4, num_rings_small, device=dev),
    symmetry_axis=2,
)

# span-1 descriptor (no ring-difference constraint)
lor_s1 = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner_small,
    parallelproj.pet_lors.Michelogram(
        scanner_small.num_rings,
        max_ring_difference=num_rings_small - 1,
        span=1,
    ),
    radial_trim=10,
)

# span-1 forward projector
img_shape = (32, 32, 11)
voxel_size = (2.0, 2.0, 1.0)

proj_s1 = parallelproj.projectors.RegularPolygonPETProjector(
    lor_s1, img_shape=img_shape, voxel_size=voxel_size
)

# axial compression operator (span 1 -> span target_span)
op = parallelproj.pet_lors.SinogramAxialCompressionOperator(
    lor_s1, target_span=target_span
)

print(op)
print(f"span-1 sinogram shape: {proj_s1.out_shape}")
print(f"span-{target_span} sinogram shape: {op.out_shape}")

# %%
# A tiny 3-D Gaussian phantom, centered off-axis at world coordinates
# ``(x, y, z) = (0, 10, 0) mm`` with isotropic ``sigma = 4 mm``.  The
# image lives in numpy; we convert to ``xp`` for the projector.

ii, jj, kk = np.meshgrid(
    np.arange(img_shape[0]),
    np.arange(img_shape[1]),
    np.arange(img_shape[2]),
    indexing="ij",
)
x_w = (ii - (img_shape[0] - 1) / 2) * voxel_size[0]
y_w = (jj - (img_shape[1] - 1) / 2) * voxel_size[1]
z_w = (kk - (img_shape[2] - 1) / 2) * voxel_size[2]
phantom_np = np.exp(
    -((x_w - 0.0) ** 2 + (y_w - 10.0) ** 2 + (z_w - 0.0) ** 2) / (2 * 4.0**2)
).astype(np.float32)

phantom = xp.asarray(phantom_np, device=dev)

# Forward-project at span 1, then axially compress to span ``target_span``.
sino_s1 = proj_s1(phantom)
sino_sn = op(sino_s1)

# %%
# Visualise: a maximum-intensity projection of the 3-D phantom along the
# ``y`` axis (so the axial structure is visible), and the resulting span-1
# and span-``target_span`` sinograms for the same view.  The *plane* axis
# of each sinogram encodes axial position; compressing reduces the number
# of plane bins (and increases per-bin values because of the summation).

view_idx = 0
s1_np = to_numpy_array(sino_s1)[:, view_idx, :]
sn_np = to_numpy_array(sino_sn)[:, view_idx, :]

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), tight_layout=True)

axes[0].imshow(
    phantom_np.max(axis=1).T, origin="lower", cmap="gray", aspect="auto"
)
axes[0].set_title("phantom (max-intensity projection along $y$)")
axes[0].set_xlabel("x voxel")
axes[0].set_ylabel("z voxel")

im1 = axes[1].imshow(s1_np, origin="lower", cmap="inferno", aspect="auto")
axes[1].set_title(
    f"span-1 sinogram (view {view_idx})\n"
    f"shape = {s1_np.shape} -> {op.num_planes_in} planes"
)
axes[1].set_xlabel("plane index $p_1$")
axes[1].set_ylabel("radial bin")
fig.colorbar(im1, ax=axes[1])

im2 = axes[2].imshow(sn_np, origin="lower", cmap="inferno", aspect="auto")
axes[2].set_title(
    f"span-{target_span} compressed sinogram (view {view_idx})\n"
    f"shape = {sn_np.shape} -> {op.num_planes_out} planes"
)
axes[2].set_xlabel("plane index $n$")
axes[2].set_ylabel("radial bin")
fig.colorbar(im2, ax=axes[2])

fig.show()
