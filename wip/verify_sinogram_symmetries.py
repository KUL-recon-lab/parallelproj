"""
Verify sinogram symmetries for a uniform cylinder phantom.

For a centred, cylindrically symmetric object, equivalent sinogram bins
(identified by sinogram_symmetries) should carry the same expected count.
This script forward-projects a uniform cylinder, reduces the sinogram over
each symmetry class, then checks that the coefficient of variation (std/mean)
within each class is small — confirming that the symmetry holds up to the
discretisation error of the finite-voxel image.

Scanner setup:
  DemoPETScannerGeometry  (34 sides × 16 cryst/side = 544 cryst/ring, r≈380 mm)
  3 axial blocks × 3 rings/block = 9 rings total

Voxel size is set to 2 mm for speed.  Use 0.5 mm for a more stringent check
(forward projection takes longer but discretisation error is much smaller).
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import array_api_compat.numpy as xp
import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
import parallelproj.projectors as pp_proj
from parallelproj import to_numpy_array

from sinogram_symmetries import (
    compute_sinogram_plane_symmetries,
    build_plane_class_indices,
    build_view_class_indices,
    build_radial_class_indices,
    reduce_sinogram_by_symmetry_class,
)

# ── 1. Scanner ────────────────────────────────────────────────────────────────

RINGS_PER_BLOCK = 9
NUM_BLOCKS = 2
NUM_RINGS = RINGS_PER_BLOCK * NUM_BLOCKS  # 9

scanner = pps.DemoPETScannerGeometry(xp, "cpu", num_rings=NUM_RINGS, symmetry_axis=2)

print(
    f"Scanner : radius={scanner.radius:.1f} mm, "
    f"{scanner.num_sides} sides × {scanner.num_lor_endpoints_per_side} cryst/side "
    f"= {scanner.num_lor_endpoints_per_ring} cryst/ring,  "
    f"{NUM_RINGS} rings ({NUM_BLOCKS} blocks × {RINGS_PER_BLOCK})"
)

# ── 2. Span-1 sinogram descriptor ─────────────────────────────────────────────

MAX_RING_DIFF = RINGS_PER_BLOCK  # one block-width: includes one-gap planes
RADIAL_TRIM = 50  # keep num_rad manageable

lor_desc = ppl.RegularPolygonPETLORDescriptor(
    scanner,
    ppl.Michelogram(NUM_RINGS, max_ring_difference=MAX_RING_DIFF, span=1),
    radial_trim=RADIAL_TRIM,
    sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
)
num_rad, num_views = lor_desc.num_rad, lor_desc.num_views
print(
    f"Sinogram: {lor_desc.spatial_sinogram_shape}  "
    f"(num_rad={num_rad} [odd={num_rad%2==1}], "
    f"num_views={num_views}, num_planes={lor_desc.num_planes})"
)

# ── 3. Uniform cylinder phantom ───────────────────────────────────────────────
#
# Centred, circular cylinder filling the axial FOV.
# Voxel size 2 mm — fast to project.  Reduce to 0.5 mm for a tighter check.

VOXEL_SIZE = (1.0, 1.0, 1.0)  # mm
CYLINDER_RADIUS = 250.0  # mm, well inside the scanner FOV

ring_z = to_numpy_array(scanner.ring_positions).astype(float)
ring_spacing = float(np.abs(np.diff(ring_z)).mean())
block_width = ring_spacing * RINGS_PER_BLOCK  # axial extent of one block

# The axial-block-shift symmetry requires the phantom to look the same when
# shifted by one block_width in z.  So the cylinder must extend at least
# block_width beyond the outermost ring on each side.
z_half = float(np.abs(ring_z).max()) + block_width
n_z = int(2 * z_half / VOXEL_SIZE[2]) + 1
n_xy = int(2 * (CYLINDER_RADIUS + VOXEL_SIZE[0]) / VOXEL_SIZE[0]) + 1
img_shape = (n_xy, n_xy, n_z)

# Voxel-centre coordinates for a centred image
x_c = (np.arange(n_xy) - (n_xy - 1) / 2.0) * VOXEL_SIZE[0]

xx, yy = np.meshgrid(x_c, x_c, indexing="ij")
in_cyl = (xx**2 + yy**2) <= CYLINDER_RADIUS**2  # (n_xy, n_xy)

# Uniform cylinder: no axial masking — it fills the full image height so the
# phantom is z-invariant over the scanner FOV and one block_width beyond it.
phantom_np = np.zeros(img_shape, dtype=np.float32)
phantom_np[in_cyl, :] = 1.0

phantom = xp.asarray(phantom_np)
print(
    f"Phantom : {img_shape}, voxel {VOXEL_SIZE} mm, "
    f"cylinder r={CYLINDER_RADIUS} mm, "
    f"{int(phantom_np.sum())} active voxels"
)

# ── 4. Forward projection ─────────────────────────────────────────────────────

proj = pp_proj.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=VOXEL_SIZE
)
print("Forward projecting …", end=" ", flush=True)
sino = proj(phantom)
sino_np = to_numpy_array(sino).astype(np.float64)
print(
    f"done.  "
    f"Non-zero bins: {(sino_np > 0).sum()} / {sino_np.size}  "
    f"(range [{sino_np.min():.2f}, {sino_np.max():.2f}])"
)

# ── 5. Build symmetry index lists ─────────────────────────────────────────────

N_PER_SIDE = scanner.num_lor_endpoints_per_side  # view period

_, c2p, n_plane_classes = compute_sinogram_plane_symmetries(
    block_size=RINGS_PER_BLOCK,
    num_blocks=NUM_BLOCKS,
    max_ring_diff=MAX_RING_DIFF,
    n_edge=0,
)

view_idx = build_view_class_indices(num_views, N_PER_SIDE)
rad_idx = build_radial_class_indices(num_rad)
plane_idx = build_plane_class_indices(
    lor_desc.michelogram.plane_for_ring_pair_table,
    c2p,
    n_plane_classes,
)

print(
    f"Classes : {len(view_idx)} view (period n={N_PER_SIDE}, "
    f"{len(view_idx[0])} views each),  "
    f"{len(rad_idx)} radial,  "
    f"{n_plane_classes} plane"
)

# ── 6. Reduced sinogram ───────────────────────────────────────────────────────

sino_red = reduce_sinogram_by_symmetry_class(
    sino, view_idx, lor_desc.view_axis_num, xp.mean
)
sino_red = reduce_sinogram_by_symmetry_class(
    sino_red, rad_idx, lor_desc.radial_axis_num, xp.mean
)
sino_red = reduce_sinogram_by_symmetry_class(
    sino_red, plane_idx, lor_desc.plane_axis_num, xp.mean
)

compression = int(np.prod(sino.shape)) // int(np.prod(sino_red.shape))
print(
    f"Reduced : {tuple(sino.shape)} → {tuple(sino_red.shape)}  "
    f"({compression}× compression)"
)

# ── 7. Verify symmetry via CoV within equivalence classes ─────────────────────
#
# For each equivalence class, the sinogram values at all member bins should be
# (nearly) identical for a perfectly symmetric object.  We quantify residual
# spread via the coefficient of variation: CoV = std / mean.  A low CoV
# confirms the symmetry; the remaining spread comes from voxel discretisation.


def cov_within_classes(sino, class_idx_list, ax):
    """Mean and max CoV of sinogram values within each multi-member class."""
    results = []
    signal_max = sino.max()
    for class_idx in class_idx_list:
        if len(class_idx) < 2:
            continue
        vals = np.take(sino, class_idx, axis=ax)  # class axis at *ax*
        vals = np.moveaxis(vals, ax, -1)  # (..., class_size)
        mean = vals.mean(axis=-1)
        std = vals.std(axis=-1)
        sig = mean > 1e-4 * signal_max  # non-trivial signal only
        if sig.any():
            results.append((std[sig] / mean[sig]).mean())
    return np.asarray(results)


print("\nSymmetry quality — CoV of sinogram values within each equivalence class")
print("(low CoV ≈ symmetry holds; residual spread is voxel-discretisation noise)\n")

for label, sym_idx, sym_ax in [
    ("View   (scanner rotation,  period n)", view_idx, lor_desc.view_axis_num),
    ("Radial (FOV mirror, r ↔ num_rad−1−r)", rad_idx, lor_desc.radial_axis_num),
    ("Plane  (axial block-shift + z-mirror)", plane_idx, lor_desc.plane_axis_num),
]:
    covs = cov_within_classes(sino_np, sym_idx, sym_ax)
    if covs.size:
        print(
            f"  {label} :  "
            f"mean CoV = {covs.mean() * 100:.3f} %   "
            f"max CoV = {covs.max() * 100:.3f} %"
        )
    else:
        print(f"  {label} :  (no multi-member classes)")
