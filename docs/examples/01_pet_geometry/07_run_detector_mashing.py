"""
Detector mashing: fewer, bigger virtual detectors
==================================================

*Mashing* groups neighbouring detectors into larger **virtual** detectors
located at the **average** endpoint position.  By reducing the number of
detectors it dramatically reduces the number of lines of response (LORs), which
shrinks the sinogram and speeds up reconstruction at the cost of spatial
resolution.

:class:`.SinogramMashingOperator` mashes a span-1 regular-polygon sinogram by

* ``transaxial_factor`` (:math:`N`) -- group :math:`N` neighbouring crystals
  **within each polygon side** (around the ring), and
* ``axial_factor`` (:math:`M`) -- group :math:`M` neighbouring **rings**
  (along the symmetry axis).

Because averaging uniformly spaced within-side crystals (and ring positions)
again gives a regular polygon, the mashed geometry is itself a
:class:`.RegularPolygonPETScannerGeometry` (``mash.coarse_scanner``) with a
matching :class:`.RegularPolygonPETLORDescriptor` (``mash.coarse_lor_descriptor``).
So there are two ways to model the mashed data:

* the **exact** model -- mash the fine forward projection: ``mash(P_fine(x))``;
* the **fast** model -- project directly along the averaged LORs with a
  :class:`.RegularPolygonPETProjector` built on ``mash.coarse_lor_descriptor``.

``mode="sum"`` preserves counts (use it for emission / measured data);
``mode="average"`` averages and matches the fast coarse projector (use it for
multiplicative factors such as attenuation or normalisation).

.. note::

   To run this example locally, download
   `example_utils.py <https://raw.githubusercontent.com/KUL-recon-lab/parallelproj/main/docs/examples/example_utils.py>`_
   into the **same folder** as this script. Make sure ``parallelproj`` is installed.
"""

# %%
import numpy as np
import matplotlib.pyplot as plt

import parallelproj.pet_scanners
import parallelproj.pet_lors
import parallelproj.projectors
from parallelproj import to_numpy_array

# %%
from example_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu")
xp, dev = suggest_array_backend_and_device(None, None)

# %%
# A fine ("true") scanner and its sinogram descriptor
# ---------------------------------------------------
#
# A cylindrical scanner with 28 sides, 4 crystals per side (112 crystals per
# ring) and 8 rings.

num_rings = 8
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=110.0,
    num_sides=28,
    num_lor_endpoints_per_side=4,
    lor_spacing=4.0,
    ring_positions=xp.linspace(-14.0, 14.0, num_rings, device=dev),
    symmetry_axis=2,
)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=7, span=1),
    radial_trim=10,
)

# %%
# Mash 2 crystals around the ring and 2 rings axially
# ---------------------------------------------------
#
# ``transaxial_factor`` must divide the number of crystals per side and
# ``axial_factor`` must divide the number of rings.  Here both are 2.

mash = parallelproj.pet_lors.SinogramMashingOperator(
    lor_desc, transaxial_factor=2, axial_factor=2, mode="sum"
)
coarse_desc = mash.coarse_lor_descriptor

n_fine = int(np.prod(mash.in_shape))
n_coarse = int(np.prod(mash.out_shape))
print(mash)
print(f"fine   sinogram shape : {mash.in_shape}  ({n_fine} LORs)")
print(f"mashed sinogram shape : {mash.out_shape}  ({n_coarse} LORs)")
print(f"LOR reduction factor  : {n_fine / n_coarse:.1f}x")

# %%
# Where do the virtual detectors sit?
# -----------------------------------
#
# The mashed (virtual) crystals lie at the **average** position of the
# within-side blocks they replace.  We plot one ring, transaxially: small dots
# are the fine crystals, large crosses the mashed virtual crystals.

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

fig1, ax1 = plt.subplots(figsize=(6, 6), tight_layout=True)
ax1.scatter(fine_pts[:, tax[0]], fine_pts[:, tax[1]], s=12, color="tab:blue",
            label=f"fine crystals ({fine_pts.shape[0]}/ring)")
ax1.scatter(coarse_pts[:, tax[0]], coarse_pts[:, tax[1]], s=90, marker="x",
            color="tab:red", label=f"mashed virtual ({coarse_pts.shape[0]}/ring)")
ax1.set_aspect("equal")
ax1.set_title("Transaxial detector mashing (one ring, N=2)")
ax1.legend(loc="upper right", fontsize="small")
fig1.show()

# %%
# Mash a (simulated) emission sinogram
# ------------------------------------
#
# Forward-project a simple phantom through the fine projector to get a fine
# emission sinogram, then mash it with ``mode="sum"`` (counts add).

img_shape = (100, 100, num_rings)
voxel_size = (4.0, 4.0, 3.5)
proj_fine = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape, voxel_size
)

img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
img[30:60, 40:55, :] = 1.0
img[60:75, 60:75, :] = 2.0

fine_sino = proj_fine(img)
mashed_sino = mash(fine_sino)

# show the central plane of each sinogram (radial x view)
rax, vax, pax = lor_desc.radial_axis_num, lor_desc.view_axis_num, lor_desc.plane_axis_num


def _central_plane(sino, desc):
    s = to_numpy_array(sino)
    s = np.moveaxis(s, (desc.radial_axis_num, desc.view_axis_num, desc.plane_axis_num),
                    (0, 1, 2))
    return s[:, :, s.shape[2] // 2]


fig2, ax2 = plt.subplots(1, 2, figsize=(11, 5), tight_layout=True)
ax2[0].imshow(_central_plane(fine_sino, lor_desc).T, aspect="auto", cmap="Greys")
ax2[0].set_title(f"fine sinogram (central plane)\n{mash.in_shape}")
ax2[0].set_xlabel("radial")
ax2[0].set_ylabel("view")
ax2[1].imshow(_central_plane(mashed_sino, coarse_desc).T, aspect="auto", cmap="Greys")
ax2[1].set_title(f"mashed sinogram (central plane)\n{mash.out_shape}")
ax2[1].set_xlabel("radial")
ax2[1].set_ylabel("view")
fig2.show()

# %%
# Fast coarse projector vs. the exact mashed model
# ------------------------------------------------
#
# A projector built on ``mash.coarse_lor_descriptor`` projects directly along
# the averaged LORs (cheap).  With ``mode="average"`` it approximates the
# averaged bundle of fine LORs, i.e. ``mash_avg(P_fine(x)) ~ P_coarse(x)``.  The
# two are not identical -- a single averaged LOR has a slightly narrower
# sensitivity profile than the bundle it replaces -- so they agree up to a
# resolution-dependent difference (shown below).  Use the exact ``mash(P_fine)``
# model when that difference matters, and the fast coarse projector otherwise.

mash_avg = parallelproj.pet_lors.SinogramMashingOperator(
    lor_desc, transaxial_factor=2, axial_factor=2, mode="average"
)
proj_coarse = parallelproj.projectors.RegularPolygonPETProjector(
    mash_avg.coarse_lor_descriptor, img_shape, voxel_size
)

exact = mash_avg(proj_fine(img))   # mash the fine forward projection
fast = proj_coarse(img)            # project directly along the averaged LORs

rel = float(
    np.linalg.norm(to_numpy_array(exact) - to_numpy_array(fast))
    / np.linalg.norm(to_numpy_array(fast))
)
print(f"relative difference  ||mash_avg(P_fine x) - P_coarse x|| / ||P_coarse x|| = {rel:.3f}")

fig3, ax3 = plt.subplots(1, 3, figsize=(15, 5), tight_layout=True)
e = _central_plane(exact, mash_avg.coarse_lor_descriptor).T
f = _central_plane(fast, mash_avg.coarse_lor_descriptor).T
vmax = float(max(e.max(), f.max()))
ax3[0].imshow(e, aspect="auto", cmap="Greys", vmin=0, vmax=vmax)
ax3[0].set_title("exact:  mash_avg(P_fine x)")
ax3[1].imshow(f, aspect="auto", cmap="Greys", vmin=0, vmax=vmax)
ax3[1].set_title("fast:   P_coarse x")
ax3[2].imshow(e - f, aspect="auto", cmap="RdBu")
ax3[2].set_title("difference")
for a in ax3:
    a.set_xlabel("radial")
    a.set_ylabel("view")
fig3.show()

plt.show()
