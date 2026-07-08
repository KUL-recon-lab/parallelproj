import matplotlib.pyplot as plt
from scipy.io import readsav

import parallelproj.pet_lors as lors
import parallelproj.pet_scanners as scanners
import parallelproj.projectors as projectors

from parallelproj import Array, to_numpy_array
from example_utils import show_vol_cuts

# %%
import array_api_compat.torch as xp

dev = "cuda"

# %%
r: Array = xp.arange(45, device=dev)
ring_positions = r * 5.31556 + (r // 9) * 2.8
ring_positions -= ring_positions.mean()  # center at 0

scanner = scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=320.37,  # (623.6 + 2*8.57)/2  -> apothem, DOI included
    num_sides=28,  # nrmodule
    num_lor_endpoints_per_side=16,  # modnrdet
    lor_spacing=4.03125,  # detsizemm_xy (mm)
    ring_positions=ring_positions,  # see below (mm)
    symmetry_axis=2,
    ring_endpoint_ordering=scanners.RingEndpointOrdering.CLOCKWISE,  # default
    phi0=0.0,  # default
)

lor_desc = lors.RegularPolygonPETLORDescriptor(
    scanner,
    lors.Michelogram.ge(
        # num_rings=scanner.num_rings, max_ring_difference=scanner.num_rings - 1
        num_rings=scanner.num_rings,
        max_ring_difference=1,
    ),
    radial_trim=45,  # 447 - 90 = 357 bins, central -> 178
    view_direction=lors.ViewDirection.PLUS,  # default
    radial_direction=lors.RadialDirection.MINUS,  # KUL radial-axis labelling
    zig_zag_order=lors.SinogramZigZagOrder.START_FIRST,
)

# %%

img_shape = (500, 500, 89)
vox_size = (0.5, 0.5, 2.79)

proj = projectors.RegularPolygonPETProjector(lor_desc, img_shape, vox_size)

# %%
# setup a sinogram that has a 1 in a single bin
sino1: Array = xp.zeros(proj.out_shape, dtype=xp.float32, device=dev)
sino2: Array = xp.zeros(proj.out_shape, dtype=xp.float32, device=dev)

rad_bin = 170  # lor_desc.num_rad // 2 - 8
view_bin = 0
plane_bin = 44

sino1[rad_bin - 1, view_bin, plane_bin] = 1.0
sino1[rad_bin, view_bin, plane_bin] = 1.4
sino1[rad_bin + 1, view_bin, plane_bin] = 2.0

sino2[rad_bin - 1, view_bin + 56, plane_bin] = 1.0
sino2[rad_bin, view_bin + 56, plane_bin] = 1.4
sino2[rad_bin + 1, view_bin + 56, plane_bin] = 2.0

# %%
# back project the sinogram
print("back projecting")
sino1_back = proj.adjoint(sino1)
sino2_back = proj.adjoint(sino2)

sino_back = sino1_back + sino2_back

# %%
vmax = max(float(sino1_back.max()), float(sino2_back.max()))

sl = img_shape[2] // 2
img_plane = to_numpy_array(sino_back[:, :, sl]).copy()

img_plane[0:5, 0:15] = vmax

# %%

img_plane_idl = readsav("img_plane_idl.sav")["img_plane"].copy().T
img_plane_idl *= img_plane.sum() / img_plane_idl.sum()


# %%
eps = img_plane_idl.max() / 1000

fig, ax = plt.subplots(
    1, 3, figsize=(18, 6), layout="constrained", sharex=True, sharey=True
)
ax[0].imshow(img_plane.T, origin="lower", vmin=0, vmax=vmax, cmap="Greys")
ax[0].set_xlabel("x0")
ax[0].set_ylabel("x1")

ax[1].imshow(img_plane_idl.T, origin="lower", vmin=0, vmax=vmax, cmap="Greys")
ax[1].set_xlabel("x0")
ax[1].set_ylabel("x1")

ax[2].imshow(
    ((img_plane - img_plane_idl) / (img_plane_idl + eps)).T,
    origin="lower",
    vmin=-0.05,
    vmax=0.05,
    cmap="seismic",
)
fig.show()

# %%

ix1 = 470

fig2, ax2 = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
ax2[0].plot(img_plane[:, ix1], "k-", drawstyle="steps-mid")
ax2[0].plot(img_plane_idl[:, ix1], "r:", drawstyle="steps-mid")

ax2[1].plot(
    (img_plane[:, ix1] - img_plane_idl[:, ix1]) / (img_plane_idl[:, ix1] + eps),
    "k-",
    drawstyle="steps-mid",
)
fig2.show()

# %%
# fig = plt.figure(figsize=(8, 8), tight_layout=True)
# ax = fig.add_subplot(111, projection="3d")
# scanner.show_lor_endpoints(ax)
# fig.show()
