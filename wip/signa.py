import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

import parallelproj.pet_lors as lors
import parallelproj.pet_scanners as scanners
import parallelproj.projectors as projectors

from parallelproj import Array, to_numpy_array

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
        max_ring_difference=3,
        segment_order=lors.SegmentOrder.NEGATIVE_FIRST,
    ),
    radial_trim=45,  # 447 - 90 = 357 bins, central -> 178
    view_direction=lors.ViewDirection.PLUS,  # default
    radial_direction=lors.RadialDirection.MINUS,  # KUL radial-axis labelling
    zig_zag_order=lors.SinogramZigZagOrder.START_FIRST,
)

fig_m, ax_m = plt.subplots(figsize=(9, 9))
lor_desc.michelogram.show(ax_m, plane_index_fontsize=8)
ax_m.set_xlim(-0.5, 11)
ax_m.set_ylim(-0.5, 11)
fig_m.show()

# %%

img_shape = (500, 500, 89)
vox_size = (0.5, 0.5, 2.79)

proj = projectors.RegularPolygonPETProjector(lor_desc, img_shape, vox_size)

# %%
# setup a sinogram that has a 1 in a single bin
sino: Array = xp.zeros(proj.out_shape, dtype=xp.float32, device=dev)

rad_bin = 170
view_bins = [0, 56, 63]
plane_bins = [0, 44, 81, 89, 173, 174, 258]

for ip, plane_bin in enumerate(plane_bins):
    for view_bin in view_bins:
        sino[rad_bin - 1, view_bin + ip, plane_bin] = 1.0 + 0.1 * ip
        sino[rad_bin, view_bin + ip, plane_bin] = 1.4 + 0.1 * ip
        sino[rad_bin + 1, view_bin + ip, plane_bin] = 2.0 + 0.1 * ip

# %%
# back project the sinogram
print("back projecting")
sino_back = to_numpy_array(proj.adjoint(sino))

sino_back_idl = np.swapaxes(
    readsav("sino_back.sav")["sino_back"].copy().squeeze(), 0, 2
)
sino_back_idl *= sino_back.sum() / sino_back_idl.sum()

# need to flip the left right direction (x0/x direction) because of different defs of the coordinate systems
sino_back_idl = np.flip(sino_back_idl, 0)

# %%
vmax = max(float(sino_back.max()), float(sino_back_idl.max()))
ims = dict(vmin=0, origin="lower", cmap="Greys")
ims2 = dict(vmin=-0.02 * vmax, vmax=0.02 * vmax, origin="lower", cmap="seismic")


img_sls = [0, 1, 44, 81, 87, 88]

ncols = len(img_sls)

fig, ax = plt.subplots(
    3,
    ncols,
    figsize=(ncols * 3.0, 3 * 3.0),
    layout="constrained",
    sharex=True,
    sharey=True,
)

for i, pl in enumerate(img_sls):
    ax[0, i].imshow(sino_back[:, :, pl].T, vmax=vmax, **ims)
    ax[1, i].imshow(sino_back_idl[:, :, pl].T, vmax=vmax, **ims)
    ax[2, i].imshow((sino_back[:, :, pl] - sino_back_idl[:, :, pl]).T, **ims2)
    ax[0, i].set_title(f"pp bp {pl}", fontsize="medium")
    ax[1, i].set_title(f"IDL bp {pl}", fontsize="medium")
    ax[2, i].set_title(f"pp - IDL {pl}", fontsize="medium")

fig.show()

fig2, ax2 = plt.subplots(
    3,
    3,
    figsize=(4 * 4.2, 1 * 4.2),
    layout="constrained",
    sharex=True,
    sharey=True,
)

for i, x1 in enumerate([50, 250, 400]):
    ax2[0, i].imshow(sino_back[:, x1, :].T, **ims)
    ax2[1, i].imshow(sino_back_idl[:, x1, :].T, **ims)
    ax2[2, i].imshow((sino_back[:, x1, :] - sino_back_idl[:, x1, :]).T, **ims2)
    ax2[0, i].set_title(f"parallelproj backproj cor plane {x1}", fontsize="medium")
    ax2[1, i].set_title(f"KUL IDL backproj cor plane {x1}", fontsize="medium")
    ax2[2, i].set_title(f"parallelproj - KUL cor plane {x1}", fontsize="medium")

fig2.show()
