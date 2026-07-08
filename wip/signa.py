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
sino: Array = xp.zeros(proj.out_shape, dtype=xp.float32, device=dev)

rad_bin = 170
view_bins = [0, 56, 63]
plane_bins = [0, 44, 81]

for plane_bin in plane_bins:
    for view_bin in view_bins:
        sino[rad_bin - 1, view_bin, plane_bin] = 1.0
        sino[rad_bin, view_bin, plane_bin] = 1.4
        sino[rad_bin + 1, view_bin, plane_bin] = 2.0

# %%
# back project the sinogram
print("back projecting")
sino_back = to_numpy_array(proj.adjoint(sino))

sino_back_idl = np.swapaxes(readsav("sino_back.sav")["sino_back"].copy(), 0, 2)
sino_back_idl *= sino_back.sum() / sino_back_idl.sum()

sino_back_idl = np.flip(sino_back_idl, (0, 1))

print(np.sum((sino_back - sino_back_idl) ** 2))

# %%
vmax = max(float(sino_back.max()), float(sino_back_idl.max()))
ims = dict(vmin=0, origin="lower", cmap="Greys")
ims2 = dict(vmin=-0.02 * vmax, vmax=0.02 * vmax, origin="lower", cmap="seismic")

ncols = len(plane_bins)

fig, ax = plt.subplots(
    3,
    ncols,
    figsize=(ncols * 4.2, 3 * 4.2),
    layout="constrained",
    sharex=True,
    sharey=True,
)

for i, pl in enumerate(plane_bins):
    ax[0, i].imshow(sino_back[:, :, pl].T, vmax=vmax, **ims)
    ax[1, i].imshow(sino_back_idl[:, :, pl].T, vmax=vmax, **ims)
    ax[2, i].imshow((sino_back[:, :, pl] - sino_back_idl[:, :, pl]).T, **ims2)
    ax[0, i].set_title(f"parallelproj backproj image plane {pl}", fontsize="medium")
    ax[1, i].set_title(f"KUL IDL backproj image plane {pl}", fontsize="medium")
    ax[2, i].set_title(f"parallelproj - KUL image plane {pl}", fontsize="medium")

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
