import matplotlib.pyplot as plt
import numpy as np
from scipy.io import readsav

import parallelproj.pet_lors as lors
import parallelproj.projectors as projectors
import parallelproj.tof as tof

from parallelproj import Array, to_numpy_array

import array_api_compat.torch as xp

# %%
dev = "cuda"
img_shape = (275, 275, 89)
vox_size = (2.0, 2.0, 2.79)
show_michelogram = False


# %%
lor_desc = lors.get_lor_descriptor_G1(xp, dev, max_ring_difference=7)
proj = projectors.RegularPolygonPETProjector(lor_desc, img_shape, vox_size)

proj.tof_parameters = tof.TOFParameters(
    num_tofbins=27,
    tofbin_width=0.169 * 0.5 * tof.C_MM_PER_NS,
    sigma_tof=0.385 * 0.5 * tof.C_MM_PER_NS / 2.35,
)

proj.tof = True


# %%
if show_michelogram:
    fig_m, ax_m = plt.subplots(figsize=(9, 9))
    lor_desc.michelogram.show(ax_m, plane_index_fontsize=8)
    ax_m.set_xlim(-0.5, 11)
    ax_m.set_ylim(-0.5, 11)
    fig_m.show()


# %%
# setup a sinogram that has a 1 in a single bin
sino: Array = xp.zeros(proj.out_shape, dtype=xp.float32, device=dev)

rad_bin = 170
view_bins = [0, 74, 148]
plane_bins = [0, 44, 81, 89, 173, 174, 258, 259, 420, 498, 536]
tof_bins = [10, 19]

for ip, plane_bin in enumerate(plane_bins):
    for view_bin in view_bins:
        for tof_bin in tof_bins:
            sino[rad_bin, view_bin, plane_bin, tof_bin] = 1.4 + 0.1 * ip
            sino[rad_bin + 11, view_bin, plane_bin, tof_bin] = 1.6 + 0.1 * ip

# %%
# back project the sinogram
print("back projecting")
sino_back = to_numpy_array(proj.adjoint(sino))

sino_back_idl = np.swapaxes(
    readsav("tof_sino_back_ob.sav")["sino_back"].copy().squeeze(), 0, 2
)
sino_back_idl *= sino_back.sum() / sino_back_idl.sum()

# need to flip the left right direction (x0/x direction) because of different defs of the coordinate systems
sino_back_idl = np.flip(sino_back_idl, 0)

# %%
vmax = max(float(sino_back.max()), float(sino_back_idl.max()))
ims = dict(vmin=0, origin="lower", cmap="Greys")
ims2 = dict(vmin=-0.03 * vmax, vmax=0.03 * vmax, origin="lower", cmap="seismic")
ims3 = dict(vmin=-0.10, vmax=0.10, origin="lower", cmap="seismic")


img_sls = [0, 1, 2, 43, 44, 81, 87]

ncols = len(img_sls)

fig, ax = plt.subplots(
    4,
    ncols,
    figsize=(ncols * 2.0, 4 * 2.0),
    layout="constrained",
    sharex=True,
    sharey=True,
)

for i, pl in enumerate(img_sls):
    ax[0, i].imshow(sino_back[:, :, pl].T, vmax=vmax, **ims)
    ax[1, i].imshow(sino_back_idl[:, :, pl].T, vmax=vmax, **ims)
    ax[2, i].imshow((sino_back[:, :, pl] - sino_back_idl[:, :, pl]).T, **ims2)
    ax[3, i].imshow(
        (
            (sino_back[:, :, pl] - sino_back_idl[:, :, pl])
            / (sino_back_idl[:, :, pl] + sino_back_idl[:, :, pl].max() * 0.01)
        ).T,
        **ims2,
    )
    ax[0, i].set_title(f"pp bp {pl}", fontsize="medium")
    ax[1, i].set_title(f"IDL bp {pl}", fontsize="medium")
    ax[2, i].set_title(f"pp - IDL {pl}", fontsize="medium")
    ax[3, i].set_title(f"(pp - IDL)/IDL {pl}", fontsize="medium")

fig.show()
