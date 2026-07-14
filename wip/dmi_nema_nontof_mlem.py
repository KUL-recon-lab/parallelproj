import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.io import loadmat

import parallelproj.operators as operators
import parallelproj.functions as functions
import parallelproj.pet_lors as lors
import parallelproj.projectors as projectors
import parallelproj.tof as tof

from parallelproj import to_numpy_array, Array
from example_utils import show_vol_cuts

import array_api_compat.torch as xp

# %%
dev = "cuda"
img_shape = (200, 200, 71)
vox_size = (2.0, 2.0, 2.78)

# %%
lor_desc = lors.get_lor_descriptor_G2(xp, dev)
proj = projectors.RegularPolygonPETProjector(lor_desc, img_shape, vox_size)

# proj.tof_parameters = tof.get_tof_parameters_G1()

# %%
# read all on-TOF sionograms we need


def _load_sino(fname: str, verbose=True):
    data_path = Path("dmi_nema") / fname
    if verbose:
        print(f"loading {data_path}")
    return xp.asarray(
        np.ascontiguousarray(np.swapaxes(loadmat(data_path)["data"].copy(), 1, 2)),
        device=dev,
    )


prompts = _load_sino("prompts_f1b1.mat")
dtPuc = _load_sino("dtPuc_f1b1.mat")
norm = _load_sino("norm.mat")


acfs = _load_sino("acf_f1b1.mat")
scatter = _load_sino("scatter_f1b1.mat")
randoms = _load_sino("randoms_f1b1.mat")

# %%
mult_corrections = acfs * dtPuc / norm
contamination = scatter + randoms

mult_op = operators.ElementwiseMultiplicationOperator(mult_corrections)

res_model = operators.GaussianFilterOperator(
    proj.in_shape, sigma=[2.0 / (2.35 * float(vs)) for vs in proj.voxel_size]
)

# compose all 3 operators into a single linear operator
pet_lin_op = operators.CompositeLinearOperator((mult_op, proj, res_model))


full_data_fidelity = functions.C2AffineObjective(
    functions.NegPoissonLogL(prompts, exact=True), pet_lin_op, contamination
)


# %%
def em_update(
    x_cur: Array,
    data_fidelity: functions.C1Function,
    adj_ones: Array,
    img_mask: Array | None = None,
) -> Array:
    if img_mask is None:
        d = x_cur / adj_ones
    else:
        d = xp.where(img_mask, x_cur / adj_ones, xp.zeros_like(x_cur))
    return x_cur - d * data_fidelity.gradient(x_cur)


# %%
# run MLEM

# setup FOV mask
cyl_mask = proj.fov_mask()
x_init = xp.astype(cyl_mask, xp.float32)
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask
del cyl_mask

adjoint_ones = pet_lin_op.adjoint(
    xp.ones(pet_lin_op.out_shape, dtype=xp.float32, device=dev)
)

num_epochs_mlem = 100

df_mlem = xp.zeros(num_epochs_mlem, dtype=xp.float32, device=dev)
x_mlem = xp.asarray(x_init, copy=True)

for i in range(num_epochs_mlem):
    print(f"MLEM epoch {(i + 1):04} / {num_epochs_mlem:04}", end="\r")
    x_mlem = em_update(x_mlem, full_data_fidelity, adjoint_ones, fov_mask)
    df_mlem[i] = full_data_fidelity(x_mlem)
print()

_, _, _ = show_vol_cuts(to_numpy_array(x_mlem))
