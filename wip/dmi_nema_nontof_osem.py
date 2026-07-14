import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path
from scipy.io import loadmat
from copy import copy

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

num_epochs_osem = 4
num_subsets = 34

res_model_fwhm_mm = 3.5

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

# mult_op = operators.ElementwiseMultiplicationOperator(mult_corrections)

res_model = operators.GaussianFilterOperator(
    proj.in_shape,
    sigma=[res_model_fwhm_mm / (2.35 * float(vs)) for vs in proj.voxel_size],
)


# %%
# split the forward model into subsets

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# clear the cached LOR endpoints since we will create many copies of the projector
proj.clear_cached_lor_endpoints()
pet_subset_linop_seq = []

# we setup a sequence of subset forward operators each constisting of
# (1) image-based resolution model
# (2) subset projector
# (3) multiplication with the corresponding subset of the attenuation sinogram
for i in range(num_subsets):
    # make a copy of the full projector and reset the views to project
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    subset_mult_op = operators.ElementwiseMultiplicationOperator(
        mult_corrections[subset_slices_non_tof[i]]
    )

    # add the resolution model and multiplication with a subset of the attenuation sinogram
    pet_subset_linop_seq.append(
        operators.CompositeLinearOperator(
            [
                subset_mult_op,
                subset_proj,
                res_model,
            ]
        )
    )

pet_subset_linop_seq = operators.LinearOperatorSequence(pet_subset_linop_seq)

# %%
# setup the subset data fidelities

# the strictly positive contamination guarantees A x + s > 0 in every bin,
# so the exact (unmodified) log-likelihood can be used
exact_mode = bool(xp.min(contamination) > 0)

subset_data_fidelities = [
    functions.C2AffineObjective(
        functions.NegPoissonLogL(prompts[sl], exact=exact_mode),
        pet_subset_linop_seq[k],
        contamination[sl],
    )
    for k, sl in enumerate(subset_slices)
]


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
# caluclate subset adjoint ones (sensitivity images per subset)
subset_adjoint_ones = xp.zeros((num_subsets,) + img_shape, dtype=xp.float32, device=dev)
for k, op in enumerate(pet_subset_linop_seq):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

## full sensitivity image A^T 1 = sum of all subset sensitivities
# adjoint_ones = xp.sum(subset_adjoint_ones, axis=0)


# %%
# run OSEM

# setup FOV mask
cyl_mask = proj.fov_mask()
x_init = xp.astype(cyl_mask, xp.float32)
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask
del cyl_mask

x_osem = xp.asarray(x_init, copy=True)


for i in range(num_epochs_osem):
    for k in range(len(subset_slices)):
        print(
            f"OSEM epoch {(i+1):04} / {num_epochs_osem:04}, subset {(k+1):04} / {num_subsets:04}",
            end="\r",
        )
        x_osem = em_update(
            x_osem, subset_data_fidelities[k], subset_adjoint_ones[k], fov_mask
        )
print()


# %%
_, _, _ = show_vol_cuts(to_numpy_array(x_osem), voxel_size=vox_size)

plt.show()
