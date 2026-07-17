import matplotlib.pyplot as plt
import numpy as np
import h5py

from pathlib import Path
from scipy.io import loadmat
from copy import copy

import parallelproj.operators as operators
import parallelproj.functions as functions
import parallelproj.pet_lors as lors
import parallelproj.projectors as projectors
import parallelproj.tof as tof

from parallelproj import to_numpy_array, Array


from parallelproj._examples_utils import show_vol_cuts

import array_api_compat.torch as xp


# %%
def _load_sino_mat(sino_path: Path, arr_mod, device, verbose=True):

    if verbose:
        print(f"loading {sino_path}")
    sino_np = np.ascontiguousarray(np.swapaxes(loadmat(sino_path)["data"].copy(), 1, 2))

    return arr_mod.asarray(sino_np, device=device)


# %%
dev = "cpu"
img_shape = (200, 200, 71)
vox_size = (2.0, 2.0, 2.78)

num_epochs_osem = 4
num_subsets = 34

res_model_fwhm_mm = 3.5

data_path = Path("dmi_nema")

# %%
print("Setting up lor descriptor and projector")
lor_desc = lors.get_lor_descriptor_G2(xp, dev)
proj = projectors.RegularPolygonPETProjector(lor_desc, img_shape, vox_size)
proj.tof_parameters = tof.get_tof_parameters_G2()


# %%
# read the prompts sinogram
print("reading TOF prompts")

prompts_npy_path = data_path / "tofPrompts_f1b1.npy"

if prompts_npy_path.exists():
    prompts_np = np.load(prompts_npy_path)
else:
    prompts_np = np.zeros(proj.out_shape, dtype=np.int8)

    # load TOF prompts from HDF file
    with h5py.File(data_path / "tofPrompts_f1b1.mat", "r") as prompts_h5_data:
        for i in range(lor_desc.num_views):
            prompts_np[:, i, :, :] = prompts_h5_data[f"view{i+1}"][:].T

    np.save(prompts_npy_path, prompts_np)


prompts = xp.asarray(prompts_np, device=dev)
del prompts_np


# %%
# read all multiplicative corrections for the fwd model
mult_corr_file = data_path / "mult_corrections.npy"

if mult_corr_file.exists():
    mult_corrections = xp.asarray(np.load(mult_corr_file), device=dev)
else:
    dtPuc = _load_sino_mat(data_path / "dtPuc_f1b1.mat", xp, dev)
    norm = _load_sino_mat(data_path / "norm.mat", xp, dev)
    acfs = _load_sino_mat(data_path / "acf_f1b1.mat", xp, dev)
    mult_corrections = acfs * dtPuc / norm
    np.save(mult_corr_file, to_numpy_array(mult_corrections))

# %%
# read all additive corrections for the fwd model (contaminations)

contam_file = data_path / "contaminations_tof.npy"

if contam_file.exists():
    print("read contamination")
    contamination = xp.asarray(np.load(contam_file), device=dev)
else:
    # scatter = _load_sino_mat(data_path / "scatter_f1b1.mat", xp, dev)
    # randoms = _load_sino_mat(data_path / "randoms_f1b1.mat", xp, dev)

    print("read TOF randoms")
    # read the randoms, and distribute across tof bins
    tmp = (
        _load_sino_mat(data_path / "randoms_f1b1.mat", xp, dev)
        / proj.tof_parameters.num_tofbins
    )
    tof_randoms = xp.broadcast_to(xp.expand_dims(tmp, axis=-1), proj.out_shape)

    print("read TOF scatter")

    with h5py.File(data_path / "tofScatter_f1b1.mat", "r") as x:
        tof_scatter = xp.asarray(
            np.moveaxis(x["data"], [0, 1, 2, 3], [1, 3, 2, 0]),
            device=dev,
            dtype=xp.float32,
        )

    contamination = tof_scatter + tof_randoms
    np.save(contam_file, to_numpy_array(contamination))

    del tof_scatter
    del tof_randoms

# %%
# setup the complete fwd model

print("setting up fwd model")

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
        xp.broadcast_to(
            xp.expand_dims(mult_corrections[subset_slices_non_tof[i]], axis=-1),
            subset_proj.out_shape,
        )
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
    return xp.clip(x_cur - d * data_fidelity.gradient(x_cur), 0, None)


# %%
# caluclate subset adjoint ones (sensitivity images per subset)
print("Calculating adjoint ones")
subset_adjoint_ones = xp.zeros((num_subsets,) + img_shape, dtype=xp.float32, device=dev)
for k, op in enumerate(pet_subset_linop_seq):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

# %%
# run OSEM

# setup FOV mask
cyl_mask = proj.fov_mask()
x_init = xp.astype(cyl_mask, xp.float32)
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask
del cyl_mask

x_osem = xp.asarray(x_init, copy=True)

x_intermed = np.zeros((num_epochs_osem,) + img_shape, dtype=np.float32)

for i in range(num_epochs_osem):
    for k in range(len(subset_slices)):
        print(
            f"OSEM epoch {(i+1):04} / {num_epochs_osem:04}, subset {(k+1):04} / {num_subsets:04}",
            end="\r",
        )
        x_osem = em_update(
            x_osem, subset_data_fidelities[k], subset_adjoint_ones[k], fov_mask
        )
    x_intermed[i, ...] = to_numpy_array(x_osem)

np.save(data_path / "tof_osem.npy", x_intermed)

print()


# %%
_, _, _ = show_vol_cuts(x_intermed, voxel_size=vox_size)

plt.show()
