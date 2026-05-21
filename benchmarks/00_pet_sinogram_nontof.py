import time
import argparse
import os
import pandas as pd
import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
import parallelproj_core
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument("--num_subsets", type=int, default=34)
parser.add_argument(
    "--mode", default="GPU", choices=["GPU", "GPU-torch", "CPU", "CPU-torch", "hybrid"]
)
parser.add_argument("--threadsperblock", type=int, default=32)
parser.add_argument("--output_file", type=int, default=None)
parser.add_argument("--output_dir", default="results")
parser.add_argument(
    "--sinogram_orders", default=["PVR", "PRV", "VPR", "VRP", "RPV", "RVP"], nargs="+"
)
parser.add_argument("--symmetry_axes", default=["0", "1", "2"], nargs="+")

args = parser.parse_args()

if args.mode == "GPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import array_api_compat.cupy as xp

    dev = xp.cuda.Device(0)
elif args.mode == "GPU-torch":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import array_api_compat.torch as xp

    dev = "cuda"
elif args.mode == "hybrid":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import array_api_compat.numpy as xp

    dev = "cpu"
elif args.mode == "CPU":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import array_api_compat.numpy as xp

    dev = "cpu"
elif args.mode == "CPU-torch":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import array_api_compat.torch as xp

    dev = "cpu"
else:
    raise ValueError

num_runs = args.num_runs
threadsperblock = args.threadsperblock
num_subsets = args.num_subsets

output_dir = args.output_dir
if args.output_file is None:
    output_file = f"nontof_sinogram__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numsubsets_{num_subsets}.csv"

# image properties
num_trans = 215
num_ax = 71
voxel_size = xp.asarray([2.78, 2.78, 2.78], dtype=xp.float32, device=dev)

# scanner properties
num_rings = 36

sinogram_orders = args.sinogram_orders
symmetry_axes = [int(x) for x in args.symmetry_axes]

# ---------------------------------------------------------------------

df = pd.DataFrame()

num_rings = 36
ring_positions = (
    5.31556 * xp.arange(num_rings, device=dev, dtype=xp.float32)
    + (xp.astype(xp.arange(num_rings, device=dev) // 9, xp.float32)) * 2.8
)
ring_positions -= 0.5 * xp.max(ring_positions)

for ia, symmetry_axis in enumerate(symmetry_axes):
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=0.5 * (744.1 + 2 * 8.51),
        num_sides=34,
        num_lor_endpoints_per_side=16,
        lor_spacing=4.03125,
        ring_positions=ring_positions,
        symmetry_axis=symmetry_axis,
    )

    # setup a box like test image
    img_shape = [num_trans, num_trans, num_trans]
    img_shape[symmetry_axis] = num_ax
    img_shape = tuple(img_shape)
    n0, n1, n2 = img_shape

    # setup an image containing a square
    img = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    sl = [
        slice(n0 // 4, 3 * n0 // 4, None),
        slice(n1 // 4, 3 * n1 // 4, None),
        slice(n2 // 4, 3 * n2 // 4, None),
    ]

    sl[symmetry_axis] = slice(0, img.shape[symmetry_axis], None)
    sl = tuple(sl)
    img[sl] = 1

    # setup the image origin = the coordinate of the [0,0,0] voxel
    img_origin = (
        -(xp.asarray(img.shape, dtype=xp.float32, device=dev) / 2) + 0.5
    ) * voxel_size

    for io, sinogram_order in enumerate(sinogram_orders):
        lor_descriptor = ppl.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=65,
            sinogram_order=ppl.SinogramSpatialAxisOrder[sinogram_order],
        )
        views = xp.arange(0, lor_descriptor.num_views, num_subsets, device=dev)

        xstart, xend = lor_descriptor.get_lor_coordinates(views)

        print(sinogram_order)
        print(symmetry_axis, img_shape)

        ones = xp.ones(xstart.shape[:-1], dtype=xp.float32, device=dev)

        for ir in range(num_runs + 1):
            # perform a complete fwd projection
            img_fwd = xp.zeros(xstart.shape[:-1], dtype=xp.float32, device=dev)
            t0 = time.time()
            parallelproj_core.joseph3d_fwd(
                xstart,
                xend,
                img,
                img_origin,
                voxel_size,
                img_fwd,
                threads_per_block=threadsperblock,
            )
            t1 = time.time()

            # perform a complete backprojection
            t2 = time.time()
            back_img = xp.zeros(img.shape, dtype=xp.float32, device=dev)
            parallelproj_core.joseph3d_back(
                xstart,
                xend,
                back_img,
                img_origin,
                voxel_size,
                ones,
                threads_per_block=threadsperblock,
            )
            t3 = time.time()
            if ir > 0:
                tmp = pd.DataFrame(
                    {
                        "sinogram order": sinogram_order,
                        "symmetry axis": str(symmetry_axis),
                        "run": ir,
                        "t forward (s)": t1 - t0,
                        "t back (s)": t3 - t2,
                    },
                    index=[0],
                )
                df = pd.concat((df, tmp))

# ----------------------------------------------------------------------------
# save results
df["data"] = "nontof_sinogram"
df["mode"] = args.mode
df["num_subsets"] = num_subsets
df["threadsperblock"] = threadsperblock

Path(output_dir).mkdir(exist_ok=True, parents=True)
df.to_csv(os.path.join(output_dir, output_file), index=False)

# ----------------------------------------------------------------------------
# show results

sns.set_context("paper")

df["t forward+back (s)"] = df["t forward (s)"] + df["t back (s)"]

fig, ax = plt.subplots(1, 3, figsize=(7, 7 / 3), sharex=False, sharey="row")

bplot_kwargs = dict(capsize=0.15, err_kws={"linewidth": 1.5}, errorbar="sd")

sns.barplot(
    data=df,
    x="sinogram order",
    y="t forward (s)",
    hue="symmetry axis",
    ax=ax[0],
    **bplot_kwargs,
)
sns.barplot(
    data=df,
    x="sinogram order",
    y="t back (s)",
    hue="symmetry axis",
    ax=ax[1],
    **bplot_kwargs,
)
sns.barplot(
    data=df,
    x="sinogram order",
    y="t forward+back (s)",
    hue="symmetry axis",
    ax=ax[2],
    **bplot_kwargs,
)

sns.move_legend(ax[0], "upper right", ncol=2)
for i, axx in enumerate(ax.ravel()):
    axx.grid(ls=":")
    if i > 0:
        axx.get_legend().remove()

fig.show()
