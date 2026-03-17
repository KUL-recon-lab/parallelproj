"""get LM data from https://zenodo.org/records/8404015"""

import time
import argparse
import h5py
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import parallelproj.pet_scanners as pps
import parallelproj_core

from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--lm_file", type=str, default="data/LIST0000.BLF")
parser.add_argument("--num_events", type=int, default=10000000)
parser.add_argument("--num_runs", type=int, default=5)
parser.add_argument(
    "--mode", default="GPU", choices=["GPU", "GPU-torch", "CPU", "CPU-torch", "hybrid"]
)
parser.add_argument("--threadsperblock", type=int, default=32)
parser.add_argument("--output_file", type=int, default=None)
parser.add_argument("--output_dir", default="results")
parser.add_argument("--symmetry_axes", default=["0", "1", "2"], nargs="+")
parser.add_argument("--presort", action="store_true")

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


np.random.seed(0)

lm_file = args.lm_file
num_events = args.num_events
num_runs = args.num_runs
threadsperblock = args.threadsperblock

data_str = "nontof_listmode"
if args.presort:
    data_str += "_presorted"

output_dir = args.output_dir
if args.output_file is None:
    output_file = f"{data_str}__mode_{args.mode}__numruns_{num_runs}__tpb_{threadsperblock}__numevents_{num_events}.csv"

# image properties
num_trans = 215
num_ax = 71
voxel_size = xp.asarray([2.78, 2.78, 2.78], dtype=xp.float32, device=dev)

# scanner properties
num_rings = 36
ring_positions = (
    5.31556 * xp.arange(num_rings, device=dev, dtype=xp.float32)
    + (xp.astype(xp.arange(num_rings, device=dev) // 9, xp.float32)) * 2.8
)
ring_positions -= 0.5 * xp.max(ring_positions)


symmetry_axes = [int(x) for x in args.symmetry_axes]

df = pd.DataFrame()

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# --- load listmode data ----------------------------------------------
# ---------------------------------------------------------------------
# ---------------------------------------------------------------------

if Path(lm_file).exists():
    with h5py.File(lm_file, "r") as data:
        events = xp.asarray(data["MiceList/TofCoinc"][:], device=dev, dtype=int)
else:
    raise ValueError(f"file {lm_file} does not exist")

if num_events is None:
    num_events = events.shape[0]

# shuffle events since events come semi sorted
print("shuffling LM data")
ie = np.arange(num_events)
np.random.shuffle(ie)
ie = xp.asarray(ie, device=dev)
events = xp.take(events, ie, axis=0)

# for the DMI the tof bins in the LM files are already meshed (only every 13th is populated)
# so we divide the small tof bin number by 13 to get the bigger tof bins
# the definition of the TOF bin sign is also reversed
events[:, -1] = -(events[:, -1] // 13)

# sort events according to in-ring difference
if args.presort:
    print("pre-sorting events")
    isorted = xp.argsort(events[:, 1] - events[:, 3])
    events = xp.take(events, isorted, axis=0)

y = xp.ones(events.shape[0], dtype=xp.float32, device=dev)

for ia, symmetry_axis in enumerate(symmetry_axes):
    image_shape = 3 * [num_trans]
    image_shape[symmetry_axis] = num_ax
    image_shape = tuple(image_shape)
    image_origin = (
        -(xp.asarray(image_shape, dtype=xp.float32, device=dev) / 2) + 0.5
    ) * voxel_size
    image = xp.ones(image_shape, dtype=xp.float32, device=dev)

    print(
        f"{symmetry_axis, image_shape} {threadsperblock} tpb  {num_events//1000000}e6 events"
    )

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

    xstart = scanner.get_lor_endpoints(events[:, 0], events[:, 1])
    xend = scanner.get_lor_endpoints(events[:, 2], events[:, 3])

    for ir in range(num_runs + 1):
        t0 = time.time()
        img_fwd = xp.zeros(xstart.shape[:-1], dtype=xp.float32, device=dev)
        parallelproj_core.joseph3d_fwd(
            xstart,
            xend,
            image,
            image_origin,
            voxel_size,
            img_fwd,
            threads_per_block=threadsperblock,
        )
        t1 = time.time()

        # peform a back projection
        t2 = time.time()
        back_img = xp.zeros(image.shape, dtype=xp.float32, device=dev)
        parallelproj_core.joseph3d_back(
            xstart,
            xend,
            back_img,
            image_origin,
            voxel_size,
            y,
            threads_per_block=threadsperblock,
        )
        t3 = time.time()
        if ir > 0:
            tmp = pd.DataFrame(
                {
                    "symmetry axis": symmetry_axis,
                    "run": ir,
                    "t forward (s)": t1 - t0,
                    "t back (s)": t3 - t2,
                },
                index=[0],
            )
            df = pd.concat((df, tmp))

# ---------------------------------------------------------------------
# save results
df["data"] = data_str
df["mode"] = args.mode
df["num_events"] = num_events
df["threadsperblock"] = threadsperblock

Path(output_dir).mkdir(exist_ok=True, parents=True)
df.to_csv(os.path.join(output_dir, output_file), index=False)

# ----------------------------------------------------------------------------
# show results

sns.set_context("paper")

df["t forward+back (s)"] = df["t forward (s)"] + df["t back (s)"]

bplot_kwargs = dict(capsize=0.15, err_kws={"linewidth": 1.5}, errorbar="sd")

fig, ax = plt.subplots(1, 3, figsize=(7, 7 / 3), sharex=False, sharey="row")
sns.barplot(data=df, x="symmetry axis", y="t forward (s)", ax=ax[0], **bplot_kwargs)
sns.barplot(data=df, x="symmetry axis", y="t back (s)", ax=ax[1], **bplot_kwargs)
sns.barplot(
    data=df, x="symmetry axis", y="t forward+back (s)", ax=ax[2], **bplot_kwargs
)
for axx in ax:
    axx.grid(ls=":")
fig.show()
