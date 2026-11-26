"""
Projection layers on multiple GPUs with CUDA streams
====================================================

Training a network that contains a forward or back projection is often limited
by the projections.  With several GPUs, the samples of a mini batch can be
projected **on all GPUs at the same time**.  This example shows how to do that
efficiently with parallelproj and PyTorch, e.g. a mini batch of 6 samples on
2 GPUs (3 samples per GPU) or on 3 GPUs (2 samples per GPU).

**You will learn**

- why GPU projections are *asynchronous* (they run on the current CUDA stream),
  and why this lets one Python thread keep several GPUs busy,
- how to keep **one projector per GPU**, so that the scanner geometry is never
  copied during training,
- how to write a projection layer (an autograd function plus a small
  :class:`torch.nn.Module`) that splits a mini batch across GPUs and gathers
  the results, with gradients flowing back to every GPU,
- how to time GPU code correctly.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_06_torch_01_run_projection_layer.py` (a
projection layer on one device).

.. note::

    The speed-up needs at least two CUDA GPUs and a CUDA build of
    parallelproj-core.  With one GPU, or on the CPU (as on Read the Docs), the
    example runs as well, on the devices that are available, but the timings
    show no multi-GPU speed-up.

Background: CUDA streams
------------------------

GPU work is queued on *CUDA streams*.  PyTorch runs every operation on the
*current stream* of the GPU of its tensors and returns immediately, before the
GPU is done.  Operations on the same stream run in order, and everything that
needs a result on the CPU (``.cpu()``, ``.item()``, ``print``) waits for it.

parallelproj's projectors (since 2.1) do the same: they are queued on the
current stream of the GPU of their arrays.  A Python loop that calls a
projection on GPU 0 and then one on GPU 1 therefore only *queues* the work, and
both GPUs compute **at the same time**.  If each call waited for its GPU (as in
parallelproj < 2.1), the GPUs would work one after another.
"""

# %%
# 1. Devices
# ----------
#
# We use all CUDA GPUs if PyTorch sees any and parallelproj-core is a CUDA
# build; otherwise the CPU.  ``torch.cuda.device_count()`` respects
# ``CUDA_VISIBLE_DEVICES``, e.g. ``CUDA_VISIBLE_DEVICES=0,1`` to use 2 GPUs.

from __future__ import annotations

import time

import array_api_compat.torch as xp
import matplotlib.pyplot as plt
import parallelproj_core as ppc
import torch

import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors

if torch.cuda.is_available() and ppc.cuda_enabled == 1:
    devices = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    devices = [torch.device("cpu")]
on_gpu = devices[0].type == "cuda"

print(f"PyTorch {torch.__version__}, parallelproj-core {ppc.__version__}")
for d in devices:
    print(f"  {d}: {torch.cuda.get_device_name(d) if on_gpu else 'CPU'}")

# %%
# 2. One projector per device
# ---------------------------
#
# A projector keeps its LOR endpoints and image geometry on one device.  We
# therefore build **one projector per device** with the same function, once,
# before training.  The scanner is a cylindrical PET scanner; it is smaller on
# the CPU to keep the documentation build short.  The image covers a
# field of view of 400 mm x 400 mm x 60 mm.

if on_gpu:
    num_rings, num_lor_endpoints_per_side, img_shape = 16, 16, (128, 128, 32)
else:
    num_rings, num_lor_endpoints_per_side, img_shape = 4, 6, (64, 64, 8)


def make_projector(dev: torch.device) -> parallelproj.projectors.RegularPolygonPETProjector:
    """non-TOF sinogram projector with all arrays on device ``dev``"""
    scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=300.0,
        num_sides=32,
        num_lor_endpoints_per_side=num_lor_endpoints_per_side,
        lor_spacing=56.0 / num_lor_endpoints_per_side,
        ring_positions=xp.linspace(-30.0, 30.0, num_rings, device=dev),
        symmetry_axis=2,
    )
    lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
        scanner,
        parallelproj.pet_lors.Michelogram(
            num_rings, max_ring_difference=num_rings - 1, span=1
        ),
        radial_trim=num_lor_endpoints_per_side * 9,
    )
    return parallelproj.projectors.RegularPolygonPETProjector(
        lor_desc,
        img_shape=img_shape,
        voxel_size=(400.0 / img_shape[0], 400.0 / img_shape[1], 60.0 / img_shape[2]),
    )


projectors = {d: make_projector(d) for d in devices}
proj0 = projectors[devices[0]]
print(f"image {proj0.in_shape} -> sinogram {proj0.out_shape}")

# %%
# 3. The projection layer
# -----------------------
#
# The layer has two parts:
#
# - :class:`ProjectChunk`, an autograd function that projects a *chunk* of the
#   mini batch that is on **one** device, sample by sample, with the projector
#   of that device.  Its ``backward`` is the adjoint (the back projection for a
#   forward projection layer, and vice versa).  No synchronization is needed:
#   every projection is queued on the current stream of the device.
# - :class:`MultiDeviceProjection`, a :class:`torch.nn.Module` that splits the
#   mini batch into one chunk per device, moves each chunk to its device,
#   applies :class:`ProjectChunk`, and gathers the results on the device of
#   the input.  The device-to-device copies (``.to``) are differentiable, so
#   the gradients of every chunk flow back to the input automatically.
#
# Images have the shape ``(batch, 1, *img_shape)`` (one channel), sinograms
# ``(batch, *sinogram_shape)``.


class ProjectChunk(torch.autograd.Function):
    """forward (``adjoint=False``) or back projection of a chunk on one device"""

    @staticmethod
    def forward(ctx, x: torch.Tensor, proj, adjoint: bool) -> torch.Tensor:
        ctx.proj, ctx.adjoint = proj, adjoint
        if adjoint:  # sinograms (b, *out_shape) -> images (b, 1, *in_shape)
            return torch.stack([proj.adjoint(xi.detach()) for xi in x])[:, None]
        # images (b, 1, *in_shape) -> sinograms (b, *out_shape)
        return torch.stack([proj(xi[0].detach()) for xi in x])

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        proj, adjoint = ctx.proj, ctx.adjoint
        if adjoint:  # gradient w.r.t. the sinograms: forward projection
            g = torch.stack([proj(gi[0].contiguous()) for gi in grad])
        else:  # gradient w.r.t. the images: back projection
            g = torch.stack([proj.adjoint(gi.contiguous()) for gi in grad])[:, None]
        return g, None, None  # no gradients for proj and adjoint


class MultiDeviceProjection(torch.nn.Module):
    """split a mini batch across devices, project, gather on the input device"""

    def __init__(self, projectors: dict, adjoint: bool = False) -> None:
        super().__init__()
        self.projectors = projectors
        self.adjoint = adjoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        devices = list(self.projectors)
        chunks = torch.tensor_split(x, len(devices))  # e.g. 6 samples on 2 GPUs -> 3 + 3
        out = [
            ProjectChunk.apply(c.to(d, non_blocking=True), self.projectors[d], self.adjoint)
            for c, d in zip(chunks, devices)
            if c.shape[0] > 0
        ]
        return torch.cat([o.to(x.device, non_blocking=True) for o in out])


fwd_layer = MultiDeviceProjection(projectors)  # A
back_layer = MultiDeviceProjection(projectors, adjoint=True)  # A^T

# %%
# 4. A mini batch of 6 images
# ---------------------------
#
# Six simple test images (a uniform cylinder with a hot and a cold rod at
# different positions), on the first device.

batch_size = 6  # splits evenly on 1, 2, 3 and 6 devices
n0, n1, n2 = img_shape
r0 = (torch.arange(n0, device=devices[0]) - (n0 - 1) / 2)[:, None]
r1 = (torch.arange(n1, device=devices[0]) - (n1 - 1) / 2)[None, :]


def disk(c0: float, c1: float, r: float) -> torch.Tensor:
    return (((r0 - c0) ** 2 + (r1 - c1) ** 2) <= r**2).float()


images = []
for i in range(batch_size):
    phi = torch.tensor(2 * torch.pi * i / batch_size)
    c0, c1 = 0.25 * n0 * torch.cos(phi), 0.25 * n1 * torch.sin(phi)
    img2d = disk(0, 0, 0.4 * n0) + 2 * disk(c0, c1, 0.08 * n0) - disk(-c0, -c1, 0.08 * n0)
    images.append(img2d[:, :, None].expand(img_shape))
x = torch.stack(images)[:, None].contiguous()  # (4, 1, *img_shape)
print(f"mini batch: {tuple(x.shape)} on {x.device}")

# %%
# Forward projection of the mini batch with all devices, and a check against
# the projector of the first device alone.

y = fwd_layer(x)
y_ref = torch.stack([proj0(xi[0]) for xi in x])
print(f"sinograms: {tuple(y.shape)} on {y.device}")
print(f"max. rel. difference to one device: {float((y - y_ref).abs().max() / y_ref.abs().max()):.1e}")

# %%
# The figure shows the 6 images (central plane) and their sinograms (one
# plane), with the device each sample was projected on.

dev_of_sample = [
    d for c, d in zip(torch.tensor_split(torch.arange(batch_size), len(devices)), devices) for _ in c
]
fig, ax = plt.subplots(2, batch_size, figsize=(1.7 * batch_size, 3.9), layout="constrained")
for i in range(batch_size):
    ax[0, i].imshow(x[i, 0, :, :, n2 // 2].cpu().T, cmap="Greys", origin="lower")
    ax[0, i].set_title(f"sample {i} → {dev_of_sample[i]}", fontsize="small")
    ax[1, i].imshow(y[i, :, :, 0].cpu().T, cmap="magma", aspect="auto", origin="lower")
for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
ax[0, 0].set_ylabel("image")
ax[1, 0].set_ylabel("sinogram (plane 0)")
fig.show()

# %%
# 5. A training step and its gradient
# -----------------------------------
#
# A toy training step: the images ``x_est`` (the "network output") are fitted
# to the sinograms with the least-squares loss :math:`\sum_i \|A x_i - y_i\|^2`.
# A single ``loss.backward()`` back projects the residual of every chunk on its
# own device.  We check the gradient of sample 0 against its analytic value
# :math:`2 A^T (A x_0 - y_0)`.

x_est = torch.ones_like(x, requires_grad=True)


def training_step() -> torch.Tensor:
    x_est.grad = None
    loss = ((fwd_layer(x_est) - y) ** 2).sum()
    loss.backward()
    return loss


loss = training_step()
expected = 2 * proj0.adjoint(proj0(x_est[0, 0].detach()) - y[0])
rel_err = float((x_est.grad[0, 0] - expected).abs().max() / expected.abs().max())
print(f"loss {loss.item():.4e}, rel. error of the gradient of sample 0: {rel_err:.1e}")

# %%
# The back projection layer works the same way, e.g. a back projection of the
# sinograms:

x_back = back_layer(y)
print(f"back projections: {tuple(x_back.shape)} on {x_back.device}")

# %%
# 6. Timing GPU code correctly
# ----------------------------
#
# Because GPU calls return before the GPU is done, a timer around them only
# measures how long it takes to *queue* the work.  To measure the real time,
# we synchronize **all** devices before starting and before stopping the
# clock, do one warm-up call (the first call includes one-time setup), and
# take the best of a few repetitions.  (``torch.utils.benchmark.Timer`` and
# CUDA events, ``torch.cuda.Event(enable_timing=True)``, are alternatives.)


def synchronize() -> None:
    for d in devices:
        if d.type == "cuda":
            torch.cuda.synchronize(d)


def best_time(fn, repeats: int = 5) -> float:
    fn()  # warm-up
    times = []
    for _ in range(repeats):
        synchronize()
        t0 = time.perf_counter()
        fn()
        synchronize()
        times.append(time.perf_counter() - t0)
    return min(times)


# %%
# We time two tasks:
#
# - the **forward projection** of the mini batch (e.g. inference with a
#   network that contains a projection layer), and
# - a **training step** (forward projection, loss, and the back projection of
#   the gradient),
#
# each in three configurations:
#
# - on **one** device,
# - on **all** devices, but waiting for the GPU after every projection, as
#   with the synchronous projections of parallelproj < 2.1 (emulated with a
#   ``torch.cuda.synchronize`` after every call),
# - on **all** devices with asynchronous projections (this example's layer).


class SynchronousProjector:
    """wraps a projector and waits for the GPU after every call (old behaviour)"""

    def __init__(self, proj) -> None:
        self.proj = proj

    def __call__(self, x):
        y = self.proj(x)
        synchronize()
        return y

    def adjoint(self, y):
        x = self.proj.adjoint(y)
        synchronize()
        return x


def forward_with(layer: MultiDeviceProjection):
    def forward():
        with torch.no_grad():
            layer(x)

    return forward


def step_with(layer: MultiDeviceProjection):
    def step():
        x_est.grad = None
        ((layer(x_est) - y) ** 2).sum().backward()

    return step


configs = {"1 device": MultiDeviceProjection({devices[0]: proj0})}
if len(devices) > 1:
    configs[f"{len(devices)} GPUs, synchronous calls"] = MultiDeviceProjection(
        {d: SynchronousProjector(p) for d, p in projectors.items()}
    )
    configs[f"{len(devices)} GPUs, asynchronous calls"] = fwd_layer

tasks = {"forward projection": forward_with, "training step": step_with}
timings = {
    task: {name: best_time(make(layer)) for name, layer in configs.items()}
    for task, make in tasks.items()
}

# the GPU with the most samples limits the speed-up, e.g. 6 samples on 4 GPUs
# would be split 2 + 2 + 1 + 1, so the best possible speed-up is 6 / 2 = 3
max_chunk = -(-batch_size // len(devices))
print(f"best possible speed-up with {len(devices)} device(s): {batch_size / max_chunk:.2f}")
for task, t in timings.items():
    print(f"{task}:")
    for name, ti in t.items():
        print(f"  {name:30s} {1000 * ti:8.1f} ms   speed-up {t['1 device'] / ti:4.2f}")

# %%
# With asynchronous projections, :math:`N` GPUs are up to :math:`N` times
# faster than one GPU, if the samples can be split evenly and the projections
# dominate the run time.  Waiting for the GPU after every call makes the GPUs
# work one after another in the forward pass.  In the training step, the
# difference is smaller: the autograd engine runs the backward pass of every
# GPU in its own thread, so the back projections overlap even with
# synchronous calls.

fig, axs = plt.subplots(
    1, 2, figsize=(10, 0.9 + 0.45 * len(configs)), layout="constrained", sharey=True
)
names = list(configs)[::-1]
for ax, (task, t) in zip(axs, timings.items()):
    bars = ax.barh(names, [1000 * t[n] for n in names], color="#3c75aa")
    ax.bar_label(bars, labels=[f"{t['1 device'] / t[n]:.2f}x" for n in names], padding=4)
    ax.set_xlabel("time [ms]")
    ax.set_title(f"{task} ({batch_size} samples)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, 1200 * max(t.values()))
fig.show()

# %%
# Summary and notes
# -----------------
#
# - Build **one projector per GPU** once, and keep the geometry on its GPU.
# - Split the mini batch with ``torch.tensor_split``, project every chunk on
#   its GPU, and gather the results with ``torch.cat``.  No explicit
#   synchronization is needed: the projections are queued on the current
#   stream of each GPU, like any PyTorch operation, so one Python thread keeps
#   all GPUs busy.  The autograd engine runs the backward pass of each GPU in
#   its own thread.
# - **Time** GPU code only with synchronization and a warm-up.
# - Choose the mini batch size as a **multiple of the number of GPUs**, so that
#   all GPUs get the same number of samples.
# - Large outputs (with the GPU scanner here: a sinogram of 58 MB per sample) are copied
#   between GPUs when they are gathered on one device.  In training, it can
#   pay off to keep the data of every chunk on its GPU, compute the loss per
#   GPU, and only sum the (scalar) losses on one device.
# - **DistributedDataParallel** (one process per GPU, e.g. with ``torchrun``)
#   is the alternative for multi-node training: each process uses the
#   single-device layer of
#   :ref:`sphx_glr_auto_examples_06_torch_01_run_projection_layer.py` with the
#   projector of its own GPU.
# - :class:`torch.nn.DataParallel` does not work out of the box: it replicates
#   the parameters and buffers of a module to all GPUs, but the projector's
#   arrays are neither, so every replica would use the projector of the first
#   GPU.  Use the explicit split shown here instead.
# - If the mini batch comes from the CPU (a data loader), use pinned memory
#   (``DataLoader(..., pin_memory=True)``) so that ``.to(device,
#   non_blocking=True)`` copies asynchronously as well.
# - **Side streams:** inside ``with torch.cuda.stream(s):`` the projections
#   are queued on ``s``, correctly ordered with the other work on ``s``.
