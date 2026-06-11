"""
RAM-efficient OSEM with disk-backed TOF sinograms
=================================================

Demonstrates a memory-efficient OSEM variant in which full TOF sinograms
are **never held in RAM simultaneously**.  Instead they are stored on disk
in a *subset-contiguous* binary layout via :mod:`numpy.memmap`, and only one
subset's data is loaded at a time during each OSEM update.

**Why subset-contiguous layout?**

The natural sinogram axis order (e.g. ``(num_rad, num_views, num_planes,
num_tofbins)`` for RVP) stores views contiguously.  OSEM subsets are
distributed across the view axis with stride ``num_subsets``, so reading
subset *k* from this layout requires many non-sequential disk seeks.

Re-organising the data to shape
``(num_subsets, num_rad, views_per_subset, num_planes, num_tofbins)``
on disk makes each subset a single contiguous block.  Sequential access
lets the OS read-ahead prefetch the next subset from disk while the
projector runs on the current one.

**Memory comparison (this toy scanner)**

.. code-block:: text

    Before disk conversion (full sinograms in RAM):
      y:                ~15 MB
      contamination:    ~15 MB
      total:            ~30 MB

    After disk conversion:
      per subset in RAM during update:
        y_k:    ~0.65 MB
        s_k:    ~0.65 MB
        total:  ~1.30 MB  (vs ~30 MB -- a 23x reduction)

On a clinical scanner (400 rad x 400 views x 837 planes x 27 TOF bins)
the full sinogram is ~3.5 GB each, and the saving scales accordingly.

**Helper code** (in :mod:`parallelproj.data`):

* :class:`parallelproj.data.SubsetArrayMmap` -- read-only wrapper
  around a ``numpy.memmap`` file.  ``mmap[k]`` returns an *owned* copy of
  subset *k* that Python frees as soon as the caller deletes the reference.
* :func:`parallelproj.data.to_subset_mmap` -- one-time conversion
  from a full in-memory sinogram to the on-disk subset-contiguous format.
"""

# %%
from __future__ import annotations

import shutil
import tempfile
from copy import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import parallelproj.operators
import parallelproj.pet_lors
import parallelproj.pet_scanners
import parallelproj.projectors
import parallelproj.tof
from parallelproj import Array, to_numpy_array
from parallelproj.functions import C1Function, C2AffineObjective, NegPoissonLogL
from parallelproj.data import to_subset_mmap

from example_utils import (
    elliptic_cylinder_phantom,
    show_vol_cuts,
)

# %%
# Backend and device
# ------------------
#
# In principle we can run on any backend or device
# However, since we will use (numpy-based) memmaps, it is best to use
# the CPU as device.

import array_api_compat.numpy as xp

dev = "cpu"

# %%
# Scanner and projector setup
# ---------------------------

num_subsets = 24
num_epochs = 5

num_rings = 5
scanner = parallelproj.pet_scanners.RegularPolygonPETScannerGeometry(
    xp,
    dev,
    radius=65.0,
    num_sides=16,
    num_lor_endpoints_per_side=12,
    lor_spacing=2.3,
    ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
    symmetry_axis=2,
)

img_shape = (55, 55, 8)
voxel_size = (2.0, 2.0, 2.0)

lor_desc = parallelproj.pet_lors.RegularPolygonPETLORDescriptor(
    scanner,
    parallelproj.pet_lors.Michelogram(scanner.num_rings, max_ring_difference=2, span=1),
    radial_trim=10,
    sinogram_order=parallelproj.pet_lors.SinogramSpatialAxisOrder.RVP,
)

proj = parallelproj.projectors.RegularPolygonPETProjector(
    lor_desc, img_shape=img_shape, voxel_size=voxel_size
)

x_true = elliptic_cylinder_phantom(
    xp, dev, image_shape=img_shape, voxel_size=voxel_size
)

# %%
# Attenuation and full forward model
# -----------------------------------

x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
att_sino = xp.exp(-proj(x_att))

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

att_values = (
    xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
    if proj.tof
    else att_sino
)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape,
    sigma=[2.0 / (2.35 * float(vs)) for vs in proj.voxel_size],
)

pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# Simulate TOF PET data
# ----------------------

noise_free_data = pet_lin_op(x_true)

contamination = xp.full(
    noise_free_data.shape,
    0.5 * float(xp.mean(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

noise_free_data += contamination

np.random.seed(1)
y = xp.asarray(
    np.random.poisson(to_numpy_array(noise_free_data)),
    device=dev,
    dtype=xp.float32,
)

del noise_free_data

# %%
# Subset view / slice definitions
# --------------------------------

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# %%
# Convert full sinograms to disk-backed subset files and free RAM
# ---------------------------------------------------------------
#
# ``to_subset_mmap`` gathers the non-contiguous subset slices from
# the full array and writes them as contiguous blocks on disk.  The
# returned :class:`parallelproj.data.SubsetArrayMmap` is a thin read-only
# wrapper; no sinogram data lives in RAM after the ``del`` calls.

tmpdir = Path(tempfile.mkdtemp())

y_np = to_numpy_array(y)
s_np = to_numpy_array(contamination)

print("Full sinograms in RAM before conversion:")
print(f"  y:             {y_np.nbytes / 1024**2:.1f} MB")
print(f"  contamination: {s_np.nbytes / 1024**2:.1f} MB")
print(f"  total:         {(y_np.nbytes + s_np.nbytes) / 1024**2:.1f} MB")

y_mmap = to_subset_mmap(y_np, subset_slices, tmpdir / "y.bin")
s_mmap = to_subset_mmap(s_np, subset_slices, tmpdir / "s.bin")

# the strictly positive contamination guarantees positive expected data in
# every bin, so the exact (unmodified) log-likelihood of NegPoissonLogL can
# be used -- evaluate before the full contamination array is freed
exact_mode = bool(xp.min(contamination) > 0)

del y_np, s_np, y, contamination  # full arrays no longer in RAM

print("\nOn disk, per-subset in RAM on demand (OS-managed):")
print(f"  y_k per subset:   {y_mmap.nbytes_per_subset() / 1024**2:.2f} MB")
print(f"  s_k per subset:   {s_mmap.nbytes_per_subset() / 1024**2:.2f} MB")
peak_mb = (y_mmap.nbytes_per_subset() + s_mmap.nbytes_per_subset()) / 1024**2
print(f"  peak per update:  {peak_mb:.2f} MB")

# %%
# Subset forward operators and sensitivity images
# ------------------------------------------------

proj.clear_cached_lor_endpoints()

pet_subset_linop_seq = []

for i in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    att_values_k = (
        xp.broadcast_to(
            xp.expand_dims(att_sino[subset_slices_non_tof[i]], axis=-1),
            subset_proj.out_shape,
        )
        if subset_proj.tof
        else att_sino[subset_slices_non_tof[i]]
    )
    subset_att_op = parallelproj.operators.ElementwiseMultiplicationOperator(
        att_values_k
    )

    pet_subset_linop_seq.append(
        parallelproj.operators.CompositeLinearOperator(
            [subset_att_op, subset_proj, res_model]
        )
    )

pet_subset_linop_seq = parallelproj.operators.LinearOperatorSequence(
    pet_subset_linop_seq
)

subset_adjoint_ones = xp.zeros(
    (num_subsets,) + pet_lin_op.in_shape, dtype=xp.float32, device=dev
)
for k, op in enumerate(pet_subset_linop_seq):
    subset_adjoint_ones[k] = op.adjoint(
        xp.ones(op.out_shape, dtype=xp.float32, device=dev)
    )

# %%
# FOV mask
# --------
#
# The scanner's cylindrical field of view does not cover every voxel of the
# image grid.  Voxels outside the FOV are never intersected by any LOR, so
# their sensitivity :math:`(A^H 1)_i = 0`.  Dividing by zero in the EM
# preconditioner would produce NaN / Inf values that corrupt the
# reconstruction.  :meth:`.RegularPolygonPETProjector.fov_mask` returns a
# boolean array that is ``True`` inside the FOV.  ``fov_mask`` is set to
# ``None`` when every image voxel is inside the FOV (no masking needed).

cyl_mask = proj.fov_mask()
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask
del cyl_mask

# %%
# EM update
# ----------


def em_update(
    x_cur: Array,
    data_fidelity: C1Function,
    adj_ones: Array,
    img_mask: Array | None = None,
) -> Array:
    """One EM update rewritten as a preconditioned gradient descent step.

    Computes :math:`x^+ = x - D \\nabla f(x)` where the diagonal
    preconditioner is :math:`D = \\operatorname{diag}(x / (A^H 1))`.
    Voxels outside the FOV are excluded via ``img_mask`` to avoid
    division by the zero sensitivity values in ``adj_ones``.

    Parameters
    ----------
    x_cur : Array
        Current image estimate.
    data_fidelity : C1Function
        Differentiable data-fidelity term whose gradient is evaluated at
        ``x_cur``.
    adj_ones : Array
        Sensitivity image :math:`A^H 1` (or subset variant
        :math:`(A^k)^H 1`).
    img_mask : Array or None, optional
        Boolean FOV mask (``True`` inside the FOV).  Preconditioner is
        zeroed outside the FOV so that zero-sensitivity voxels do not
        produce NaN / Inf.  Pass ``None`` when every voxel is in the FOV.

    Returns
    -------
    Array
        Updated image :math:`x^+`, same shape as ``x_cur``.
    """
    if img_mask is None:
        d = x_cur / adj_ones
    else:
        d = xp.where(img_mask, x_cur / adj_ones, xp.zeros_like(x_cur))
    return x_cur - d * data_fidelity.gradient(x_cur)


# %%
# Full objective evaluated subset-by-subset
# ------------------------------------------
#
# The negative Poisson log-likelihood is separable:
# :math:`f(x) = \sum_k f_k(x)`.  We accumulate the value over subsets,
# loading only one subset's data at a time.
#
# .. note::
#     By default :class:`.NegPoissonLogL` evaluates a "safe epsilon"
#     (shifted Poisson) surrogate: a tiny ``eps = rel_eps * mean(y)`` is
#     added to the measured and the expected data.  This is finite for any
#     non-negative expectation (never ``nan`` / ``inf``), at the price of a
#     tiny (~``rel_eps``) bias that vanishes at the fit.  Since our
#     contamination is strictly positive, the expected data ``A x + s`` are
#     positive in every bin and we can use ``exact=True`` (``exact_mode``
#     was derived from the contamination above).  Keep the default whenever
#     the expected data can reach zero in bins with counts.  Note that each
#     per-subset instance would derive its own ``eps`` from its subset mean
#     -- pass one global ``eps`` explicitly if exact separability of the
#     subset objectives matters.


def full_objective_from_subsets(x: Array) -> float:
    """Compute f(x) by accumulating over subsets from disk."""
    total = 0.0
    for subset_idx in range(num_subsets):
        y_sub = xp.asarray(y_mmap[subset_idx], device=dev, dtype=xp.float32)
        s_sub = xp.asarray(s_mmap[subset_idx], device=dev, dtype=xp.float32)
        total += C2AffineObjective(
            NegPoissonLogL(y_sub, exact=exact_mode),
            pet_subset_linop_seq[subset_idx],
            s_sub,
        )(x)
        del y_sub, s_sub
    return total


# %%
# Warm-start: one OSEM epoch before tracking convergence
# -------------------------------------------------------

x_osem = xp.ones(pet_lin_op.in_shape, dtype=xp.float32, device=dev)
if fov_mask is not None:
    x_osem = xp.where(fov_mask, x_osem, xp.zeros_like(x_osem))

for k in range(num_subsets):
    y_k = xp.asarray(y_mmap[k], device=dev, dtype=xp.float32)
    s_k = xp.asarray(s_mmap[k], device=dev, dtype=xp.float32)
    df_k = C2AffineObjective(
        NegPoissonLogL(y_k, exact=exact_mode), pet_subset_linop_seq[k], s_k
    )
    x_osem = em_update(x_osem, df_k, subset_adjoint_ones[k], fov_mask)
    del df_k, y_k, s_k

# %%
# OSEM reconstruction
# --------------------
#
# Only ``y_k`` and ``s_k`` (one subset each) reside in RAM during a subset
# update.  They are loaded from the memory-mapped files and freed by
# ``del`` after each update, keeping the sinogram footprint minimal.

df_osem = xp.zeros(num_epochs, dtype=xp.float32, device=dev)

for i in range(num_epochs):
    for k in range(num_subsets):
        print(
            f"OSEM epoch {i + 1:03}/{num_epochs:03},"
            f" subset {k + 1:03}/{num_subsets:03}",
            end="\r",
        )
        # --- load subset k from disk (one sequential read) ---
        y_k = xp.asarray(y_mmap[k], device=dev, dtype=xp.float32)
        s_k = xp.asarray(s_mmap[k], device=dev, dtype=xp.float32)

        df_k = C2AffineObjective(
            NegPoissonLogL(y_k, exact=exact_mode), pet_subset_linop_seq[k], s_k
        )
        x_osem = em_update(x_osem, df_k, subset_adjoint_ones[k], fov_mask)

        # --- release subset data from RAM / GPU VRAM ---
        del df_k, y_k, s_k

    df_osem[i] = full_objective_from_subsets(x_osem)

print()

# %%
# Remove temporary files
# ----------------------

shutil.rmtree(tmpdir)

# %%
# Convergence
# -----------

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
ax.plot(np.arange(1, num_epochs + 1), to_numpy_array(df_osem), marker="o")
ax.set_xlabel("Epoch")
ax.set_ylabel("Negative Poisson log-likelihood")
ax.set_title(f"OSEM ({num_subsets} subsets, disk-backed sinograms)")
ax.grid(ls=":")
fig.show()

# %%
# Reconstructed image
# --------------------

fig, axs, widgets = show_vol_cuts(
    to_numpy_array(x_osem),
    voxel_size=voxel_size,
    fig_title=f"OSEM {num_epochs} epochs (disk-backed)",
    vmin=0,
    vmax=float(xp.max(x_osem)),
)
fig.show()
