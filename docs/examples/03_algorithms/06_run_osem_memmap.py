"""
RAM-efficient OSEM with disk-backed TOF sinograms
=================================================

:bdg-primary:`reconstruction` :bdg-secondary:`intermediate` :bdg-light:`CPU or GPU`

TOF sinograms of clinical scanners need several GB of memory each, and OSEM
needs at least two of them (the data and the contamination).  OSEM only
works on one subset at a time, so the full sinograms never have to be in RAM
together.  In this example we store them on disk in a *subset-contiguous*
layout via :class:`numpy.memmap` and load only one subset at a time during
the OSEM updates.

**You will learn**

- why a subset-contiguous layout on disk makes reading a subset fast,
- how to convert full sinograms to disk-backed subset files with
  :func:`parallelproj.data.to_subset_mmap`,
- how to run OSEM (and evaluate the full objective function) with only one
  subset of the data in RAM, using :class:`parallelproj.data.SubsetArrayMmap`,
- how much memory this saves.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(OSEM with subsets).
"""

# %%
# 0. Setup
# --------
#
# The example could run on any backend and device.  Since the memory-mapped
# files are NumPy arrays (:class:`numpy.memmap`), we use NumPy on the CPU:
# then the subset data are read directly from the files, without an
# additional copy to a GPU.

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
from parallelproj._examples_utils import (
    elliptic_cylinder_phantom,
    show_vol_cuts,
    suggest_array_backend_and_device,
)
from parallelproj.data import to_subset_mmap
from parallelproj.functions import C1Function, C2AffineObjective, NegPoissonLogL

# NumPy on the CPU on purpose: the memmaps are NumPy arrays
xp, dev = suggest_array_backend_and_device("numpy", "cpu")

# %%
# 1. Scanner, projector and phantom
# ---------------------------------
#
# A small scanner with 16 sides and 5 rings, a sinogram with a maximum ring
# difference of 2 in radial-view-plane (RVP) order, an image of 55 x 55 x 8
# voxels of 2 mm and an elliptic cylinder phantom with hot and cold inserts.
# OSEM uses 24 subsets and runs for ``num_epochs`` epochs after a one-epoch
# warm start.

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
# 2. TOF forward model with attenuation
# -------------------------------------
#
# The attenuation sinogram is computed with the non-TOF projector (water,
# :math:`\mu = 0.01` / mm, inside the phantom).  Then TOF is switched on (13
# TOF bins of 12 mm, :math:`\sigma_\text{TOF}` = 12 mm), the attenuation
# factors are broadcast over the TOF bins, and a Gaussian resolution model
# (2 mm FWHM) is added: :math:`A = \text{diag}(a) P G`.

x_att = 0.01 * xp.astype(x_true > 0, xp.float32)
att_sino = xp.exp(-proj(x_att))

proj.tof_parameters = parallelproj.tof.TOFParameters(
    num_tofbins=13, tofbin_width=12.0, sigma_tof=12.0
)

att_values = xp.broadcast_to(xp.expand_dims(att_sino, axis=-1), proj.out_shape)
att_op = parallelproj.operators.ElementwiseMultiplicationOperator(att_values)

res_model = parallelproj.operators.GaussianFilterOperator(
    proj.in_shape,
    sigma=[2.0 / (2.3548 * float(vs)) for vs in proj.voxel_size],
)

pet_lin_op = parallelproj.operators.CompositeLinearOperator((att_op, proj, res_model))

# %%
# 3. Simulation of TOF data
# -------------------------
#
# Poisson data of the noise-free expectation :math:`A x + s`, with a constant
# contamination :math:`s` (50 % of the mean of :math:`A x`).

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
# 4. Subset definitions
# ---------------------
#
# The subsets are groups of views.  We need the slices of each subset for the
# 4D TOF sinograms (data and contamination) and for the 3D non-TOF
# attenuation sinogram.

subset_views, subset_slices = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, len(proj.out_shape)
)

_, subset_slices_non_tof = proj.lor_descriptor.get_distributed_views_and_slices(
    num_subsets, 3
)

# %%
# 5. Disk-backed subset files
# ---------------------------
#
# **Why a subset-contiguous layout?**  The natural sinogram axis order (e.g.
# ``(num_rad, num_views, num_planes, num_tofbins)`` for RVP) stores views
# contiguously.  OSEM subsets take every ``num_subsets``-th view, so reading
# subset *k* from this layout requires many non-sequential disk reads.
# Re-organising the data to the shape
# ``(num_subsets, num_rad, views_per_subset, num_planes, num_tofbins)``
# on disk makes each subset a single contiguous block that can be read
# sequentially (and prefetched by the operating system).
#
# :func:`parallelproj.data.to_subset_mmap` gathers the subset slices of a
# full array once and writes them as contiguous blocks to a file.  It returns
# a :class:`parallelproj.data.SubsetArrayMmap`, a thin read-only wrapper
# around a :class:`numpy.memmap`: ``mmap[k]`` returns an *owned* copy of
# subset *k* that is freed as soon as it is no longer referenced.  After the
# ``del`` below, the full TOF sinograms are no longer in RAM.

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
# During an OSEM update, only 1.29 MB of sinogram data are in RAM instead of
# 30.9 MB, a reduction by the number of subsets (24).  On a clinical scanner
# (400 radial bins x 400 views x 837 planes x 27 TOF bins, i.e.
# :math:`3.6 \cdot 10^9` bins) each full sinogram needs about 14.5 GB in
# float32, and the saving scales accordingly.

# %%
# 6. Subset forward operators and sensitivity images
# --------------------------------------------------
#
# As in the in-memory OSEM, every subset gets its own forward operator (with
# the attenuation factors of its views) and its sensitivity image
# :math:`(A^k)^T 1`.  These operators and images are small compared with the
# TOF sinograms.

proj.clear_cached_lor_endpoints()

pet_subset_linop_seq = []

for i in range(num_subsets):
    subset_proj = copy(proj)
    subset_proj.views = subset_views[i]

    att_values_k = xp.broadcast_to(
        xp.expand_dims(att_sino[subset_slices_non_tof[i]], axis=-1),
        subset_proj.out_shape,
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
# 7. FOV mask and EM update
# -------------------------
#
# The cylindrical field of view (FOV) of the scanner does not cover every
# voxel of the image grid.  Voxels outside the FOV are not intersected by any
# LOR, so their sensitivity :math:`(A^T 1)_j` is 0, and dividing by it in the
# EM preconditioner would produce NaN / Inf values.
# :meth:`.RegularPolygonPETProjector.fov_mask` returns a boolean array that
# is ``True`` inside the FOV; ``fov_mask`` is ``None`` if every voxel is
# inside the FOV (no masking needed).

cyl_mask = proj.fov_mask()
fov_mask = None if bool(xp.all(cyl_mask)) else cyl_mask
del cyl_mask


def em_update(
    x_cur: Array,
    data_fidelity: C1Function,
    adj_ones: Array,
    img_mask: Array | None = None,
) -> Array:
    """One EM update rewritten as a preconditioned gradient descent step.

    Computes :math:`x^+ = x - D \\nabla f(x)` where the diagonal
    preconditioner is :math:`D = \\operatorname{diag}(x / (A^T 1))`.
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
        Sensitivity image :math:`A^T 1` (or subset variant
        :math:`(A^k)^T 1`).
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
        # divide by 1 outside the FOV to avoid 0 / 0 (those voxels are zeroed)
        safe_adj_ones = xp.where(img_mask, adj_ones, xp.ones_like(adj_ones))
        d = xp.where(img_mask, x_cur / safe_adj_ones, xp.zeros_like(x_cur))
    return x_cur - d * data_fidelity.gradient(x_cur)


# %%
# 8. Full objective evaluated subset by subset
# --------------------------------------------
#
# The negative Poisson log-likelihood is separable,
# :math:`f(x) = \sum_k f_k(x)`.  To monitor the convergence, we accumulate its
# value over the subsets, loading only one subset of the data at a time.
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
# 9. OSEM reconstruction
# ----------------------
#
# A first OSEM epoch from a uniform image serves as a warm start; then we run
# ``num_epochs`` further epochs and evaluate the full objective after each of
# them.  Only ``y_k`` and ``s_k`` (one subset each) are in RAM during a subset
# update: they are loaded from the memory-mapped files (one sequential read
# each) and released with ``del`` after the update.

# --- warm start: 1 OSEM epoch ---
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

# --- OSEM epochs with objective tracking ---
df_osem = xp.zeros(num_epochs, dtype=xp.float32, device=dev)

for i in range(num_epochs):
    for k in range(num_subsets):
        print(
            f"OSEM epoch {i + 1:03}/{num_epochs:03},"
            f" subset {k + 1:03}/{num_subsets:03}",
            end="\r",
        )
        # load subset k from disk (one sequential read)
        y_k = xp.asarray(y_mmap[k], device=dev, dtype=xp.float32)
        s_k = xp.asarray(s_mmap[k], device=dev, dtype=xp.float32)

        df_k = C2AffineObjective(
            NegPoissonLogL(y_k, exact=exact_mode), pet_subset_linop_seq[k], s_k
        )
        x_osem = em_update(x_osem, df_k, subset_adjoint_ones[k], fov_mask)

        # release the subset data
        del df_k, y_k, s_k

    df_osem[i] = full_objective_from_subsets(x_osem)

print()

# %%
# 10. Removing the temporary files
# --------------------------------

shutil.rmtree(tmpdir)

# %%
# 11. Convergence
# ---------------
#
# The negative Poisson log-likelihood after each OSEM epoch (epoch 1, the
# warm start, is not tracked).

fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
epochs = np.arange(2, num_epochs + 2)
ax.plot(epochs, to_numpy_array(df_osem), marker="o")
ax.set_xticks(epochs)
ax.set_xlabel("OSEM epoch")
ax.set_ylabel("negative Poisson log-likelihood")
ax.set_title(f"OSEM ({num_subsets} subsets, disk-backed sinograms)")
ax.grid(ls=":")
fig.show()

# %%
# The objective decreases monotonically, with decreasing steps.  Loading the
# data subset by subset from disk does not change the algorithm: the iterates
# are those of standard OSEM.

# %%
# 12. Reconstructed image
# -----------------------

fig, _, _ = show_vol_cuts(
    x_osem,
    voxel_size=voxel_size,
    fig_title=f"OSEM, {num_epochs + 1} epochs with {num_subsets} subsets (disk-backed)",
    vmin=0,
    vmax=float(xp.max(x_osem)),
)
fig.show()

# %%
# The hot and cold inserts of the phantom are visible; after 6 epochs with 24
# subsets the image is already noisy, as is typical for unregularised OSEM.

# %%
# Summary
# -------
#
# - OSEM needs only one subset of the data and the contamination at a time,
#   so the full TOF sinograms can stay on disk.
# - :func:`parallelproj.data.to_subset_mmap` writes a sinogram in a
#   subset-contiguous layout, so that each subset is read in one sequential
#   access; :class:`parallelproj.data.SubsetArrayMmap` returns one subset per
#   index.
# - The sinogram memory during the reconstruction drops by the number of
#   subsets (here from 30.9 MB to 1.29 MB), and the full objective can still
#   be evaluated subset by subset.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_07_run_mlem_negpoissonlogl_modes.py`
#    -- the exact and the "safe epsilon" mode of the negative Poisson
#    log-likelihood.

# sphinx_gallery_thumbnail_number = 2
