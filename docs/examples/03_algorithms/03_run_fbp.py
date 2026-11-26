"""
2D non-TOF filtered back projection of Poisson data
===================================================

:bdg-primary:`reconstruction` :bdg-secondary:`beginner` :bdg-light:`CPU or GPU`

Filtered back projection (FBP) is the classical analytic reconstruction
method: the measured projections are filtered with a ramp filter and back
projected.  It is fast, but it assumes noise-free line integrals, so PET data
must first be corrected for attenuation and contamination.  In this example we
simulate a 2D non-TOF sinogram of a phantom with a known Radon transform,
reconstruct it with FBP and compare the result with an iterative MLEM
reconstruction of the same data.

**You will learn**

- how to simulate a 2D sinogram from an analytic phantom, including
  attenuation, limited resolution, contamination and Poisson noise,
- how to pre-correct the emission sinogram for FBP,
- how to set up and apply a discrete ramp filter,
- how to back project with :class:`.ParallelViewProjector2D` and how to scale
  the result quantitatively,
- how FBP compares with MLEM for noisy data.

**Prerequisites:**
:ref:`sphx_glr_auto_examples_03_algorithms_00_run_mlem_osem_svrg.py`
(the MLEM update).
"""

# %%
# 0. Setup
# --------
#
# Every example selects its array backend (NumPy, CuPy or PyTorch) and device
# (CPU or GPU) with ``suggest_array_backend_and_device``: PyTorch on a CUDA
# GPU if available, otherwise the best available backend on the CPU.  The
# same code runs unchanged on all of them.

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

import parallelproj.projectors
from parallelproj import to_numpy_array
from parallelproj._examples_utils import (
    RadonDisk,
    RadonObjectSequence,
    suggest_array_backend_and_device,
)

# to choose explicitly, e.g. suggest_array_backend_and_device("numpy", "cpu")
xp, dev = suggest_array_backend_and_device()

# %%
# 1. Input parameters
# -------------------
#
# All lengths are in cm.

# add Poisson noise to the sinogram
add_noise = True
# total "prompt" counts of the emission sinogram
total_counts = 1e7
# sigma (in radial bins) of the Gaussian filter that smooths the sinogram in
# the radial direction to simulate limited resolution
sino_res = 1.0
# linear attenuation coefficient of phantom in 1/cm
mu = 0.1
# number of MLEM iterations to run
num_iter = 50
# maximum angle of the sinogram in radians
phi_max = xp.pi
# undersampling factor of the sinogram in the angular direction
phi_undersampling_factor = 1

# %%
# 2. Sinogram and image grids
# ---------------------------
#
# The sinogram has ``num_rad`` radial bins between -30 and 30 cm and enough
# views to sample the angular direction as finely as the radial one.  The
# reconstructed image uses the radial sampling in both directions; the true
# object is evaluated on a finer grid for display.

num_rad = 201
num_phi = int(0.5 * num_rad * xp.pi * (phi_max / xp.pi)) + 1
num_phi = num_phi // phi_undersampling_factor

r = xp.linspace(-30, 30, num_rad, device=dev, dtype=xp.float32)
phi = xp.linspace(0, phi_max, num_phi, endpoint=False, device=dev, dtype=xp.float32)
R, PHI = xp.meshgrid(r, phi, indexing="ij")
x = xp.linspace(float(xp.min(r)), float(xp.max(r)), 1001, device=dev, dtype=xp.float32)
X0hr, X1hr = xp.meshgrid(x, x, indexing="ij")

print(f"num rad:   {num_rad}")
print(f"phi max:   {180*phi_max/xp.pi:.2f} deg")
print(f"delta phi: {180*float(phi[1]-phi[0])/xp.pi:.2f} deg")

# extents of the image and sinogram plots
ext_img = [float(xp.min(r)), float(xp.max(r)), float(xp.min(r)), float(xp.max(r))]
ext_sino = [float(xp.min(r)), float(xp.max(r)), float(xp.min(phi)), float(xp.max(phi))]

# %%
# 3. An object with a known Radon transform
# -----------------------------------------
#
# The phantom is a combination of (scaled) disks whose Radon transform is
# known analytically: an elliptic background (a disk of radius 8 cm stretched
# by a factor 3 along ``x0``), a hot disk, and cold and hot inserts.

disk0 = RadonDisk(xp, dev, 8.0)
disk0.amplitude = 1.0
disk0.s0 = 3.0

disk1 = RadonDisk(xp, dev, 2.0)
disk1.amplitude = 0.5
disk1.x1_offset = 4.67

disk2 = RadonDisk(xp, dev, 1.4)
disk2.amplitude = -0.5
disk2.x0_offset = -10.0

disk3 = RadonDisk(xp, dev, 0.93)
disk3.amplitude = -0.5
disk3.x1_offset = -4.67

disk4 = RadonDisk(xp, dev, 0.67)
disk4.amplitude = 1.0
disk4.x1_offset = -4.67

radon_object = RadonObjectSequence([disk0, disk1, disk2, disk3, disk4])
true_img = radon_object.values(X0hr, X1hr)

fig, ax = plt.subplots(figsize=(5, 4.2), layout="constrained")
im = ax.imshow(
    to_numpy_array(true_img.T), cmap="Greys", extent=ext_img, origin="lower"
)
ax.set_xlabel("x0 [cm]")
ax.set_ylabel("x1 [cm]")
ax.set_title("True object")
fig.colorbar(im, ax=ax, shrink=0.8)
fig.show()

# %%
# The background has the value 1, the hot disk at the top 1.5, the cold disk
# on the left 0.5, and the small insert at the bottom is a hot core (1.5)
# inside a cold ring (0.5).

# %%
# 4. Simulation of attenuation, resolution, contamination and noise
# -----------------------------------------------------------------
#
# The emission sinogram is the Radon transform of the object multiplied by
# the attenuation factors :math:`\exp(-\mu \, l)`, where :math:`l` is the
# length of the LOR inside the elliptic background, smoothed in radial
# direction, plus a constant contamination (10 % of the mean of the
# attenuated sinogram).  It is then scaled to the total number of counts and
# Poisson noise is added.
#
# FBP assumes the line integrals of the object.  We therefore **pre-correct**
# the noisy sinogram: subtract the contamination and divide by the
# attenuation factors.

rt_transform = radon_object.radon_transform(R, PHI)

sens_sino = xp.exp(-mu * disk0.radon_transform(R, PHI))
contam = xp.full(
    rt_transform.shape, 0.1 * xp.mean(sens_sino * rt_transform), device=dev
)

emis_sino = sens_sino * rt_transform

if sino_res > 0:
    for i in range(num_phi):
        emis_sino[:, i] = xp.asarray(
            gaussian_filter(
                to_numpy_array(emis_sino[:, i]),
                sino_res,
            ),
            device=dev,
        )

emis_sino = emis_sino + contam

count_fac = total_counts / float(xp.sum(emis_sino))

emis_sino *= count_fac
contam *= count_fac

if add_noise:
    emis_sino = xp.asarray(
        np.random.poisson(to_numpy_array(emis_sino)).astype(np.float32),
        device=dev,
    )

# pre-correct sinogram - needed to run FBP
pre_corrected_sino = (emis_sino - contam) / sens_sino

sinograms = {
    "Radon transform of object": rt_transform,
    "Emission sinogram": emis_sino,
    "Pre-corrected sinogram": pre_corrected_sino,
}
fig, axes = plt.subplots(
    1, 3, figsize=(10, 3.8), sharex=True, sharey=True, layout="constrained"
)
for ax, (title, sino) in zip(axes, sinograms.items()):
    im = ax.imshow(
        to_numpy_array(sino.T),
        cmap="Greys",
        aspect=20,
        extent=ext_sino,
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("s [cm]")
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9)
axes[0].set_ylabel(r"$\phi$ [rad]")
fig.show()

# %%
# The emission sinogram has the fewest counts (white) around
# :math:`\phi = \pi/2`, where the LORs run along the long axis of the ellipse
# and are attenuated most.  Pre-correction restores the shape of the Radon
# transform, but strongly amplifies the noise in these heavily attenuated
# LORs; outside the object it contains noise around zero.

# %%
# 5. The ramp filter
# ------------------
#
# We use the discrete ramp filter (Ram-Lak filter) in the spatial domain for a
# unit sampling distance: :math:`h(0) = 1/4`, :math:`h(n) = 0` for even
# :math:`n \neq 0` and :math:`h(n) = -1 / (\pi^2 n^2)` for odd :math:`n`.

n_filter = r.shape[0]
r_shift = xp.arange(n_filter, device=dev, dtype=xp.float32) - n_filter // 2
f = xp.zeros(n_filter, device=dev, dtype=xp.float32)
f[r_shift != 0] = -1 / (xp.pi**2 * r_shift[r_shift != 0] ** 2)
f[(r_shift % 2) == 0] = 0
f[r_shift == 0] = 0.25

fig, ax = plt.subplots(figsize=(6, 3.5), layout="constrained")
ax.plot(to_numpy_array(r_shift), to_numpy_array(f), ".-")
ax.set_xlabel("radial offset n [bins]")
ax.set_ylabel("h(n)")
ax.set_xlim(-20, 20)
ax.set_title("Ramp filter (central part)")
fig.show()

# %%
# The filter has a positive peak at the centre and negative side lobes at odd
# offsets that decay as :math:`1/n^2`.  Its values sum to almost zero (zero for
# an infinitely long filter): it removes the mean of each view.

# %%
# 6. Ramp filtering of the pre-corrected sinogram
# -----------------------------------------------
#
# Every view of the pre-corrected sinogram is convolved with the ramp filter
# in radial direction.

filtered_pre_corrected_sino = 1.0 * pre_corrected_sino

for i in range(num_phi):
    filtered_pre_corrected_sino[:, i] = xp.asarray(
        np.convolve(
            to_numpy_array(filtered_pre_corrected_sino[:, i]),
            to_numpy_array(f),
            mode="same",
        ),
        device=dev,
    )

fig, axes = plt.subplots(
    1, 2, figsize=(8, 4.2), sharex=True, sharey=True, layout="constrained"
)
for ax, sino, title in zip(
    axes,
    (pre_corrected_sino, filtered_pre_corrected_sino),
    ("Pre-corrected sinogram", "Filtered pre-corrected sinogram"),
):
    im = ax.imshow(
        to_numpy_array(sino.T),
        cmap="Greys",
        aspect=20,
        extent=ext_sino,
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("s [cm]")
    fig.colorbar(im, ax=ax, orientation="horizontal", shrink=0.9)
axes[0].set_ylabel(r"$\phi$ [rad]")
fig.show()

# %%
# The ramp filter acts as a high-pass filter: it enhances the edges of the
# object in every view (with negative undershoots next to them) and the noise.

# %%
# 7. Filtered back projection
# ---------------------------
#
# :class:`.ParallelViewProjector2D` is a 2D projector for parallel views,
# here with the image grid of section 2.  (The views are passed as ``-phi`` to
# match the angle convention of the analytic Radon transform.)  Its adjoint
# back projects the filtered sinogram.
#
# The back projection approximates the integral over :math:`\phi` by a sum
# over the views, and the adjoint of the projector weights every view by the
# pixel size :math:`\Delta s`.  To obtain the FBP image in the units of the
# object, we multiply by :math:`\Delta \phi / \Delta s^2`: :math:`\Delta \phi`
# for the integral over the views and :math:`1 / \Delta s^2` because the ramp
# filter above is defined for a unit sampling distance.

delta_s = float(r[1] - r[0])
delta_phi = float(phi[1] - phi[0])

proj = parallelproj.projectors.ParallelViewProjector2D(
    (num_rad, num_rad),
    r,
    -phi,
    2 * float(xp.max(r)),
    (float(xp.min(r)), float(xp.min(r))),
    (delta_s, delta_s),
)

filtered_back_proj = (delta_phi / delta_s**2) * proj.adjoint(
    filtered_pre_corrected_sino
)

# %%
# 8. MLEM reconstruction for comparison
# -------------------------------------
#
# MLEM uses the same projector, includes the attenuation factors and the
# contamination in its forward model, and works directly with the noisy
# (not pre-corrected) emission sinogram.

x_mlem = xp.ones((num_rad, num_rad), device=dev, dtype=xp.float32)
sens_img = proj.adjoint(sens_sino)

for i in range(num_iter):
    exp = sens_sino * proj(x_mlem) + contam
    ratio = emis_sino / exp
    ratio_back = proj.adjoint(sens_sino * ratio)

    update_img = ratio_back / sens_img
    x_mlem *= update_img

# %%
# 9. Comparison of FBP and MLEM
# -----------------------------
#
# Both reconstructions are in the units of the scaled object (object values
# times ``count_fac``), so we show them next to the scaled true object with
# the same grey scale.

images = {
    "True object (scaled)": count_fac * true_img,
    "Filtered back projection": filtered_back_proj,
    f"MLEM, {num_iter} iterations": x_mlem,
}
vmax = 1.2 * count_fac * float(xp.max(true_img))

fig, axes = plt.subplots(
    1, 3, figsize=(10, 3.7), sharex=True, sharey=True, layout="constrained"
)
for ax, (title, img) in zip(axes, images.items()):
    im = ax.imshow(
        to_numpy_array(img.T),
        cmap="Greys",
        vmin=0,
        vmax=vmax,
        extent=ext_img,
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("x0 [cm]")
axes[0].set_ylabel("x1 [cm]")
fig.colorbar(im, ax=axes, shrink=0.8)
fig.show()

# %%
# Both methods reconstruct the object with the correct values.  The FBP image
# shows strong high-frequency noise inside the ellipse, with streaks along
# ``x0`` caused by the strongly amplified noise of the heavily attenuated LORs
# along the long axis, and fine streaks outside the object.  MLEM models the Poisson statistics and the attenuation
# directly, which gives a less noisy image with a clean background.

# %%
# Summary
# -------
#
# - FBP needs pre-corrected data: the contamination is subtracted and the
#   result is divided by the attenuation factors, which amplifies the noise.
# - The ramp filter is applied per view in radial direction; the filtered
#   sinogram is back projected with the adjoint of the projector, scaled by
#   :math:`\Delta \phi / \Delta s^2`.
# - For noisy PET data, MLEM, which models the Poisson statistics,
#   attenuation and contamination, gives a less noisy image than FBP.
#
# .. seealso::
#
#    **Next:** :ref:`sphx_glr_auto_examples_03_algorithms_04_run_depierro.py`
#    -- MAP reconstruction with a quadratic prior using DePierro's algorithm.

# sphinx_gallery_thumbnail_number = 5
