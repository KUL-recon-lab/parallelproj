"""
2D non-TOF filtered back projection (FBP) of Poisson data
=========================================================

This example demonstrates the run 2D filtered back projection (FBP)
on pre-corrected Poisson emission data.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

import parallelproj.projectors
from parallelproj import to_numpy_array
from utils import RadonDisk, RadonObjectSequence
from scipy.ndimage import gaussian_filter

# %%
from array_utils import suggest_array_backend_and_device

# To use a specific backend and/or device, replace the None arguments, e.g.:
#   xp, dev = suggest_array_backend_and_device(backend="numpy", dev="cpu") or by setting xp and dev manually
xp, dev = suggest_array_backend_and_device(None, None)


# %%
# Input parameters
# ----------------

# add Poisson noise to the sinogram
add_noise = True
# total "prompt" counts of the emission sinogram
total_counts = 1e7
# sigma of the Gaussian filter to smooth sinogram in radial
# direction to simulate limited resolution
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
# Setup of arrays
# ---------------

num_rad = 201
num_phi = int(0.5 * num_rad * xp.pi * (phi_max / xp.pi)) + 1
num_phi = num_phi // phi_undersampling_factor

r = xp.linspace(-30, 30, num_rad, device=dev, dtype=xp.float32)
phi = xp.linspace(0, phi_max, num_phi, endpoint=False, device=dev, dtype=xp.float32)
R, PHI = xp.meshgrid(r, phi, indexing="ij")
X0, X1 = xp.meshgrid(r, r, indexing="ij")
x = xp.linspace(float(xp.min(r)), float(xp.max(r)), 1001, device=dev, dtype=xp.float32)
X0hr, X1hr = xp.meshgrid(x, x, indexing="ij")

print(f"num rad:   {num_rad}")
print(f"phi max:   {180*phi_max/xp.pi:.2f} deg")
print(f"delta phi: {180*float(phi[1]-phi[0])/xp.pi:.2f} deg")


# %%
# Define an object with known Radon transform
# -------------------------------------------
#
# We setup a combination of (scaled) disks as a simple phantom.

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

fig, ax = plt.subplots(tight_layout=True)
ax.imshow(
    to_numpy_array(radon_object.values(X0hr, X1hr).T),
    cmap="Greys",
    origin="lower",
)
ax.set_xlabel(r"$x_0$")
ax.set_ylabel(r"$x_1$")
ax.set_title("true object", fontsize="medium")
fig.show()

# %%
# Calculate the radon transform of the object
# -------------------------------------------

rt_transform = radon_object.radon_transform(R, PHI)

# %%
# Simulate the effect of attenuation and Poisson noise
# ----------------------------------------------------

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

# %%
ext_sino = [float(xp.min(r)), float(xp.max(r)), float(xp.min(phi)), float(xp.max(phi))]

fig, ax = plt.subplots(1, 3, figsize=(12, 4), tight_layout=True)
ax[0].imshow(
    to_numpy_array(rt_transform.T),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)
ax[1].imshow(
    to_numpy_array(emis_sino.T),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)
ax[2].imshow(
    to_numpy_array(pre_corrected_sino.T),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)

for axx in ax.ravel():
    axx.set_xlabel(r"$s$")
    axx.set_ylabel(r"$\phi$")

ax[0].set_title("radon transform of object", fontsize="medium")
ax[1].set_title("emission sinogram", fontsize="medium")
ax[2].set_title("pre-corrected sinogram", fontsize="medium")
fig.show()


# %%
# Setup of the ramp filter
# ------------------------

n_filter = r.shape[0]
r_shift = xp.arange(n_filter, device=dev, dtype=xp.float32) - n_filter // 2
f = xp.zeros(n_filter, device=dev, dtype=xp.float32)
f[r_shift != 0] = -1 / (xp.pi**2 * r_shift[r_shift != 0] ** 2)
f[(r_shift % 2) == 0] = 0
f[r_shift == 0] = 0.25

fig, ax = plt.subplots(tight_layout=True)
ax.plot(to_numpy_array(r_shift), to_numpy_array(f), ".-")
ax.set_xlabel(r"$s$")
ax.set_ylabel(r"$f$")
ax.set_title("ramp filter", fontsize="medium")
fig.show()

# %%
# Ramp filtering of the pre-corrected emission sinogram
# -----------------------------------------------------

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

fig, ax = plt.subplots(1, 2, figsize=(8, 4), tight_layout=True)
ax[0].imshow(
    to_numpy_array(pre_corrected_sino.T),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)
ax[1].imshow(
    to_numpy_array(filtered_pre_corrected_sino.T),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)
for axx in ax.ravel():
    axx.set_xlabel(r"$s$")
    axx.set_ylabel(r"$\phi$")

ax[0].set_title("pre-corrected sinogram", fontsize="medium")
ax[1].set_title("filtered pre-corr. sinogram", fontsize="medium")
fig.show()


# %%
# Define a projector and run filtered back projection (FBP)
# ---------------------------------------------------------

proj = parallelproj.projectors.ParallelViewProjector2D(
    (num_rad, num_rad),
    r,
    -phi,
    2 * float(xp.max(r)),
    (float(xp.min(r)), float(xp.min(r))),
    (float(r[1] - r[0]), float(r[1] - r[0])),
)

filtered_back_proj = proj.adjoint(filtered_pre_corrected_sino)

fig, ax = plt.subplots(tight_layout=True)
ax.imshow(to_numpy_array(filtered_back_proj).T, cmap="Greys", origin="lower")
ax.set_xlabel(r"$x_0$")
ax.set_ylabel(r"$x_1$")
fig.show()


# %%
# Run iterative MLEM reconstruction for comparison
# ------------------------------------------------

x_mlem = xp.ones((num_rad, num_rad), device=dev, dtype=xp.float32)
sens_img = proj.adjoint(sens_sino)

for i in range(num_iter):
    exp = sens_sino * proj(x_mlem) + contam
    ratio = emis_sino / exp
    ratio_back = proj.adjoint(sens_sino * ratio)

    update_img = ratio_back / sens_img
    x_mlem *= update_img


# %%
# Visualize the results
# ---------------------

ext_img = [float(xp.min(r)), float(xp.max(r)), float(xp.min(r)), float(xp.max(r))]

fig, ax = plt.subplots(1, 4, figsize=(16, 4), tight_layout=True)
ax[0].imshow(
    to_numpy_array(radon_object.values(X0hr, X1hr).T),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[1].imshow(
    to_numpy_array(emis_sino.T),
    cmap="Greys",
    aspect=20,
    extent=ext_sino,
    origin="lower",
)
ax[2].imshow(
    to_numpy_array(filtered_back_proj.T),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[3].imshow(
    to_numpy_array(x_mlem.T),
    cmap="Greys",
    extent=ext_img,
    origin="lower",
)
ax[0].set_xlabel(r"$x_0$")
ax[0].set_ylabel(r"$x_1$")
ax[1].set_xlabel(r"$s$")
ax[1].set_ylabel(r"$\phi$")
ax[2].set_xlabel(r"$x_0$")
ax[2].set_ylabel(r"$x_1$")
ax[3].set_xlabel(r"$x_0$")
ax[3].set_ylabel(r"$x_1$")

ax[0].set_title("true object", fontsize="medium")
ax[1].set_title("emission sino", fontsize="medium")
ax[2].set_title("filtered back projection", fontsize="medium")
ax[3].set_title(f"MLEM {num_iter} it.", fontsize="medium")

fig.show()
