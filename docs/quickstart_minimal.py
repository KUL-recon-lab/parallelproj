import array_api_compat.numpy as xp  # the array backend (swap for .torch / .cupy)

import matplotlib.pyplot as plt

from parallelproj.pet_scanners import DemoPETScannerGeometry
from parallelproj.pet_lors import Michelogram, RegularPolygonPETLORDescriptor
from parallelproj.projectors import RegularPolygonPETProjector

# the device used for every array we create
dev = "cpu"  # or "cuda"

# demo cylindrical PET scanner (here trimmed to 4 rings)
# describes world coordinates of LOR endpoints
scanner = DemoPETScannerGeometry(xp, dev, num_rings=4)

# line of response descriptor that describes how order pairs of LOR endpoints
# including options for radial trimming and axial compression
lor_desc = RegularPolygonPETLORDescriptor(
    scanner,
    Michelogram(scanner.num_rings, max_ring_difference=3, span=1),
    radial_trim=50
)

# non-TOF projector
proj = RegularPolygonPETProjector(
    lor_desc, img_shape=(100, 100, 8), voxel_size=(4.0, 4.0, 4.0)
)

# a simple test image: a hot box in the centre
img = xp.zeros(proj.in_shape, dtype=xp.float32, device=dev)
img[50:90, 10:40, :] = 1.0

img_fwd = proj(img)           # forward projection:  image  -> sinogram
back = proj.adjoint(img_fwd)  # back projection (adjoint);  proj.H(sino) works too

print("image shape:    ", img.shape)
print("sinogram shape: ", img_fwd.shape)

fig = plt.figure(figsize=(8, 8), tight_layout=True)
ax = fig.add_subplot(111, projection="3d")
proj.show_geometry(ax)
plt.show()
