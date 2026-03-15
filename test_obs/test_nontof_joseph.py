from __future__ import annotations

import parallelproj
import array_api_compat.numpy as np

from types import ModuleType

# import the global pytestmark variable containing the xp/dev matrix we
# want to test
from .config import pytestmark


def test_fwd(
    xp: ModuleType,
    dev: str,
    verbose: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    n0, n1, n2 = (2, 3, 4)

    img_dim = (n0, n1, n2)
    voxel_size = xp.asarray([4.0, 3.0, 2.0], dtype=xp.float32, device=dev)
    img_origin = (
        -xp.asarray(img_dim, dtype=xp.float32, device=dev) / 2 + 0.5
    ) * voxel_size
    img = xp.reshape(
        xp.arange(n0 * n1 * n2, dtype=xp.float32, device=dev), (n0, n1, n2)
    )

    # LOR start points in voxel coordinates
    vstart = xp.asarray(
        [
            [0, -1, 0],  #
            [0, -1, 0],  #
            [0, -1, 1],  #
            [0, -1, 0.5],  #
            [0, 0, -1],  #
            [-1, 0, 0],  #
            [n0 - 1, -1, 0],  #
            [n0 - 1, -1, n2 - 1],  #
            [n0 - 1, 0, -1],  #
            [n0 - 1, n1 - 1, -1],
        ],
        device=dev,
    )

    vend = xp.asarray(
        [
            [0, n1, 0],
            [0, n1, 0],
            [0, n1, 1],
            [0, n1, 0.5],
            [0, 0, n2],
            [n0, 0, 0],
            [n0 - 1, n1, 0],
            [n0 - 1, n1, n2 - 1],  #
            [n0 - 1, 0, n2],
            [n0 - 1, n1 - 1, n2],
        ],
        device=dev,
    )

    xstart = vstart * voxel_size + img_origin
    xend = vend * voxel_size + img_origin

    img_fwd = parallelproj.joseph3d_fwd(
        xstart, xend, img, img_origin, voxel_size, num_chunks=3
    )

    # setup the expected values for the projection
    expected_projections = xp.zeros_like(img_fwd, device=dev)
    expected_projections[0] = xp.sum(img[0, :, 0]) * voxel_size[1]
    expected_projections[1] = xp.sum(img[0, :, 0]) * voxel_size[1]
    expected_projections[2] = xp.sum(img[0, :, 1]) * voxel_size[1]
    expected_projections[3] = 0.5 * (expected_projections[0] + expected_projections[2])
    expected_projections[4] = xp.sum(img[0, 0, :]) * voxel_size[2]
    expected_projections[5] = xp.sum(img[:, 0, 0]) * voxel_size[0]
    expected_projections[6] = xp.sum(img[n0 - 1, :, 0]) * voxel_size[1]
    expected_projections[7] = xp.sum(img[n0 - 1, :, n2 - 1]) * voxel_size[1]
    expected_projections[8] = xp.sum(img[n0 - 1, 0, :]) * voxel_size[2]
    expected_projections[9] = xp.sum(img[n0 - 1, n1 - 1, :]) * voxel_size[2]

    if verbose:
        print(
            f"module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}"
        )
        print("calculated projection = ", img_fwd)
        print("expected   projection = ", expected_projections)
        print("abs diff              = ", xp.abs(img_fwd - expected_projections))
        print(
            "rel diff              = ",
            xp.abs(img_fwd - expected_projections) / expected_projections,
        )
        print("")

    isclose = bool(
        xp.all(
            xp.less_equal(
                xp.abs(img_fwd - expected_projections),
                atol + rtol * xp.abs(expected_projections),
            )
        )
    )

    assert isclose


# --------------------------------------------------------------------------


def test_adjointness(
    xp: ModuleType,
    dev: str,
    nLORs: int = 1000000,
    seed: int = 1,
    verbose: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    """test whether backprojection is the adjoint of forward projection
    indirect test whether back projection is correct (assuming fwd projection is correct)
    """

    np.random.seed(seed)
    n0, n1, n2 = (16, 15, 17)

    img_dim = (n0, n1, n2)
    voxel_size = xp.asarray([0.7, 0.8, 0.6], dtype=xp.float32, device=dev)
    img_origin = (
        -xp.asarray(img_dim, dtype=xp.float32, device=dev) / 2 + 0.5
    ) * voxel_size

    img = xp.asarray(np.random.rand(n0, n1, n2), dtype=xp.float32, device=dev)

    # generate random LORs on a sphere around the image volume
    R = 0.8 * xp.max((xp.asarray(img_dim, dtype=xp.float32, device=dev) * voxel_size))

    phis = xp.asarray(np.random.rand(nLORs) * 2 * np.pi, device=dev, dtype=xp.float32)
    costheta = xp.asarray(np.random.rand(nLORs) * 2 - 1, device=dev, dtype=xp.float32)
    sintheta = xp.astype(xp.sqrt(1 - costheta**2), xp.float32)

    xstart = xp.zeros((nLORs, 3), dtype=xp.float32, device=dev)
    xstart[:, 0] = R * sintheta * xp.cos(phis)
    xstart[:, 1] = R * sintheta * xp.sin(phis)
    xstart[:, 2] = R * costheta

    phis = xp.asarray(np.random.rand(nLORs) * 2 * np.pi, device=dev, dtype=xp.float32)
    costheta = xp.asarray(np.random.rand(nLORs) * 2 - 1, device=dev, dtype=xp.float32)
    sintheta = xp.astype(xp.sqrt(1 - costheta**2), xp.float32)

    xend = xp.zeros((nLORs, 3), dtype=xp.float32, device=dev)
    xend[:, 0] = R * sintheta * xp.cos(phis)
    xend[:, 1] = R * sintheta * xp.sin(phis)
    xend[:, 2] = R * costheta

    # forward project
    img_fwd = parallelproj.joseph3d_fwd(
        xstart, xend, img, img_origin, voxel_size, num_chunks=3
    )

    # backward project
    sino = xp.asarray(np.random.rand(*img_fwd.shape), dtype=xp.float32, device=dev)
    back_img = parallelproj.joseph3d_back(
        xstart, xend, img.shape, img_origin, voxel_size, sino, num_chunks=5
    )

    ip_a = float(xp.sum((back_img * img)))
    ip_b = float(xp.sum((img_fwd * sino)))

    if verbose:
        print(
            f"module = {xp.__name__}  -  cuda_enabled {parallelproj.num_visible_cuda_devices > 0}"
        )
        print("ip_a = ", ip_a)
        print("ip_b = ", ip_b)
        print("")

    isclose = abs(ip_a - ip_b) <= atol + rtol * abs(ip_b)

    assert isclose
