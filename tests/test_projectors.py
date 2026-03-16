from __future__ import annotations

import pytest
import parallelproj
import array_api_compat
import matplotlib.pyplot as plt

from .config import pytestmark


def allclose(x, y, atol: float = 1e-8, rtol: float = 1e-5) -> bool:
    """check if two arrays are close to each other, given absolute and relative error
    inspired by numpy.allclose
    """
    xp = array_api_compat.array_namespace(x)
    return bool(xp.all(xp.less_equal(xp.abs(x - y), atol + rtol * xp.abs(y))))


def test_parallelviewprojector(xp, dev, verbose=True):
    image_shape = (2, 2)
    voxel_size = (2.0, 2.0)
    image_origin = (-1.0, -1.0)

    radial_positions = xp.asarray([-1, 0, 1], dtype=xp.float32, device=dev)
    view_angles = xp.asarray([0, xp.pi / 2], dtype=xp.float32, device=dev)
    radius = 3.0

    proj2d = parallelproj.ParallelViewProjector2D(
        image_shape, radial_positions, view_angles, radius, image_origin, voxel_size
    )

    assert proj2d.num_views == array_api_compat.size(view_angles)
    assert proj2d.num_rad == array_api_compat.size(radial_positions)

    xstart = proj2d.xstart
    xend = proj2d.xend

    assert allclose(proj2d.image_origin[1:], xp.asarray(image_origin, device=dev))
    assert proj2d.image_shape == image_shape
    assert allclose(proj2d.voxel_size[1:], xp.asarray(voxel_size, device=dev))
    assert proj2d.dev == array_api_compat.device(xstart)

    assert proj2d.adjointness_test(xp, dev, verbose=verbose, dtype=xp.float32)

    # test a simple 2D projection
    x2d = xp.reshape(xp.arange(4, dtype=xp.float32, device=dev), (2, 2))
    x_fwd = proj2d(x2d)

    exp_result = xp.asarray(
        [[2.0, 8.0], [6.0, 6.0], [10.0, 4.0]], dtype=xp.float32, device=dev
    )

    if verbose:
        print("calculated 2d projection = ", x_fwd)
        print("expected   2d projection = ", exp_result)
        print("abs diff                 = ", xp.abs(x_fwd - exp_result))
        print("rel diff                 = ", xp.abs(x_fwd - exp_result) / exp_result)
        print("")

    assert allclose(x_fwd, exp_result)

    import numpy as np

    fig = proj2d.show_views(image=np.ones(image_shape))

    # setup a simple 3D projector with 2 rings

    image_shape3d = (2, 2, 2)
    image_origin3d = (-1, -1.0, -1.0)
    voxel_size3d = (2.0, 2.0, 2.0)
    ring_positions = xp.asarray([-1, 0, 1.0], dtype=xp.float32, device=dev)

    proj3d = parallelproj.ParallelViewProjector3D(
        image_shape3d,
        radial_positions,
        view_angles,
        radius,
        image_origin3d,
        voxel_size3d,
        ring_positions,
        max_ring_diff=1,
    )

    xstart = proj3d.xstart
    xend = proj3d.xend

    assert proj3d.adjointness_test(xp, dev, verbose=verbose, dtype=xp.float32)

    # test a simple 3D projection
    x3d = xp.reshape(xp.arange(8, dtype=xp.float32, device=dev), (2, 2, 2))
    x3d_fwd = proj3d(x3d)

    # check if we get the expected results for the 3 direct planes
    exp_result_dp0 = xp.asarray(
        [[4.0, 16.0], [12.0, 12.0], [20.0, 8.0]], dtype=xp.float32, device=dev
    )
    exp_result_dp1 = xp.asarray(
        [[6.0, 18.0], [14.0, 14.0], [22.0, 10.0]], dtype=xp.float32, device=dev
    )
    exp_result_dp2 = xp.asarray(
        [[8.0, 20.0], [16.0, 16.0], [24.0, 12.0]], dtype=xp.float32, device=dev
    )

    assert allclose(x3d_fwd[..., 0], exp_result_dp0)
    assert allclose(x3d_fwd[..., 1], exp_result_dp1)
    assert allclose(x3d_fwd[..., 2], exp_result_dp2)

    # test is max_ring_diff = None works
    proj3d_2 = parallelproj.ParallelViewProjector3D(
        image_shape3d,
        radial_positions,
        view_angles,
        radius,
        image_origin3d,
        voxel_size3d,
        ring_positions,
        max_ring_diff=None,
    )

    assert proj3d_2.max_ring_diff == array_api_compat.size(ring_positions) - 1

    # test whether span > 1 raises execption
    with pytest.raises(Exception) as e_info:
        proj3d_2 = parallelproj.ParallelViewProjector3D(
            image_shape3d,
            radial_positions,
            view_angles,
            radius,
            image_origin3d,
            voxel_size3d,
            ring_positions,
            max_ring_diff=None,
            span=3,
        )


def test_lmprojector(
    xp, dev, verbose=True, rtol: float = 1e-5, atol: float = 1e-8
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

    lm_proj = parallelproj.ListmodePETProjector(
        xstart, xend, img_dim, voxel_size, img_origin
    )

    assert xp.all(lm_proj.event_start_coordinates == xstart)
    assert xp.all(lm_proj.event_end_coordinates == xend)

    assert lm_proj.num_events == xstart.shape[0]

    assert lm_proj.adjointness_test(xp, dev, dtype=xp.float32)

    assert xp.all(lm_proj.voxel_size == xp.asarray(voxel_size, device=dev))

    img_fwd = lm_proj(img)

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

    isclose = bool(
        xp.all(
            xp.less_equal(
                xp.abs(img_fwd - expected_projections),
                atol + rtol * xp.abs(expected_projections),
            )
        )
    )

    assert isclose

    # TOF LM tests
    tof_params = parallelproj.TOFParameters()
    with pytest.raises(Exception) as e_info:
        # raises an exception because TOFParameters and event_tofbins are not
        # set
        lm_proj.tof = True

    with pytest.raises(Exception) as e_info:
        # tof_parameters must be of type TOFParameters or None
        lm_proj.tof_parameters = "nonsense"

    # set TOF parameters
    lm_proj.tof_parameters = tof_params

    with pytest.raises(Exception) as e_info:
        # raises an exception because event_tofbins are not set
        lm_proj.tof = True

    # set event TOF bins
    with pytest.raises(Exception) as e_info:
        # raises an exception because event_tofbins have not the
        # same length as num_events
        lm_proj.event_tofbins = xp.asarray(
            [0, 1, -1, 0, 2, -2, 0, 1, -1], dtype=xp.int16, device=dev
        )

    lm_proj.event_tofbins = xp.asarray(
        [0, 1, -1, 0, 2, -2, 0, 1, -1, 0], dtype=xp.int16, device=dev
    )
    # now we can set the tof property to True
    lm_proj.tof = True

    assert lm_proj.tof

    assert lm_proj.adjointness_test(xp, dev)

    # unset the tof parameters and check if tof gets set to False
    lm_proj.tof_parameters = None
    assert lm_proj.tof == False

    lm_proj.tof_parameters = tof_params
    lm_proj.tof = True

    # unset the event_tofbins and check if tof gets set to False
    lm_proj.event_tofbins = None
    assert lm_proj.tof == False

    # test a projector with img_origin = None
    lm_proj2 = parallelproj.ListmodePETProjector(xstart, xend, img_dim, voxel_size)


def test_equalblock_projector(xp, dev, verbose=True):

    # grid shape of LOR endpoints forming a block module
    block_shape = (2, 2, 2)
    # spacing between LOR endpoints in a block module
    block_spacing = (4.0, 3.0, 2.0)
    # radius of the scanner
    scanner_radius = 10

    aff1 = xp.eye(4, device=dev)
    aff1[1, -1] = scanner_radius

    aff2 = xp.eye(4, device=dev)
    aff2[1, -1] = -scanner_radius

    block1 = parallelproj.BlockPETScannerModule(
        xp,
        dev,
        block_shape,
        block_spacing,
        affine_transformation_matrix=aff1,
    )

    block2 = parallelproj.BlockPETScannerModule(
        xp,
        dev,
        block_shape,
        block_spacing,
        affine_transformation_matrix=aff2,
    )

    scanner = parallelproj.ModularizedPETScannerGeometry([block1, block2])

    lor_desc = parallelproj.EqualBlockPETLORDescriptor(
        scanner,
        xp.asarray(
            [
                [0, 1],
            ]
        ),
    )

    img_shape = (14, 10, 3)
    voxel_size = (1.0, 1.0, 1.0)
    img = xp.ones(img_shape, dtype=xp.float32, device=dev)
    img_origin = xp.asarray([-6.5, -4.5, -1.0], dtype=xp.float32, device=dev)

    proj = parallelproj.EqualBlockPETProjector(lor_desc, img_shape, voxel_size)
    assert proj.adjointness_test(xp, dev)
    assert proj.xp == xp
    assert proj.dev == dev
    assert proj.in_shape == img_shape
    assert proj.out_shape == (
        lor_desc.num_block_pairs,
        lor_desc.num_lors_per_block_pair,
    )
    assert proj.tof == False
    assert proj.lor_descriptor == lor_desc
    assert allclose(img_origin, proj.img_origin)
    assert allclose(
        proj.voxel_size, xp.asarray(voxel_size, dtype=xp.float32, device=dev)
    )

    img_fwd = proj(img)
    ones_back = proj.adjoint(xp.ones_like(img_fwd))

    with pytest.raises(Exception):
        proj.tof = True

    # test TOF projector

    proj_tof = parallelproj.EqualBlockPETProjector(
        lor_desc, img_shape, voxel_size, img_origin=img_origin
    )
    tof_params = parallelproj.TOFParameters(
        num_tofbins=27, tofbin_width=0.8, sigma_tof=2.0, num_sigmas=3.0
    )

    proj_tof.tof_parameters = tof_params

    assert proj_tof.adjointness_test(xp, dev)
    assert proj_tof.tof == True
    assert proj_tof.tof_parameters == tof_params

    img_fwd_tof = proj_tof(img)
    ones_back_tof = proj_tof.adjoint(xp.ones_like(img_fwd_tof))

    fig = plt.figure(figsize=(8, 4), tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    proj.show_geometry(ax)
    fig.show()
    plt.close(fig)

    proj_tof.tof = False
    proj_tof.tof = True

    with pytest.raises(Exception):
        proj_tof.tof_parameters = 2

    proj_tof.tof_parameters = None
    assert proj_tof.tof == False
    proj_tof.tof_parameters = tof_params
    assert proj_tof.tof == True
