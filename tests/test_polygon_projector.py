from __future__ import annotations

import array_api_compat.numpy as np
import pytest
import matplotlib.pyplot as plt
from types import ModuleType

import parallelproj.pet_lors as ppl
import parallelproj.pet_scanners as pps
import parallelproj.projectors as ppp
import parallelproj.tof as ppt
from parallelproj import to_numpy_array

from .config import pytestmark

from copy import copy


def test_polygon_projector(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    symmetry_axis = 2
    num_sides = 17
    radius = 180
    radial_trim = 51
    max_ring_difference = 1

    voxel_size = (4.0, 4.0, 2.66)
    img_shape = (53, 53, 5)
    sinogram_order = ppl.SinogramSpatialAxisOrder.RVP

    # setup a test image with 3 hot rods
    x = xp.zeros(img_shape, dtype=xp.float32, device=dev)
    x[img_shape[0] // 2, img_shape[1] // 2, 1:] = 1.0
    x[-3, img_shape[1] // 2, :-1] = 1.0
    x[img_shape[0] // 2, -3, 1:] = 1.0

    # define the scanner geometry, lor descriptor and projector
    scanner = pps.DemoPETScannerGeometry(
        xp,
        dev,
        num_rings=num_rings,
        num_sides=num_sides,
        radius=radius,
        symmetry_axis=symmetry_axis,
    )

    lor_desc = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, max_ring_difference, span=1),
        radial_trim=radial_trim,
        sinogram_order=sinogram_order,
    )

    proj = ppp.RegularPolygonPETProjector(lor_desc, img_shape, voxel_size)
    assert proj.out_shape == (lor_desc.num_rad, lor_desc.num_views, lor_desc.num_planes)

    # non-TOF projections
    assert isinstance(str(proj), str)
    x_fwd = proj(x)
    y = xp.ones(x_fwd.shape, dtype=xp.float32, device=dev)
    y_back = proj.adjoint(y)

    # test that a non-integer sinogram raises a TypeError
    with pytest.raises(TypeError):
        proj.convert_sinogram_to_listmode(
            xp.zeros(proj.out_shape, dtype=xp.float32, device=dev)
        )

    # test conversion to LM
    # LM converter run over views and TOF bins which determines the output order
    # of the events
    test_sino = xp.zeros(proj.out_shape, dtype=xp.int16, device=dev)
    test_sino[0, 0, 1] = 1
    test_sino[1, 0, 0] = 2
    test_sino[0, 1, 1] = 1

    (
        event_start_coords,
        event_end_coords,
        event_tofbins,
    ) = proj.convert_sinogram_to_listmode(test_sino)

    # test shuffle=True (non-TOF): output should have same shape, tofbins still None
    (
        event_start_coords_sh,
        event_end_coords_sh,
        event_tofbins_sh,
    ) = proj.convert_sinogram_to_listmode(test_sino, shuffle=True)
    assert event_start_coords_sh.shape == event_start_coords.shape
    assert event_end_coords_sh.shape == event_end_coords.shape
    assert event_tofbins_sh is None

    assert event_start_coords.shape == (4, 3)
    assert event_end_coords.shape == (4, 3)
    assert event_tofbins is None

    xstart, xend = lor_desc.get_lor_coordinates()

    assert xp.all(event_start_coords[0, :] == xstart[0, 0, 1, :])
    assert xp.all(event_start_coords[1, :] == xstart[1, 0, 0, :])
    assert xp.all(event_start_coords[2, :] == xstart[1, 0, 0, :])
    assert xp.all(event_start_coords[3, :] == xstart[0, 1, 1, :])

    assert xp.all(event_end_coords[0, :] == xend[0, 0, 1, :])
    assert xp.all(event_end_coords[1, :] == xend[1, 0, 0, :])
    assert xp.all(event_end_coords[2, :] == xend[1, 0, 0, :])
    assert xp.all(event_end_coords[3, :] == xend[0, 1, 1, :])

    tof_params = ppt.TOFParameters(num_tofbins=7, tofbin_width=30.6, sigma_tof=15.0, num_sigmas=3.0)
    proj.tof_parameters = tof_params
    assert proj.out_shape == (
        lor_desc.num_rad,
        lor_desc.num_views,
        lor_desc.num_planes,
        tof_params.num_tofbins,
    )

    assert isinstance(str(proj), str)
    x_fwd_tof = proj(x)
    y_tof = xp.ones(x_fwd_tof.shape, dtype=xp.float32, device=dev)
    y_back_tof = proj.adjoint(y_tof)

    # test conversion to LM
    # LM converter run over views and TOF bins which determines the output order
    # of the events
    test_sino = xp.zeros(proj.out_shape, dtype=xp.int16, device=dev)
    test_sino[0, 0, 1, 0] = 1
    test_sino[1, 0, 0, 0] = 1
    test_sino[1, 0, 0, 1] = 1
    test_sino[0, 1, 1, 0] = 1
    test_sino[0, 1, 1, -1] = 2

    (
        event_start_coords,
        event_end_coords,
        event_tofbins,
    ) = proj.convert_sinogram_to_listmode(test_sino)

    assert event_start_coords.shape == (6, 3)
    assert event_end_coords.shape == (6, 3)
    assert event_tofbins.shape == (6,)

    xstart, xend = lor_desc.get_lor_coordinates()

    assert xp.all(event_start_coords[0, :] == xstart[0, 0, 1, :])
    assert xp.all(event_start_coords[1, :] == xstart[1, 0, 0, :])
    assert xp.all(event_start_coords[2, :] == xstart[1, 0, 0, :])
    assert xp.all(event_start_coords[3, :] == xstart[0, 1, 1, :])
    assert xp.all(event_start_coords[4, :] == xstart[0, 1, 1, :])
    assert xp.all(event_start_coords[5, :] == xstart[0, 1, 1, :])

    assert xp.all(event_end_coords[0, :] == xend[0, 0, 1, :])
    assert xp.all(event_end_coords[1, :] == xend[1, 0, 0, :])
    assert xp.all(event_end_coords[2, :] == xend[1, 0, 0, :])
    assert xp.all(event_end_coords[3, :] == xend[0, 1, 1, :])
    assert xp.all(event_end_coords[4, :] == xend[0, 1, 1, :])
    assert xp.all(event_end_coords[5, :] == xend[0, 1, 1, :])

    assert xp.all(
        event_tofbins
        == (
            xp.asarray(
                [
                    0,
                    0,
                    1,
                    0,
                    proj.tof_parameters.num_tofbins - 1,
                    proj.tof_parameters.num_tofbins - 1,
                ],
                device=dev,
                dtype=xp.int16,
            )
        )
    )

    # test shuffle=True (TOF): output should have same shape, tofbins not None
    (
        event_start_coords_sh,
        event_end_coords_sh,
        event_tofbins_sh,
    ) = proj.convert_sinogram_to_listmode(test_sino, shuffle=True)
    assert event_start_coords_sh.shape == event_start_coords.shape
    assert event_end_coords_sh.shape == event_end_coords.shape
    assert event_tofbins_sh is not None
    assert event_tofbins_sh.shape == event_tofbins.shape

    # setup a projector with non default image origin and views
    views = xp.asarray([0, 1], device=dev)
    img_origin = xp.asarray([-100, -100, -5], device=dev, dtype=xp.float32)
    proj2 = ppp.RegularPolygonPETProjector(
        lor_desc, img_shape, voxel_size, views=views, img_origin=img_origin
    )

    assert xp.all(proj2.views == views)
    assert xp.all(proj2.img_origin == img_origin)

    assert proj2.in_shape == img_shape
    assert proj2.out_shape == (lor_desc.num_rad, views.shape[0], lor_desc.num_planes)

    with pytest.raises(ValueError):
        # should raise an error since we have not set the TOF parameters
        proj2.tof = True

    proj2.tof_parameters = tof_params
    proj2.tof = True

    assert proj2.tof_parameters == tof_params
    assert proj2.tof

    proj2.tof = False
    assert proj2.tof == False

    proj2.tof_parameters = tof_params
    assert proj2.tof

    # setting tof_parameters to None should set tof to False
    proj2.tof_parameters = None
    assert proj2.tof == False

    with pytest.raises(ValueError):
        # should raise an error if we don't pass None | TOFParameters
        proj2.tof_parameters = 3.5

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(121, projection="3d")
    proj.show_geometry(ax)
    plt.close(fig)


def _get_slice(r, v, p, lor_descriptor):
    sl = [slice(0), slice(0), slice(0)]

    sl[lor_descriptor.radial_axis_num] = slice(r, r + 1, None)
    sl[lor_descriptor.view_axis_num] = slice(v, v + 1, None)
    sl[lor_descriptor.plane_axis_num] = slice(p, p + 1, None)

    return tuple(sl)


def test_minimal_reg_polygon_projector(xp, dev) -> None:
    """test forward joseph non-tof forward projection with a minimal scanner geometry
    using a 3x3x3 image and a single voxel !=0 at the center of the image
    """

    radius = 12.0
    z = radius / float(np.sqrt(2.0))
    vox_size = 1.8
    vox_value = 2.7
    img_size = 3

    img_shape = (img_size, img_size, img_size)

    # setup a test image with a single voxel != 0 at the center of the image
    x = xp.zeros(img_shape, device=dev, dtype=xp.float32)
    x[img_shape[0] // 2, img_shape[1] // 2, img_shape[2] // 2] = vox_value
    # setup a test image where all voxels have the same value
    x2 = vox_value * xp.ones(img_shape, device=dev, dtype=xp.float32)

    for symmetry_axis in [0, 1, 2]:
        scanner = pps.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=radius,
            num_sides=8,
            num_lor_endpoints_per_side=1,
            lor_spacing=1.0,
            ring_positions=xp.asarray([-z, 0, z], device=dev),
            symmetry_axis=symmetry_axis,
        )

        for sinogram_order in ppl.SinogramSpatialAxisOrder:
            lor_desc = ppl.RegularPolygonPETLORDescriptor(
                scanner, radial_trim=1, sinogram_order=sinogram_order
            )

            proj = ppp.RegularPolygonPETProjector(
                lor_desc, img_shape=img_shape, voxel_size=(vox_size, vox_size, vox_size)
            )

            x_fwd = proj(x)

            assert proj.xstart is not None
            assert proj.xend is not None

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(3)
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 0, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 2, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 0, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd[_get_slice(2, 2, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                vox_value * vox_size * np.sqrt(3),
            )

            # check "central" (straight through) projection which should be vox
            # size * vox value
            assert np.isclose(
                float(x_fwd[_get_slice(2, 1, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size,
            )
            assert np.isclose(
                float(x_fwd[_get_slice(2, 3, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size,
            )

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(2)
            assert np.isclose(
                float(x_fwd[_get_slice(2, 0, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size * np.sqrt(2),
            )
            assert np.isclose(
                float(x_fwd[_get_slice(2, 2, 1, lor_desc)][0, 0, 0]),
                vox_value * vox_size * np.sqrt(2),
            )

            x_fwd2 = proj(x2)

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(3)
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 0, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 2, lor_desc.num_planes - 1, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 0, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )
            assert np.isclose(
                float(
                    x_fwd2[_get_slice(2, 2, lor_desc.num_planes - 2, lor_desc)][0, 0, 0]
                ),
                img_size * vox_value * vox_size * np.sqrt(3),
            )

            # check "central" (straight through) projection which should be vox
            # size * vox value
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 1, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size,
            )
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 3, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size,
            )

            # check "corner to corner" projection which should be vox size *
            # vox value * sqrt(2)
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 0, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size * np.sqrt(2),
            )
            assert np.isclose(
                float(x_fwd2[_get_slice(2, 2, 1, lor_desc)][0, 0, 0]),
                img_size * vox_value * vox_size * np.sqrt(2),
            )

            # setup the same projector without caching of the LOR endpoints
            projb = ppp.RegularPolygonPETProjector(
                lor_desc,
                img_shape=img_shape,
                voxel_size=(vox_size, vox_size, vox_size),
                cache_lor_endpoints=False,
            )
            x_fwd2b = projb(x2)

            assert projb.adjointness_test(xp, dev, dtype=xp.float32)

            assert projb.xstart is None
            assert projb.xend is None
            assert projb.lor_descriptor == lor_desc

            # check whether the projections with and without caching the LOR
            # endpoints are the same
            assert np.allclose(
                to_numpy_array(x_fwd2b),
                to_numpy_array(x_fwd2),
            )

            # check whether the caching works if we first call the adjoint
            proj4 = ppp.RegularPolygonPETProjector(
                lor_desc,
                img_shape=img_shape,
                voxel_size=(vox_size, vox_size, vox_size),
                cache_lor_endpoints=True,
            )

            y = xp.ones(proj4.out_shape, dtype=xp.float32, device=dev)
            tmp = proj4.adjoint(y)

            assert proj4.xstart is not None
            assert proj4.xend is not None

            proj4.clear_cached_lor_endpoints()
            assert proj4.xstart is None
            assert proj4.xend is None

            tmp = proj4.adjoint(y)

            assert proj4.xstart is not None
            assert proj4.xend is not None

            proj5 = copy(proj)
            subset_views = xp.asarray([0, 1], device=dev)
            proj5.views = xp.asarray(subset_views, device=dev)

            # if we reset the views, the cached LOR endpoints should be cleared
            assert proj5.xstart is None
            assert proj5.xend is None

            x_fwd5 = proj5(x)

            assert bool(xp.all(proj5.views == subset_views))
