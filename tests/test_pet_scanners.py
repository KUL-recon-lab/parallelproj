from __future__ import annotations

import pytest
import parallelproj.pet_scanners as pps
import matplotlib.pyplot as plt

from types import ModuleType

from .config import pytestmark


def test_regular_polygon_pet_module(xp: ModuleType, dev: str) -> None:
    radius = 700.0
    num_sides = 28
    num_lor_endpoints_per_side = 16
    lor_spacing = 4.0

    aff_mat = xp.zeros((4, 4), device=dev, dtype=xp.float32)
    aff_mat[0, 0] = 1.0
    aff_mat[1, 1] = 1.0
    aff_mat[2, 2] = 1.0
    aff_mat[0, -1] = 0.0
    aff_mat[1, -1] = 0.0
    aff_mat[2, -1] = 10.0

    ax0 = 1
    ax1 = 0

    mod = pps.RegularPolygonPETScannerModule(
        xp,
        dev,
        radius=radius,
        num_sides=num_sides,
        num_lor_endpoints_per_side=num_lor_endpoints_per_side,
        lor_spacing=lor_spacing,
        affine_transformation_matrix=aff_mat,
        ax0=ax0,
        ax1=ax1,
    )

    assert mod.radius == radius
    assert mod.num_sides == num_sides
    assert mod.num_lor_endpoints_per_side == num_lor_endpoints_per_side
    assert mod.lor_spacing == lor_spacing
    assert mod.xp == xp
    assert mod.dev == dev

    assert mod.num_lor_endpoints == num_sides * num_lor_endpoints_per_side
    assert bool(
        xp.all(mod.lor_endpoint_numbers == xp.arange(mod.num_lor_endpoints, device=dev))
    )
    assert xp.all(mod.affine_transformation_matrix == aff_mat)

    assert ax0 == mod.ax0
    assert ax1 == mod.ax1
    assert lor_spacing == mod.lor_spacing

    raw_points = mod.get_raw_lor_endpoints()
    transformed_points = mod.get_lor_endpoints()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    mod.show_lor_endpoints(ax, transformed=False)
    mod.show_lor_endpoints(ax, transformed=True, annotation_fontsize=8)
    fig.show()
    plt.close(fig)

    # test module withput affine transformation matrix
    mod2 = pps.RegularPolygonPETScannerModule(
        xp,
        dev,
        radius=radius,
        num_sides=num_sides,
        num_lor_endpoints_per_side=num_lor_endpoints_per_side,
        lor_spacing=lor_spacing,
        ax0=ax0,
        ax1=ax1,
    )

    aff_mat2 = xp.eye(4, device=dev)
    aff_mat2[-1, -1] = 1

    assert xp.all(mod2.affine_transformation_matrix == aff_mat2)


def test_regular_polygon_pet_scanner(xp: ModuleType, dev: str) -> None:
    num_rings = 4

    for symmetry_axis in [0, 1, 2]:
        scanner = pps.DemoPETScannerGeometry(
            xp, dev, num_rings=num_rings, symmetry_axis=symmetry_axis
        )

        num_sides = 34
        num_lor_endpoints_per_side = 16

        assert scanner.num_rings == num_rings
        assert scanner.symmetry_axis == symmetry_axis

        assert scanner.radius == 0.5 * (744.1 + 2 * 8.51)
        assert scanner.num_sides == num_sides

        assert scanner.num_lor_endpoints_per_side == num_lor_endpoints_per_side
        assert scanner.lor_spacing == 4.03125

        assert (
            scanner.num_lor_endpoints_per_ring == num_sides * num_lor_endpoints_per_side
        )

        ring_positions = scanner.ring_positions

        mods = scanner.modules
        assert scanner.num_modules == num_rings

        endpoint_coords = scanner.all_lor_endpoints
        endpoint_mod_number = scanner.all_lor_endpoints_module_number
        endpoint_index_offset = scanner.all_lor_endpoints_index_offset

        mods = xp.asarray([0, 0, 1], device=dev)
        in_mods = xp.asarray([0, 1, 0], device=dev)
        lin_index = scanner.linear_lor_endpoint_index(mods, in_mods)

        assert xp.all(
            lin_index
            == xp.asarray([0, 1, scanner.num_lor_endpoints_per_ring], device=dev)
        )

        x_lor = scanner.get_lor_endpoints(mods, in_mods)

        assert xp.all(x_lor == xp.take(endpoint_coords, lin_index, axis=0))

        i_in_ring = scanner.all_lor_endpoints_index_in_ring

        assert xp.all(
            i_in_ring
            == (
                xp.arange(num_rings * scanner.num_lor_endpoints_per_ring, device=dev)
                % scanner.num_lor_endpoints_per_ring
            )
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    scanner.show_lor_endpoints(ax, show_linear_index=False)
    scanner.show_lor_endpoints(ax, show_linear_index=True)
    fig.show()
    plt.close(fig)

    # test scanner with manually specified azimuthal angles of the sides
    phis = xp.asarray([0.0, xp.pi / 4], dtype=xp.float32, device=dev)

    scanner2 = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=150,
        num_sides=2,
        num_lor_endpoints_per_side=num_lor_endpoints_per_side,
        lor_spacing=2.5,
        ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
        symmetry_axis=1,
        phis=phis,
    )

    assert xp.all(scanner2.modules[0].phis == phis)


def test_regular_equal_block_scanner(xp: ModuleType, dev: str) -> None:

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

    block1 = pps.BlockPETScannerModule(
        xp,
        dev,
        block_shape,
        block_spacing,
        affine_transformation_matrix=aff1,
    )

    block2 = pps.BlockPETScannerModule(
        xp,
        dev,
        block_shape,
        block_spacing,
        affine_transformation_matrix=aff2,
    )

    assert block1.shape == block_shape
    assert block1.spacing == block_spacing
    lor_endpoints1a = block1.lor_endpoints
    lor_endpoints1b = xp.asarray(
        [
            [-2.0, 8.5, -1.0],
            [-2.0, 8.5, 1.0],
            [-2.0, 11.5, -1.0],
            [-2.0, 11.5, 1.0],
            [2.0, 8.5, -1.0],
            [2.0, 8.5, 1.0],
            [2.0, 11.5, -1.0],
            [2.0, 11.5, 1.0],
        ],
        device=dev,
    )

    assert xp.max(xp.abs(lor_endpoints1a - lor_endpoints1b)) < 1e-7

    scanner = pps.ModularizedPETScannerGeometry([block1, block2])

    fig = plt.figure(tight_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    scanner.show_lor_endpoints(ax, annotation_fontsize=4, show_linear_index=False)
    fig.show()
    plt.close(fig)


def test_pet_scanner_module_get_raw_lor_endpoints_not_implemented(
    xp: ModuleType, dev: str
) -> None:
    class MinimalModule(pps.PETScannerModule):
        def get_raw_lor_endpoints(self, inds=None):
            return super().get_raw_lor_endpoints(inds)

    mod = MinimalModule(xp, dev, num_lor_endpoints=4)
    with pytest.raises(NotImplementedError):
        mod.get_raw_lor_endpoints()
