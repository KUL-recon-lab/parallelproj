from __future__ import annotations

import warnings
import numpy as np
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


def test_regular_polygon_pet_module_custom_positions(xp: ModuleType, dev: str) -> None:
    """lor_endpoint_positions overrides uniform spacing and gives same result for uniform case."""
    radius = 700.0
    num_sides = 28
    N = 8
    d = 4.0

    # anti-symmetric uniform positions — should match the uniform constructor
    positions = d * (np.arange(N, dtype=np.float32) - (N - 1) / 2.0)

    mod_uniform = pps.RegularPolygonPETScannerModule(
        xp, dev, radius=radius, num_sides=num_sides,
        num_lor_endpoints_per_side=N, lor_spacing=d,
    )
    mod_custom = pps.RegularPolygonPETScannerModule(
        xp, dev, radius=radius, num_sides=num_sides,
        lor_endpoint_positions=positions,
    )

    # shapes and counts must match
    assert mod_custom.num_lor_endpoints_per_side == N
    assert mod_custom.lor_spacing is None
    assert mod_custom.lor_endpoint_positions.shape == (N,)

    # endpoints must be numerically identical
    import array_api_compat
    xp_ = array_api_compat.array_namespace(mod_uniform.get_raw_lor_endpoints())
    assert bool(xp_.all(xp_.abs(
        mod_custom.get_raw_lor_endpoints() - mod_uniform.get_raw_lor_endpoints()
    ) < 1e-4))

    # missing both args raises ValueError
    with pytest.raises(ValueError):
        pps.RegularPolygonPETScannerModule(xp, dev, radius=radius, num_sides=num_sides)

    # asymmetric positions raise UserWarning
    asymmetric = np.array([-3.0, -1.0, 0.5, 2.0], dtype=np.float32)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        pps.RegularPolygonPETScannerModule(
            xp, dev, radius=radius, num_sides=num_sides,
            lor_endpoint_positions=asymmetric,
        )
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
        assert "anti-symmetric" in str(w[0].message)

    # geometry raises ValueError when neither uniform nor custom positions given
    with pytest.raises(ValueError):
        pps.RegularPolygonPETScannerGeometry(
            xp, dev, radius=radius, num_sides=num_sides,
            ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
            symmetry_axis=2,
        )

    # geometry also supports lor_endpoint_positions
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp, dev,
        radius=radius,
        num_sides=num_sides,
        ring_positions=xp.asarray([0.0, 10.0], dtype=xp.float32, device=dev),
        symmetry_axis=2,
        lor_endpoint_positions=positions,
    )
    assert scanner.num_lor_endpoints_per_side == N
    assert scanner.lor_spacing is None
    assert scanner.lor_endpoint_positions.shape == (N,)


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


def test_ring_endpoint_ordering(xp: ModuleType, dev: str) -> None:
    """RingEndpointOrdering property, CCW branch, and reversal invariant."""
    import numpy as np
    from parallelproj import to_numpy_array

    n_sides = 8
    n_per_side = 2

    for symmetry_axis in [0, 1, 2]:
        s_cw = pps.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=1.0,
            num_sides=n_sides,
            num_lor_endpoints_per_side=n_per_side,
            lor_spacing=0.1,
            ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
            symmetry_axis=symmetry_axis,
            ring_endpoint_ordering=pps.RingEndpointOrdering.CLOCKWISE,
        )
        s_ccw = pps.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=1.0,
            num_sides=n_sides,
            num_lor_endpoints_per_side=n_per_side,
            lor_spacing=0.1,
            ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
            symmetry_axis=symmetry_axis,
            ring_endpoint_ordering=pps.RingEndpointOrdering.COUNTERCLOCKWISE,
        )

        # properties are readable and propagate to the module level
        assert s_cw.ring_endpoint_ordering is pps.RingEndpointOrdering.CLOCKWISE
        assert s_ccw.ring_endpoint_ordering is pps.RingEndpointOrdering.COUNTERCLOCKWISE
        assert s_cw.modules[0].ring_endpoint_ordering is pps.RingEndpointOrdering.CLOCKWISE
        assert s_ccw.modules[0].ring_endpoint_ordering is pps.RingEndpointOrdering.COUNTERCLOCKWISE

        # CCW[i] maps to CW[j] where side order is reversed (keeping side 0
        # fixed) and within-side order is also reversed:
        #   j = ((n_sides - i//n_per_side) % n_sides) * n_per_side
        #       + (n_per_side - 1 - i % n_per_side)
        cw_pts  = to_numpy_array(s_cw.all_lor_endpoints)
        ccw_pts = to_numpy_array(s_ccw.all_lor_endpoints)
        n_total = n_sides * n_per_side
        expected_indices = [
            ((n_sides - i // n_per_side) % n_sides) * n_per_side
            + (n_per_side - 1 - i % n_per_side)
            for i in range(n_total)
        ]
        assert np.allclose(ccw_pts, cw_pts[expected_indices], atol=1e-5)


def test_phi0(xp: ModuleType, dev: str) -> None:
    """phi0 rotates side 0; ignored when phis is supplied explicitly."""
    import math
    import numpy as np
    from parallelproj import to_numpy_array

    n_sides = 8
    n_per_side = 2
    phi0_val = math.pi / n_sides  # half a polygon step

    # --- scanner-level property propagates to module level ---
    s = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=1.0,
        num_sides=n_sides,
        num_lor_endpoints_per_side=n_per_side,
        lor_spacing=0.1,
        ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
        symmetry_axis=2,
        phi0=phi0_val,
    )
    assert s.phi0 == phi0_val
    assert s.modules[0].phi0 == phi0_val

    # --- phis are offset by phi0 relative to phi0=0 case ---
    s0 = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=1.0,
        num_sides=n_sides,
        num_lor_endpoints_per_side=n_per_side,
        lor_spacing=0.1,
        ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
        symmetry_axis=2,
        phi0=0.0,
    )
    phis_with = to_numpy_array(s.modules[0].phis)
    phis_base = to_numpy_array(s0.modules[0].phis)
    assert np.allclose(phis_with, phis_base + phi0_val, atol=1e-5)

    # --- endpoint coordinates are consistent with the shifted phis ---
    # phi0 rotates the ring in the (ax0, ax1) plane.
    # For symmetry_axis=2: ax0=col1, ax1=col0.
    # A shift phi → phi+phi0 acts as a rotation in (ax0, ax1) by phi0:
    #   new[:, ax0] = cos(phi0)*old[:, ax0] - sin(phi0)*old[:, ax1]
    #   new[:, ax1] = sin(phi0)*old[:, ax0] + cos(phi0)*old[:, ax1]
    ax0, ax1 = 1, 0  # for symmetry_axis=2
    pts_new = to_numpy_array(s.all_lor_endpoints)
    pts_old = to_numpy_array(s0.all_lor_endpoints)
    cos_a, sin_a = math.cos(phi0_val), math.sin(phi0_val)
    pts_expected = pts_old.copy()
    pts_expected[:, ax0] = cos_a * pts_old[:, ax0] - sin_a * pts_old[:, ax1]
    pts_expected[:, ax1] = sin_a * pts_old[:, ax0] + cos_a * pts_old[:, ax1]
    assert np.allclose(pts_new, pts_expected, atol=1e-5)

    # --- phi0 is ignored when phis are supplied explicitly ---
    phis_explicit = xp.asarray([0.0, math.pi / 2], dtype=xp.float32, device=dev)
    s_explicit = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=1.0,
        num_sides=2,
        num_lor_endpoints_per_side=n_per_side,
        lor_spacing=0.1,
        ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
        symmetry_axis=2,
        phis=phis_explicit,
        phi0=99.0,  # should be ignored
    )
    # phis on the module should be the supplied array, unaffected by phi0
    assert xp.all(s_explicit.modules[0].phis == phis_explicit)


def test_regular_polygon_pet_scanner_invalid_symmetry_axis(
    xp: ModuleType, dev: str
) -> None:
    with pytest.raises(ValueError, match="symmetry_axis"):
        pps.RegularPolygonPETScannerGeometry(
            xp,
            dev,
            radius=150,
            num_sides=4,
            num_lor_endpoints_per_side=2,
            lor_spacing=4.0,
            ring_positions=xp.asarray([0.0], dtype=xp.float32, device=dev),
            symmetry_axis=3,
        )


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
