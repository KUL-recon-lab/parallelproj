from __future__ import annotations

import pytest
import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
import matplotlib.pyplot as plt

from types import ModuleType

from .config import pytestmark


def test_pet_lors(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    symmetry_axis = 2
    scanner = pps.DemoPETScannerGeometry(
        xp, dev, num_rings, symmetry_axis=symmetry_axis
    )

    radial_trim = 65
    max_ring_difference = 2

    for sinogram_order in ppl.SinogramSpatialAxisOrder:
        lor_desc = ppl.RegularPolygonPETLORDescriptor(
            scanner,
            radial_trim=radial_trim,
            max_ring_difference=max_ring_difference,
            sinogram_order=sinogram_order,
        )

        assert lor_desc.scanner == scanner
        assert lor_desc.max_ring_difference == max_ring_difference
        assert lor_desc.radial_trim == radial_trim
        assert lor_desc.num_views == scanner.num_lor_endpoints_per_ring // 2

        assert lor_desc.plane_axis_num == sinogram_order.name.find("P")
        assert lor_desc.radial_axis_num == sinogram_order.name.find("R")
        assert lor_desc.view_axis_num == sinogram_order.name.find("V")

        lor_coords = lor_desc.get_lor_coordinates()

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        scanner.show_lor_endpoints(ax, show_linear_index=False)
        lor_desc.show_views(
            ax,
            views=xp.asarray([0], device=dev),
            planes=xp.asarray([0], device=dev),
            lw=0.1,
        )
        fig.show()
        plt.close(fig)

    # test lor descriptor without max_ring_difference and radial_trim
    scanner2 = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=15,
        lor_spacing=2.3,
        ring_positions=xp.linspace(-10, 10, num_rings, device=dev),
        symmetry_axis=2,
    )

    num_subsets = 9

    # test the distribution of views into subsets
    for sinogram_order in ppl.SinogramSpatialAxisOrder:
        lor_desc2 = ppl.RegularPolygonPETLORDescriptor(
            scanner2,
            radial_trim=10,
            max_ring_difference=None,
            sinogram_order=sinogram_order,
        )

        for num_dim in [3, 4]:
            subset_views, subset_slices = lor_desc2.get_distributed_views_and_slices(
                num_subsets, num_dim
            )

            assert bool(
                xp.all(
                    xp.sort(xp.concat(subset_views))
                    == xp.arange(lor_desc2.num_views, device=dev)
                )
            )

            assert bool(
                xp.all(
                    subset_views[0]
                    == xp.asarray([0, 9, 18, 27, 36, 45, 54, 63, 72, 81], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[1]
                    == xp.asarray([4, 13, 22, 31, 40, 49, 58, 67, 76, 85], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[2]
                    == xp.asarray([8, 17, 26, 35, 44, 53, 62, 71, 80, 89], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[3]
                    == xp.asarray([1, 10, 19, 28, 37, 46, 55, 64, 73, 82], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[4]
                    == xp.asarray([5, 14, 23, 32, 41, 50, 59, 68, 77, 86], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[5]
                    == xp.asarray([2, 11, 20, 29, 38, 47, 56, 65, 74, 83], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[6]
                    == xp.asarray([6, 15, 24, 33, 42, 51, 60, 69, 78, 87], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[7]
                    == xp.asarray([3, 12, 21, 30, 39, 48, 57, 66, 75, 84], device=dev)
                )
            )
            assert bool(
                xp.all(
                    subset_views[8]
                    == xp.asarray([7, 16, 25, 34, 43, 52, 61, 70, 79, 88], device=dev)
                )
            )

            sl = num_dim * [slice(None)]
            sl[lor_desc2.view_axis_num] = slice(0, None, num_subsets)
            assert subset_slices[0] == tuple(sl)


def test_abstract_get_lor_coordinates(xp: ModuleType, dev: str) -> None:
    """line 58: NotImplementedError in abstract base get_lor_coordinates"""
    num_rings = 3
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings, symmetry_axis=2)

    class _ConcreteDesc(ppl.PETLORDescriptor):
        def get_lor_coordinates(self):
            return super().get_lor_coordinates()

    desc = _ConcreteDesc(scanner)
    with pytest.raises(NotImplementedError):
        desc.get_lor_coordinates()


def test_regular_polygon_lor_desc_span(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings, symmetry_axis=2)

    # line 270: even span must raise
    with pytest.raises(ValueError, match="span must be odd"):
        ppl.RegularPolygonPETLORDescriptor(scanner, span=2)

    # span=1 descriptor
    lor_desc_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=1
    )

    # line 319: .span property
    assert lor_desc_s1.span == 1

    # lines 328, 337: start/end_plane_index happy path (span=1)
    _ = lor_desc_s1.start_plane_index
    _ = lor_desc_s1.end_plane_index

    # lines 342, 347, 352, 357: z/multiplicity/segment properties (span=1)
    _ = lor_desc_s1.start_plane_z
    _ = lor_desc_s1.end_plane_z
    _ = lor_desc_s1.plane_multiplicity
    _ = lor_desc_s1.plane_segment

    # line 401: __str__
    s = str(lor_desc_s1)
    assert "RegularPolygonPETLORDescriptor" in s

    # span=3 descriptor (lines 439, 506-556: _setup_spanned_plane_indices)
    lor_desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=3
    )

    assert lor_desc_s3.span == 3
    assert lor_desc_s3.num_planes > 0

    # lines 324-328: start_plane_index raises for span > 1
    with pytest.raises(AttributeError):
        _ = lor_desc_s3.start_plane_index

    # lines 333-337: end_plane_index raises for span > 1
    with pytest.raises(AttributeError):
        _ = lor_desc_s3.end_plane_index

    # lines 342, 347, 352, 357: same properties accessible for span > 1
    _ = lor_desc_s3.start_plane_z
    _ = lor_desc_s3.end_plane_z
    _ = lor_desc_s3.plane_multiplicity
    _ = lor_desc_s3.plane_segment

    # lines 426-432: _ring_diff_to_segment with span=3 (half_span=1)
    assert lor_desc_s3._ring_diff_to_segment(0) == 0
    assert lor_desc_s3._ring_diff_to_segment(1) == 0  # |rd|=1 <= half_span=1
    assert lor_desc_s3._ring_diff_to_segment(-1) == 0
    assert lor_desc_s3._ring_diff_to_segment(2) == 1  # ceil((2-1)/3)=1
    assert lor_desc_s3._ring_diff_to_segment(-2) == -1

    # get_lor_coordinates works end-to-end with span > 1
    xs, xe = lor_desc_s3.get_lor_coordinates()
    assert xs.shape[-1] == 3
    assert xe.shape[-1] == 3


def test_show_michelogram(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings, symmetry_axis=2)

    # span=1: basic path, no merge lines (lines 711-799, 825)
    lor_desc_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=1
    )
    fig, ax = plt.subplots()
    lor_desc_s1.show_michelogram(ax, show_merge_lines=True)
    plt.close(fig)

    # span=3: exercises merge-line branch (line 767 condition True)
    lor_desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=3
    )
    fig, ax = plt.subplots()
    lor_desc_s3.show_michelogram(ax, show_merge_lines=True)
    plt.close(fig)

    # show_merge_lines=False: skips the merge-line branch
    fig, ax = plt.subplots()
    lor_desc_s3.show_michelogram(ax, show_merge_lines=False)
    plt.close(fig)


def test_show_segment_lors(xp: ModuleType, dev: str) -> None:
    import numpy as np

    num_rings = 3
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings, symmetry_axis=2)

    # span=1, mrd=1: only segment 0, n_rows=1 (no negative segments)
    lor_desc_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=1, span=1
    )
    fig = lor_desc_s1.show_segment_lors()
    plt.close(fig)

    # span=3, mrd=2: segments -1, 0, +1 → n_rows=2, n_cols=2 (lines 863-1013)
    # covers: Michelogram inset (row=1, col=0), legend (row=0, col=0),
    # compressed kwargs, uncompressed kwargs, tight_layout
    lor_desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=3
    )
    fig = lor_desc_s3.show_segment_lors()
    plt.close(fig)

    # custom kwargs (lines 881-885)
    fig = lor_desc_s3.show_segment_lors(
        uncompressed_lor_kwargs={"alpha": 0.3},
        compressed_lor_kwargs={"alpha": 0.8, "color": "blue"},
    )
    plt.close(fig)

    # pre-existing axes: covers the else branch (lines 896-900)
    # with span=3, mrd=2 on 3 rings: abs_segs={0,1} → n_cols=2, n_rows=2
    fig_pre, axs_pre = plt.subplots(2, 2, squeeze=False)
    lor_desc_s3.show_segment_lors(axs=axs_pre)
    plt.close(fig_pre)

    # lines 940-941: ax.axis("off") branch when a negative segment is absent.
    # Symmetric scanners always produce symmetric segments, so we monkeypatch
    # _plane_segment to {-1, 0, 1, 2} — negative seg -1 exists (→ n_rows=2),
    # but +2 has no matching -2, so the (row=1, col=2) cell hits ax.axis("off").
    lor_desc_patch = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=1
    )
    lor_desc_patch._plane_segment = xp.asarray([-1, 0, 1, 2], device=dev, dtype=xp.int32)
    lor_desc_patch._start_plane_z = xp.asarray([0.0, 0.0, 0.0, 0.0], device=dev, dtype=xp.float32)
    lor_desc_patch._end_plane_z = xp.asarray([0.0, 0.0, 0.0, 0.0], device=dev, dtype=xp.float32)
    fig = lor_desc_patch.show_segment_lors()
    plt.close(fig)


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

    block3 = pps.BlockPETScannerModule(
        xp,
        dev,
        (2, 2, 3),
        (4.0, 4.0, 4.0),
        affine_transformation_matrix=aff2,
    )

    scanner = pps.ModularizedPETScannerGeometry([block1, block2])

    lor_desc = ppl.EqualBlockPETLORDescriptor(
        scanner,
        xp.asarray(
            [
                [0, 1],
            ],
            device=dev,
        ),
    )

    assert lor_desc.scanner == scanner
    assert (
        xp.max(xp.abs(lor_desc.all_block_pairs - xp.asarray([[0, 1]], device=dev))) == 0
    )
    assert lor_desc.num_block_pairs == 1
    assert lor_desc.num_lorendpoints_per_block == 8
    assert lor_desc.num_lors_per_block_pair == 64
    assert lor_desc.xp == xp
    assert lor_desc.dev == dev

    scanner2 = pps.ModularizedPETScannerGeometry([block1, block2, block3])
    with pytest.raises(Exception):
        lor_desc2 = ppl.EqualBlockPETLORDescriptor(
            scanner2,
            xp.asarray(
                [
                    [0, 1],
                    [1, 2],
                ],
                device=dev,
            ),
        )

    fig3 = plt.figure(tight_layout=True)
    ax3 = fig3.add_subplot(111, projection="3d")
    scanner.show_lor_endpoints(ax3, annotation_fontsize=4, show_linear_index=False)
    lor_desc.show_block_pair_lors(ax3, block_pair_nums=None, color=plt.cm.tab10(0))
    fig3.show()
    plt.close(fig3)
