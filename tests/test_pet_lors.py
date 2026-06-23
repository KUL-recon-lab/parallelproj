from __future__ import annotations

import pytest
import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
import parallelproj.tof as tof
import matplotlib.pyplot as plt

from parallelproj import to_numpy_array

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
            ppl.Michelogram(scanner.num_rings, max_ring_difference, span=1),
            radial_trim=radial_trim,
            sinogram_order=sinogram_order,
        )

        assert lor_desc.scanner == scanner
        assert lor_desc.max_ring_difference == max_ring_difference
        assert lor_desc.radial_trim == radial_trim
        assert lor_desc.num_views == scanner.num_lor_endpoints_per_ring // 2

        assert lor_desc.plane_axis_num == sinogram_order.name.find("P")
        assert lor_desc.radial_axis_num == sinogram_order.name.find("R")
        assert lor_desc.view_axis_num == sinogram_order.name.find("V")

        lor_desc.get_lor_coordinates()

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

    # even span must raise (validated by Michelogram constructor)
    with pytest.raises(ValueError, match="odd"):
        ppl.Michelogram(scanner.num_rings, 2, span=2)

    # mismatched num_rings between Michelogram and scanner must raise
    with pytest.raises(ValueError, match="num_rings"):
        ppl.RegularPolygonPETLORDescriptor(
            scanner, ppl.Michelogram(scanner.num_rings + 1, 2, span=1)
        )

    # span=1 descriptor
    lor_desc_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=1)
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

    # span=3 descriptor
    lor_desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=3)
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

    # segment formula is exposed via the Michelogram (see also
    # test_michelogram_ring_diff_to_segment for thorough coverage).
    assert lor_desc_s3.michelogram.ring_diff_to_segment(0) == 0
    assert lor_desc_s3.michelogram.ring_diff_to_segment(1) == 0
    assert lor_desc_s3.michelogram.ring_diff_to_segment(-1) == 0
    assert lor_desc_s3.michelogram.ring_diff_to_segment(2) == 1
    assert lor_desc_s3.michelogram.ring_diff_to_segment(-2) == -1

    # get_lor_coordinates works end-to-end with span > 1
    xs, xe = lor_desc_s3.get_lor_coordinates()
    assert xs.shape[-1] == 3
    assert xe.shape[-1] == 3

    # Verify correctness of compressed plane z-coordinates for span=3.
    # Use a 3-ring scanner with known, non-uniform ring positions z=[0,1,3].
    #
    # span=3 → half_span=1: rd in {0,±1} → seg 0; rd=±2 → seg ±1
    # Planes are sorted by (|seg|, -seg, axial_midpoint = s+e):
    #
    # plane | (seg, mid) | ring pairs       | start_z | end_z | mult | seg
    #   0   | (0,  0)    | (0,0)            |  0.0    |  0.0  |  1   |  0
    #   1   | (0,  1)    | (0,1),(1,0)      |  0.5    |  0.5  |  2   |  0
    #   2   | (0,  2)    | (1,1)            |  1.0    |  1.0  |  1   |  0
    #   3   | (0,  3)    | (1,2),(2,1)      |  2.0    |  2.0  |  2   |  0
    #   4   | (0,  4)    | (2,2)            |  3.0    |  3.0  |  1   |  0
    #   5   | (+1, 2)    | (0,2)            |  0.0    |  3.0  |  1   | +1
    #   6   | (-1, 2)    | (2,0)            |  3.0    |  0.0  |  1   | -1
    ring_z = xp.asarray([0.0, 1.0, 3.0], device=dev)
    scanner_z = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=1,
        lor_spacing=2.3,
        ring_positions=ring_z,
        symmetry_axis=2,
    )
    lor_desc_z = ppl.RegularPolygonPETLORDescriptor(
        scanner_z, ppl.Michelogram(scanner_z.num_rings, 2, span=3)
    )

    assert lor_desc_z.num_planes == 7

    exp_start_z = xp.asarray(
        [0.0, 0.5, 1.0, 2.0, 3.0, 0.0, 3.0], device=dev, dtype=xp.float32
    )
    exp_end_z = xp.asarray(
        [0.0, 0.5, 1.0, 2.0, 3.0, 3.0, 0.0], device=dev, dtype=xp.float32
    )
    exp_mult = xp.asarray([1, 2, 1, 2, 1, 1, 1], device=dev, dtype=xp.int32)
    exp_seg = xp.asarray([0, 0, 0, 0, 0, 1, -1], device=dev, dtype=xp.int32)

    assert bool(xp.all(xp.abs(lor_desc_z.start_plane_z - exp_start_z) < 1e-5))
    assert bool(xp.all(xp.abs(lor_desc_z.end_plane_z - exp_end_z) < 1e-5))
    assert bool(xp.all(lor_desc_z.plane_multiplicity == exp_mult))
    assert bool(xp.all(lor_desc_z.plane_segment == exp_seg))

    # Verify compressed plane z-coordinates for a 13-ring scanner, span=9, mrd=11.
    # Use uniform ring positions z[k]=k (k=0..12) for easy hand-calculation.
    #
    # span=9 → half_span=4: |rd|≤4 → seg 0; 5≤|rd|≤11 → seg ±1
    #
    # Plane count per segment (grouped by (seg, axial_midpoint=s+e)):
    #   seg  0: mid=0..24  → 25 planes, 97  ring pairs total
    #   seg +1: mid=5..19  → 15 planes, 35  ring pairs total
    #   seg -1: mid=5..19  → 15 planes, 35  ring pairs total
    #   total: 55 planes, 167 ring pairs
    #
    # Spot-checked planes (sorted by (|seg|, -seg, mid)):
    #   plane  0 (seg=0,  mid= 0): pair  (0, 0)                       → sz=0.0,  ez=0.0,  mult=1
    #   plane  4 (seg=0,  mid= 4): pairs (0,4),(1,3),(2,2),(3,1),(4,0) → sz=2.0,  ez=2.0,  mult=5
    #   plane 12 (seg=0,  mid=12): pairs (4,8),(5,7),(6,6),(7,5),(8,4) → sz=6.0,  ez=6.0,  mult=5
    #   plane 24 (seg=0,  mid=24): pair  (12,12)                       → sz=12.0, ez=12.0, mult=1
    #   plane 25 (seg=+1, mid= 5): pair  (0, 5)                        → sz=0.0,  ez=5.0,  mult=1
    #   plane 31 (seg=+1, mid=11): pairs (0,11),(1,10),(2,9),(3,8)     → sz=1.5,  ez=9.5,  mult=4
    #   plane 40 (seg=-1, mid= 5): pair  (5, 0)                        → sz=5.0,  ez=0.0,  mult=1
    #   plane 46 (seg=-1, mid=11): pairs (11,0),(10,1),(9,2),(8,3)     → sz=9.5,  ez=1.5,  mult=4
    ring_z_13 = xp.asarray([float(i) for i in range(13)], device=dev)
    scanner_13 = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=1,
        lor_spacing=2.3,
        ring_positions=ring_z_13,
        symmetry_axis=2,
    )
    lor_desc_13 = ppl.RegularPolygonPETLORDescriptor(
        scanner_13, ppl.Michelogram(scanner_13.num_rings, 11, span=9)
    )

    assert lor_desc_13.num_planes == 55
    assert bool(xp.sum(lor_desc_13.plane_multiplicity) == 167)

    seg_13 = lor_desc_13.plane_segment
    assert bool(
        xp.sum(xp.astype(seg_13 == xp.asarray(0, device=dev, dtype=xp.int32), xp.int32))
        == 25
    )
    assert bool(
        xp.sum(xp.astype(seg_13 == xp.asarray(1, device=dev, dtype=xp.int32), xp.int32))
        == 15
    )
    assert bool(
        xp.sum(
            xp.astype(seg_13 == xp.asarray(-1, device=dev, dtype=xp.int32), xp.int32)
        )
        == 15
    )

    def _check_plane(idx, exp_sz, exp_ez, exp_mult, exp_seg):
        assert bool(xp.abs(lor_desc_13.start_plane_z[idx] - exp_sz) < 1e-5)
        assert bool(xp.abs(lor_desc_13.end_plane_z[idx] - exp_ez) < 1e-5)
        assert bool(lor_desc_13.plane_multiplicity[idx] == exp_mult)
        assert bool(lor_desc_13.plane_segment[idx] == exp_seg)

    _check_plane(0, 0.0, 0.0, 1, 0)
    _check_plane(4, 2.0, 2.0, 5, 0)
    _check_plane(12, 6.0, 6.0, 5, 0)
    _check_plane(24, 12.0, 12.0, 1, 0)
    _check_plane(25, 0.0, 5.0, 1, 1)
    _check_plane(31, 1.5, 9.5, 4, 1)
    _check_plane(40, 5.0, 0.0, 1, -1)
    _check_plane(46, 9.5, 1.5, 4, -1)


def test_zig_zag_order(xp: ModuleType, dev: str) -> None:
    """zig_zag_order property and START_FIRST branch are exercised."""
    import numpy as np

    num_rings = 1
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=100.0,
        num_sides=8,
        num_lor_endpoints_per_side=1,
        lor_spacing=1.0,
        ring_positions=xp.asarray([0.0], device=dev),
        symmetry_axis=2,
    )

    lor_ef = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=0,
        zig_zag_order=ppl.SinogramZigZagOrder.END_FIRST,
    )
    lor_sf = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        radial_trim=0,
        zig_zag_order=ppl.SinogramZigZagOrder.START_FIRST,
    )

    # property is readable and returns the correct enum value
    assert lor_ef.zig_zag_order is ppl.SinogramZigZagOrder.END_FIRST
    assert lor_sf.zig_zag_order is ppl.SinogramZigZagOrder.START_FIRST

    # both share the same sinogram shape
    assert lor_ef.spatial_sinogram_shape == lor_sf.spatial_sinogram_shape

    # view 0: END_FIRST has end step first, START_FIRST has start step first
    # n=8: END_FIRST pairs: (0,7),(0,6),(1,6),...  START_FIRST: (0,7),(1,7),(1,6),...
    ef_start = to_numpy_array(lor_ef.start_in_ring_index[0, :])
    ef_end = to_numpy_array(lor_ef.end_in_ring_index[0, :])
    sf_start = to_numpy_array(lor_sf.start_in_ring_index[0, :])
    sf_end = to_numpy_array(lor_sf.end_in_ring_index[0, :])

    # first radial bin is the same for both (central LOR)
    assert int(ef_start[0]) == int(sf_start[0])
    assert int(ef_end[0]) == int(sf_end[0])

    # second radial bin differs: END_FIRST keeps start, advances end;
    # START_FIRST advances start, keeps end
    assert int(ef_start[1]) == int(ef_start[0])  # start unchanged
    assert int(sf_end[1]) == int(sf_end[0])       # end unchanged
    assert int(ef_end[1]) != int(ef_end[0])        # end stepped
    assert int(sf_start[1]) != int(sf_start[0])   # start stepped

    # the two conventions produce different index arrays
    assert not np.array_equal(ef_start, sf_start)
    assert not np.array_equal(ef_end, sf_end)


def test_show_michelogram(xp: ModuleType, dev: str) -> None:
    num_rings = 3
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings, symmetry_axis=2)

    # span=1: basic path, no merge lines (lines 711-799, 825)
    lor_desc_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=1)
    )
    fig, ax = plt.subplots()
    lor_desc_s1.show_michelogram(ax, show_merge_lines=True)
    plt.close(fig)

    # span=3: exercises merge-line branch (line 767 condition True)
    lor_desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=3)
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
        scanner, ppl.Michelogram(scanner.num_rings, 1, span=1)
    )
    fig = lor_desc_s1.show_segment_lors()
    plt.close(fig)

    # span=3, mrd=2: segments -1, 0, +1 → n_rows=2, n_cols=2 (lines 863-1013)
    # covers: Michelogram inset (row=1, col=0), legend (row=0, col=0),
    # compressed kwargs, uncompressed kwargs, tight_layout
    lor_desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=3)
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

    # NOTE: the descriptor's show_segment_lors is now a thin delegator to
    # Michelogram.show_segment_lors.  The asymmetric-segment branch (the
    # ax.axis("off") code path when a negative segment is absent for a given
    # |seg|) is exercised in test_michelogram_show, where the Michelogram's
    # internal arrays can be monkey-patched directly.


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


def test_sinogram_axial_compression_operator(xp: ModuleType, dev: str) -> None:
    """Tests for :class:`SinogramAxialCompressionOperator`.

    Covers:

    * constructor validation,
    * shape / companion-descriptor consistency,
    * adjointness via the inherited random-array helper,
    * closed-form operator norm :math:`\\|G\\|_2 = \\sqrt{\\max_n m_n}`,
    * closed-form roundtrip :math:`G^T G \\mathbf{1} = m[\\tau(p_1)]`,
    * a hand-computed mini-scanner verification of the forward and adjoint,
    * ``target_span=1`` identity behaviour,
    * all :class:`SinogramSpatialAxisOrder` permutations,
    * TOF pass-through.
    """
    import numpy as _np

    # 3-ring scanner with tiny radial/view counts so things are tractable.
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=4,
        num_lor_endpoints_per_side=2,
        lor_spacing=4.0,
        ring_positions=xp.asarray([0.0, 1.0, 2.0], device=dev),
        symmetry_axis=2,
    )

    lor_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, 2, span=1),
        radial_trim=2,
        sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
    )

    # ------------------------------------------------------------------
    # Constructor validation
    # ------------------------------------------------------------------
    with pytest.raises(TypeError):
        ppl.SinogramAxialCompressionOperator("not a descriptor", 3)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="target_span"):
        ppl.SinogramAxialCompressionOperator(lor_s1, 2)  # even

    with pytest.raises(ValueError, match="target_span"):
        ppl.SinogramAxialCompressionOperator(lor_s1, 0)

    with pytest.raises(ValueError, match="mode"):
        ppl.SinogramAxialCompressionOperator(lor_s1, 3, mode="bogus")

    with pytest.raises(ValueError, match="num_tof_bins"):
        ppl.SinogramAxialCompressionOperator(lor_s1, 3, num_tof_bins=0)

    lor_s3_input = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=3)
    )
    with pytest.raises(ValueError, match="span=1"):
        ppl.SinogramAxialCompressionOperator(lor_s3_input, 3)

    # ------------------------------------------------------------------
    # Build operator and check shapes / companion descriptor
    # ------------------------------------------------------------------
    op = ppl.SinogramAxialCompressionOperator(lor_s1, target_span=3)

    assert op.target_span == 3
    assert op.mode == "sum"  # default
    assert op.num_tof_bins is None
    assert op.lor_descriptor is lor_s1
    assert op.in_shape == lor_s1.spatial_sinogram_shape
    assert op.out_shape == op.out_lor_descriptor.spatial_sinogram_shape
    assert op.num_planes_in == lor_s1.num_planes
    assert op.num_planes_out == op.out_lor_descriptor.num_planes
    assert "target_span=3" in str(op)
    assert "mode='sum'" in str(op)

    op_mult_np = _np.asarray(to_numpy_array(op.plane_multiplicity))
    desc_mult_np = _np.asarray(
        to_numpy_array(op.out_lor_descriptor.plane_multiplicity)
    )
    assert _np.array_equal(op_mult_np, desc_mult_np)

    # ------------------------------------------------------------------
    # adjointness (random arrays via the LinearOperator base-class helper)
    # ------------------------------------------------------------------
    assert op.adjointness_test(xp, dev)

    # ------------------------------------------------------------------
    # Closed-form operator norm
    # ------------------------------------------------------------------
    expected_norm = float(_np.sqrt(int(op_mult_np.max())))
    assert abs(op.norm(xp, dev) - expected_norm) < 1e-6
    assert op.max_plane_multiplicity == int(op_mult_np.max())

    # ------------------------------------------------------------------
    # Closed-form roundtrip:  G^T G 1  =  m[target(p1)]
    # ------------------------------------------------------------------
    x_ones = xp.ones(op.in_shape, dtype=xp.float32, device=dev)
    rt = op.adjoint(op(x_ones))

    target = op.target_plane_for_input_plane
    expected_plane_vals = xp.astype(
        xp.take(op.plane_multiplicity, target, axis=0), xp.float32
    )

    P_axis = op.lor_descriptor.plane_axis_num
    rt_np = _np.asarray(to_numpy_array(rt))
    epv_np = _np.asarray(to_numpy_array(expected_plane_vals))
    rt_moved = _np.moveaxis(rt_np, P_axis, -1)
    assert _np.all(rt_moved == epv_np[None, None, :])

    # ------------------------------------------------------------------
    # Hand-computed mini-scanner verification (3 rings, mrd=2, span=1->3)
    # ------------------------------------------------------------------
    # Span-1 plane order (rd = 0, +1, -1, +2, -2) -> p1 = 0..8:
    #
    #     p1   (s, e)   rd   seg(S=3)   mid = s + e
    #      0   (0,0)    0       0           0
    #      1   (1,1)    0       0           2
    #      2   (2,2)    0       0           4
    #      3   (0,1)   +1       0           1
    #      4   (1,2)   +1       0           3
    #      5   (1,0)   -1       0           1
    #      6   (2,1)   -1       0           3
    #      7   (0,2)   +2      +1           2
    #      8   (2,0)   -2      -1           2
    #
    # Output planes (sorted by (|seg|, -seg, mid)):
    #
    #      n = 0   (seg  0, mid 0)  <- p1 = 0
    #      n = 1   (seg  0, mid 1)  <- p1 = 3, 5
    #      n = 2   (seg  0, mid 2)  <- p1 = 1
    #      n = 3   (seg  0, mid 3)  <- p1 = 4, 6
    #      n = 4   (seg  0, mid 4)  <- p1 = 2
    #      n = 5   (seg +1, mid 2)  <- p1 = 7
    #      n = 6   (seg -1, mid 2)  <- p1 = 8
    int_dtype = op.target_plane_for_input_plane.dtype
    expected_target = xp.asarray(
        [0, 2, 4, 1, 3, 1, 3, 5, 6], device=dev, dtype=int_dtype
    )
    assert bool(xp.all(op.target_plane_for_input_plane == expected_target))

    mult_dtype = op.plane_multiplicity.dtype
    expected_mult = xp.asarray(
        [1, 2, 1, 2, 1, 1, 1], device=dev, dtype=mult_dtype
    )
    assert bool(xp.all(op.plane_multiplicity == expected_mult))

    # Forward on a deterministic input  x[r, v, p1] = p1.
    _, _, P = op.in_shape
    plane_vals_in = xp.astype(xp.arange(P, device=dev), xp.float32)
    x_in = xp.zeros(op.in_shape, dtype=xp.float32, device=dev) + xp.reshape(
        plane_vals_in, (1, 1, P)
    )

    y_out = op(x_in)

    # Expected per-output-plane sum (constant in r, v):
    #   y[n=0] = 0;  y[n=1] = 3+5 = 8;  y[n=2] = 1;  y[n=3] = 4+6 = 10;
    #   y[n=4] = 2;  y[n=5] = 7;        y[n=6] = 8.
    expected_plane_sums = xp.asarray(
        [0.0, 8.0, 1.0, 10.0, 2.0, 7.0, 8.0], device=dev, dtype=xp.float32
    )
    expected_y = xp.zeros(op.out_shape, dtype=xp.float32, device=dev) + xp.reshape(
        expected_plane_sums, (1, 1, op.num_planes_out)
    )
    assert bool(xp.all(y_out == expected_y))

    # Adjoint on  y[r, v, n] = 10 + n  gives  x[r, v, p1] = 10 + target(p1).
    plane_vals_out = 10.0 + xp.astype(
        xp.arange(op.num_planes_out, device=dev), xp.float32
    )
    y_in = xp.zeros(op.out_shape, dtype=xp.float32, device=dev) + xp.reshape(
        plane_vals_out, (1, 1, op.num_planes_out)
    )
    x_adj = op.adjoint(y_in)
    expected_adj = xp.asarray(
        [10.0, 12.0, 14.0, 11.0, 13.0, 11.0, 13.0, 15.0, 16.0],
        device=dev,
        dtype=xp.float32,
    )
    expected_x_adj = xp.zeros(op.in_shape, dtype=xp.float32, device=dev) + xp.reshape(
        expected_adj, (1, 1, op.num_planes_in)
    )
    assert bool(xp.all(x_adj == expected_x_adj))

    # ------------------------------------------------------------------
    # target_span = 1 is identity (each input plane in its own output)
    # ------------------------------------------------------------------
    op_id = ppl.SinogramAxialCompressionOperator(lor_s1, target_span=1)
    assert op_id.in_shape == op_id.out_shape
    assert op_id.num_planes_in == op_id.num_planes_out
    assert op_id.max_plane_multiplicity == 1

    expected_id_target = xp.astype(
        xp.arange(op_id.num_planes_in, device=dev),
        op_id.target_plane_for_input_plane.dtype,
    )
    assert bool(xp.all(op_id.target_plane_for_input_plane == expected_id_target))

    rng = _np.random.RandomState(0)
    x_rand = xp.asarray(
        rng.rand(*op_id.in_shape).astype(_np.float32), device=dev
    )
    assert bool(xp.all(op_id(x_rand) == x_rand))
    assert bool(xp.all(op_id.adjoint(x_rand) == x_rand))

    # ------------------------------------------------------------------
    # All sinogram_orders work end-to-end (in/out plane axes line up,
    # adjointness still holds)
    # ------------------------------------------------------------------
    for order in ppl.SinogramSpatialAxisOrder:
        lor = ppl.RegularPolygonPETLORDescriptor(
            scanner,
            ppl.Michelogram(scanner.num_rings, 2, span=1),
            sinogram_order=order,
        )
        op_o = ppl.SinogramAxialCompressionOperator(lor, target_span=3)
        p_ax = order.name.find("P")
        r_ax = order.name.find("R")
        v_ax = order.name.find("V")
        assert op_o.in_shape[p_ax] == lor.num_planes
        assert op_o.out_shape[p_ax] == op_o.num_planes_out
        assert op_o.in_shape[r_ax] == op_o.out_shape[r_ax]
        assert op_o.in_shape[v_ax] == op_o.out_shape[v_ax]
        assert op_o.adjointness_test(xp, dev)

    # ------------------------------------------------------------------
    # TOF pass-through: trailing axis carried unchanged, and the
    # per-TOF-bin output equals the non-TOF operator applied to
    # that TOF slice.
    # ------------------------------------------------------------------
    num_tof = 5
    op_tof = ppl.SinogramAxialCompressionOperator(
        lor_s1, target_span=3, num_tof_bins=num_tof
    )
    assert op_tof.num_tof_bins == num_tof
    assert op_tof.in_shape == lor_s1.spatial_sinogram_shape + (num_tof,)
    assert (
        op_tof.out_shape
        == op_tof.out_lor_descriptor.spatial_sinogram_shape + (num_tof,)
    )
    assert op_tof.adjointness_test(xp, dev)

    rng2 = _np.random.RandomState(42)
    x_tof = xp.asarray(
        rng2.rand(*op_tof.in_shape).astype(_np.float32), device=dev
    )
    y_tof = op_tof(x_tof)
    for t in range(num_tof):
        # op_tof(x)[..., t]  ==  op(x[..., t])
        assert bool(xp.all(y_tof[..., t] == op(x_tof[..., t])))


def test_sinogram_axial_compression_operator_average_mode(
    xp: ModuleType, dev: str
) -> None:
    """``mode="average"`` produces the per-multiplicity mean and a matching adjoint.

    For the 3-ring span=1 -> span=3 layout used in the main test, the
    multiplicities are ``[1, 2, 1, 2, 1, 1, 1]``.  With the synthetic input
    ``x[r, v, p1] = p1``, the sum-mode output per plane is
    ``[0, 8, 1, 10, 2, 7, 8]``; dividing by the multiplicity gives the
    average-mode output ``[0, 4, 1, 5, 2, 7, 8]``.

    Likewise, on a deterministic ``y[r, v, n] = 10 + n``, the sum-mode
    adjoint replicates ``10 + tau(p1)``; the average-mode adjoint divides
    by ``m_{tau(p1)}``.
    """
    import numpy as _np

    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=4,
        num_lor_endpoints_per_side=2,
        lor_spacing=4.0,
        ring_positions=xp.asarray([0.0, 1.0, 2.0], device=dev),
        symmetry_axis=2,
    )
    lor_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, 2, span=1),
        radial_trim=2,
    )

    op_sum = ppl.SinogramAxialCompressionOperator(
        lor_s1, target_span=3, mode="sum"
    )
    op_avg = ppl.SinogramAxialCompressionOperator(
        lor_s1, target_span=3, mode="average"
    )

    assert op_sum.mode == "sum"
    assert op_avg.mode == "average"
    assert "mode='average'" in str(op_avg)

    # The two operators share the same plane structure
    assert op_avg.num_planes_in == op_sum.num_planes_in
    assert op_avg.num_planes_out == op_sum.num_planes_out
    assert op_avg.in_shape == op_sum.in_shape
    assert op_avg.out_shape == op_sum.out_shape
    assert bool(xp.all(op_avg.plane_multiplicity == op_sum.plane_multiplicity))
    assert bool(
        xp.all(op_avg.target_plane_for_input_plane == op_sum.target_plane_for_input_plane)
    )

    # ------------------------------------------------------------------
    # Closed-form norms
    # ------------------------------------------------------------------
    mult_np = _np.asarray(to_numpy_array(op_avg.plane_multiplicity))
    expected_sum_norm = float(_np.sqrt(int(mult_np.max())))
    expected_avg_norm = float(1.0 / _np.sqrt(int(mult_np.min())))
    assert abs(op_sum.norm(xp, dev) - expected_sum_norm) < 1e-6
    assert abs(op_avg.norm(xp, dev) - expected_avg_norm) < 1e-6

    # ------------------------------------------------------------------
    # Adjointness for average mode
    # ------------------------------------------------------------------
    assert op_avg.adjointness_test(xp, dev)

    # ------------------------------------------------------------------
    # Forward in average mode = sum-mode result / multiplicity
    # ------------------------------------------------------------------
    P_in = op_avg.num_planes_in
    P_out = op_avg.num_planes_out

    plane_vals = xp.astype(xp.arange(P_in, device=dev), xp.float32)
    x_in = xp.zeros(op_avg.in_shape, dtype=xp.float32, device=dev) + xp.reshape(
        plane_vals, (1, 1, P_in)
    )

    y_avg = op_avg(x_in)

    # Expected per-plane: sum mode [0, 8, 1, 10, 2, 7, 8] divided by
    # multiplicity [1, 2, 1, 2, 1, 1, 1] -> [0, 4, 1, 5, 2, 7, 8].
    expected_avg = xp.asarray(
        [0.0, 4.0, 1.0, 5.0, 2.0, 7.0, 8.0], device=dev, dtype=xp.float32
    )
    expected_y_avg = xp.zeros(op_avg.out_shape, dtype=xp.float32, device=dev) + xp.reshape(
        expected_avg, (1, 1, P_out)
    )
    assert bool(xp.all(y_avg == expected_y_avg))

    # ------------------------------------------------------------------
    # Adjoint in average mode = sum-mode adjoint / m_{tau(p1)}
    # ------------------------------------------------------------------
    plane_vals_out = 10.0 + xp.astype(
        xp.arange(P_out, device=dev), xp.float32
    )
    y_in = xp.zeros(op_avg.out_shape, dtype=xp.float32, device=dev) + xp.reshape(
        plane_vals_out, (1, 1, P_out)
    )
    x_adj = op_avg.adjoint(y_in)

    # Sum-mode adjoint values along the plane axis would be
    # [10, 12, 14, 11, 13, 11, 13, 15, 16] (see the main test). Divided by
    # multiplicity at the target: m[target_for_p1] = [1, 1, 1, 2, 2, 2, 2, 1, 1]
    # -> [10, 12, 14, 5.5, 6.5, 5.5, 6.5, 15, 16].
    expected_adj = xp.asarray(
        [10.0, 12.0, 14.0, 5.5, 6.5, 5.5, 6.5, 15.0, 16.0],
        device=dev,
        dtype=xp.float32,
    )
    expected_x_adj = xp.zeros(op_avg.in_shape, dtype=xp.float32, device=dev) + xp.reshape(
        expected_adj, (1, 1, P_in)
    )
    assert bool(xp.all(x_adj == expected_x_adj))

    # ------------------------------------------------------------------
    # Round-trip identity check:  G_avg^T G_avg 1 = 1 / m_{tau(p1)} * m_{tau(p1)}
    #
    # Because G_avg = D^{-1} G_sum, we have
    #     G_avg^T G_avg = G_sum^T D^{-2} G_sum
    # and  (G_avg^T G_avg 1)_{p1} = m_{tau(p1)} / m_{tau(p1)}^2 = 1 / m_{tau(p1)}.
    # ------------------------------------------------------------------
    x_ones = xp.ones(op_avg.in_shape, dtype=xp.float32, device=dev)
    rt = op_avg.adjoint(op_avg(x_ones))
    target = op_avg.target_plane_for_input_plane
    inv_mult_at_target = 1.0 / xp.astype(
        xp.take(op_avg.plane_multiplicity, target, axis=0), xp.float32
    )
    rt_np = _np.asarray(to_numpy_array(rt))
    expected_np = _np.asarray(to_numpy_array(inv_mult_at_target))
    P_axis = op_avg.lor_descriptor.plane_axis_num
    moved = _np.moveaxis(rt_np, P_axis, -1)
    assert _np.all(moved == expected_np[None, None, :])


def test_michelogram_basic(xp: ModuleType, dev: str) -> None:
    """Construction, validation, and the basic per-plane metadata for span=1.

    Compares Michelogram outputs directly against the existing
    :class:`RegularPolygonPETLORDescriptor` for the same parameters, so this
    test pins Michelogram to the current descriptor behaviour before the
    descriptor is rewritten to consume it.
    """
    import numpy as _np

    # ------------------------------------------------------------------
    # constructor validation
    # ------------------------------------------------------------------
    with pytest.raises(ValueError, match="num_rings"):
        ppl.Michelogram(num_rings=0, max_ring_difference=0)
    with pytest.raises(ValueError, match="num_rings"):
        ppl.Michelogram(num_rings=-1, max_ring_difference=0)
    with pytest.raises(ValueError, match="max_ring_difference"):
        ppl.Michelogram(num_rings=3, max_ring_difference=-1)
    with pytest.raises(ValueError, match="span"):
        ppl.Michelogram(num_rings=3, max_ring_difference=2, span=0)
    with pytest.raises(ValueError, match="span"):
        ppl.Michelogram(num_rings=3, max_ring_difference=2, span=2)
    with pytest.raises(ValueError, match="span"):
        ppl.Michelogram(num_rings=3, max_ring_difference=2, span=-3)

    # ------------------------------------------------------------------
    # span=1: compare to RegularPolygonPETLORDescriptor outputs
    # ------------------------------------------------------------------
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=4,
        num_lor_endpoints_per_side=2,
        lor_spacing=4.0,
        ring_positions=xp.asarray([0.0, 1.0, 2.0], device=dev),
        symmetry_axis=2,
    )
    lor_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, 2, span=1),
        radial_trim=2,
    )
    m1 = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=1)

    assert m1.num_rings == 3
    assert m1.max_ring_difference == 2
    assert m1.span == 1
    assert m1.num_planes == lor_s1.num_planes
    assert m1.max_multiplicity == 1
    assert "span=1" in repr(m1)

    desc_seg = _np.asarray(to_numpy_array(lor_s1.plane_segment), dtype=_np.int32)
    desc_mult = _np.asarray(to_numpy_array(lor_s1.plane_multiplicity), dtype=_np.int32)
    assert _np.array_equal(m1.plane_segment, desc_seg)
    assert _np.array_equal(m1.plane_multiplicity, desc_mult)

    # span=1 axial midpoint is just (start + end) per plane, taken directly
    # from the descriptor's start_plane_index / end_plane_index.
    desc_start = _np.asarray(to_numpy_array(lor_s1.start_plane_index), dtype=_np.int64)
    desc_end = _np.asarray(to_numpy_array(lor_s1.end_plane_index), dtype=_np.int64)
    expected_mid = (desc_start + desc_end).astype(_np.int32)
    assert _np.array_equal(m1.plane_axial_midpoint_int, expected_mid)

    # For span=1 the padded layout collapses to a single column; the
    # contributing start/end rings must match the descriptor's plane indices.
    assert _np.array_equal(m1.plane_start_rings[:, 0], desc_start.astype(_np.int32))
    assert _np.array_equal(m1.plane_end_rings[:, 0], desc_end.astype(_np.int32))
    assert _np.all(m1.plane_mask == 1.0)


def test_michelogram_spanned(xp: ModuleType, dev: str) -> None:
    """Spanned layouts match the descriptor: 3-ring span=3 and 13-ring span=9."""
    import numpy as _np

    # ------------------------------------------------------------------
    # 3-ring scanner, span=3 — also pinned by the existing
    # test_regular_polygon_lor_desc_span hand-computed expectations.
    # ------------------------------------------------------------------
    ring_z_3 = xp.asarray([0.0, 1.0, 3.0], device=dev)
    scanner_3 = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=1,
        lor_spacing=2.3,
        ring_positions=ring_z_3,
        symmetry_axis=2,
    )
    lor_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner_3, ppl.Michelogram(scanner_3.num_rings, 2, span=3)
    )
    m3 = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=3)

    assert m3.num_planes == lor_s3.num_planes == 7
    assert m3.max_multiplicity == 2

    desc_seg = _np.asarray(to_numpy_array(lor_s3.plane_segment), dtype=_np.int32)
    desc_mult = _np.asarray(to_numpy_array(lor_s3.plane_multiplicity), dtype=_np.int32)
    assert _np.array_equal(m3.plane_segment, desc_seg)
    assert _np.array_equal(m3.plane_multiplicity, desc_mult)

    # Hand-derived per-plane data (matches the test_regular_polygon_lor_desc_span
    # comment block):
    #   n=0 (seg 0,  mid 0): (0,0)
    #   n=1 (seg 0,  mid 1): (0,1),(1,0)
    #   n=2 (seg 0,  mid 2): (1,1)
    #   n=3 (seg 0,  mid 3): (1,2),(2,1)
    #   n=4 (seg 0,  mid 4): (2,2)
    #   n=5 (seg +1, mid 2): (0,2)
    #   n=6 (seg -1, mid 2): (2,0)
    assert _np.array_equal(
        m3.plane_axial_midpoint_int,
        _np.asarray([0, 1, 2, 3, 4, 2, 2], dtype=_np.int32),
    )
    assert _np.array_equal(
        m3.plane_segment,
        _np.asarray([0, 0, 0, 0, 0, 1, -1], dtype=_np.int32),
    )

    # Padded contributing ring pairs
    expected_start = _np.asarray(
        [[0, 0], [0, 1], [1, 0], [1, 2], [2, 0], [0, 0], [2, 0]], dtype=_np.int32
    )
    expected_end = _np.asarray(
        [[0, 0], [1, 0], [1, 0], [2, 1], [2, 0], [2, 0], [0, 0]], dtype=_np.int32
    )
    expected_mask = _np.asarray(
        [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0], [1, 0], [1, 0]], dtype=_np.float32
    )
    assert _np.array_equal(m3.plane_start_rings, expected_start)
    assert _np.array_equal(m3.plane_end_rings, expected_end)
    assert _np.array_equal(m3.plane_mask, expected_mask)

    # average_z_per_plane against the descriptor's spanned setup
    desc_start_z = _np.asarray(to_numpy_array(lor_s3.start_plane_z), dtype=_np.float32)
    desc_end_z = _np.asarray(to_numpy_array(lor_s3.end_plane_z), dtype=_np.float32)
    m_start_z, m_end_z = m3.average_z_per_plane(_np.asarray([0.0, 1.0, 3.0]))
    assert _np.allclose(m_start_z, desc_start_z, atol=1e-6)
    assert _np.allclose(m_end_z, desc_end_z, atol=1e-6)

    # ------------------------------------------------------------------
    # 13-ring scanner, span=9 — the bigger scenario hand-verified in
    # test_regular_polygon_lor_desc_span; spot-checks here.
    # ------------------------------------------------------------------
    ring_z_13 = xp.asarray([float(i) for i in range(13)], device=dev)
    scanner_13 = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=12,
        num_lor_endpoints_per_side=1,
        lor_spacing=2.3,
        ring_positions=ring_z_13,
        symmetry_axis=2,
    )
    lor_13 = ppl.RegularPolygonPETLORDescriptor(
        scanner_13, ppl.Michelogram(scanner_13.num_rings, 11, span=9)
    )
    m13 = ppl.Michelogram(num_rings=13, max_ring_difference=11, span=9)

    assert m13.num_planes == lor_13.num_planes == 55
    assert int(m13.plane_multiplicity.sum()) == 167

    desc_seg_13 = _np.asarray(to_numpy_array(lor_13.plane_segment), dtype=_np.int32)
    desc_mult_13 = _np.asarray(
        to_numpy_array(lor_13.plane_multiplicity), dtype=_np.int32
    )
    assert _np.array_equal(m13.plane_segment, desc_seg_13)
    assert _np.array_equal(m13.plane_multiplicity, desc_mult_13)

    # Same hand-computed plane checks as in the existing descriptor test
    desc_start_z_13 = _np.asarray(
        to_numpy_array(lor_13.start_plane_z), dtype=_np.float32
    )
    desc_end_z_13 = _np.asarray(
        to_numpy_array(lor_13.end_plane_z), dtype=_np.float32
    )
    m_start_z_13, m_end_z_13 = m13.average_z_per_plane(_np.arange(13, dtype=_np.float64))
    assert _np.allclose(m_start_z_13, desc_start_z_13, atol=1e-5)
    assert _np.allclose(m_end_z_13, desc_end_z_13, atol=1e-5)


def test_michelogram_ring_diff_to_segment(xp: ModuleType, dev: str) -> None:
    """The segment formula matches the descriptor's _ring_diff_to_segment."""
    scanner = pps.DemoPETScannerGeometry(xp, dev, 3, symmetry_axis=2)
    desc_s3 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=3)
    )
    desc_s9 = ppl.RegularPolygonPETLORDescriptor(
        scanner, ppl.Michelogram(scanner.num_rings, 2, span=9)
    )

    m3 = ppl.Michelogram(num_rings=5, max_ring_difference=4, span=3)
    m9 = ppl.Michelogram(num_rings=5, max_ring_difference=4, span=9)
    m1 = ppl.Michelogram(num_rings=5, max_ring_difference=4, span=1)

    for rd in range(-15, 16):
        assert m3.ring_diff_to_segment(rd) == desc_s3.michelogram.ring_diff_to_segment(rd)
        assert m9.ring_diff_to_segment(rd) == desc_s9.michelogram.ring_diff_to_segment(rd)

    # Spot checks against the documented formula
    # span=1: every rd is its own segment
    for rd in range(-4, 5):
        assert m1.ring_diff_to_segment(rd) == rd
    # span=3, half_span=1: rd in {-1,0,1} -> seg 0; rd=±2,±3,±4 -> seg ±1
    assert m3.ring_diff_to_segment(0) == 0
    assert m3.ring_diff_to_segment(1) == 0
    assert m3.ring_diff_to_segment(-1) == 0
    assert m3.ring_diff_to_segment(2) == 1
    assert m3.ring_diff_to_segment(-2) == -1
    assert m3.ring_diff_to_segment(4) == 1
    assert m3.ring_diff_to_segment(-4) == -1
    # span=9, half_span=4: rd in [-4,+4] -> seg 0; rd=±5..±13 -> seg ±1
    assert m9.ring_diff_to_segment(4) == 0
    assert m9.ring_diff_to_segment(-4) == 0
    assert m9.ring_diff_to_segment(5) == 1
    assert m9.ring_diff_to_segment(-5) == -1
    assert m9.ring_diff_to_segment(13) == 1
    assert m9.ring_diff_to_segment(-13) == -1


def test_michelogram_ge_layout(xp: ModuleType, dev: str) -> None:
    """GE (Discovery MI / Signa) layout: mixed segmentation, span ignored."""
    from collections import defaultdict

    # GE Discovery MI-3: 27 rings, max ring difference 26 -> 703 planes
    m = ppl.Michelogram.ge_signa(num_rings=27, max_ring_difference=26)
    assert m.layout is ppl.MichelogramLayout.GE
    assert m.span is None
    assert int(m.num_planes) == 703
    assert int(m.max_multiplicity) == 2

    # per-segment plane counts in GE order 0, +1, -1, +2, -2, ...
    per = defaultdict(int)
    for seg in m.plane_segment.tolist():
        per[seg] += 1
    assert [per[s] for s in (0, 1, -1, 2, -2)] == [53, 49, 49, 45, 45]
    assert per[13] == 1 and per[-13] == 1

    # cross planes (multiplicity 2) are exactly segment 0 at odd axial midpoint
    for pi in range(int(m.num_planes)):
        if int(m.plane_multiplicity[pi]) == 2:
            assert int(m.plane_segment[pi]) == 0
            assert int(m.plane_axial_midpoint_int[pi]) % 2 == 1
    # full bijection over all ring pairs (max ring diff = num_rings - 1)
    assert int((m.plane_for_ring_pair_table >= 0).sum()) == 27 * 27

    # explicit-enum constructor is equivalent; span is ignored (with a warning)
    m_enum = ppl.Michelogram(27, 26, layout=ppl.MichelogramLayout.GE)
    assert int(m_enum.num_planes) == 703 and m_enum.span is None
    with pytest.warns(UserWarning, match="ignored"):
        ppl.Michelogram(27, 26, span=3, layout=ppl.MichelogramLayout.GE)

    # GE ring-difference -> segment rule: 0 for |rd|<=1, else sign*(|rd|//2)
    assert m.ring_diff_to_segment(0) == 0
    assert m.ring_diff_to_segment(1) == 0 and m.ring_diff_to_segment(-1) == 0
    assert m.ring_diff_to_segment(2) == 1 and m.ring_diff_to_segment(3) == 1
    assert m.ring_diff_to_segment(-2) == -1 and m.ring_diff_to_segment(-3) == -1
    assert m.ring_diff_to_segment(4) == 2 and m.ring_diff_to_segment(26) == 13

    # axial compression is not supported for the GE layout
    with pytest.raises(ValueError, match="GE layout"):
        m.compression_index_maps_to(ppl.Michelogram(27, 26, span=3))

    # a descriptor built on a GE michelogram works; span is None and the
    # per-plane single-ring-pair indices are undefined (planes can hold 2 pairs)
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings=6, symmetry_axis=2)
    ge = ppl.Michelogram.ge_signa(scanner.num_rings, 5)
    desc = ppl.RegularPolygonPETLORDescriptor(scanner, ge)
    assert desc.span is None
    assert int(desc.num_planes) == int(ge.num_planes) == 31
    with pytest.raises(AttributeError):
        _ = desc.start_plane_index


def test_michelogram_plane_for_ring_pair(xp: ModuleType, dev: str) -> None:
    """plane_for_ring_pair lookup matches the (start, end) -> plane mapping
    derived from the padded plane_start_rings / plane_end_rings arrays."""
    import numpy as _np

    m = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=3)

    # Forward consistency: for every (pi, k) with mask=1, the inverse must agree.
    for pi in range(m.num_planes):
        for k in range(int(m.plane_multiplicity[pi])):
            s = int(m.plane_start_rings[pi, k])
            e = int(m.plane_end_rings[pi, k])
            assert m.plane_for_ring_pair(s, e) == pi
            assert int(m.plane_for_ring_pair_table[s, e]) == pi

    # Invalid lookups
    with pytest.raises(IndexError):
        m.plane_for_ring_pair(-1, 0)
    with pytest.raises(IndexError):
        m.plane_for_ring_pair(0, 3)

    # |rd| > max_ring_difference  (max_rd=2, num_rings=3 means all rd fit; use
    # a smaller max_rd to trigger this case)
    m_small = ppl.Michelogram(num_rings=3, max_ring_difference=0, span=1)
    with pytest.raises(ValueError, match="max_ring_difference"):
        m_small.plane_for_ring_pair(0, 2)


def test_michelogram_compression_index_maps(xp: ModuleType, dev: str) -> None:
    """compression_index_maps validity rules + correctness vs the operator."""
    import numpy as _np

    m1 = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=1)

    # ------------------------------------------------------------------
    # validity errors
    # ------------------------------------------------------------------
    with pytest.raises(ValueError, match="target_span"):
        m1.compression_index_maps(0)
    with pytest.raises(ValueError, match="target_span"):
        m1.compression_index_maps(2)  # even

    # span_a=3 -> span_b=5 : 5/3 not integer
    m3 = ppl.Michelogram(num_rings=5, max_ring_difference=4, span=3)
    with pytest.raises(ValueError, match="multiple"):
        m3.compression_index_maps(5)
    # span_a=3 -> span_b=1 : smaller than self
    with pytest.raises(ValueError, match=">= self.span"):
        m3.compression_index_maps(1)

    # ------------------------------------------------------------------
    # compression_index_maps_to (direct entry point) validation branches.
    # These are only reachable when a target Michelogram is passed
    # directly, since the compression_index_maps(target_span) wrapper
    # always builds a target with matching num_rings and
    # max_ring_difference.
    # ------------------------------------------------------------------
    with pytest.raises(TypeError, match="Michelogram"):
        m1.compression_index_maps_to("not a michelogram")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="num_rings"):
        m1.compression_index_maps_to(
            ppl.Michelogram(num_rings=m1.num_rings + 1, max_ring_difference=2, span=3)
        )

    # target.max_ring_difference smaller than self's (would leave some
    # input ring pairs without a target plane).
    m1_with_mrd = ppl.Michelogram(num_rings=5, max_ring_difference=4, span=1)
    with pytest.raises(ValueError, match="max_ring_difference"):
        m1_with_mrd.compression_index_maps_to(
            ppl.Michelogram(num_rings=5, max_ring_difference=2, span=3)
        )

    # ------------------------------------------------------------------
    # span 1 -> 3 on the 3-ring scanner: must equal the operator's
    # target_plane_for_input_plane / multiplicity exactly.
    # ------------------------------------------------------------------
    scanner = pps.RegularPolygonPETScannerGeometry(
        xp,
        dev,
        radius=65.0,
        num_sides=4,
        num_lor_endpoints_per_side=2,
        lor_spacing=4.0,
        ring_positions=xp.asarray([0.0, 1.0, 2.0], device=dev),
        symmetry_axis=2,
    )
    lor_s1 = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, 2, span=1),
        radial_trim=2,
    )
    op = ppl.SinogramAxialCompressionOperator(lor_s1, target_span=3)

    tgt, idx2d, mask, mult = m1.compression_index_maps(3)

    op_target = _np.asarray(
        to_numpy_array(op.target_plane_for_input_plane), dtype=_np.int64
    )
    op_mult = _np.asarray(
        to_numpy_array(op.plane_multiplicity), dtype=_np.int32
    )
    assert _np.array_equal(tgt, op_target)
    assert _np.array_equal(mult, op_mult)
    # The padded inverse map must match what the operator builds internally.
    # The operator stores the flattened version on self._idx2d_flat; reshape
    # and compare.
    op_idx_flat = _np.asarray(to_numpy_array(op._idx2d_flat), dtype=_np.int64)
    op_max_mult = int(op._max_mult)
    op_idx2d = op_idx_flat.reshape(op.num_planes_out, op_max_mult)
    op_mask2d = _np.asarray(to_numpy_array(op._mask2d), dtype=_np.float32)
    assert _np.array_equal(idx2d, op_idx2d)
    assert _np.array_equal(mask, op_mask2d)

    # ------------------------------------------------------------------
    # Self-mapping: target_span == self.span must be an identity-style map
    # (each input plane is its own output, multiplicity all 1).
    # ------------------------------------------------------------------
    tgt_id, _, _, mult_id = m3.compression_index_maps(3)
    assert _np.array_equal(tgt_id, _np.arange(m3.num_planes, dtype=_np.int64))
    assert _np.all(mult_id == 1)

    # ------------------------------------------------------------------
    # span 3 -> 9 on a 13-ring scanner: every input plane must map to
    # exactly one output plane, and the union of contributions covers all
    # input planes (a partition).
    # ------------------------------------------------------------------
    m3_13 = ppl.Michelogram(num_rings=13, max_ring_difference=11, span=3)
    m9_13 = ppl.Michelogram(num_rings=13, max_ring_difference=11, span=9)
    tgt_39, idx2d_39, mask_39, mult_39 = m3_13.compression_index_maps(9)
    assert tgt_39.shape == (m3_13.num_planes,)
    assert idx2d_39.shape[0] == m9_13.num_planes
    assert int(mult_39.sum()) == m3_13.num_planes
    # Each input plane appears exactly once in idx2d_39 (filtered by mask)
    flat_used = idx2d_39[mask_39 > 0].astype(_np.int64)
    flat_sorted = _np.sort(flat_used)
    assert _np.array_equal(flat_sorted, _np.arange(m3_13.num_planes))


def test_michelogram_show(xp: ModuleType, dev: str) -> None:
    """Smoke tests for show() and show_segment_lors()."""
    import numpy as _np

    # span=1: show only (no merge lines)
    m1 = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=1)
    fig, ax = plt.subplots()
    m1.show(ax, show_merge_lines=False)
    plt.close(fig)
    fig, ax = plt.subplots()
    m1.show(ax)  # default show_merge_lines=True (no-op for span=1)
    plt.close(fig)

    # span=3 on 3 rings: show + show_segment_lors (auto-create axes)
    m3 = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=3)
    fig, ax = plt.subplots()
    m3.show(ax)
    plt.close(fig)
    fig = m3.show_segment_lors(_np.asarray([0.0, 1.0, 3.0]))
    plt.close(fig)

    # show_segment_lors with custom kwargs and pre-existing axes
    fig = m3.show_segment_lors(
        _np.asarray([0.0, 1.0, 3.0]),
        uncompressed_lor_kwargs={"alpha": 0.3},
        compressed_lor_kwargs={"alpha": 0.8, "color": "blue"},
    )
    plt.close(fig)

    fig_pre, axs_pre = plt.subplots(2, 2, squeeze=False)
    m3.show_segment_lors(_np.asarray([0.0, 1.0, 3.0]), axs=axs_pre)
    plt.close(fig_pre)

    # ring_positions shape validation
    with pytest.raises(ValueError, match="ring_positions"):
        m3.show_segment_lors(_np.asarray([0.0, 1.0]))  # wrong length
    with pytest.raises(ValueError, match="ring_positions"):
        m3.show_segment_lors(_np.zeros((3, 2)))  # not 1-D

    # average_z_per_plane shape validation (covered by show_segment_lors above,
    # but also exercise the standalone entry point)
    with pytest.raises(ValueError, match="ring_positions"):
        m3.average_z_per_plane(_np.zeros(5))

    # ------------------------------------------------------------------
    # Asymmetric-segment branch: ax.axis("off") fires when, for a given
    # |seg|, one of the +/- segments is missing in plane_segment.
    #
    # Symmetric scanners always produce symmetric segments, so we
    # construct a normal Michelogram and overwrite its internals to
    # simulate an asymmetric layout: segments {-1, 0, +1, +2}.  Because
    # there's a negative segment, n_rows == 2; because +2 has no
    # matching -2, the (row=1, col=2) cell falls through to ax.axis("off").
    # ------------------------------------------------------------------
    m_patch = ppl.Michelogram(num_rings=3, max_ring_difference=2, span=1)
    m_patch._num_planes = 4
    m_patch._max_multiplicity = 1
    m_patch._plane_segment = _np.asarray([-1, 0, 1, 2], dtype=_np.int32)
    m_patch._plane_multiplicity = _np.asarray([1, 1, 1, 1], dtype=_np.int32)
    m_patch._plane_start_rings = _np.zeros((4, 1), dtype=_np.int32)
    m_patch._plane_end_rings = _np.zeros((4, 1), dtype=_np.int32)
    m_patch._plane_mask = _np.ones((4, 1), dtype=_np.float32)
    m_patch._plane_axial_midpoint_int = _np.zeros(4, dtype=_np.int32)

    fig = m_patch.show_segment_lors(_np.asarray([0.0, 1.0, 2.0]))
    plt.close(fig)


def test_show_tof_bins(xp: ModuleType, dev: str) -> None:
    scanner = pps.DemoPETScannerGeometry(xp, dev, num_rings=1, symmetry_axis=2)
    lor_desc = ppl.RegularPolygonPETLORDescriptor(
        scanner,
        ppl.Michelogram(scanner.num_rings, max_ring_difference=0, span=1),
        radial_trim=0,
        sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
    )
    tof_params = tof.TOFParameters(num_tofbins=5, tofbin_width=40.0, sigma_tof=20.0)

    # views=int, show_colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lor_desc.show_tof_bins(ax, tof_params, views=0, plane=0, show_colorbar=True)
    plt.close(fig)

    # views as array-like (covers the else branch in views normalisation)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lor_desc.show_tof_bins(ax, tof_params, views=xp.asarray([0, 1], device=dev))
    plt.close(fig)

    # show_bin_labels with bins wider than the LOR so some centres fall outside
    # the physical LOR extent (covers the inner `continue` in the labels loop)
    tof_wide = tof.TOFParameters(num_tofbins=3, tofbin_width=400.0, sigma_tof=20.0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    lor_desc.show_tof_bins(
        ax, tof_wide, show_bin_labels=True, show_endpoints=False
    )
    plt.close(fig)


