from __future__ import annotations

import pytest
import parallelproj.pet_scanners as pps
import parallelproj.pet_lors as ppl
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
        scanner_z, max_ring_difference=2, span=3
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
        scanner_13, max_ring_difference=11, span=9
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
    lor_desc_patch._plane_segment = xp.asarray(
        [-1, 0, 1, 2], device=dev, dtype=xp.int32
    )
    lor_desc_patch._start_plane_z = xp.asarray(
        [0.0, 0.0, 0.0, 0.0], device=dev, dtype=xp.float32
    )
    lor_desc_patch._end_plane_z = xp.asarray(
        [0.0, 0.0, 0.0, 0.0], device=dev, dtype=xp.float32
    )
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
        radial_trim=2,
        max_ring_difference=2,
        sinogram_order=ppl.SinogramSpatialAxisOrder.RVP,
        span=1,
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

    with pytest.raises(ValueError, match="num_tof_bins"):
        ppl.SinogramAxialCompressionOperator(lor_s1, 3, num_tof_bins=0)

    lor_s3_input = ppl.RegularPolygonPETLORDescriptor(
        scanner, max_ring_difference=2, span=3
    )
    with pytest.raises(ValueError, match="span=1"):
        ppl.SinogramAxialCompressionOperator(lor_s3_input, 3)

    # ------------------------------------------------------------------
    # Build operator and check shapes / companion descriptor
    # ------------------------------------------------------------------
    op = ppl.SinogramAxialCompressionOperator(lor_s1, target_span=3)

    assert op.target_span == 3
    assert op.num_tof_bins is None
    assert op.lor_descriptor is lor_s1
    assert op.in_shape == lor_s1.spatial_sinogram_shape
    assert op.out_shape == op.out_lor_descriptor.spatial_sinogram_shape
    assert op.num_planes_in == lor_s1.num_planes
    assert op.num_planes_out == op.out_lor_descriptor.num_planes
    assert "target_span=3" in str(op)

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
            scanner, max_ring_difference=2, sinogram_order=order, span=1
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
