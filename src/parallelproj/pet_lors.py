"""description of PET LORs (and sinograms bins) consisting of two detector endpoints"""

from __future__ import annotations

import abc
import enum
import numpy as np
from parallelproj import Array, to_numpy_array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.axes import Axes

from types import ModuleType

from .pet_scanners import (
    ModularizedPETScannerGeometry,
    RegularPolygonPETScannerGeometry,
)


class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)

    Examples
    --------
    .. minigallery:: parallelproj.SinogramSpatialAxisOrder
    """

    RVP = enum.auto()
    """[radial,view,plane]"""
    RPV = enum.auto()
    """[radial,plane,view]"""
    VRP = enum.auto()
    """[view,radial,plane]"""
    VPR = enum.auto()
    """[view,plane,radial]"""
    PRV = enum.auto()
    """[plane,radial,view]"""
    PVR = enum.auto()
    """[plane,view,radial]"""


class PETLORDescriptor(abc.ABC):
    """abstract base class to describe which modules / indices in modules of a
    modularized PET scanner are in coincidence; defining geometrical LORs"""

    def __init__(self, scanner: ModularizedPETScannerGeometry) -> None:
        """
        Parameters
        ----------
        scanner : ModularizedPETScannerGeometry
            a modularized PET scanner
        """
        self._scanner = scanner

    @abc.abstractmethod
    def get_lor_coordinates(self) -> tuple[Array, Array]:
        """return the start and end coordinates of all (or a subset of) LORs"""
        raise NotImplementedError

    @property
    def scanner(self) -> ModularizedPETScannerGeometry:
        """the scanner for which coincidences are described"""
        return self._scanner

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self.scanner.xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self.scanner.dev


class EqualBlockPETLORDescriptor(PETLORDescriptor):
    """LOR descriptor for scanner consisting of block modules where each
    block module has the same number of LOR endpoints"""

    def __init__(
        self, scanner: ModularizedPETScannerGeometry, all_block_pairs: Array
    ) -> None:
        """
        Parameters
        ----------
        scanner : ModularizedPETScannerGeometry
            A modularized PET scanner consisting of block modules
            with the same number of LOR endpoints.
        all_block_pairs : Array
            An array containing pairs of integer numbers encoding
            which block pairs are in coincidence and form valid LORs.

        Returns
        -------
        None
        """
        # check if all modules (blocks) have the same number of LOR enpoints
        lor_endpoints_per_block = [x.num_lor_endpoints for x in scanner.modules]
        if not all(x == lor_endpoints_per_block[0] for x in lor_endpoints_per_block):
            raise ValueError(
                "All modules (blocks) must have the same number of LOR endpoints"
            )

        super().__init__(scanner)
        self._scanner = scanner
        self._all_block_pairs = all_block_pairs
        self._num_lorendpoints_per_block = self.scanner.modules[0].num_lor_endpoints
        self._num_lors_per_block_pair = self._num_lorendpoints_per_block**2

    @property
    def all_block_pairs(self) -> Array:
        """all block pairs in coincidence"""
        return self._all_block_pairs

    @property
    def num_block_pairs(self) -> int:
        """number of block pairs in coincidence"""
        return self._all_block_pairs.shape[0]

    @property
    def num_lorendpoints_per_block(self) -> int:
        """number of LOR endpoints per block"""
        return self._num_lorendpoints_per_block

    @property
    def num_lors_per_block_pair(self) -> int:
        """number of LORs per block pair"""
        return self._num_lors_per_block_pair

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self.scanner.xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self.scanner.dev

    def get_lor_coordinates(
        self, block_pair_nums: None | Array = None
    ) -> tuple[Array, Array]:
        """
        Get the coordinates of LORs for the given block pair numbers.

        Parameters
        ----------
        block_pair_nums : None or Array, optional
            The block pair numbers for which to retrieve the LOR coordinates.
            If None, all block pair numbers will be used.

        Returns
        -------
        tuple[Array, Array]
        A tuple containing two arrays:
            - the start coordinates of the LORs, with shape (N, 3), where N is the total number of LORs.
            - the end coordinates of the LORs, with shape (N, 3)
        """
        if block_pair_nums is None:
            block_pair_nums = self.xp.arange(
                self._all_block_pairs.shape[0], device=self.dev
            )

        assert block_pair_nums is not None

        xstart = self.xp.zeros(
            (block_pair_nums.shape[0], self._num_lors_per_block_pair, 3),
            device=self.dev,
            dtype=self.xp.float32,
        )

        xend = self.xp.zeros(
            (block_pair_nums.shape[0], self._num_lors_per_block_pair, 3),
            device=self.dev,
            dtype=self.xp.float32,
        )

        for i, block_pair_num in enumerate(block_pair_nums):
            bs = int(self._all_block_pairs[block_pair_num, 0])
            be = int(self._all_block_pairs[block_pair_num, 1])

            eps = self.scanner.get_lor_endpoints(
                self.xp.asarray([bs], device=self.dev),
                self.xp.arange(self._num_lorendpoints_per_block, device=self.dev),
            )
            epe = self.scanner.get_lor_endpoints(
                self.xp.asarray([be], device=self.dev),
                self.xp.arange(self._num_lorendpoints_per_block, device=self.dev),
            )

            tmp = self.xp.arange(self._num_lorendpoints_per_block, device=self.dev)
            a, b = self.xp.meshgrid(tmp, tmp, indexing="ij")
            a = self.xp.reshape(a, (-1,))
            b = self.xp.reshape(b, (-1,))

            xstart[i, ...] = self.xp.take(eps, a, axis=0)
            xend[i, ...] = self.xp.take(epe, b, axis=0)

        return self.xp.reshape(xstart, (-1, 3)), self.xp.reshape(xend, (-1, 3))

    def show_block_pair_lors(
        self, ax: Axes, block_pair_nums: Array, lw: float = 0.2, **kwargs
    ) -> None:
        """show all LORs connecting all endpoints between blocks forming a block pairs

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        block_pair_nums : int
            the block pair numbers to show
        lw : float, optional
            the line width, by default 0.2
        """

        xs, xe = self.get_lor_coordinates(block_pair_nums=block_pair_nums)

        p1s = to_numpy_array(xs)
        p2s = to_numpy_array(xe)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)


class RegularPolygonPETLORDescriptor(PETLORDescriptor):
    """LOR descriptor for a regular polygon PET scanner where
    we have coincidences within and between "rings (polygons of modules)"
    The geometrical LORs can be sorted into a sinogram having a
    "plane", "view" and "radial" axis.

    Examples
    --------
    .. minigallery:: parallelproj.RegularPolygonPETLORDescriptor
    """

    def __init__(
        self,
        scanner: RegularPolygonPETScannerGeometry,
        radial_trim: int = 3,
        max_ring_difference: int | None = None,
        sinogram_order: SinogramSpatialAxisOrder = SinogramSpatialAxisOrder.RVP,
        span: int = 1,
    ) -> None:
        """

        Parameters
        ----------
        scanner : RegularPolygonPETScannerGeometry
            a regular polygon PET scanner
        radial_trim : int, optional
            number of geometrial LORs to disregard in the radial direction, by default 3
        max_ring_difference : int | None, optional
            maximim ring difference to consider for coincidences, by default None means
            all ring differences are included
        sinogram_order : SinogramSpatialAxisOrder, optional
            the order of the sinogram axes, by default SinogramSpatialAxisOrder.RVP
        span : int, optional
            axial compression factor (must be odd), by default 1 (no compression).
            With span > 1, ring pairs whose ring difference falls in the same
            segment and share the same axial midpoint are merged into a single sinogram plane
            whose LOR geometry is the average of the constituent ring-pair geometries.
        """

        if span % 2 == 0:
            raise ValueError("span must be odd")

        super().__init__(scanner)

        self._scanner = scanner
        self._radial_trim = radial_trim
        self._span = span

        if max_ring_difference is None:
            self._max_ring_difference = scanner.num_rings - 1
        else:
            self._max_ring_difference = max_ring_difference

        self._num_rad = (scanner.num_lor_endpoints_per_ring + 1) - 2 * self._radial_trim
        self._num_views = scanner.num_lor_endpoints_per_ring // 2

        self._sinogram_order = sinogram_order

        self._setup_plane_indices()
        self._setup_view_indices()

    @property
    def scanner(self) -> RegularPolygonPETScannerGeometry:
        """the scanner for which coincidences are described"""
        return self._scanner

    @property
    def radial_trim(self) -> int:
        """number of geometrial LORs to disregard in the radial direction"""
        return self._radial_trim

    @property
    def max_ring_difference(self) -> int:
        """the maximum ring difference"""
        return self._max_ring_difference

    @property
    def num_planes(self) -> int:
        """number of planes in the sinogram"""
        return self._num_planes

    @property
    def num_rad(self) -> int:
        """number of radial elements in the sinogram"""
        return self._num_rad

    @property
    def num_views(self) -> int:
        """number of views in the sinogram"""
        return self._num_views

    @property
    def span(self) -> int:
        """axial compression factor (1 = no compression)"""
        return self._span

    @property
    def start_plane_index(self) -> Array:
        """start ring index for all planes (only defined for span=1)"""
        if self._span > 1:
            raise AttributeError(
                "start_plane_index is not defined for span > 1. Use start_plane_z instead."
            )
        return self._start_plane_index

    @property
    def end_plane_index(self) -> Array:
        """end ring index for all planes (only defined for span=1)"""
        if self._span > 1:
            raise AttributeError(
                "end_plane_index is not defined for span > 1. Use end_plane_z instead."
            )
        return self._end_plane_index

    @property
    def start_plane_z(self) -> Array:
        """start z-coordinate for all planes (averaged over constituent ring pairs for span > 1)"""
        return self._start_plane_z

    @property
    def end_plane_z(self) -> Array:
        """end z-coordinate for all planes (averaged over constituent ring pairs for span > 1)"""
        return self._end_plane_z

    @property
    def plane_multiplicity(self) -> Array:
        """number of ring pairs contributing to each plane (always 1 for span=1)"""
        return self._plane_multiplicity

    @property
    def plane_segment(self) -> Array:
        """segment number for each plane (equals abs(rd) for span=1)"""
        return self._plane_segment

    @property
    def start_in_ring_index(self) -> Array:
        """start index within ring for all views - shape (num_view, num_rad)"""
        return self._start_in_ring_index

    @property
    def end_in_ring_index(self) -> Array:
        """end index within ring for all views - shape (num_view, num_rad)"""
        return self._end_in_ring_index

    @property
    def sinogram_order(self) -> SinogramSpatialAxisOrder:
        """the order of the sinogram axes"""
        return self._sinogram_order

    @property
    def plane_axis_num(self) -> int:
        """the axis number of the plane axis"""
        return self.sinogram_order.name.find("P")

    @property
    def radial_axis_num(self) -> int:
        """the axis number of the radial axis"""
        return self.sinogram_order.name.find("R")

    @property
    def view_axis_num(self) -> int:
        """the axis number of the view axis"""
        return self.sinogram_order.name.find("V")

    @property
    def spatial_sinogram_shape(self) -> tuple[int, ...]:
        """the shape of the sinogram in spatial order"""
        shape = [0, 0, 0]
        shape[self.plane_axis_num] = self.num_planes
        shape[self.view_axis_num] = self.num_views
        shape[self.radial_axis_num] = self.num_rad
        return tuple(shape)

    def __str__(self) -> str:
        """string representation"""

        return (
            self.__class__.__name__
            + " with spatial sinogram shape ("
            + ", ".join(
                [
                    f"{self.spatial_sinogram_shape[i]}{self.sinogram_order.name[i]}"
                    for i in range(3)
                ]
            )
            + ")"
        )

    def _ring_diff_to_segment(self, rd: int) -> int:
        """Return the signed segment number for a given ring difference.

        Parameters
        ----------
        rd : int
            Ring difference ``end_ring - start_ring``.

        Returns
        -------
        int
            Segment number (0 for segment 0, ±k for outer segments).
        """
        S = self._span
        half_span = (S - 1) // 2
        abs_rd = abs(rd)
        if abs_rd <= half_span:
            return 0
        k = (abs_rd - half_span + S - 1) // S  # ceil division
        return k if rd > 0 else -k

    def _setup_plane_indices(self) -> None:
        """dispatch to unspanned or spanned plane index setup"""
        if self._span == 1:
            self._setup_unspanned_plane_indices()
        else:
            self._setup_spanned_plane_indices()

    def _setup_unspanned_plane_indices(self) -> None:
        """setup the start / end plane indices for span=1 (similar to a Michelogram)"""
        self._start_plane_index = self.xp.arange(
            self._scanner.num_rings, dtype=self.xp.int32, device=self.dev
        )
        self._end_plane_index = self.xp.arange(
            self._scanner.num_rings, dtype=self.xp.int32, device=self.dev
        )

        for i in range(1, self._max_ring_difference + 1):
            tmp1 = self.xp.arange(
                self._scanner.num_rings - i, dtype=self.xp.int16, device=self.dev
            )
            tmp2 = (
                self.xp.arange(
                    self._scanner.num_rings - i, dtype=self.xp.int16, device=self.dev
                )
                + i
            )

            self._start_plane_index = self.xp.concat(
                (self._start_plane_index, tmp1, tmp2)
            )
            self._end_plane_index = self.xp.concat((self._end_plane_index, tmp2, tmp1))

        self._num_planes = self._start_plane_index.shape[0]

        self._start_plane_z = self.xp.astype(
            self.xp.take(
                self._scanner.ring_positions,
                self.xp.astype(self._start_plane_index, self.xp.int64),
            ),
            self.xp.float32,
        )
        self._end_plane_z = self.xp.astype(
            self.xp.take(
                self._scanner.ring_positions,
                self.xp.astype(self._end_plane_index, self.xp.int64),
            ),
            self.xp.float32,
        )
        self._plane_multiplicity = self.xp.ones(
            self._num_planes, dtype=self.xp.int32, device=self.dev
        )
        self._plane_segment = self.xp.astype(
            self.xp.astype(self._end_plane_index, self.xp.int32)
            - self.xp.astype(self._start_plane_index, self.xp.int32),
            self.xp.int32,
        )

    def _setup_spanned_plane_indices(self) -> None:
        """Setup spanned plane z-coordinates using axial compression.

        Ring pairs (start_ring, end_ring) whose ring difference falls in the same
        segment and share the same axial midpoint (start_ring + end_ring) are grouped
        into one sinogram plane.  The LOR geometry of that plane is the average of the
        constituent ring-pair z-positions.

        Segment assignment for ring difference rd (span S, half_span = (S-1)//2):
          seg = 0                              if |rd| <= half_span
          seg = ±ceil((|rd|-half_span) / S)   otherwise  (sign follows rd)

        Planes are ordered: segment 0, then +1, -1, +2, -2, … each sorted by
        increasing axial midpoint (start_ring + end_ring).
        """
        import numpy as _np

        R = self._scanner.num_rings
        D = self._max_ring_difference

        ring_pos = _np.asarray(
            to_numpy_array(self._scanner.ring_positions), dtype=_np.float64
        )

        # Group (start_ring, end_ring) pairs by (segment, axial_midpoint_int)
        # where axial_midpoint_int = start_ring + end_ring  (= 2 * midpoint, integer)
        plane_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}

        for s in range(R):
            for e in range(R):
                rd = e - s
                if abs(rd) > D:
                    continue
                seg = self._ring_diff_to_segment(rd)
                key = (seg, s + e)
                if key not in plane_groups:
                    plane_groups[key] = []
                plane_groups[key].append((s, e))

        # Order: seg 0, then +1, -1, +2, -2, …; within each segment by axial midpoint
        sorted_keys = sorted(
            plane_groups.keys(), key=lambda k: (abs(k[0]), -k[0], k[1])
        )

        self._num_planes = len(sorted_keys)

        start_z = _np.empty(self._num_planes, dtype=_np.float32)
        end_z = _np.empty(self._num_planes, dtype=_np.float32)
        mult = _np.empty(self._num_planes, dtype=_np.int32)
        seg_arr = _np.empty(self._num_planes, dtype=_np.int32)

        for pi, key in enumerate(sorted_keys):
            pairs = plane_groups[key]
            start_z[pi] = _np.mean([ring_pos[s] for s, _ in pairs])
            end_z[pi] = _np.mean([ring_pos[e] for _, e in pairs])
            mult[pi] = len(pairs)
            seg_arr[pi] = key[0]

        self._start_plane_z = self.xp.asarray(start_z, device=self.dev)
        self._end_plane_z = self.xp.asarray(end_z, device=self.dev)
        self._plane_multiplicity = self.xp.asarray(mult, device=self.dev)
        self._plane_segment = self.xp.asarray(seg_arr, device=self.dev)

        # integer ring indices are not defined for span > 1
        self._start_plane_index = None
        self._end_plane_index = None

    def _setup_view_indices(self) -> None:
        """setup the start / end view indices"""
        n = self._scanner.num_lor_endpoints_per_ring

        m = 2 * (n // 2)

        self._start_in_ring_index = self.xp.zeros(
            (self._num_views, self._num_rad), dtype=self.xp.int32, device=self.dev
        )
        self._end_in_ring_index = self.xp.zeros(
            (self._num_views, self._num_rad), dtype=self.xp.int32, device=self.dev
        )

        for view in np.arange(self._num_views):
            self._start_in_ring_index[view, :] = self.xp.astype(
                (
                    self.xp.concat((self.xp.arange(m) // 2, self.xp.asarray([n // 2])))
                    - int(view)
                )[self._radial_trim : -self._radial_trim],
                self.xp.int32,
            )
            self._end_in_ring_index[view, :] = self.xp.astype(
                (
                    self.xp.concat(
                        (self.xp.asarray([-1]), -((self.xp.arange(m) + 4) // 2))
                    )
                    - int(view)
                )[self._radial_trim : -self._radial_trim],
                self.xp.int32,
            )

        # shift the negative indices
        self._start_in_ring_index = self.xp.where(
            self._start_in_ring_index >= 0,
            self._start_in_ring_index,
            self._start_in_ring_index + n,
        )
        self._end_in_ring_index = self.xp.where(
            self._end_in_ring_index >= 0,
            self._end_in_ring_index,
            self._end_in_ring_index + n,
        )

    def get_lor_coordinates(
        self,
        views: None | Array = None,
    ) -> tuple[Array, Array]:
        """return the start and end coordinates of all LORs / or a subset of views

        Parameters
        ----------
        views : None | Array, optional
            the views to consider, by default None means all views

        Returns
        -------
        xstart, xend : Array
           2 dimensional floating point arrays containing the start and end coordinates of all LORs
        """

        if views is None:
            views = self.xp.arange(self.num_views, device=self.dev)

        # --- (1) setup the LOR start / end points for all views of plane 0

        start_in_ring_index = self.xp.take(self.start_in_ring_index, views, axis=0)
        end_in_ring_index = self.xp.take(self.end_in_ring_index, views, axis=0)

        if self.view_axis_num > self.radial_axis_num:
            start_in_ring_index = start_in_ring_index.T
            end_in_ring_index = end_in_ring_index.T

        shape_2d = start_in_ring_index.shape

        start_inds_2d = self.xp.reshape(start_in_ring_index, (-1,))
        end_inds_2d = self.xp.reshape(end_in_ring_index, (-1,))

        xstart_2d = self.xp.reshape(
            self.scanner.get_lor_endpoints(
                self.xp.zeros_like(start_inds_2d), start_inds_2d
            ),
            shape_2d + (3,),
        )
        xend_2d = self.xp.reshape(
            self.scanner.get_lor_endpoints(
                self.xp.zeros_like(end_inds_2d), end_inds_2d
            ),
            shape_2d + (3,),
        )

        xstart_3d = []
        xend_3d = []

        # --- (2) stack copies of the plane 0 LOR start / end points for all planes with updated "z" coordinates

        for i in range(self.num_planes):
            # make a copy of the 2D coordinates
            # stupid way of adding 0, since asarray with torch and cuda does
            # not seem to work
            xstart = xstart_2d + 0
            xend = xend_2d + 0

            xstart[..., self._scanner.symmetry_axis] = float(self._start_plane_z[i])
            xend[..., self._scanner.symmetry_axis] = float(self._end_plane_z[i])

            xstart_3d.append(xstart)
            xend_3d.append(xend)

        xstart_3d = self.xp.stack(xstart_3d, axis=self.plane_axis_num)
        xend_3d = self.xp.stack(xend_3d, axis=self.plane_axis_num)

        return xstart_3d, xend_3d

    def show_views(
        self, ax: Axes, views: Array, planes: Array, lw: float = 0.2, **kwargs
    ) -> None:
        """show all LORs of a single view in a given plane

        Parameters
        ----------
        ax : plt.Axes
            a 3D matplotlib axes
        view : int
            the view number
        plane : int
            the plane number
        lw : float, optional
            the line width, by default 0.2
        """

        xs, xe = self.get_lor_coordinates(views=views)

        xs = self.xp.reshape(
            self.xp.take(xs, planes, axis=self.plane_axis_num), (-1, 3)
        )
        xe = self.xp.reshape(
            self.xp.take(xe, planes, axis=self.plane_axis_num), (-1, 3)
        )

        p1s = to_numpy_array(xs)
        p2s = to_numpy_array(xe)

        ls = np.hstack([p1s, p2s]).copy()
        ls = ls.reshape((-1, 2, 3))
        lc = Line3DCollection(ls, linewidths=lw, **kwargs)
        ax.add_collection(lc)

    def _draw_michelogram(self, ax, show_merge_lines: bool = True, **kwargs) -> None:
        """Draw the Michelogram onto an existing axes.

        Internal helper shared by :meth:`show_michelogram` and
        :meth:`show_segment_lors`.
        """
        import numpy as _np
        from matplotlib.colors import BoundaryNorm, ListedColormap

        R = self._scanner.num_rings
        D = self._max_ring_difference

        start_list, end_list, seg_list, z_list = [], [], [], []
        for s in range(R):
            for e in range(R):
                rd = e - s
                if abs(rd) > D:
                    continue
                seg = self._ring_diff_to_segment(rd)
                start_list.append(s)
                end_list.append(e)
                seg_list.append(seg)
                z_list.append(s + e)

        start_arr = _np.array(start_list, dtype=_np.float32)
        end_arr = _np.array(end_list, dtype=_np.float32)
        seg_arr = _np.array(seg_list, dtype=_np.int32)
        z_arr = _np.array(z_list, dtype=_np.int32)

        abs_seg_arr = _np.abs(seg_arr)
        n_colors = int(abs_seg_arr.max()) + 1

        base_cmap = plt.get_cmap("tab10" if n_colors <= 10 else "tab20")
        cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
        norm = BoundaryNorm(_np.arange(-0.5, n_colors, 1.0), cmap.N)

        # build plane_groups in the same order used by the projector
        plane_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for s in range(R):
            for e in range(R):
                rd = e - s
                if abs(rd) > D:
                    continue
                seg = self._ring_diff_to_segment(rd)
                key = (seg, s + e)
                if key not in plane_groups:
                    plane_groups[key] = []
                plane_groups[key].append((s, e))
        sorted_keys = sorted(
            plane_groups.keys(), key=lambda k: (abs(k[0]), -k[0], k[1])
        )

        kwargs.setdefault("s", 20)
        ax.scatter(
            start_arr,
            end_arr,
            c=abs_seg_arr.astype(_np.float32),
            cmap=cmap,
            norm=norm,
            **kwargs,
        )

        if show_merge_lines and self._span > 1:
            for seg, z_int in set(zip(seg_list, z_list)):
                mask = (seg_arr == seg) & (z_arr == z_int)
                xs, ys = start_arr[mask], end_arr[mask]
                if xs.size > 1:
                    order = _np.argsort(xs)
                    ax.plot(xs[order], ys[order], color="gray", lw=0.5, alpha=0.5)

        # annotate each compressed plane with its plane index at the group centroid
        for pi, key in enumerate(sorted_keys):
            pairs = plane_groups[key]
            cx = _np.mean([s for s, _ in pairs])
            cy = _np.mean([e for _, e in pairs])
            ax.text(
                cx,
                cy,
                str(pi),
                ha="center",
                va="center",
                fontsize=4,
                color="black",
                fontweight="bold",
                zorder=10,
            )

        ax.set_xlabel("start ring")
        ax.set_ylabel("end ring")
        ax.set_title(
            f"Michelogram\n(span={self._span}, max Δring={D})", fontsize="small"
        )
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, R - 0.5)
        ax.set_ylim(-0.5, R - 0.5)

    def show_michelogram(
        self,
        ax,
        show_merge_lines: bool = True,
        **kwargs,
    ):
        """Visualize the Michelogram with ring pairs colored by segment number.

        Each point represents a valid ring pair (start_ring, end_ring).  Points are
        colored by their segment number.  For span > 1, ring pairs that share the same
        segment *and* the same axial midpoint (start_ring + end_ring) are merged into a
        single sinogram plane; ``show_merge_lines=True`` draws short lines connecting
        those merged ring pairs so the compression structure is visible.

        Parameters
        ----------
        ax : plt.Axes
            2D matplotlib axes (not 3D).
        show_merge_lines : bool, optional
            draw lines connecting ring pairs that are merged into the same sinogram
            plane, by default True.  Only has a visible effect for span > 1.
        **kwargs
            forwarded to ``ax.scatter`` (e.g. ``s=4``, ``cmap="RdBu_r"``).
        """
        self._draw_michelogram(ax, show_merge_lines=show_merge_lines, **kwargs)

    def show_segment_lors(
        self,
        axs=None,
        uncompressed_lor_kwargs: dict | None = None,
        compressed_lor_kwargs: dict | None = None,
    ):
        """Side-view LOR diagram per segment with the Michelogram inset.

        Subplots are arranged in a 2-row grid (when negative segments exist):

        * **columns** – indexed by ``abs(segment)``: 0, 1, 2, …
        * **row 0** – non-negative segments (0, +1, +2, …)
        * **row 1, col 0** – Michelogram
        * **row 1, col ≥ 1** – negative segments (−1, −2, …)

        Each LOR subplot shows:

        * **solid black lines** – every individual ring-pair LOR in the segment
          (the "uncompressed" planes);
        * **dashed coloured lines** – the compressed (axially-averaged) LOR geometry
          (one line per spanned plane).

        Parameters
        ----------
        axs : 2-D array-like of Axes, optional
            Pre-existing axes of shape ``(n_rows, n_cols)``.  If None a new
            figure is created automatically.
        uncompressed_lor_kwargs : dict, optional
            Style overrides for the uncompressed LOR lines.
        compressed_lor_kwargs : dict, optional
            Style overrides for the compressed LOR lines.

        Returns
        -------
        matplotlib.figure.Figure
        """
        import numpy as _np
        from matplotlib.lines import Line2D
        from matplotlib.patches import Circle

        R = self._scanner.num_rings
        D = self._max_ring_difference
        ring_pos = _np.asarray(
            to_numpy_array(self._scanner.ring_positions), dtype=_np.float64
        )

        seg_arr_np = _np.asarray(to_numpy_array(self._plane_segment), dtype=_np.int32)
        all_segs = sorted(set(int(v) for v in seg_arr_np))
        abs_segs = sorted(set(abs(s) for s in all_segs))  # column indices
        n_cols = len(abs_segs)
        neg_segs = [s for s in all_segs if s < 0]
        n_rows = 2 if neg_segs else 1

        unc_kw: dict = {"color": "black", "lw": 1.0, "alpha": 0.5}
        if uncompressed_lor_kwargs:
            unc_kw.update(uncompressed_lor_kwargs)
        com_kw: dict = {"lw": 1.5, "alpha": 0.9, "linestyle": "--"}
        if compressed_lor_kwargs:
            com_kw.update(compressed_lor_kwargs)

        created_fig = axs is None
        if created_fig:
            fig, raw = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3 * n_cols, 4 * n_rows),
                squeeze=False,
            )
            _axs = raw  # shape (n_rows, n_cols)
        else:
            import numpy as _np2

            _axs = _np2.asarray(axs)
            fig = _axs.flat[0].get_figure()

        # --- coordinate normalisation -----------------------------------
        z_min, z_max = ring_pos.min(), ring_pos.max()
        z_span = max(z_max - z_min, 1.0)
        margin = 0.12 * z_span
        x_L, x_R = -z_span / 2.0, z_span / 2.0
        ring_r = 0.35 * z_span / max(R, 1)

        # --- build uncompressed ring-pair lookup (keyed by signed segment) ---
        uncompressed: dict[int, list[tuple[int, int]]] = {s: [] for s in all_segs}
        for s in range(R):
            for e in range(R):
                rd = e - s
                if abs(rd) > D:
                    continue
                seg = self._ring_diff_to_segment(rd)
                if seg in uncompressed:
                    uncompressed[seg].append((s, e))

        start_z_np = _np.asarray(to_numpy_array(self._start_plane_z))
        end_z_np = _np.asarray(to_numpy_array(self._end_plane_z))

        n_colors = len(abs_segs)
        base_cmap = plt.get_cmap("tab10" if n_colors <= 10 else "tab20")

        for col_idx, abs_seg in enumerate(abs_segs):
            color = base_cmap(col_idx)

            for row_idx in range(n_rows):
                ax = _axs[row_idx, col_idx]

                # [1, 0]: michelogram inset instead of the non-existent segment -0
                if row_idx == 1 and abs_seg == 0:
                    self._draw_michelogram(ax)
                    continue

                seg_val = abs_seg if row_idx == 0 else -abs_seg

                if seg_val not in all_segs:
                    ax.axis("off")
                    continue

                # (a) uncompressed ring-pair LORs
                for s, e in uncompressed[seg_val]:
                    ax.plot([x_L, x_R], [ring_pos[s], ring_pos[e]], **unc_kw)

                # (b) compressed (averaged) LORs
                mask = seg_arr_np == seg_val
                kw = dict(com_kw)
                if "color" not in kw:
                    kw["color"] = color
                for sz, ez in zip(start_z_np[mask], end_z_np[mask]):
                    ax.plot([x_L, x_R], [float(sz), float(ez)], **kw)

                # detector rings: one Circle per ring at each detector side
                for xpos in [x_L, x_R]:
                    for z in ring_pos:
                        ax.add_patch(
                            Circle(
                                (xpos, float(z)),
                                ring_r,
                                edgecolor="black",
                                facecolor="lightgray",
                                lw=0.6,
                                zorder=5,
                            )
                        )

                if seg_val > 0:
                    seg_label = f"+{abs_seg}"
                elif seg_val < 0:
                    seg_label = f"−{abs_seg}"
                else:
                    seg_label = "0"
                n_compressed = int((seg_arr_np == seg_val).sum())
                n_uncompressed = len(uncompressed[seg_val])
                ax.set_title(
                    f"seg {seg_label}  {n_compressed} / {n_uncompressed}",
                    fontsize="small",
                )
                ax.set_xlim(x_L - 2 * ring_r, x_R + 2 * ring_r)
                ax.set_ylim(z_min - margin * 2.0, z_max + margin * 2.0)
                ax.set_aspect("equal")
                ax.axis("off")

                # legend on top-left subplot only
                if row_idx == 0 and col_idx == 0:
                    ax.legend(
                        handles=[
                            Line2D(
                                [0],
                                [0],
                                color="black",
                                lw=1.0,
                                alpha=0.5,
                                label="uncompressed",
                            ),
                            Line2D(
                                [0],
                                [0],
                                color="black",
                                lw=1.5,
                                linestyle="--",
                                label="compressed",
                            ),
                        ],
                        loc="upper right",
                        fontsize="x-small",
                    )

        if created_fig:
            fig.tight_layout()
        return fig

    def get_distributed_views_and_slices(
        self, num_subsets: int, num_dim: int
    ) -> tuple[list[Array], list[tuple[slice, ...]]]:
        """distribute sinogram views numbers into subsets

        Parameters
        ----------
        num_subsets : int
            number of subsets
        num_dim : int
            number of dimensions of the sinogram
            to setup the subset slices
            (e.g. 3 for non-TOF, 4 for TOF)

        Returns
        -------
        tuple[list[Array], list[tuple[slice, ...]]]
            subset views numbers and subset slices
        """
        subset_nums = []

        for i in range(num_subsets // 2):
            subset_nums += [x for x in range(i, num_subsets, num_subsets // 2)]

        subset_slices = []
        subset_views = []
        all_views = self.xp.arange(self.num_views, device=self.dev)

        for i in subset_nums:
            sl = num_dim * [slice(None)]
            sl[self.view_axis_num] = slice(i, None, num_subsets)
            sl = tuple(sl)
            subset_slices.append(sl)
            subset_views.append(all_views[sl[self.view_axis_num]])

        return subset_views, subset_slices
