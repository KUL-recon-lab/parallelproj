"""description of PET LORs (and sinograms bins) consisting of two detector endpoints"""

from __future__ import annotations

import abc
import enum
import array_api_compat.numpy as np
from parallelproj import Array
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from types import ModuleType

from .pet_scanners import (
    ModularizedPETScannerGeometry,
    RegularPolygonPETScannerGeometry,
)

from .backend import to_numpy_array


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
        self, ax: plt.Axes, block_pair_nums: Array, lw: float = 0.2, **kwargs
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
        """

        super().__init__(scanner)

        self._scanner = scanner
        self._radial_trim = radial_trim

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
    def start_plane_index(self) -> Array:
        """start plane for all planes"""
        return self._start_plane_index

    @property
    def end_plane_index(self) -> Array:
        """end plane for all planes"""
        return self._end_plane_index

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
    def spatial_sinogram_shape(self) -> tuple[int, int, int]:
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

    def _setup_plane_indices(self) -> None:
        """setup the start / end plane indices (similar to a Michelogram)"""
        self._start_plane_index = self.xp.arange(
            self.scanner.num_rings, dtype=self.xp.int32, device=self.dev
        )
        self._end_plane_index = self.xp.arange(
            self.scanner.num_rings, dtype=self.xp.int32, device=self.dev
        )

        for i in range(1, self._max_ring_difference + 1):
            tmp1 = self.xp.arange(
                self.scanner.num_rings - i, dtype=self.xp.int16, device=self.dev
            )
            tmp2 = (
                self.xp.arange(
                    self.scanner.num_rings - i, dtype=self.xp.int16, device=self.dev
                )
                + i
            )

            self._start_plane_index = self.xp.concat(
                (self._start_plane_index, tmp1, tmp2)
            )
            self._end_plane_index = self.xp.concat((self._end_plane_index, tmp2, tmp1))

        self._num_planes = self._start_plane_index.shape[0]

    def _setup_view_indices(self) -> None:
        """setup the start / end view indices"""
        n = self.scanner.num_lor_endpoints_per_ring

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

            xstart[..., self.scanner.symmetry_axis] = float(
                self.scanner.ring_positions[self.start_plane_index[i]]
            )
            xend[..., self.scanner.symmetry_axis] = float(
                self.scanner.ring_positions[self.end_plane_index[i]]
            )

            xstart_3d.append(xstart)
            xend_3d.append(xend)

        xstart_3d = self.xp.stack(xstart_3d, axis=self.plane_axis_num)
        xend_3d = self.xp.stack(xend_3d, axis=self.plane_axis_num)

        return xstart_3d, xend_3d

    def show_views(
        self, ax: plt.Axes, views: Array, planes: Array, lw: float = 0.2, **kwargs
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
