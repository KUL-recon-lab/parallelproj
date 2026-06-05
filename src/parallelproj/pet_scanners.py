"""PET scanner geometry classes describing detector module layouts and endpoint coordinates.

Covers both modular scanners (:class:`ModularizedPETScannerGeometry`, where
modules can be placed arbitrarily) and the regular-polygon scanner
(:class:`RegularPolygonPETScannerGeometry`), which is the primary geometry
used throughout parallelproj.  Provides methods for computing LOR endpoint
world coordinates and visualising the detector arrangement.
"""

from __future__ import annotations

import abc
import enum
import warnings
from collections.abc import Sequence
from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np

from ._backend import Array, to_numpy_array

if TYPE_CHECKING:
    from mpl_toolkits.mplot3d import Axes3D


class RingEndpointOrdering(enum.Enum):
    """Direction in which endpoint indices increase around a detector ring.

    The ordering is defined relative to the scanner's symmetry axis
    (set via the ``symmetry_axis`` parameter of
    :class:`RegularPolygonPETScannerGeometry`).  When the ring is viewed
    from the **positive** symmetry-axis direction, the two conventions are:

    ``CLOCKWISE``
        Index 0 is at the 12 o'clock position and subsequent indices advance
        clockwise.  This is the default and matches the original behaviour.

    ``COUNTERCLOCKWISE``
        Index 0 is at the 12 o'clock position and subsequent indices advance
        counterclockwise.

    For the common case ``symmetry_axis=2`` (axial = z), the ring lies in
    the x-y plane and is viewed from above (+z).  Index 0 is at the top
    (+y direction) and advances clockwise (toward +x) or counterclockwise
    (toward -x) depending on the chosen convention.
    """

    CLOCKWISE = enum.auto()
    """Indices increase clockwise (default)."""
    COUNTERCLOCKWISE = enum.auto()
    """Indices increase counterclockwise."""


class PETScannerModule(abc.ABC):
    """Abstract base class for a single detector module in a PET scanner.

    A module groups a fixed number of LOR endpoints (detector elements) and
    exposes their world coordinates via :meth:`get_raw_lor_endpoints` (before
    any affine transform) and :meth:`get_lor_endpoints` (after).  An optional
    4x4 affine transformation matrix can be supplied to reposition the module
    in world space.

    Concrete subclasses:

    - :class:`BlockPETScannerModule` -- a 3-D rectangular grid of crystals
    - :class:`RegularPolygonPETScannerModule` -- one flat side of a regular-polygon scanner
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        num_lor_endpoints: int,
        affine_transformation_matrix: Array | None = None,
    ) -> None:
        """

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        num_lor_endpoints : int
            number of LOR endpoints in the module
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        """

        self._xp = xp
        self._dev = dev
        self._num_lor_endpoints = num_lor_endpoints
        self._lor_endpoint_numbers = self.xp.arange(num_lor_endpoints, device=self.dev)

        if affine_transformation_matrix is None:
            self._affine_transformation_matrix = self.xp.eye(4, device=self.dev)
            self._has_affine_transformation = False
        else:
            self._affine_transformation_matrix = affine_transformation_matrix
            self._has_affine_transformation = True

    @property
    def xp(self) -> ModuleType:
        """array module to use for storing the LOR endpoints"""
        return self._xp

    @property
    def dev(self) -> str:
        """device to use for storing the LOR endpoints"""
        return self._dev

    @property
    def num_lor_endpoints(self) -> int:
        """total number of LOR endpoints in the module

        Returns
        -------
        int
        """
        return self._num_lor_endpoints

    @property
    def lor_endpoint_numbers(self) -> Array:
        """array enumerating all the LOR endpoints in the module

        Returns
        -------
        Array
        """
        return self._lor_endpoint_numbers

    @property
    def affine_transformation_matrix(self) -> Array:
        """4x4 affine transformation matrix

        Returns
        -------
        Array
            4x4 identity matrix if no transformation was provided
        """
        return self._affine_transformation_matrix

    @abc.abstractmethod
    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """mapping from LOR endpoint indices within module to an array of "raw" world coordinates

        Parameters
        ----------
        inds : Array | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        Array
            float array of shape (len(inds), 3) with the world coordinates of the LOR endpoints
        """
        raise NotImplementedError

    def get_lor_endpoints(self, inds: Array | None = None) -> Array:
        """mapping from LOR endpoint indices within module to an array of "transformed" world coordinates

        Parameters
        ----------
        inds : Array | None, optional
            an non-negative integer array of indices, default None
            if None means all possible indices [0, ... , num_lor_endpoints - 1]

        Returns
        -------
        Array
            float array of shape (len(inds), 3) with the world coordinates of the LOR endpoints
            after applying the affine transformation (if any)
        """

        lor_endpoints = self.get_raw_lor_endpoints(inds)

        if self._has_affine_transformation:
            tmp = self.xp.ones(
                (lor_endpoints.shape[0], 4), device=self.dev, dtype=self.xp.float32
            )
            tmp[:, :-1] = lor_endpoints
            lor_endpoints = (tmp @ self.affine_transformation_matrix.T)[:, :3]

        return lor_endpoints

    def show_lor_endpoints(
        self,
        ax: Axes3D,
        annotation_fontsize: float = 0,
        annotation_prefix: str = "",
        annotation_offset: int = 0,
        transformed: bool = True,
        **kwargs,
    ) -> None:
        """show the LOR coordinates in a 3D scatter plot

        Parameters
        ----------
        ax : Axes3D
            3D matplotlib axes
        annotation_fontsize : float, optional
            fontsize of LOR endpoint number annotation, by default 0
        annotation_prefix : str, optional
            prefix for annotation, by default ''
        annotation_offset : int, optional
            number to add to crystal number, by default 0
        transformed : bool, optional
            use transformed instead of raw coordinates, by default True
        """

        if transformed:
            all_lor_endpoints = self.get_lor_endpoints()
        else:
            all_lor_endpoints = self.get_raw_lor_endpoints()

        # convert to numpy array
        all_lor_endpoints = to_numpy_array(all_lor_endpoints)

        ax.scatter(
            all_lor_endpoints[:, 0],
            all_lor_endpoints[:, 1],
            all_lor_endpoints[:, 2],  # type: ignore[arg-type]
            **kwargs,
        )

        ax.set_box_aspect(
            [ub - lb for lb, ub in (getattr(ax, f"get_{a}lim")() for a in "xyz")]
        )

        ax.set_xlabel("x0")
        ax.set_ylabel("x1")
        ax.set_zlabel("x2")

        if annotation_fontsize > 0:
            for i in self.lor_endpoint_numbers:
                ax.text(
                    all_lor_endpoints[int(i), 0],
                    all_lor_endpoints[int(i), 1],
                    all_lor_endpoints[int(i), 2],
                    f"{annotation_prefix}{i+annotation_offset}",
                    fontsize=annotation_fontsize,
                )


class BlockPETScannerModule(PETScannerModule):
    """Block (rectangular cuboid) PET scanner module"""

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        shape: tuple[int, int, int],
        spacing: tuple[float, float, float],
        affine_transformation_matrix: Array | None = None,
    ) -> None:
        """

        Parameters
        ----------
        xp : ModuleType
            array module to use for storing the LOR endpoints
        dev : str
            device to use for storing the LOR endpoints
        shape : tuple[int, int, int]
            shape of the regular grid of LOR endpoints forming the block module
        spacing : tuple[float, float, float]
            spacing between the LOR endpoints in each direction
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied eagerly to the LOR endpoint
            coordinates at construction time.  The transformed positions are
            stored directly; the matrix itself is not retained, so subsequent
            calls to :meth:`get_lor_endpoints` apply no further transform.
            ``None`` (default) leaves the endpoints in their local frame.
        """

        self._shape = shape
        self._spacing = spacing

        # calculate the LOR endpoints
        x0 = spacing[0] * (np.arange(shape[0], dtype=np.float32) - (shape[0] - 1) / 2)
        x1 = spacing[1] * (np.arange(shape[1], dtype=np.float32) - (shape[1] - 1) / 2)
        x2 = spacing[2] * (np.arange(shape[2], dtype=np.float32) - (shape[2] - 1) / 2)

        # in the current version (1.12.0) of array_api_compat.torch the indexing kwargs is ignored
        # which is why we stick to numpy
        X0, X1, X2 = np.meshgrid(x0, x1, x2, indexing="ij")

        lor_endpoints_np = np.stack(
            (X0.ravel(), X1.ravel(), X2.ravel()),
            axis=-1,
        )

        self._lor_endpoints: Array = xp.asarray(lor_endpoints_np, device=dev)

        if affine_transformation_matrix is not None:
            tmp = xp.ones(
                (self._lor_endpoints.shape[0], 4), device=dev, dtype=xp.float32
            )
            tmp[:, :-1] = self._lor_endpoints

            self._lor_endpoints = (tmp @ affine_transformation_matrix.T)[:, :3]

        super().__init__(xp, dev, shape[0] * shape[1] * shape[2], None)

    @property
    def shape(self) -> tuple[int, int, int]:
        """shape of the block module

        Returns
        -------
        tuple[int, int, int]
        """
        return self._shape

    @property
    def spacing(self) -> tuple[float, float, float]:
        """spacing of the block module

        Returns
        -------
        tuple[float, float, float]
        """
        return self._spacing

    @property
    def lor_endpoints(self) -> Array:
        """LOR endpoints of the block module

        Returns
        -------
        Array
        """
        return self._lor_endpoints

    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """Return world coordinates of the requested crystal endpoints.

        Looks up pre-computed endpoint positions from the stored grid array.

        Parameters
        ----------
        inds : Array | None, optional
            Integer indices into the module's endpoint list.
            ``None`` returns all endpoints (default).

        Returns
        -------
        Array
            Float array of shape ``(len(inds), 3)`` with world coordinates.
        """
        if inds is None:
            inds = self.lor_endpoint_numbers

        return self.xp.take(self.lor_endpoints, inds, axis=0)


class RegularPolygonPETScannerModule(PETScannerModule):
    """Regular polygon PET scanner module (detectors on a regular polygon)"""

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        radius: float,
        num_sides: int,
        num_lor_endpoints_per_side: int | None = None,
        lor_spacing: float | None = None,
        ax0: int = 2,
        ax1: int = 1,
        affine_transformation_matrix: Array | None = None,
        phis: None | Array = None,
        ring_endpoint_ordering: RingEndpointOrdering = RingEndpointOrdering.CLOCKWISE,
        phi0: float = 0.0,
        lor_endpoint_positions: np.ndarray | None = None,
    ) -> None:
        """

        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon
        num_sides: int
            number of sides of the regular polygon
        num_lor_endpoints_per_side: int or None, optional
            number of LOR endpoints per side.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        lor_spacing : float or None, optional
            uniform spacing between LOR endpoints in mm.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        ax0 : int, optional
            axis number for the first direction, by default 2
        ax1 : int, optional
            axis number for the second direction, by default 1
        affine_transformation_matrix : Array | None, optional
            4x4 affine transformation matrix applied to the LOR endpoint coordinates, default None
            if None, the 4x4 identity matrix is used
        phis : None | Array, optional
            angle of each side, by default None
            means that the sides are equally spaced around a circle
        ring_endpoint_ordering : RingEndpointOrdering, optional
            direction in which endpoint indices increase around the ring, by
            default ``RingEndpointOrdering.CLOCKWISE``.
        phi0 : float, optional
            azimuthal offset of side 0 in radians, by default 0.
            Only applied when ``phis`` is ``None``; ignored when ``phis`` is
            provided explicitly.
        lor_endpoint_positions : np.ndarray or None, optional
            1-D array of crystal positions (in mm) along each polygon side,
            with 0 at the centre of the side.  When given, overrides
            ``num_lor_endpoints_per_side`` and ``lor_spacing``.

            For radial sinogram symmetry to hold (see
            :func:`~parallelproj.sinogram_symmetries.build_radial_class_indices`),
            the array must be **anti-symmetric about 0**:
            ``pos[i] == -pos[N-1-i]`` for all ``i``.  A ``UserWarning`` is
            issued if this condition is not met.

            Examples for a side with **even** N=6 (3 crystals each half,
            uniform 2 mm pitch, 1 mm gap)::

                lor_endpoint_positions = np.array([-3.5, -1.5, -0.5,
                                                    0.5,  1.5,  3.5])

            Examples for **odd** N=5 (2 crystals each half + one centre
            crystal, uniform 2 mm pitch)::

                lor_endpoint_positions = np.array([-4.0, -2.0,  0.0,
                                                    2.0,  4.0])
        """

        self._radius = radius
        self._num_sides = num_sides
        self._ax0 = ax0
        self._ax1 = ax1
        self._ring_endpoint_ordering = ring_endpoint_ordering
        self._phi0 = phi0

        if lor_endpoint_positions is not None:
            pos = np.asarray(lor_endpoint_positions, dtype=np.float32)
            # warn if not anti-symmetric about 0
            if not np.allclose(pos + pos[::-1], 0, atol=1e-4 * max(float(np.max(np.abs(pos))), 1.0)):
                warnings.warn(
                    "lor_endpoint_positions is not anti-symmetric about 0 "
                    "(pos[i] != -pos[N-1-i]). Radial sinogram symmetry "
                    "(build_radial_class_indices) may not be physically valid.",
                    UserWarning,
                    stacklevel=2,
                )
            self._lor_endpoint_positions = xp.asarray(pos, device=dev, dtype=xp.float32)
            self._num_lor_endpoints_per_side = len(pos)
            self._lor_spacing = None
        elif num_lor_endpoints_per_side is not None and lor_spacing is not None:
            N = num_lor_endpoints_per_side
            pos_np = lor_spacing * (np.arange(N, dtype=np.float32) - (N - 1) / 2.0)
            self._lor_endpoint_positions = xp.asarray(pos_np, device=dev, dtype=xp.float32)
            self._num_lor_endpoints_per_side = N
            self._lor_spacing = lor_spacing
        else:
            raise ValueError(
                "Provide either lor_endpoint_positions or both "
                "num_lor_endpoints_per_side and lor_spacing."
            )

        super().__init__(
            xp,
            dev,
            num_sides * self._num_lor_endpoints_per_side,
            affine_transformation_matrix,
        )

        # angle of each "side"
        if phis is None:
            self._phis = (
                phi0
                + 2
                * self.xp.pi
                * self.xp.arange(self._num_sides, dtype=xp.float32, device=dev)
                / self.num_sides
            )
        else:
            self._phis = phis

    @property
    def radius(self) -> float:
        """inner radius of the regular polygon

        Returns
        -------
        float
        """
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides of the regular polygon

        Returns
        -------
        int
        """
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side

        Returns
        -------
        int
        """
        return self._num_lor_endpoints_per_side

    @property
    def ax0(self) -> int:
        """axis number for the first module direction

        Returns
        -------
        int
        """
        return self._ax0

    @property
    def ax1(self) -> int:
        """axis number for the second module direction

        Returns
        -------
        int
        """
        return self._ax1

    @property
    def lor_spacing(self) -> float | None:
        """Uniform spacing between LOR endpoints in mm, or ``None`` when
        custom ``lor_endpoint_positions`` were supplied."""
        return self._lor_spacing

    @property
    def lor_endpoint_positions(self) -> Array:
        """1-D float32 Array of crystal positions along each polygon side (mm),
        on the same device as the scanner.

        Anti-symmetric about 0 for standard scanners (uniform or gap layout).
        """
        return self._lor_endpoint_positions

    @property
    def phis(self) -> Array:
        """azimuthal angle of each side

        Returns
        -------
        Array
        """
        return self._phis

    @property
    def ring_endpoint_ordering(self) -> RingEndpointOrdering:
        """direction in which endpoint indices increase around the ring"""
        return self._ring_endpoint_ordering

    @property
    def phi0(self) -> float:
        """azimuthal offset of side 0 in radians (only applied when phis=None)"""
        return self._phi0

    # abstract method from base class to be implemented
    def get_raw_lor_endpoints(self, inds: Array | None = None) -> Array:
        """Compute world coordinates for the requested crystal endpoints.

        Calculates endpoint positions analytically from the scanner geometry
        (radius, number of sides, crystals per side, azimuthal offset),
        respecting the configured :class:`RingEndpointOrdering`.

        Parameters
        ----------
        inds : Array | None, optional
            Integer indices into the module's endpoint list.
            ``None`` returns all endpoints (default).

        Returns
        -------
        Array
            Float array of shape ``(len(inds), 3)`` with world coordinates.
        """
        if inds is None:
            inds = self.lor_endpoint_numbers

        if self._ring_endpoint_ordering is RingEndpointOrdering.COUNTERCLOCKWISE:
            side = inds // self._num_lor_endpoints_per_side
            within = inds - side * self._num_lor_endpoints_per_side
            new_side = self.xp.astype(
                (self._num_sides - side) % self._num_sides, self.xp.int32
            )
            new_within = self.xp.astype(
                self._num_lor_endpoints_per_side - 1 - within, self.xp.int32
            )
            inds = new_side * self._num_lor_endpoints_per_side + new_within

        side = inds // self.num_lor_endpoints_per_side
        within_side = inds - side * self.num_lor_endpoints_per_side

        # Look up the physical position along the side for each endpoint.
        # For uniform spacing this is lor_spacing*(i - (N-1)/2); for custom
        # positions it is whatever the user supplied.
        pos = self.xp.take(self._lor_endpoint_positions, within_side)

        phi = self.xp.take(self._phis, side)

        lor_endpoints = self.xp.zeros(
            (inds.shape[0], 3), device=self.dev, dtype=self.xp.float32
        )

        cosphi = self.xp.cos(phi)
        sinphi = self.xp.sin(phi)

        lor_endpoints[:, self.ax0] = cosphi * self.radius - sinphi * pos
        lor_endpoints[:, self.ax1] = sinphi * self.radius + cosphi * pos

        return lor_endpoints


class ModularizedPETScannerGeometry:
    """A PET scanner geometry built from an ordered list of :class:`PETScannerModule` objects.

    Each module contributes a contiguous block of LOR endpoints to the global
    flat index space.  The global index of endpoint ``k`` in module ``i`` is
    ``all_lor_endpoints_index_offset[i] + k``.  All modules must share the
    same array namespace (``xp``) and device.

    Use :class:`RegularPolygonPETScannerGeometry` for the common case of a
    cylindrical scanner with stacked regular-polygon rings.  Use this class
    directly when the scanner has an irregular or custom module layout.
    """

    def __init__(self, modules: Sequence[PETScannerModule]):
        """
        Parameters
        ----------
        modules : Sequence[PETScannerModule]
            a sequence of scanner modules
        """

        # member variable that determines whether we want to use
        # a numpy or cupy array to store the array of all lor endpoints
        self._modules = modules
        self._num_modules = len(self._modules)
        self._num_lor_endpoints_per_module = self.xp.asarray(
            [x.num_lor_endpoints for x in self._modules], device=self.dev
        )
        self._num_lor_endpoints = int(self.xp.sum(self._num_lor_endpoints_per_module))

        # declare attributes set by setup_all_lor_endpoints so they are visible in __init__
        self._all_lor_endpoints_index_offset: Array = None  # type: ignore[assignment]
        self._all_lor_endpoints: Array = None  # type: ignore[assignment]
        self._all_lor_endpoints_module_number: Array = None  # type: ignore[assignment]

        self.setup_all_lor_endpoints()

    def setup_all_lor_endpoints(self) -> None:
        """calculate the position of all lor endpoints by iterating over
        the modules and calculating the transformed coordinates of all
        module endpoints
        """

        offsets = [0]
        for module in self._modules[:-1]:
            offsets.append(offsets[-1] + module.num_lor_endpoints)
        self._all_lor_endpoints_index_offset = self.xp.asarray(offsets, device=self.dev)

        self._all_lor_endpoints = self.xp.zeros(
            (self._num_lor_endpoints, 3), device=self.dev, dtype=self.xp.float32
        )

        for i, module in enumerate(self._modules):
            self._all_lor_endpoints[
                int(self._all_lor_endpoints_index_offset[i]) : int(
                    self._all_lor_endpoints_index_offset[i] + module.num_lor_endpoints
                ),
                :,
            ] = self.xp.astype(module.get_lor_endpoints(), self.xp.float32)

        self._all_lor_endpoints_module_number = self.xp.asarray(
            [
                i
                for i, module in enumerate(self._modules)
                for _ in range(module.num_lor_endpoints)
            ],
            device=self.dev,
        )

    @property
    def modules(self) -> Sequence[PETScannerModule]:
        """sequence of modules defining the scanner"""
        return self._modules

    @property
    def num_modules(self) -> int:
        """the number of modules defining the scanner"""
        return self._num_modules

    @property
    def num_lor_endpoints_per_module(self) -> Array:
        """array showing how many LOR endpoints are in every module"""
        return self._num_lor_endpoints_per_module

    @property
    def num_lor_endpoints(self) -> int:
        """the total number of LOR endpoints in the scanner"""
        return self._num_lor_endpoints

    @property
    def all_lor_endpoints_index_offset(self) -> Array:
        """the offset in the linear (flattend) index for all LOR endpoints"""
        return self._all_lor_endpoints_index_offset

    @property
    def all_lor_endpoints_module_number(self) -> Array:
        """the module number of all LOR endpoints"""
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints(self) -> Array:
        """the world coordinates of all LOR endpoints"""
        return self._all_lor_endpoints

    @property
    def xp(self) -> ModuleType:
        """Array module of the first module.

        All modules in the scanner must share the same array namespace;
        this property returns the namespace of the first one as representative.
        """
        return self._modules[0].xp

    @property
    def dev(self) -> str:
        """Device of the first module.

        All modules in the scanner must reside on the same device;
        this property returns the device of the first one as representative.
        """
        return self._modules[0].dev

    def linear_lor_endpoint_index(
        self,
        module: Array,
        index_in_module: Array,
    ) -> Array:
        """transform the module + index_in_modules indices into a flattened / linear LOR endpoint index

        Parameters
        ----------
        module : Array
            containing module numbers
        index_in_module : Array
            containing index in modules

        Returns
        -------
        Array
            the flattened LOR endpoint index
        """

        return (
            self.xp.take(self.all_lor_endpoints_index_offset, module, axis=0)
            + index_in_module
        )

    def get_lor_endpoints(self, module: Array, index_in_module: Array) -> Array:
        """get the coordinates for LOR endpoints defined by module and index in module

        Parameters
        ----------
        module : Array
            the module number of the LOR endpoints
        index_in_module : Array
            the index in module number of the LOR endpoints

        Returns
        -------
        Array
            the 3 world coordinates of the LOR endpoints
        """
        return self.xp.take(
            self.all_lor_endpoints,
            self.linear_lor_endpoint_index(module, index_in_module),
            axis=0,
        )

    def show_lor_endpoints(
        self, ax: Axes3D, show_linear_index: bool = True, **kwargs
    ) -> None:
        """show all LOR endpoints in a 3D plot

        Parameters
        ----------
        ax : Axes3D
            a 3D matplotlib axes
        show_linear_index : bool, optional
            annotate the LOR endpoints with the linear LOR endpoint index
        **kwargs : keyword arguments
            passed to show_lor_endpoints() of the scanner module
        """
        for i, module in enumerate(self.modules):
            if show_linear_index:
                offset = int(to_numpy_array(self.all_lor_endpoints_index_offset[i]))
                prefix = ""
            else:
                offset = 0
                prefix = f"{i},"

            module.show_lor_endpoints(
                ax, annotation_offset=offset, annotation_prefix=prefix, **kwargs
            )


class RegularPolygonPETScannerGeometry(ModularizedPETScannerGeometry):
    """A cylindrical PET scanner built from stacked regular-polygon rings.

    Each axial ring is a :class:`RegularPolygonPETScannerModule` with
    ``num_sides`` sides and ``num_lor_endpoints_per_side`` crystals per side,
    giving ``num_lor_endpoints_per_ring = num_sides * num_lor_endpoints_per_side``
    endpoints per ring.  Rings are stacked axially; the global flat endpoint
    index increases first within a ring and then across rings, so endpoint
    ``r * num_lor_endpoints_per_ring + k`` belongs to ring ``r`` and in-ring
    position ``k``.
    """

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        radius: float,
        num_sides: int,
        num_lor_endpoints_per_side: int | None = None,
        lor_spacing: float | None = None,
        ring_positions: Array = None,
        symmetry_axis: int = None,
        phis: None | Array = None,
        ring_endpoint_ordering: RingEndpointOrdering = RingEndpointOrdering.CLOCKWISE,
        phi0: float = 0.0,
        lor_endpoint_positions: np.ndarray | None = None,
    ) -> None:
        """
        Parameters
        ----------
        xp: ModuleType
            array module to use for storing the LOR endpoints
        dev: str
            device to use for storing the LOR endpoints
        radius : float
            inner radius of the regular polygon (distance from centre to detector face) in mm
        num_sides : int
            number of sides (faces) of each regular polygon
        num_lor_endpoints_per_side : int or None, optional
            number of LOR endpoints in each side.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        lor_spacing : float or None, optional
            uniform spacing between LOR endpoints in mm.  Required when
            ``lor_endpoint_positions`` is not given; ignored otherwise.
        ring_positions : Array
            1D array with the coordinate of the rings along the ring axis
        symmetry_axis : int
            the ring axis (0,1,2)
        phis : None | Array, optional
            angle of each side, by default None
            means that the sides are equally spaced around a circle
        ring_endpoint_ordering : RingEndpointOrdering, optional
            direction in which endpoint indices increase around the ring, by
            default ``RingEndpointOrdering.CLOCKWISE``.
        phi0 : float, optional
            azimuthal offset of side 0 in radians, by default 0.
            Only applied when ``phis`` is ``None``; ignored when ``phis`` is
            provided explicitly.
        lor_endpoint_positions : np.ndarray or None, optional
            Custom 1-D array of crystal positions along each polygon side in mm.
            When given, overrides ``num_lor_endpoints_per_side`` and
            ``lor_spacing``.  See :class:`RegularPolygonPETScannerModule` for
            details and anti-symmetry requirements.
        """

        self._radius = radius
        self._num_sides = num_sides
        self._symmetry_axis = symmetry_axis
        self._ring_positions = ring_positions
        self._ring_endpoint_ordering = ring_endpoint_ordering
        self._phi0 = phi0

        # Resolve positions: custom or uniform
        if lor_endpoint_positions is not None:
            _positions = np.asarray(lor_endpoint_positions, dtype=np.float32)
            self._num_lor_endpoints_per_side = len(_positions)
            self._lor_spacing = None
        elif num_lor_endpoints_per_side is not None and lor_spacing is not None:
            N = num_lor_endpoints_per_side
            _positions = lor_spacing * (
                np.arange(N, dtype=np.float32) - (N - 1) / 2.0
            )
            self._num_lor_endpoints_per_side = N
            self._lor_spacing = lor_spacing
        else:
            raise ValueError(
                "Provide either lor_endpoint_positions or both "
                "num_lor_endpoints_per_side and lor_spacing."
            )

        if symmetry_axis == 0:
            self._ax0 = 2
            self._ax1 = 1
        elif symmetry_axis == 1:
            self._ax0 = 0
            self._ax1 = 2
        elif symmetry_axis == 2:
            self._ax0 = 1
            self._ax1 = 0
        else:
            raise ValueError("symmetry_axis must be 0, 1, or 2")

        modules = []

        for ring in range(self.num_rings):
            aff_mat = xp.eye(4, device=dev, dtype=xp.float32)
            aff_mat[symmetry_axis, -1] = xp.astype(ring_positions[ring], xp.float32)

            modules.append(
                RegularPolygonPETScannerModule(
                    xp,
                    dev,
                    radius,
                    num_sides,
                    affine_transformation_matrix=aff_mat,
                    ax0=self._ax0,
                    ax1=self._ax1,
                    phis=phis,
                    ring_endpoint_ordering=ring_endpoint_ordering,
                    phi0=phi0,
                    lor_endpoint_positions=_positions,
                )
            )

        super().__init__(tuple(modules))

        self._all_lor_endpoints_index_in_ring = (
            self.xp.arange(self.num_lor_endpoints, device=dev)
            - self.all_lor_endpoints_ring_number * self.num_lor_endpoints_per_module[0]
        )

    @property
    def radius(self) -> float:
        """Inner radius of the regular polygon (distance from centre to detector face) in mm."""
        return self._radius

    @property
    def num_sides(self) -> int:
        """number of sides (faces) of each polygon"""
        return self._num_sides

    @property
    def num_lor_endpoints_per_side(self) -> int:
        """number of LOR endpoints per side (face) in each polygon"""
        return self._num_lor_endpoints_per_side

    @property
    def num_rings(self) -> int:
        """number of rings (regular polygons)"""
        return self._ring_positions.shape[0]

    @property
    def lor_spacing(self) -> float | None:
        """Uniform spacing between LOR endpoints in mm, or ``None`` when
        custom ``lor_endpoint_positions`` were supplied."""
        return self._lor_spacing

    @property
    def lor_endpoint_positions(self) -> Array:
        """1-D float32 Array of crystal positions along each polygon side (mm),
        on the same device as the scanner."""
        return self.modules[0].lor_endpoint_positions

    @property
    def symmetry_axis(self) -> int:
        """The symmetry axis. Also called axial (or ring) direction."""
        return self._symmetry_axis

    @property
    def ring_endpoint_ordering(self) -> RingEndpointOrdering:
        """direction in which endpoint indices increase around the ring"""
        return self._ring_endpoint_ordering

    @property
    def phi0(self) -> float:
        """azimuthal offset of side 0 in radians (only applied when phis=None)"""
        return self._phi0

    @property
    def all_lor_endpoints_ring_number(self) -> Array:
        """Ring (axial module) index for every LOR endpoint.

        For a regular-polygon scanner each axial ring is one module, so this
        is an alias for ``all_lor_endpoints_module_number``.  Values range
        from ``0`` to ``num_rings - 1``.
        """
        return self._all_lor_endpoints_module_number

    @property
    def all_lor_endpoints_index_in_ring(self) -> Array:
        """the index within the ring (regular polygon) of all LOR endpoints"""
        return self._all_lor_endpoints_index_in_ring

    @property
    def num_lor_endpoints_per_ring(self) -> int:
        """the number of LOR endpoints per ring (regular polygon)"""
        return int(self._num_lor_endpoints_per_module[0])

    @property
    def ring_positions(self) -> Array:
        """the ring (regular polygon) positions"""
        return self._ring_positions


class DemoPETScannerGeometry(RegularPolygonPETScannerGeometry):
    """Demo PET scanner geometry consisting of a 34-ogon with 16 LOR endpoints per side and 36 rings"""

    def __init__(
        self,
        xp: ModuleType,
        dev: str,
        radius: float = 0.5 * (744.1 + 2 * 8.51),
        num_sides: int = 34,
        num_lor_endpoints_per_side: int = 16,
        lor_spacing: float = 4.03125,
        num_rings: int = 36,
        symmetry_axis: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        xp : ModuleType
            array module
        dev : str
            the device to use
        radius : float, optional
            radius of the regular polygon, by default 0.5*(744.1 + 2 * 8.51)
        num_sides : int, optional
            number of sides of the polygon, by default 34
        num_lor_endpoints_per_side : int, optional
            number of LOR endpoints per side, by default 16
        lor_spacing : float, optional
            spacing between the LOR endpoints, by default 4.03125
        num_rings : int, optional
            number of rings, by default 36
        symmetry_axis : int, optional
            symmetry (axial) axis of the scanner, by default 2
        """

        ring_positions = (
            5.32 * xp.arange(num_rings, device=dev, dtype=xp.float32)
            + (xp.astype(xp.arange(num_rings, device=dev) // 9, xp.float32)) * 2.8
        )
        ring_positions = ring_positions - 0.5 * xp.max(ring_positions)

        super().__init__(
            xp,
            dev,
            radius=radius,
            num_sides=num_sides,
            num_lor_endpoints_per_side=num_lor_endpoints_per_side,
            lor_spacing=lor_spacing,
            ring_positions=ring_positions,
            symmetry_axis=symmetry_axis,
        )
