"""Forward and back projectors for PET sinograms, histograms and listmode.

Array-API-compatible :class:`~parallelproj.operators.LinearOperator` subclasses
that call the compiled ``parallelproj_core`` backend for Joseph-based ray
tracing.  Covers 2-D parallel-view projection, regular-polygon PET in sinogram
and listmode mode (non-TOF and TOF), and equal-block PET geometries.
"""

from __future__ import annotations

from typing import Any
from types import ModuleType

import numpy as np
import array_api_compat
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from array_api_compat import device, get_namespace

import parallelproj_core

from ._backend import Array, to_numpy_array, empty_cuda_cache
from .operators import LinearOperator
from .pet_lors import Michelogram, RegularPolygonPETLORDescriptor, EqualBlockPETLORDescriptor
from .tof import TOFParameters


class ParallelViewProjector2D(LinearOperator):
    """2D non-TOF parallel view projector"""

    def __init__(
        self,
        image_shape: tuple[int, int],
        radial_positions: Array,
        view_angles: Array,
        radius: float,
        image_origin: tuple[float, float],
        voxel_size: tuple[float, float],
    ) -> None:
        """Set up a 2D parallel-beam projector using Joseph's ray-tracing method.

        LOR start and end points are computed from the scanner radius and the
        supplied radial positions and view angles.  The image is treated as a
        2D slice; internally a unit axial dimension is prepended so the 3D
        ``parallelproj_core`` kernels can be used directly.

        Parameters
        ----------
        image_shape : tuple[int, int]
            Shape of the 2D input image ``(n1, n2)``.
        radial_positions : Array
            Radial positions of the detector bins in world coordinates (mm).
        view_angles : Array
            View angles in radians, one per projection view.
        radius : float
            Scanner radius in mm (distance from centre to detector).
        image_origin : tuple[float, float]
            World coordinates of the ``[0, 0]`` voxel centre in mm ``(o1, o2)``.
            Internally promoted to ``(0, o1, o2)`` so the 3D backend can be used.
        voxel_size : tuple[float, float]
            Voxel size ``(d1, d2)`` in mm.
            Internally promoted to ``(1, d1, d2)`` with a unit axial dimension.
        """
        super().__init__()
        self._xp = array_api_compat.get_namespace(radial_positions)

        self._radial_positions = radial_positions
        self._device = array_api_compat.device(radial_positions)

        self._image_shape = image_shape
        self._image_origin = array_api_compat.to_device(
            self.xp.asarray((0,) + image_origin, dtype=self.xp.float32), self._device
        )
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray((1,) + voxel_size, dtype=self.xp.float32), self._device
        )

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]

        self._num_rad = radial_positions.shape[0]

        self._radius = radius

        self._xstart = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 3), dtype=self.xp.float32),
            self._device,
        )
        self._xend = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 3), dtype=self.xp.float32),
            self._device,
        )

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            self._xstart[:, i, 1] = (
                self._xp.cos(phi) * self._radial_positions
                + self._xp.sin(phi) * self._radius
            )
            self._xstart[:, i, 2] = (
                -self._xp.sin(phi) * self._radial_positions
                + self._xp.cos(phi) * self._radius
            )
            # world coordinates of LOR endpoints
            self._xend[:, i, 1] = (
                self._xp.cos(phi) * self._radial_positions
                - self._xp.sin(phi) * self._radius
            )
            self._xend[:, i, 2] = (
                -self._xp.sin(phi) * self._radial_positions
                - self._xp.cos(phi) * self._radius
            )

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._xp

    @property
    def in_shape(self) -> tuple[int, ...]:
        """Image shape ``(n1, n2)``."""
        return self._image_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Projection shape ``(num_rad, num_views)``."""
        return (self._num_rad, self._num_views)

    @property
    def num_views(self) -> int:
        """number of views"""
        return self._num_views

    @property
    def num_rad(self) -> int:
        """number of radial elements"""
        return self._num_rad

    @property
    def xstart(self) -> Array:
        """coordinates of LOR start points"""
        return self._xstart

    @property
    def xend(self) -> Array:
        """coordinates of LOR end points"""
        return self._xend

    @property
    def image_origin(self) -> Array:
        """World coordinates of the ``[0, 0]`` voxel, shape ``(3,)``.

        The first element is always ``0`` (the prepended axial dimension used
        internally by the 3D backend); elements 1 and 2 correspond to the
        ``image_origin`` passed at construction.
        """
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int]:
        """image shape"""
        return self._image_shape

    @property
    def voxel_size(self) -> Array:
        """Voxel size, shape ``(3,)``.

        The first element is always ``1`` (the unit axial dimension used
        internally by the 3D backend); elements 1 and 2 correspond to the
        ``voxel_size`` passed at construction.
        """
        return self._voxel_size

    @property
    def dev(self) -> str:
        """device used for storage of LOR endpoints"""
        return self._device

    def _apply(self, x: Array) -> Array:
        y = self.xp.zeros(self.out_shape, dtype=self.xp.float32, device=self._device)
        parallelproj_core.joseph3d_fwd(
            self._xstart,
            self._xend,
            self.xp.expand_dims(x, axis=0),
            self._image_origin,
            self._voxel_size,
            y,
        )
        return y

    def _adjoint(self, y: Array) -> Array:
        x = self.xp.zeros(
            (1,) + self._image_shape, dtype=self.xp.float32, device=self._device
        )

        parallelproj_core.joseph3d_back(
            self._xstart,
            self._xend,
            x,
            self._image_origin,
            self._voxel_size,
            y,
        )
        return self.xp.squeeze(x, axis=0)

    def show_views(
        self,
        views_to_show: None | Array = None,
        image: None | Array = None,
        **kwargs: Any,
    ) -> Figure:
        """Visualize the geometry of selected projection views.

        Parameters
        ----------
        views_to_show : None | Array
            view numbers to show
        image : None | Array
            show an image inside the projector geometry
        **kwargs : dict
            passed to matplotlib.pyplot.imshow

        Returns
        -------
        plt.Figure
            the matplotlib figure
        """
        if views_to_show is None:
            views_to_show = np.linspace(  # type: ignore[assignment]
                0, self._num_views - 1, 5
            ).astype(int)

        assert views_to_show is not None

        num_cols = len(views_to_show)
        fig, ax = plt.subplots(1, num_cols, figsize=(num_cols * 3, 3))

        tmp1 = float(self._image_origin[1] - 0.5 * self._voxel_size[1])
        tmp2 = float(self._image_origin[2] - 0.5 * self._voxel_size[2])
        img_extent = [tmp1, -tmp1, tmp2, -tmp2]

        for i, ip in enumerate(views_to_show):
            ax[i].plot(
                to_numpy_array(self._xstart[:, ip, 1]),
                to_numpy_array(self._xstart[:, ip, 2]),
                ".",
                ms=0.5,
            )
            ax[i].plot(
                to_numpy_array(self._xend[:, ip, 1]),
                to_numpy_array(self._xend[:, ip, 2]),
                ".",
                ms=0.5,
            )

            for k in np.linspace(0, self._num_rad - 1, 7).astype(int):
                ax[i].plot(
                    [float(self._xstart[k, ip, 1]), float(self._xend[k, ip, 1])],
                    [float(self._xstart[k, ip, 2]), float(self._xend[k, ip, 2])],
                    "k-",
                    lw=0.5,
                )
                ax[i].annotate(
                    f"{k}",
                    (float(self._xstart[k, ip, 1]), float(self._xstart[k, ip, 2])),
                    fontsize="xx-small",
                )

            pmax = 1.5 * float(self.xp.max(self._xstart[..., 1]))
            ax[i].set_xlim(-pmax, pmax)
            ax[i].set_ylim(-pmax, pmax)
            ax[i].grid(ls=":")
            ax[i].set_aspect("equal")

            if image is not None:
                ax[i].add_patch(
                    Rectangle(
                        (tmp1, tmp2),
                        float(self.in_shape[0] * self._voxel_size[1]),
                        float(self.in_shape[1] * self._voxel_size[2]),
                        edgecolor="r",
                        facecolor="none",
                        linestyle=":",
                    )
                )

                ax[i].imshow(
                    to_numpy_array(image).T,
                    origin="lower",
                    extent=img_extent,
                    **kwargs,
                )

            ax[i].set_title(
                f"view {ip:03} - phi {(180/np.pi)*self._view_angles[ip]} deg",
                fontsize="small",
            )

        fig.tight_layout()

        return fig


class ParallelViewProjector3D(LinearOperator):
    """3D non-TOF parallel view projector supporting any odd span."""

    def __init__(
        self,
        image_shape: tuple[int, int, int],
        radial_positions: Array,
        view_angles: Array,
        radius: float,
        image_origin: tuple[float, float, float],
        voxel_size: tuple[float, float, float],
        ring_positions: Array,
        michelogram: Michelogram,
    ) -> None:
        """Set up a 3D parallel-beam projector using Joseph's ray-tracing method.

        Extends :class:`ParallelViewProjector2D` to 3D by adding axial rings
        with support for any odd sinogram span via a :class:`.Michelogram`.
        Each sinogram plane's axial LOR position is determined by the
        average z-coordinate of the ring pairs contributing to that plane
        (exact for span=1; the standard averaged-LOR approximation for span>1).

        Parameters
        ----------
        image_shape : tuple[int, int, int]
            Shape of the 3D input image ``(n0, n1, n2)`` where ``n2`` is axial.
        radial_positions : Array
            Radial positions of the detector bins in world coordinates (mm).
        view_angles : Array
            View angles in radians, one per projection view.
        radius : float
            Scanner radius in mm.
        image_origin : tuple[float, float, float]
            World coordinates of the ``[0, 0, 0]`` voxel centre in mm.
        voxel_size : tuple[float, float, float]
            Voxel size ``(d0, d1, d_axial)`` in mm.
        ring_positions : Array
            Axial positions of the detector rings in world coordinates (mm).
            Must have length equal to ``michelogram.num_rings``.
        michelogram : Michelogram
            Axial plane layout encoding the span, max ring difference, and
            ring-pair grouping.  Use :class:`.Michelogram` with ``span=1``
            for uncompressed data or any odd ``span`` for compressed data.
        """

        super().__init__()

        self._xp = array_api_compat.get_namespace(radial_positions)
        self._radial_positions = radial_positions
        self._device = array_api_compat.device(radial_positions)

        self._image_shape = image_shape
        self._image_origin = array_api_compat.to_device(
            self.xp.asarray(image_origin, dtype=self.xp.float32), self._device
        )
        self._voxel_size = array_api_compat.to_device(
            self.xp.asarray(voxel_size, dtype=self.xp.float32), self._device
        )

        self._view_angles = view_angles
        self._num_views = self._view_angles.shape[0]
        self._num_rad = radial_positions.shape[0]
        self._radius = radius

        num_rings = ring_positions.shape[0]
        if michelogram.num_rings != num_rings:
            raise ValueError(
                f"michelogram.num_rings ({michelogram.num_rings}) must match "
                f"len(ring_positions) ({num_rings})"
            )
        self._michelogram = michelogram

        xstart2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2), dtype=self.xp.float32),
            self._device,
        )
        xend2d = array_api_compat.to_device(
            self.xp.zeros((self._num_rad, self._num_views, 2), dtype=self.xp.float32),
            self._device,
        )

        for i, phi in enumerate(self._view_angles):
            # world coordinates of LOR start points
            xstart2d[:, i, 0] = (
                self._xp.cos(phi) * self._radial_positions
                + self._xp.sin(phi) * self._radius
            )
            xstart2d[:, i, 1] = (
                -self._xp.sin(phi) * self._radial_positions
                + self._xp.cos(phi) * self._radius
            )
            # world coordinates of LOR endpoints
            xend2d[:, i, 0] = (
                self._xp.cos(phi) * self._radial_positions
                - self._xp.sin(phi) * self._radius
            )
            xend2d[:, i, 1] = (
                -self._xp.sin(phi) * self._radial_positions
                - self._xp.cos(phi) * self._radius
            )

        # Average z-coordinate per plane (exact for span=1, averaged-LOR for span>1)
        start_z, end_z = michelogram.average_z_per_plane(to_numpy_array(ring_positions))
        self._num_planes = michelogram.num_planes

        self._xstart = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32,
            ),
            self._device,
        )
        self._xend = array_api_compat.to_device(
            self._xp.zeros(
                (self._num_rad, self._num_views, self._num_planes, 3),
                dtype=self._xp.float32,
            ),
            self._device,
        )

        for i in range(self._num_planes):
            self._xstart[:, :, i, :2] = xstart2d
            self._xend[:, :, i, :2] = xend2d
            self._xstart[:, :, i, 2] = float(start_z[i])
            self._xend[:, :, i, 2] = float(end_z[i])

    @property
    def michelogram(self) -> Michelogram:
        """the Michelogram defining the axial plane layout"""
        return self._michelogram

    @property
    def max_ring_diff(self) -> int:
        """maximum ring difference"""
        return self._michelogram.max_ring_difference

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._xp

    @property
    def in_shape(self) -> tuple[int, int, int]:
        """Image shape ``(n0, n1, n2)`` where ``n2`` is axial."""
        return self._image_shape

    @property
    def out_shape(self) -> tuple[int, int, int]:
        """Sinogram shape ``(num_rad, num_views, num_planes)``."""
        return (self._num_rad, self._num_views, self._num_planes)

    @property
    def voxel_size(self) -> Array:
        """the voxel size in all directions"""
        return self._voxel_size

    @property
    def image_origin(self) -> Array:
        """image origin - world coordinates of the [0,0,0] voxel"""
        return self._image_origin

    @property
    def image_shape(self) -> tuple[int, int, int]:
        """image shape"""
        return self._image_shape

    @property
    def xstart(self) -> Array:
        """coordinates of LOR start points"""
        return self._xstart

    @property
    def xend(self) -> Array:
        """coordinates of LOR end points"""
        return self._xend

    def _apply(self, x: Array) -> Array:
        y = self.xp.zeros(self.out_shape, dtype=self.xp.float32, device=self._device)
        parallelproj_core.joseph3d_fwd(
            self._xstart, self._xend, x, self.image_origin, self.voxel_size, y
        )
        return y

    def _adjoint(self, y: Array) -> Array:
        x = self.xp.zeros(self._image_shape, dtype=self.xp.float32, device=self._device)
        parallelproj_core.joseph3d_back(
            self._xstart,
            self._xend,
            x,
            self.image_origin,
            self.voxel_size,
            y,
        )
        return x


class RegularPolygonPETProjector(LinearOperator):
    """geometric non-TOF and TOF sinogram projector for regular polygon PET scanners"""

    def __init__(
        self,
        lor_descriptor: RegularPolygonPETLORDescriptor,
        img_shape: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        img_origin: None | Array = None,
        views: None | Array = None,
        cache_lor_endpoints: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        lor_descriptor : RegularPolygonPETLORDescriptor
            descriptor of the LOR start / end points
        img_shape : tuple[int, int, int]
            shape of the image to be projected
        voxel_size : tuple[float, float, float]
            the voxel size of the image to be projected
        img_origin : None | Array, optional
            the origin of the image to be projected, by default None
            means that the center of the image is at world coordinate (0,0,0)
        views : None | Array, optional
            sinogram views to be projected, by default None
            means that all views are being projected
        cache_lor_endpoints : bool, optional
            whether to cache the LOR endpoints, by default True
            setting it to False will save memory but will slow down computations
        """

        super().__init__()
        self._dev = lor_descriptor.dev

        self._lor_descriptor = lor_descriptor
        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(
            voxel_size, dtype=self.xp.float32, device=self._dev
        )

        if img_origin is None:
            self._img_origin = (
                -(
                    self.xp.asarray(
                        self._img_shape, dtype=self.xp.float32, device=self._dev
                    )
                    / 2
                )
                + 0.5
            ) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(
                img_origin, dtype=self.xp.float32, device=self._dev
            )

        if views is None:
            self._views = self.xp.arange(
                self._lor_descriptor.num_views, device=self._dev
            )
        else:
            self._views = views

        self._tof_parameters = None
        self._tof = False

        self._cache_lor_endpoints = cache_lor_endpoints

        self._xstart = None
        self._xend = None

        self._out_shape = self._compute_out_shape()

    @property
    def in_shape(self) -> tuple[int, int, int]:
        """Image shape ``(n0, n1, n2)``."""
        return self._img_shape

    def _compute_out_shape(self) -> tuple[int, ...]:
        out_shape = list(self._lor_descriptor.spatial_sinogram_shape)
        out_shape[self._lor_descriptor.view_axis_num] = self._views.shape[0]

        if self._tof and self._tof_parameters is not None:
            out_shape += [self._tof_parameters.num_tofbins]

        return tuple(out_shape)

    @property
    def out_shape(self) -> tuple[int, ...]:
        """Sinogram shape respecting ``sinogram_order``, optionally with a trailing TOF axis."""
        return self._out_shape

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._lor_descriptor.xp

    @property
    def tof(self) -> bool:
        """Enable or disable TOF mode.

        Setting to ``True`` requires ``tof_parameters`` to be set first;
        raises ``ValueError`` otherwise.  Setting to ``False`` always
        succeeds and is a no-op when TOF is already disabled.
        """
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        if value and self.tof_parameters is None:
            raise ValueError("tof_parameters must not be None")
        self._tof = value

    @property
    def tof_parameters(self) -> TOFParameters | None:
        """TOF kernel parameters, or ``None`` for non-TOF mode.

        Assigning a :class:`.TOFParameters` instance automatically enables TOF
        and recomputes ``out_shape``.  Assigning ``None`` disables TOF.
        """
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters | None) -> None:
        if not (isinstance(value, TOFParameters) or value is None):
            raise ValueError("tof_parameters must be a TOFParameters object or None")
        self._tof_parameters = value

        if value is None:
            self._tof = False
        else:
            self._tof = True

        self._out_shape = self._compute_out_shape()

    @property
    def lor_descriptor(self) -> RegularPolygonPETLORDescriptor:
        """LOR descriptor"""
        return self._lor_descriptor

    @property
    def img_origin(self) -> Array:
        """image origin - world coordinates of the [0,0,0] voxel"""
        return self._img_origin

    @property
    def views(self) -> Array:
        """View indices to project.

        Assigning a new array recomputes ``out_shape`` and clears any cached
        LOR endpoint arrays.
        """
        return self._views

    @views.setter
    def views(self, value: Array) -> None:
        self._views = value
        self._out_shape = self._compute_out_shape()
        # we need to reset the LOR start and end points in case
        # they were cached
        self.clear_cached_lor_endpoints()

    @property
    def xstart(self) -> Array | None:
        """cached coordinates of LOR start points"""
        return self._xstart

    @property
    def xend(self) -> Array | None:
        """cached coordinates of LOR end points"""
        return self._xend

    @property
    def voxel_size(self) -> Array:
        """voxel size"""
        return self._voxel_size

    def clear_cached_lor_endpoints(self) -> None:
        """clear cached LOR endpoints"""
        self._xstart = None
        self._xend = None

        empty_cuda_cache(self.xp)

    def fov_mask(self) -> Array:
        """Boolean cylindrical FOV mask for this projector's image grid.

        The cylinder radius equals the transaxial distance of the midpoint
        of the first LOR of the first sinogram view from the scanner
        isocenter.  The cylinder axis is aligned with the scanner's
        symmetry axis.

        Returns
        -------
        Array of bool, shape ``in_shape``
            ``True`` inside the cylindrical FOV, ``False`` outside.
        """
        xp = self.xp
        dev = self._dev
        sym_ax = self.lor_descriptor.scanner.symmetry_axis
        ax0, ax1 = [ax for ax in range(3) if ax != sym_ax]

        # midpoint of the first LOR of the first view in world coordinates
        xstart, xend = self.lor_descriptor.get_lor_coordinates(
            views=xp.asarray([0], device=dev)
        )
        mid = (xstart.reshape(-1, 3)[0] + xend.reshape(-1, 3)[0]) / 2
        lor_radius = float(xp.sqrt(mid[ax0] ** 2 + mid[ax1] ** 2))

        # voxel centre coordinates along the two transaxial axes
        c0 = (
            self._img_origin[ax0]
            + xp.arange(self._img_shape[ax0], device=dev, dtype=xp.float32)
            * self._voxel_size[ax0]
        )
        c1 = (
            self._img_origin[ax1]
            + xp.arange(self._img_shape[ax1], device=dev, dtype=xp.float32)
            * self._voxel_size[ax1]
        )

        # transaxial distance from isocenter, shape (n_ax0, n_ax1)
        r = xp.sqrt(c0[:, None] ** 2 + c1[None, :] ** 2)

        # expand along symmetry axis and broadcast to full image shape
        return (
            xp.broadcast_to(xp.expand_dims(r, axis=sym_ax), self._img_shape)
            <= lor_radius
        )

    def __str__(self) -> str:
        """string representation"""

        st = (
            self.__class__.__name__
            + " with sinogram shape ("
            + ", ".join(
                [
                    f"{self.lor_descriptor.spatial_sinogram_shape[i]}"
                    f" {self.lor_descriptor.sinogram_order.name[i]}"
                    for i in range(3)
                ]
            )
        )

        if self.tof and self._tof_parameters is not None:
            st += f", {self._tof_parameters.num_tofbins} TOF bins"

        st += ")"

        return st

    def _apply(self, x: Array) -> Array:
        """Forward projection of input image x (non-TOF and TOF), including image-based resolution model."""

        dev = array_api_compat.device(x)

        # calculate LOR endpoints if not done yet
        needs_compute = (self.xstart is None) or (self.xend is None)
        if needs_compute:
            xstart, xend = self._lor_descriptor.get_lor_coordinates(views=self._views)
            empty_cuda_cache(self.xp)
        else:
            xstart = self.xstart
            xend = self.xend
            assert xstart is not None
            assert xend is not None

        # cache LOR endpoints if requested
        if self._cache_lor_endpoints and needs_compute:
            self._xstart = xstart
            self._xend = xend

        x_fwd = self.xp.zeros(self.out_shape, dtype=self.xp.float32, device=dev)

        if not self.tof:
            parallelproj_core.joseph3d_fwd(
                xstart, xend, x, self._img_origin, self._voxel_size, x_fwd
            )
        else:
            assert self._tof_parameters is not None
            parallelproj_core.joseph3d_tof_sino_fwd(
                xstart,
                xend,
                x,
                self._img_origin,
                self._voxel_size,
                x_fwd,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self._tof_parameters.num_tofbins,
                self._tof_parameters.num_sigmas,
            )

        return x_fwd

    def _adjoint(self, y: Array) -> Array:
        """Back projection of sinogram y (non-TOF and TOF)."""
        dev = array_api_compat.device(y)

        # calculate LOR endpoints if not done yet
        needs_compute = (self.xstart is None) or (self.xend is None)
        if needs_compute:
            xstart, xend = self._lor_descriptor.get_lor_coordinates(views=self._views)
            empty_cuda_cache(self.xp)
        else:
            xstart = self.xstart
            xend = self.xend
            assert xstart is not None
            assert xend is not None

        # cache LOR endpoints if requested
        if self._cache_lor_endpoints and needs_compute:
            self._xstart = xstart
            self._xend = xend

        y_back = self.xp.zeros(self.in_shape, dtype=self.xp.float32, device=dev)

        if not self.tof:
            parallelproj_core.joseph3d_back(
                xstart,
                xend,
                y_back,
                self._img_origin,
                self._voxel_size,
                y,
            )
        else:
            assert self._tof_parameters is not None
            parallelproj_core.joseph3d_tof_sino_back(
                xstart,
                xend,
                y_back,
                self._img_origin,
                self._voxel_size,
                y,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof], dtype=self.xp.float32, device=dev
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self._tof_parameters.num_tofbins,
                self._tof_parameters.num_sigmas,
            )

        return y_back

    def show_geometry(
        self,
        ax: Axes3D,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        edgecolor: str = "grey",
        alpha: float = 0.1,
    ) -> None:
        """show the geometry of the scanner and the FOV of the image

        Parameters
        ----------
        ax : Axes3D
            matplotlib axes object with projection = '3d'
        color : tuple[float, float, float], optional
            color to use for the FOV cube, by default (1.,0.,0.)
        edgecolor : str, optional
            edgecolor to use for the FOV cube, by default 'grey'
        alpha : float, optional
            alpha value of the FOV cube, by default 0.1
        """

        # dimensions of the "voxel" array for the FOV cube
        # (1,1,1) means that FOV cube is represented by a single voxel
        sh = (1, 1, 1)

        x, y, z = np.indices((sh[0] + 1, sh[1] + 1, sh[2] + 1)).astype(float)

        x /= sh[0]
        y /= sh[1]
        z /= sh[2]

        x *= int(self.in_shape[0]) * float(self.voxel_size[0])
        y *= int(self.in_shape[1]) * float(self.voxel_size[1])
        z *= int(self.in_shape[2]) * float(self.voxel_size[2])

        x += float(self.img_origin[0]) - 0.5 * float(self.voxel_size[0])
        y += float(self.img_origin[1]) - 0.5 * float(self.voxel_size[1])
        z += float(self.img_origin[2]) - 0.5 * float(self.voxel_size[2])

        colors = np.empty(sh + (4,), dtype=np.float32)
        colors[..., 0] = color[0]
        colors[..., 1] = color[1]
        colors[..., 2] = color[2]
        colors[..., 3] = alpha

        ax.voxels(
            x,
            y,
            z,
            filled=np.ones(sh, dtype=bool),
            facecolors=colors,
            edgecolors=edgecolor,
        )

        self.lor_descriptor.scanner.show_lor_endpoints(ax)

    def convert_sinogram_to_crystal_index_events(
        self, sinogram: Array, shuffle: bool = False
    ) -> np.ndarray:
        """Convert a non-TOF or TOF span-1 sinogram to crystal-index events.

        Each count in the sinogram becomes one row in the output array.
        Non-TOF rows are ``(d_red, r_red, d_blue, r_blue)``; TOF rows add a
        trailing ``tof_bin`` column in the projector convention
        (bin 0 = closest to the canonical xstart crystal).
        The output is ready for direct use with
        :func:`.regular_polygon_events_to_sinogram`.

        The *red* crystal is the canonical xstart of each LOR as defined by
        the LOR descriptor; *blue* is the xend crystal.  Unpack the returned
        array with ``events[:, 0]`` (d_red), ``events[:, 1]`` (r_red), etc.

        Parameters
        ----------
        sinogram : Array
            Integer span-1 sinogram.
            Non-TOF shape: ``lor_descriptor.spatial_sinogram_shape``.
            TOF shape: ``(*lor_descriptor.spatial_sinogram_shape, num_tof_bins)``.
        shuffle : bool, optional
            Randomly shuffle the output rows (default ``False``).
            Uses numpy's global random state; call ``numpy.random.seed``
            before this method for reproducible results.

        Returns
        -------
        events : np.ndarray, shape (N, 4) or (N, 5), dtype int32
            Crystal-index events.  Columns are
            ``(d_red, r_red, d_blue, r_blue)`` or
            ``(d_red, r_red, d_blue, r_blue, tof_bin)``.

        Raises
        ------
        TypeError
            If ``sinogram`` does not have an integer dtype.
        ValueError
            If the LOR descriptor has ``span > 1``; :attr:`start_plane_index`
            is only defined for span-1 descriptors.
        """
        lor_desc = self.lor_descriptor
        if lor_desc.michelogram.span != 1:
            raise ValueError(
                "convert_sinogram_to_crystal_index_events requires a span-1 LOR descriptor"
            )

        integer_dtypes = (
            self.xp.int8,
            self.xp.int16,
            self.xp.int32,
            self.xp.int64,
            self.xp.uint8,
            self.xp.uint16,
            self.xp.uint32,
            self.xp.uint64,
        )
        if sinogram.dtype not in integer_dtypes:
            raise TypeError(
                f"sinogram must have an integer dtype, got {sinogram.dtype}"
            )

        sc = to_numpy_array(lor_desc.start_in_ring_index)  # (num_views, num_rad)
        ec = to_numpy_array(lor_desc.end_in_ring_index)
        sr = to_numpy_array(lor_desc.start_plane_index)  # (num_planes,)
        er = to_numpy_array(lor_desc.end_plane_index)

        p_ax = lor_desc.plane_axis_num
        v_ax = lor_desc.view_axis_num
        r_ax = lor_desc.radial_axis_num

        sino_np = to_numpy_array(sinogram).astype(np.int32)
        tof_mode = sino_np.ndim == 4

        valid_vr = sc != ec  # self-pair bins are unphysical; skip them

        # Reorder axes to (view, radial, plane[, tof]) so that np.where
        # yields events in the same view-first, radial-second, plane-third
        # order as the original view-by-view loop in convert_sinogram_to_listmode.
        if tof_mode:
            sino_vrpt = np.transpose(sino_np, (v_ax, r_ax, p_ax, 3))
            counts = sino_vrpt * valid_vr[:, :, None, None]
            v_idx, r_idx, p_idx, t_idx = np.where(counts > 0)
            cnt = counts[v_idx, r_idx, p_idx, t_idx]
            rv = np.repeat(v_idx, cnt)
            rr = np.repeat(r_idx, cnt)
            rp = np.repeat(p_idx, cnt)
            rt = np.repeat(t_idx, cnt)
            events = np.column_stack(
                [sc[rv, rr], sr[rp], ec[rv, rr], er[rp], rt]
            ).astype(np.int32)
        else:
            sino_vrp = np.transpose(sino_np, (v_ax, r_ax, p_ax))
            counts = sino_vrp * valid_vr[:, :, None]
            v_idx, r_idx, p_idx = np.where(counts > 0)
            cnt = counts[v_idx, r_idx, p_idx]
            rv = np.repeat(v_idx, cnt)
            rr = np.repeat(r_idx, cnt)
            rp = np.repeat(p_idx, cnt)
            events = np.column_stack([sc[rv, rr], sr[rp], ec[rv, rr], er[rp]]).astype(
                np.int32
            )

        if shuffle:
            perm = np.random.permutation(len(events))
            events = events[perm]

        return events

    def convert_sinogram_to_listmode(
        self, sinogram: Array, shuffle: bool = False
    ) -> tuple[Array, Array, Array | None]:
        """convert a non-TOF or TOF emission sinogram to listmode events

        Parameters
        ----------
        sinogram : Array
            an integer (TOF or non-TOF) emission sinogram
        shuffle : bool, optional
            if True, randomly shuffle the order of the output events,
            by default False. Shuffling is implemented via
            ``numpy.random.permutation(num_events)``, which draws from
            numpy's global random state. Use ``numpy.random.seed()``
            before calling this method for reproducible results.

        Returns
        -------
        tuple[Array, Array, Array | None]
            event_start_coordinates, event_end_coordinates, event_tofbins
            in case of non-TOF, event_tofbins is None
        """
        events = self.convert_sinogram_to_crystal_index_events(
            sinogram, shuffle=shuffle
        )

        scanner = self.lor_descriptor.scanner
        d1 = self.xp.asarray(events[:, 0].astype(np.int64), device=self._dev)
        r1 = self.xp.asarray(events[:, 1].astype(np.int64), device=self._dev)
        d2 = self.xp.asarray(events[:, 2].astype(np.int64), device=self._dev)
        r2 = self.xp.asarray(events[:, 3].astype(np.int64), device=self._dev)

        event_start_coords = scanner.get_lor_endpoints(r1, d1)
        event_end_coords = scanner.get_lor_endpoints(r2, d2)

        if events.shape[1] == 5:
            event_tofbins = self.xp.asarray(
                events[:, 4].astype(np.int16), device=self._dev
            )
        else:
            event_tofbins = None

        return event_start_coords, event_end_coords, event_tofbins


class ListmodePETProjector(LinearOperator):
    """Non-TOF and TOF listmode projector for regular-polygon PET scanners.

    To enable TOF mode after construction, set the properties in this order:

    1. ``projector.tof_parameters = TOFParameters(...)`` — sets the TOF kernel
       parameters (TOF remains disabled until step 3).
    2. ``projector.event_tofbins = <Array>`` — per-event TOF bin indices.
    3. ``projector.tof = True`` — activates TOF projection.

    Setting ``tof = True`` before both ``tof_parameters`` and
    ``event_tofbins`` are assigned raises ``ValueError``.  Setting
    ``event_tofbins = None`` or ``tof_parameters = None`` automatically
    resets ``tof`` to ``False``.
    """

    def __init__(
        self,
        event_start_coordinates: Array,
        event_end_coordinates: Array,
        img_shape: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        img_origin: None | Array = None,
    ) -> None:
        """
        Parameters
        ----------
        event_start_coordinates : Array
            float world coordinates of event LOR start points, shape (num_events, 3)
        event_end_coordinates : Array
            float world coordinates of event LOR end points, shape (num_events, 3)
        img_shape : tuple[int, int, int]
            shape of the image to be projected
        voxel_size : tuple[float, float, float]
            the voxel size of the image to be projected
        img_origin : None | Array, optional
            the origin of the image to be projected, by default None
            means that the center of the image is at world coordinate (0,0,0)
        """

        super().__init__()

        self._xstart = event_start_coordinates
        self._xend = event_end_coordinates

        self._xp = get_namespace(self._xstart)

        self._dev = device(event_start_coordinates)

        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(
            voxel_size, dtype=self.xp.float32, device=self._dev
        )

        if img_origin is None:
            self._img_origin = (
                -(
                    self.xp.asarray(
                        self._img_shape, dtype=self.xp.float32, device=self._dev
                    )
                    / 2
                )
                + 0.5
            ) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(
                img_origin, dtype=self.xp.float32, device=self._dev
            )

        self._tof_parameters = None
        self._tof = False
        self._tofbin = None

    @property
    def in_shape(self) -> tuple[int, int, int]:
        """Image shape ``(n0, n1, n2)``."""
        return self._img_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        """``(num_events,)`` — one value per detected event."""
        return (self._xstart.shape[0],)

    @property
    def num_events(self) -> int:
        """number of events"""
        return self._xstart.shape[0]

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._xp

    @property
    def tof(self) -> bool:
        """Enable or disable TOF mode.

        Must set ``tof_parameters`` and ``event_tofbins`` before setting to
        ``True``; raises ``ValueError`` otherwise.  Setting ``event_tofbins``
        to ``None`` automatically disables TOF.
        """
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        if (value) and (self.tof_parameters is None):
            raise ValueError("must set tof_parameters first")
        if (value) and (self.event_tofbins is None):
            raise ValueError("must set event_tofbins first")

        self._tof = value

    @property
    def tof_parameters(self) -> TOFParameters | None:
        """TOF kernel parameters, or ``None`` for non-TOF mode.

        Assigning ``None`` disables TOF.
        """
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters | None) -> None:
        if not (isinstance(value, TOFParameters) or value is None):
            raise ValueError("tof_parameters must be a TOFParameters object or None")
        self._tof_parameters = value

        if value is None:
            self._tof = False

    @property
    def event_tofbins(self) -> None | Array:
        """Integer TOF bin index for each event, or ``None`` for non-TOF mode.

        Assigning an array enables per-event TOF binning; its length must
        match ``num_events``.  Assigning ``None`` clears the TOF bins and
        disables TOF.
        """
        return self._tofbin

    @event_tofbins.setter
    def event_tofbins(self, value: None | Array) -> None:
        if value is None:
            self._tofbin = None
            self._tof = False
        else:
            if value.shape[0] != self.num_events:
                raise ValueError(
                    "tofbin must have the same number of elements as events"
                )
            self._tofbin = value

    @property
    def event_start_coordinates(self) -> Array:
        """coordinates of LOR start points"""
        return self._xstart

    @property
    def event_end_coordinates(self) -> Array:
        """coordinates of LOR end points"""
        return self._xend

    @property
    def voxel_size(self) -> Array:
        """voxel size"""
        return self._voxel_size

    def _apply(self, x: Array) -> Array:
        dev = array_api_compat.device(x)

        x_fwd = self.xp.zeros(self.out_shape, dtype=self.xp.float32, device=dev)

        if not self.tof:
            parallelproj_core.joseph3d_fwd(
                self._xstart,
                self._xend,
                x,
                self._img_origin,
                self._voxel_size,
                x_fwd,
            )
        else:
            assert self._tof_parameters is not None
            assert self._tofbin is not None
            parallelproj_core.joseph3d_tof_lm_fwd(
                self._xstart,
                self._xend,
                x,
                self._img_origin,
                self._voxel_size,
                x_fwd,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self._tofbin,
                self._tof_parameters.num_tofbins,
                self._tof_parameters.num_sigmas,
            )

        return x_fwd

    def _adjoint(self, y: Array) -> Array:
        dev = array_api_compat.device(y)
        y_back = self.xp.zeros(self.in_shape, dtype=self.xp.float32, device=dev)

        if not self.tof:
            parallelproj_core.joseph3d_back(
                self._xstart,
                self._xend,
                y_back,
                self._img_origin,
                self._voxel_size,
                y,
            )
        else:
            assert self._tof_parameters is not None
            assert self._tofbin is not None
            parallelproj_core.joseph3d_tof_lm_back(
                self._xstart,
                self._xend,
                y_back,
                self._img_origin,
                self._voxel_size,
                y,
                self._tof_parameters.tofbin_width,
                self.xp.asarray(
                    [self._tof_parameters.sigma_tof], dtype=self.xp.float32, device=dev
                ),
                self.xp.asarray(
                    [self._tof_parameters.tofcenter_offset],
                    dtype=self.xp.float32,
                    device=dev,
                ),
                self._tofbin,
                self._tof_parameters.num_tofbins,
                self.tof_parameters.num_sigmas,
            )

        return y_back


class EqualBlockPETProjector(LinearOperator):
    """geometric non-TOF and TOF sinogram projector for equal block PET scanners"""

    def __init__(
        self,
        lor_descriptor: EqualBlockPETLORDescriptor,
        img_shape: tuple[int, int, int],
        voxel_size: tuple[float, float, float],
        img_origin: None | Array = None,
        num_chunks: int = 1,
    ) -> None:
        """
        Parameters
        ----------
        lor_descriptor : EqualBlockPETLORDescriptor
            descriptor of the LOR start / end points
        img_shape : tuple[int, int, int]
            shape of the image to be projected
        voxel_size : tuple[float, float, float]
            the voxel size of the image to be projected
        img_origin : None | Array, optional
            the origin of the image to be projected, by default None
            means that the center of the image is at world coordinate (0,0,0)
        num_chunks : int, optional
            number of chunks to split the block pairs into during projection,
            by default 1 (all block pairs processed in a single call).
            Increase this value to reduce peak memory usage at the cost of
            more projection kernel calls.
        """

        super().__init__()
        self._dev = lor_descriptor.dev

        self._lor_descriptor = lor_descriptor
        self._img_shape = img_shape
        self._voxel_size = self.xp.asarray(
            voxel_size, dtype=self.xp.float32, device=self._dev
        )

        if img_origin is None:
            self._img_origin = (
                -(
                    self.xp.asarray(
                        self._img_shape, dtype=self.xp.float32, device=self._dev
                    )
                    / 2
                )
                + 0.5
            ) * self._voxel_size
        else:
            self._img_origin = self.xp.asarray(
                img_origin, dtype=self.xp.float32, device=self._dev
            )

        self._tof_parameters = None
        self._tof = False
        self._num_chunks = num_chunks

    @property
    def xp(self) -> ModuleType:
        """array module"""
        return self._lor_descriptor.xp

    @property
    def dev(self) -> str:
        """device"""
        return self._dev

    @property
    def in_shape(self) -> tuple[int, int, int]:
        """Image shape ``(n0, n1, n2)``."""
        return self._img_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        """``(num_block_pairs, num_lors_per_block_pair)`` for non-TOF, with a trailing ``num_tofbins`` axis for TOF."""
        out_shape = [
            self._lor_descriptor.num_block_pairs,
            self._lor_descriptor.num_lors_per_block_pair,
        ]

        if self.tof and self._tof_parameters is not None:
            out_shape += [self._tof_parameters.num_tofbins]

        return tuple(out_shape)

    @property
    def tof(self) -> bool:
        """Enable or disable TOF mode.

        Setting to ``True`` requires ``tof_parameters`` to be set first;
        raises ``ValueError`` otherwise.  Setting to ``False`` always
        succeeds and is a no-op when TOF is already disabled.
        """
        return self._tof

    @tof.setter
    def tof(self, value: bool) -> None:
        if value and self.tof_parameters is None:
            raise ValueError("tof_parameters must not be None")
        self._tof = value

    @property
    def tof_parameters(self) -> TOFParameters | None:
        """TOF kernel parameters, or ``None`` for non-TOF mode.

        Assigning a :class:`.TOFParameters` instance automatically enables TOF.
        Assigning ``None`` disables TOF.
        """
        return self._tof_parameters

    @tof_parameters.setter
    def tof_parameters(self, value: TOFParameters | None) -> None:
        if not (isinstance(value, TOFParameters) or value is None):
            raise ValueError("tof_parameters must be a TOFParameters object or None")
        self._tof_parameters = value

        if value is None:
            self._tof = False
        else:
            self._tof = True

    @property
    def lor_descriptor(self) -> EqualBlockPETLORDescriptor:
        """LOR descriptor"""
        return self._lor_descriptor

    @property
    def img_origin(self) -> Array:
        """image origin - world coordinates of the [0,0,0] voxel"""
        return self._img_origin

    @property
    def voxel_size(self) -> Array:
        """voxel size"""
        return self._voxel_size

    @property
    def num_chunks(self) -> int:
        """Number of chunks to split block pairs into during projection.

        Increasing this reduces peak GPU memory usage at the cost of more
        kernel launches.
        """
        return self._num_chunks

    @num_chunks.setter
    def num_chunks(self, value: int) -> None:
        self._num_chunks = value

    def _apply(self, x: Array) -> Array:
        """forward projection of input image x"""

        dev = array_api_compat.device(x)

        x_fwd = self.xp.zeros(self.out_shape, dtype=self.xp.float32, device=dev)

        num_bp = self.lor_descriptor.num_block_pairs
        chunk_size = -(-num_bp // self._num_chunks)  # ceiling division

        for chunk_start in range(0, num_bp, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_bp)
            bp_chunk = self.xp.arange(chunk_start, chunk_end, device=dev)
            xstart, xend = self.lor_descriptor.get_lor_coordinates(bp_chunk)

            if not self.tof:
                parallelproj_core.joseph3d_fwd(
                    xstart,
                    xend,
                    x,
                    self._img_origin,
                    self._voxel_size,
                    self.xp.reshape(x_fwd[chunk_start:chunk_end, ...], (-1,)),
                )
            else:
                assert self._tof_parameters is not None
                parallelproj_core.joseph3d_tof_sino_fwd(
                    xstart,
                    xend,
                    x,
                    self._img_origin,
                    self._voxel_size,
                    self.xp.reshape(
                        x_fwd[chunk_start:chunk_end, ...],
                        (-1, self._tof_parameters.num_tofbins),
                    ),
                    self._tof_parameters.tofbin_width,
                    self.xp.asarray(
                        [self._tof_parameters.sigma_tof],
                        dtype=self.xp.float32,
                        device=dev,
                    ),
                    self.xp.asarray(
                        [self._tof_parameters.tofcenter_offset],
                        dtype=self.xp.float32,
                        device=dev,
                    ),
                    self.tof_parameters.num_tofbins,
                    self.tof_parameters.num_sigmas,
                )

        return x_fwd

    def _adjoint(self, y: Array) -> Array:
        """back projection of sinogram y"""
        dev = array_api_compat.device(y)

        y_back = self.xp.zeros(self.in_shape, dtype=self.xp.float32, device=dev)

        num_bp = self.lor_descriptor.num_block_pairs
        chunk_size = -(-num_bp // self._num_chunks)  # ceiling division

        for chunk_start in range(0, num_bp, chunk_size):
            chunk_end = min(chunk_start + chunk_size, num_bp)
            bp_chunk = self.xp.arange(chunk_start, chunk_end, device=dev)
            xstart, xend = self.lor_descriptor.get_lor_coordinates(bp_chunk)

            if not self.tof:
                parallelproj_core.joseph3d_back(
                    xstart,
                    xend,
                    y_back,
                    self._img_origin,
                    self._voxel_size,
                    self.xp.reshape(y[chunk_start:chunk_end, ...], (-1,)),
                )
            else:
                assert self._tof_parameters is not None
                parallelproj_core.joseph3d_tof_sino_back(
                    xstart,
                    xend,
                    y_back,
                    self._img_origin,
                    self._voxel_size,
                    self.xp.reshape(
                        y[chunk_start:chunk_end, ...],
                        (-1, self._tof_parameters.num_tofbins),
                    ),
                    self._tof_parameters.tofbin_width,
                    self.xp.asarray(
                        [self._tof_parameters.sigma_tof],
                        dtype=self.xp.float32,
                        device=dev,
                    ),
                    self.xp.asarray(
                        [self._tof_parameters.tofcenter_offset],
                        dtype=self.xp.float32,
                        device=dev,
                    ),
                    self.tof_parameters.num_tofbins,
                    self.tof_parameters.num_sigmas,
                )

        return y_back

    def show_geometry(
        self,
        ax: Axes3D,
        color: tuple[float, float, float] = (1.0, 0.0, 0.0),
        edgecolor: str = "grey",
        alpha: float = 0.1,
    ) -> None:
        """show the geometry of the scanner and the FOV of the image

        Parameters
        ----------
        ax : Axes3D
            matplotlib axes object with projection = '3d'
        color : tuple[float, float, float], optional
            color to use for the FOV cube, by default (1.,0.,0.)
        edgecolor : str, optional
            edgecolor to use for the FOV cube, by default 'grey'
        alpha : float, optional
            alpha value of the FOV cube, by default 0.1
        """

        # dimensions of the "voxel" array for the FOV cube
        # (1,1,1) means that FOV cube is represented by a single voxel
        sh = (1, 1, 1)

        x, y, z = np.indices((sh[0] + 1, sh[1] + 1, sh[2] + 1)).astype(float)

        x /= sh[0]
        y /= sh[1]
        z /= sh[2]

        x *= int(self.in_shape[0]) * float(self.voxel_size[0])
        y *= int(self.in_shape[1]) * float(self.voxel_size[1])
        z *= int(self.in_shape[2]) * float(self.voxel_size[2])

        x += float(self.img_origin[0]) - 0.5 * float(self.voxel_size[0])
        y += float(self.img_origin[1]) - 0.5 * float(self.voxel_size[1])
        z += float(self.img_origin[2]) - 0.5 * float(self.voxel_size[2])

        colors = np.empty(sh + (4,), dtype=np.float32)
        colors[..., 0] = color[0]
        colors[..., 1] = color[1]
        colors[..., 2] = color[2]
        colors[..., 3] = alpha

        ax.voxels(
            x,
            y,
            z,
            filled=np.ones(sh, dtype=bool),
            facecolors=colors,
            edgecolors=edgecolor,
        )

        self.lor_descriptor.scanner.show_lor_endpoints(ax)
