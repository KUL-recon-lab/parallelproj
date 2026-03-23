from __future__ import annotations

from collections.abc import Sequence
import abc
import itertools
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, TextBox

from array_api_compat import device, to_device
from parallelproj import Array
from types import ModuleType


class RadonObject(abc.ABC):
    """abstract base class for objects with known radon transform"""

    def __init__(self, xp: ModuleType, dev: str) -> None:
        self._xp = xp
        self._dev = dev

        self._x0_offset: float = 0.0
        self._x1_offset: float = 0.0
        self._s0: float = 1.0
        self._s1: float = 1.0
        self._amplitude: float = 1.0
        self._rotation: float = 0.0

    @abc.abstractmethod
    def _centered_radon_transform(self, r: Array, phi: Array) -> Array:
        pass

    @abc.abstractmethod
    def _centered_values(self, x0: Array, x1: Array) -> Array:
        pass

    @property
    def xp(self) -> ModuleType:
        return self._xp

    @property
    def dev(self) -> str:
        return self._dev

    @property
    def x0_offset(self) -> float:
        return self._x0_offset

    @x0_offset.setter
    def x0_offset(self, value: float) -> None:
        self._x0_offset = value

    @property
    def x1_offset(self) -> float:
        return self._x1_offset

    @x1_offset.setter
    def x1_offset(self, value: float) -> None:
        self._x1_offset = value

    @property
    def s0(self) -> float:
        return self._s0

    @s0.setter
    def s0(self, value: float) -> None:
        self._s0 = value

    @property
    def s1(self) -> float:
        return self._s1

    @s1.setter
    def s1(self, value: float) -> None:
        self._s1 = value

    @property
    def amplitude(self) -> float:
        return self._amplitude

    @amplitude.setter
    def amplitude(self, value: float) -> None:
        self._amplitude = value

    @property
    def rotation(self) -> float:
        return self._rotation

    @rotation.setter
    def rotation(self, value: float) -> None:
        self._rotation = value

    def radon_transform(self, s, phi) -> float:

        phi_rotated = (phi + self.rotation) % (2 * self.xp.pi)

        s_prime = s / self.xp.sqrt(
            self._s0**2 * self.xp.cos(phi_rotated) ** 2
            + self._s1**2 * self.xp.sin(phi_rotated) ** 2
        )
        phi_prime = self.xp.atan2(
            self._s0 * self.xp.cos(phi_rotated),
            self._s1 * self.xp.sin(phi_rotated),
        )

        fac = (
            self._s0
            * self._s1
            / self.xp.sqrt(
                self._s0**2 * self.xp.cos(phi_rotated) ** 2
                + self._s1**2 * self.xp.sin(phi_rotated) ** 2
            )
        )

        return (
            self._amplitude
            * fac
            * self._centered_radon_transform(
                s_prime
                - self._x0_offset * self.xp.sin(phi_prime)
                - self._x1_offset * self.xp.cos(phi_prime),
                phi_prime,
            )
        )

    def values(self, x0: Array, x1: Array) -> Array:
        x0_p = x0 * math.cos(self._rotation) - x1 * math.sin(self._rotation)
        x1_p = x0 * math.sin(self._rotation) + x1 * math.cos(self._rotation)

        x0_pp = x0_p / self._s0 - self._x0_offset
        x1_pp = x1_p / self._s1 - self._x1_offset

        return self._amplitude * self._centered_values(x0_pp, x1_pp)


class RadonObjectSequence(Sequence[RadonObject]):
    def __init__(self, objects: Sequence[RadonObject]) -> None:
        super().__init__()
        self._objects: Sequence[RadonObject] = objects

    def __len__(self) -> int:
        return len(self._objects)

    def __getitem__(self, i: int) -> RadonObject:
        return self._objects[i]

    def radon_transform(self, r, phi) -> float:
        return sum([x.radon_transform(r, phi) for x in self])

    def values(self, x0: Array, x1: Array) -> Array:
        return sum([x.values(x0, x1) for x in self])


class RadonDisk(RadonObject):
    """2D disk with known radon transform"""

    def __init__(self, xp: ModuleType, dev: str, radius: float) -> None:
        super().__init__(xp, dev)
        self._radius: float = radius

    def _centered_radon_transform(self, r: Array, phi: Array) -> Array:
        mask = self.xp.abs(r) <= self._radius
        safe = self.xp.where(mask, self._radius**2 - r**2, self.xp.zeros_like(r))
        return self.xp.where(mask, 2 * self.xp.sqrt(safe), self.xp.zeros_like(r))

    def _centered_values(self, x0: Array, x1: Array) -> Array:
        return self.xp.where(
            x0**2 + x1**2 <= self._radius**2,
            self.xp.ones_like(x0),
            self.xp.zeros_like(x0),
        )

    @property
    def radius(self) -> float:
        return self._radius

    @radius.setter
    def radius(self, value: float) -> None:
        self._radius = value


class RadonSquare(RadonObject):
    """2D square with known radon transform"""

    def __init__(self, xp: ModuleType, dev: str, width: float) -> None:
        super().__init__(xp, dev)
        self._width: float = width
        self._eps = 1e-6

    def _centered_radon_transform(self, r: Array, phi: Array) -> Array:
        # calculate the angle alpha which is "distance to pi/4"
        f1 = phi // (self.xp.pi / 4)
        f2 = f1 % 2
        m = -(2 * f2 - 1)
        alpha = m * (phi - f1 * self.xp.pi / 4) + f2 * self.xp.pi / 4

        abs_r = self.xp.abs(r)

        # height of the plateau
        h = 2 * self.a / self.xp.cos(alpha)

        sqrt2 = float(self.xp.sqrt(self.xp.asarray(2.0)))

        # r2 is the distance until we have the plateau
        r2 = sqrt2 * self.a * self.xp.cos(self.xp.pi / 4 + alpha)
        # between r2 and r1 we have the triangle part
        r1 = sqrt2 * self.a * self.xp.cos(self.xp.pi / 4 - alpha)

        mask1 = self.xp.where(
            (r1 - r2) > self.eps, self.xp.ones_like(r), self.xp.zeros_like(r)
        )
        mask2 = self.xp.where(abs_r >= r2, self.xp.ones_like(r), self.xp.zeros_like(r))
        mask3 = self.xp.where(abs_r < r1, self.xp.ones_like(r), self.xp.zeros_like(r))

        # triangle mask
        t_mask = mask1 * mask2 * mask3

        # plateau aprt
        p1 = self.xp.where(abs_r < r2, h * self.xp.ones_like(r), self.xp.zeros_like(r))

        # triangle part avoiding division by zero
        denom = self.xp.where(
            r1 - r2 > self.eps, r1 - r2, self.eps * self.xp.ones_like(r)
        )
        tmp = -self.xp.sign(r) * h * self.xp.divide(r - self.xp.sign(r) * r2, denom) + h
        p2 = self.xp.where(t_mask == 1, tmp, self.xp.zeros_like(r))

        return p1 + p2

    def _centered_values(self, x0: Array, x1: Array) -> Array:
        return self.xp.where(
            self.xp.logical_and(self.xp.abs(x0) <= self.a, self.xp.abs(x1) <= self.a),
            self.xp.ones_like(x0),
            self.xp.zeros_like(x0),
        )

    @property
    def width(self) -> float:
        return self._width

    @width.setter
    def width(self, value: float) -> None:
        self._width = value

    @property
    def a(self) -> float:
        return self._width / 2

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        self._eps = value


def neighbor_offsets(ndim):
    # all offsets in {-1,0,1}^ndim except (0,...,0), as Python int tuples
    return [
        tuple(v - 1 for v in ind)
        for ind in itertools.product(range(3), repeat=ndim)
        if any(v != 1 for v in ind)
    ]


def neighbor_difference_and_sum(x, xp):
    """
    x: array_api-compatible array, shape (n1, n2, ..., nD)
    returns:
        d, s: shape (num_neigh, n1, n2, ..., nD)
              differences and sums with all nearest neighbors.
              If a neighbor is outside the image, the corresponding
              entry is 0 by definition.
    """
    ndim = x.ndim
    shape = x.shape
    offsets = neighbor_offsets(ndim)  # list of (num_neigh,) int tuples
    num_neigh = len(offsets)

    dev = device(x)

    d = xp.zeros((num_neigh,) + shape, dtype=x.dtype, device=dev)
    s = xp.zeros((num_neigh,) + shape, dtype=x.dtype, device=dev)

    for k in range(num_neigh):
        off = offsets[k]  # e.g. [-1, 0, 1, ...], shape (ndim,)

        # build slices for "center" and "neighbor" so that
        # center[...] and neighbor[...] have exactly the same shape
        center_sl = []
        neigh_sl = []
        for _ax, (n, o) in enumerate(zip(shape, off)):
            if o == 0:
                # full axis in both
                center_sl.append(slice(0, n))
                neigh_sl.append(slice(0, n))
            elif o > 0:
                # neighbor is shifted +o → valid center is [0 : n-o]
                center_sl.append(slice(0, n - o))
                neigh_sl.append(slice(o, n))
            else:  # o < 0
                # neighbor is shifted -|o| → valid center is [|o| : n]
                o_abs = -o
                center_sl.append(slice(o_abs, n))
                neigh_sl.append(slice(0, n - o_abs))

        center_sl = tuple(center_sl)
        neigh_sl = tuple(neigh_sl)

        # view with aligned shapes
        xc = x[center_sl]
        xn = x[neigh_sl]

        # place results into the corresponding region of d[k], s[k]
        # outside that region, d and s remain zero
        d_view = d[(k,) + center_sl]
        s_view = s[(k,) + center_sl]

        d_view[...] = xc - xn
        s_view[...] = xc + xn

    return d, s


def neighbor_product(x, xp):
    """
    Same neighbor definition as neighbor_difference_and_sum, but returns
    products x * neighbor. Products with neighbors outside the image are 0.
    """
    ndim = x.ndim
    shape = x.shape
    offsets = neighbor_offsets(ndim)
    num_neigh = len(offsets)
    dev = device(x)

    p = xp.zeros((num_neigh,) + shape, dtype=x.dtype, device=dev)

    for k in range(num_neigh):
        off = offsets[k]

        center_sl = []
        neigh_sl = []
        for _ax, (n, o) in enumerate(zip(shape, off)):
            if o == 0:
                center_sl.append(slice(0, n))
                neigh_sl.append(slice(0, n))
            elif o > 0:
                center_sl.append(slice(0, n - o))
                neigh_sl.append(slice(o, n))
            else:  # o < 0
                o_abs = -o
                center_sl.append(slice(o_abs, n))
                neigh_sl.append(slice(0, n - o_abs))

        center_sl = tuple(center_sl)
        neigh_sl = tuple(neigh_sl)

        xc = x[center_sl]
        xn = x[neigh_sl]

        p_view = p[(k,) + center_sl]
        p_view[...] = xc * xn

    return p


class SmoothFunction(abc.ABC):

    def __init__(self, in_shape, xp, dev, scale: float = 1.0) -> None:

        self._in_shape = in_shape
        self._scale = scale
        self._xp = xp
        self._dev = dev

    @property
    def scale(self) -> float:
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def xp(self):
        return self._xp

    @property
    def dev(self):
        return self._dev

    @abc.abstractmethod
    def _call(self, x: Array) -> float:
        raise NotImplementedError

    @abc.abstractmethod
    def _gradient(self, x: Array) -> Array:
        raise NotImplementedError

    def __call__(self, x: Array) -> float:
        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._call(x)
        else:
            res = self._scale * self._call(x)

        return res

    def gradient(self, x: Array) -> Array:
        dev_input = device(x)

        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._gradient(x)
        else:
            res = self._scale * self._gradient(x)

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        res = to_device(res, dev_input)

        return res

    def prox_function(self, z: Array, x: Array, T: Array) -> float:
        r"""returns the function h(z) = \sum_i f_i(z) + 0.5 * \sum_i (z_i - x_i)^2 / T_i
        which when minimized over z is the proximal operator of the function at x
        """
        return self.__call__(z) + 0.5 * float((((z - x) ** 2) / T).sum())

    def prox_gradient(self, z: Array, x: Array, T: Array) -> Array:
        """return the gradient of the prox function h(z), needed for numeric evaluation of the proximal operator"""
        return self.gradient(z) + (z - x) / T


class SmoothFunctionWithDiagonalHessian(SmoothFunction):
    @abc.abstractmethod
    def _diag_hessian(self, x: Array) -> Array:
        """(approximation) of the diagonal of the Hessian"""
        raise NotImplementedError

    def diag_hessian(self, x: Array) -> Array:
        dev_input = device(x)

        x = self._xp.asarray(x, device=self._dev)

        flat_input = x.ndim == 1
        if flat_input:
            x = self._xp.reshape(x, self._in_shape)

        if self._scale == 1.0:
            res = self._diag_hessian(x)
        else:
            res = self._scale * self._diag_hessian(x)

        if flat_input:
            res = self._xp.reshape(res, (res.size,))

        res = to_device(res, dev_input)

        return res


class RDP(SmoothFunctionWithDiagonalHessian):
    def __init__(
        self,
        in_shape: tuple[int, ...],
        xp: ModuleType,
        dev: str,
        voxel_size: Array,
        eps: float | None = None,
        gamma: float = 2.0,
    ) -> None:
        self._gamma = gamma

        if eps is None:
            self._eps = xp.finfo(xp.float32).eps
        else:
            self._eps = eps

        self._ndim = len(in_shape)

        if self._ndim != 3:
            raise ValueError("RDP only implemented for 3D images")

        super().__init__(in_shape=in_shape, xp=xp, dev=dev)

        # number of nearest neighbors
        self._num_neigh = 3**self._ndim - 1

        self._voxel_size = voxel_size

        # array for differences and sums with nearest neighbors
        self._voxel_size_weights = xp.zeros(
            (self._num_neigh,) + in_shape, dtype=xp.float32, device=dev
        )

        for i, ind in enumerate(itertools.product(range(3), repeat=self._ndim)):
            if i != (self._num_neigh // 2):
                offset = xp.asarray(ind, device=dev) - 1

                # this only works for 3D
                vw = voxel_size[2] / xp.linalg.norm(offset * voxel_size)

                if i < self._num_neigh // 2:
                    self._voxel_size_weights[i, ...] = vw
                else:
                    self._voxel_size_weights[i - 1, ...] = vw

        self._weights = self._voxel_size_weights
        self._kappa = None

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def eps(self) -> float:
        return self._eps

    @property
    def weights(self) -> Array:
        return self._weights

    @property
    def kappa(self) -> Array | None:
        return self._kappa

    @kappa.setter
    def kappa(self, image: Array) -> None:
        self._kappa = image
        self._weights = (
            neighbor_product(self._kappa, self._xp) * self._voxel_size_weights
        )

    def _call(self, x: Array) -> float:

        if float(self.xp.min(x)) < 0:
            return self.xp.inf

        d, s = neighbor_difference_and_sum(x, self.xp)
        # phi = s + self.gamma * self.xp.abs(d) + self.eps
        phi = self.xp.abs(d)
        phi *= self.gamma
        phi += s
        phi += self.eps

        # tmp = (d**2) / phi
        # reuse d to save memory
        tmp = d
        tmp *= tmp
        tmp /= phi

        if self._weights is not None:
            tmp *= self._weights

        return 0.5 * float(self.xp.sum(tmp, dtype=self.xp.float64))

    def _gradient(self, x: Array) -> Array:
        d, s = neighbor_difference_and_sum(x, self.xp)
        # phi = s + self.gamma * self.xp.abs(d) + self.eps
        tmp = self.xp.abs(d)
        tmp *= self.gamma
        phi = s + tmp
        phi += self.eps

        # tmp = d * (2 * phi - (d + self.gamma * self.xp.abs(d))) / (phi**2)
        tmp += d
        tmp -= phi
        tmp -= phi
        tmp *= -1
        tmp *= d
        tmp /= phi
        tmp /= phi

        if self._weights is not None:
            tmp *= self._weights

        return tmp.sum(axis=0)

    def _diag_hessian(self, x: Array) -> Array:
        d, s = neighbor_difference_and_sum(x, self.xp)
        # phi = s + self.gamma * self.xp.abs(d) + self.eps
        phi = self.xp.abs(d)
        phi *= self.gamma
        phi += s
        phi += self.eps

        # tmp = ((s - d + self.eps) ** 2) / (phi**3)
        tmp = s - d
        tmp += self.eps
        tmp *= tmp
        tmp /= phi
        tmp /= phi
        tmp /= phi

        if self._weights is not None:
            tmp *= self._weights

        return 2 * tmp.sum(axis=0)


def show_3d_cuts(
    vol: np.ndarray,
    voxel_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
    fig_title: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "Greys",
    figsize: tuple[int, int] = (12, 6),
) -> tuple:
    """Display interactive 2D cuts through a 3D volume along all three axes.

    Controls below the images:

    * Three sliders to scroll through cuts along x, y, z.
    * Two text boxes to update the colour scale limits (vmin / vmax).
    * One text box to change the colormap (any valid matplotlib name).

    Non-uniform voxel sizes are respected: each image is displayed with the
    correct pixel aspect ratio so that physical distances are represented
    accurately.

    Parameters
    ----------
    vol : np.ndarray, shape (nx, ny, nz)
        3-D volume to visualise.
    voxel_size : tuple of three floats, optional
        Physical size ``(dx, dy, dz)`` of one voxel.  Default ``(1, 1, 1)``.
    fig_title : str or None, optional
        Overall figure title.
    vmin, vmax : float or None, optional
        Initial colour scale limits.  Defaults to the global min / max of
        ``vol``.
    cmap : str, optional
        Initial matplotlib colormap name.  Default ``"Greys"``.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axs : list of three Axes (image panels)
    widgets : dict
        All interactive widgets keyed by name (``"sx"``, ``"sy"``, ``"sz"``,
        ``"tb_vmin"``, ``"tb_vmax"``, ``"tb_cmap"``).  Keep a reference to
        this dict to prevent garbage collection of the callbacks.
    """
    assert vol.ndim == 3, "vol must be a 3-D array"
    nx, ny, nz = vol.shape
    dx, dy, dz = voxel_size

    if vmin is None:
        vmin = float(vol.min())
    if vmax is None:
        vmax = float(vol.max())

    ix0, iy0, iz0 = nx // 2, ny // 2, nz // 2

    # --- layout via GridSpec ---
    # row 0: images
    # row 1: slice sliders
    # row 2: vmin / vmax / cmap text boxes
    # row 3: horizontal colorbar (spans all columns)
    fig = plt.figure(figsize=figsize)
    if fig_title is not None:
        fig.suptitle(fig_title)

    gs = fig.add_gridspec(
        4, 3,
        height_ratios=[8, 1, 1, 0.5],
        hspace=0.6,
        wspace=0.35,
    )

    ax_im = [fig.add_subplot(gs[0, c]) for c in range(3)]
    ax_sl = [fig.add_subplot(gs[1, c]) for c in range(3)]
    ax_tb = [fig.add_subplot(gs[2, c]) for c in range(3)]
    ax_cb = fig.add_subplot(gs[3, :])

    # --- images ---
    # cut along x: shows (y, z) plane  → aspect = dz / dy
    # cut along y: shows (x, z) plane  → aspect = dz / dx
    # cut along z: shows (x, y) plane  → aspect = dy / dx
    aspects = [dz / dy, dz / dx, dy / dx]
    labels  = [("y", "z"), ("x", "z"), ("x", "y")]

    imkw = dict(vmin=vmin, vmax=vmax, cmap=cmap, origin="lower")
    ims = [
        ax_im[0].imshow(vol[ix0, :, :].T, aspect=aspects[0], **imkw),
        ax_im[1].imshow(vol[:, iy0, :].T, aspect=aspects[1], **imkw),
        ax_im[2].imshow(vol[:, :, iz0].T, aspect=aspects[2], **imkw),
    ]
    titles = [f"x = {ix0}", f"y = {iy0}", f"z = {iz0}"]
    for i, ax in enumerate(ax_im):
        ax.set_xlabel(labels[i][0])
        ax.set_ylabel(labels[i][1])
        ax.set_title(titles[i])

    # --- slice sliders ---
    sx = Slider(ax_sl[0], "x", 0, nx - 1, valinit=ix0, valstep=1)
    sy = Slider(ax_sl[1], "y", 0, ny - 1, valinit=iy0, valstep=1)
    sz = Slider(ax_sl[2], "z", 0, nz - 1, valinit=iz0, valstep=1)

    def update_slices(_):
        ix, iy, iz = int(sx.val), int(sy.val), int(sz.val)
        ims[0].set_data(vol[ix, :, :].T)
        ims[1].set_data(vol[:, iy, :].T)
        ims[2].set_data(vol[:, :, iz].T)
        ax_im[0].set_title(f"x = {ix}")
        ax_im[1].set_title(f"y = {iy}")
        ax_im[2].set_title(f"z = {iz}")
        fig.canvas.draw_idle()

    sx.on_changed(update_slices)
    sy.on_changed(update_slices)
    sz.on_changed(update_slices)

    # --- colorbar ---
    cb = fig.colorbar(ims[0], cax=ax_cb, orientation="horizontal")

    # --- vmin / vmax text boxes ---
    tb_vmin = TextBox(ax_tb[0], "vmin", initial=f"{vmin:.4g}")
    tb_vmax = TextBox(ax_tb[1], "vmax", initial=f"{vmax:.4g}")

    def update_clim(_):
        try:
            lo = float(tb_vmin.text)
            hi = float(tb_vmax.text)
        except ValueError:
            return
        for im in ims:
            im.set_clim(lo, hi)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_vmin.on_submit(update_clim)
    tb_vmax.on_submit(update_clim)

    # --- colormap text box ---
    tb_cmap = TextBox(ax_tb[2], "cmap", initial=cmap)

    def update_cmap(name):
        try:
            cm = plt.get_cmap(name.strip())
        except (ValueError, KeyError):
            tb_cmap.set_val(ims[0].cmap.name)  # revert to current
            return
        for im in ims:
            im.set_cmap(cm)
        cb.update_normal(ims[0])
        fig.canvas.draw_idle()

    tb_cmap.on_submit(update_cmap)

    widgets = dict(sx=sx, sy=sy, sz=sz, tb_vmin=tb_vmin, tb_vmax=tb_vmax, tb_cmap=tb_cmap)
    return fig, ax_im, widgets
