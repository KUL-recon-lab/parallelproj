from collections.abc import Sequence
import abc
import math
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
        rt = self.xp.zeros_like(r)
        rt[self.xp.abs(r) <= self._radius] = 2 * self.xp.sqrt(
            self._radius**2 - r[self.xp.abs(r) <= self._radius] ** 2
        )

        return rt

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
    def radius(self, value: float) -> None:
        self._radius = value

    @property
    def a(self) -> float:
        return self._width / 2

    @property
    def eps(self) -> float:
        return self._eps

    @eps.setter
    def eps(self, value: float) -> None:
        self._eps = value
