"""Demo 3D images for parallelproj examples."""

from __future__ import annotations

from importlib import import_module
from typing import Any
from types import ModuleType
from parallelproj import Array


def elliptic_cylinder_phantom(
    xp: ModuleType,
    dev: Any,
    image_shape: tuple[int, int, int] = (40, 40, 8),
    voxel_size: tuple[float, float, float] = (2.0, 2.0, 2.0),
) -> Array:
    """3D elliptic cylinder phantom with hot and cold spherical inserts.

    The cylinder has an elliptical cross-section and is oriented along the
    third axis (axis 2).  Six spherical inserts of varying radii and
    contrasts are embedded inside the uniform cylinder background.

    The coordinate origin is at the centre of the image volume.  All
    geometric parameters (cylinder semi-axes, insert positions and radii)
    scale automatically with ``image_shape`` and ``voxel_size``, so the
    phantom looks the same regardless of resolution.

    Parameters
    ----------
    image_shape:
        Number of voxels along each axis ``(n0, n1, n2)``.
        Default: ``(40, 40, 8)``.
    voxel_size:
        Physical voxel size along each axis.
        Default: ``(2.0, 2.0, 2.0)``.
    xp:
        Array API-compatible namespace (e.g. ``array_api_compat.torch`` or
        ``array_api_compat.numpy``).  Defaults to ``array_api_compat.numpy``.
    dev:
        Device passed to array-creation functions (e.g. ``"cpu"`` or
        ``"cuda"``).  Default: ``"cpu"``.

    Returns
    -------
    Array
        ``float32`` array of shape ``image_shape`` on device ``dev``.

        * Outside the cylinder: ``0.0``
        * Cylinder background: ``1.0``
        * Hot inserts: ``> 1.0`` (contrasts 2x, 4x, 8x)
        * Cold inserts: ``< 1.0`` (contrasts 0x, 0.25x, 0.5x)
    """

    n0, n1, n2 = image_shape
    v0, v1, v2 = voxel_size

    fov0 = n0 * v0
    fov1 = n1 * v1
    fov2 = n2 * v2

    # Voxel-centre coordinate grids in physical units, centred at the origin
    x0 = (xp.arange(n0, device=dev, dtype=xp.float32) - (n0 - 1) / 2.0) * v0
    x1 = (xp.arange(n1, device=dev, dtype=xp.float32) - (n1 - 1) / 2.0) * v1
    x2 = (xp.arange(n2, device=dev, dtype=xp.float32) - (n2 - 1) / 2.0) * v2

    X0, X1, X2 = xp.meshgrid(x0, x1, x2, indexing="ij")

    # ------------------------------------------------------------------
    # Elliptic cylinder
    # ------------------------------------------------------------------
    # Transaxial semi-axes (40 % of each FOV dimension) and axial half-height
    a0 = 0.45 * fov0
    a1 = 0.30 * fov1
    hz = 0.50 * fov2

    inside_cyl = ((X0 / a0) ** 2 + (X1 / a1) ** 2 <= 1.0) & (xp.abs(X2) <= hz)

    # Boolean -> float32: 1.0 inside the cylinder, 0.0 outside
    img = xp.astype(inside_cyl, xp.float32)

    # ------------------------------------------------------------------
    # Spherical inserts
    # ------------------------------------------------------------------
    # Insert sizes are expressed as a fraction of the smaller transaxial
    # semi-axis so they scale consistently with the phantom dimensions.
    r_ref = min(a0, a1)

    # Each row: (c0/a0, c1/a1, c2/hz, radius/r_ref, fill_value)
    #   fill_value > 1  ->  hot insert
    #   fill_value < 1  ->  cold insert
    #   fill_value = 0  ->  signal-free (dark) cold insert
    inserts: list[tuple[float, float, float, float, float]] = [
        # --- hot inserts ---
        (0.50, 0.00, 0.00, 0.08, 2.0),  # small,  4x contrast
        (-0.50, 0.00, 0.30, 0.12, 1.5),  # medium, 2x contrast
        (0.00, 0.50, -0.30, 0.18, 3.0),  # large,  8x contrast
        # --- cold inserts ---
        (-0.45, 0.00, -0.30, 0.08, 0.5),  # medium, 0.5x contrast
        (0.30, -0.50, 0.00, 0.18, 0.0),  # medium, signal-free
        (0.00, -0.45, 0.30, 0.12, 0.25),  # medium, 0.25x contrast
    ]

    for c0f, c1f, c2f, r_frac, value in inserts:
        cx0 = c0f * a0
        cx1 = c1f * a1
        cx2 = c2f * hz
        radius = r_frac * r_ref

        dist2 = (X0 - cx0) ** 2 + (X1 - cx1) ** 2 + (X2 - cx2) ** 2
        fill = xp.full(image_shape, value, dtype=xp.float32, device=dev)
        img = xp.where(dist2 <= radius**2, fill, img)

    return img
