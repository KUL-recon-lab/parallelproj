"""description of PET LORs (and sinograms bins) consisting of two detector endpoints"""

from __future__ import annotations

import abc
import enum
from types import ModuleType

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.axes import Axes
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Circle


from mpl_toolkits.mplot3d import Axes3D

from ._backend import Array, to_numpy_array

from .operators import LinearOperator
from .tof import TOFParameters
from .pet_scanners import (
    ModularizedPETScannerGeometry,
    RegularPolygonPETScannerGeometry,
)


class SinogramSpatialAxisOrder(enum.Enum):
    """order of spatial axis in a sinogram R (radial), V (view), P (plane)"""

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


class SinogramZigZagOrder(enum.Enum):
    """Zig-zag ordering of in-ring detector pairs for each sinogram view.

    For a scanner with :math:`n` detector endpoints per ring and view index 0,
    the two variants differ in which detector (start or end) steps first as the
    radial bin index increases from the central LOR outward.

    ``END_FIRST``
        The *end* detector steps first for each new radial pair.
        Pairs (start, end) at view 0: (0,n-1), (0,n-2), (1,n-2), (1,n-3), …

    ``START_FIRST``
        The *start* detector steps first for each new radial pair.
        Pairs (start, end) at view 0: (0,n-1), (1,n-1), (1,n-2), (2,n-2), …
    """

    END_FIRST = enum.auto()
    """End crystal steps first (default, historically used convention)."""
    START_FIRST = enum.auto()
    """Start crystal steps first."""


class Michelogram:
    """Axial plane layout for a cylindrical PET scanner under odd span.

    Encapsulates the segment / axial-position combinatorics that map every
    valid ring pair :math:`(s, e)` onto a sinogram plane index under
    span conventions.

    For span :math:`S` (odd) and a maximum ring difference :math:`D`, each
    ring pair with :math:`|e - s| \\le D` is assigned a segment via
    :meth:`ring_diff_to_segment`.  Ring pairs sharing the same
    :math:`(\\text{segment},\\; s + e)` collapse into a single plane.
    Planes are ordered by :math:`(|\\text{seg}|,\\; -\\text{seg},\\; s + e)`
    :math:`[0, +1, -1, +2, -2, \\ldots]` with axial bins increasing in
    :math:`s + e` (equivalently in z for equispaced rings).

    The class knows nothing about ring z-positions, scanner radius, or
    sinogram axis ordering — it operates on pure integer indices.  Consumers
    (e.g. :class:`RegularPolygonPETLORDescriptor`,
    :class:`SinogramAxialCompressionOperator`) combine it with the geometry-
    and array-API-specific information they need.

    For span ``= 1`` the layout reduces to the unspanned Michelogram (each
    ring pair is its own plane with :attr:`max_multiplicity` ``== 1``); the
    ordering ``rd = 0, +1, -1, +2, -2, ...`` with each ring difference
    sorted by ring sum.

    Parameters
    ----------
    num_rings : int
        Number of detector rings (:math:`R \\ge 1`).
    max_ring_difference : int
        Maximum ring difference :math:`|e - s|` considered (:math:`\\ge 0`).
        Values larger than ``num_rings - 1`` have no extra effect.
    span : int, optional
        Axial compression factor — must be odd and :math:`\\ge 1`.
        Default ``1`` (no compression).

    Examples
    --------
    >>> m = Michelogram(num_rings=3, max_ring_difference=2, span=3)
    >>> int(m.num_planes)
    7
    >>> int(m.max_multiplicity)
    2
    >>> int(m.ring_diff_to_segment(0)), int(m.ring_diff_to_segment(2)), \
        int(m.ring_diff_to_segment(-2))
    (0, 1, -1)
    """

    # ------------------------------------------------------------------
    # Construction / core formula
    # ------------------------------------------------------------------

    def __init__(
        self,
        num_rings: int,
        max_ring_difference: int,
        span: int = 1,
    ) -> None:
        if not isinstance(num_rings, int) or num_rings < 1:
            raise ValueError("num_rings must be a positive integer")
        if not isinstance(max_ring_difference, int) or max_ring_difference < 0:
            raise ValueError("max_ring_difference must be a non-negative integer")
        if not isinstance(span, int) or span < 1 or span % 2 == 0:
            raise ValueError("span must be an odd positive integer")

        self._num_rings = int(num_rings)
        self._max_ring_difference = int(max_ring_difference)
        self._span = int(span)
        self._half_span = (self._span - 1) // 2

        self._build()

    def ring_diff_to_segment(self, rd: int) -> int:
        """Signed segment number for a given ring difference :math:`e - s`.

        Returns
        -------
        int
            ``0`` if :math:`|rd| \\le \\text{half\\_span}`, otherwise
            :math:`\\pm k` with :math:`k = \\lceil (|rd| - \\text{half\\_span}) / S \\rceil`
            and sign equal to that of :math:`rd`.
        """
        S = self._span
        half_span = self._half_span
        abs_rd = abs(rd)
        if abs_rd <= half_span:
            return 0
        k = (abs_rd - half_span + S - 1) // S
        return k if rd > 0 else -k

    def _build(self) -> None:
        """Compute and cache the full plane layout."""
        R = self._num_rings
        D = self._max_ring_difference

        # Group every valid ring pair (s, e) by (segment, s + e).  Iteration
        # order here does not matter — we sort the result.
        plane_groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for s in range(R):
            for e in range(R):
                rd = e - s
                if abs(rd) > D:
                    continue
                seg = self.ring_diff_to_segment(rd)
                key = (seg, s + e)
                plane_groups.setdefault(key, []).append((s, e))

        # standard segment sequence + within-segment axial-midpoint order.
        sorted_keys = sorted(
            plane_groups.keys(), key=lambda k: (abs(k[0]), -k[0], k[1])
        )

        num_planes = len(sorted_keys)
        plane_segment = np.empty(num_planes, dtype=np.int32)
        plane_axial_midpoint_int = np.empty(num_planes, dtype=np.int32)
        plane_multiplicity = np.empty(num_planes, dtype=np.int32)

        for pi, key in enumerate(sorted_keys):
            plane_segment[pi] = key[0]
            plane_axial_midpoint_int[pi] = key[1]
            plane_multiplicity[pi] = len(plane_groups[key])

        max_mult = int(plane_multiplicity.max()) if num_planes > 0 else 0

        plane_start_rings = np.zeros((num_planes, max_mult), dtype=np.int32)
        plane_end_rings = np.zeros((num_planes, max_mult), dtype=np.int32)
        plane_mask = np.zeros((num_planes, max_mult), dtype=np.float32)

        # Inverse lookup table; -1 indicates an invalid pair (|rd| > D).
        plane_for_ring_pair_table = np.full((R, R), -1, dtype=np.int32)

        for pi, key in enumerate(sorted_keys):
            pairs = plane_groups[key]
            for k, (s, e) in enumerate(pairs):
                plane_start_rings[pi, k] = s
                plane_end_rings[pi, k] = e
                plane_mask[pi, k] = 1.0
                plane_for_ring_pair_table[s, e] = pi

        self._num_planes = num_planes
        self._max_multiplicity = max_mult
        self._plane_segment = plane_segment
        self._plane_axial_midpoint_int = plane_axial_midpoint_int
        self._plane_multiplicity = plane_multiplicity
        self._plane_start_rings = plane_start_rings
        self._plane_end_rings = plane_end_rings
        self._plane_mask = plane_mask
        self._plane_for_ring_pair_table = plane_for_ring_pair_table

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def num_rings(self) -> int:
        """Number of rings."""
        return self._num_rings

    @property
    def max_ring_difference(self) -> int:
        """Maximum ring difference :math:`|e - s|`."""
        return self._max_ring_difference

    @property
    def span(self) -> int:
        """Axial compression factor (odd)."""
        return self._span

    @property
    def num_planes(self) -> int:
        """Total number of sinogram planes."""
        return self._num_planes

    @property
    def max_multiplicity(self) -> int:
        """Largest plane multiplicity (most ring pairs in any one plane)."""
        return self._max_multiplicity

    @property
    def plane_segment(self) -> np.ndarray:
        """Signed segment number for each plane, shape ``(num_planes,)``,
        dtype ``int32``."""
        return self._plane_segment

    @property
    def plane_axial_midpoint_int(self) -> np.ndarray:
        """Integer axial midpoint :math:`s + e` (= twice the actual midpoint)
        for each plane, shape ``(num_planes,)``, dtype ``int32``."""
        return self._plane_axial_midpoint_int

    @property
    def plane_multiplicity(self) -> np.ndarray:
        """Number of ring pairs contributing to each plane,
        shape ``(num_planes,)``, dtype ``int32``."""
        return self._plane_multiplicity

    @property
    def plane_start_rings(self) -> np.ndarray:
        """Contributing start ring indices per plane, right-padded with ``0``.

        Shape ``(num_planes, max_multiplicity)``, dtype ``int32``.  Use
        :attr:`plane_mask` to identify the valid entries.
        """
        return self._plane_start_rings

    @property
    def plane_end_rings(self) -> np.ndarray:
        """Contributing end ring indices per plane, right-padded with ``0``.

        Shape ``(num_planes, max_multiplicity)``, dtype ``int32``.  Use
        :attr:`plane_mask` to identify the valid entries.
        """
        return self._plane_end_rings

    @property
    def plane_mask(self) -> np.ndarray:
        """Validity mask for :attr:`plane_start_rings` / :attr:`plane_end_rings`.

        Shape ``(num_planes, max_multiplicity)``, dtype ``float32``.  Entries
        are ``1.0`` for valid contributing ring pairs and ``0.0`` for
        right-padding.
        """
        return self._plane_mask

    @property
    def plane_for_ring_pair_table(self) -> np.ndarray:
        """``(num_rings, num_rings)`` lookup table whose entry ``[s, e]``
        is the plane index for ring pair ``(s, e)``, or ``-1`` if
        :math:`|e - s| > \\text{max\\_ring\\_difference}`."""
        return self._plane_for_ring_pair_table

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def plane_for_ring_pair(self, s: int, e: int) -> int:
        """Plane index for the ring pair ``(s, e)``.

        Raises
        ------
        IndexError
            If either ``s`` or ``e`` is outside ``[0, num_rings)``.
        ValueError
            If :math:`|e - s| > \\text{max\\_ring\\_difference}`.
        """
        if not (0 <= s < self._num_rings) or not (0 <= e < self._num_rings):
            raise IndexError(
                f"ring indices out of range: ({s}, {e}); "
                f"num_rings={self._num_rings}"
            )
        pi = int(self._plane_for_ring_pair_table[s, e])
        if pi < 0:
            raise ValueError(
                f"ring pair ({s}, {e}) has |rd|={abs(e - s)} > "
                f"max_ring_difference={self._max_ring_difference}"
            )
        return pi

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------

    def average_z_per_plane(self, ring_positions) -> tuple[np.ndarray, np.ndarray]:
        """Mean ring z-coordinate per plane, separately for start and end rings.

        Equivalent to averaging ``ring_positions`` over the contributing ring
        pairs of each plane.  For span ``=1`` planes this is trivially the
        single contributing ring's z; for span ``> 1`` planes it produces the
        averaged-LOR z-position used by the spanned setup of
        :class:`RegularPolygonPETLORDescriptor` and by
        :meth:`show_segment_lors`.

        Parameters
        ----------
        ring_positions : array-like, shape ``(num_rings,)``
            z-coordinate of each ring (any backend; converted via ``np.asarray``).

        Returns
        -------
        start_z : np.ndarray, shape ``(num_planes,)``, dtype ``float32``
        end_z : np.ndarray, shape ``(num_planes,)``, dtype ``float32``
        """
        ring_pos = np.asarray(ring_positions, dtype=np.float64)
        if ring_pos.ndim != 1 or ring_pos.shape[0] != self._num_rings:
            raise ValueError(
                "ring_positions must be a 1-D array of length "
                f"num_rings={self._num_rings}"
            )
        mult = self._plane_multiplicity.astype(np.float64)
        start_z = (ring_pos[self._plane_start_rings] * self._plane_mask).sum(
            axis=1
        ) / mult
        end_z = (ring_pos[self._plane_end_rings] * self._plane_mask).sum(axis=1) / mult
        return start_z.astype(np.float32), end_z.astype(np.float32)

    # ------------------------------------------------------------------
    # Axial compression
    # ------------------------------------------------------------------

    def compression_index_maps_to(
        self, target: "Michelogram"
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build gather/scatter index maps to a pre-built target Michelogram.

        Returns the integer index structures needed to map planes of this
        Michelogram onto planes of ``target``.

        Both Michelograms must describe the same scanner geometry
        (``target.num_rings == self.num_rings``), and ``target.span`` must
        be an integer multiple of ``self.span``.  Because both spans are
        odd by construction, the ratio ``target.span / self.span`` is then
        automatically odd, which guarantees that every ring pair of any
        input plane shares the same target plane — so the operation is a
        single-valued gather.

        The target's ``max_ring_difference`` must be at least
        ``self.max_ring_difference`` so every input ring pair has a target
        plane.  If it is strictly greater, the resulting maps still work but
        some output planes will have zero multiplicity (output bins that no
        input ring pair contributes to).

        Parameters
        ----------
        target : Michelogram
            Pre-built target Michelogram.  Validation rules above.

        Returns
        -------
        target_for_p1 : np.ndarray, shape ``(self.num_planes,)``, dtype ``int64``
            For each plane of this Michelogram, the corresponding plane index
            in ``target``.
        idx2d : np.ndarray, shape ``(target.num_planes, target_max_mult)``, ``int64``
            For each target plane, the indices in this Michelogram that
            contribute, right-padded with ``0``.  Use ``mask`` to filter.
        mask : np.ndarray, same shape as ``idx2d``, dtype ``float32``
            ``1.0`` for valid entries, ``0.0`` for right-padding.
        target_multiplicity : np.ndarray, shape ``(target.num_planes,)``, ``int32``
            Number of self-planes folded into each target plane.

        Raises
        ------
        TypeError
            If ``target`` is not a :class:`Michelogram` instance.
        ValueError
            If ``target.num_rings`` differs from ``self.num_rings``;
            if ``target.span < self.span``;
            if ``self.span`` does not divide ``target.span``;
            or if ``target.max_ring_difference < self.max_ring_difference``.
        """
        if not isinstance(target, Michelogram):
            raise TypeError("target must be a Michelogram instance")
        if target.num_rings != self._num_rings:
            raise ValueError(
                f"target.num_rings ({target.num_rings}) must match "
                f"self.num_rings ({self._num_rings})"
            )
        if target.span < self._span:
            raise ValueError(
                f"target.span ({target.span}) must be >= self.span ({self._span})"
            )
        if target.span % self._span != 0:
            raise ValueError(
                f"target.span ({target.span}) must be an integer multiple "
                f"of self.span ({self._span})"
            )
        if target.max_ring_difference < self._max_ring_difference:
            raise ValueError(
                f"target.max_ring_difference ({target.max_ring_difference}) "
                f"must be >= self.max_ring_difference "
                f"({self._max_ring_difference}) so every input ring pair "
                "has a target plane"
            )

        num_planes_in = self._num_planes
        target_for_p1 = np.empty(num_planes_in, dtype=np.int64)

        # Any ring pair in an input plane maps to the same target plane
        # under the divisibility condition checked above, so we take the
        # first one as a representative.
        for pi in range(num_planes_in):
            s = int(self._plane_start_rings[pi, 0])
            e = int(self._plane_end_rings[pi, 0])
            target_for_p1[pi] = target.plane_for_ring_pair(s, e)

        # Invert: per target plane, list contributing input plane indices.
        num_planes_out = target.num_planes
        groups: list[list[int]] = [[] for _ in range(num_planes_out)]
        for pi in range(num_planes_in):
            groups[int(target_for_p1[pi])].append(pi)

        target_multiplicity = np.fromiter(
            (len(g) for g in groups), dtype=np.int32, count=num_planes_out
        )
        target_max_mult = int(target_multiplicity.max()) if num_planes_out > 0 else 0

        idx2d = np.zeros((num_planes_out, target_max_mult), dtype=np.int64)
        mask = np.zeros((num_planes_out, target_max_mult), dtype=np.float32)
        for n, g in enumerate(groups):
            if g:
                idx2d[n, : len(g)] = g
                mask[n, : len(g)] = 1.0

        return target_for_p1, idx2d, mask, target_multiplicity

    def compression_index_maps(
        self, target_span: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build gather/scatter index maps to a higher-span Michelogram.

        Convenience wrapper around :meth:`compression_index_maps_to` that
        builds the target Michelogram internally as
        ``Michelogram(self.num_rings, self.max_ring_difference, span=target_span)``.

        Parameters
        ----------
        target_span : int
            Odd integer ``>= self.span``; additionally
            ``(target_span // self.span)`` must be odd.

        Returns
        -------
        See :meth:`compression_index_maps_to`.

        Raises
        ------
        ValueError
            If ``target_span`` is not a positive odd integer.  Further
            validation errors are raised by
            :meth:`compression_index_maps_to`.
        """
        if not isinstance(target_span, int) or target_span < 1 or target_span % 2 == 0:
            raise ValueError("target_span must be an odd positive integer")
        target = Michelogram(
            self._num_rings, self._max_ring_difference, span=target_span
        )
        return self.compression_index_maps_to(target)

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show(
        self,
        ax: Axes,
        show_merge_lines: bool = True,
        plane_index_fontsize: float = 6,
        **kwargs,
    ) -> None:
        """Draw the Michelogram scatter plot onto ``ax``.

        Each point represents a valid ring pair ``(s, e)``, colored by
        ``abs(segment)``.  For ``span > 1``, ring pairs that share the same
        ``(segment, s + e)`` and therefore collapse into the same sinogram
        plane are connected by a thin grey line when ``show_merge_lines``
        is ``True``.

        Parameters
        ----------
        ax : plt.Axes
            2-D matplotlib axes (not 3-D).
        show_merge_lines : bool, optional
            Draw lines connecting ring pairs that merge into the same plane.
            Defaults to ``True``.  Only has a visible effect for ``span > 1``.
        plane_index_fontsize : float, optional
            Font size of the per-plane index annotations placed at each
            ring-pair (or merged-group) centroid.  Defaults to ``6``.  Useful
            knob when the Michelogram is large (lower to avoid overlap) or
            small (raise for readability).
        **kwargs
            Forwarded to ``ax.scatter`` (e.g. ``s=4``, ``cmap="RdBu_r"``).
        """
        self._draw_axes(
            ax,
            show_merge_lines=show_merge_lines,
            plane_index_fontsize=plane_index_fontsize,
            **kwargs,
        )

    def show_segment_lors(
        self,
        ring_positions,
        axs=None,
        uncompressed_lor_kwargs: dict | None = None,
        compressed_lor_kwargs: dict | None = None,
        inset_plane_index_fontsize: float = 4,
    ):
        """Side-view LOR diagram per segment with a Michelogram inset.

        Mirrors the descriptor's
        :meth:`RegularPolygonPETLORDescriptor.show_segment_lors`, but takes
        ``ring_positions`` explicitly so the Michelogram can be visualised
        standalone (e.g. with ``np.arange(num_rings)`` for a purely
        schematic plot, or with the user's actual ring z-positions).

        Subplots are arranged in a 2-row grid (when negative segments exist):

        * **columns** indexed by ``abs(segment)``: 0, 1, 2, ...
        * **row 0** non-negative segments (0, +1, +2, ...)
        * **row 1, col 0** Michelogram inset
        * **row 1, col >= 1** negative segments (-1, -2, ...)

        Each LOR subplot shows the uncompressed (per-ring-pair) LORs as
        solid black lines and the compressed (axially-averaged) LORs as
        dashed coloured lines.

        Parameters
        ----------
        ring_positions : array-like, shape ``(num_rings,)``
            z-coordinate of each ring.
        axs : 2-D array-like of Axes, optional
            Pre-existing axes of shape ``(n_rows, n_cols)``.  If ``None``,
            a new figure is created.
        uncompressed_lor_kwargs : dict, optional
            Style overrides for the uncompressed LOR lines.
        compressed_lor_kwargs : dict, optional
            Style overrides for the compressed LOR lines.

        Returns
        -------
        matplotlib.figure.Figure
        """
        R = self._num_rings
        D = self._max_ring_difference

        ring_pos = np.asarray(ring_positions, dtype=np.float64)
        if ring_pos.ndim != 1 or ring_pos.shape[0] != R:
            raise ValueError(
                f"ring_positions must be a 1-D array of length num_rings={R}"
            )

        start_z, end_z = self.average_z_per_plane(ring_pos)
        start_z_np = np.asarray(start_z, dtype=np.float64)
        end_z_np = np.asarray(end_z, dtype=np.float64)
        seg_arr_np = np.asarray(self._plane_segment, dtype=np.int32)

        all_segs = sorted(set(int(v) for v in seg_arr_np))
        abs_segs = sorted(set(abs(s) for s in all_segs))
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
            _axs = raw
        else:
            _axs = np.asarray(axs)
            fig = _axs.flat[0].get_figure()

        # coordinate normalisation
        z_min, z_max = ring_pos.min(), ring_pos.max()
        z_span = max(z_max - z_min, 1.0)
        margin = 0.12 * z_span
        x_L, x_R = -z_span / 2.0, z_span / 2.0
        ring_r = 0.35 * z_span / max(R, 1)

        # uncompressed ring-pair lookup keyed by signed segment
        uncompressed: dict[int, list[tuple[int, int]]] = {s: [] for s in all_segs}
        for s_ring in range(R):
            for e_ring in range(R):
                rd = e_ring - s_ring
                if abs(rd) > D:
                    continue
                seg = self.ring_diff_to_segment(rd)
                if seg in uncompressed:
                    uncompressed[seg].append((s_ring, e_ring))

        n_colors = len(abs_segs)
        base_cmap = plt.get_cmap("tab10" if n_colors <= 10 else "tab20")

        for col_idx, abs_seg in enumerate(abs_segs):
            color = base_cmap(col_idx)

            for row_idx in range(n_rows):
                ax = _axs[row_idx, col_idx]

                # [1, 0]: Michelogram inset instead of the non-existent
                # segment -0.
                if row_idx == 1 and abs_seg == 0:
                    self._draw_axes(
                        ax, plane_index_fontsize=inset_plane_index_fontsize
                    )
                    continue

                seg_val = abs_seg if row_idx == 0 else -abs_seg

                if seg_val not in all_segs:
                    ax.axis("off")
                    continue

                # (a) uncompressed ring-pair LORs
                for s_r, e_r in uncompressed[seg_val]:
                    ax.plot(
                        [x_L, x_R],
                        [ring_pos[s_r], ring_pos[e_r]],
                        **unc_kw,
                    )

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
                    seg_label = f"-{abs_seg}"
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

    def _draw_axes(
        self,
        ax: Axes,
        show_merge_lines: bool = True,
        plane_index_fontsize: float = 6,
        **kwargs,
    ) -> None:
        """Internal helper: draw the Michelogram onto an existing axes.

        Shared between :meth:`show` (main use) and :meth:`show_segment_lors`
        (inset).
        """
        R = self._num_rings
        D = self._max_ring_difference
        S = self._span

        # Unroll the padded layout into flat arrays of valid ring pairs.
        total = int(self._plane_mask.sum())
        start_arr = np.empty(total, dtype=np.float32)
        end_arr = np.empty(total, dtype=np.float32)
        seg_arr = np.empty(total, dtype=np.int32)
        idx = 0
        for pi in range(self._num_planes):
            seg = int(self._plane_segment[pi])
            m = int(self._plane_multiplicity[pi])
            for k in range(m):
                start_arr[idx] = int(self._plane_start_rings[pi, k])
                end_arr[idx] = int(self._plane_end_rings[pi, k])
                seg_arr[idx] = seg
                idx += 1

        abs_seg_arr = np.abs(seg_arr)
        n_colors = int(abs_seg_arr.max()) + 1

        base_cmap = plt.get_cmap("tab10" if n_colors <= 10 else "tab20")
        cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
        norm = BoundaryNorm(np.arange(-0.5, n_colors, 1.0), cmap.N)

        kwargs.setdefault("s", 20)
        ax.scatter(
            start_arr,
            end_arr,
            c=abs_seg_arr.astype(np.float32),
            cmap=cmap,
            norm=norm,
            **kwargs,
        )

        if show_merge_lines and S > 1:
            for pi in range(self._num_planes):
                m = int(self._plane_multiplicity[pi])
                if m > 1:
                    xs = self._plane_start_rings[pi, :m].astype(np.float32)
                    ys = self._plane_end_rings[pi, :m].astype(np.float32)
                    order = np.argsort(xs)
                    ax.plot(
                        xs[order],
                        ys[order],
                        color="gray",
                        lw=0.5,
                        alpha=0.5,
                    )

        # Annotate each plane at the group centroid with its plane index.
        for pi in range(self._num_planes):
            m = int(self._plane_multiplicity[pi])
            xs = [int(self._plane_start_rings[pi, k]) for k in range(m)]
            ys = [int(self._plane_end_rings[pi, k]) for k in range(m)]
            cx = float(np.mean(xs))
            cy = float(np.mean(ys))
            ax.text(
                cx,
                cy,
                str(pi),
                ha="center",
                va="center",
                fontsize=plane_index_fontsize,
                color="black",
                fontweight="bold",
                zorder=10,
            )

        ax.set_xlabel("start ring")
        ax.set_ylabel("end ring")
        ax.set_title(f"Michelogram\n(span={S}, max Dring={D})", fontsize="small")
        ax.set_aspect("equal")
        ax.set_xlim(-0.5, R - 0.5)
        ax.set_ylim(-0.5, R - 0.5)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_rings={self._num_rings}, "
            f"max_ring_difference={self._max_ring_difference}, "
            f"span={self._span})"
        )


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
        self._all_block_pairs = self.xp.asarray(all_block_pairs, device=self.dev)
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

        num_selected = block_pair_nums.shape[0]

        # get start and end block indices for all selected block pairs
        bp = self.xp.take(
            self._all_block_pairs, block_pair_nums, axis=0
        )  # (num_selected, 2)
        start_blocks = bp[:, 0]  # (num_selected,)
        end_blocks = bp[:, 1]  # (num_selected,)

        # build the within-block endpoint index pairs via meshgrid (computed once)
        tmp = self.xp.arange(self._num_lorendpoints_per_block, device=self.dev)
        a, b = self.xp.meshgrid(tmp, tmp, indexing="ij")
        a = self.xp.reshape(a, (-1,))  # (num_lors_per_block_pair,)
        b = self.xp.reshape(b, (-1,))  # (num_lors_per_block_pair,)

        # flat index over all (block_pair, lor) combinations
        lor_idx = self.xp.arange(
            num_selected * self._num_lors_per_block_pair, device=self.dev
        )
        within_pair_idx = lor_idx % self._num_lors_per_block_pair
        pair_idx = lor_idx // self._num_lors_per_block_pair

        # tile endpoint indices and repeat block indices across all LORs
        a_all = self.xp.take(a, within_pair_idx, axis=0)
        b_all = self.xp.take(b, within_pair_idx, axis=0)
        start_blocks_all = self.xp.take(start_blocks, pair_idx, axis=0)
        end_blocks_all = self.xp.take(end_blocks, pair_idx, axis=0)

        xstart = self.scanner.get_lor_endpoints(start_blocks_all, a_all)
        xend = self.scanner.get_lor_endpoints(end_blocks_all, b_all)

        return xstart, xend

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
    "plane", "view" and "radial" axis."""

    def __init__(
        self,
        scanner: RegularPolygonPETScannerGeometry,
        michelogram: Michelogram | None = None,
        radial_trim: int = 3,
        sinogram_order: SinogramSpatialAxisOrder = SinogramSpatialAxisOrder.RVP,
        zig_zag_order: SinogramZigZagOrder = SinogramZigZagOrder.END_FIRST,
    ) -> None:
        """

        Parameters
        ----------
        scanner : RegularPolygonPETScannerGeometry
            a regular polygon PET scanner.
        michelogram : Michelogram, optional
            the axial plane layout — the single source of truth for the
            spanning combinatorics (segments, axial midpoints, ring-pair
            grouping, ordering).  If ``None`` (default), a span-1 layout with
            no constraint on the ring difference is used, i.e.
            ``Michelogram(scanner.num_rings, scanner.num_rings - 1, span=1)``.
            The Michelogram must have ``num_rings == scanner.num_rings``.
        radial_trim : int, optional
            number of geometrial LORs to disregard in the radial direction.
            Defaults to 3.
        sinogram_order : SinogramSpatialAxisOrder, optional
            the order of the sinogram axes.  Defaults to
            ``SinogramSpatialAxisOrder.RVP``.
        zig_zag_order : SinogramZigZagOrder, optional
            the zig-zag ordering convention for in-ring detector pairs.
            Defaults to ``SinogramZigZagOrder.END_FIRST``.
        """

        super().__init__(scanner)

        if michelogram is None:
            michelogram = Michelogram(
                num_rings=scanner.num_rings,
                max_ring_difference=scanner.num_rings - 1,
                span=1,
            )
        elif michelogram.num_rings != scanner.num_rings:
            raise ValueError(
                f"michelogram.num_rings ({michelogram.num_rings}) must equal "
                f"scanner.num_rings ({scanner.num_rings})"
            )

        self._scanner = scanner
        self._radial_trim = radial_trim
        self._michelogram = michelogram
        self._max_ring_difference = self._michelogram.max_ring_difference
        self._span = self._michelogram.span

        self._num_rad = scanner.num_lor_endpoints_per_ring - 1 - 2 * self._radial_trim
        self._num_views = scanner.num_lor_endpoints_per_ring // 2

        self._sinogram_order = sinogram_order
        self._zig_zag_order = zig_zag_order

        # declare all attributes set by the setup methods so they are
        # visible in __init__
        self._num_planes: int = 0
        # None only when span > 1; the properties guard with AttributeError
        # before returning.
        self._start_plane_index: Array | None = None
        self._end_plane_index: Array | None = None
        # always set to a real Array by the setup methods
        self._start_plane_z: Array = None  # type: ignore[assignment]
        self._end_plane_z: Array = None  # type: ignore[assignment]
        self._plane_multiplicity: Array = None  # type: ignore[assignment]
        self._plane_segment: Array = None  # type: ignore[assignment]
        self._start_in_ring_index: Array = None  # type: ignore[assignment]
        self._end_in_ring_index: Array = None  # type: ignore[assignment]

        self._setup_plane_data()
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
    def michelogram(self) -> Michelogram:
        """The :class:`Michelogram` describing the axial plane layout.

        This is the single source of truth for the spanning combinatorics
        (segments, axial midpoints, ring-pair grouping, ordering).  Useful
        for visualization, axial compression operators, or any user code
        that needs access to the integer ring-pair structure.
        """
        return self._michelogram

    @property
    def start_plane_index(self) -> Array:
        """start ring index for all planes (only defined for span=1)"""
        if self._span > 1:
            raise AttributeError(
                "start_plane_index is not defined for span > 1. Use start_plane_z instead."
            )
        assert self._start_plane_index is not None
        return self._start_plane_index

    @property
    def end_plane_index(self) -> Array:
        """end ring index for all planes (only defined for span=1)"""
        if self._span > 1:
            raise AttributeError(
                "end_plane_index is not defined for span > 1. Use end_plane_z instead."
            )
        assert self._end_plane_index is not None
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
    def zig_zag_order(self) -> SinogramZigZagOrder:
        """the zig-zag ordering convention for in-ring detector pairs"""
        return self._zig_zag_order

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

    def _setup_plane_data(self) -> None:
        """Project the Michelogram's per-plane data onto the scanner's xp/device.

        Reads ``plane_segment`` and ``plane_multiplicity`` directly from the
        Michelogram and casts them to ``xp`` arrays on the scanner's device.
        Computes ``start_plane_z`` / ``end_plane_z`` by averaging the scanner's
        ring positions over each plane's contributing ring pairs via
        :meth:`Michelogram.average_z_per_plane`.

        For ``span == 1`` the per-plane data is single-valued, so
        ``start_plane_index`` and ``end_plane_index`` are exposed as 1-D
        arrays of ring indices (the first column of the Michelogram's padded
        layout).  For ``span > 1`` they remain ``None`` and the
        :attr:`start_plane_index` / :attr:`end_plane_index` properties raise
        ``AttributeError``; the padded per-plane ring indices are available
        via ``self.michelogram.plane_start_rings`` / ``plane_end_rings``.
        """
        m = self._michelogram
        xp = self.xp
        dev = self.dev

        self._num_planes = m.num_planes
        self._plane_segment = xp.asarray(m.plane_segment, device=dev)
        self._plane_multiplicity = xp.asarray(m.plane_multiplicity, device=dev)

        ring_positions_np = np.asarray(
            to_numpy_array(self._scanner.ring_positions), dtype=np.float64
        )
        start_z, end_z = m.average_z_per_plane(ring_positions_np)
        self._start_plane_z = xp.asarray(start_z, device=dev)
        self._end_plane_z = xp.asarray(end_z, device=dev)

        if self._span == 1:
            self._start_plane_index = xp.asarray(
                m.plane_start_rings[:, 0], device=dev
            )
            self._end_plane_index = xp.asarray(
                m.plane_end_rings[:, 0], device=dev
            )
        else:
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

        # slice for radial trimming; -0 == 0 in Python so guard explicitly
        trim = self._radial_trim
        rad_slc = slice(trim, -trim if trim > 0 else None)

        for view in np.arange(self._num_views):
            if self._zig_zag_order is SinogramZigZagOrder.END_FIRST:
                # end crystal steps first: (0,n-1),(0,n-2),(1,n-2),(1,n-3),...
                start_seq = self.xp.arange(m - 1) // 2
                end_seq = self.xp.concat(
                    (self.xp.asarray([-1]), -((self.xp.arange(m - 2) + 4) // 2))
                )
            else:
                # start crystal steps first: (0,n-1),(1,n-1),(1,n-2),(2,n-2),...
                start_seq = (self.xp.arange(m - 1) + 1) // 2
                end_seq = self.xp.concat(
                    (self.xp.asarray([-1]), -((self.xp.arange(m - 2) + 3) // 2))
                )

            self._start_in_ring_index[view, :] = self.xp.astype(
                (start_seq - int(view))[rad_slc],
                self.xp.int32,
            )
            self._end_in_ring_index[view, :] = self.xp.astype(
                (end_seq - int(view))[rad_slc],
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
            xstart = self.xp.asarray(xstart_2d, copy=True)
            xend = self.xp.asarray(xend_2d, copy=True)

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

    def show_tof_bins(
        self,
        ax: Axes3D,
        tof_parameters: TOFParameters,
        views: int | Array | None = None,
        plane: int = 0,
        show_endpoints: bool = True,
        bin_cmap: str = "seismic",
        show_bin_labels: bool = False,
        label_fontsize: float = 8.0,
        lw: float = 2.0,
        show_colorbar: bool = False,
    ) -> None:
        """Visualise the TOF bin grid for the specified sinogram views and plane.

        Each LOR is drawn as a sequence of coloured line segments — one per
        TOF bin — directly along the LOR ("zebra" style).  Bin colour runs
        from blue (bin 0, xstart side) to red (bin N−1, xend side) via
        ``bin_cmap``.  Bins whose extent falls completely outside the physical
        LOR (i.e. beyond the detector positions) are silently skipped, so
        short edge LORs naturally show fewer coloured segments than central
        ones.

        Parameters
        ----------
        ax : Axes3D
            3-D matplotlib axes to draw on.  The caller is responsible for
            creating the figure and axes.
        tof_parameters : TOFParameters
            TOF bin geometry (number of bins, bin width, centre offset).
        views : int, array-like, or None
            Sinogram view index / indices to draw.

            * ``int`` — draw only that view.
            * array-like — draw those specific views.
            * ``None`` (default) — draw the single middle view
              (``num_views // 2``).

        plane : int
            Sinogram plane index (axial ring pair).  Default ``0``.
        show_endpoints : bool
            Call :meth:`~.RegularPolygonPETScannerGeometry.show_lor_endpoints`
            to annotate detector positions.  Default ``True``.
        bin_cmap : str
            Matplotlib colourmap for the bin segments.  Default ``"seismic"``.
        show_bin_labels : bool
            Annotate the bin numbers on the central LOR of the drawn set.
            Default ``False`` (useful only for scanners with few LORs).
        label_fontsize : float
            Font size for bin labels when ``show_bin_labels=True``.
        lw : float
            Line width of the coloured bin segments.  Default ``2.0``.
        show_colorbar : bool
            Add a colorbar mapping bin index to colour.  Default ``False``.
        """
        num_tof_bins  = tof_parameters.num_tofbins
        tofbin_width  = tof_parameters.tofbin_width
        tofcenter_off = tof_parameters.tofcenter_offset

        fig = ax.get_figure()

        if views is None:
            views = self.xp.asarray([self.num_views // 2], device=self.dev)
        elif isinstance(views, int):
            views = self.xp.asarray([views], device=self.dev)
        else:
            views = self.xp.asarray(views, device=self.dev)

        if show_endpoints:
            self.scanner.show_lor_endpoints(ax, annotation_fontsize=8)

        xstart_all, xend_all = self.get_lor_coordinates(views=views)
        xs_np = to_numpy_array(xstart_all)
        xe_np = to_numpy_array(xend_all)

        p_ax     = self.plane_axis_num
        xs_plane = np.take(xs_np, [plane], axis=p_ax).squeeze(axis=p_ax)
        xe_plane = np.take(xe_np, [plane], axis=p_ax).squeeze(axis=p_ax)
        xs_flat  = xs_plane.reshape(-1, 3)
        xe_flat  = xe_plane.reshape(-1, 3)

        bin_colors = plt.get_cmap(bin_cmap)(np.linspace(0, 1, num_tof_bins))

        sym_hat = np.zeros(3)
        sym_hat[self.scanner.symmetry_axis] = 1.0

        n_lors    = xs_flat.shape[0]
        label_idx = n_lors // 2

        for idx in range(n_lors):
            xs      = xs_flat[idx]
            xe      = xe_flat[idx]
            lor_vec = xe - xs
            lor_len = np.linalg.norm(lor_vec)
            lor_dir  = lor_vec / lor_len
            midpoint = (xs + xe) / 2
            lor_half = lor_len / 2.0

            for k in range(num_tof_bins):
                t_a = (k     - num_tof_bins / 2) * tofbin_width + tofcenter_off
                t_b = (k + 1 - num_tof_bins / 2) * tofbin_width + tofcenter_off
                t_a = max(t_a, -lor_half)
                t_b = min(t_b,  lor_half)
                if t_a >= t_b:
                    continue
                p1 = midpoint + t_a * lor_dir
                p2 = midpoint + t_b * lor_dir
                ax.plot(
                    [p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    color=bin_colors[k], lw=lw, solid_capstyle="butt", zorder=3,
                )

            if show_bin_labels and idx == label_idx:
                lor_proj = lor_dir - np.dot(lor_dir, sym_hat) * sym_hat
                proj_len = np.linalg.norm(lor_proj)
                if proj_len > 1e-6:
                    lor_proj /= proj_len
                perp     = np.cross(sym_hat, lor_proj)
                perp_len = np.linalg.norm(perp)
                if perp_len > 1e-6:
                    perp /= perp_len
                label_off = perp * self.scanner.radius * 0.12

                for k in range(num_tof_bins):
                    t_c = (k - (num_tof_bins - 1) / 2) * tofbin_width + tofcenter_off
                    if abs(t_c) > lor_half:
                        continue
                    lc = midpoint + t_c * lor_dir + label_off
                    ax.text(lc[0], lc[1], lc[2], str(k),
                            fontsize=label_fontsize, ha="center")

        ax.set_xlabel("x (mm)")
        ax.set_ylabel("y (mm)")
        ax.set_zlabel("z (mm)")

        if show_colorbar:
            from matplotlib.colors import Normalize
            import matplotlib.cm as mpl_cm

            norm = Normalize(vmin=0, vmax=num_tof_bins - 1)
            sm   = mpl_cm.ScalarMappable(cmap=bin_cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, shrink=0.4, pad=0.05, label="TOF bin")
            cbar.set_ticks(range(num_tof_bins))

    def show_michelogram(
        self,
        ax: Axes,
        show_merge_lines: bool = True,
        **kwargs,
    ) -> None:
        """Visualize the Michelogram.

        Thin wrapper around :meth:`Michelogram.show`; see that method for full
        documentation of arguments.
        """
        self._michelogram.show(ax, show_merge_lines=show_merge_lines, **kwargs)

    def show_segment_lors(
        self,
        axs=None,
        uncompressed_lor_kwargs: dict | None = None,
        compressed_lor_kwargs: dict | None = None,
    ):
        """Side-view LOR diagram per segment with the Michelogram inset.

        Thin wrapper around :meth:`Michelogram.show_segment_lors`; this
        method supplies the scanner's ring positions automatically.  See
        :meth:`Michelogram.show_segment_lors` for full documentation.
        """
        ring_positions_np = np.asarray(
            to_numpy_array(self._scanner.ring_positions), dtype=np.float64
        )
        return self._michelogram.show_segment_lors(
            ring_positions_np,
            axs=axs,
            uncompressed_lor_kwargs=uncompressed_lor_kwargs,
            compressed_lor_kwargs=compressed_lor_kwargs,
        )

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


class SinogramAxialCompressionOperator(LinearOperator):
    """Linear operator that axially compresses a span-1 PET sinogram into a higher odd span.

    For an input :class:`RegularPolygonPETLORDescriptor` with ``span=1`` and a
    target odd span :math:`S`, every span-1 ring pair
    :math:`(s, e)` is assigned to an output bin
    :math:`(\\text{segment}, \\text{axial midpoint})` where

    * ``segment`` is determined by the ring difference :math:`rd = e - s` under
      target span :math:`S`
      :meth:`RegularPolygonPETLORDescriptor._ring_diff_to_segment`),
    * ``axial midpoint`` is :math:`s + e` (an integer equal to twice the actual
      midpoint).

    All span-1 ring pairs sharing the same
    :math:`(\\text{segment}, s + e)` collapse into a single output plane.

    Two reduction modes are supported:

    * ``mode="sum"`` (default).  The output plane is the **sum** of the
      contributing input planes:

      .. math::

          y_n \\;=\\; \\sum_{p_1 \\in \\mathcal{G}(n)} x_{p_1}
          \\qquad
          \\left(G^T y\\right)_{p_1} \\;=\\; y_{\\,\\tau(p_1)}\\,.

      This is the natural reduction for **counts-like** sinograms — emission
      data, measured counts, randoms, etc. — which add when ring pairs are
      grouped together.

    * ``mode="average"``.  The output plane is the **mean** of the
      contributing input planes:

      .. math::

          y_n \\;=\\; \\frac{1}{m_n} \\sum_{p_1 \\in \\mathcal{G}(n)} x_{p_1}
          \\qquad
          \\left(G_{\\rm avg}^T y\\right)_{p_1}
              \\;=\\; \\frac{y_{\\,\\tau(p_1)}}{m_{\\,\\tau(p_1)}}\\,.

      This is the natural reduction for **multiplicative-factor** sinograms —
      attenuation factors, sensitivity / normalisation factors, geometric
      efficiency — which should *average* rather than *sum* when ring pairs
      are grouped together.

    In both expressions, :math:`\\mathcal{G}(n)` is the set of input plane
    indices mapped to output plane :math:`n`, :math:`m_n = |\\mathcal{G}(n)|`
    is the plane multiplicity, and :math:`\\tau(p_1)` is the output plane
    index for input plane :math:`p_1`.

    Output plane ordering matches that of
    :class:`RegularPolygonPETLORDescriptor` constructed with the same scanner,
    ``radial_trim``, ``max_ring_difference``, and ``sinogram_order`` but with
    ``span=target_span``.  That companion descriptor is exposed as
    :attr:`out_lor_descriptor` for visualization (e.g. ``show_michelogram``,
    ``show_segment_lors``) or for composing the operator with a span-:math:`S`
    projector.

    The closed-form operator 2-norms are

    .. math::

        \\|G_{\\rm sum}\\|_2 = \\sqrt{\\max_n m_n}\\,, \\qquad
        \\|G_{\\rm avg}\\|_2 = 1 / \\sqrt{\\min_n m_n}\\,,

    derived from :math:`G_{\\rm sum} G_{\\rm sum}^T = \\operatorname{diag}(m_n)`
    and :math:`G_{\\rm avg} G_{\\rm avg}^T = \\operatorname{diag}(1/m_n)`.
    :meth:`norm` returns these directly without power iteration.

    Parameters
    ----------
    lor_descriptor : RegularPolygonPETLORDescriptor
        A ``span=1`` LOR descriptor whose sinogram is to be compressed.
    target_span : int
        Odd integer ``>= 1`` giving the target axial compression.  ``1`` is
        accepted and yields an identity-like operator (each input plane maps to
        a single output plane in the same span-1 order).
    mode : {"sum", "average"}, optional
        Reduction mode.  ``"sum"`` (default) is appropriate for counts-like
        sinograms; ``"average"`` is appropriate for multiplicative-factor
        sinograms such as attenuation or sensitivity factors.
    num_tof_bins : int or None, optional
        If ``None`` (default), the operator acts on the 3D spatial sinogram
        with shape :attr:`RegularPolygonPETLORDescriptor.spatial_sinogram_shape`.
        If a positive integer, the operator acts on a 4D TOF sinogram whose
        trailing axis (size ``num_tof_bins``) is the TOF axis and is passed
        through unchanged.

    Examples
    --------
    >>> import array_api_compat.numpy as xp
    >>> import parallelproj.pet_scanners as pps
    >>> import parallelproj.pet_lors as ppl
    >>> scanner = pps.RegularPolygonPETScannerGeometry(
    ...     xp, "cpu", radius=65.0, num_sides=12, num_lor_endpoints_per_side=4,
    ...     lor_spacing=4.0, ring_positions=xp.asarray([0.0, 1.0, 2.0]),
    ...     symmetry_axis=2,
    ... )
    >>> lor_s1 = ppl.RegularPolygonPETLORDescriptor(
    ...     scanner, ppl.Michelogram(scanner.num_rings, 2, span=1), radial_trim=1,
    ... )
    >>> comp = ppl.SinogramAxialCompressionOperator(lor_s1, target_span=3)
    >>> comp.in_shape, comp.out_shape  # doctest: +SKIP
    ((..., ..., 9), (..., ..., 7))
    >>> comp.adjointness_test(xp, "cpu")
    True
    """

    def __init__(
        self,
        lor_descriptor: RegularPolygonPETLORDescriptor,
        target_span: int,
        mode: str = "sum",
        num_tof_bins: int | None = None,
    ) -> None:
        if not isinstance(lor_descriptor, RegularPolygonPETLORDescriptor):
            raise TypeError("lor_descriptor must be a RegularPolygonPETLORDescriptor")
        if lor_descriptor.span != 1:
            raise ValueError("input lor_descriptor must have span=1")
        if not isinstance(target_span, int) or target_span < 1 or target_span % 2 == 0:
            raise ValueError("target_span must be an odd positive integer")
        if mode not in ("sum", "average"):
            raise ValueError(
                f"mode must be 'sum' or 'average', got {mode!r}"
            )
        if num_tof_bins is not None and (
            not isinstance(num_tof_bins, int) or num_tof_bins < 1
        ):
            raise ValueError("num_tof_bins must be a positive integer or None")

        super().__init__()

        self._lor_descriptor = lor_descriptor
        self._target_span = int(target_span)
        self._mode = mode
        self._num_tof_bins = num_tof_bins

        self._xp = lor_descriptor.xp
        self._dev = lor_descriptor.dev
        self._plane_axis = lor_descriptor.plane_axis_num

        # Build the target Michelogram exactly once and reuse it for both
        # the companion descriptor and the compression index maps below.
        target_michelogram = Michelogram(
            num_rings=lor_descriptor.scanner.num_rings,
            max_ring_difference=lor_descriptor.max_ring_difference,
            span=self._target_span,
        )

        self._out_lor_descriptor = RegularPolygonPETLORDescriptor(
            scanner=lor_descriptor.scanner,
            michelogram=target_michelogram,
            radial_trim=lor_descriptor.radial_trim,
            sinogram_order=lor_descriptor.sinogram_order,
        )

        self._build_index_maps(target_michelogram)

        # in/out shapes honour sinogram_order's plane_axis_num and optional TOF.
        spatial_in = tuple(lor_descriptor.spatial_sinogram_shape)
        spatial_out = tuple(self._out_lor_descriptor.spatial_sinogram_shape)
        if num_tof_bins is None:
            self._in_shape = spatial_in
            self._out_shape = spatial_out
        else:
            self._in_shape = spatial_in + (int(num_tof_bins),)
            self._out_shape = spatial_out + (int(num_tof_bins),)

    def _build_index_maps(self, target_michelogram: Michelogram) -> None:
        """Build the gather/scatter index structures from the Michelogram.

        All the combinatorial work — segment assignment, ring-pair grouping,
        STIR-standard plane ordering, padded index construction — lives on
        :class:`Michelogram`.  This method just converts those numpy arrays
        to the descriptor's ``xp`` and ``dev`` and stores them.

        ``target_michelogram`` is the pre-built target Michelogram already
        used to construct the companion span-N descriptor, reused here so
        the layout is built only once per operator.

        Stores on ``self``:

        * ``_target_for_p1`` : shape ``(num_planes_1,)``, the output plane
          index for every input plane.  Used by :meth:`_adjoint`.
        * ``_idx2d_flat``    : shape ``(num_planes_n * max_mult,)``, flattened
          gather index.  ``xp.take(x, _idx2d_flat, axis=plane_axis)`` followed
          by a reshape gives ``(num_planes_n, max_mult)`` along the plane axis.
        * ``_mask2d``        : shape ``(num_planes_n, max_mult)``, ``1.0`` for
          valid entries and ``0.0`` for right-padding.  Used to zero-out
          padding contributions in :meth:`_apply`.
        * ``_multiplicity``  : shape ``(num_planes_n,)``, multiplicity of each
          output plane.
        * ``_max_mult``      : the largest plane multiplicity.

        Because the companion span-N descriptor is built from the same
        Michelogram instance, its plane ordering and per-plane multiplicity
        agree with this operator's by construction; no cross-check is needed.
        """
        target_for_p1, idx2d, mask2d, multiplicity = (
            self._lor_descriptor.michelogram.compression_index_maps_to(
                target_michelogram
            )
        )

        num_planes_n = int(multiplicity.shape[0])
        max_mult = int(idx2d.shape[1]) if num_planes_n > 0 else 0

        xp = self._xp
        dev = self._dev

        self._num_planes_1 = int(target_for_p1.shape[0])
        self._num_planes_n = num_planes_n
        self._max_mult = max_mult
        self._min_mult = int(multiplicity.min()) if num_planes_n > 0 else 0
        self._target_for_p1 = xp.asarray(target_for_p1, device=dev)
        # store flat indices because the array API standard only requires
        # 1-D indices for xp.take (multi-dim take is not portable).
        self._idx2d_flat = xp.asarray(idx2d.reshape(-1), device=dev)
        self._mask2d = xp.asarray(mask2d, device=dev)
        self._multiplicity = xp.asarray(multiplicity, device=dev)

        # Pre-computed reciprocals used by mode="average".  Stored as
        # multiplications instead of divisions inside the hot path.
        inv_multiplicity = (1.0 / multiplicity.astype(np.float32)).astype(np.float32)
        self._inv_multiplicity = xp.asarray(inv_multiplicity, device=dev)
        # inv-multiplicity broadcast onto the input-plane axis (one entry per
        # input plane, equal to 1/m_{tau(p1)}).  Used by the average-mode
        # adjoint.
        inv_multiplicity_at_target = inv_multiplicity[target_for_p1]
        self._inv_multiplicity_at_target = xp.asarray(
            inv_multiplicity_at_target, device=dev
        )

    # ------------------------------------------------------------------
    # LinearOperator interface
    # ------------------------------------------------------------------

    @property
    def in_shape(self) -> tuple[int, ...]:
        return self._in_shape

    @property
    def out_shape(self) -> tuple[int, ...]:
        return self._out_shape

    def _apply(self, x: Array) -> Array:
        """Compress along the plane axis.

        For ``mode="sum"``: ``y_n = sum_{p1 in group(n)} x_{p1}``.
        For ``mode="average"``: divide the sum by the per-plane multiplicity.
        """
        xp = self._xp
        ax = self._plane_axis

        # gather all contributing input planes per output plane.  After the
        # 1-D take we still have x.ndim axes, with the plane axis enlarged to
        # num_planes_n * max_mult; the reshape then splits that into
        # (num_planes_n, max_mult).
        gathered = xp.take(x, self._idx2d_flat, axis=ax)

        new_shape = list(x.shape)
        new_shape[ax] = self._num_planes_n
        new_shape.insert(ax + 1, self._max_mult)
        gathered = xp.reshape(gathered, tuple(new_shape))

        # Broadcast the (num_planes_n, max_mult) mask across all other axes.
        mask_shape = [1] * gathered.ndim
        mask_shape[ax] = self._num_planes_n
        mask_shape[ax + 1] = self._max_mult
        gathered = gathered * xp.reshape(self._mask2d, tuple(mask_shape))

        # Sum over the multiplicity axis (at ax + 1).
        result = xp.sum(gathered, axis=ax + 1)

        if self._mode == "average":
            # Divide every output plane by its multiplicity m_n.
            inv_shape = [1] * result.ndim
            inv_shape[ax] = self._num_planes_n
            result = result * xp.reshape(self._inv_multiplicity, tuple(inv_shape))

        return result

    def _adjoint(self, y: Array) -> Array:
        """Expand the compressed sinogram back along the plane axis.

        For ``mode="sum"``: ``x_{p1} = y_{tau(p1)}`` — each input plane gets
        the value of its target output plane.
        For ``mode="average"``: the broadcast value is additionally divided
        by the multiplicity of the target output plane.
        """
        xp = self._xp
        ax = self._plane_axis
        result = xp.take(y, self._target_for_p1, axis=ax)

        if self._mode == "average":
            # Divide every input plane by m_{tau(p1)}.
            inv_shape = [1] * result.ndim
            inv_shape[ax] = self._num_planes_1
            result = result * xp.reshape(
                self._inv_multiplicity_at_target, tuple(inv_shape)
            )

        return result

    def norm(
        self,
        xp: ModuleType,
        dev: str,
        num_iter: int = 30,
        iscomplex: bool = False,
        verbose: bool = False,
    ) -> float:
        """Operator 2-norm in closed form.

        Because each input plane belongs to exactly one output plane,

        * ``mode="sum"``:    :math:`G G^T = \\operatorname{diag}(m_n)`
          and therefore :math:`\\|G\\|_2 = \\sqrt{\\max_n m_n}`.
        * ``mode="average"``: :math:`G_{\\rm avg} G_{\\rm avg}^T
          = \\operatorname{diag}(1/m_n)`
          and therefore :math:`\\|G_{\\rm avg}\\|_2 = 1 / \\sqrt{\\min_n m_n}`.

        Both norms are independent of TOF, ``xp``, and ``dev``; the inherited
        signature is retained for compatibility with
        :class:`LinearOperator.norm` but its arguments (``xp``, ``dev``,
        ``num_iter``, ``iscomplex``, ``verbose``) are ignored.
        """
        if self._mode == "sum":
            return float(np.sqrt(self._max_mult))
        # mode == "average"
        return float(1.0 / np.sqrt(self._min_mult))

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def lor_descriptor(self) -> RegularPolygonPETLORDescriptor:
        """The input (span-1) LOR descriptor."""
        return self._lor_descriptor

    @property
    def out_lor_descriptor(self) -> RegularPolygonPETLORDescriptor:
        """Auto-built companion descriptor whose plane ordering matches this
        operator's output."""
        return self._out_lor_descriptor

    @property
    def target_span(self) -> int:
        """Target span (odd, >= 1)."""
        return self._target_span

    @property
    def mode(self) -> str:
        """Reduction mode, either ``"sum"`` or ``"average"``."""
        return self._mode

    @property
    def num_tof_bins(self) -> int | None:
        """Number of TOF bins, or ``None`` for a non-TOF operator."""
        return self._num_tof_bins

    @property
    def plane_multiplicity(self) -> Array:
        """Number of span-1 planes that collapse into each output plane.

        Shape ``(num_planes_n,)``.  Equals the diagonal of
        :math:`G G^T`.
        """
        return self._multiplicity

    @property
    def target_plane_for_input_plane(self) -> Array:
        """Output plane index for each span-1 input plane.

        Shape ``(num_planes_1,)``.  Useful for the closed-form check
        :math:`(G^T G\\,\\mathbf{1})_{p_1} = m_{\\,\\tau(p_1)}`.
        """
        return self._target_for_p1

    @property
    def max_plane_multiplicity(self) -> int:
        """Largest plane multiplicity (:math:`\\|G\\|_2^2`)."""
        return self._max_mult

    @property
    def num_planes_in(self) -> int:
        """Number of span-1 input planes."""
        return self._num_planes_1

    @property
    def num_planes_out(self) -> int:
        """Number of span-:math:`S` output planes."""
        return self._num_planes_n

    def __str__(self) -> str:
        tof_str = (
            f", {self._num_tof_bins} TOF bins" if self._num_tof_bins is not None else ""
        )
        return (
            f"{self.__class__.__name__}("
            f"target_span={self._target_span}, mode={self._mode!r}, "
            f"num_planes: {self._num_planes_1} -> {self._num_planes_n}, "
            f"max_multiplicity={self._max_mult}{tof_str})"
        )
