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


from ._backend import Array, to_numpy_array

from .operators import LinearOperator
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


class Michelogram:
    """Axial plane layout for a cylindrical PET scanner under odd span.

    Encapsulates the segment / axial-position combinatorics that map every
    valid ring pair :math:`(s, e)` onto a sinogram plane index under
    Siemens / STIR odd-span conventions.

    For span :math:`S` (odd) and a maximum ring difference :math:`D`, each
    ring pair with :math:`|e - s| \\le D` is assigned a segment via
    :meth:`ring_diff_to_segment`.  Ring pairs sharing the same
    :math:`(\\text{segment},\\; s + e)` collapse into a single plane.
    Planes are ordered by :math:`(|\\text{seg}|,\\; -\\text{seg},\\; s + e)`
    — the STIR "standard segment sequence"
    :math:`[0, +1, -1, +2, -2, \\ldots]` with axial bins increasing in
    :math:`s + e` (equivalently in z for equispaced rings).

    The class knows nothing about ring z-positions, scanner radius, or
    sinogram axis ordering — it operates on pure integer indices.  Consumers
    (e.g. :class:`RegularPolygonPETLORDescriptor`,
    :class:`SinogramAxialCompressionOperator`) combine it with the geometry-
    and array-API-specific information they need.

    For span ``= 1`` the layout reduces to the unspanned Michelogram (each
    ring pair is its own plane with :attr:`max_multiplicity` ``== 1``); the
    ordering is the same as STIR's standard segment sequence applied to
    span 1, i.e. ``rd = 0, +1, -1, +2, -2, ...`` with each ring difference
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
        if self._num_planes == 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
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

    def compression_index_maps(
        self, target_span: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Build gather/scatter index maps to a higher-span Michelogram.

        Returns the integer index structures needed to map planes of this
        Michelogram onto planes of
        ``Michelogram(num_rings, max_ring_difference, span=target_span)``.

        The compression is well-defined only when ``target_span / self.span``
        is an *odd positive integer* — under that condition every ring pair
        of any input plane shares the same target plane, so the operation is
        a single-valued gather.  Otherwise some input plane's contributing
        ring pairs would split across multiple target planes and the
        operation is no longer linear; the method raises ``ValueError``.

        For ``self.span = 1`` any odd ``target_span >= 1`` is valid.

        Parameters
        ----------
        target_span : int
            Odd integer ``>= self.span``; additionally
            ``(target_span // self.span)`` must be odd.

        Returns
        -------
        target_for_p1 : np.ndarray, shape ``(self.num_planes,)``, dtype ``int64``
            For each plane of this Michelogram, the corresponding plane index
            in the target Michelogram.
        idx2d : np.ndarray, shape ``(target.num_planes, target_max_mult)``, ``int64``
            For each target plane, the indices in this Michelogram that
            contribute, right-padded with ``0``.  Use ``mask`` to filter.
        mask : np.ndarray, same shape as ``idx2d``, dtype ``float32``
            ``1.0`` for valid entries, ``0.0`` for right-padding.
        target_multiplicity : np.ndarray, shape ``(target.num_planes,)``, ``int32``
            Number of self-planes folded into each target plane.

        Raises
        ------
        ValueError
            If ``target_span`` is not a positive odd integer, if
            ``target_span < self.span``, if ``self.span`` does not divide
            ``target_span``, or if the resulting ratio is even.
        """
        if not isinstance(target_span, int) or target_span < 1 or target_span % 2 == 0:
            raise ValueError("target_span must be an odd positive integer")
        if target_span < self._span:
            raise ValueError(
                f"target_span ({target_span}) must be >= self.span ({self._span})"
            )
        if target_span % self._span != 0:
            raise ValueError(
                f"target_span ({target_span}) must be an integer multiple "
                f"of self.span ({self._span})"
            )
        ratio = target_span // self._span
        if ratio % 2 == 0:
            raise ValueError(
                f"target_span / self.span = {target_span} / {self._span} "
                f"= {ratio} must be an odd integer"
            )

        target = Michelogram(
            self._num_rings, self._max_ring_difference, span=target_span
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

    # ------------------------------------------------------------------
    # Visualization
    # ------------------------------------------------------------------

    def show(self, ax: Axes, show_merge_lines: bool = True, **kwargs) -> None:
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
        **kwargs
            Forwarded to ``ax.scatter`` (e.g. ``s=4``, ``cmap="RdBu_r"``).
        """
        self._draw_axes(ax, show_merge_lines=show_merge_lines, **kwargs)

    def show_segment_lors(
        self,
        ring_positions,
        axs=None,
        uncompressed_lor_kwargs: dict | None = None,
        compressed_lor_kwargs: dict | None = None,
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
                    self._draw_axes(ax)
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

    def _draw_axes(self, ax: Axes, show_merge_lines: bool = True, **kwargs) -> None:
        """Internal helper: draw the Michelogram onto an existing axes.

        Shared between :meth:`show` (main use) and :meth:`show_segment_lors`
        (inset).
        """
        R = self._num_rings
        D = self._max_ring_difference
        S = self._span

        if self._num_planes == 0:
            ax.set_title(
                f"Michelogram\n(span={S}, max Dring={D}, empty)",
                fontsize="small",
            )
            return

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
                fontsize=4,
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

        # declare all attributes set by setup methods so they are visible in __init__
        self._num_planes: int = 0
        # None only when span > 1; properties guard with AttributeError before returning
        self._start_plane_index: Array | None = None
        self._end_plane_index: Array | None = None
        # always set to a real Array by the setup methods
        self._start_plane_z: Array = None  # type: ignore[assignment]
        self._end_plane_z: Array = None  # type: ignore[assignment]
        self._plane_multiplicity: Array = None  # type: ignore[assignment]
        self._plane_segment: Array = None  # type: ignore[assignment]
        self._start_in_ring_index: Array = None  # type: ignore[assignment]
        self._end_in_ring_index: Array = None  # type: ignore[assignment]

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
            Segment number (0 for segment 0, +/-k for outer segments).
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

        assert self._start_plane_index is not None
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
          seg = +/-ceil((|rd|-half_span) / S)   otherwise  (sign follows rd)

        Planes are ordered: segment 0, then +1, -1, +2, -2, ... each sorted by
        increasing axial midpoint (start_ring + end_ring).
        """
        R = self._scanner.num_rings
        D = self._max_ring_difference

        ring_pos = np.asarray(
            to_numpy_array(self._scanner.ring_positions), dtype=np.float64
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

        # Order: seg 0, then +1, -1, +2, -2, ...; within each segment by axial midpoint
        sorted_keys = sorted(
            plane_groups.keys(), key=lambda k: (abs(k[0]), -k[0], k[1])
        )

        self._num_planes = len(sorted_keys)

        start_z = np.empty(self._num_planes, dtype=np.float32)
        end_z = np.empty(self._num_planes, dtype=np.float32)
        mult = np.empty(self._num_planes, dtype=np.int32)
        seg_arr = np.empty(self._num_planes, dtype=np.int32)

        for pi, key in enumerate(sorted_keys):
            pairs = plane_groups[key]
            start_z[pi] = np.mean([ring_pos[s] for s, _ in pairs])
            end_z[pi] = np.mean([ring_pos[e] for _, e in pairs])
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

    def _draw_michelogram(self, ax, show_merge_lines: bool = True, **kwargs) -> None:
        """Draw the Michelogram onto an existing axes.

        Internal helper shared by :meth:`show_michelogram` and
        :meth:`show_segment_lors`.
        """

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

        start_arr = np.array(start_list, dtype=np.float32)
        end_arr = np.array(end_list, dtype=np.float32)
        seg_arr = np.array(seg_list, dtype=np.int32)
        z_arr = np.array(z_list, dtype=np.int32)

        abs_seg_arr = np.abs(seg_arr)
        n_colors = int(abs_seg_arr.max()) + 1

        base_cmap = plt.get_cmap("tab10" if n_colors <= 10 else "tab20")
        cmap = ListedColormap([base_cmap(i) for i in range(n_colors)])
        norm = BoundaryNorm(np.arange(-0.5, n_colors, 1.0), cmap.N)

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
            c=abs_seg_arr.astype(np.float32),
            cmap=cmap,
            norm=norm,
            **kwargs,
        )

        if show_merge_lines and self._span > 1:
            for seg, z_int in set(zip(seg_list, z_list)):
                mask = (seg_arr == seg) & (z_arr == z_int)
                xs, ys = start_arr[mask], end_arr[mask]
                if xs.size > 1:
                    order = np.argsort(xs)
                    ax.plot(xs[order], ys[order], color="gray", lw=0.5, alpha=0.5)

        # annotate each compressed plane with its plane index at the group centroid
        for pi, key in enumerate(sorted_keys):
            pairs = plane_groups[key]
            cx = np.mean([s for s, _ in pairs])
            cy = np.mean([e for _, e in pairs])
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
            f"Michelogram\n(span={self._span}, max Dring={D})", fontsize="small"
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

        * **columns** - indexed by ``abs(segment)``: 0, 1, 2, ...
        * **row 0** - non-negative segments (0, +1, +2, ...)
        * **row 1, col 0** - Michelogram
        * **row 1, col >= 1** - negative segments (-1, -2, ...)

        Each LOR subplot shows:

        * **solid black lines** - every individual ring-pair LOR in the segment
          (the "uncompressed" planes);
        * **dashed coloured lines** - the compressed (axially-averaged) LOR geometry
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

        R = self._scanner.num_rings
        D = self._max_ring_difference
        ring_pos = np.asarray(
            to_numpy_array(self._scanner.ring_positions), dtype=np.float64
        )

        seg_arr_np = np.asarray(to_numpy_array(self._plane_segment), dtype=np.int32)
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
            _axs = np.asarray(axs)
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

        start_z_np = np.asarray(to_numpy_array(self._start_plane_z))
        end_z_np = np.asarray(to_numpy_array(self._end_plane_z))

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


class SinogramAxialCompressionOperator(LinearOperator):
    """Linear operator that axially compresses a span-1 PET sinogram into a higher odd span.

    For an input :class:`RegularPolygonPETLORDescriptor` with ``span=1`` and a
    target odd span :math:`S`, every span-1 ring pair
    :math:`(s, e)` is assigned to an output bin
    :math:`(\\text{segment}, \\text{axial midpoint})` where

    * ``segment`` is determined by the ring difference :math:`rd = e - s` under
      target span :math:`S` (matches Siemens / STIR conventions; see
      :meth:`RegularPolygonPETLORDescriptor._ring_diff_to_segment`),
    * ``axial midpoint`` is :math:`s + e` (an integer equal to twice the actual
      midpoint).

    All span-1 ring pairs sharing the same
    :math:`(\\text{segment}, s + e)` collapse into a single output plane.

    .. math::

        y_n \\;=\\; \\sum_{p_1 \\in \\mathcal{G}(n)} x_{p_1}
        \\qquad
        \\left(G^T y\\right)_{p_1} \\;=\\; y_{\\,\\tau(p_1)}

    where :math:`\\mathcal{G}(n)` is the set of input plane indices mapped to
    output plane :math:`n` and :math:`\\tau(p_1)` is the output plane index for
    input plane :math:`p_1`.

    Output plane ordering matches that of
    :class:`RegularPolygonPETLORDescriptor` constructed with the same scanner,
    ``radial_trim``, ``max_ring_difference``, and ``sinogram_order`` but with
    ``span=target_span``.  That companion descriptor is exposed as
    :attr:`out_lor_descriptor` for visualization (e.g. ``show_michelogram``,
    ``show_segment_lors``) or for composing the operator with a span-:math:`S`
    projector.

    Because :math:`G` is a 0/1 row-stochastic gather (each column has exactly
    one ``1``),

    .. math::

        G G^T = \\operatorname{diag}(m_n), \\qquad
        \\|G\\|_2 \\;=\\; \\sqrt{\\max_n m_n}

    where :math:`m_n` is the multiplicity of output plane :math:`n` (the number
    of span-1 planes that collapse into it).  The operator's :meth:`norm` returns
    this closed-form value directly without power iteration.

    Parameters
    ----------
    lor_descriptor : RegularPolygonPETLORDescriptor
        A ``span=1`` LOR descriptor whose sinogram is to be compressed.
    target_span : int
        Odd integer ``>= 1`` giving the target axial compression.  ``1`` is
        accepted and yields an identity-like operator (each input plane maps to
        a single output plane in the same span-1 order).
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
    >>> lor_s1 = ppl.RegularPolygonPETLORDescriptor(scanner, radial_trim=1,
    ...                                             max_ring_difference=2, span=1)
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
        num_tof_bins: int | None = None,
    ) -> None:
        if not isinstance(lor_descriptor, RegularPolygonPETLORDescriptor):
            raise TypeError("lor_descriptor must be a RegularPolygonPETLORDescriptor")
        if lor_descriptor.span != 1:
            raise ValueError("input lor_descriptor must have span=1")
        if not isinstance(target_span, int) or target_span < 1 or target_span % 2 == 0:
            raise ValueError("target_span must be an odd positive integer")
        if num_tof_bins is not None and (
            not isinstance(num_tof_bins, int) or num_tof_bins < 1
        ):
            raise ValueError("num_tof_bins must be a positive integer or None")

        super().__init__()

        self._lor_descriptor = lor_descriptor
        self._target_span = int(target_span)
        self._num_tof_bins = num_tof_bins

        self._xp = lor_descriptor.xp
        self._dev = lor_descriptor.dev
        self._plane_axis = lor_descriptor.plane_axis_num

        # Build the companion span-N descriptor.  Its plane ordering is what
        # we map onto; we'll cross-check our internally computed multiplicity
        # against descriptor's to catch any silent convention drift.
        self._out_lor_descriptor = RegularPolygonPETLORDescriptor(
            scanner=lor_descriptor.scanner,
            radial_trim=lor_descriptor.radial_trim,
            max_ring_difference=lor_descriptor.max_ring_difference,
            sinogram_order=lor_descriptor.sinogram_order,
            span=self._target_span,
        )

        self._build_index_maps()

        # in/out shapes honour sinogram_order's plane_axis_num and optional TOF.
        spatial_in = tuple(lor_descriptor.spatial_sinogram_shape)
        spatial_out = tuple(self._out_lor_descriptor.spatial_sinogram_shape)
        if num_tof_bins is None:
            self._in_shape = spatial_in
            self._out_shape = spatial_out
        else:
            self._in_shape = spatial_in + (int(num_tof_bins),)
            self._out_shape = spatial_out + (int(num_tof_bins),)

    def _build_index_maps(self) -> None:
        """Build the gather/scatter index structures.

        Computes (and stores on ``self``):

        * ``_target_for_p1`` : shape ``(num_planes_1,)``, the output plane
          index for every input plane.  Used by :meth:`_adjoint`.
        * ``_idx2d_flat``    : shape ``(num_planes_n * max_mult,)``, flattened
          index array.  ``xp.take(x, _idx2d_flat, axis=plane_axis)`` followed
          by a reshape gives ``(num_planes_n, max_mult)`` along the plane axis.
        * ``_mask2d``        : shape ``(num_planes_n, max_mult)``, ``1.0`` for
          valid entries and ``0.0`` for right-padding.  Used to zero-out
          padding contributions in :meth:`_apply`.
        * ``_multiplicity``  : shape ``(num_planes_n,)``, multiplicity of each
          output plane.
        * ``_max_mult``      : the largest plane multiplicity.
        """
        lor = self._lor_descriptor
        S = self._target_span
        half_span = (S - 1) // 2

        # Span-1 plane endpoints in the *input* sinogram order.  We deliberately
        # read these from the descriptor rather than recompute, so the operator
        # is locked to whatever the descriptor produces.
        start_idx = np.asarray(to_numpy_array(lor.start_plane_index), dtype=np.int64)
        end_idx = np.asarray(to_numpy_array(lor.end_plane_index), dtype=np.int64)

        rd = end_idx - start_idx
        mid_int = end_idx + start_idx  # s + e (= 2 * midpoint)

        # Segment under the *target* span S (Siemens / STIR convention).
        abs_rd = np.abs(rd)
        seg = np.where(
            abs_rd <= half_span,
            0,
            np.sign(rd) * ((abs_rd - half_span + S - 1) // S),
        ).astype(np.int64)

        num_planes_1 = int(start_idx.shape[0])

        # Group span-1 planes by (segment, s+e).  Sort key matches
        # _setup_spanned_plane_indices so the operator's output plane indexing
        # agrees with the companion span-N descriptor bin-for-bin.
        keys = list(zip(seg.tolist(), mid_int.tolist()))
        unique_keys = sorted(set(keys), key=lambda k: (abs(k[0]), -k[0], k[1]))
        key_to_n = {k: i for i, k in enumerate(unique_keys)}
        target_for_p1 = np.fromiter(
            (key_to_n[k] for k in keys), dtype=np.int64, count=num_planes_1
        )
        num_planes_n = len(unique_keys)

        # Invert: for each output plane, list its contributing input planes.
        groups: list[list[int]] = [[] for _ in range(num_planes_n)]
        for p1, k in enumerate(keys):
            groups[key_to_n[k]].append(p1)
        multiplicity = np.fromiter(
            (len(g) for g in groups), dtype=np.int32, count=num_planes_n
        )
        max_mult = int(multiplicity.max()) if num_planes_n > 0 else 0

        # Padded (num_planes_n, max_mult) gather index + 0/1 mask.  Padding
        # entries hold index 0 with mask 0 so the gather is safe and the
        # padded reads contribute zero to the sum.
        idx2d = np.zeros((num_planes_n, max_mult), dtype=np.int64)
        mask2d = np.zeros((num_planes_n, max_mult), dtype=np.float32)
        for i, g in enumerate(groups):
            if g:
                idx2d[i, : len(g)] = g
                mask2d[i, : len(g)] = 1.0

        # Convention guard: the auto-built companion descriptor must agree on
        # the per-plane multiplicity.  If it doesn't, our output plane ordering
        # has drifted from the descriptor's and downstream visualization tools
        # would mis-align.
        comp_mult = np.asarray(
            to_numpy_array(self._out_lor_descriptor.plane_multiplicity),
            dtype=np.int32,
        )
        if comp_mult.shape != multiplicity.shape or not np.array_equal(
            comp_mult, multiplicity
        ):
            raise RuntimeError(
                "internal: plane multiplicity disagrees with auto-built span-"
                f"{S} descriptor — output plane ordering convention drift"
            )

        xp = self._xp
        dev = self._dev

        self._num_planes_1 = num_planes_1
        self._num_planes_n = num_planes_n
        self._max_mult = max_mult
        self._target_for_p1 = xp.asarray(target_for_p1, device=dev)
        # store flat indices because the array API standard only requires
        # 1-D indices for xp.take (multi-dim take is not portable).
        self._idx2d_flat = xp.asarray(idx2d.reshape(-1), device=dev)
        self._mask2d = xp.asarray(mask2d, device=dev)
        self._multiplicity = xp.asarray(multiplicity, device=dev)

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
        """Compress: ``y_n = sum_{p1 in group(n)} x_{p1}`` along the plane axis."""
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
        return xp.sum(gathered, axis=ax + 1)

    def _adjoint(self, y: Array) -> Array:
        """Expand: replicate each output plane back to every contributing input plane."""
        return self._xp.take(y, self._target_for_p1, axis=self._plane_axis)

    def norm(
        self,
        xp: ModuleType,
        dev: str,
        num_iter: int = 30,
        iscomplex: bool = False,
        verbose: bool = False,
    ) -> float:
        """Operator 2-norm in closed form.

        Because :math:`G` is a 0/1 gather with each input plane belonging to
        exactly one output plane, :math:`G G^T = \\operatorname{diag}(m_n)`
        and therefore :math:`\\|G\\|_2 = \\sqrt{\\max_n m_n}`.  This is
        independent of TOF, ``xp``, and ``dev``; the inherited signature is
        retained for compatibility with :class:`LinearOperator.norm` but its
        arguments (``xp``, ``dev``, ``num_iter``, ``iscomplex``, ``verbose``)
        are ignored.
        """
        return float(np.sqrt(self._max_mult))

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
            f"target_span={self._target_span}, "
            f"num_planes: {self._num_planes_1} -> {self._num_planes_n}, "
            f"max_multiplicity={self._max_mult}{tof_str})"
        )
