"""Manual Mask — AnalysisPipeline that rasterizes user-drawn shapes.

The Analysis page collects shapes (rectangle, ellipse, free-polygon) drawn
on the viewer canvas and stores them keyed by (M position, frame index,
and a Z-slot — either a specific Z slice index or the string ``"all"``
meaning "applies uniformly to every Z slice").

At run time the page injects the per-M shape dict into
``params["frame_shapes"]`` and this pipeline rasterizes them into a
(T, H, W) int32 label stack for visualization, and — when any per-Z
shapes exist — also emits per-label voxel counts so the Results tab can
report true 3D volume rather than the uniform-Z approximation.

Shape dict schema (V1.31+):
    {frame_idx (int): {
        z_key (int | "all"): [
            {"type": "rect" | "ellipse" | "polygon",
             "vertices": [[y, x], ...]},
            ...
        ],
        ...
    }}

Each shape becomes its own ``label_id`` (1, 2, 3, …) within a frame, in a
deterministic order: ``"all"`` shapes first, then z=0, z=1, ….  For
projected files there is only one slot (``"all"``) and the behavior is
identical to V1.30.

Legacy ``{frame_idx: [shapes]}`` (pre-V1.31) is accepted and treated as
``{frame_idx: {"all": shapes}}`` for backward compatibility with sessions
saved by V1.30.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from skimage.draw import ellipse as sk_ellipse
from skimage.draw import polygon as sk_polygon
from skimage.measure import find_contours
from skimage.morphology import disk

from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.core.plugin_registry import ParamSpec


# Default vertex density for "convert to editable polygon" — chosen to feel
# responsive on small ROIs while remaining cheap to drag. The user can keep
# adding implicit detail by editing freehand; this is just the starting count.
DEFAULT_EDIT_VERTEX_COUNT = 24


@AnalysisPipeline.register
class ManualMaskPipeline(AnalysisPipeline):
    """User-drawn binary masks.

    Draw rectangles, ellipses, or free-polygons on each frame in the viewer
    while this pipeline is active; the page collects the shapes and the
    pipeline rasterizes them to a label stack.  Each shape becomes its own
    ``label_id`` so the Results tab reports per-shape area and centroid.
    """

    name = "Manual Mask"
    description = (
        "User-drawn binary masks. Draw rectangles, ellipses, or free-polygons "
        "on each frame using the drawing tools that appear below the "
        "parameters; the pipeline rasterizes them into a label stack. "
        "Each shape on a frame becomes its own label_id."
    )

    def get_params(self) -> List[ParamSpec]:
        return [
            ParamSpec(
                "channel_name", "Channel", "choice", "",
                choices=[],
                tooltip="Channel the manual mask is associated with. "
                        "The mask's frame count comes from this channel's "
                        "(T, H, W) shape.",
            ),
            # Internal state injected by AnalysisPage at run time. Hidden from
            # the ParamEditor UI — the user manipulates it via the canvas.
            # Schema: {t: {z_key: [shapes]}} where z_key is an int Z index or
            # the string "all" meaning "every Z slice".
            ParamSpec("frame_shapes", "", "hidden", default={}),
        ]

    def run(
        self,
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        progress_cb: Optional[Callable[[int], None]] = None,
        cancelled_cb: Optional[Callable[[], bool]] = None,
    ) -> AnalysisResult:
        ch_name = params.get("channel_name") or next(iter(channels))
        arr = channels.get(ch_name)
        if arr is None:
            raise ValueError(
                f"ManualMaskPipeline: channel '{ch_name}' not in {list(channels)}"
            )
        if arr.ndim != 3:
            raise ValueError(
                f"ManualMaskPipeline: expected (T, H, W) array, got shape {arr.shape}"
            )
        T, H, W = arr.shape
        n_zslices = max(1, int(metadata.get("n_zslices") or 1))
        raw_shapes: Dict[Any, Any] = params.get("frame_shapes") or {}
        # Normalise: {t: {z_key: [shapes]}}. Accept legacy {t: [shapes]} too.
        normalised: Dict[int, Dict[Any, List[Dict[str, Any]]]] = {}
        any_per_z = False
        for t_key, frame_val in raw_shapes.items():
            try:
                t = int(t_key)
            except (TypeError, ValueError):
                continue
            if not (0 <= t < T):
                continue
            by_z = _normalise_frame_slot(frame_val)
            if not by_z:
                continue
            normalised[t] = by_z
            if any(k != "all" for k in by_z):
                any_per_z = True

        label_stack = np.zeros((T, H, W), dtype=np.int32)
        voxel_counts: Dict[Tuple[int, int], int] = {}

        total = max(T, 1)
        for t in range(T):
            if cancelled_cb is not None and cancelled_cb():
                break
            by_z = normalised.get(t, {})
            if by_z:
                ordered = _ordered_shapes(by_z)  # [(label_id, z_key, shape), ...]
                # 2D union for visualization. Paint in label order — later
                # shapes overwrite earlier where they overlap.
                for label_id, _z_key, shape in ordered:
                    _rasterize(label_stack[t], shape, label_id, H, W)
                # Per-label voxel counts — needed only when at least one
                # shape lives on a specific Z slice. Otherwise the uniform-Z
                # fallback in results_engine is exact.
                if any_per_z and n_zslices > 1:
                    counts = _voxel_counts_for_frame(ordered, n_zslices, H, W)
                    for label_id, vox in counts.items():
                        voxel_counts[(t, label_id)] = vox
            if progress_cb is not None:
                progress_cb(int(100 * (t + 1) / total))

        n_frames_with = int(sum(1 for t in range(T) if label_stack[t].max() > 0))
        pixel_size = float(metadata.get("pixel_size_um") or 1.0)
        total_objects = sum(
            sum(len(shapes) for shapes in by_z.values())
            for by_z in normalised.values()
        )
        areas_px: List[int] = []
        for t in range(T):
            n_labels = int(label_stack[t].max())
            for label_id in range(1, n_labels + 1):
                a = int((label_stack[t] == label_id).sum())
                if a > 0:
                    areas_px.append(a)
        mean_area_px = float(np.mean(areas_px)) if areas_px else 0.0
        std_area_px = float(np.std(areas_px)) if areas_px else 0.0

        result = AnalysisResult(
            label_masks={ch_name: label_stack},
            measurements=[],  # results_engine recomputes via regionprops
            summary={
                "total_objects": int(total_objects),
                "n_frames_with_objects": n_frames_with,
                "mean_area_px": mean_area_px,
                "std_area_px": std_area_px,
                "mean_area_um2": round(mean_area_px * pixel_size ** 2, 4),
                "std_area_um2": round(std_area_px * pixel_size ** 2, 4),
            },
            # Yellow — distinct from the multi-color palette used by automated
            # instance-segmentation pipelines.
            overlay_color=(255, 220, 0),
            overlay_alpha=0.45,
        )
        if any_per_z and n_zslices > 1:
            result.volumetric_voxel_counts = {ch_name: voxel_counts}
        return result


def rasterize_shapes(frame: np.ndarray,
                     shapes: List[Dict[str, Any]],
                     H: int, W: int) -> None:
    """Paint each shape into ``frame`` with label_id 1..N (in place).

    Public helper so the Analysis page can render a live preview of the
    drawn-but-not-yet-run shapes onto the viewer.
    """
    for label_id, shape in enumerate(shapes, start=1):
        _rasterize(frame, shape, label_id, H, W)


def _normalise_frame_slot(value: Any) -> Dict[Any, List[Dict[str, Any]]]:
    """Coerce a raw frame value into ``{z_key: [shapes]}`` form.

    Accepts either the new dict-of-lists schema or the legacy bare list
    (V1.30 sessions), which is treated as ``{"all": shapes}``.  Returns an
    empty dict if the input is empty or malformed.
    """
    if isinstance(value, list):
        return {"all": list(value)} if value else {}
    if not isinstance(value, dict):
        return {}
    out: Dict[Any, List[Dict[str, Any]]] = {}
    for k, shapes in value.items():
        if not shapes:
            continue
        if k == "all" or k == "ALL":
            out["all"] = list(shapes)
        else:
            try:
                out[int(k)] = list(shapes)
            except (TypeError, ValueError):
                continue
    return out


def _ordered_shapes(
    by_z: Dict[Any, List[Dict[str, Any]]],
) -> List[Tuple[int, Any, Dict[str, Any]]]:
    """Enumerate shapes across z-slots in a stable label order.

    Order: ``"all"`` first, then z=0, z=1, ….  Returns
    ``[(label_id, z_key, shape), ...]`` with label_id starting at 1.
    """
    ordered_keys: List[Any] = []
    if "all" in by_z:
        ordered_keys.append("all")
    for k in sorted(k for k in by_z.keys() if k != "all"):
        ordered_keys.append(k)
    out: List[Tuple[int, Any, Dict[str, Any]]] = []
    label_id = 1
    for k in ordered_keys:
        for shape in by_z.get(k, []):
            out.append((label_id, k, shape))
            label_id += 1
    return out


def _voxel_counts_for_frame(
    ordered: List[Tuple[int, Any, Dict[str, Any]]],
    n_zslices: int,
    H: int,
    W: int,
) -> Dict[int, int]:
    """Compute true 3D voxel counts per label_id for one frame.

    Rasterizes the per-Z mask for each Z slice (taking into account
    label-order overwrites for overlaps) and accumulates per-label voxel
    counts.  This is the path used when at least one shape is bound to a
    specific Z slice; the projected-uniform case bypasses it entirely.
    """
    counts: Dict[int, int] = {}
    for z in range(n_zslices):
        slice_mask = np.zeros((H, W), dtype=np.int32)
        for label_id, z_key, shape in ordered:
            if z_key == "all" or z_key == z:
                _rasterize(slice_mask, shape, label_id, H, W)
        # Tally each label_id present in this z slice.
        if slice_mask.max() == 0:
            continue
        labels_here, areas_here = np.unique(slice_mask, return_counts=True)
        for lab, area in zip(labels_here, areas_here):
            if lab == 0:
                continue
            counts[int(lab)] = counts.get(int(lab), 0) + int(area)
    return counts


def _rasterize(frame: np.ndarray, shape: Dict[str, Any],
               label_id: int, H: int, W: int) -> None:
    """Paint a single shape into a frame at the given label id (in place)."""
    s_type = shape.get("type")
    raw_verts = shape.get("vertices") or []
    if not raw_verts:
        return
    verts = np.asarray(raw_verts, dtype=float)

    if s_type == "rect" and verts.shape[0] == 2:
        y0, y1 = sorted([verts[0, 0], verts[1, 0]])
        x0, x1 = sorted([verts[0, 1], verts[1, 1]])
        iy0 = max(0, int(round(y0)))
        ix0 = max(0, int(round(x0)))
        iy1 = min(H, int(round(y1)) + 1)
        ix1 = min(W, int(round(x1)) + 1)
        if iy1 > iy0 and ix1 > ix0:
            frame[iy0:iy1, ix0:ix1] = label_id
        return

    if s_type == "ellipse" and verts.shape[0] == 2:
        cy = (verts[0, 0] + verts[1, 0]) / 2.0
        cx = (verts[0, 1] + verts[1, 1]) / 2.0
        ry = abs(verts[1, 0] - verts[0, 0]) / 2.0
        rx = abs(verts[1, 1] - verts[0, 1]) / 2.0
        if ry <= 0 or rx <= 0:
            return
        rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(H, W))
        frame[rr, cc] = label_id
        return

    if s_type == "polygon" and verts.shape[0] >= 3:
        rr, cc = sk_polygon(verts[:, 0], verts[:, 1], shape=(H, W))
        frame[rr, cc] = label_id
        return


# ── Edit helpers (V1.30+): convert / resample / expand polygons ───────────────

def shape_to_editable_polygon(
    shape: Dict[str, Any],
    n_vertices: int = DEFAULT_EDIT_VERTEX_COUNT,
) -> List[List[float]]:
    """Convert a shape into a polygon with ~``n_vertices`` evenly-spaced vertices.

    The result is a list of ``[y, x]`` pairs ready to be stored back in the
    shape's ``vertices`` slot.  Rectangles become 4-corner polygons resampled
    at uniform arc length; ellipses become parametric N-point polygons;
    polygons are resampled (interpolated if sparse, decimated if dense).

    This is the "automatically created uniformly and at medium density around
    the mask" step from the feature spec.
    """
    s_type = shape.get("type")
    raw_verts = shape.get("vertices") or []
    verts = np.asarray(raw_verts, dtype=float)

    if s_type == "rect" and verts.shape[0] == 2:
        y0, y1 = sorted([verts[0, 0], verts[1, 0]])
        x0, x1 = sorted([verts[0, 1], verts[1, 1]])
        corners = np.array([
            [y0, x0], [y0, x1], [y1, x1], [y1, x0], [y0, x0],
        ])
        return _resample_closed_polygon(corners, n_vertices)

    if s_type == "ellipse" and verts.shape[0] == 2:
        cy = (verts[0, 0] + verts[1, 0]) / 2.0
        cx = (verts[0, 1] + verts[1, 1]) / 2.0
        ry = abs(verts[1, 0] - verts[0, 0]) / 2.0
        rx = abs(verts[1, 1] - verts[0, 1]) / 2.0
        if ry <= 0 or rx <= 0:
            return [[float(v[0]), float(v[1])] for v in verts]
        theta = np.linspace(0.0, 2.0 * np.pi, n_vertices, endpoint=False)
        out = [
            [float(cy + ry * np.sin(t)), float(cx + rx * np.cos(t))]
            for t in theta
        ]
        return out

    if s_type == "polygon" and verts.shape[0] >= 3:
        closed = np.vstack([verts, verts[:1]])
        return _resample_closed_polygon(closed, n_vertices)

    return [[float(v[0]), float(v[1])] for v in verts]


def expand_polygon_uniformly(
    vertices: List[List[float]],
    expand_px: float,
    H: int,
    W: int,
    n_vertices: int = DEFAULT_EDIT_VERTEX_COUNT,
) -> List[List[float]]:
    """Uniformly grow (or shrink, if negative) a polygon by ``expand_px`` pixels.

    Implementation: rasterize the polygon to a binary mask, apply
    morphological dilation / erosion with a disk structuring element of
    radius ``|expand_px|``, then re-extract the outline via marching squares
    and resample to ``n_vertices`` evenly-spaced points.  This is robust for
    arbitrary polygon shapes (convex, concave, freehand) and behaves like a
    proper "uniform extend in all directions" operation.

    Returns the new vertex list.  Falls back to the input if the operation
    cannot be performed (e.g., erosion eats the entire mask).
    """
    if not vertices or abs(expand_px) < 1e-9:
        return list(vertices)

    verts = np.asarray(vertices, dtype=float)
    if verts.shape[0] < 3:
        return list(vertices)

    rr, cc = sk_polygon(verts[:, 0], verts[:, 1], shape=(H, W))
    mask = np.zeros((H, W), dtype=bool)
    mask[rr, cc] = True
    if not mask.any() and expand_px < 0:
        return list(vertices)

    radius = max(1, int(round(abs(expand_px))))
    selem = disk(radius)
    if expand_px > 0:
        mask = binary_dilation(mask, structure=selem)
    else:
        mask = binary_erosion(mask, structure=selem)

    if not mask.any():
        return list(vertices)

    # Pad with a 1-px False border so contours never run off-edge — find_contours
    # would otherwise yield open paths along image boundaries.
    padded = np.zeros((H + 2, W + 2), dtype=float)
    padded[1:-1, 1:-1] = mask.astype(float)
    contours = find_contours(padded, 0.5)
    if not contours:
        return list(vertices)
    # Largest contour by point count = outer boundary of the (possibly fattened)
    # main connected component. Subtract the padding offset.
    biggest = max(contours, key=len) - 1.0
    return _resample_closed_polygon(biggest, n_vertices)


def _resample_closed_polygon(verts: np.ndarray, n: int) -> List[List[float]]:
    """Resample a closed polyline (first == last) to ``n`` evenly-spaced vertices.

    "Evenly-spaced" is by arc length. Returns a list of ``[y, x]`` pairs of
    length exactly ``n``, with no explicit closing duplicate.
    """
    verts = np.asarray(verts, dtype=float)
    if verts.shape[0] < 2 or n < 3:
        return [[float(v[0]), float(v[1])] for v in verts[:n]]

    diffs = np.diff(verts, axis=0)
    seglen = np.linalg.norm(diffs, axis=1)
    cumlen = np.concatenate([[0.0], np.cumsum(seglen)])
    total = float(cumlen[-1])
    if total <= 0:
        # Degenerate (all coincident) — just return the first vertex repeated.
        return [[float(verts[0, 0]), float(verts[0, 1])]] * n

    targets = np.linspace(0.0, total, n, endpoint=False)
    out: List[List[float]] = []
    for tgt in targets:
        idx = int(np.searchsorted(cumlen, tgt, side="right") - 1)
        idx = max(0, min(idx, verts.shape[0] - 2))
        denom = max(cumlen[idx + 1] - cumlen[idx], 1e-9)
        frac = (tgt - cumlen[idx]) / denom
        p = verts[idx] + frac * (verts[idx + 1] - verts[idx])
        out.append([float(p[0]), float(p[1])])
    return out
