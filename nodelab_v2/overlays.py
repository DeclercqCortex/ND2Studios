"""Viewer overlay system — per-domain settings, persistence and the painter.

One place owns *what an overlay looks like* for every attribute domain the Viewer can
draw on top of the image: **Points**, **Labels**, **Tracks**, **Mesh**, and (reserved,
not drawn yet) **Voxels**. The Viewer keeps a single :class:`OverlaySettings` and a
single :class:`OverlayRenderer`; the popup in :mod:`nodelab_v2.overlay_dialog` edits the
settings and the renderer paints them.

Three properties are structural, not incidental:

* **Everything is painted in WIDGET space, in screen pixels.** The renderer is handed a
  ``plane px → widget px`` mapping (:attr:`OverlayFrame.map_pt`) and every size in the
  settings (outline width, glyph spread, vertex radius) is a *screen* size. So an outline
  stays exactly as thick when you zoom into a label as it was zoomed out — which the old
  overlays could not do on the CPU path, because they were baked into the image pixmap
  and magnified with it. Region **fills** are the one thing that scales, because a fill
  *is* the region.
* **The two backends share this code.** Both :class:`~nodelab_v2.glview.GLImageView` and
  the CPU ``_ImageView`` call back with a ``QPainter`` in widget coordinates and expose
  ``plane_to_widget``, so there is exactly one implementation of every overlay look.
* **Field specs drive the UI.** :data:`FIELDS` describes each setting (kind, range,
  choices, what it depends on, and its cross-domain *role*). The dialog builds its
  controls from that, and :func:`spread_settings` copies a tab's look onto the other tabs
  *by role* — so "opacity" travels from Points to Labels even though "arm spread" cannot.

Settings persist as JSON (partial dicts merge, so a file written by an older build still
loads). Three layers, later wins: built-in defaults → the project file shipped in this
package (:data:`PROJECT_FILE`, meant to be committed) → this machine's file
(:data:`USER_FILE`). ``NODELAB_OVERLAYS`` points at one explicit file instead.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field, fields as dc_fields
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap, QPolygonF

SCHEMA = 1

#: Golden-angle hue step — successive integer indices land far apart on the colour wheel,
#: so "each label / point / track a different colour" stays legible for hundreds of items.
GOLDEN_ANGLE = 137.507764


# ── colour helpers (ONE hue→rgb implementation, shared by outlines and fills) ────
def _hues_to_rgb(hues: np.ndarray, sat: float, val: float) -> np.ndarray:
    """Vectorized HSV→RGB for an array of hues (degrees) at a common ``sat``/``val``
    (both 0..1) → ``(N, 3)`` uint8.

    Both the per-item outline colours and the per-label fill LUT go through this, so a
    label's outline and its fill can never disagree about what colour it is.
    """
    hh = (np.asarray(hues, dtype=float) / 60.0) % 6.0
    i = np.floor(hh).astype(int)
    f = hh - i
    v = np.full(hh.shape, float(val))
    p = v * (1.0 - sat)
    q = v * (1.0 - sat * f)
    t = v * (1.0 - sat * (1.0 - f))
    conds = [i == 0, i == 1, i == 2, i == 3, i == 4, i == 5]
    r = np.select(conds, [v, q, p, p, t, v], default=v)
    g = np.select(conds, [t, v, v, q, p, p], default=v)
    b = np.select(conds, [p, p, t, v, v, q], default=v)
    return np.clip(np.stack([r, g, b], axis=-1) * 255.0, 0, 255).round().astype(np.uint8)


def distinct_color(index: int, sat: int = 205, val: int = 255,
                   opacity: int = 100) -> QColor:
    """A deterministic, well-spread colour for item ``index`` (golden-angle hue).

    ``sat``/``val`` are 0..255 (as stored in the settings) and ``opacity`` is a percent.
    Stable across frames — which is what makes a track keep its colour over T.
    """
    rgb = _hues_to_rgb(np.array([(int(index) * GOLDEN_ANGLE) % 360.0]),
                       max(0.0, min(1.0, sat / 255.0)),
                       max(0.0, min(1.0, val / 255.0)))[0]
    col = QColor(int(rgb[0]), int(rgb[1]), int(rgb[2]))
    col.setAlpha(_alpha(opacity))
    return col


def _alpha(opacity_pct: float) -> int:
    return max(0, min(255, int(round(255.0 * float(opacity_pct) / 100.0))))


def qcolor(spec: str, opacity: int = 100) -> QColor:
    """A settings colour string (``#rrggbb``) as a QColor at ``opacity`` percent."""
    col = QColor(str(spec))
    if not col.isValid():
        col = QColor("#ffffff")
    col.setAlpha(_alpha(opacity))
    return col


def _mix(a: QColor, b: QColor, t: float) -> QColor:
    """Blend ``a`` toward ``b`` by ``t``, keeping ``a``'s alpha."""
    out = QColor(round(a.red() * (1 - t) + b.red() * t),
                 round(a.green() * (1 - t) + b.green() * t),
                 round(a.blue() * (1 - t) + b.blue() * t))
    out.setAlpha(a.alpha())
    return out


# ── settings model ──────────────────────────────────────────────────────────────
@dataclass
class PointsOverlay:
    """Point-domain detections. The default look is the requested **golden star**: a
    bright centre pixel exactly on the detection, with arms stepping ``spread`` pixels
    out in each direction and dimming as they go."""
    enabled: bool = True
    shape: str = "star"
    spread: int = 3
    unit_px: float = 3.0
    thickness: float = 1.6
    gradient: bool = True
    center_boost: bool = True
    color_mode: str = "single"
    color: str = "#ffc83c"          # gold
    sat: int = 205
    val: int = 255
    opacity: int = 100
    z_project: bool = False
    off_opacity: int = 30


@dataclass
class LabelsOverlay:
    """Label rasters (an integer Voxel-domain layer). Defaults deliberately favour
    *visibility*: a distinct colour per label, a 2 px outline and a light fill."""
    enabled: bool = True
    style: str = "both"             # outline | fill | both
    width: float = 2.0
    color_mode: str = "per_label"
    color: str = "#e08a3a"          # Domain.LABEL
    sat: int = 205
    val: int = 255
    opacity: int = 100
    fill_opacity: int = 30
    show_ids: bool = False
    id_px: int = 11


@dataclass
class TracksOverlay:
    """Track trajectories. ``per_track`` colouring is golden-angle by ``track_id``, so a
    track keeps exactly its colour across every T frame — the same rule as labels."""
    enabled: bool = True
    width: float = 1.8
    color_mode: str = "per_track"
    color: str = "#c264a0"          # Domain.TRACK
    sat: int = 205
    val: int = 255
    opacity: int = 100
    vertex_px: float = 1.8
    head_px: float = 5.0
    trail: str = "all"              # all | past | window
    window: int = 5
    fade: bool = False
    show_ids: bool = False
    id_px: int = 10


@dataclass
class VoxelsOverlay:
    """Scalar Voxel layers as a colour-mapped wash. **Reserved** — the settings are
    stored and spread like any other tab, but nothing is drawn yet."""
    enabled: bool = False
    style: str = "heatmap"          # heatmap | mask | contour
    colormap: str = "viridis"
    opacity: int = 45
    threshold: float = 0.0
    width: float = 1.4


@dataclass
class MeshOverlay:
    """Mesh-domain boundary surfaces (V2.08), drawn as their **cross-section at the viewed
    Z** — the honest 2-D reading of a 3-D surface, and the one that lines up with the
    image underneath. Projecting all the 3-D edges instead would be an unreadable tangle
    that says nothing about the plane you are looking at."""
    enabled: bool = True
    style: str = "wireframe"        # wireframe | surface | points
    width: float = 1.4
    color_mode: str = "per_object"
    color: str = "#b06ad8"          # Domain.MESH
    sat: int = 205
    val: int = 255
    opacity: int = 100
    fill_opacity: int = 25
    vertex_px: float = 1.6
    near_z: float = 1.0             # 'points' style: how many Z planes count as "near"


@dataclass
class OverlaySettings:
    """The whole overlay configuration — one group per tab in the popup."""
    points: PointsOverlay = field(default_factory=PointsOverlay)
    labels: LabelsOverlay = field(default_factory=LabelsOverlay)
    tracks: TracksOverlay = field(default_factory=TracksOverlay)
    voxels: VoxelsOverlay = field(default_factory=VoxelsOverlay)
    mesh: MeshOverlay = field(default_factory=MeshOverlay)

    def group(self, tab: str):
        return getattr(self, tab)

    # ── (de)serialization — partial dicts MERGE onto the current values, so a file
    #    from an older build (or a hand-edited one) never wipes unknown settings ──
    def to_dict(self) -> Dict[str, Any]:
        return {"schema": SCHEMA, "overlays": {t: asdict(self.group(t)) for t in TABS}}

    def update_from_dict(self, data: Dict[str, Any]) -> List[str]:
        """Merge ``data`` in; returns the ``tab.key`` names actually changed."""
        blob = data.get("overlays", data) if isinstance(data, dict) else {}
        changed: List[str] = []
        for tab in TABS:
            src = blob.get(tab)
            if not isinstance(src, dict):
                continue
            grp = self.group(tab)
            valid = {f.name: f.type for f in dc_fields(grp)}
            for key, raw in src.items():
                if key not in valid:
                    continue                      # unknown key: ignore, never crash
                try:
                    val = _coerce(getattr(grp, key), raw)
                except (TypeError, ValueError):
                    continue
                if val != getattr(grp, key):
                    setattr(grp, key, val)
                    changed.append(f"{tab}.{key}")
        return changed

    def copy(self) -> "OverlaySettings":
        out = OverlaySettings()
        out.update_from_dict(self.to_dict())
        return out


def defaults_for(tab: str) -> Dict[str, Any]:
    """The built-in default values for one tab (what "Reset tab" restores)."""
    return asdict(OverlaySettings().group(tab))


def _coerce(current: Any, raw: Any) -> Any:
    """Cast ``raw`` to the type of the current value (JSON gives us floats for ints)."""
    if isinstance(current, bool):
        return bool(raw)
    if isinstance(current, int):
        return int(round(float(raw)))
    if isinstance(current, float):
        return float(raw)
    return str(raw)


# ── tab metadata ────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class TabInfo:
    key: str
    title: str
    implemented: bool
    blurb: str


TAB_INFO: Tuple[TabInfo, ...] = (
    TabInfo("points", "Points", True,
            "Point-domain detections on the viewed plane (detect.spots, "
            "detect.particles, DVC/DIC field samples)."),
    TabInfo("labels", "Labels", True,
            "Integer label rasters — the Voxel-domain label layer with the most "
            "regions on the viewed plane (analysis.segment, analysis.label)."),
    TabInfo("tracks", "Tracks", True,
            "Track-domain trajectories: member positions joined across time "
            "(track.link, track.objects)."),
    TabInfo("voxels", "Voxels", False,
            "Scalar Voxel layers as a colour-mapped wash (masks, rasterized DVC "
            "fields). Not drawn yet — settings are stored for when it lands."),
    TabInfo("mesh", "Mesh", True,
            "Mesh-domain boundary surfaces, drawn as their cross-section at the viewed "
            "Z (analysis.tessellate)."),
)
TABS: Tuple[str, ...] = tuple(t.key for t in TAB_INFO)
TAB_BY_KEY: Dict[str, TabInfo] = {t.key: t for t in TAB_INFO}

#: Per tab, the ``color_mode`` value that means "give every item its own colour".
PER_ITEM_MODE: Dict[str, Optional[str]] = {
    "points": "per_point", "labels": "per_label", "tracks": "per_track",
    "mesh": "per_object", "voxels": None,
}


# ── field specs (drive the dialog AND the spread-to-other-tabs button) ──────────
@dataclass(frozen=True)
class FieldSpec:
    key: str
    kind: str                    # bool | int | float | choice | color
    label: str
    tip: str = ""
    lo: float = 0.0
    hi: float = 100.0
    step: float = 1.0
    decimals: int = 1
    choices: Tuple[Tuple[str, str], ...] = ()
    role: Optional[str] = None                        # cross-tab spread role
    enable_if: Optional[Tuple[str, Tuple[Any, ...]]] = None
    unit: str = ""


_SAT = FieldSpec("sat", "int", "Palette saturation", role="sat", lo=40, hi=255,
                 tip="Saturation of the auto-generated per-item colours (0–255)")
_VAL = FieldSpec("val", "int", "Palette brightness", role="val", lo=40, hi=255,
                 tip="Brightness of the auto-generated per-item colours (0–255)")
_OPACITY = FieldSpec("opacity", "int", "Opacity", role="opacity", lo=5, hi=100,
                     unit="%", tip="Overall opacity of this overlay")

_POINT_SHAPES = (("star", "Golden star (8 arms, gradient)"),
                 ("cross", "Cross + (4 arms, gradient)"),
                 ("diag", "Diagonal × (4 arms, gradient)"),
                 ("circle", "Filled dot"),
                 ("ring", "Hollow ring"),
                 ("square", "Hollow square"))
_GRADIENT_SHAPES = ("star", "cross", "diag")

FIELDS: Dict[str, Tuple[FieldSpec, ...]] = {
    "points": (
        FieldSpec("shape", "choice", "Marker", choices=_POINT_SHAPES,
                  tip="The glyph stamped on each detection. The gradient shapes put a "
                      "bright pixel on the exact position and step outward."),
        FieldSpec("spread", "int", "Arm spread", lo=1, hi=5, unit=" steps",
                  tip="How many pixel steps the arms reach in each direction (1–5). "
                      "For the ring/dot/square shapes this is the radius."),
        FieldSpec("unit_px", "float", "Step size", lo=1.0, hi=12.0, step=0.5,
                  decimals=1, unit=" px",
                  tip="Size of ONE arm step in SCREEN pixels — so the marker keeps its "
                      "size while you zoom."),
        FieldSpec("thickness", "float", "Line width", role="line_width",
                  lo=0.5, hi=6.0, step=0.1, decimals=1, unit=" px",
                  tip="Stroke width for the ring / square outlines (screen px)"),
        FieldSpec("gradient", "bool", "Brightness gradient",
                  enable_if=("shape", _GRADIENT_SHAPES),
                  tip="Dim each arm step as it moves away from the centre"),
        FieldSpec("center_boost", "bool", "Bright centre pixel",
                  enable_if=("shape", _GRADIENT_SHAPES),
                  tip="Brighten the single pixel sitting on the actual position"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("single", "One colour"),
                           ("per_point", "Every point a different colour"),
                           ("per_layer", "One colour per Point layer")),
                  tip="A single colour, a distinct colour per point, or one per layer"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The marker colour (default: gold)"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("z_project", "bool", "Show points from every Z",
                  tip="Also draw detections that sit on other Z planes, dimmed — "
                      "useful for a sparse 3-D point cloud"),
        FieldSpec("off_opacity", "int", "Off-plane opacity", lo=5, hi=100, unit="%",
                  enable_if=("z_project", (True,)),
                  tip="Opacity of the points that are NOT on the viewed plane"),
    ),
    "labels": (
        FieldSpec("style", "choice", "Style",
                  choices=(("outline", "Outline only"), ("fill", "Fill only"),
                           ("both", "Outline + fill")),
                  tip="Region outlines, a translucent fill, or both"),
        FieldSpec("width", "float", "Outline width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  enable_if=("style", ("outline", "both")),
                  tip="Contour width in SCREEN pixels — constant at any zoom"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("per_label", "Every label a different colour"),
                           ("single", "One colour")),
                  tip="Distinct colour per label id, or one colour for all"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The outline / fill colour when not colouring per label"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("fill_opacity", "int", "Fill opacity", lo=0, hi=100, unit="%",
                  enable_if=("style", ("fill", "both")),
                  tip="Opacity of the region fill (the outline uses Opacity)"),
        FieldSpec("show_ids", "bool", "Label the ids", role="show_ids",
                  tip="Draw each region's integer id at its centroid"),
        FieldSpec("id_px", "int", "Id text size", lo=6, hi=24, unit=" px",
                  enable_if=("show_ids", (True,)),
                  tip="Font size of the id text in screen pixels"),
    ),
    "tracks": (
        FieldSpec("width", "float", "Line width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  tip="Trajectory line width in SCREEN pixels"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("per_track", "Every track a different colour"),
                           ("single", "One colour")),
                  tip="A distinct colour per track id — kept identical across every T "
                      "frame — or one colour for all"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The trajectory colour when not colouring per track"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("vertex_px", "float", "Vertex dot", lo=0.0, hi=6.0, step=0.1,
                  decimals=1, unit=" px",
                  tip="Radius of the per-timepoint dots (0 hides them)"),
        FieldSpec("head_px", "float", "Current-T marker", lo=0.0, hi=14.0, step=0.5,
                  decimals=1, unit=" px",
                  tip="Radius of the enlarged dot on the vertex at the viewed T"),
        FieldSpec("trail", "choice", "Trail",
                  choices=(("all", "Whole trajectory"),
                           ("past", "Up to the viewed T"),
                           ("window", "A window of frames before T")),
                  tip="How much of each trajectory to draw relative to the viewed T"),
        FieldSpec("window", "int", "Trail length", lo=1, hi=50, unit=" frames",
                  enable_if=("trail", ("window",)),
                  tip="How many timepoints back the trail reaches"),
        FieldSpec("fade", "bool", "Fade older segments",
                  tip="Ramp the line's opacity so the newest segment is brightest"),
        FieldSpec("show_ids", "bool", "Label the track ids", role="show_ids",
                  tip="Draw each track's id next to its current position"),
        FieldSpec("id_px", "int", "Id text size", lo=6, hi=24, unit=" px",
                  enable_if=("show_ids", (True,)),
                  tip="Font size of the id text in screen pixels"),
    ),
    "voxels": (
        FieldSpec("style", "choice", "Style",
                  choices=(("heatmap", "Colour-mapped wash"), ("mask", "Flat mask"),
                           ("contour", "Iso-contours")),
                  tip="How a scalar Voxel layer will be drawn"),
        FieldSpec("colormap", "choice", "Colour map",
                  choices=(("viridis", "viridis"), ("magma", "magma"),
                           ("gray", "gray"), ("turbo", "turbo")),
                  tip="The colour ramp for the wash"),
        _OPACITY,
        FieldSpec("threshold", "float", "Threshold", lo=0.0, hi=1.0, step=0.01,
                  decimals=2,
                  tip="Normalized value below which a voxel is left transparent"),
        FieldSpec("width", "float", "Contour width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  enable_if=("style", ("contour",)),
                  tip="Iso-contour line width in screen pixels"),
    ),
    "mesh": (
        FieldSpec("style", "choice", "Style",
                  choices=(("wireframe", "Cross-section outline"),
                           ("surface", "Cross-section filled"),
                           ("points", "Vertices near this Z")),
                  tip="A 3-D surface has no single 2-D picture: the outline is where the "
                      "mesh crosses the viewed Z plane, the filled style shades that "
                      "cross-section's interior, and the vertex style stamps the mesh "
                      "vertices that sit near this plane."),
        FieldSpec("width", "float", "Line width", role="line_width",
                  lo=0.5, hi=8.0, step=0.1, decimals=1, unit=" px",
                  enable_if=("style", ("wireframe", "surface")),
                  tip="Cross-section outline width in SCREEN pixels — constant at any zoom"),
        FieldSpec("color_mode", "choice", "Colour by", role="color_mode",
                  choices=(("per_object", "Every object a different colour"),
                           ("single", "One colour")),
                  tip="A distinct colour per mesh element id, or one colour for all"),
        FieldSpec("color", "color", "Colour", role="color",
                  enable_if=("color_mode", ("single",)),
                  tip="The mesh colour when not colouring per object"),
        _SAT, _VAL, _OPACITY,
        FieldSpec("fill_opacity", "int", "Fill opacity", lo=0, hi=100, unit="%",
                  enable_if=("style", ("surface",)),
                  tip="Opacity of the cross-section fill (the outline uses Opacity)"),
        FieldSpec("vertex_px", "float", "Vertex dot", lo=0.5, hi=8.0, step=0.1,
                  decimals=1, unit=" px", enable_if=("style", ("points",)),
                  tip="Radius of each vertex dot in screen pixels"),
        FieldSpec("near_z", "float", "Z tolerance", lo=0.0, hi=10.0, step=0.5,
                  decimals=1, unit=" planes", enable_if=("style", ("points",)),
                  tip="How far off the viewed Z a vertex may sit and still be drawn"),
    ),
}

#: What "Spread this tab's look to the others" copies — a *role*, not a key, because the
#: same idea wears a different name per domain (``points.thickness`` ↔ ``labels.width``).
SPREAD_ROLES: Tuple[str, ...] = ("opacity", "color", "color_mode", "line_width",
                                 "sat", "val", "show_ids")

SPREAD_ROLE_LABEL: Dict[str, str] = {
    "opacity": "opacity", "color": "single colour", "color_mode": "colour-by mode",
    "line_width": "line width", "sat": "palette saturation",
    "val": "palette brightness", "show_ids": "id labels",
}


def _spec(tab: str, key: str) -> Optional[FieldSpec]:
    for sp in FIELDS[tab]:
        if sp.key == key:
            return sp
    return None


def _role_key(tab: str, role: str) -> Optional[str]:
    for sp in FIELDS[tab]:
        if sp.role == role:
            return sp.key
    return None


def field_enabled(group: Any, spec: FieldSpec) -> bool:
    """Whether ``spec``'s control is live given the group's other values."""
    if spec.enable_if is None:
        return True
    other, allowed = spec.enable_if
    return getattr(group, other, None) in allowed


def spread_settings(settings: OverlaySettings, source: str) -> List[str]:
    """Copy ``source``'s look onto every other tab **where the role applies**.

    Returns one human-readable line per change (so the dialog can say exactly what it
    did, and say nothing when a role had nowhere to go). ``color_mode`` travels as the
    *notion* "one colour" vs "a colour per item" — each tab keeps its own spelling of the
    per-item value (``per_label`` / ``per_track`` / …).
    """
    src = settings.group(source)
    notes: List[str] = []
    for role in SPREAD_ROLES:
        skey = _role_key(source, role)
        if skey is None:
            continue
        sval = getattr(src, skey)
        for tab in TABS:
            if tab == source:
                continue
            tkey = _role_key(tab, role)
            if tkey is None:
                continue
            grp = settings.group(tab)
            if role == "color_mode":
                per_src, per_dst = PER_ITEM_MODE.get(source), PER_ITEM_MODE.get(tab)
                if per_dst is None:
                    continue
                want = per_dst if sval == per_src else "single"
            else:
                want = _coerce(getattr(grp, tkey), sval)
            if getattr(grp, tkey) != want:
                setattr(grp, tkey, want)
                notes.append(f"{TAB_BY_KEY[tab].title}: {SPREAD_ROLE_LABEL[role]} "
                             f"→ {want}")
    return notes


# ── persistence ─────────────────────────────────────────────────────────────────
FILE_SUFFIX = ".nd2overlay.json"
FILE_FILTER = f"Overlay settings (*{FILE_SUFFIX});;JSON (*.json);;All files (*)"

#: this machine's permanent overlay ("make default") — outside the repo
USER_FILE = Path.home() / ".nd2studios" / "overlays.json"
#: the project default, shipped inside the package so committing it shares the look
PROJECT_FILE = Path(__file__).resolve().parent / "overlay_defaults.json"
#: an explicit override (tests / a shared lab config)
ENV_VAR = "NODELAB_OVERLAYS"


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict):
        raise ValueError("overlay settings must be a JSON object")
    return data


def write_json(path: Path, settings: OverlaySettings) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(settings.to_dict(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return path


def default_sources() -> List[Path]:
    """The layers :func:`load_defaults` merges, in increasing precedence."""
    env = os.environ.get(ENV_VAR)
    if env:
        return [Path(env)]
    return [PROJECT_FILE, USER_FILE]


def load_defaults() -> Tuple[OverlaySettings, List[str]]:
    """Built-in defaults with each existing settings layer merged on top.

    Returns ``(settings, notes)``; ``notes`` names the files that were applied and any
    that failed to parse, so the UI can say where the current look came from instead of
    silently falling back.
    """
    settings = OverlaySettings()
    notes: List[str] = []
    for path in default_sources():
        try:
            if not path.is_file():
                continue
            settings.update_from_dict(read_json(path))
            notes.append(f"loaded {path}")
        except Exception as exc:                     # noqa: BLE001 — never block the GUI
            notes.append(f"ignored {path} ({exc})")
    return settings, notes


# ── the painter ─────────────────────────────────────────────────────────────────
@dataclass
class PointMark:
    """One point ready to draw: position in **displayed-plane** pixels, the id used for
    per-point colouring, its layer index, and whether it sits on the viewed Z."""
    y: float
    x: float
    key: int
    layer: int
    on_plane: bool = True


@dataclass
class TrackPath:
    """One trajectory: ``track_id``, its member positions in displayed-plane pixels, the
    index of the vertex at the viewed T (or ``None``), and each vertex's timepoint."""
    track_id: int
    path: List[Tuple[float, float]]
    current: Optional[int]
    times: List[int]


@dataclass
class MeshSection:
    """One mesh element's intersection with the viewed Z plane, ready to draw.

    ``loops`` are polylines of ``(y, x)`` in full-resolution image pixels — closed where
    the cross-section closed, open where it did not (a mesh that is not watertight, or one
    clipped by the plane's edge). ``verts`` are the element's vertices near the plane, for
    the vertex style. ``object_id`` is the per-object colour key.
    """
    object_id: int
    loops: List[List[Tuple[float, float]]] = field(default_factory=list)
    verts: List[Tuple[float, float]] = field(default_factory=list)
    closed: List[bool] = field(default_factory=list)


#: Above this many crossing triangles one element's cross-section is decimated, so a pan
#: over a marching-cubes surface (which can put 10⁴ triangles on one plane) stays live.
MAX_SECTION_TRIANGLES = 40_000


def mesh_section(verts_zyx: np.ndarray, faces: np.ndarray, z: float
                 ) -> Tuple[List[List[Tuple[float, float]]], List[bool]]:
    """Cross-section of a triangle mesh at the plane ``z``, as chained ``(y, x)`` loops.

    Each triangle straddling the plane contributes exactly one segment (the two edges whose
    endpoints fall on opposite sides); the segments are then chained end-to-end into
    polylines. Vertices sitting *exactly* on the plane are nudged to one side, so a
    triangle can never yield zero or three crossings — the degenerate case that would
    otherwise drop or double a segment.

    Returns ``(loops, closed_flags)``.
    """
    v = np.asarray(verts_zyx, dtype=float).reshape(-1, 3)
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0 or len(v) == 0:
        return [], []
    tol = 1e-9
    d = v[:, 0] - float(z)
    d = np.where(np.abs(d) < tol, tol, d)          # never let a vertex sit ON the plane
    dd = d[f]
    pos = dd > 0
    n_pos = pos.sum(axis=1)
    hit = np.flatnonzero((n_pos == 1) | (n_pos == 2))
    if hit.size == 0:
        return [], []
    if hit.size > MAX_SECTION_TRIANGLES:
        hit = hit[::int(np.ceil(hit.size / MAX_SECTION_TRIANGLES))]
    fh, ddh = f[hit], dd[hit]
    P = np.zeros((len(fh), 3, 2), dtype=float)
    SC = np.zeros((len(fh), 3), dtype=bool)
    for k, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        da, db = ddh[:, i], ddh[:, j]
        sc = (da > 0) != (db > 0)
        tt = np.where(sc, da / np.where(da == db, 1.0, da - db), 0.0)
        a, b = v[fh[:, i]][:, 1:], v[fh[:, j]][:, 1:]      # (y, x) only
        P[:, k, :] = a + tt[:, None] * (b - a)
        SC[:, k] = sc
    keep = SC.sum(axis=1) == 2
    segs = [(tuple(pair[0]), tuple(pair[1]))
            for pair in (P[i][SC[i]] for i in np.flatnonzero(keep))]
    return _chain_segments(segs)


def _chain_segments(segs: Sequence[Tuple[Tuple[float, float], Tuple[float, float]]],
                    quant: float = 1e-6
                    ) -> Tuple[List[List[Tuple[float, float]]], List[bool]]:
    """Chain unordered segments into polylines by matching endpoints.

    Endpoints are matched on a quantized key rather than exact equality: the two triangles
    sharing an edge compute the same crossing point from the same two vertices, but not
    necessarily in the same operand order, so the results can differ in the last bit.
    """
    if not segs:
        return [], []

    def key(pt: Tuple[float, float]) -> Tuple[int, int]:
        return (int(round(pt[0] / quant)), int(round(pt[1] / quant)))

    ends: Dict[Tuple[int, int], List[int]] = {}
    for i, (a, b) in enumerate(segs):
        ends.setdefault(key(a), []).append(i)
        ends.setdefault(key(b), []).append(i)
    used = [False] * len(segs)
    loops: List[List[Tuple[float, float]]] = []
    closed: List[bool] = []
    for start in range(len(segs)):
        if used[start]:
            continue
        used[start] = True
        a, b = segs[start]
        path = [a, b]
        # walk forward from b, then (if it did not close) backward from a
        for direction in (0, 1):
            if direction == 1:
                if key(path[0]) == key(path[-1]):
                    break
                path.reverse()
            while True:
                nxt = None
                for i in ends.get(key(path[-1]), ()):
                    if not used[i]:
                        nxt = i
                        break
                if nxt is None:
                    break
                used[nxt] = True
                p, q = segs[nxt]
                path.append(q if key(p) == key(path[-1]) else p)
                if key(path[0]) == key(path[-1]):
                    break
        loops.append(path)
        closed.append(key(path[0]) == key(path[-1]) and len(path) > 3)
    return loops, closed


@dataclass
class OverlayFrame:
    """Everything the renderer needs about *this* frame, in one object.

    ``map_pt`` maps **displayed-plane** pixel coordinates to widget pixels (the backend's
    ``plane_to_widget``); ``sy``/``sx`` scale *structure* coordinates (full-resolution
    image pixels, as the axes report them) into displayed-plane pixels, which is how a
    decimated display plane still lands the geometry in the right place.
    """
    map_pt: Callable[[float, float], QPointF]
    plane_wh: Tuple[int, int]                       # displayed plane (w, h)
    sy: float = 1.0
    sx: float = 1.0
    points: Sequence[PointMark] = ()
    tracks: Sequence[TrackPath] = ()
    mesh: Sequence[MeshSection] = ()
    label_plane: Optional[np.ndarray] = None
    current_t: int = 0


#: Above this many boundary segments the label outline is uniformly decimated so a pan
#: stays interactive. Disclosed in the Labels tab tooltip and warned about once on stderr.
MAX_OUTLINE_SEGMENTS = 250_000

_DIRS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "cross": ((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)),
    "diag": ((-0.70710678, -0.70710678), (-0.70710678, 0.70710678),
             (0.70710678, -0.70710678), (0.70710678, 0.70710678)),
}
_DIRS["star"] = _DIRS["cross"] + _DIRS["diag"]


class OverlayRenderer:
    """Paints the overlays in widget space. Owns two caches — stamped point glyphs and
    the label fill image — both keyed so a pure pan/zoom never rebuilds them."""

    def __init__(self) -> None:
        self._glyphs: Dict[tuple, QPixmap] = {}
        self._fill_key: Optional[tuple] = None
        self._fill_img: Optional[QImage] = None
        self._fill_buf: Optional[np.ndarray] = None      # keeps the QImage's memory alive
        self._warned_truncate = False

    def invalidate(self) -> None:
        """Drop the caches (a settings change or a new node)."""
        self._glyphs.clear()
        self._fill_key = None
        self._fill_img = None
        self._fill_buf = None

    # ── entry point ────────────────────────────────────────────────────────────
    def paint(self, p: QPainter, s: OverlaySettings, frame: OverlayFrame) -> None:
        """Draw every enabled overlay, bottom to top: labels, mesh, points, tracks.

        Mesh sits above the label fill (a cross-section outline is meant to be read
        *against* the region it bounds) and below the point/track markers, which are the
        smallest marks and must never be covered.
        """
        if s.labels.enabled and frame.label_plane is not None:
            self._paint_labels(p, s.labels, frame)
        if s.mesh.enabled and frame.mesh:
            self._paint_mesh(p, s.mesh, frame)
        if s.points.enabled and frame.points:
            self._paint_points(p, s.points, frame)
        if s.tracks.enabled and frame.tracks:
            self._paint_tracks(p, s.tracks, frame)

    # ── mesh ──────────────────────────────────────────────────────────────────
    def mesh_color(self, s: MeshOverlay, object_id: int,
                   opacity: Optional[int] = None) -> QColor:
        op = s.opacity if opacity is None else opacity
        if s.color_mode == "per_object":
            return distinct_color(int(object_id), s.sat, s.val, op)
        return qcolor(s.color, op)

    def _paint_mesh(self, p: QPainter, s: MeshOverlay, frame: OverlayFrame) -> None:
        """Draw each element's Z cross-section (outline / filled) or its near-plane
        vertices. Only a CLOSED cross-section is filled — filling an open polyline would
        invent an edge that the mesh does not have."""
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        for sec in frame.mesh:
            col = self.mesh_color(s, sec.object_id)
            if s.style == "points":
                p.setPen(QPen(col, 1.0))
                p.setBrush(col)
                rad = float(s.vertex_px)
                if rad > 0:
                    for y, x in sec.verts:
                        p.drawEllipse(frame.map_pt(x * frame.sx, y * frame.sy), rad, rad)
                continue
            polys = [QPolygonF([frame.map_pt(x * frame.sx, y * frame.sy)
                                for y, x in loop]) for loop in sec.loops]
            if s.style == "surface" and s.fill_opacity > 0:
                fill = self.mesh_color(s, sec.object_id, s.fill_opacity)
                p.setPen(Qt.NoPen)
                p.setBrush(fill)
                for poly, is_closed in zip(polys, sec.closed):
                    if is_closed:
                        p.drawPolygon(poly)
            if s.width > 0:
                pen = QPen(col, float(s.width))
                pen.setCapStyle(Qt.RoundCap)
                pen.setJoinStyle(Qt.RoundJoin)
                p.setPen(pen)
                p.setBrush(Qt.NoBrush)
                for poly in polys:
                    p.drawPolyline(poly)
        p.restore()

    # ── labels ────────────────────────────────────────────────────────────────
    def _label_color(self, s: LabelsOverlay, value: int, opacity: int) -> QColor:
        if s.color_mode == "per_label":
            return distinct_color(int(value), s.sat, s.val, opacity)
        return qcolor(s.color, opacity)

    def _paint_labels(self, p: QPainter, s: LabelsOverlay, frame: OverlayFrame) -> None:
        lab = frame.label_plane
        if lab is None or lab.size == 0:
            return
        W, H = frame.plane_wh
        lh, lw = lab.shape[:2]
        lsy, lsx = H / max(1, lh), W / max(1, lw)

        if s.style in ("fill", "both") and s.fill_opacity > 0:
            img = self._fill_image(s, lab)
            if img is not None:
                tl = frame.map_pt(0.0, 0.0)
                br = frame.map_pt(float(W), float(H))
                p.save()
                # nearest-neighbour: a zoomed-in label mask must stay pixel-crisp, and
                # interpolating distinct label colours invents regions that aren't there.
                p.setRenderHint(QPainter.SmoothPixmapTransform, False)
                p.drawImage(QRectF(tl, br), img)
                p.restore()

        if s.style in ("outline", "both") and s.width > 0:
            self._paint_label_outline(p, s, frame, lab, lsy, lsx)

        if s.show_ids:
            self._paint_label_ids(p, s, frame, lab, lsy, lsx)

    def _fill_image(self, s: LabelsOverlay, lab: np.ndarray) -> Optional[QImage]:
        """A cached RGBA image of the label plane — one colour per region."""
        key = (id(lab), lab.shape, s.color_mode, s.color, s.sat, s.val, s.fill_opacity)
        if key == self._fill_key and self._fill_img is not None:
            return self._fill_img
        top = int(lab.max()) if lab.size else 0
        if top <= 0 or top > 1 << 20:            # nothing to fill / absurd id space
            return None
        alpha = _alpha(s.fill_opacity)
        lut = np.zeros((top + 1, 4), dtype=np.uint8)
        if s.color_mode == "per_label":
            idx = np.arange(1, top + 1, dtype=float)
            lut[1:, :3] = _hues_to_rgb((idx * GOLDEN_ANGLE) % 360.0,
                                       max(0.0, min(1.0, s.sat / 255.0)),
                                       max(0.0, min(1.0, s.val / 255.0)))
        else:
            col = qcolor(s.color)
            lut[1:, :3] = np.array([col.red(), col.green(), col.blue()], dtype=np.uint8)
        lut[1:, 3] = alpha
        rgba = np.ascontiguousarray(np.take(lut, np.clip(lab, 0, top), axis=0))
        h, w = rgba.shape[:2]
        img = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888)
        self._fill_key, self._fill_img, self._fill_buf = key, img, rgba
        return img

    def _paint_label_outline(self, p: QPainter, s: LabelsOverlay, frame: OverlayFrame,
                             lab: np.ndarray, lsy: float, lsx: float) -> None:
        """Draw region boundaries as **segments** grouped by colour.

        Segments, not pixels: the old overlay stamped one dot per boundary pixel, which
        zoomed in turns into a dotted line with gaps as wide as the magnification. A
        vertical boundary between ``lab[i, j]`` and ``lab[i, j+1]`` is the unit segment
        from ``(j+1, i)`` to ``(j+1, i+1)`` in label-pixel space, so the contour stays a
        continuous line at any zoom and the pen width stays in screen pixels.
        """
        v_i, v_j = np.nonzero(lab[:, :-1] != lab[:, 1:])
        h_i, h_j = np.nonzero(lab[:-1, :] != lab[1:, :])
        if v_i.size == 0 and h_i.size == 0:
            return
        # colour key = the larger of the two sides, so a label/background edge takes the
        # label's colour and a label/label edge is drawn once, deterministically.
        v_key = np.maximum(lab[v_i, v_j], lab[v_i, v_j + 1])
        h_key = np.maximum(lab[h_i, h_j], lab[h_i + 1, h_j])
        # (x0, y0, x1, y1) in label-pixel space
        v_seg = np.stack([v_j + 1.0, v_i * 1.0, v_j + 1.0, v_i + 1.0], axis=1)
        h_seg = np.stack([h_j * 1.0, h_i + 1.0, h_j + 1.0, h_i + 1.0], axis=1)
        segs = np.concatenate([v_seg, h_seg], axis=0)
        keys = np.concatenate([v_key, h_key], axis=0).astype(np.int64)
        keep = keys > 0
        segs, keys = segs[keep], keys[keep]
        if segs.shape[0] == 0:
            return
        if segs.shape[0] > MAX_OUTLINE_SEGMENTS:
            stride = int(np.ceil(segs.shape[0] / MAX_OUTLINE_SEGMENTS))
            segs, keys = segs[::stride], keys[::stride]
            if not self._warned_truncate:
                self._warned_truncate = True
                print(f"[overlays] label outline decimated 1/{stride} "
                      f"(>{MAX_OUTLINE_SEGMENTS} boundary segments)",
                      file=sys.stderr, flush=True)
        mp = frame.map_pt
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        p.setBrush(Qt.NoBrush)
        single = s.color_mode != "per_label"
        order = np.array([0]) if single else np.unique(keys)
        for value in order.tolist():
            sel = slice(None) if single else (keys == value)
            group = segs[sel]
            if group.shape[0] == 0:
                continue
            pen = QPen(self._label_color(s, value, s.opacity), float(s.width))
            pen.setCapStyle(Qt.FlatCap)
            p.setPen(pen)
            p.drawLines([QLineF(mp(x0 * lsx, y0 * lsy), mp(x1 * lsx, y1 * lsy))
                         for x0, y0, x1, y1 in group.tolist()])
        p.restore()

    def _paint_label_ids(self, p: QPainter, s: LabelsOverlay, frame: OverlayFrame,
                         lab: np.ndarray, lsy: float, lsx: float) -> None:
        """Stamp each region's id at its centroid (capped — hundreds of numbers on one
        plane is noise, not information)."""
        top = int(lab.max()) if lab.size else 0
        if top <= 0 or top > 400:
            return
        flat = lab.ravel()
        counts = np.bincount(flat, minlength=top + 1).astype(float)
        yy, xx = np.mgrid[0:lab.shape[0], 0:lab.shape[1]]
        cy = np.bincount(flat, weights=yy.ravel(), minlength=top + 1)
        cx = np.bincount(flat, weights=xx.ravel(), minlength=top + 1)
        p.save()
        font = QFont(p.font())
        font.setPixelSize(int(s.id_px))
        font.setBold(True)
        p.setFont(font)
        for value in range(1, top + 1):
            n = counts[value]
            if n <= 0:
                continue
            pt = frame.map_pt(cx[value] / n * lsx, cy[value] / n * lsy)
            p.setPen(QPen(QColor(0, 0, 0, _alpha(s.opacity) // 2), 1.0))
            p.drawText(pt + QPointF(1.0, 1.0), str(value))
            p.setPen(QPen(self._label_color(s, value, s.opacity), 1.0))
            p.drawText(pt, str(value))
        p.restore()

    # ── points ────────────────────────────────────────────────────────────────
    def _point_color(self, s: PointsOverlay, mark: PointMark) -> QColor:
        opacity = s.opacity if mark.on_plane else min(s.opacity, s.off_opacity)
        if s.color_mode == "per_point":
            return distinct_color(mark.key, s.sat, s.val, opacity)
        if s.color_mode == "per_layer":
            return distinct_color(mark.layer, s.sat, s.val, opacity)
        return qcolor(s.color, opacity)

    def _paint_points(self, p: QPainter, s: PointsOverlay,
                      frame: OverlayFrame) -> None:
        dpr = float(p.device().devicePixelRatio() or 1.0)
        for mark in frame.points:
            col = self._point_color(s, mark)
            pm, size = self.glyph(s, col, dpr)
            centre = frame.map_pt(mark.x * frame.sx, mark.y * frame.sy)
            p.drawPixmap(QPointF(centre.x() - size / 2.0, centre.y() - size / 2.0), pm)

    def glyph(self, s: PointsOverlay, col: QColor, dpr: float) -> Tuple[QPixmap, float]:
        """A cached marker pixmap for ``col`` plus its logical size.

        Pre-rendering the glyph once and stamping it is what keeps a 10 000-point overlay
        cheap, and it is also what makes the size zoom-invariant: the pixmap is built in
        screen pixels and never scaled.
        """
        radius = max(1.0, float(s.spread) * float(s.unit_px))
        pad = max(2.0, float(s.thickness) + 1.0)
        size = 2.0 * radius + 2.0 * pad
        key = (s.shape, s.spread, round(s.unit_px, 2), round(s.thickness, 2),
               s.gradient, s.center_boost, col.rgba(), round(dpr, 3))
        cached = self._glyphs.get(key)
        if cached is not None:
            return cached, size
        if len(self._glyphs) > 512:                 # a per-point palette is bounded by
            self._glyphs.clear()                    # 360 hues; this is the safety net
        px = max(1, int(round(size * dpr)))
        pm = QPixmap(px, px)
        pm.setDevicePixelRatio(dpr)
        pm.fill(Qt.transparent)
        gp = QPainter(pm)
        c = size / 2.0
        rays = _DIRS.get(s.shape)
        if rays is not None:
            unit = float(s.unit_px)
            # crisp square "pixels": antialiasing a 3 px block just blurs its edges
            gp.setRenderHint(QPainter.Antialiasing, False)
            gp.setPen(Qt.NoPen)
            for k in range(int(s.spread), 0, -1):
                frac = (1.0 - k / (s.spread + 1.0)) if s.gradient else 1.0
                step = QColor(col)
                step.setAlpha(max(0, min(255, int(round(col.alpha() * frac)))))
                gp.setBrush(step)
                for dy, dx in rays:
                    gp.drawRect(QRectF(c + dx * k * unit - unit / 2.0,
                                       c + dy * k * unit - unit / 2.0, unit, unit))
            centre = _mix(col, QColor(255, 255, 255), 0.5) if s.center_boost else col
            gp.setBrush(centre)
            gp.drawRect(QRectF(c - unit / 2.0, c - unit / 2.0, unit, unit))
        else:
            gp.setRenderHint(QPainter.Antialiasing, True)
            if s.shape == "circle":
                gp.setPen(Qt.NoPen)
                gp.setBrush(col)
                gp.drawEllipse(QPointF(c, c), radius * 0.5, radius * 0.5)
            elif s.shape == "square":
                gp.setPen(QPen(col, float(s.thickness)))
                gp.setBrush(Qt.NoBrush)
                gp.drawRect(QRectF(c - radius, c - radius, 2 * radius, 2 * radius))
            else:                                    # ring
                gp.setPen(QPen(col, float(s.thickness)))
                gp.setBrush(Qt.NoBrush)
                gp.drawEllipse(QPointF(c, c), radius, radius)
        gp.end()
        self._glyphs[key] = pm
        return pm, size

    # ── tracks ────────────────────────────────────────────────────────────────
    def track_color(self, s: TracksOverlay, track_id: int,
                    opacity: Optional[int] = None) -> QColor:
        op = s.opacity if opacity is None else opacity
        if s.color_mode == "per_track":
            return distinct_color(int(track_id), s.sat, s.val, op)
        return qcolor(s.color, op)

    def _paint_tracks(self, p: QPainter, s: TracksOverlay,
                      frame: OverlayFrame) -> None:
        p.save()
        p.setRenderHint(QPainter.Antialiasing, True)
        font = QFont(p.font())
        font.setPixelSize(int(s.id_px))
        font.setBold(True)
        for tr in frame.tracks:
            keep = self._trail_mask(s, tr, frame.current_t)
            verts = [frame.map_pt(x * frame.sx, y * frame.sy)
                     for i, (y, x) in enumerate(tr.path) if keep[i]]
            if not verts:
                continue
            col = self.track_color(s, tr.track_id)
            if len(verts) >= 2:
                if s.fade:
                    n = len(verts) - 1
                    for i in range(n):
                        frac = 0.25 + 0.75 * ((i + 1) / n)
                        seg = QColor(col)
                        seg.setAlpha(int(round(col.alpha() * frac)))
                        p.setPen(QPen(seg, float(s.width)))
                        p.drawLine(verts[i], verts[i + 1])
                else:
                    p.setPen(QPen(col, float(s.width)))
                    p.setBrush(Qt.NoBrush)
                    p.drawPolyline(QPolygonF(verts))
            # the current-T vertex keeps its identity in the ORIGINAL path, so recover
            # its index within the drawn subset
            cur_drawn = None
            if tr.current is not None and keep[tr.current]:
                cur_drawn = int(np.count_nonzero(keep[:tr.current]))
            p.setPen(QPen(col, 1.0))
            p.setBrush(col)
            for k, pt in enumerate(verts):
                rad = float(s.head_px) if k == cur_drawn else float(s.vertex_px)
                if rad > 0:
                    p.drawEllipse(pt, rad, rad)
            if s.show_ids:
                anchor = verts[cur_drawn if cur_drawn is not None else -1]
                p.setFont(font)
                off = QPointF(float(s.head_px) + 3.0, -float(s.head_px) - 2.0)
                p.setPen(QPen(QColor(0, 0, 0, col.alpha() // 2), 1.0))
                p.drawText(anchor + off + QPointF(1.0, 1.0), str(tr.track_id))
                p.setPen(QPen(col, 1.0))
                p.drawText(anchor + off, str(tr.track_id))
        p.restore()

    @staticmethod
    def _trail_mask(s: TracksOverlay, tr: TrackPath, current_t: int) -> np.ndarray:
        """Which vertices of ``tr`` the ``trail`` mode draws at the viewed T."""
        times = np.asarray(tr.times if tr.times else [0] * len(tr.path))
        if times.size != len(tr.path):
            return np.ones(len(tr.path), dtype=bool)
        if s.trail == "past":
            return times <= current_t
        if s.trail == "window":
            return (times <= current_t) & (times >= current_t - int(s.window))
        return np.ones(len(tr.path), dtype=bool)


# ── live preview (used by the dialog; proves the renderer, not a second look) ────
def render_preview(p: QPainter, rect: QRectF, s: OverlaySettings, tab: str,
                   renderer: Optional[OverlayRenderer] = None) -> None:
    """Draw a small sample of ``tab``'s overlay inside ``rect`` using the REAL renderer,
    over a synthetic checker so opacity is readable. Reserved tabs get a plain note."""
    p.save()
    p.setClipRect(rect)
    p.fillRect(rect, QColor(18, 21, 26))
    cell = 12.0                              # a checker, so opacity is readable
    shade = QColor(31, 36, 44)
    ny = int(rect.height() // cell) + 1
    nx = int(rect.width() // cell) + 1
    for iy in range(ny):
        for ix in range(nx):
            if (ix + iy) % 2 == 0:
                p.fillRect(QRectF(rect.left() + ix * cell, rect.top() + iy * cell,
                                  cell, cell), shade)
    info = TAB_BY_KEY[tab]
    if not info.implemented:
        p.setPen(QPen(QColor(150, 160, 175), 1.0))
        p.drawText(rect, int(Qt.AlignCenter | Qt.TextWordWrap),
                   f"{info.title} overlays are not drawn yet —\n"
                   "these settings are stored for when they land.")
        p.restore()
        return

    ren = renderer or OverlayRenderer()
    w, h = 40.0, 24.0                       # a tiny synthetic plane
    sx = rect.width() / w
    sy = rect.height() / h

    def map_pt(px: float, py: float) -> QPointF:
        return QPointF(rect.left() + px * sx, rect.top() + py * sy)

    frame = OverlayFrame(map_pt=map_pt, plane_wh=(int(w), int(h)), current_t=2)
    sub = OverlaySettings()
    sub.update_from_dict(s.to_dict())
    for other in TABS:                      # preview exactly one overlay at a time
        sub.group(other).enabled = (other == tab)
    if tab == "points":
        frame.points = [PointMark(6.0, 8.0, 1, 0), PointMark(12.0, 20.0, 2, 0),
                        PointMark(17.0, 32.0, 3, 1), PointMark(9.0, 30.0, 4, 1)]
    elif tab == "labels":
        lab = np.zeros((int(h), int(w)), dtype=np.int32)
        lab[4:12, 5:16] = 1
        lab[6:18, 18:27] = 2
        lab[3:9, 29:38] = 3
        frame.label_plane = lab
    elif tab == "tracks":
        frame.tracks = [
            TrackPath(1, [(6.0, 4.0), (8.0, 11.0), (10.0, 18.0), (13.0, 26.0),
                          (15.0, 34.0)], 2, [0, 1, 2, 3, 4]),
            TrackPath(2, [(18.0, 6.0), (16.0, 13.0), (15.0, 21.0), (12.0, 29.0)],
                      2, [0, 1, 2, 3]),
        ]
    elif tab == "mesh":
        # two closed cross-sections — a hexagon and an L, so the fill / outline / vertex
        # styles all show something honest (the L proves a concave section renders)
        hexa = [(6.0 + 4.0 * np.sin(a), 11.0 + 5.0 * np.cos(a))
                for a in np.linspace(0, 2 * np.pi, 7)]
        ell = [(4.0, 22.0), (4.0, 34.0), (9.0, 34.0), (9.0, 28.0),
               (18.0, 28.0), (18.0, 22.0), (4.0, 22.0)]
        frame.mesh = [MeshSection(1, [hexa], hexa[:-1], [True]),
                      MeshSection(2, [ell], ell[:-1], [True])]
    ren.paint(p, sub, frame)
    p.restore()


__all__ = [
    "SCHEMA", "GOLDEN_ANGLE", "distinct_color", "qcolor",
    "PointsOverlay", "LabelsOverlay", "TracksOverlay", "VoxelsOverlay", "MeshOverlay",
    "OverlaySettings", "TabInfo", "TAB_INFO", "TABS", "TAB_BY_KEY", "PER_ITEM_MODE",
    "FieldSpec", "FIELDS", "SPREAD_ROLES", "SPREAD_ROLE_LABEL", "field_enabled",
    "spread_settings", "FILE_SUFFIX", "FILE_FILTER", "USER_FILE", "PROJECT_FILE",
    "ENV_VAR", "read_json", "write_json", "default_sources", "load_defaults",
    "defaults_for",
    "PointMark", "TrackPath", "MeshSection", "OverlayFrame", "OverlayRenderer",
    "MAX_OUTLINE_SEGMENTS", "MAX_SECTION_TRIANGLES", "mesh_section", "render_preview",
]
