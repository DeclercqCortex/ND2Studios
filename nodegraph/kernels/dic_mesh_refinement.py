"""DIC Mesh Refinement kernel — ROI/mesh-region rasterization + al_dic refinement policy.

PURPOSE
    Rasterize a serializable list of vector shapes into a boolean (H, W) mesh-domain /
    refinement-brush mask (``build_roi_mask``), and build an optional pyALDIC
    ``RefinementPolicy`` from a refinement spec dict (``_refinement_policy``).

WHERE THE REAL MATH LIVES
    - ``build_roi_mask`` / ``_rasterize_region`` / ``has_region``: the shape-replay
      rasterization math is IN-REPO (pure numpy + scikit-image draw primitives).
    - ``_refinement_policy``: the actual adaptive-mesh refinement policy is built by the
      EXTERNAL, OPTIONAL package ``al_dic`` (pyALDIC) — specifically
      ``al_dic.mesh.refinement.build_refinement_policy``. This module only marshals the
      spec dict into that call and returns None if al_dic is missing or raises.

PROVENANCE (branch: Version-1.45)
    - nd2studios/backend/dic/roi.py  (build_roi_mask, _rasterize_region, has_region — verbatim)
    - nd2studios/backend/dic/engine.py  (_refinement_policy function ONLY — verbatim)

Vendored verbatim; imports nothing from nd2studios; caller owns all prep.

DROPPED UI/REGISTRY-ONLY MEMBERS: none (no ParamSpec/get_params/registry members were on
the vendored path). ADDED IMPORT: ``import importlib`` (stdlib) — required by the copied
_refinement_policy, which lived in engine.py where importlib was imported at module top.
The duplicate ``from __future__ import annotations`` from roi.py was removed (kept once
here at the file top) so the concatenation is valid Python; no code semantics change.
"""
from __future__ import annotations
"""ROI / mesh-region rasterization for the DIC mesh nodes (Qt-free).

Reproduces pyALDIC's ``ROIController`` semantics — Add (boolean OR) / Cut
(boolean AND-NOT) of Rectangle / Polygon / Circle shapes plus a freehand Refine
Brush, with Invert / Clear — but from a *serializable list of vector shapes* (the
same idea as :mod:`nd2studios.backend.analysis.manual_mask`), so the drawn region
round-trips through the pipeline file and is rasterized on Run. This module has no
Qt or ``al_dic`` dependency: it is used both by the drawing dialog (live preview)
and by the page's Run handlers (publishing the region side-artifact).

Shape schema (one dict per user action, applied **in order**)::

    {"type": "rect"|"ellipse"|"circle"|"polygon"|"brush"|"invert"|"clear",
     "op":   "add"|"cut",                       # ignored for invert/clear
     "vertices": [[y, x], ...],                 # rect/ellipse: 2 bbox corners;
                                                #  polygon: >=3; brush: polyline
     "center": [y, x], "radius": r}             # circle / brush stroke width

``build_roi_mask`` replays the list to a boolean ``(H, W)`` mask (True = inside
the mesh domain / refinement region). Applying actions in sequence makes Invert
and Cut deterministic and order-dependent exactly like the interactive tool.
"""

import importlib
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

# Marks a shape whose ``op`` is Cut (subtract from the mask) rather than Add.
OP_ADD = "add"
OP_CUT = "cut"


def _rasterize_region(shape: Dict[str, Any], H: int, W: int) -> np.ndarray:
    """Rasterize a single shape to a boolean ``(H, W)`` region (True = painted)."""
    from skimage.draw import disk as sk_disk
    from skimage.draw import ellipse as sk_ellipse
    from skimage.draw import polygon as sk_polygon

    region = np.zeros((H, W), dtype=bool)
    s_type = str(shape.get("type", ""))
    verts = shape.get("vertices") or []
    v = np.asarray(verts, dtype=float) if verts else np.empty((0, 2))

    if s_type == "rect" and v.shape[0] == 2:
        y0, y1 = sorted([v[0, 0], v[1, 0]])
        x0, x1 = sorted([v[0, 1], v[1, 1]])
        iy0, iy1 = max(0, int(round(y0))), min(H, int(round(y1)) + 1)
        ix0, ix1 = max(0, int(round(x0))), min(W, int(round(x1)) + 1)
        if iy1 > iy0 and ix1 > ix0:
            region[iy0:iy1, ix0:ix1] = True
        return region

    if s_type == "ellipse" and v.shape[0] == 2:
        cy = (v[0, 0] + v[1, 0]) / 2.0
        cx = (v[0, 1] + v[1, 1]) / 2.0
        ry = abs(v[1, 0] - v[0, 0]) / 2.0
        rx = abs(v[1, 1] - v[0, 1]) / 2.0
        if ry > 0 and rx > 0:
            rr, cc = sk_ellipse(cy, cx, ry, rx, shape=(H, W))
            region[rr, cc] = True
        return region

    if s_type == "circle":
        c = shape.get("center")
        r = float(shape.get("radius", 0) or 0)
        if c is not None and r > 0:
            rr, cc = sk_disk((float(c[0]), float(c[1])), r, shape=(H, W))
            region[rr, cc] = True
        return region

    if s_type == "polygon" and v.shape[0] >= 3:
        rr, cc = sk_polygon(v[:, 0], v[:, 1], shape=(H, W))
        region[rr, cc] = True
        return region

    if s_type == "brush" and v.shape[0] >= 1:
        r = max(1.0, float(shape.get("radius", 8) or 8))
        # Stamp a disk at each vertex and along each segment (sampled at ~r/2)
        # so a freehand stroke paints a continuous thick band.
        pts: List[np.ndarray] = []
        for i in range(v.shape[0]):
            pts.append(v[i])
            if i + 1 < v.shape[0]:
                seg = v[i + 1] - v[i]
                dist = float(np.hypot(seg[0], seg[1]))
                n = int(dist / max(1.0, r / 2.0))
                for k in range(1, n):
                    pts.append(v[i] + seg * (k / float(n)))
        for p in pts:
            rr, cc = sk_disk((float(p[0]), float(p[1])), r, shape=(H, W))
            region[rr, cc] = True
        return region

    return region


def build_roi_mask(shapes: Optional[Sequence[Dict[str, Any]]],
                   H: int, W: int) -> np.ndarray:
    """Replay a shape list into a boolean ``(H, W)`` mask (True = inside).

    Actions apply in order: Add ``|=`` region, Cut ``&= ~`` region, Invert flips
    the whole mask, Clear zeros it. An empty / missing list yields an all-False
    mask (the caller decides whether "no ROI" means the full frame)."""
    mask = np.zeros((int(H), int(W)), dtype=bool)
    for shape in (shapes or []):
        if not isinstance(shape, dict):
            continue
        s_type = str(shape.get("type", ""))
        if s_type == "invert":
            mask = ~mask
            continue
        if s_type == "clear":
            mask[:] = False
            continue
        region = _rasterize_region(shape, int(H), int(W))
        if str(shape.get("op", OP_ADD)) == OP_CUT:
            mask &= ~region
        else:
            mask |= region
    return mask


def has_region(shapes: Optional[Sequence[Dict[str, Any]]]) -> bool:
    """True if the shape list contains at least one drawable (non-clear) action."""
    for shape in (shapes or []):
        if isinstance(shape, dict) and str(shape.get("type", "")) not in ("", "clear"):
            return True
    return False


def _refinement_policy(refinement: Optional[Dict[str, Any]], half_win: int):
    """Build an ``al_dic`` ``RefinementPolicy`` from the refine node's spec, or None."""
    if not refinement:
        return None
    try:
        refine = importlib.import_module("al_dic.mesh.refinement")
    except Exception:  # noqa: BLE001 — refinement is best-effort
        return None
    crit = refinement.get("criteria", {}) if isinstance(refinement, dict) else {}
    brush = refinement.get("brush")
    mask = None
    if brush is not None:
        mask = np.asarray(brush).astype(np.float64)
    try:
        return refine.build_refinement_policy(
            refine_inner_boundary=bool(crit.get("mask_boundary", False)),
            refine_outer_boundary=bool(crit.get("roi_edge", False)),
            refinement_mask=(mask if bool(crit.get("brush", mask is not None)) else None),
            min_element_size=int(refinement.get("min_element_size", 8) or 8),
            half_win=int(half_win),
        )
    except Exception:  # noqa: BLE001 — never fail the solve on a bad policy
        return None
