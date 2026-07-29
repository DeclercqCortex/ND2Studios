# DIC Mesh Refinement — Integration Contract

## 1. Purpose + where the real math lives
Two related jobs for the 2D DIC (pyALDIC) mesh path:

- **`build_roi_mask` / `_rasterize_region` / `has_region`** — replay a serializable list
  of vector shapes into a boolean `(H, W)` mask that defines the mesh domain / freehand
  refinement region. **Math is IN-REPO** (pure `numpy` + `scikit-image` `draw` primitives:
  `disk`, `ellipse`, `polygon`).
- **`_refinement_policy`** — marshal a refinement spec dict into an `al_dic`
  `RefinementPolicy`. **The real refinement math is EXTERNAL**, in the optional package
  `al_dic` (pyALDIC), function `al_dic.mesh.refinement.build_refinement_policy`. This
  module only translates the spec and returns `None` when `al_dic` is absent or errors.

## 2. Entry points
```python
build_roi_mask(shapes: Optional[Sequence[Dict[str, Any]]], H: int, W: int) -> np.ndarray
has_region(shapes: Optional[Sequence[Dict[str, Any]]]) -> bool
_refinement_policy(refinement: Optional[Dict[str, Any]], half_win: int)  # -> RefinementPolicy | None
_rasterize_region(shape: Dict[str, Any], H: int, W: int) -> np.ndarray   # single-shape helper
```

## 3. Inputs
| name | Python type | shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------|-------|-----------|-------|---------------------|-----------------------|
| `shapes` | `Sequence[dict]` \| None | N shape dicts | — | vertices are `[y, x]` | pixels | required (None/[] → all-False mask) | ordered user actions; see shape schema |
| `H` | int | scalar | — | rows (y) | pixels | required | mask height — the **drawn (cropped) frame** height |
| `W` | int | scalar | — | cols (x) | pixels | required | mask width |
| `refinement` | dict \| None | — | — | — | — | required (None → `None`) | refinement spec (see below) |
| `half_win` | int | scalar | — | — | pixels | required | **= DIC winsize // 2**; comes from the DIC correlation node, NOT this node |

Shape dict schema (one per action, applied in order):
`{"type": "rect"|"ellipse"|"circle"|"polygon"|"brush"|"invert"|"clear", "op": "add"|"cut",
"vertices": [[y,x],...], "center": [y,x], "radius": r}`.
`rect`/`ellipse` need exactly 2 bbox-corner vertices; `polygon` needs ≥3; `brush` is a
polyline (≥1 vertex) stamped with disks of `radius`; `circle` uses `center`+`radius`.

Refinement spec schema:
`{"brush": (H,W) mask | None, "criteria": {"mask_boundary": bool, "roi_edge": bool,
"brush": bool}, "min_element_size": int}`.

## 4. Parameters
| name | type | default | valid range / choices | semantics |
|------|------|---------|------------------------|-----------|
| `op` (per shape) | str | `"add"` | `"add"` / `"cut"` | add = `mask |= region`; cut = `mask &= ~region` |
| `radius` (brush/circle) | float | 8 (brush) | > 0 | disk radius in px; brush clamps to ≥1 |
| `criteria.mask_boundary` | bool | False | — | → `refine_inner_boundary` |
| `criteria.roi_edge` | bool | False | — | → `refine_outer_boundary` |
| `criteria.brush` | bool | `mask is not None` | — | gate: pass `refinement_mask` only if True |
| `min_element_size` | int | 8 | ≥1 | smallest FE element size the policy refines to |
| `half_win` | int | — | ≥1 | correlation half-window; sizes the policy |

## 5. Output
- **`build_roi_mask` / `_rasterize_region`** → `np.ndarray`, shape `(H, W)`, dtype `bool`,
  axis order `[y, x]`, unitless. `True` = inside mesh domain / refinement region.
- **`has_region`** → `bool`.
- **`_refinement_policy`** → an opaque `al_dic` `RefinementPolicy` object (fed straight to
  the al_dic solver) **or `None`** (al_dic missing, no spec, or any error).

## 6. Conventions & GOTCHAS
- **Vertices are `[y, x]` (row, col), NOT `[x, y]`.** `center` is `[y, x]` too.
- **`half_win` is external.** This node ALONE cannot build a complete policy — it MUST be
  given `half_win = winsize // 2` from the DIC correlation params. Do not derive it here.
- **Mask resolution flip.** The brush/refinement mask must be rasterized at the **drawn
  (cropped) frame size**, then **resized to the post-downsample correlated frame size**
  before the policy sees it. That resize is the CALLER's job (not done in this module).
- **Order matters.** Shapes apply sequentially; `invert` flips the whole mask, `clear`
  zeros it — so Cut/Invert are order-dependent, matching the interactive tool.
- **Empty/None `shapes` → all-False mask**; the caller decides whether "no ROI" means the
  full frame.
- **al_dic optional & fail-soft.** `_refinement_policy` swallows all al_dic errors and
  returns `None`; the solve then proceeds with no refinement.
- **`criteria.brush` default** is `mask is not None` — so a supplied brush mask is used
  unless the caller explicitly sets `criteria.brush = False`.
- `_rasterize_region` clips to frame bounds via `shape=(H, W)` on the skimage draws; rects
  are clamped manually.

## 7. Dependencies
| pip package | why | import timing |
|-------------|-----|---------------|
| `numpy` | array math / masks | import-time (module top) |
| `scikit-image` | `skimage.draw.{disk,ellipse,polygon}` rasterization | **lazy** — inside `_rasterize_region` |
| `al-dic` (pyALDIC) | builds the actual `RefinementPolicy` | **lazy & OPTIONAL** — inside `_refinement_policy` |

## 8. Failure modes / edge cases
- `shapes=None`/`[]` → all-False `(H,W)` mask (no error).
- Malformed shape dict (wrong vertex count, missing center/radius) → that shape contributes
  an empty region (silently skipped).
- Non-dict entries in `shapes` are skipped.
- `refinement=None`/falsy → `_refinement_policy` returns `None`.
- `al_dic` not installed, or `build_refinement_policy` raises → `None` (fail-soft).
- `scikit-image` missing → `ImportError` at first `_rasterize_region` call (needed for any
  drawable shape).

## 9. Minimal runnable example
```python
import sys; sys.path.insert(0, "pure_analysis")
import numpy as np
import dic_mesh_refinement as m

shapes = [
    {"type": "rect",   "op": "add", "vertices": [[2, 2], [6, 7]]},
    {"type": "circle", "op": "add", "center": [5, 5], "radius": 3},
]
mask = m.build_roi_mask(shapes, H=10, W=10)   # -> (10, 10) bool, True inside
assert mask.shape == (10, 10) and mask.dtype == bool

policy = m._refinement_policy(
    {"brush": mask.astype(float), "criteria": {"brush": True, "roi_edge": True},
     "min_element_size": 8},
    half_win=4,   # = DIC winsize // 2, supplied by the correlation node
)
# policy is an al_dic RefinementPolicy, or None if al_dic is not installed.
```

## 10. Pipeline wiring
In the original app: the DIC drawing dialog (`ROIController`-style) produces the `shapes`
list, which round-trips through the pipeline file. On Run, `build_roi_mask` rasterizes it
at the drawn/cropped frame size; the caller resizes the mask to the downsampled correlated
frame, packs the refinement spec dict, and calls `_refinement_policy` with `half_win =
winsize//2` from the DIC correlation node. The resulting `RefinementPolicy` feeds
`al_dic.core.pipeline.run_aldic` (the 2D DIC solver) to adaptively refine its quadtree FE
mesh. The ROI mask itself also gates the mesh domain. This node feeds the **DIC correlation
node**; it does not itself perform correlation.

## 11. Provenance
Branch `Version-1.45`:
- `nd2studios/backend/dic/roi.py` — `build_roi_mask`, `_rasterize_region`, `has_region` (verbatim).
- `nd2studios/backend/dic/engine.py` — `_refinement_policy` function only (verbatim).

Edits: removed duplicate `from __future__ import annotations` from roi.py's copy (kept once
at file top); added stdlib `import importlib` (used by the copied `_refinement_policy`).
No compute-path code was altered; no UI/registry members were present to drop.
