# DIC Mesh Region — Integration Contract

## 1. Purpose + where the real math lives
Rasterizes a serializable, ordered list of vector **shapes** into a boolean
`(H, W)` mask that defines the DIC mesh domain / refinement region (True =
inside the region). The mask round-trips through the pipeline file as vector
shapes and is rasterized on Run.

**Real math lives IN-REPO.** It is plain `numpy` + `skimage.draw`
(`disk`, `ellipse`, `polygon`). There is no external DIC / `al_dic`
dependency in this kernel — replaying shapes to a mask *is* the whole
algorithm. Vendored verbatim from `nd2studios/backend/dic/roi.py`
(branch `Version-1.45`).

## 2. Entry point
```python
build_roi_mask(shapes: Optional[Sequence[Dict[str, Any]]], H: int, W: int) -> np.ndarray
# helpers:
_rasterize_region(shape: Dict[str, Any], H: int, W: int) -> np.ndarray   # single shape -> bool (H,W)
has_region(shapes: Optional[Sequence[Dict[str, Any]]]) -> bool           # any drawable (non-clear) action?
```
Primary call is `build_roi_mask`.

## 3. Inputs
| name | Python type | shape | dtype | axis order | units | required? | meaning & constraints |
|------|-------------|-------|-------|-----------|-------|-----------|-----------------------|
| `shapes` | `list[dict]` or `None` | list of action dicts | — | coords are `[row, col] = [y, x]` | pixels | required (may be `None`/`[]`) | ordered list; each dict is one user action (see schema below). `None`/empty → all-False mask. |
| `H` | `int` | scalar | int | rows (y) | pixels | required | mask height = drawn (post-crop, pre-downsample) frame height. |
| `W` | `int` | scalar | int | cols (x) | pixels | required | mask width. |

### Shape dict schema (per action)
`{"type": ..., "op": "add"|"cut", "vertices": [[y,x],...], "center": [y,x], "radius": r}`

| type | required fields | geometry |
|------|-----------------|----------|
| `rect` | `vertices` = exactly 2 `[y,x]` bbox corners | axis-aligned filled rectangle |
| `ellipse` | `vertices` = exactly 2 `[y,x]` bbox corners | filled ellipse inscribed in bbox |
| `circle` | `center` `[y,x]`, `radius` r>0 | filled disk |
| `polygon` | `vertices` = ≥3 `[y,x]` verts | filled polygon |
| `brush` | `vertices` = polyline (≥1) `[y,x]`, `radius` (stroke half-width, default 8) | disks stamped along polyline at ~`r/2` spacing |
| `invert` | (none) | flips whole mask; `op` ignored |
| `clear` | (none) | zeros whole mask; `op` ignored |

## 4. Parameters
| name | type | default | valid range / choices | semantics |
|------|------|---------|-----------------------|-----------|
| `op` (per shape) | str | `"add"` | `"add"`, `"cut"` | Add: `mask \|= region`. Cut: `mask &= ~region`. Ignored for `invert`/`clear`. |
| `radius` (circle) | float | 0 | `>0` to paint | disk radius, pixels. `<=0` → no-op. |
| `radius` (brush) | float | 8 | clamped to `>=1.0` | stroke half-width, pixels; also sets segment sampling step `r/2`. |
| `type` | str | `""` | see schema | unknown/malformed type → empty region (no-op). |

## 5. Output
Single `np.ndarray`.

| field | shape | dtype | axis order | units | meaning |
|-------|-------|-------|-----------|-------|---------|
| return | `(H, W)` | `bool` | `[row, col] = [y, x]` | pixels | True = inside mesh domain / refinement region. |

## 6. Conventions & GOTCHAS
- **Coordinates are `[row, col] = [y, x]`, NOT `[x, y]`.** Both `vertices` and
  `center` use this order. This is the #1 integration risk.
- **Shapes apply STRICTLY IN ORDER.** Add (`|=`), Cut (`&= ~`), invert (`~`
  whole mask), clear (reset to all-False). `invert` and `cut` are **stateful /
  order-dependent** — reordering the list changes the result.
- **Empty / all-False mask is a VALID result.** `None`, `[]`, or a list that
  nets to nothing all yield all-False. **The CALLER decides** whether "no ROI"
  means full-frame vs. empty.
- **Resolution-agnostic.** `H, W` are the *drawn* (post-crop, pre-downsample)
  frame size. The DIC job is responsible for resizing this mask to its
  correlated-frame size.
- `rect` uses rounded integer bbox slicing (clamped to frame); `ellipse`/
  `circle`/`polygon`/`brush` use `skimage.draw` with `shape=(H,W)` clipping.
- `rect`/`ellipse` require exactly 2 vertices; `polygon` requires ≥3; otherwise
  that shape is skipped (empty region).
- Non-dict list items are silently skipped.

## 7. Dependencies
| package | why | when |
|---------|-----|------|
| `numpy` | mask arrays, boolean ops | import-time (top-level) |
| `scikit-image` (`skimage.draw`) | `disk`, `ellipse`, `polygon` rasterization | **lazy** — imported inside `_rasterize_region`, only when a curved/polygon/brush shape is drawn. `rect`/`invert`/`clear` never touch skimage. |

## 8. Failure modes / edge cases
- `shapes=None` or `[]` → all-False `(H,W)` mask (no error).
- Degenerate geometry (rect/ellipse with ≠2 verts, polygon <3, radius ≤0,
  zero-area bbox) → that shape contributes nothing.
- Out-of-frame coords → clipped to `(H,W)` (skimage) or clamped (rect).
- `scikit-image` missing → `ImportError` only when a curved/polygon/brush shape
  is rasterized (lazy import). Pure rect/invert/clear masks work without it.
- Non-dict entries → skipped.

## 9. Minimal runnable example
```python
import sys; sys.path.insert(0, "pure_analysis")
import dic_mesh_region as m

shapes = [
    {"type": "rect",   "op": "add", "vertices": [[2, 2], [8, 8]]},  # [y,x] corners
    {"type": "circle", "op": "cut", "center": [5, 5], "radius": 2}, # punch a hole
]
mask = m.build_roi_mask(shapes, H=10, W=10)
# mask.shape == (10, 10); mask.dtype == bool; mask.sum() == 40
assert mask.shape == (10, 10) and mask.dtype == bool
```

## 10. Pipeline wiring (original)
- **Feeds this kernel:** the DIC Mesh Region drawing dialog produces the
  serializable `shapes` list (stored in the pipeline file); the Run handler
  supplies `H, W` = the drawn (post-crop, pre-downsample) frame size.
- **This kernel feeds:** the boolean mask is published as a DIC side-artifact /
  region directive; the DIC correlation job resizes it to the correlated-frame
  size and uses it to restrict the mesh domain / refinement region.

## 11. Provenance
Vendored verbatim from `nd2studios/backend/dic/roi.py`, branch `Version-1.45`.
Source was already Qt-free / `al_dic`-free / standalone. No members dropped
(no UI/registry/ParamSpec code existed). Only the original module docstring was
replaced by the vendor header; all executable code is byte-identical.
