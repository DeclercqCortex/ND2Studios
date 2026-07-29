# Checkpoint — integration contract

## 1. Purpose + where the real math lives

The Checkpoint node is a **freeze/resume cache**. **It carries NO analysis
kernel — there is no math here.** All real computation (segmentation, tracking,
measurement) happens in *other* nodes upstream. This module only serializes an
already-computed, in-memory checkpoint store to disk and rebuilds it, so a later
Run can resume from the frozen result instead of recomputing.

The "real math" therefore lives elsewhere entirely (the upstream analysis nodes,
in-repo). This file is pure serialization built on `numpy` + `json`.

## 2. Entry point

```python
save_checkpoints(dir_path: str,
                 store: Dict[str, Dict[str, Any]],
                 signature: Optional[Dict[str, Any]] = None) -> int

load_checkpoints(dir_path: str
                 ) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]

checkpoints_dir_for(pipeline_path: str) -> str   # helper: derive cache dir path
```

Also exported: the minimal `AnalysisResult` dataclass (load rebuilds instances of
it inside the restored store).

## 3. Inputs (the serialized-store contract)

`save_checkpoints` input `store`:

| name | Python type | shape | dtype | axis order | units | required | meaning |
|------|-------------|-------|-------|-----------|-------|----------|---------|
| store | `{node_id: {"hash": str, "data": snapshot}}` | — | — | — | — | required (may be `{}`) | one entry per checkpoint node |
| node_id | str | — | — | — | — | required | already `node-<hex>`; sanitized to NPZ filename |
| hash | str | — | — | — | — | required | upstream validity hash (opaque here) |
| signature | dict / None | — | — | — | — | optional | source-file signature; stored verbatim in manifest |

`snapshot` (`store[node_id]["data"]`) keys read by save:

| key | type | notes |
|-----|------|-------|
| `results_by_m` | `{int m: AnalysisResult}` | per-multipoint results |
| `ctx_result` | AnalysisResult | must be identical object (`is`) to one value in `results_by_m` to be re-pointed on load |
| `results_crop` | tuple/list or None | crop bbox, stored as list |
| `all_rows`, `results_rows`, `ctx_rows` | list[dict] | measurement rows (JSON via `_json_default`) |
| `track_colormap` | `{int: (r,g,b)}` or None | keys stringified in JSON, re-int on load |
| `track_long_ids` | set[int] | stored sorted, rebuilt as set |
| `track_overlay_rows` | list | verbatim |

`AnalysisResult` fields touched (array shape `(T, H, W)`, `int32` label masks,
axis order T-then-H,W, background = 0):

| field | type | meaning |
|-------|------|---------|
| `label_masks` | `{channel: (T,H,W) array}` | primary label masks (may be lazy readers → materialized via `np.asarray`) |
| `secondary_label_masks` | `{name: (T,H,W) array}` | secondary/inverse masks |
| `overlay_color` | `(r,g,b)` or None | primary overlay color |
| `overlay_alpha` | float (0.45) | primary blend |
| `overlay_outline` | bool | outline vs fill |
| `secondary_overlay_color` | `(r,g,b)` or None | |
| `secondary_overlay_alpha` | float (0.3) | |
| `volumetric_voxel_counts` | `{channel: {(frame,label): count}}` or None | per-(frame,label) voxel counts |

## 4. Parameters

None. There are no tunable compute parameters — behavior is fully determined by
the store contents and the on-disk cache state.

## 5. Output

`save_checkpoints` → `int` (number of checkpoints written).

On-disk layout in `dir_path` (a `<pipeline>.checkpoints/` directory):

| artifact | structure |
|----------|-----------|
| `index.json` | `{kind:"nd2studios.checkpoints", version:1, signature:{...}, checkpoints:{node_id: {...}}}` |
| `<node_id>.npz` | one compressed NPZ per node; keys `a0,a1,…` each a `(T,H,W)` label-mask array. Written only if the node has ≥1 mask. |

Per-node manifest entry: `hash`, `results_crop`, `results_by_m` (list of
per-m dicts: `m`, `channels`=[{channel,key}], `secondary`=[{name,key}],
`overlay_color/alpha/outline`, `secondary_overlay_color/alpha`,
`voxel_counts`=`{ch:[[f,l,c],…]}`), `ctx_result_m`, the row lists,
`track_colormap`, `track_long_ids`, `track_overlay_rows`.

`load_checkpoints` → `(store, signature)` in exactly the shape `save` consumed:
`store[node_id] = {"hash", "data": snapshot}`. On load, `snapshot` additionally
contains `ctx_results_by_m` (a shallow copy of `results_by_m`). Returns
`({}, None)` on an absent / foreign / newer cache.

## 6. Conventions & GOTCHAS

- **NO analysis math** — freeze/resume only. Nothing here validates results;
  validity is a caller concern (the upstream `hash`).
- **`<pipeline>.checkpoints/` dir** = one NPZ per checkpoint node + one
  `index.json`. The NPZ holds label/secondary masks under generated keys
  `a0,a1,…`; the manifest maps `channel`/`name` → key.
- **AnalysisResult identity**: `ctx_result` is re-pointed on load by matching the
  recorded `ctx_result_m` back to `results_by_m[m]`. If `ctx_result` in the input
  was **not** the same object (`is`) as an entry in `results_by_m`, `ctx_m` is
  `None` and load falls back to the first result. Preserve object identity.
- **voxel_counts flattening**: `{(frame,label): count}` is flattened to `[f,l,c]`
  integer triples in JSON and rebuilt as `(int(f),int(l)): int(c)` on load.
- **Masks materialized**: lazy/disk-backed readers are `np.asarray`-materialized
  at save (session scratch won't survive). A mask that fails to materialize is
  silently skipped.
- **Directory is clobbered**: `save_checkpoints` `rmtree`s an existing `dir_path`
  first. Do not point it at a directory holding anything else.
- **`({}, None)` on reject**: absent dir, wrong `kind`, or `version` > 1.
- Axis order is `(T, H, W)`; masks are integer labels with 0 = background.

## 7. Dependencies

| package | why | when |
|---------|-----|------|
| numpy | array (de)serialization, NPZ, `np.asarray` | import-time (top-level) |

Standard lib only otherwise (`json`, `os`, `shutil`, `dataclasses`, `typing`).
No lazy third-party deps. No nd2studios imports.

## 8. Failure modes / edge cases

- Empty `store` (`{}` or None): writes just `index.json`, returns 0.
- Node with no masks: no NPZ written; manifest entry still recorded.
- Mask that won't `np.asarray`: skipped (`_stash` returns None).
- Missing/absent cache dir on load → `({}, None)`.
- Foreign cache (`kind` mismatch) or newer `version` → `({}, None)`.
- `ctx_result` not in `results_by_m` → `ctx_result_m=None`, load uses first entry.
- NPZ key referenced by manifest but absent in NPZ → that mask is dropped.

## 9. Minimal runnable example

```python
import sys, tempfile, numpy as np
sys.path.insert(0, "pure_analysis")
from checkpoint import save_checkpoints, load_checkpoints, AnalysisResult

r = AnalysisResult(
    label_masks={"ch0": np.zeros((2, 4, 4), np.int32)},
    secondary_label_masks={"bg": np.ones((2, 4, 4), np.int32)},
    overlay_color=(255, 0, 0),
    volumetric_voxel_counts={"ch0": {(0, 1): 5, (1, 2): 7}},
)
store = {"node-abc": {"hash": "H", "data": {
    "results_by_m": {0: r}, "ctx_result": r,
    "all_rows": [], "track_long_ids": {1, 2}}}}

d = tempfile.mkdtemp() + "/p.checkpoints"
n = save_checkpoints(d, store, {"sig": 1})          # -> 1
s2, sig = load_checkpoints(d)                        # -> (store, {"sig":1})
rr = s2["node-abc"]["data"]["results_by_m"][0]
assert rr.label_masks["ch0"].shape == (2, 4, 4)      # (T,H,W)
assert rr.volumetric_voxel_counts == {"ch0": {(0, 1): 5, (1, 2): 7}}
assert s2["node-abc"]["data"]["ctx_result"] is rr    # identity re-pointed
```

## 10. Pipeline wiring

In the original pipeline: upstream analysis nodes (segmentation / tracking /
measurement) populate the page's in-memory `_checkpoint_store`
(`{node_id: {hash, data}}`). The Checkpoint node's Run freezes that snapshot;
`save_checkpoints` persists it beside a saved `.nd2s_pipeline.json` (dir from
`checkpoints_dir_for`). On pipeline load, `load_checkpoints` restores the store;
at Run time the checkpoint's upstream `hash` gates whether the cache is used or a
full Run re-freezes. Downstream, the restored `AnalysisResult`s feed the Results
tab / overlays / exporters exactly as a fresh Run would.

## 11. Provenance

Branch `Version-1.45`:
- `nd2studios/pipeline_graph/checkpoint_io.py` — save/load logic (vendored verbatim).
- `nd2studios/core/analysis_registry.py` — `AnalysisResult` dataclass (minimal
  copy: only the fields checkpoint (de)serialization touches; `measurements`,
  `summary`, and the `AnalysisPipeline` registry/base class were dropped).
