# NodeLab — Architecture

> ## ⚠️ SUPERSEDED 2026-07-29 — this describes the REMOVED first-generation app.
> `nodelab/` and the vendored `nd2studios`/`pipeline_kit` backend were deleted on
> 2026-07-29 (record: [../ClaudesPlan/V2.05_phase7_capability_matrix.md](../ClaudesPlan/V2.05_phase7_capability_matrix.md)
> §6). Every path, module and invariant below is gone. The current architecture is
> **[ENGINEERING_NOTES.md](ENGINEERING_NOTES.md)** (the `nodegraph` engine +
> `nodelab_v2` GUI); the user-facing guide is [../../MANUAL.md](../../MANUAL.md).
> This file is kept only as history.

NodeLab is a thin, well-layered PySide6 GUI over the vendored, Qt-free
`nd2studios.pipeline_kit` backend. All Qt lives in `nodelab/`; the backend is a
copy consumed only through its public API.

## Layers

```
        ┌────────────────────────── nodelab (Qt) ──────────────────────────┐
run.py→ app.build_app → main_window.NodeLabWindow                          │
        │   ├── chrome/  menu · toolbar · console dock · status bar         │
        │   ├── panels/  palette · properties · results · viewport         │
        │   ├── canvas/  view → scene → node/socket/edge items             │
        │   └── run/     controller → worker(QThread) → handler seam        │
        └───────────────────────────────┬──────────────────────────────────┘
                                         │ public API only
        ┌────────────────────────── nd2studios (vendored, Qt-free) ─────────┐
        │  pipeline_kit: model · registry · catalog · executor · io · nodes │
        │  backend / compute / core / utils (engines, lazy-imported)        │
        └───────────────────────────────────────────────────────────────────┘
```

## Data flow

- **State** — a `pipeline_kit.PipelineDoc`; NodeLab edits its **merged** slice
  (`doc.merged`), where processing/analysis/results/logic/special/channel nodes
  all live. `NodeScene` owns `NodeItem`/`EdgeItem` mirroring that slice.
- **Every edit mutates the model, then items resync.** `NodeScene` delegates the
  rules to the backend: `can_connect`, `would_create_cycle`, `structural_edges`,
  `build_node`, `clone_node`. No graph state is invented in the GUI.
- **Palette / properties** are registry-driven: `specs.palette_groups()` reads
  the live catalog; Properties renders `param_specs_for(op_key)`.
- **Run** — `RunController` starts a `RunWorker(QThread)` that walks
  `GraphRunner`: enhancement nodes → `recipe_for_node` + `apply_recipe`
  (real compute); if-else → `evaluate_simple_condition` + `GraphRunner.complete`
  branch pruning; all else → the `handlers` seam. Per-node + summary signals
  drive the canvas (running/progress/done glow, edge flow), console, results,
  toolbar, and status bar. The UI thread never blocks.
- **Persistence** — `save_pipeline` / `load_pipeline` (`.nd2s_pipeline.json`,
  schema 6, with migrations) — unchanged backend code.

## Key modules

| Module | Responsibility |
|---|---|
| `nodelab/bootstrap.py` | Find the vendored backend, force-import nodes to populate `NODES`. |
| `nodelab/theme.py` | Design tokens, accent management, category tints, QSS build. |
| `nodelab/specs.py` | Palette grouping + `op_key → NodeSpec` resolution. |
| `nodelab/glyphs.py` | Shared shape-glyph drawing (node headers + palette tiles). |
| `canvas/scene.py` | Model binding, connect, splice, selection, counts. |
| `canvas/view.py` | Pan/zoom/grid/fit + mouse interaction + menus + drops. |
| `canvas/node_item.py` | Node card painting + socket layout + state. |
| `run/worker.py` | Off-thread GraphRunner walk + per-node execution. |
| `run/handlers.py` | GUI-agnostic special-node execution seam. |
| `main_window.py` | Integration hub — wires all signals. |

## Boundaries & invariants

- **No edits to `nd2studios/`** → `pipeline_kit` parity stays byte-for-byte
  (`scripts/_pipeline_kit_parity.py --check`).
- **`op_key`s are frozen** (saved-file contract) — never renamed.
- **Backend purity** — nothing in `nd2studios/pipeline_kit` imports Qt; NodeLab
  imports the backend, never the reverse.
- **Threading** — all load/run work is in `QThread` workers; signals cross back
  to the UI thread.

## Extension points

- **Real special-node handlers** — `@register_handler(op_key)` in
  `nodelab/run/handlers.py` (port from the old `pages/pipelines_page.py`).
- **Real image I/O** — feed `RunController(channel_provider=…)` and
  `ViewportPanel.set_frame(...)` from a loaded ND2/TIFF.
- **Loops / condition-block editor / per-edge levers** — the backend already
  supports loop edges, `ConditionBlock`, and edge scope/view-only; add the UI.
