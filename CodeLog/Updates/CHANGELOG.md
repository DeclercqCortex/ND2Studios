# Changelog — NodeLab

All notable changes to the standalone NodeLab GUI. The vendored `nd2studios/`
backend is a copy and is **not** modified here (see `pipeline_kit` parity).

## [0.1.0] — NodeLab initial build (ND2Studios V1.90)

### Added
- **Standalone project** `ND2Studios_Blender` with `run.py` launcher and the
  `nodelab` package, built on a **vendored copy** of `nd2studios` (backend
  reached via `nodelab.bootstrap.ensure_backend`, override `ND2STUDIOS_ROOT`).
- **Theme** (`nodelab/theme.py`) — full design-token set + single configurable
  accent (`set_accent`, `accent_glow`, `glow_blur_px`) and `build_stylesheet()`
  which templates `nodelab.qss` (`@ACCENT@ / @ACCENT_12@ / @ACCENT_22@`).
- **Canvas** (`nodelab/canvas/`):
  - `NodeGraphView` — manual pan, wheel zoom-to-cursor (clamp 0.3–2.2×),
    dot-grid `drawBackground`, fit-to-view, delete/backspace, right-click
    quick-add & node context menus, palette drag-drop (`application/x-nodelab-op`).
  - `NodeScene` — bound to a `pipeline_kit` `GraphSlice`; connect mirroring
    `can_connect` + `would_create_cycle`, one-wire-per-input, wire pick-up,
    splice-on-wire, duplicate/bypass/disconnect/delete, selection lift.
  - `NodeItem` — card with category header tint, shape glyph, tag chip,
    output/param/input rows, live progress bar, bypass opacity, selection/running
    glow; channel-source **pills**; rainbow `CHANNEL` header ports.
  - `SocketItem` — `PortType` glyphs (circle/square/diamond/dashed/accent) with
    hover scale + connected glow. `EdgeItem` — cubic-Bézier wires with
    horizontal control handles, channel dashing, and Run flow animation.
- **Panels** (`nodelab/panels/`): registry-driven **Add-Node palette**
  (grouped, searchable, click/drag add), **Properties** (typed widgets from
  `param_specs_for`, honoring `visible_when`, Bypass/Preview), **Results** (stat
  cards + notes), **Viewport** placeholder.
- **Chrome** (`nodelab/chrome/`): menu bar (logo + wordmark + File/Edit/Node/
  View/Help + file readout), toolbar (Open/Save/RUN GRAPH/Stop/zoom/counters),
  console+batch dock (SYS/IO/GRAPH/RUN levels), status bar.
- **Run** (`nodelab/run/`): `RunController` + `RunWorker` (QThread) drive a
  `GraphRunner` topological walk off the UI thread; enhancement recipes execute
  via `apply_recipe`; if-else branch pruning via `evaluate_simple_condition`;
  **`SpecialContext` handler seam** (`register_handler`) for special/results/
  logic nodes (simulated by default).
- **Image I/O** (`nodelab/loading.py`) — **Open TIFF…** loads `.nd2` (via the
  vendored `backend.nd2_loader`) or `.tif/.tiff` (tifffile) into
  `{channel: (T,H,W)}` off the UI thread (`LoadWorker`); the viewport shows the
  frame, the palette's CHANNELS group updates to the file's real channel names,
  the file readout shows `W×H · dtype · frames · ch`, and **Run executes on the
  loaded pixels** (with a processed-frame preview back to the viewport).
- **Save / Load** — `.nd2s_pipeline.json` via `pipeline_kit.io` (schema 6).
- **Verification scripts** copied into `scripts/` (`_pipeline_kit_parity.py`,
  `_pipeline_graph_selftest.py`).

### Notes
- Backend parity preserved: `scripts/_pipeline_kit_parity.py --check` reports no
  drift; all `pipeline_graph` self-tests pass against the vendored copy.
- Run input is a synthetic `(T,H,W)` volume until real TIFF loading is wired.
