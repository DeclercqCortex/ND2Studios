# NodeLab — User Manual

**ND2Studios_Blender** is a Blender-geometry-nodes-style node editor for microscopy image
analysis. You build an **acquire → enhance → segment → measure → track** pipeline by wiring
nodes on a canvas, tune each node in the inspector, and *pull* a node to see its result in a
live multi-channel viewer or as a table.

The engine is [`nodegraph/`](nodegraph/) (Qt-free); the editor is
[`nodelab_v2/`](nodelab_v2/). For how it works internally, see
[CodeLog/Architecture/ENGINEERING_NOTES.md](CodeLog/Architecture/ENGINEERING_NOTES.md).

> **State as of 2026-07-29:** catalog **54 node types** (V2.12 folded Watershed +
> StarDist into the one **Segmentation** node and added CellSAM); headless gate
> `python -m nodegraph.selftest` → **56 `[ok]` lines green**; driven GUI gate
> `scripts/_nodelab_v2_phase5_probe.py` → **ALL PASS**. There is exactly one editor — the
> first-generation `nodelab`/`pipeline_kit` app was removed on 2026-07-29 and `--legacy`
> no longer exists.

---

## Contents

1. [Install & launch](#1-install--launch)
2. [The window](#2-the-window)
3. [Your first graph in five minutes](#3-your-first-graph-in-five-minutes)
4. [Loading data](#4-loading-data)
5. [Building graphs](#5-building-graphs)
6. [Running: pull, progress, errors](#6-running-pull-progress-errors)
7. [The Viewer](#7-the-viewer)
8. [The Inspector — parameters, units, auto/pinned](#8-the-inspector--parameters-units-autopinned)
9. [The 2D/3D lever](#9-the-2d3d-lever)
10. [Domains, layers and the layer picker](#10-domains-layers-and-the-layer-picker)
11. [Spreadsheet & export](#11-spreadsheet--export)
12. [Organising a big graph: frames, reroutes, groups, zones](#12-organising-a-big-graph-frames-reroutes-groups-zones)
13. [Saving & loading](#13-saving--loading)
14. [Keyboard & mouse reference](#14-keyboard--mouse-reference)
15. [Node reference (all 54)](#15-node-reference-all-54)
16. [Worked workflows](#16-worked-workflows)
17. [Headless / scripted use](#17-headless--scripted-use)
18. [Troubleshooting](#18-troubleshooting)
19. [Verifying a build](#19-verifying-a-build)

---

## 1. Install & launch

```bash
pip install -r requirements.txt
python run.py
```

Python 3.13 is the tested interpreter. Required: PySide6, numpy, scipy, scikit-image,
PyWavelets, opencv-python, nd2 + dask, tifffile, blosc2, pyarrow.

**Optional, feature-gated at import.** The app launches and the *whole palette registers*
without these; the node raises a friendly install hint only when you pull it:

| Extra | Unlocks |
|---|---|
| `scikit-learn` | `analysis.cluster_points` (GMM / KMeans + BIC) |
| `numba`, `pandas` | `track.objects` (the 5-method tracker) |
| `al-dic` | `analysis.dic_correlate` (2D DIC, pyALDIC) |
| `stardist`, `tensorflow`, `csbdeep` | `analysis.segment` → method **stardist** |
| `cellSAM` (+ `torch`) | `analysis.segment` → method **cellsam**; the weights also need
  `DEEPCELL_ACCESS_TOKEN` from <https://users.deepcell.org> (non-commercial academic
  licence), or point `model_path` at a local `.pt` |
| `zarr` | only `scripts/_bench_provider_granularity.py` |

### Setting up CellSAM (once per machine)

The weights are ~1.7 GB and licensed for **non-commercial academic use** through an
authenticated endpoint, so they cannot ship with the repo — every user fetches their own:

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
python scripts/setup_cellsam.py     # downloads + verifies the weights
python scripts/_cellsam_smoke.py    # proves the whole path: model -> node -> labels
```

`setup_cellsam.py` asks for a token from <https://users.deepcell.org> the first time and
tells you exactly how to set it. **After that, loading is fully offline** — the token is
never consulted again, including from the GUI, because `get_model()` returns early once
`~/.deepcell/models/cellsam_v<ver>/` exists.

> **If your machine intercepts HTTPS** (corporate proxy, or antivirus with HTTPS scanning),
> Python's downloader will fail with `CERTIFICATE_VERIFY_FAILED` even though your browser
> and `pip` work — the interceptor's root is trusted by the OS but absent from certifi, and
> on Python 3.13 adding it to a CA bundle does **not** help either (verification is strict
> and rejects most antivirus-generated roots). `setup_cellsam.py` detects this and tells you
> to `pip install truststore`, which routes verification through the OS trust store; it then
> retries automatically. This was the actual experience on the development machine (Norton
> HTTPS scanning) — see [`nodegraph/kernels/cellsam_segment.md`](nodegraph/kernels/cellsam_segment.md) §8.

Reference timings on a CPU-only torch build: model load ~1.5 s, then ~10 s per image
(matching the paper's benchmark; cost scales with cell count because the mask decoder runs
once per detected cell). A CUDA torch build is picked up automatically
(`NODELAB_CELLSAM_DEVICE=cpu|cuda|auto` overrides).

**Environment switches**

| Variable | Effect |
|---|---|
| `NODELAB_GL=0` | force the CPU viewer path (skip requesting a GL context) |
| `NODELAB_OVERLAYS=<path>` | read/write overlay settings from one explicit JSON file |
| `PYTHONUTF8=1` | needed on a cp1252 Windows console for the gates' `µ`/`σ`/`↔` glyphs |

---

## 2. The window

NodeLab opens on a **blank canvas** with a welcome card: *Load image… / Browse nodes /
Example graph*. The Viewer pane starts collapsed — the canvas owns the whole centre until
your first pull returns an image, which unfolds the Viewer to a ~2.5:1 split.

```
┌──────────────────────────────────────────────────────────────────────┐
│ File  Edit  Run  Graph  View  Help                                   │
├──────────┬───────────────────────────────────────────┬───────────────┤
│ Palette  │              VIEWER                       │  Properties   │
│ (search, │  image · channel strip · LUT · overlays   │  (inspector)  │
│  grouped │───────────────────────────────────────────┤               │
│  by      │              NODE CANVAS              [⛶] │  Spreadsheet  │
│  category│  cards · wires · frames · mini-map        │  (tables)     │
├──────────┴───────────────────────────────────────────┴───────────────┤
│ Console (log + full tracebacks, Copy all / Clear)                    │
├──────────────────────────────────────────────────────────────────────┤
│ status:  ● LED   node · progress · wall time                          │
└──────────────────────────────────────────────────────────────────────┘
```

* **Palette** — searchable, grouped by category. Double-click a row to place the node at
  the view centre, or drag it onto the canvas.
* **Node canvas** — pan by dragging empty space, zoom with the wheel, `Home` fits the graph.
  The `⛶` button in the top-right (or `Ctrl+Space`) maximises it.
* **Viewer** — the pulled node's image. See [§7](#7-the-viewer).
* **Properties (Inspector)** — the selected node's parameters. See [§8](#8-the-inspector--parameters-units-autopinned).
* **Spreadsheet** — the pulled node's structure tables (Label / Point / Track / Mesh).
* **Console** — a selectable log; every failed pull lands here with its **full traceback**
  and a *Copy all* button. This is where you look when a node goes red.
* **Status bar** — a pulsing LED (idle / busy / error) plus the current node and timing.

Light theme: **View → Light theme**.

---

## 3. Your first graph in five minutes

1. **File → Load ND2/TIFF file…** (`Ctrl+L`), pick a file. A source card appears titled
   with the file name, carrying one output socket **per channel** plus an *All channels*
   output. No pixels have been read yet — only metadata.
2. Drag **Gaussian Blur** from the palette onto the canvas. Wire the source's `ch0`
   output into its `data` input.
3. Select the Gaussian node. In the Inspector, note that `sigma` is in **µm** and shows an
   **auto** value derived from the file's own calibration. Type over it to pin it.
4. Add **Threshold** → wire it after the blur. Leave `method` at `otsu`.
5. Add **Connected Components** → wire it after the threshold.
6. Add **Measure** → wire it after the labels.
7. **Double-click the Measure card** (or select it and press `F5`). The engine walks
   backwards, computes only what is needed, and the Viewer opens with the result; the
   Spreadsheet fills with one row per label.
8. **Ctrl+E** exports that table to CSV / Parquet / Arrow.
9. **Ctrl+S** saves the graph as `*.nd2graph.json`.

Now change `sigma`. Only the blur and everything downstream of it recompute — the source
ingest is memoized. That incremental behaviour is the whole point of the engine.

---

## 4. Loading data

### File → Load ND2/TIFF file… (`Ctrl+L`)

Reads **metadata only** and drops a pre-configured `io.load` source node:

* **Axes** are normalised to canonical `(M, T, Z, C, Y, X)`, inserting size-1 axes for
  absent dimensions.
* **Calibration** is parsed into the engine's vocabulary: `pixel_size_um`, `z_step_um`,
  `dt_s`, `channel_emission_nm`, `objective_na`, `objective_magnification`, `bit_depth`.
  Every metadata-intelligent parameter in the graph re-seeds from this immediately.
* **Channel display** (names, emission λ, colours) drives the per-channel output sockets,
  their wire tints, and the Viewer's channel buttons.
* ND2 pixels are read through `nd2.ND2File.to_dask()`; TIFF through `tifffile` (first
  series, best-effort calibration from the TIFF tags).

**Pixels ingest lazily on the first pull.** The first pull writes a planar-block `.b2nd`
store next to the file and re-opens it lazily thereafter, so subsequent runs skip the
conversion. Expect the first pull on a large file to be slower than the rest.

### Per-channel output sockets

`io.load` and `channel.split` grow one synthetic output socket per channel (`ch0`, `ch1`, …).
These are a GUI convenience: at run time each `chK` wire is rewritten into a real
`channel.select` tap with `channels=[K]`, so the engine sees an ordinary node. The `out`
socket carries the full multi-channel bundle.

### No file? You still get a working app

An `io.load` node with an empty `path` falls back to a deterministic synthetic source
(`1×1×5×2×512×512`) with real-looking calibration — so the whole GUI, every node, and the
derive pills work out of the box. The welcome card's *Example graph* button builds an
8-node demo chain on it.

---

## 5. Building graphs

### Placing nodes

* Palette **double-click** → placed at the view centre.
* Palette **drag & drop** → placed where you drop it (a drop that lands on the welcome
  card is forwarded to the canvas beneath).
* **Drop a node onto an existing wire** → it **splices** in (the wire is rerouted through
  it). Refused cleanly on value/field wires.
* Drag from a socket and **release on empty canvas** → the **link-drag search** popup opens,
  filtered to ops with a compatible socket; picking one places it there and connects it.

### Wiring

* Press a socket and drag; the target socket rings **green** (valid) or **red** (invalid —
  wrong type, wrong direction, or it would create a cycle).
* `DATASET` pairs only with `DATASET` (the thick main wire). Value sockets connect on an
  exact type match or an implicit conversion (bool→int→float, scalar→vector broadcast,
  vector→colour). Vector arity **widens only** — 2→3 pads the axial component with 0, and
  3→2 is rejected.
* A second wire into a **non-multi** input **replaces** the existing one (Blender behaviour).
* Pressing a **connected** non-multi input **detaches** the wire and re-drags it from its
  source.
* Field sockets draw as a **diamond**, plain value sockets as a circle, Dataset as a large dot.

### Wire colours and domain chips

Dataset sockets carry a **domain rail** — small chips (`VOX`, `LBL`, `PT`, `TRK`, `MSH`, …)
naming the attribute domains present on that wire. Wires are tinted by domain, and a node
whose required domain is **missing upstream** paints a red validation chip. That is your
at-a-glance answer to "why does Measure complain?" — it needs `LABEL`, and nothing upstream
produced labels.

### Editing the graph

| Action | How |
|---|---|
| Delete | `Del` / `Backspace`, the ✕ badge that appears when you hover a card, or right-click → Delete |
| **Dissolve** (delete but keep the chain) | `Ctrl+X`, right-click → Dissolve, or Edit menu |
| Mute (bypass, pass-through) | select + `M` |
| Collapse a card to a compact strip | select + `C` |
| Select all nodes | `Ctrl+A` |
| Insert a reroute dot on a wire | **double-click the wire** |

Muted nodes dim and are bypassed at run time — the engine never sees them.

---

## 6. Running: pull, progress, errors

Nothing computes until you **pull** a node. The engine walks *backwards* from the node you
asked for and computes only what that request needs.

| Action | How |
|---|---|
| Pull the selected node | `F5`, or **Run → Pull selected node** |
| Pull & view a node | **double-click its card** |
| Re-pull the currently viewed node | `Shift+F5` |
| Preview whatever you click | **View → Preview clicked node** (always on while the canvas is maximised) |

One pull runs at a time (latest-wins queueing) on a worker thread, so the UI never blocks.
Edits invalidate in-flight results by epoch, so a stale result can never land in the Viewer.

### Reading per-node progress on the canvas

Each card wears a rail under its header plus a status dot:

| Look | Meaning |
|---|---|
| hollow dot | queued |
| pulsing dot, rail **filling** | running, and the compute reports a real fraction |
| pulsing dot, rail **sweeping** | running, cost not fractionally knowable |
| solid glowing dot, full rail | done |
| hollow ring | **memo hit** — nothing was recomputed |
| red | error |

A working card also takes an accent border and its outgoing wires flow. Exact percentages
and wall times live in the **tooltip** and the status bar, so a busy canvas stays readable.

**A lazy node finishing in microseconds is correct, not a bug.** Most enhancement nodes
return a *lazy provider*, not pixels; the real cost appears on whichever node actually reads
planes (often the Viewer's decode). The bars report the truth rather than a comforting
fiction.

### When something fails

The card turns red, the status LED goes red, and the **Console** gets the full traceback
with *Copy all*. Dependency-gated nodes raise a one-line install hint (e.g. "install
scikit-learn") rather than a mysterious `ImportError` deep in a kernel.

---

## 7. The Viewer

The Viewer renders the pulled Dataset as a **multi-channel colour composite**. There is no
title bar — the image fills the top; all controls sit in one strip beneath it, and the
viewed node's name is folded into the status line at the bottom.

### Navigation

* **M / T / Z sliders**, one row each, with a **▶ play/pause** button and an fps spinner per
  axis. Playback advances one axis frame-by-frame, self-throttled to the target fps, and
  reports the achieved rate next to the slider.
* **Scroll to zoom, drag to pan.** The view re-fits on resize unless you have zoomed or
  panned yourself.

### Channels and contrast

* One **toggle button per channel**, tinted by its emission colour (or the file's native
  channel colour when it is unambiguous). Toggle channels in and out of the composite.
* Per channel: a **histogram with draggable LUT handles**. Drag the handles to set the
  window; drag the middle dot up/down for gamma; or type `lo`/`hi` values directly.
* **Auto** re-applies percentile auto-contrast; **Fit** zooms every histogram to its window.

On the GPU path (default), LUT/gamma/colour changes are **uniform updates** — instant, with
zero decode and zero re-upload. Moving T or Z only binds a different texture (uploading a
new plane once). `NODELAB_GL=0` forces the CPU path; a GL failure falls back automatically.

### Overlays

The **◈ Overlays** button (also `Ctrl+Shift+O`, **View → Overlays…**) opens a popup with a
tab per domain: **Points**, **Labels**, **Tracks**, **Mesh** (and a reserved Voxels tab).
Quick on/off checkboxes for Points / Labels / Tracks sit in the channel strip.

* **Points** — golden-star glyphs, per-point palette, configurable arm spread.
* **Labels** — outlines and/or fills, per-label palette (fill hue matches outline hue).
* **Tracks** — one hued polyline per track through its members' `(y,x)` ordered by t, with
  the vertex at the **currently viewed T enlarged** — so scrubbing T walks each track's
  current position.
* **Mesh** — the **cross-section at the viewed Z**: closed loops where the surface cuts the
  plane, in three styles.

Every overlay size is a **screen** size, so an outline keeps its thickness as you zoom into
a label instead of being magnified with the image. Region *fills* scale, because a fill is
the region. A tab's look can be **spread** to the other tabs by role (opacity travels;
"arm spread" does not).

Settings persist as JSON in three layers, later wins: built-in defaults → the project file
shipped in the package (meant to be committed) → this machine's file. Partial files merge,
so a file written by an older build still loads.

### Maximised canvas + mini-map

`Ctrl+Space` (or the `⛶` button, or **View → Maximize node canvas**) gives the whole centre
to the graph and re-homes **the same Viewer** into a mini-map HUD pinned to the canvas's
top-left corner — channels, LUT, playback and overlays all carry across. While maximised,
**clicking any node previews it live** (debounced, so marquee-selecting a chain is one pull),
and the previewed card wears an accent spine. Drag the mini-map header to move it (it
re-anchors to the nearest corner), drag the bottom-right grip to resize, and double-click the
header — or press its dock button, or `Esc` — to put the Viewer back at its old split size.

---

## 8. The Inspector — parameters, units, auto/pinned

Selecting a node builds an editable form from its definition: the 2D/3D switch, the resolved
data-access footprint, and one row per **active** parameter, each with its unit label.

### Physical units, not pixels

Spatial and temporal parameters are authored in **physical units** and converted for you:

| Unit | Meaning |
|---|---|
| `um` / `nm` | lateral, divided by `pixel_size_um` |
| `um_axial` | axial, divided by `z_step_um` (anisotropic sampling) |
| `um2`, `um3` | areas / volumes |
| `s` | seconds, divided by `dt_s` |
| `px` | already pixels |

So a `radius = 0.5 µm` becomes the right pixel count for *this* objective, and the same
graph run on differently calibrated data does the right thing without edits.

### auto vs pinned (the sticky override)

A parameter that can be **derived from metadata** shows an **auto / pinned** toggle and an
`ƒmd` pill on the node card:

* **auto** — the value is derived from the incoming edge's metadata right now. Edit an
  upstream node and the pill updates live, before anything is computed.
* **pinned** — typing a value, or clicking the pin, fixes it. It stays fixed across
  calibration changes and file swaps. Unpinning reverts to auto.

Examples: Deconvolve's PSF derives from emission λ, NA, and pixel/z size (**per channel**);
Spot Detection's radii derive from the diffraction limit; DoG's low σ derives the same way
and its high σ auto-fills at 1.6× low.

### Mode-gated parameters

A parameter that the currently selected mode would ignore is **hidden**, not shown and
discarded. Switch `analysis.histogram_threshold`'s method and you see 1–2 live fields
instead of all eight; switch `analysis.threshold` to `otsu` and the manual `threshold` field
disappears (the method self-derives it). If a control is visible, the selected code path
reads it — that is enforced by a build gate, not by convention.

A control whose relevance depends on a *wire* rather than a mode stays visible on purpose
(DVC/DIC's `reference_frame` is dead under `previous_frame` but live again the moment you
wire an external `reference` Dataset).

---

## 9. The 2D/3D lever

Nodes whose kernel has a spatial extent carry a **2D / 3D switch in the card header**. It is
a genuine dimensionality choice, not a hint:

* **2D** — the op runs per `(Y, X)` plane.
* **3D** — the op runs on the whole `(Z, Y, X)` volume, with **anisotropic** axial
  parameters (`sigma_z`, `radius_z`, … appear only in 3D mode, defaulting to the lateral
  value).
* The default is **metadata-adaptive**: `z > 1 ⇒ 3D`.
* The switch is **greyed out when the incoming z is known to be 1** (unknown ≠ 1, so an
  unresolved source never greys it). A saved graph locked to 3D on `z == 1` paints a red
  validation badge.
* 2D and 3D memoize **distinctly**, so flipping the lever recomputes rather than serving the
  other dimensionality's cached result.

Some nodes deliberately have **no** lever:

* Pointwise ops (Gamma, Threshold) are dimension-agnostic.
* A **scope** choice is not a lever: Normalize's `plane / volume / series` is a plain mode.
* A node whose dimensionality is **inherited from its input geometry** has no lever by
  design — `transform.rasterize_field` and `analysis.accumulate_field` read it from the
  field's own provenance, `analysis.boundary_band` and `analysis.tessellate` are 3D-only.
  A lever there could disagree with the data and silently corrupt it.
* A node declaring `stack_of_2d` honesty (Wavelet Denoise, Bilateral) still shows a 3D mode
  but tells you it stacks 2D results — it never pretends to be volumetric.

---

## 10. Domains, layers and the layer picker

A Dataset on the wire is not just pixels: it carries **attribute layers** on **domains**.

**Acquisition-lattice domains** (coarsenings of the image hypercube):
`Voxel (m,t,z,c,y,x)` → `Plane (m,t,z)` → `Frame (m,t)` → `Timepoint (t)` / `Multipoint (m)`
→ `Global ()`, plus the orthogonal `Channel (c)`.

**Detected-structure domains** (produced by analysis): `Label`, `Point`, `Track`, `Mesh`.

That is what the domain chips on Dataset sockets mean. Practical consequences:

* A mask, a distance field, a band raster are **Voxel** layers.
* Segmentation results are a Voxel raster **and** a `Label` table (one row per region).
* Detections are a `Point` table; tracking adds a `Track` table **and** a `track_id` column
  written back onto the member layer.
* `analysis.tessellate` produces a `Mesh` — the one domain carrying topology (which vertices
  form which face) — which `transform.rasterize_mesh` turns back into a filled Label volume.
* `transform.transfer_domain` moves a **lattice** attribute between lattice domains
  (coarsening reduces over the dropped axes; refining broadcasts). Structure hops use the
  built-in bridges inside the relevant nodes.

### The layer picker

Nodes that consume a named layer (`mask`, `labels`, `points`, `source`, `mesh`, …) present an
**editable combo offering exactly the layer names actually present upstream** — you pick
instead of retyping. Free text is still accepted for hand-authored graphs. A node is never
offered its own output.

Two behaviours worth knowing:

* The catalog is **not monotone**. An axis-changing node (crop, resample, z-project, stack,
  channel select) **drops** lattice layers whose shape no longer matches. If a mask
  disappears after a crop, that is the rule working — re-derive it after the crop.
* Some producers write layers no socket names (drift's `drift_y`/`drift_x`, extract-boundary's
  `<labels>_boundary`, measure's columns on the label table). Those appear in the picker too.

---

## 11. Spreadsheet & export

The Spreadsheet groups the pulled Dataset's structure layers into **one table per
`(domain, layer)`** — rows are elements ordered by id, columns are attributes with the
coordinate columns (`id,m,t,c,z,y,x`) first. A mesh shows as **three** tables (elements /
vertices / faces), which is exactly how it is stored.

Voxel/lattice attributes are per-voxel rasters — those belong in the Viewer, not here.

**File → Export table…** (`Ctrl+E`):

| Format | Notes |
|---|---|
| `.csv` | one file; several tables are written in long form with leading `domain` / `layer` columns |
| `.parquet` / `.arrow` | one table, same long form, via pyarrow |

---

## 12. Organising a big graph: frames, reroutes, groups, zones

### Reroute dots

**Double-click a wire** to insert a reroute — a real identity pass-through node rendered as
a compact 22 px dot. Use them to keep long wires tidy. Refused on value/field wires (without
dropping the wire).

### Labelled frames — `Ctrl+J`

**Graph → Frame selection…** draws a titled, tinted rectangle *behind* the selected nodes.
It auto-sizes to enclose them, reflows when a member moves, and dragging the frame moves all
members. Frames are **GUI-only** — they never enter the run graph. A frame always has ≥1
member (emptying it removes it); deleting a frame keeps its nodes.

### Node groups — `Ctrl+G` / `Ctrl+Shift+G`

**Graph → Group selection…** collapses a selected **linear** sub-chain (exactly one external
Dataset input and one output) into a reusable group **definition** plus a single purple
`group:<name>` **instance** card wired to the same frontier. **Ungroup** reverses it.

* A source node cannot be grouped — its seed would be buried.
* At run time the instance is **expanded inline**, so the engine, memo and metadata pass need
  no group awareness; downstream domain rails read correctly **through** the opaque instance.
* Groups are nestable, and a group that contains itself is rejected with a clear error.
* Both the instance and its definition are saved, and round-trip.

### Repeat / Simulation zones

**Graph → Wrap selection in Repeat zone…** bounds a body between paired `In`/`Out` markers
with a feedback back-edge, then **unrolls** it into a flat per-iteration chain. Because the
result is an ordinary acyclic graph, memoization works per iteration: re-pulling an unchanged
graph is fully cached, editing the seed or a body param invalidates from that point on, and a
downstream edit leaves the zone cached.

* Iteration 0 is fed by the external Dataset wired into `In`; later iterations get the
  feedback instead. A loop-invariant input wired straight into a *body* node feeds every
  iteration.
* **Simulation** zones are the same mechanism with temporal intent. Put a **`zone.frame`**
  node in the body and iteration *t* processes frame *t* while the `In`/`Out` feedback carries
  cross-frame **state** — a scan over frames.
* A genuinely stateful/random body can be flagged impure, which makes it non-cacheable across
  pulls rather than silently wrong.

---

## 13. Saving & loading

`*.nd2graph.json` (`Ctrl+S` / `Ctrl+Shift+S` / `Ctrl+O`, **File → New** re-welcomes).

* The **structure** — nodes (op_key, params, modes), every edge including zone back-edges,
  zones, and group definitions (recursively) — is written by the engine's serializer.
  Output is deterministic and key-sorted, so saved files diff cleanly.
* **GUI-only state** rides in a top-level `ui` object the headless loader ignores: canvas
  positions, mute/collapse, frames, the source node's display title, its captured channel
  descriptors, and the sticky `__locked__` pin list.
* Loading validates: an unknown or absent `format_version`, or malformed structure, is a
  clear error rather than a half-loaded graph.
* Files from the retired first-generation editor (`.nd2s_pipeline.json`) **cannot** be
  opened, and no importer will be built.

---

## 14. Keyboard & mouse reference

### Menus

| Shortcut | Action |
|---|---|
| `Ctrl+N` / `Ctrl+O` / `Ctrl+S` / `Ctrl+Shift+S` | New / Open / Save / Save As |
| `Ctrl+L` | Load ND2/TIFF file… |
| `Ctrl+E` | Export table… |
| `Del` / `Backspace` | Delete selected nodes / wires / frames |
| `Ctrl+X` | **Dissolve** — delete and reconnect the chain |
| `Ctrl+A` | Select all nodes |
| `F5` | Pull selected node |
| `Shift+F5` | Pull viewed node again |
| `Ctrl+J` | Frame selection |
| `Ctrl+G` / `Ctrl+Shift+G` | Group / Ungroup selection |
| `Home` | Fit graph |
| `Ctrl+Space` | Maximize node canvas (`Esc` to leave) |
| `Ctrl+Shift+O` | Overlays… |

### Canvas

| Input | Action |
|---|---|
| Double-click a **node** | pull it and show it in the Viewer |
| Double-click a **wire** | insert a reroute dot |
| Drag a socket → socket | connect (green ring = valid, red = invalid) |
| Drag a **connected** non-multi input | detach and re-drag from its source |
| Drag a socket → empty canvas | link-drag search popup |
| Drop a palette node **onto a wire** | splice it in |
| Hover a card → `✕` | delete that node |
| Right-click a node / wire / frame | context menu (Delete, Dissolve, …) |
| `M` | mute / unmute selected |
| `C` | collapse / expand selected |
| `Esc` | cancel a wire drag; leave maximised canvas |
| Drag empty space / wheel | pan / zoom |

### Viewer

| Input | Action |
|---|---|
| Wheel / drag | zoom / pan the image |
| M/T/Z slider, `▶` | scrub / play that axis (fps spinner per axis) |
| Channel button | toggle that channel in the composite |
| Drag LUT handles | set the window; drag the middle dot for gamma; or type `lo`/`hi` |
| `Auto` / `Fit` | re-apply percentile auto-contrast / zoom histograms |
| `◈ Overlays` | overlay settings popup |

---

## 15. Node reference (all 54)

Notation: **lever** = has the 2D/3D header switch · **modes** = in-body dropdowns ·
`µm`/`µm²`/`µm³` = the parameter's unit · *(dep)* = needs an optional package.

### Source & channel

| Node | `op_key` | What it does |
|---|---|---|
| Load (GUI source) | `io.load` | The pipeline source. Resolves `path` to a lazy provider (ND2/TIFF → planar-block `.b2nd`), or a synthetic fallback when empty. Grows per-channel output sockets. |
| Select Channel | `channel.select` | Subset/reorder the channel axis; the per-channel emission list follows in lockstep. |
| Split Channels | `channel.split` | Fan a multi-channel Dataset into per-channel outputs; `out` still carries the full bundle. |
| Reroute | `rr.reroute` | Identity pass-through for wire tidiness (created by double-clicking a wire; hidden from the palette). |
| Viewer tap | `view.viewer` | Pure pass-through inspection tap. |

### Enhancement (15)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| Gaussian Blur | `enhance.gaussian` | `sigma` µm, `sigma_z` µm_axial (3D) | lever; tiled in 2D |
| Median | `enhance.median` | `radius` µm, `radius_z` | lever; edge-preserving |
| Morphology | `enhance.morphology` | `radius` µm, mode `erode/dilate/open/close` | lever |
| Top-Hat | `enhance.tophat` | `radius` µm, `variant white/black` | lever; background flattening |
| Difference of Gaussians | `enhance.dog` | `low_sigma` µm (diffraction-derived), `high_sigma` (0 ⇒ 1.6× low) | lever; blob band-pass |
| Unsharp Mask | `enhance.unsharp` | `radius` µm, `amount` | lever |
| Morphological Gradient | `enhance.morphological_gradient` | `radius` µm | lever; edge map |
| TV Denoise | `enhance.tv_denoise` | `weight` | lever; **true 3D** |
| Wavelet Denoise | `enhance.wavelet_denoise` | — | lever; **stack-of-2D** (declared) |
| Non-Local Means | `enhance.nlm` | `h`, `patch_size` px, `patch_distance` px | lever; **true 3D** |
| Bilateral Denoise | `enhance.bilateral` | `sigma_spatial` µm, `sigma_color` | lever; **stack-of-2D** (declared) |
| CLAHE | `enhance.clahe` | `clip_limit` | lever; rescales back into the input range |
| Gamma | `enhance.gamma` | `gamma` | no lever (pointwise), per-plane normalised |
| Normalize | `enhance.normalize` | `low_pct`, `high_pct`, scope `plane/volume/series` | **drops `bit_depth`** — output is no longer integer counts |
| Deconvolve | `enhance.deconvolve` | `na`, `emission_nm` nm, `z_step_um`, `iterations` | lever; Richardson–Lucy with a PSF derived **per channel** from optics |

### Segmentation & analysis (20)

| Node | `op_key` | Key controls | Produces |
|---|---|---|---|
| Threshold | `analysis.threshold` | method `fixed/otsu/li/yen/triangle/mean`, `threshold` (fixed only), `name` | Voxel mask |
| Multi-Otsu | `analysis.multiotsu` | `classes`, `name` | Voxel class raster (0..K-1) |
| Local Threshold | `analysis.threshold_local` | `block_size` µm, `offset`, `name` | Voxel mask (uneven illumination) |
| Connected Components | `analysis.label` | `mask`, `connectivity` (8 / 26 default), `name` | Voxel raster **+ Label table** |
| **Segmentation** | `analysis.segment` | method **threshold** (`level` otsu/li/yen/triangle/mean/fixed, `connectivity`) · **watershed** (optional `mask` layer, `min_distance` µm) · **stardist** (`prob_thresh`, `nms_thresh`, `scale`, `model_name`) · **cellsam** (`bbox_threshold`, `cellsam_model`, `model_path`, `normalize`, `postprocess`, `remove_boundaries`, `tile`+`tile_size`/`tile_overlap` px); shared: `name`, `fill_holes`, min/max **area µm²** (2D) or **volume µm³** (3D) | Voxel label raster **+ Label table**; lever: 2D = per-plane instances, 3D = z-connected *(the two learned methods refuse 3D)* |
| Distance Transform | `analysis.edt` | `mask`, `name` | µm distance field (anisotropic in 3D) |
| Measure | `analysis.measure` | `labels`, `stats`, optional **`raw`** Dataset | per-label stats on the Label table |
| Histogram Threshold | `analysis.histogram_threshold` | method `single/hysteresis/percentile/relative` × direction `below/above/between/outside`, morphology cleanup, area filters µm², optional `raw` | mask + Label raster + region table (2D) |
| Spot Detection | `detect.spots` | `min/max_radius` µm (+ `_z`), method `log/dog`, polarity `bright/dark` | Point table; lever |
| Particle Detection | `detect.particles` | `min_distance` µm, `threshold`, `min_intensity`, `min_size`, `subpixel`, mode `log/components` | Point table; lever |
| Extract Boundary | `analysis.extract_boundary` | `labels`, `name` | boundary Points (2D contours / 3D surface verts) |
| Boundary Band | `analysis.boundary_band` | method `dilation/edt`, `band_voxels`, `band_um` µm, `include_neighbors` | outward band Label raster; **3D only** |
| Cluster Points | `analysis.cluster_points` | method `gmm/kmeans`, `n_clusters`, `relax_pct`, `n_init` | per-point cluster-id column *(dep: scikit-learn)* |
| Tessellate | `analysis.tessellate` | boundary `convex_hull/alpha_shape/voronoi/label_surface`, `alpha_um` µm, `min_points`, `labels`, `iso_level`, `decimate` | **Mesh** (one closed surface per label + analytic volume/area/density); 3D |
| ROI Mask | `analysis.roi_mask` | `shapes` (serialisable shape list), `name` | boolean Voxel ROI mask; empty ⇒ whole frame |
| DVC (ALDVC) | `analysis.dvc_field` | reference `fixed_frame/previous_frame` + optional external `reference` Dataset, `subset_size/spacing` px, `search_radius`, correlation `zncc/phase`, strain `infinitesimal/green-lagrange/almansi/hencky`, `newFFTSearch` | Point displacement+strain field (µm); lever |
| Accumulate DVC Field | `analysis.accumulate_field` | `source`, `name` | cumulative Lagrangian disp+strain; **inherits** its config from the upstream DVC (refuses `fixed_frame`/no provenance) |
| DIC (pyALDIC) | `analysis.dic_correlate` | reference lever + optional `reference` Dataset, `winsize/winstepsize` px, ICGN/ADMM iterations, smoothness, optional `roi` mask | 2D Point displacement field *(dep: al-dic)* |
| Track Linking | `track.link` | target `label/point`, `max_distance` µm, `iou_threshold` | Track membership (IoU overlap / nearest neighbour) |
| Track Objects | `track.objects` | target `label/point`, method `centroid/serialtrack/topology/fingerprint/overlap` + per-method params | Track table **+ `track_id` write-back**; *(dep: numba/pandas)*. `serialtrack` is 1–2 orders of magnitude slower than the rest |

### Registration (2)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| Drift Correction | `align.drift` | — | phase cross-correlation across T; stores per-frame shift as Frame attrs (`drift_y`/`drift_x`) |
| Registration | `registration.stabilize` | model `translation/euclidean/affine/feature`, reference `first/previous/mean/template`, `ref_channel`, `upsample`, `highpass_sigma` px, `min_confidence` | **register once on a reference channel, apply to all** — preserves colocalisation |

### Transform & utility (7)

| Node | `op_key` | Key controls | Notes |
|---|---|---|---|
| Z-Project | `util.zproject` | method `max/mean/sum/min/median` | Z→1; drops `z_step_um`, marks `z_collapsed`; `sum` widens `bit_depth` |
| Crop | `util.crop` | `y0/y1/x0/x1` px (+ `z0/z1` in 3D) | pixel size preserved; lever |
| Resample | `util.resample` | `scale_xy`, `scale_z` | pixel size scales inversely; lever |
| Stack (T→1) | `util.stack` | method `mean/median/sigma_clip/trimmed_mean/max/sum` | SNR stacking; drops `dt_s`; `sum` widens `bit_depth` |
| Transfer Domain | `transform.transfer_domain` | `from_domain`, `to_domain`, `reducer`, `attr` | lattice ↔ lattice only |
| Rasterize Mesh | `transform.rasterize_mesh` | `mesh`, `smooth_um` µm, `min_voxels`, `fill_holes`, `name` | Mesh → filled Voxel Label + per-element table (concavity survives) |
| Rasterize Field | `transform.rasterize_field` | method `linear/nearest`, `source`, `prefix` | Point field → full-res Voxel layers; **no lever** (dim inherited from the field) |

### Abstraction (7)

`zone.repeat_in` / `zone.repeat_out` / `zone.sim_in` / `zone.sim_out` — zone boundary markers
(pass-through). `zone.frame` — inside a zone body, iteration *t* yields frame *t*.
`group.input` / `group.output` — the group interface markers.

---

## 16. Worked workflows

### A. Segment and measure (the bread and butter)

```
io.load ─ch0→ enhance.tophat ─→ enhance.gaussian
        ─→ analysis.segment(method=watershed) ─→ analysis.measure
                                                  ↑ raw (optional)
```

* Top-hat flattens the background; Gaussian smooths at a µm scale.
* **One Segmentation node does the whole job**: it cuts its own foreground (Otsu by
  default) and splits touching objects, emitting the label raster *and* the region table.
  Swap `method` to `stardist` or `cellsam` and nothing downstream changes — that is the
  point of the single node. Point its `mask` socket at an existing mask layer instead
  (from Threshold / ROI Mask / Histogram Threshold) when you have already made one.
* **Wire the raw source into Measure's optional `raw` input.** Then segmentation happens on
  the enhanced chain you tuned, but the reported intensities come from the pixels the camera
  actually recorded. The mask, labels and thresholds always stay on the main input.
* Geometry must match: a cropped/resampled/z-projected `raw` is refused with both shapes
  named, because reading them voxel-for-voxel would report a neighbour's intensity.
* Pull Measure → Spreadsheet → `Ctrl+E`.

### B. Detect and track puncta

```
io.load ─ch0→ enhance.dog ─→ detect.spots ─→ track.link (point, max_distance µm)
```

Turn on the **Tracks** overlay and play T: each track draws its whole trail with the current
timepoint's vertex enlarged. For heavy-duty tracking swap in `track.objects` and choose a
linker; it also writes `track_id` back onto the member layer, which is what makes the result
usable downstream (nothing consumes the Track domain directly).

### C. Drift, then analyse

```
io.load ─→ registration.stabilize (ref_channel=0, model=translation) ─→ …
```

Register **once** on one reference channel and apply the same transform to every channel and
z — that is what preserves colocalisation. `align.drift` is the lighter phase-correlation
variant; both store the per-frame shift as Frame attributes you can inspect.

### D. Strain fields (DVC / DIC)

```
io.load ─→ analysis.dvc_field (previous_frame) ─→ analysis.accumulate_field
                                              └─→ transform.rasterize_field
```

* `dvc_field` returns a **Point** field at subset centres with displacement in µm plus
  strain; 2D per-plane or 3D volumetric via the lever.
* `accumulate_field` composes `previous_frame` increments into a **cumulative Lagrangian**
  series. It does not ask you to restate the reference mode — it **inherits** it from the
  upstream node's stamped provenance, and refuses a `fixed_frame` field (already cumulative)
  or a non-DVC input.
* `rasterize_field` interpolates the point field up to full-resolution Voxel layers for
  display. It has no 2D/3D lever: it reads the field's own dimensionality, so a 2D per-plane
  field on a `z > 1` image is never misread as a 3D grid.
* For 2D image pairs use `analysis.dic_correlate` instead (it can also take an ROI mask from
  `analysis.roi_mask`). Note the first al-dic call JIT-compiles numba (~5–6 s, once per
  process).

### E. Point cloud → mesh → label volume

```
detect.particles ─→ analysis.cluster_points ─→ analysis.tessellate (alpha_shape)
                                            ─→ transform.rasterize_mesh ─→ analysis.boundary_band
```

`tessellate` builds one closed surface per label with analytic volume/area/density;
`rasterize_mesh` fills it into a Label volume — and because the interior test is derived from
the mesh's own provenance, a concave alpha-shape fills **less** than its convex hull (which is
a real bug this split fixed). `label_surface` boundary mode meshes an existing label raster
via marching cubes instead of a point cloud.

### F. Per-frame simulation with carried state

Wrap the body in a **Sim zone**, put a `zone.frame` node inside it, and set iterations = T.
Iteration *t* processes frame *t* while the `In`/`Out` feedback carries state across frames.
Each iteration memoizes independently, so editing frame 40's parameters does not recompute
frames 1–39.

---

## 17. Headless / scripted use

The engine is Qt-free, and a graph saved by the GUI can be run without PySide6:

```python
import nodegraph.nodes                       # importing registers the 55-node catalog
from nodegraph.serialize import from_json
from nodegraph.groups import expand
from nodelab_v2.ops import headless_engine   # Qt-free by design

graph, zones, groups = from_json(open("my.nd2graph.json").read())
if groups:
    graph = expand(graph, groups)            # inline group instances
if zones:
    from nodegraph.zones import unroll
    graph = unroll(graph, zones)             # flatten Repeat/Sim zones (iterations ride the Zone)

engine = headless_engine(graph, seeds={"load1": my_dataset},
                         meta_seeds={"load1": my_envelope})
result = engine.pull("node_id")              # → a Dataset
```

Notes:

* You supply the source: a headless consumer provides its own seed `Dataset` for each
  `io.load` node (`nodelab_v2.ingest.ingest_nd2(path, store_path)` gives you a provider +
  a metadata envelope). `headless_engine` calls `ensure_ops()` and materializes the
  per-channel taps for you; group expansion and zone unrolling are yours to do (the GUI's
  `document.to_graph` does both).
* `Engine(strict_reads=True)` turns any un-fenced calibration read into a hard error — use it
  when developing nodes.
* `Engine(memo_bytes=…)` caps the memo with a byte-budget LRU (the GUI uses 1 GiB);
  `cache_bytes=…` sizes the streaming tile cache.
* `Engine(observer=fn)` gives you `start` / `progress` / `done` / `cached` / `error` events
  per node.

---

## 18. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| "install scikit-learn / al-dic / stardist / numba" | that node is dependency-gated; install the extra from [§1](#1-install--launch) |
| A node shows a **red domain chip** | it requires a domain nothing upstream produced (e.g. Measure needs `LABEL` — add Connected Components) |
| The **3D switch is greyed out** | the incoming `z` is known to be 1. Z-project or a `z==1` file will do that |
| Red validation badge on a card | the graph is locked to 3D but `z == 1` |
| A **mask vanished** after Crop/Resample/Z-Project | the layer catalog is not monotone: an axis change drops lattice layers whose shape no longer matches. Re-derive the mask after the geometry change |
| `analysis.histogram_threshold` refuses the input | it needs raw integer counts; something upstream (Normalize, CLAHE) produced `[0,1]` floats and honestly dropped `bit_depth` |
| Raw-input geometry error on Measure | the optional `raw` Dataset must match voxel-for-voxel; both shapes are named in the message |
| A node finishes instantly with a full bar | it returned a lazy provider; the real cost lands on whichever node reads planes |
| Viewer went black after docking/undocking | the GL context was recreated; it rebuilds and replays automatically. `NODELAB_GL=0` forces the CPU path |
| First pull on a big ND2 is slow | the one-time `.b2nd` planar-block ingest next to the file; later pulls reopen it lazily |
| Console shows a `mojibake`/`UnicodeEncodeError` running the gates | prefix `PYTHONUTF8=1` (Windows cp1252 vs the `µ`/`σ`/`↔` glyphs) |
| Re-pulling recomputes a chain you expected cached | an ancestor was evicted by the memo's byte budget; eviction only costs a recompute, never correctness. Raise `memo_bytes` if you have RAM |
| `serialtrack` seems hung | it is 1–2 orders of magnitude slower than the other linkers (≈8–18 s on 1k–15k detections) |
| An old `.nd2s_pipeline.json` will not open | that is the retired first-generation format; it is not supported and no importer exists |

---

## 19. Verifying a build

```bash
PYTHONUTF8=1 python -m nodegraph.selftest                        # headless core → 55 groups
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png  # driven GUI probe
```

Both must end in `ALL NODEGRAPH SELF-TESTS PASSED` / `ALL PHASE-5 GUI PROBES PASSED`.
Heavier, not in the fast gate:

```bash
python scripts/_ingest_nd2_smoke.py            # a real ND2 end-to-end
python scripts/_bench_provider_granularity.py  # the storage-layout keystone benchmark
python scripts/_bench_ccl_watershed.py         # CCL / watershed cost
python scripts/_nodelab_v2_shot.py out.png     # one offscreen render (--welcome for launch state)
```

**Adding a node?** Read the **`wire-node-v2`** skill for the concepts, then follow
**`build-node-v2`** for the procedure. Both gates above are that procedure's exit criteria.
