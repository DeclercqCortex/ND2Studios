# CONTEXT — nodegraph v2 handoff (fresh-chat brief) · 2026-07-22

> **Read this first.** Self-contained state + what's next for the **nodegraph v2** effort
> (a greenfield, Blender-geometry-nodes-style node system for the ND2Studios microscope
> image-analysis app). Depth lives in the sibling docs (§8); this is the orientation +
> forward map.

---

## 0b. Phase 7 — DONE (2026-07-22, scoped coexistence)

The LAST phase landed as **scoped coexistence** (user-chosen after a parity assessment
showed v2 is NOT at full parity — ~30 specialized v1 nodes unported: DIC/DVC, granule,
cell-tracker, DL seg, interaction/workflow). **`run.py` now launches NodeLab v2 by
default; `python run.py --legacy` launches v1.** Nothing was deleted; the `pipeline_kit`
parity gate stays green. Full matrix + retirement exit-criterion in
[`V2.05_phase7_capability_matrix.md`](V2.05_phase7_capability_matrix.md). A hard v1
removal is deferred until the gap list closes or is declared obsolete. With this, the v2
build sequence (Phases 1–7) is complete for the covered workflow; the open work is
porting the §2 gaps and the deferred GUI/viewer refinements.

## 0. Where we are — one paragraph

The **headless core is complete, hardened, and broad**: a lazy pull + two-hash-memo
engine; a 39-node metadata-intelligent catalog with the 2D/3D lever; domain transfer +
structure bridges (incl. multi-hop); Repeat/Simulation **zones** (with per-frame-T) and
nestable node **groups**; frame-to-frame **tracking**; real **ND2 ingest** to an on-disk
b2nd store; JSON **serialization** of the graph; and — as of 2026-07-22 — **C1 per-tile
streaming eval** (`nodegraph/streaming.py`, design locked in
[`V2.04`](V2.04_c1_streaming_eval.md)): computes return lazy chained providers instead of
realizing the 6-D array, with probe-verified overlap-recompute halos, an engine-owned
byte-budget tile/field cache keyed by staleness-proof flat fingerprints, PartialReducer
tree-reduce (zproject), and windowed per-tile field evaluation. Everything is Qt-free
except `nodelab_v2/` (a first GUI slice). The big thing left is **Phase 5** (fleshing out
the `nodelab_v2` Qt GUI, which hits the open asyncio↔Qt decision at G7). Dep-gated
ML/sparse nodes are blocked on uninstalled packages.

---

## 1. Gates — must stay green (run these first in any session)

```bash
PYTHONUTF8=1 python -m nodegraph.selftest          # → 38 groups, "ALL NODEGRAPH SELF-TESTS PASSED"
python scripts/_pipeline_kit_parity.py --check     # → "no drift" (v1 pipeline_kit untouched)
```
- **`PYTHONUTF8=1` on Windows** — some `[ok]` lines carry `↔`/`σ`/`µ` glyphs the cp1252
  console chokes on (cosmetic, not a failure).
- Catalog = **39 node types**. `python -m nodelab_v2` boots the GUI slice; offscreen render:
  `python scripts/_nodelab_v2_shot.py out.png` (see §7 offscreen gotchas).
- Heavier, file-dependent, NOT in the fast gate: `scripts/_ingest_nd2_smoke.py` (real ND2),
  `scripts/_bench_ccl_watershed.py` (C6), `scripts/_bench_provider_granularity.py` (keystone).

**Env:** Python 3.13.1; numpy 2.4.2, scipy 1.17.1, scikit-image 0.26.0, blosc2 4.9.1,
zarr 3.2.1, pyarrow 23.0.1, nd2 0.11.2, PySide6 (+PyQt6). **Installed but model/GPU-gated:**
stardist, csbdeep, dask. **Absent:** cellpose, bart, tensorstore, PySAP. **Never `pip
install` unprompted; re-verify any backend API in-env before writing against it.**

---

## 2. What's BUILT (orientation only — don't rebuild)

**`nodegraph/` core (Qt-free):** `domains`, `reducers` (mean/…/median + fusion
`sigma_clip`(MAD-robust)/`trimmed_mean` + `PartialReducer` tree-reduce), `revision`,
`dataset` (axes + calibration + attribute layers + `reshaped_axes`), `transfer`
(lattice-generated + bridge routing + **C2 `execute_bridge_plan`** multi-hop), `sockets`,
`registry` (`NodeSpec`/`define_node`, `DimMode` lever, `Granularity`, `available_in`
variant sockets, `meta_transform`), `graph` (+ `Edge.kind` back-edges), `metadata`
(edit-time `MetaEnvelope` pass + the meta_transforms), `provider` (Synthetic + in-memory
& **on-disk** `B2ndProvider` + `ArrayProvider`; content-`fingerprint`/`version` — **C5**),
`memo` (two-hash; recipe_hash = op+params+upstream recipe_hashes+**revisions**; declared
`ctx.calib` reads re-validated on hit; `output_fingerprint` cutoff/dedup), `engine` (lazy
pull + `ReadContext` + granularity routing + **C7 `strict_reads`**), `structure`
(CCL / seeded-watershed / columnar tables / `TrackMembership`), `bridges`, `field`,
`boundary`, `nodes` (the 39-node catalog), **`zones`** (Repeat/Sim unroll + revision-fold +
impure epoch + `zone.frame` per-frame-T), **`groups`** (nestable inline `expand`),
**`tracking`** (`link_labels` IoU / `link_points` NN → `TrackMembership` — **C3**),
**`serialize`** (`*.nd2graph.json` to/from dict/json).

**Catalog (39):** `channel.select`; `enhance.{deconvolve, gamma, gaussian, median,
morphology, tophat, dog, unsharp, tv_denoise, wavelet_denoise, clahe, normalize,
morphological_gradient, bilateral, nlm}`; `analysis.{threshold(+otsu/li/yen/triangle/mean),
multiotsu, threshold_local, label, measure(multi-stat), edt, watershed, extract_boundary}`;
`detect.spots (LoG/DoG × bright/dark)`; `util.{zproject, crop, resample, stack}`;
`align.drift`; `transform.transfer_domain`; `track.link`; the zone boundaries
`zone.{repeat_in, repeat_out, sim_in, sim_out, frame}`; the group boundaries
`group.{input, output}`.

**App layer:** `nodelab_v2/ingest.py` (**C4** ND2→6-D→b2nd + calibration; reuses v1
`read_nd2_metadata_extended`). `nodelab_v2/` = the Qt GUI first slice (canvas + inspector).

**Core roadmap items:** ~~C2~~ ~~C3~~ ~~C4~~ ~~C5~~ ~~C6~~ ~~C7~~ done. **C1, C8 open.**

**LOCKED DESIGN (2026-07-22, user):** Simulation-zone × memo = *unroll + revision-fold*
(the back-edge is a forward edge between iteration copies; iteration t folds iteration
t-1's `Out` revision → the memo works unchanged + incremental invalidation) + *debug-verify
(`assert_zone_pure`) + impure escape hatch (`Zone.impure` → epoch salt)*. This resolved the
V2.00 §16 open risk. See `nodegraph/zones.py`.

---

## 3. What REMAINS — prioritized

**A. ~~C1 — per-tile lazy/streaming eval~~ ✓ DONE 2026-07-22.** Design locked by the user
([`V2.04`](V2.04_c1_streaming_eval.md): A1 provider-chaining, B1 overlap-recompute halos,
C1 two-level cache, D1 windowed-minimal fields) and implemented (`nodegraph/streaming.py`
+ engine/field/provider/nodes/zones wiring; selftest 39 groups incl. `streaming eval`).
**Remaining C1 slivers (V2.04):** `util.stack` tree-reduce; resample/normalize/drift lazy
units; the kernel-param field gate (activates when kernel params gain field consumers);
Memo GC (eager attr nodes in high-T zones are still a memory hazard); computed-provider
pyramids (a Viewer/G4 decision).

**B. Phase 5 — GUI (`nodelab_v2/`) — CORE SLICE LANDED 2026-07-22.** Done:
**G1** interactive wiring (drag-connect w/ live validity, detach-redrag, link-drag
search, delete), **G2** registry palette (search + drag-to-canvas), **G3** mute-with-
passthrough, **G4** baseline Viewer (Voxel heatmap + m/t/z/c + auto-contrast; plane read
in the worker), **G6** save/load `*.nd2graph.json` (wraps `nodegraph.serialize` + a `ui`
extras object), **G7** canvas→Engine (the LOCKED QThreadPool + epoch-registry bridge, no
qasync; persistent Memo; double-click/F5 to pull), **G8** z==1 lever guard + live ƒmd
re-seed, **G10** chrome + title elide. Architecture: Qt-free **`document.py`**
(`GraphDocument`) is the model; **`runner.py`** (`EngineRunner`) is the thread bridge;
the scene mirrors the document. Verified by `scripts/_nodelab_v2_phase5_probe.py`
(offscreen, real off-thread pull). **Remaining:** G5 spreadsheet, G9 light theme,
splice-on-wire + multi-input-order UI, collapse/reroute/frames, GUI zone/group *creation*
(a loaded file's zones/groups round-trip in serialize but the GUI drops them on re-save),
Label/Point/Track viewer overlays + the optional QRhi GPU path (Phase 6).

**C. Catalog / node refinements (mostly clean, additive):**
- **CCL fast path (C6-justified):** swap `structure.label_components`' pure-python flood-fill
  for `scipy.ndimage.label` (~1200× faster; ~198 s→fast at 6554²) — **but preserve the
  raster-canonical id ordering + invariant table schema** (behavior-sensitive; its own task).
- **Fill Boundary** node (inverse of `extract_boundary`; needs multi-contour reconstruction +
  geometry-authority), **Store/Read Attribute**, **Attribute→Field** (grows the field IR —
  design decision §6), **structure-domain Transfer Domain** node (wrap `execute_bridge_plan`),
  **Channel Merge** (multi-input → grow c; exposes a `propagate_meta` single-pred limitation),
  **Bead detect** (≈ a `detect.spots` preset), a real **Gibson–Lanni PSF** for deconvolve.
- **Dep-gated (blocked):** Nuclei/Cellpose (absent), StarDist/CARE (need models+TF), sparse
  recon BART/ODL/SPORCO (BART absent), multiscale starlet PySAP (absent).

**D. Zone/group refinements:** nested/overlapping zones (unroll rejects them today — subtle:
inner-per-outer-iteration feedback composition); field-driven N (iterations from a value
socket — currently a static `Zone.iterations` int); zone×2D/3D toggle validator (V2.03 §2 A7);
group multi-socket interface + parameterized instances + a reusable group library.

**E. C8 — per-channel derive resolution (`ctx.channel`)** for c-iterating nodes (deconvolve
PSF per channel). The per-c PSF path works via `_channel_emission`; a general `ctx.channel`
eval-time accessor is the cleanup (fuzzy — clarify the concrete need first).

**F. Phase 6/7:** inspection & export wired to the pull engine (Viewer+Spreadsheet+export);
retire `pipeline_kit` from NodeLab once v2 reaches parity (keep v1 runnable until then).

---

## 4. THE decision that gates the most work

~~C1 design~~ — **RESOLVED 2026-07-22 (user locked V2.04).**
~~G7 asyncio↔Qt bridge~~ — **RESOLVED 2026-07-22 (user locked QThreadPool + epoch
registry over qasync).** Implemented in `nodelab_v2/runner.py`. The next open items are
scoped decisions inside Phase 6 (Label/Point/Track viewer overlays; whether the QRhi GPU
viewer path is needed — only if the CPU baseline can't hold 60fps) and the light theme
(G9, mechanical). No blocking cross-cutting decision remains.

Other still-open decisions (grill before committing): multi-layer keying UX (a "which mask /
point set / tracking" selector — GUI); field IR depth (how expressive: arithmetic +
`Sample`/`Nearest`); migration reality (greenfield — old `.nd2s_pipeline.json` not opened).

---

## 5. How to add / change a node (the workflow)

Use the **`build-node-v2`** skill (procedure) + **`wire-node-v2`** (concepts) — the ACTIVE
node-creation skills. (`build-node` / `wire-node` are LEGACY v1 `pipeline_kit`.) In short:
a node = `register_node(compute, op_key=..., **spec)` in `nodegraph/nodes.py`
(`define_node` builds the `NodeSpec`; `COMPUTES[op_key]=compute`). Declare per-dim
`granularity`/`kernel_axes`; unit/derive on every spatial/temporal param; a `meta_transform`
if it changes `AxisSizes`/calibration; read calibration via `ctx.calib` (recorded → memo-
fenced) and convert with `to_pixels_v2`; a realized array wraps in `ArrayProvider`; structure
results attach via `Dataset.with_structure`. Then add an end-to-end pull to `selftest.py`
(assert: finite/correct; 2D vs 3D distinct recipe hashes; axis-changing payload axes ==
`engine.env(node).axes`; the right calibration key appears in `dict(engine.entry(node).reads)`).

---

## 6. Working norms (observed) + gotchas

- **Grill before big design; present before locking.** Keep `V2.00` canonical + add addenda.
  For big autonomous decisions the user reserves, ASK (one question at a time, with a recommendation).
- **Rigor + adversarial verification.** After building, review; mark unverified claims. **Note:**
  the review/verify *workflows have intermittently HUNG* (agents start, 0 results — an API/spend
  stall). Fallback that has worked: **stop waiting and self-review with direct `python` probes**
  covering the same lenses (build a real Engine, assert edge cases). Don't block on a stuck workflow.
- **Verify code you didn't write.** Read a subagent's module before integrating/shipping it.
- **Parallel subagents only when non-conflicting:** each owns ONE new file; the orchestrator
  does the shared-file edits (`nodes.py` registration, `selftest.py` wiring) sequentially.
- **Gotchas:** test/demo fixtures MUST use fake op_keys (`test.*`/`io.*`/`eng.*`) — a fixture
  `define_node`-ing a *real* op_key clobbers it in the global `NODES` registry (order-dependent
  "passes alone, fails in suite"). An axis/calibration-changing compute must SYNC to its
  `meta_transform`'s post-value, never re-derive (`ctx.calib` returns the already-transformed
  env → double-count). Identity is the monotonic `revision`, never `id()`, never a content hash
  for lookup. Offscreen Qt: register Windows TTFs (else tofu), `os._exit(0)` to bypass the exit-5
  teardown crash, `setParent(None)` before an offscreen grab.

---

## 7. Key files

- **Design (canonical + addenda):** `V2.00_nodegraph_blender_revamp.md` (11 locked decisions),
  `V2.03_directive_metadata_toggle.md` (per-edge metadata + 2D/3D lever), `V2.02_phase2a_spec.md`
  (Phase-2a spec), `V2.01_data_io_research.md`, `../Architecture/node_backend_reference.md`.
- **Remaining-work map (per-area, with ✓/deferred marks):** `CONTEXT_v2_remaining.md`.
- **Running build log (dated, what-was-done):** `CONTEXT_nodegraph_v2_handoff.md`.
- **Code:** `nodegraph/` (24 modules), `nodelab_v2/` (GUI + `ingest.py`), `scripts/_*.py`
  (parity, selftest is `python -m nodegraph.selftest`, benchmarks, ingest smoke, GUI shot).
- **Skills:** `.claude/skills/{wire-node-v2, build-node-v2}` (active), `{wire-node, build-node}`
  (legacy v1), `grilling`.
