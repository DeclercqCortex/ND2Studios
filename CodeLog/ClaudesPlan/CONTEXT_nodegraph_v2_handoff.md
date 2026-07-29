# CONTEXT — nodegraph v2 continuation brief (handoff for a fresh chat)

> **For the clean "everything that REMAINS" map, read [`CONTEXT_v2_remaining.md`](CONTEXT_v2_remaining.md)
> (2026-07-21).** This file is the running build log (what's been done, per section); the
> remaining doc is the forward-looking roadmap organized by area with gates, gotchas, and pointers.

Read this first. It is the self-contained state + remaining plan for the **nodegraph v2**
effort (a greenfield, Blender-geometry-nodes-style node system for the ND2Studios microscope
image-analysis app). Everything below is current as of 2026-07-21.

## 0. TL;DR — where we are  (updated 2026-07-22)

- **PHASE 5 GUI CORE SLICE LANDED (2026-07-22).** NodeLab v2 is now a working editor,
  not a static canvas. New Qt-free **`nodelab_v2/document.py`** (`GraphDocument`: node
  records, validated wiring [`sockets.can_connect` + cycle rejection + non-multi
  replace], live `propagate_meta` re-seed, mute bypass, `*.nd2graph.json` save/load via
  `nodegraph.serialize` + a `ui` extras object) and **`runner.py`** (`EngineRunner`: the
  **G7 QThreadPool + epoch-registry** bridge — user-locked over qasync; persistent Memo
  across engine rebuilds → an unrelated edit recomputes only the invalidated chain;
  `io.load` source resolution with a `SyntheticProvider` fallback so the GUI runs out of
  the box). `scene.py` (document-mirroring canvas + wire-drag state machine + detach-
  redrag + link-drag search + delete/mute), `node_item.py` (bound to a `NodeRecord`,
  env-driven ƒmd pills, H11 z==1 lever guard where **unknown ≠ 1**, mute paint, elided
  titles), `edge_item.py`, `inspector.py` (edits → `doc.touch`, live auto-box re-seed),
  new `palette.py` + `viewer.py`. **Done:** G1/G2/G3-mute/G4-baseline/G6/G7/G8/G10.
  **Remaining:** G5 spreadsheet, G9 light theme, splice/multi-input-order UI, collapse/
  reroute/frames, GUI zone/group creation, viewer overlays + optional QRhi (Phase 6).
  Verified offscreen by `scripts/_nodelab_v2_phase5_probe.py` (drives wiring rules,
  H11, mute, save/load round-trip, and a REAL off-thread pull → viewer pixels: 1.3s
  first / ~0.26s memoized re-pull). Core selftest 39 groups + parity stay green.

- **C1 — PER-TILE STREAMING EVAL LANDED (2026-07-22; design LOCKED by user same day,
  [`V2.04_c1_streaming_eval.md`](V2.04_c1_streaming_eval.md)).** The biggest core item:
  node computes no longer realize the whole 6-D array. Four forks locked on the
  recommended options after a grilled design pass (39-node audit fan-out + probe-verified
  halos + a 4-lens adversarial design review whose 27 findings were folded in):
  **A1 provider-chaining** (a TILEABLE/WHOLE_PLANE/WHOLE_VOLUME compute returns a lazy
  `StreamProvider` — `_map_image` is the choke point; realization only at sinks/`WHOLE_*`
  gathers/`realize()`), **B1 overlap-recompute halos** (bytes-exact, probe-verified:
  gaussian `int(4σ+0.5)` tight, median `w//2`, open/close/tophat `2·(w//2)`, DoG from
  `max(lo, hi or 1.6·lo)`), **C1 two-level memo** (node-level Memo unchanged; NEW
  engine-owned byte-budget LRU `TileCache` keyed by a FLAT provider fp digest folding
  op+params+**declared calibration reads**+**field expr hashes**+base fp — closes the
  reseed_meta/layer-revision staleness holes; canonical-tile decomposition gives every
  chain level cache reuse; cum-halo fence promotes blown-up tile chains to plane units),
  **D1 windowed-minimal fields** (windowed `FieldContext`, full-unit-address tokens,
  `ctx.input(name)`, engine `FieldCache` sharing the budget; flagship = per-tile field
  fixture + `analysis.threshold` consuming a wired per-voxel Field). Six
  misdeclared-TILEABLE traps re-declared honestly (gamma/tv/wavelet/clahe/bilateral/nlm →
  plane/volume units — their plane-global stats would be WRONG per tile); crop → lazy
  `WindowView` (dtype-preserving); zproject → `ZReduceProvider` PartialReducer
  tree-reduce; deconvolve → lazy per-plane/per-volume with eagerly-resolved per-c PSFs;
  `assert_zone_pure` now `realize()`s (structural fps are equal by construction);
  `ReadContext.freeze` makes a late `ctx.calib` from a lazy closure a hard error; scoped
  `recursion_headroom` covers deep unrolled chains (300-deep chain in selftest).
  Selftest = **39 groups** (+`streaming eval`: tiled≡eager bytes, laziness/one-window
  reads, reseed re-keys fp, windowed fields, tree-reduce, cum-halo fence, deep chain,
  late-read fence, impure-lazy-body caught); parity clean; real-ND2 ingest smoke passes
  through the lazy path. **Post-impl adversarial review (17 findings → 9 probe-confirmed
  + 8 minors, ALL fixed + guarded as selftest R1–R9):** tile grids folded into
  fingerprints (BLOCKER — cross-grid TileCache aliasing), cum-halo resets at realized
  units, windowed refine/broadcast for coarser-domain fields, threshold field token
  folds the provider fp, WEAK cache refs (no orphaned-cache pinning across engine
  rebuilds), one NaN policy for zproject, effective-unit oversize sizing, seed
  `__seed_version__` identity, z-guards/frozen-empties/relative recursion headroom/
  non-multi edge guard/DoG validation. **Deferred (V2.04):** `util.stack` tree-reduce,
  resample/normalize/drift lazy units, kernel-param field gate (activates when kernel
  params gain field consumers), Memo GC, computed-provider pyramids (Viewer G4).

- **Catalog batch 3 + C6 benchmark LANDED (2026-07-22).** `analysis.threshold` gained
  histogram methods (otsu/li/yen/triangle/mean); new `analysis.multiotsu` (K-class raster),
  `analysis.threshold_local` (adaptive), `transform.transfer_domain` (lattice attribute
  A→B reduce/broadcast, wraps `execute_transfer`). C6 = `scripts/_bench_ccl_watershed.py`:
  pure-numpy flood-fill CCL is ~1200× slower than `scipy.ndimage.label` at 512², ~198 s
  extrapolated at 6554² → whole-frame CCL **needs a scipy fast path** (watershed already
  fast). Selftest = **38 groups** (+`catalog (batch 3)`); parity clean; catalog = **39 node
  types**. **Deliberately deferred (need a decision / deps / Qt, not autonomous-safe):**
  C1 per-tile streaming eval (big engine refactor — warrants a grilled design, several open
  choices: halo policy, lazy-tiled-provider protocol, per-tile memo granularity); Phase-5 Qt
  GUI (also hits the open asyncio↔Qt G7 decision); dep-gated nodes (Cellpose/BART/PySAP absent);
  nested zones, field-driven N, C8 per-channel derive, Fill-Boundary, Gibson–Lanni PSF.

- **Per-frame-T Sim specialization + graph serialization LANDED (2026-07-22, parallel:
  per-frame-T by the orchestrator, `serialize.py` by a background subagent).** (1) Per-frame-T:
  a `zone.frame` node (`nodes.py`) slices frame t (lazy `_FrameView` + t-bearing-layer keepdim
  slice; `frame_slice` meta_transform T→1); `unroll` stamps `__frame__=i` per iteration
  (`zones.FRAME_OP`) — iteration t processes frame t while Sim feedback carries state. **No
  `Zone` schema change**, so it stayed orthogonal to serialization (that's why the two ran in
  parallel). (2) `nodegraph/serialize.py` — `to_dict/from_dict` + `to_json/from_json`
  (`FORMAT_VERSION="2.0"`) round-tripping graph (nodes+params+modes+`__locked__`, edges incl.
  `kind="back"`), zones, and groups (nested bodies); deterministic + version-validated — the
  headless core of `*.nd2graph.json` (G6). Selftest = **37 groups** (+`zones per-frame-T`,
  `serialize`); parity clean; catalog = **36 node types**. Reviewed by direct edge-case probes.

- **Node groups (Phase 4b) + C3 tracking LANDED (2026-07-22, built in parallel by two
  subagents, then orchestrator-integrated + reviewed).** `nodegraph/groups.py` — `Group` +
  `expand()` inline transform (nestable to a fixed point; `group.input`/`group.output`
  pass-throughs). `nodegraph/tracking.py` — `link_labels` (max-IoU) + `link_points`
  (nearest-neighbour) → a `TrackMembership` (the missing producer for the 4 Track bridges);
  the `track.link` node (Mode label/point) attaches it. Selftest = **35 groups** (+`groups`,
  `tracking`); v1 parity clean; catalog = **35 node types**. The subagents each built ONE
  new module in isolation (no shared-file edits); the orchestrator did the `nodes.py`
  registrations + selftest wiring sequentially. Reviewed by direct edge-case probes (two
  instances of one group; tracking merge/empty-frame/out-of-range) — all clean.

- **Phase 4a — ZONE MODEL LANDED (2026-07-22).** `nodegraph/zones.py` (`Zone` + `unroll` +
  `assert_zone_pure`), `graph.py` `Edge.kind` back-edges (invisible to `preds`/`roots`/
  `topo_order`), 4 `zone.*` boundary pass-through nodes. Implements the locked decision
  below. Selftest = **33 groups** (`zones`: Repeat N×, Sim feedback, revision-fold caching,
  impure epoch escape hatch, debug-verify); v1 parity clean; catalog = 32 node types.
  Adversarial-review workflow **stalled** (agents hung, 0 results — an API/spend stall);
  substituted by a direct 3-lens self-review (unroll wiring incl. multi-input/N=1/empty-body,
  memo revision-fold non-collision, impure/determinism guard) — **all clean, no defects**.
  Deferred (Phase 4b): node groups, nesting/cross-zone edges, per-frame-T Sim slicing,
  field-driven N, serialization.

- **DECISION LOCKED (2026-07-22, user) — Simulation-zone × memo (V2.00 §16 risk resolved).**
  (1) **Keying = unroll + revision-fold:** a zone is an unrolled per-iteration chain; the
  back-edge (`Out`(t) → `In`(t+1)) is a *forward* edge between iteration copies, and
  iteration t's `Sim/Repeat Out` **revision** folds into iteration t+1's `recipe_hash` —
  reusing the EXISTING upstream-revision machinery, so the memo works untouched and
  incremental invalidation is automatic (edit frame k → iters ≥k recompute, <k cached),
  consistent with the "identity = monotonic revision, never content-hash-for-lookup"
  invariant. (2) **Determinism = debug-verify + impure escape hatch:** assume body purity
  in the hot path; an opt-in debug double-compute-and-compare catches impurity in tests;
  a zone flagged `impure` is non-cacheable (recomputes). Implemented in `nodegraph/zones.py`.

- **C4 — real ND2 ingest LANDED (2026-07-22).** New `nodelab_v2/ingest.py` (Qt-free,
  nd2-coupled; `nodegraph` stays nd2-free): `ingest_nd2(path, store_path)` → a
  `B2ndProvider` (in-memory or an on-disk planar-block `.b2nd` store) + a source
  `MetaEnvelope`. Pixels via `nd2.to_dask()` → canonical `(M,T,Z,C,Y,X)`; calibration
  reuses v1 `read_nd2_metadata_extended` (filtered to `CALIBRATION_KEYS`). Added
  `B2ndProvider.write`/`.open` (disk store; `version`/`fingerprint` fold in `mtime_ns`,
  closing the C5 disk hook). Fast gate = 32 selftest groups (disk-provider round-trip
  folded into `provider`); real-file path verified by `scripts/_ingest_nd2_smoke.py`
  on the 6554² NileBlue sample (pixels match raw nd2; Load→SelectChannel→Gamma pulls
  end-to-end). **Remaining sliver:** a graph-level Load-ND2 *source node* (GUI-facing).

- **Phase 3 batch 2 + core hardening LANDED (2026-07-21).** `nodegraph.selftest` = **32 groups**
  green; v1 parity clean; catalog = **28 node types**. Added: core **C5** (`TileProvider.version`
  → source memo key; `ArrayProvider`+`B2ndProvider` content-hash their identity), **C7**
  (`Engine(strict_reads=True)` + `_StrictCalibMetadata` dict-subclass, laundered out of outputs),
  **C2** (`transfer.execute_bridge_plan` chains routed multi-hop transfers per frame/volume);
  nodes `enhance.{morphological_gradient, bilateral (stack-2D), nlm (true-3D)}`, `detect.spots`
  +method(log/dog)+polarity(bright/dark)+anisotropic-3D, `analysis.{edt, watershed (+`_labeled_table`
  region-props), extract_boundary, measure(multi-stat)}`, `util.{resample, stack}`, `align.drift`;
  fusion reducers `sigma_clip`(MAD-robust)+`trimmed_mean` in `reducers.py`. **Skills rewritten
  for v2** (`wire-node-v2`/`build-node-v2`; v1 re-scoped to legacy). Adversarially reviewed
  (6-lens workflow → 8 confirmed findings, all fixed + regression-guarded): B2nd structural-identity
  collision, strict-wrapper output leak, 3D-LoG sub-voxel-σ flood (floor→1.0), dark-polarity flat-guard
  bypass, watershed µm-vs-index peak radius (→ anisotropic footprint), `trimmed_mean` NaN-blindness,
  `sigma_clip` MAD==0 std-fallback re-admitting outliers, `execute_bridge_plan` empty-axes broadcast.

- **Phase 3 node-catalog port — FIRST BATCH LANDED (2026-07-21).** 12 nodes ported into
  `nodegraph/nodes.py` (catalog now **19 node types**); `nodegraph.selftest` = **28 groups**
  green (+`catalog (ported)`), parity green. Nodes: `enhance.median/morphology/tophat/dog/
  unsharp/tv_denoise/wavelet_denoise/clahe/normalize`, `detect.spots`, `util.zproject`,
  `util.crop` — each declares per-dim `granularity`/`kernel_axes` + `unit`/`derive` params;
  the axis-changing pair carries a `meta_transform` (`z_project`/`crop`, discharging directive
  §7 for those two). Honest true-3D-vs-stack-of-2D: `tv_denoise` genuinely 3D, `wavelet_denoise`
  stack-of-2D (H15). `normalize` uses a `scope` Mode not the dim lever (H24). Backend API
  re-checked live in-env (scipy 1.17.1 / skimage 0.26.0). **Adversarially reviewed** (6-lens
  workflow → per-finding verify): 4 bugs found + fixed + regression-guarded (R1–R3 in
  `test_catalog_ported`): (1) `detect.spots` 3D used isotropic σ + never read `z_step_um` →
  now anisotropic σ tuple + axial sockets + memo-fenced; (2) `util.crop` `meta_transform`
  `span()` disagreed with the compute's `bound()` on one-sided/out-of-range/negative bounds →
  `span()` now mirrors `bound()` (fill-then-clamp) so header==payload; (3) `wavelet_denoise`
  emitted all-NaN on flat/sparse planes → flat guard + finite-fallback pass-through; (4) the
  `test_registry` fixture squatted on the real `detect.spots` op_key and clobbered the node in
  the global registry → fixture renamed to `test.registry_demo` (**lesson: fixtures must use
  fake op_keys**). A follow-up fix-verification workflow was blocked by the org monthly spend
  limit; the fixes were instead verified by the R1–R3 regression guards + direct edge-case probes.


- **Phase 1 (headless core) is BUILT and green** in `nodegraph/` (Qt-free; no PySide6/nd2 imports).
- The design is locked in **`CodeLog/ClaudesPlan/V2.00_nodegraph_blender_revamp.md`** (11 decisions).
- Data-I/O + node-transfer are researched in **`V2.01_data_io_research.md`** (56 routes, 2 passes,
  adversarially verified) and distilled into an implementation spec **`V2.02_phase2a_spec.md`**.
- The image-processing **backend menu** is in **`CodeLog/Architecture/node_backend_reference.md`**.
- **2026-07-21 user directive → `V2.03_directive_metadata_toggle.md`** (amendment addendum;
  V2.00 stays canonical): (A) metadata intelligence = the **per-edge Dataset package** (as
  transformed upstream), not static file metadata; (B) every 2D/3D-capable node gets a
  **top-right 2D/3D lever**. Verified by a 36-agent fan-out (26 holes; 8 blockers). Four
  interpretations LOCKED by the user: per-edge source (reading a); lever = distinguished
  in-body Mode (header-rendered); anisotropic params = paired floats + `um_axial`; build
  mechanisms-first. See V2.03 §0.
- **Phase 2a-prime (V2.03 §7) FIRST CODE DROP is BUILT and green.** New: `revision.py`,
  `graph.py`, `metadata.py`; amended `dataset.py` (revision+immutability+`with_metadata`+
  `CALIBRATION_KEYS`+`is_volumetric`+`reshaped_axes`), `reducers.py` (`PartialReducer`
  tree-reduce), `sockets.py` (VECTOR `dims`+`domain`), `registry.py` (`Granularity`,
  `meta_transform`, `available_in`, `active_sockets`, granularity/kernel-axes resolvers,
  `DimMode`). Selftest = 13 groups green; parity green.
- **Phase 2a COMPLETE (2026-07-21):** keystone benchmark run → TILED / planar z-blocks;
  `provider.py` (b2nd + synthetic), `memo.py` (two-hash), `engine.py` (lazy pull + ReadContext
  + Granularity routing), `structure.py` (CCL/watershed → columnar) all BUILT — **20 selftest
  groups green**, parity green. Deps: blosc2 4.9.1, zarr 3.2.1, pyarrow 23.0.1, scipy 1.17.1,
  skimage 0.26.0 (Python 3.13.1). Next: field realization, structure-bridge execution, node port.

## 1. The vision + the 11 locked decisions (condensed — full text in V2.00 §2)

One unified lazy **Dataset** flows on the main wire (image + label masks + points + tracks +
measurements as **named attribute layers over domains**) + a **value/field** socket layer.
Locked: (1) unified Dataset + value/field layer; (2) 10 domains; (3) one Dataset socket + field-able
Float/Int/Bool/Vector/Color/String/Menu; (4) value params → field-able sockets, modes → in-body
dropdowns; (5) implicit domain-transfer with the rule always visible; (6) values convert implicitly,
data-layer transforms are explicit nodes; (7) multi-input · link-drag search · mute-passthrough ·
reroute + frames; (8) node groups + Repeat & Simulation zones; (9) **lazy demand-driven PULL +
memoization** (see §4 for the restated memo invariant); (10) Viewer node + Spreadsheet; (11)
**greenfield**, no auto-migration of old `.nd2s_pipeline.json`.

## 2. The domain model — as IMPLEMENTED (V2.00 §3.2, amended)

Ten domains, two families:
- **Acquisition lattice** (coarsenings; transfers GENERATED as reduce/broadcast):
  `Voxel(m,t,z,c,y,x) → Plane(m,t,z) → Frame(m,t) → {Multipoint(m) | Timepoint(t)} → Global()`,
  plus **Channel(c)** — a lattice axis-domain **orthogonal** to the spatial/temporal chain.
  `meet` is total; `join` is **partial** (`join(Channel, Frame)` = unnamed → None). Coarsening
  Voxel→Frame/Plane reduces over `c` by default (channel-mean).
- **Detected structures** (defined by analysis, explicit bridges): **Label** (region of a label mask),
  **Point** (sub-pixel detections + optional vector), **Track** (temporal identity over Timepoint).
  Point↔Label is *constructive* (points→boundary→Fill→Label; Extract Boundary inverts).
  Label/Point/Track are multi-instance (keyed by source layer); lattice domains are single-instance.

> **The c-axis was added 2026-07-20** (user directive): `AXIS_ORDER=("m","t","z","c","y","x")`,
> Channel promoted to a lattice domain. Fixed the store/tile/memo addressing inconsistency.

## 3. What's built — Phase 1 (`nodegraph/`, all selftest groups green)

| File | Contents |
|---|---|
| `domains.py` | 10 `Domain`s, lattice axis-sets, `is_finer/comparable/join(partial)/meet(total)/dropped_axes` |
| `reducers.py` | mean/sum/max/min/median/count/first (NaN-aware) over array axes |
| `dataset.py` | `AxisSizes(m,t,z,c,y,x)`, `AttributeLayer`, `Dataset` (structural-sharing, shape validation) |
| `transfer.py` | **lattice transfer generator** `plan_transfer` (generated + executed on numpy) + structure bridge registry + BFS routing (bridge execution deferred to Phase 3) |
| `sockets.py` | `SocketType` + implicit value conversion + `can_connect` |
| `registry.py` | `NodeSpec` + `NODES` + `In*/Out*/Mode` factories (value sockets carry V1.91 `unit`/`derive`) |
| `selftest.py` | `python -m nodegraph.selftest` — 7 groups, all green |

Verify: `python -m nodegraph.selftest` and `python scripts/_pipeline_kit_parity.py --check` (old
system untouched, still green).

## 4. Settled Phase-2a decisions (V2.01 §H/§I/§J → V2.02)

- **Store = Blosc2 b2nd** on disk; ingest ND2 once; double-partition **block = tile = memo unit**
  (native sub-block ROI, no Zarr ingest). Tile = **512² ≡ block** (config; benchmark default).
- **Structure domains = Arrow columnar** RecordBatch (CSR connectivity as `ListArray`), cheap **full**
  content-hash; computed **whole-domain** (off the tile path), sliced per display tile.
- **Track/Label id stability** via id-carrying **seeded watershed** (not renumber-prone tile-local CCL).
- **Memo invariant RESTATED** (not "content-hash"): a cheap **proxy recipe-hash** for lookup
  (`file_id,offset,mtime,dtype,shape,(m,t,z,c),params` for leaves; `op+params+declared-reads+upstream
  recipe-hashes+upstream revisions` for nodes) **+** a content **output-fingerprint** for
  cutoff/dedup. Identity is a monotonic **`revision`**, never `id()`, never an 85 MB read.
- **`AttributeLayer` gains `revision` + SAFE immutability** (copy-then-freeze — avoids the §J
  aliasing footgun). **Reducers → associative monoids** (tree-reduce across tiles; `first` IS
  tileable; median falls back to whole-domain). **`ReadContext`** records calibration reads into the
  memo key. **Field memo** in a disjoint namespace; constant/broadcast → `VirtualArray`.
- **Viewer = QRhiWidget** (Metal/D3D/Vulkan/GL) with a runtime `maxTextureSize()` tiling guard
  (GL fallback caps at 2048–4096; a 6554² plane is NOT one texture on that path).

## 5. This survey's outcome (2026-07-21) — what changed / didn't

The supplied image-processing package survey is a **node-backend menu** (deconvolution, TV denoise,
multiscale enhancement, registration/stacking/fusion, sparse reconstruction), distilled into
`CodeLog/Architecture/node_backend_reference.md`. Verdict:
- **Phase 1: NO change.** The data model is backend-agnostic and already supports true 3D voxel +
  temporal + multi-channel + multipoint. Survey confirms it.
- **Phase 2a: ONE change — extended the `Granularity` taxonomy** (V2.02 §7b) from {TILEABLE,
  WHOLE_FRAME} to **TILEABLE / WHOLE_PLANE / WHOLE_VOLUME / WHOLE_SERIES / MULTI_VIEW**, because 3D
  deconvolution, temporal registration/stacking, and multi-view fusion need different whole-axis
  footprints. Also noted: **axis-changing nodes** (stack T→1; stitch M→1 + grow Y,X) return a new-
  geometry Dataset and store transforms as Frame-domain attributes.
- **Phase 2 node port: EXPANDED** (V2.00 §14.3) into 5 compute families + the true-3D-vs-stack flag +
  the **metadata-intelligent PSF** deconvolution node (PSF derived from NA/emission/pixel/z via
  Gibson–Lanni — the V1.91 unit/derive showcase).
- Caveat: survey versions/maintenance are as-of 2026-07-20 and un-re-verified — re-check any dep.

## 6. Remaining roadmap

- **Phase 2a-prime — directive mechanisms (V2.03 §7): FIRST CODE DROP DONE.** Built &
  green: `revision.py`, `graph.py`, `metadata.py` (MetaEnvelope pass + named
  meta_transforms + `propagate_meta` + `resolve_dim_default`); `dataset.py`
  (revision+immutability, `with_metadata`, `CALIBRATION_KEYS`, `is_volumetric`,
  `reshaped_axes`); `reducers.py` (`PartialReducer` tree-reduce); `sockets.py` (VECTOR
  `dims` widen-only + `domain` threaded); `registry.py` (`Granularity`, `meta_transform`,
  `available_in`, `active_sockets`, granularity/kernel-axes resolvers, `DimMode`).
  **Remaining Phase-2a-prime (§7 2a′.3):** wire toggle state + resolved granularity/
  kernel-axes + declared reads into `memo.py`/`engine.py` when those are built.
- **Phase 2a — lazy tiled provider: `provider.py` BUILT & green (2026-07-21).** Keystone
  benchmark resolved → TILED, planar `(1,512,512)` blocks. `nodegraph/provider.py`:
  `TileProvider` ABC (implement `read_region`; base derives `get_region`/`get_tile`/
  `get_subvolume`/`get_region_volume` per V2.03 §5 D1 + multiscale `level_axes`),
  `SyntheticProvider` (numpy-only, deterministic — always testable), `B2ndProvider`
  (blosc2 **lazily imported**; `from_array` ingests a 6-D `(M,T,Z,C,Y,X)` numpy volume →
  planar-block b2nd + mean pyramid). ND2→numpy ingest is an app-layer concern (nodegraph
  stays nd2-free). Selftest = 15 groups green (b2nd row runs on blosc2 4.9.1).
- **Phase 2a — memo + engine: `memo.py` + `engine.py` BUILT & green (2026-07-21).**
  `memo.py`: two-hash — structural `recipe_hash` lookup (op + params[incl. folded mode/dim
  state] + upstream recipe_hashes + upstream **revisions**) + content `output_fingerprint`
  (cutoff/dedup); blob dedup; per-node last-fp cutoff; `Entry` carries declared reads +
  `OutputHeader` (H4(5)); blake2b (stdlib, not blake3). `engine.py`: lazy recursive pull +
  `Memo` + `ReadContext` (records calibration reads; `calib` validates vs `CALIBRATION_KEYS`;
  reads re-validated on a hit — Salsa verify) + `Granularity` routing (2D→tile, 3D→subvolume)
  + `EvalContext`. Phase-2a-prime 2a′.3 (toggle/granularity/declared-reads into memo) = DONE
  here. Selftest = **18 groups green**: precise read-invalidation (unread key change ⇒ no
  recompute), downstream isolation, 2D/3D lever → distinct memo keys. Compute is supplied per
  op_key (the node port is the harness's client — Phase 2/3).
- **Phase 2a — structure.py: BUILT & green (2026-07-21). PHASE 2a COMPLETE.**
  `nodegraph/structure.py`: `StructureTable` (numpy columnar; invariant `id,m,t,c,z,y,x`
  schema, z never NaN — V2.03 §4 C3; `content_hash` cheap full hash; `to_arrow()` lazily
  imports pyarrow → RecordBatch w/ z_kind/channel_kind metadata), `connectivity_offsets`
  (2D 4/8, 3D 6/18/26), `label_components` (numpy flood-fill CCL, **raster-canonical ids**,
  2D→WHOLE_PLANE/z_kind=plane_index, 3D→WHOLE_VOLUME/subpixel-z; area=px²/px³),
  `seeded_watershed` (id-carrying — output label == marker id; lazy scipy EDT + skimage
  watershed, anisotropy `sampling`), `point_table` (invariant schema, 2D plane-index z vs
  3D subpixel). scipy/skimage/pyarrow all **lazily imported** (core = numpy-only). Selftest
  = **20 groups green** (pyarrow 23.0.1, scipy 1.17.1, skimage 0.26.0 all present, all
  branches exercised). Flood-fill CCL is the correctness baseline; a scipy/cc3d fast path
  is a later benchmark-gated optimization (V2.02 §7).
- **Phase 2 — structure-bridge EXECUTION: `bridges.py` BUILT & green (2026-07-21).** The
  geometric spine now executes (was NotImplementedError): `voxel_to_label` (mean-in-mask +
  sum/count/max/min/median via bincount grouping), `label_to_voxel` (paint-by-label, LUT),
  `voxel_to_point` (nearest numpy / linear lazy-scipy), `point_to_voxel` (splat sum/mean/max),
  `containing_label`, `points_in_label`. `BRIDGE_FUNCS` maps (src,dst)→fn. `transfer.py`'s
  generic `execute_transfer` still raises on a BridgeStep (it lacks the label-raster/point
  inputs) but now points to `bridges.py`. Selftest = **21 groups green** (round-trip paint↔
  reduce verified). **DEFERRED: Track bridges** (gather-by-track / broadcast-track) — need the
  Track membership table (member per timepoint), which the structure model grows next.
- **Phase 2 — field realization: `field.py` BUILT & green (2026-07-21).** Minimal field IR
  (`Const`/`Attr`/`Input`/`BinOp`/`UnaryOp`/`Where`; V2.00 §16 "start minimal"), `VirtualArray`
  (non-materializing const/broadcast, V2.02 §9), `evaluate` (Attr auto-transfers across the
  lattice by the default reducer — shown on the wire; structure-domain field transfer deferred),
  `field_expr_hash` (folds referenced layer **revisions**) + `field_key` = disjoint namespace
  `("field", expr_hash, domain, kernel_axes, token)` (V2.02 §9 / V2.03 §4 C2) + `FieldCache`.
  Selftest = **22 groups green**.
- **Adversarial code review (ultracode, 2026-07-21): 14 confirmed bugs, ALL FIXED + regression-guarded**
  (`test_review_regressions`; selftest now **23 groups green**). Notable: a **CRITICAL** `memo._canon`
  hash-collision (non-injective encoding → wrong memo hits) → rewrote injective; distinct seed/provider
  SOURCE nodes collided in the memo → engine folds source identity into the recipe hash; a `ctx.env.metadata`
  read-fence bypass → recording proxy (V2.02 §8); multi-input socket-order arg misrouting → canonical
  socket ordering; a read-only-view aliasing leak in `AttributeLayer`; `label_components` missing m,t,c
  columns; plus point_to_voxel/label_to_voxel/channel_select/get_subvolume edge cases. **Still DEFERRED**
  (needs provider identity API): a same-provider on-disk file change (mtime/id) invalidating the memo —
  `__provider_version__` hook is wired, fold it in via `leaf_recipe_hash` when providers carry mtime/file_id.
- **Track membership + Track bridges: BUILT & green (2026-07-21).** `structure.TrackMembership`
  (parallel `track_id`/`t`/`member_id` columns defining the Track domain — the temporal
  identity `t→member`; `member_domain` LABEL/POINT; `to_table`). `bridges.py`: `gather_by_track`
  (Label/Point→Track reduce), `broadcast_track` (Track→Label/Point), `tracks_per_timepoint`
  (Track→Timepoint count/reduce active tracks), `timepoint_to_members` (Timepoint→member×t).
  `_group_reduce` gained `drop_nonpositive` (timepoints may be 0). All in `BRIDGE_FUNCS`. The
  tracking *algorithm* (frame-to-frame linking) is a node/Simulation-zone concern — bridges
  consume a given membership. Selftest = **24 groups green**.
- **Fill/Extract Boundary: BUILT & green (2026-07-21).** `nodegraph/boundary.py` (the
  constructive Point↔Label spine, V2.03 §4 C4 — **explicit data-layer ops, NOT in
  `transfer._BRIDGES`**): `boundary_dim` (geometry authority: single-z ⇒ 2D contour,
  multi-z ⇒ 3D surface), `fill_boundary` (points → Label raster; 2D polygon / per-z-plane
  fill — the 2D-ROI-on-3D case fills only its plane, never extruded), `extract_boundary`
  (Label raster → boundary Points: 2D find_contours + contour_id / 3D marching_cubes).
  The 2D/3D toggle is a **consistency assertion** (hard error on contradiction, never an
  override). skimage lazily imported. Selftest = **25 groups green**.
- **Node port — foundation + flagship vertical slice: BUILT & green (2026-07-21).**
  `nodegraph/nodes.py` establishes the node-port CONVENTION: a node = a registered
  NodeSpec + a `compute(ctx)->Dataset` in the `COMPUTES` dict (engine looks it up by
  op_key); `register_node(compute, **spec)`; `to_pixels_v2` (adds the `um_axial` axial
  unit, V2.03 §2 A5); `diffraction_sigmas`/`gaussian_psf` = the **metadata-intelligent
  PSF derived from optics** (σ_xy≈0.21λ/NA, σ_z≈0.66λn/NA² → pixels). Two reference
  nodes: **`channel.select`** (H12: lazy `_ChannelView` + channel_select meta_transform)
  and **`enhance.deconvolve`** (the two-mode showcase: 2D lateral PSF / WHOLE_PLANE vs 3D
  anisotropic PSF / WHOLE_VOLUME via the DimMode lever; 3D-only `z_step_um` socket
  (`available_in`); Richardson–Lucy on **installed scikit-image** — RedLionfish GPU is a
  later swap). `provider.ArrayProvider` wraps a realized volume back into `Dataset.image`.
  Selftest = **26 groups green** (end-to-end through the Engine: metadata-derived PSF,
  Select Channel, 2D/3D lever → distinct memo keys + resolved granularity/kernel-axes +
  3D-only socket). PATTERN is set for the rest of the catalog.
- **Catalog batch + structure-storage convention: BUILT & green (2026-07-21).** Added to
  `nodes.py`: **Gamma** (pointwise/no-lever, TILEABLE), **Gaussian Blur** (toggle; metadata-
  intelligent σ µm→px, scipy.ndimage; 2D per-plane vs 3D anisotropic), **Threshold** (→ Voxel
  mask; fixed/Otsu Mode), **Connected Components** (`analysis.label`: mask → per-plane/volume
  CCL, connectivity per dim, global-unique ids), **Measure** (`analysis.measure`: Voxel→Label
  mean-in-mask via bridges). **Structure-storage convention RESOLVED**: `Dataset.with_structure(table)`
  stores each StructureTable column as a per-element `AttributeLayer` on its domain keyed by
  source layer (V2.00 §3.2). Full **threshold→label→measure pipeline** runs end-to-end.
  **Two MORE memo bugs found+fixed by the pipeline test** (review missed — need Dataset+provider
  payloads): (a) `output_fingerprint(Dataset)` ignored the **image provider** → datasets differing
  only by image collided + blob-dedup returned the wrong image; fixed via `provider.fingerprint()`
  (ArrayProvider=content hash, base=structural, ChannelView=base+channels) folded into the Dataset
  fingerprint; (b) `sorted(payload.attributes)` compared Domain enums (unorderable across mixed
  domains) → sort by `(domain.value, layer, name)`. Selftest = **27 groups green**; parity green.
- **Phase 5 GUI STARTED (2026-07-21).** Decisions locked (user): mockup-first → then Qt; first
  slice = **node card + minimal scene**; new **`nodelab_v2/`** package (greenfield; reuse NodeLab's
  theme.py/QGraphicsView patterns; v1 NodeLab stays runnable until Phase 7 parity). PySide6 (+PyQt6)
  available; V2.00 §14.5 baseline = QGraphicsView/Scene, QRhi viewer separate/later. **Interactive
  HTML mockup published** (Artifact) — driven by the REAL registry dump (sockets/levers/available_in/
  granularity): sockets-on-card, top-right 2D/3D header lever with LIVE socket relayout (sigma_z /
  z_step appear in 3D) + granularity chip change, field diamonds ◇ vs value circles ●, metadata-
  derived defaults (ƒmd badge), bezier Dataset wires, draggable cards, light/dark. Honors NodeLab
  accent #6FE3FF + the sockets.py SOCKET_COLOR palette. Mockup file:
  scratchpad/nodelab_v2_canvas.html. **Awaiting user design feedback before Qt implementation.**
- **Phase 5 GUI — Qt FIRST SLICE BUILT & verified (2026-07-21).** New **`nodelab_v2/`** package (9
  modules, PySide6; separate from v1 `nodelab`), bound to the real `nodegraph` registry:
  `theme.py` (palette: NodeLab cyan accent + amber `--dim2d` for 2D; socket colors reused from
  `sockets.py`), `node_item.py` (`SocketItem` field-diamond/value-circle/Dataset-dot; `SwitchItem`
  two-color 2D/3D on-off switch — amber 2D / cyan 3D; `NodeItem` renders a NodeSpec card, header
  switch → `active_sockets` relayout + granularity chip, value pills w/ ƒmd badges), `edge_item.py`
  (bezier Dataset wires), `scene.py` (`GraphScene` reroute + `GraphView` dotted grid + wheel-zoom),
  `inspector.py` (`InspectorPanel` QDockWidget: 2D/3D `SwitchWidget`, footprint chip, editable param
  rows — QSpinBox/QComboBox + unit + **auto/pinned lock** = sticky `__locked__`, connections;
  edits write node.params/locked → live relayout), `window.py` (MainWindow + demo pipeline
  Load→Select→Gaussian→Threshold→Label→Measure + Deconvolve), `app.py`/`__main__.py`. Run:
  `python -m nodelab_v2`. **Verified offscreen** via `scripts/_nodelab_v2_shot.py` (QT_QPA_PLATFORM=
  offscreen → PNG; note: offscreen loads NO system fonts, so the script registers Windows TTFs for the
  shot only — the on-display app has them; and it `os._exit(0)` to skip a harmless Qt teardown crash).
  Faithfully matches the approved mockup. Design decisions from the mockup are LOCKED (user): sockets-
  on-card, the amber-2D/cyan-3D sliding switch, click-to-edit inspector with lock/derive semantics.
- **NEXT (GUI):** (1) interactive **connect/disconnect wires** (drag from socket, link-drag search
  menu, splice-on-wire, cycle/type validation via `can_connect`); (2) **palette panel** (registry-
  driven, drag to add); (3) mute/collapse, reroute, frames; (4) **Viewer node** (QRhi/QGraphics image
  + label/point overlays via the engine's pull) + **Spreadsheet**; (5) save/load `*.nd2graph.json`
  (V2.00 §12); (6) wire the canvas to the `Engine` to actually run. **NEXT (core):** port more catalog
  nodes; nd2→6D ingest; per-tile lazy field eval; Phase 4 zones/groups.
- **Phase 2 — eval engine**: lazy pull scheduler + two-hash memo + field evaluator + domain-transfer
  execution (currently generated for lattice; structure bridges still deferred — implement here).
- **Phase 2 — node port** (V2.00 §14.3): the 5 backend families above, each declaring `Granularity`.
- **Phase 4** — Repeat + Simulation zones; node-group encapsulation.
- **Phase 5** — GUI: sockets-on-card canvas (inline widgets, diamonds/circles, multi-input, reroute,
  frames, link-drag search, mute), Viewer node (QRhi), Spreadsheet.
- **Phase 6** — inspection & export wired to the pull engine.
- **Phase 7** — retire `pipeline_kit` from NodeLab once v2 reaches capability parity.
- **Skills**: `wire-node` / `build-node` describe v1 — rewrite (or add `-v2`) once §1–§2 land.

## 7. Open decisions / blockers (settle before/while coding)

1. ~~**Run the keystone benchmark**~~ — **DONE 2026-07-21 → TILED confirmed.** Real 6554² ND2
   plane: b2nd 512² block ROI = 1.5–4.8% of whole-plane (block=512 balanced: 4.0%, cratio 2.0×).
   **3D (H26): planar `(1,512,512)` z-blocks win** (4z subvol 11% at bz=1 vs 47% at bz=32; plane 3%
   vs 44%) — block = 2D tile per z, no z-spanning blocks; `get_subvolume` gathers planar blocks.
   blosc2 4.9.1 / zarr 3.2.1 installed. (Item 6 below folded in.)
2. **asyncio↔Qt bridge** for the pull scheduler (qasync vs hand-rolled QThreadPool + epoch registry) —
   unchosen; the cooperative cancel-token layer does the real cancellation, so the asyncio shell may
   be optional.
3. **Channel-mean-by-default**: ~~confirm whether per-channel spatial stats are the common case~~ —
   **elevated by V2.03 §2 A6 (H12) to a correctness prerequisite**: a **Select Channel** node +
   `ctx.channel` per-channel derive resolution is a blocker for the deconvolve PSF. Optics never
   auto-coarsen (`join(Channel,Frame)=None` already blocks it).
4. **TensorStore vs {b2nd + hand-rolled scheduler}** for the async/coarse-view layer — pick one store;
   **criteria now include efficient 3D z-range subvolume + whole-volume reads (V2.03 §5 D2 / H26)**.
5. **Fusion reducers**: stacking wants robust/sigma-clipped/weighted mean — add to the reducer roadmap.
6. **Keystone benchmark gains a 3D section (V2.03 §5 D2):** z-range subvolume vs whole-volume vs
   single-plane reads on a `(Z,512,512)` b2nd stack — add before choosing the store's z-block shape.
7. **Interpretation forks are RESOLVED (V2.03 §0):** per-edge source; lever = header-rendered Mode;
   paired floats + `um_axial`; mechanisms-first. No longer open.

## 8. Key files, commands, pointers

- Plan: `CodeLog/ClaudesPlan/V2.00_…` (locked design), `V2.01_data_io_research.md` (routes+resolutions),
  `V2.02_phase2a_spec.md` (implementation spec). Backend menu: `CodeLog/Architecture/node_backend_reference.md`.
- Code: `nodegraph/` (Phase-1 core). Benchmark: `scripts/_bench_provider_granularity.py`.
- Verify: `python -m nodegraph.selftest`; `python scripts/_pipeline_kit_parity.py --check`.
- Skills: `.claude/skills/{wire-node,build-node,grilling,grill-me}` (wire/build describe v1).
- Metadata-intelligence (V1.91, reused on v2 value sockets): `nd2studios/pipeline_kit/metadata_adapt.py`
  (`unit`/`derive`, `to_pixels`, `adapt_defaults`, sticky `__locked__`).

## 9. How the user likes to work (norms observed)

- **Grill before big design.** For architecture decisions, run the `grilling` method (one question at
  a time, each with your recommendation) and reach shared understanding before writing.
- **Rigor + adversarial verification.** For research, fan out (workflows when opted in / ultracode),
  then adversarially verify claims; mark unverified. Present routes with pros/cons.
- **Present before locking.** Big changes are proposed for review, not silently baked into the locked
  plan; keep V2.00 the canonical locked doc and add pointers.
- **Honor real constraints.** Qt-free backend; don't modify the environment (pip installs) unprompted;
  metadata-intelligent params (unit/derive) on everything spatial/temporal.
