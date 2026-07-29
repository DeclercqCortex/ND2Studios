# CONTEXT — nodegraph v2: everything that REMAINS (handoff)

> **⚠️ UPDATE 2026-07-29 — v1 IS REMOVED.** The `pipeline_kit` parity gate in §0 is gone
> (and with it the warning not to delete `nd2studios/` — that deletion has now happened
> deliberately, to the Recycle Bin). §5's "full v1 removal is deferred" is **resolved: it
> was done**, with the remaining gaps declared obsolete rather than ported. The v1
> `wire-node`/`build-node` skills referenced in §5 were deleted; `*-v2` remain. §8's pointer
> to vendored `nd2studios/pipeline_kit/metadata_adapt.py` is dead. Record:
> **`V2.05_phase7_capability_matrix.md` §6**.

> Self-contained forward-looking brief for a fresh chat. **State as of 2026-07-21.**
> For *what's already built and why*, see the running log in
> [`CONTEXT_nodegraph_v2_handoff.md`](CONTEXT_nodegraph_v2_handoff.md); this doc is the
> **remaining-work** map. Locked design lives in [`V2.00`](V2.00_nodegraph_blender_revamp.md)
> (11 decisions), amended by [`V2.03`](V2.03_directive_metadata_toggle.md) (the 2026-07-21
> per-edge-metadata + 2D/3D-toggle directive, 26 verified holes). Phase-2a spec:
> [`V2.02`](V2.02_phase2a_spec.md). Backend menu:
> [`../Architecture/node_backend_reference.md`](../Architecture/node_backend_reference.md).

## 0. Where we are (context) + how to verify

The **headless core is complete and hardened**; a **first GUI slice** exists. Two things
work end-to-end today: a lazy-pull, two-hash-memoized engine, and a metadata-intelligent,
2D/3D-toggle-bearing node catalog (small but representative). Everything is Qt-free except
`nodelab_v2/`.

**Gates (must stay green):**
- `python -m nodegraph.selftest` → **38 groups green** (the headless core; +`catalog (ported)`,
  `catalog (batch 2)`, `catalog (batch 3)`, `fusion reducers`, `transfer C2`, `engine hardening`,
  `zones`, `zones per-frame-T`, `groups`, `tracking`, `serialize` as of 2026-07-21/22).
  Disk-B2ndProvider round-trip folded into `provider`. Catalog = **39 node types**.
  Run with `PYTHONUTF8=1` on Windows — a bare cp1252 console chokes on the `↔`/`σ` glyphs in some
  `[ok]` lines (cosmetic, not a failure). **Test fixtures must use a clearly-fake op_key (e.g.
  `test.*`) — a fixture that `define_node`s a *real* op_key clobbers that node in the global `NODES`
  registry (bit us once: a `detect.spots` fixture vs the real node).**
- `python scripts/_pipeline_kit_parity.py --check` → **no drift** (v1 backend untouched;
  requires the vendored `nd2studios/` at repo root — it was once deleted, restored from the
  Recycle Bin; don't delete it, Phase 7 retires it).
- `python scripts/_nodelab_v2_shot.py out.png` → offscreen GUI render (see §9 gotchas).
- `python scripts/_bench_provider_granularity.py` → keystone benchmark (needs blosc2+zarr).

**Env:** Python 3.13.1; numpy 2.4.2, blosc2 4.9.1, zarr 3.2.1, pyarrow 23.0.1, scipy 1.17.1,
scikit-image 0.26.0, PySide6 (+PyQt6). Never `pip install` unprompted.

**Built (for orientation only):** `nodegraph/` (18 modules) = domains, reducers (+PartialReducer
tree-reduce), revision, dataset (+with_metadata/CALIBRATION_KEYS/reshaped_axes/with_structure/
is_volumetric + AttributeLayer revision & safe immutability), transfer (lattice generated +
bridge routing), sockets (+VECTOR dims), registry (+Granularity, meta_transform, available_in,
active_sockets, DimMode, resolvers), graph, metadata (MetaEnvelope pass), provider (TileProvider,
Synthetic, **B2nd**, ArrayProvider), memo (two-hash), engine (lazy pull + ReadContext + granularity
routing), structure (CCL / seeded-watershed / columnar tables / TrackMembership), bridges (Voxel↔
Label/Point, Point↔Label, Track ×4), field (field IR + VirtualArray + disjoint memo), boundary
(Fill/Extract), nodes (Select Channel, Deconvolve 2-mode PSF, Gamma, Gaussian, Threshold, Label,
Measure). `nodelab_v2/` (9 modules) = the Qt canvas + inspector first slice. Keystone benchmark
run → **TILED, planar (1,512,512) z-blocks**. An adversarial review fixed 14 bugs (all regression-
guarded).

---

## 1. Remaining — core / engine

> **2026-07-21 (batch 2): C2, C5, C7 DONE** (selftest groups `transfer C2`,
> `engine hardening`). C5 = `TileProvider.version` (defaults to `fingerprint()`;
> disk providers override to fold `mtime_ns`) folded into the source key via the
> existing `__provider_version__` hook. C7 = `Engine(strict_reads=True)` +
> `_StrictCalibMetadata` (a dict subclass — fast-path `{**}`/`items()` unaffected)
> makes an un-contexted `Dataset.metadata` calibration read a hard error. C2 =
> `transfer.execute_bridge_plan(carrier, plan, *, label_raster, points, membership,
> shape)` chains a routed plan per frame/volume (Voxel→Track, Voxel→Point→Label,
> →Frame); Track↔Timepoint (member×t) + Frame→structure broadcast still raise (need
> the extra representation / target id-set) — call the `bridges` fns directly there.

| # | Item | Where | Notes |
|---|---|---|---|
| ~~C1~~ | ~~**Per-tile lazy field/compute eval**~~ ✓ | `nodegraph/streaming.py` (+engine/field/nodes/zones) | DONE 2026-07-22 — design **[`V2.04`](V2.04_c1_streaming_eval.md)** locked by the user (A1 provider-chaining / B1 overlap-recompute / C1 two-level cache / D1 windowed-minimal fields), then implemented same day. `StreamProvider` family (`MapComputeProvider` tile+halo & plane units + cum-halo fence, `VolumeComputeProvider` per-z slabs, `ZReduceProvider` PartialReducer tree-reduce, `WindowView` crop), engine-owned byte-budget `TileCache` + `FieldCache` (flat fp digests folding params+declared reads+field hashes+base fp → no stale tiles on reseed/layer change), windowed `FieldContext`, `ctx.input(name)`, `ReadContext.freeze`, `recursion_headroom`, `realize()` (used by `assert_zone_pure`). Six misdeclared-TILEABLE traps re-declared (gamma/tv/wavelet/clahe/bilateral/nlm). Selftest group `streaming eval` (39 total). **Remaining slivers (V2.04):** `util.stack` tree-reduce; resample/normalize/drift lazy units; kernel-param field gate (when kernel params gain field consumers); ~~Memo GC~~ **✅ DONE 2026-07-26** (byte-budget LRU `Memo(budget_bytes=…)`, per-blob refcount, correctness-safe eviction, `_last_fp` kept; GUI caps at 1 GiB via `Engine(memo_bytes=…)`; selftest `memo GC`; V2.04 §6b addendum); computed-provider pyramids (Viewer G4). |
| ~~C2~~ | ~~**Indirect multi-hop bridge execution**~~ ✓ | `transfer.py` | DONE 2026-07-21 — `execute_bridge_plan` (geometric spine). |
| ~~C3~~ | ~~**Tracking node (produces a TrackMembership)**~~ ✓ | `nodegraph/tracking.py` + `nodes.py` | DONE 2026-07-22 (subagent-built). `link_labels` (max-IoU overlap) + `link_points` (nearest-neighbour) → a `TrackMembership` (deterministic union-find, first-appearance track ids); the `track.link` node (Mode label/point, `max_distance` µm) attaches it via `with_structure`, consumable by the 4 Track bridges. Selftest `tracking`. **Refinement:** run it *inside* a Simulation zone for streaming cross-frame state (currently links the whole realized T stack per (m,c)); the per-frame-T Sim slicing (§3) is the hook. |
| ~~C4~~ | ~~**nd2 → 6-D numpy ingest + on-disk b2nd store**~~ ✓ | `nodelab_v2/ingest.py` + `nodegraph/provider.py` | DONE 2026-07-22. `nodelab_v2.ingest.ingest_nd2(path, store_path)` → `(B2ndProvider, MetaEnvelope)`: reads pixels via `nd2.to_dask()` → canonical `(M,T,Z,C,Y,X)`, reuses the proven v1 `read_nd2_metadata_extended` for calibration (filtered to `CALIBRATION_KEYS`). `B2ndProvider.write`/`.open` persist/reopen a disk planar-block store; disk `version`/`fingerprint` fold in `mtime_ns` (cheap; closes the C5 disk hook). Verified end-to-end on the real 6554² NileBlue sample (`scripts/_ingest_nd2_smoke.py`): pixels match raw nd2, engine pull Load→SelectChannel→Gamma works. **Remaining sliver:** a real **Load ND2 graph/source node** (the GUI-facing node that wraps `ingest_nd2` as a source in the graph — currently a demo stub in `nodelab_v2/window.py`). |
| ~~C5~~ | ~~**Provider identity for the memo**~~ ✓ | `provider.py`, `engine.py` | DONE 2026-07-21 — `TileProvider.version`. |
| ~~C6~~ | ~~**Whole-frame CCL/watershed cost benchmark**~~ ✓ | `scripts/_bench_ccl_watershed.py` | DONE 2026-07-22. **Finding:** the pure-numpy flood-fill `label_components` is ~1200× slower than `scipy.ndimage.label` at 512² and extrapolates to **~198 s at 6554²** → whole-frame CCL **needs a scipy/cc3d fast path** (drop-in `scipy.ndimage.label`); `seeded_watershed` (scipy-backed) is already fast (~0.02 s/512²). **Follow-up ✓ DONE 2026-07-24:** `structure.label_components` now uses `scipy.ndimage.label` (structure = `generate_binary_structure(ndim, rank)` from the shared `_connectivity_rank`) + a **raster-canonical relabel** (`_canonical_relabel`: 1..K in first-appearance C-order) + `bincount` areas + `center_of_mass` centroids. **Byte-identical** to the pure-numpy flood-fill, which is kept as `_label_components_flood` (the reference + scipy-absent fallback); the selftest `structure` group asserts scipy≡flood across 2D/3D connectivities. ~40× faster at 512² (bench: 1.17 s → 0.028 s); the 191 s cliff at 6554² is gone. |
| ~~C7~~ | ~~**ReadContext hard-error guard**~~ ✓ | `engine.py` | DONE 2026-07-21 — `strict_reads` + `_StrictCalibMetadata`. |
| ~~C8~~ | ~~**Per-channel derive resolution (`ctx.channel`) for c>1 (H12)**~~ ✓ | `engine.py`, `nodes.py` | **DONE 2026-07-24.** `EvalContext.channel(c)` → `ChannelContext`: `.param(name)` returns a user override (one value, all channels) else the socket `derive` re-evaluated with **channel c's** optics (`envelope_symbols(ctx.env, c)`, auto memo-fenced via the recording wrapper) else the socket default; `.emission_nm()` = that channel's λ. `detect.spots` now resolves min/max radius **per channel** (was one value for all c); `enhance.deconvolve` resolves emission/NA per channel via `ctx.channel` (also now honors an `emission_nm`/`na` override — previously ignored). Dead `_channel_emission` removed. Also makes `derive` work **headless** for these nodes (not only as a GUI seed). Selftest group `channel derive (C8/H12)`; concepts in `wire-node-v2` §7. |

---

## 2. Remaining — node catalog port (the bulk of Phase 3)

Pattern is proven in `nodes.py` (`register_node(compute, **spec)`, `to_pixels_v2`,
metadata-intelligent PSF, `Dataset.with_structure`, DimMode + `available_in` variant sockets +
per-dim `granularity`/`kernel_axes`). **Port the rest**, each declaring its footprint per dim and
its metadata-intelligent params, with a **30-second dependency re-check** against the 2026-07-20
survey (`node_backend_reference.md`) before adopting any backend.

> **DONE 2026-07-21 — TWO batches ported (24 nodes; catalog now 28 node types;
> `nodegraph.selftest` = 32 groups).** All declare per-dim `granularity`/`kernel_axes`
> + metadata-intelligent `unit`/`derive` params; axis-changing ones carry a
> `meta_transform`; true-3D vs stack-of-2D is declared honestly (H15). Both batches
> adversarially reviewed (fan-out lens workflows → per-finding verify) with all
> confirmed bugs fixed + regression-guarded. Backend API re-checked live in-env
> (scipy 1.17.1 / skimage 0.26.0). **Shipped:** `enhance.{median, morphology,
> tophat, dog, unsharp, tv_denoise, wavelet_denoise, clahe, normalize,
> morphological_gradient, bilateral, nlm}`, `detect.spots` (LoG/DoG × bright/dark,
> anisotropic 3D), `analysis.{threshold, label, measure (multi-stat), edt, watershed,
> extract_boundary}`, `util.{zproject, crop, resample, stack}`, `align.drift`,
> `channel.select`, `enhance.{gamma, gaussian}`, `enhance.deconvolve`. Fusion
> reducers (`sigma_clip` MAD-robust, `trimmed_mean`) added to `reducers.py`.

- **Enhancement filters:** ~~Median~~ ✓, ~~Difference-of-Gaussians~~ ✓, ~~Top-Hat~~ ✓,
  ~~Morphological Gradient~~ ✓, ~~Unsharp Mask~~ ✓, ~~Morphology~~ ✓, Local Contrast,
  ~~Background Subtract (≈ tophat white)~~ ✓, ~~CLAHE~~ ✓, ~~Normalize (scope Mode, H24)~~ ✓.
- **Regularized denoise:** ~~TV (true-3D)~~ ✓, ~~wavelet (stack-of-2D, H15)~~ ✓,
  ~~NLM (true-3D)~~ ✓, ~~bilateral (stack-of-2D, H15)~~ ✓.
- **Multiscale enhancement:** wavelet / starlet à-trous (PySAP — **not installed**) / curvelet. *Deferred.*
- **Registration · stacking · fusion (axis-changing!):** ~~drift align (`align.drift`, phase
  cross-correlation, stores the shift as Frame attrs)~~ ✓, ~~SNR stacking `util.stack` (T→1, robust
  fusion reducers)~~ ✓; deformable align (ANTs/elastix — dep-gated), tile stitching / multi-view
  fusion (M→1, `stitch` meta_transform exists — **still open**, needs `MULTI_VIEW` + a real
  registration backend). ~~**Fusion reducers**~~ ✓ (`reducers.py`). Weighted-mean fusion (needs a
  weights input, doesn't fit the `(array,axis)` reducer signature) — *deferred*.
- **Sparse reconstruction:** BART / ODL / SPORCO (GPU/CUDA — **BART not installed**). *Deferred.*
- **DL restoration:** CSBDeep/CARE, StarDist (import OK but need trained model weights + TF);
  Cellpose (**not installed**). *Deferred — dep-gated + model-gated.*
- **Detection / segmentation / structure:** ~~Spot detection (LoG + DoG, bright/dark)~~ ✓,
  ~~EDT node~~ ✓, ~~Watershed node~~ ✓, ~~histogram threshold methods (`analysis.threshold`
  +otsu/li/yen/triangle/mean)~~ ✓, ~~multi-class threshold (`analysis.multiotsu`)~~ ✓,
  ~~adaptive/local threshold (`analysis.threshold_local`)~~ ✓; Bead detect (≈ a `detect.spots`
  preset), Nuclei (Cellpose, dep-gated), StarDist (dep/model-gated).
- **Structure/measure:** ~~Compute Measurements (multi-stat `analysis.measure`)~~ ✓,
  ~~Extract Boundary node~~ ✓, ~~Transfer Domain (`transform.transfer_domain`, lattice
  reduce/broadcast)~~ ✓ (structure-bridge transfer as a node still open — wrap
  `execute_bridge_plan`); **Fill Boundary** node (inverse — needs multi-contour Point-set
  reconstruction + geometry-authority; **still open**), Store/Read Attribute, Attribute→Field.
  *Remaining.*
- **Axis-changing utility nodes:** ~~Z-Project~~ ✓, ~~Rescale/Resample~~ ✓, ~~Crop~~ ✓,
  ~~Stack~~ ✓, Channel Split (≈ `channel.select`) / Channel Merge (needs multi-input) — **still open**.
- **Source/logic:** a real **Load ND2** node (currently only a GUI-demo stub registered in
  `nodelab_v2/window.py`), If/Else, Summary/Export.
- **Flagship polish:** Deconvolve's PSF is a Gaussian approximation — a real **Gibson–Lanni** (and a
  GPU RedLionfish backend) is a swap behind the same derived sampling.

**Node-port verify gate** (adapt `build-node` for v2): every registered `NodeSpec` declares
`granularity`/`kernel_axes`; a dim-lever node passes the eligibility test (H24); 2D vs 3D produce
distinct recipe hashes; end-to-end pull through the `Engine` in `selftest`.

---

## 3. Remaining — Phase 4: zones & groups

> **Phase 4a — the ZONE MODEL LANDED (2026-07-22)** in `nodegraph/zones.py` (+ `graph.py`
> `Edge.kind` back-edges + 4 `zone.*` boundary pass-through nodes in `nodes.py`).
> **Design locked by the user** (see §6 / running log §0): *unroll + revision-fold* +
> *debug-verify + impure escape hatch* — resolves the V2.00 §16 open risk.
> `zones.unroll(graph, zones, epoch)` expands a zone into a flat per-iteration chain
> (back-edge → forward `Out@(i-1)→In@i`); the stock Engine + memo run unchanged and
> incremental invalidation falls out of the existing upstream-revision fold. Selftest
> group `zones` (33 total) covers Repeat (N× body), Simulation (feedback accumulate),
> revision-fold caching, the impure epoch escape hatch, and `assert_zone_pure`.

- ~~**Repeat zone**~~ ✓ (paired `Repeat In`/`Repeat Out`, N iterations via the feedback
  chain). **Still open:** N as a *field-able value socket* (currently a static
  `Zone.iterations` int).
- ~~**Simulation zone**~~ ✓ (paired `Sim In`/`Sim Out`, feedback across iterations;
  determinism × memo **resolved**). ~~**per-frame-T specialization**~~ ✓ **DONE 2026-07-22**:
  a `zone.frame` node (op_key `zone.frame`, `FRAME_OP`) in a zone body is stamped
  `__frame__ = <iteration index>` by `unroll`; its compute slices frame t (lazy `_FrameView`
  for the image + keepdim slice of any t-bearing lattice layer) and declares a `frame_slice`
  meta_transform (T→1). So iteration t processes frame t while the Sim feedback carries state
  (Blender split). **No `Zone` schema change** → orthogonal to serialization. Selftest
  `zones per-frame-T` (scan accumulates 1+2+3→6). Caller sets `iterations = T`. **Still open:**
  field-driven N; slicing structure (Point/Track) layers by frame.
- ~~**Node groups**~~ ✓ **DONE 2026-07-22** (subagent-built) — `nodegraph/groups.py`:
  `Group` + `expand(graph, groups)` inline transform (mirror of `zones.unroll`), **nestable**
  (recursive to a fixed point, self-reference rejected), `group.input`/`group.output`
  boundary pass-throughs (in `nodes.py`). Selftest `groups` (simple + nested + error paths +
  end-to-end pull). **Refinements (Phase 4b+):** a single DATASET interface only (multi-socket
  group interface); the instance node's params/modes are not propagated into the body
  (parameterized group instances); a reusable group *library* + serialization.
- **Nested / overlapping zones + cross-zone edges** — `unroll` explicitly rejects these
  today (Phase 4b).
- **Zone × 2D/3D toggle (V2.03 §2 A7)** — toggle default freezes to the zone-input metadata
  (iteration 0); back-edge excluded from the MetaEnvelope pass (already excluded by
  `propagate_meta`, which walks forward edges). **Still open** (validator warning).
- ~~**Serialization** of the graph + zones/groups~~ ✓ **DONE 2026-07-22** (subagent-built) —
  `nodegraph/serialize.py`: `to_dict`/`from_dict` + `to_json`/`from_json`, `FORMAT_VERSION="2.0"`.
  Round-trips nodes (id/op_key/params/modes incl. the `__locked__` sticky set), edges (incl.
  `kind="back"`), Zones, and Groups (nested `body` graph recursed); deterministic output;
  version-validated on load. Selftest `serialize`. This is the headless core of
  `*.nd2graph.json` (the GUI save/load, §4 G6, wraps it + adds node canvas position / mute /
  collapse / frames — the documented extension point in `serialize._node_to_dict`).

---

## 4. Remaining — Phase 5: GUI (`nodelab_v2/`)

**Built:** canvas (cards, amber-2D/cyan-3D sliding switch, field/value sockets, granularity chip,
bezier wires, dotted grid, zoom, draggable nodes, selection) + the click-to-edit **Properties
inspector** (switch, footprint, editable params, the **auto/pinned** lock = sticky `__locked__`,
connections). Run `python -m nodelab_v2`.

> **PHASE 5 CORE SLICE LANDED (2026-07-22).** G1/G2/G3(mute)/G4(baseline)/G6/G7/G8/G10
> all done; verified by `scripts/_nodelab_v2_phase5_probe.py` (offscreen, incl. a REAL
> off-thread engine pull → viewer pixels, 1.3s first / ~0.26s memoized re-pull). New
> Qt-free **`nodelab_v2/document.py`** (`GraphDocument` — the editing model + wiring
> rules + envelope propagation + save/load) and **`runner.py`** (`EngineRunner` — the
> **G7 QThreadPool + epoch registry** bridge, user-locked over qasync; persistent Memo
> across engine rebuilds; `io.load` source resolution w/ a `SyntheticProvider` fallback
> so it runs out of the box). `scene.py`/`node_item.py`/`edge_item.py`/`inspector.py`
> rewritten to mirror the document; new `palette.py`, `viewer.py`. Core gates stay green.

**Status by goal:**
- ~~**G1 — Interactive wiring**~~ ✓ drag-connect with live green/red validity (via
  `document.can_connect` = `sockets.can_connect` + cycle rejection + non-multi replace),
  **detach-redrag** (press a wired input → re-drag from its source), **link-drag search**
  popup (drop on empty canvas → type-filtered ops → place + auto-connect), **splice-on-wire**
  (drop a node from the palette onto a link → `scene.splice_onto` rewires src→node→dst,
  transactional: validates both new connects before removing the original, so a value/field
  wire is never dropped), delete key. **Deleting is now discoverable (2026-07-28)**: the
  key was the *only* path and needed canvas focus, so a user who had just touched the
  Viewer or the inspector saw nothing happen. Added (all landing in
  `GraphScene.delete_selection`/`delete_nodes`): a **hover ✕ badge** on each card
  (`node_item.CloseItem`), a **right-click context menu** (node/wire/frame/empty variants,
  incl. mute/collapse/pull), and an **Edit menu** (Delete + *Dissolve* Ctrl+X + Select all)
  whose shortcuts are `WidgetWithChildrenShortcut`-scoped to the view so Del still edits
  text in the inspector. **Dissolve** = delete a mid-chain node and reconnect its input to
  everything it fed (a plain delete leaves the chain broken).
  *Remaining:* explicit multi-input ordering UI.
- ~~**G2 — Palette panel**~~ ✓ `palette.py` — registry-driven, searchable, grouped by
  category, double-click-to-add + drag-onto-canvas (mime `application/x-nd2studios-op`).
- ~~**G3 — Mute + collapse**~~ ✓ mute-with-passthrough (M key; `document._bypass_muted`
  rewires around a muted node in the RUN graph) + **collapse** (C key; compact header-only
  card keeping all sockets at the edges so wires stay valid; serialized in the `ui` extras).
  ~~reroute dots~~ ✓ + ~~labelled frames~~ ✓ **DONE 2026-07-26**: reroute = hidden
  `rr.reroute` identity pass-through node (compact 22 px dot via a `NodeItem` branch,
  double-click-a-wire to splice in, refuses value wires); frames = GUI-only `FrameItem`
  (`frame_item.py`) grouping nodes behind a titled tinted rect (`document.FrameRecord`,
  Graph→Frame selection / Ctrl+J, `ui`-serialized, auto-reflows, drag moves members,
  emptied⇒removed). ~~Group creation in the GUI~~ ✓ **DONE 2026-07-26** (`document.make_group`
  collapses a linear sub-chain → a `group:<name>` instance + stored `Group` def;
  `to_graph(materialize)` expands via `groups.expand`; `ungroup` reverses; propagate patches
  the instance env from the body output so downstream domain rails read through; Ctrl+G /
  Ctrl+Shift+G; instance renders as a purple GROUP card).
- ~~**G4 — Viewer (baseline + Point overlay)**~~ ✓ `viewer.py` — Voxel-plane heatmap w/
  percentile auto-contrast + (m,t,z,c) spinners (plane read in the worker via
  `runner.render_plane`); **Point-domain overlay** draws markers for points on the
  viewed (m,t,z,c) plane (full-res→shown scale from the plane vs axes), toggleable.
  ~~Label region-outline~~ ✓ + ~~Track-trajectory overlays~~ ✓ **DONE 2026-07-26** (a
  "Tracks" checkbox; per-track golden-angle-hued polylines join `(track_id,t,member_id)`
  to member `(y,x)` by id, ordered by t, current-T vertex enlarged; `viewer._tracks_here`,
  phase5-probe covered). *Remaining:* the QRhi GPU path only if 60fps needs it
  (`maxTextureSize` guard).
- ~~**G5 — Spreadsheet + Export**~~ ✓ `spreadsheet.py` — per-``(domain, layer)`` structure
  table (rows = elements by id, cols = attributes with coordinate columns first);
  `structure_tables()` groups the pulled Dataset's Label/Point/Track attribute layers; the
  window feeds it the same `payload` the Viewer gets. **Export** (`export.py`, Qt-free):
  File→Export writes the tables long-form to `.csv` / `.parquet` / `.arrow` (pyarrow lazily
  imported; `None`-filled missing cells so numeric columns stay numeric). Verified against
  real `detect.spots` / `measure` output.
- ~~**G6 — Save / load `*.nd2graph.json`**~~ ✓ `document.to_dict`/`load_dict` wrap
  `nodegraph.serialize` + a top-level **`ui`** extras object (node x/y + mute + collapsed);
  the headless loader ignores `ui`, so a GUI file still reads back headless. File menu wired.
  A loaded file's zones/groups + zone back-edges round-trip **verbatim** (preserved in
  `document._zones`/`_groups`/`_back_edges`, warned on open). **Repeat-zone CREATION**
  landed: Graph→"Wrap selection in Repeat zone" (`document.wrap_repeat_zone`) inserts the
  paired `zone.repeat_in`/`repeat_out` boundary nodes, rewires the single dataset frontier,
  adds the Out→In back-edge, records a `Zone`, and **validates via `zones.unroll` with
  snapshot-rollback** before commit. **Group creation ✓ DONE 2026-07-26** (`make_group`/
  `ungroup` — instance-node + stored `Group` def, expanded at run/propagate). *Remaining:*
  Sim/nested zones, in-GUI editing of iterations, drawing the (currently invisible)
  back-edge wire.
- ~~**G7 — Canvas → Engine**~~ ✓ `EngineRunner` (QThreadPool worker + epoch registry;
  latest-wins queueing; per-node progress in the status bar; persistent Memo → an
  unrelated edit recomputes only the invalidated chain). Double-click / F5 to pull.
  **Per-node progress ON THE CARDS ✓ DONE 2026-07-28.** New `Engine(observer=…)` hook
  (`engine.Observer`) emits `start` / `cached` / `done(+seconds)` / `error(+repr)` per node
  plus `ctx.progress(done, total, note)` fractions from inside eager computes; results are
  untouched and observer exceptions are swallowed (a progress sink must never break a run).
  The runner forwards them worker→GUI over a queued signal (`node_progress`, throttled to
  `PROGRESS_MIN_INTERVAL_S` per node but never dropping the final 100%), plus a `plan`
  signal (the pulled node's ancestor closure = the *queued* set) and a runner-level
  `decode` event for the plane read. Instrumented eager nodes (real determinate bars):
  DVC, DIC, StarDist, histogram-threshold, spot detection, threshold, local threshold,
  multi-otsu, measure's plane gather. **Honesty note:** a lazy node finishes in microseconds
  by design, so its cost lands on whoever reads the planes — reported as `decoding` on the
  viewed node, *not* faked as a filled bar on the lazy node.
  **Visual language (rebuilt 2026-07-28 from the user's NodeLab HTML mockup — the first
  pass' amber/green bottom bar + footprint status word read as dated and fought the
  chrome):** one accent for all activity (red only for a failure), *graphic on the canvas,
  numeric on hover*.
  (1) a **2 px accent rail on the header's bottom edge** — determinate fill when the compute
  reports `ctx.progress`, ping-pong sweep when it does not, and **live work only**:
  queued/done/cached/error paint no rail, so a finished graph stays calm.
  (2) a **status dot at the header's right** (stepped left of the 2D/3D switch): pulsing
  bead while working, solid glowing bead on `done`, hollow ring for `cached`/`queued`
  (nothing was recomputed), red on `error` — and it works on a collapsed card, which the
  old bottom bar could not.
  (3) working/selected/failed cards take an **accent border + outer glow**. Glow and dot
  halos are hand-drawn translucent passes (`NodeItem._glow_pill` / `_paint_card_glow`):
  QPainter has no `box-shadow` and a `QGraphicsDropShadowEffect` would be slow + blurry.
  (4) **flowing wires** — while a pull is in flight, edges out of an already-produced node
  carry travelling accent dashes (`EdgeItem.set_flow`, 7-on/5-off) *over* their normal
  tint, so the run state never overwrites the domain/channel coloring (which is what the
  wire MEANS). The dash pen is deliberately **wider** than the tinted base; a narrower one
  just fringes the wire instead of breaking it up.
  (5) a **pulsing run LED** in the status bar (`MainWindow._set_led`) — added as a
  *permanent* widget, because `showMessage()` hides normal status-bar widgets.
  One shared `GraphScene` timer (`PROGRESS_TICK_MS`) drives every pulse/sweep/dash and runs
  only while something needs animating. Percentages and wall times live in each card's
  **tooltip** + the status bar instead of on the card.
  *Remaining:* the decode phase itself is indeterminate (no tile-level counter yet).
- ~~**G8 — Directive UX**~~ ✓ z==1 greys the 3D lever + red-badges a locked-3D node
  (H11; **unknown_axes ≠ z==1** so an unresolved source never greys it); metadata-driven
  ƒmd pills re-seed live from `propagate_meta` on every edit (node cards + inspector).
- ~~**G9 — Light theme**~~ ✓ `theme.py` rewritten to switchable dark/light palettes;
  `apply(mode)` rebinds the module-global color tokens (node cards read them live at paint);
  each QSS panel gained `restyle()` and the window a `set_theme` (View→Light theme toggle).
- ~~**G10 — Chrome + elide**~~ ✓ menu bar (File/Run/View), status bar, docked palette/
  inspector/viewer; long titles elided under the header switch.

**Design reference:** the approved interactive mockup (Artifact) —
`scratchpad/nodelab_v2_canvas.html`. **Offscreen GUI verify:** `scripts/_nodelab_v2_shot.py`
(render) + `scripts/_nodelab_v2_phase5_probe.py` (the driven end-to-end probe, incl. review R1–R6).

> **Phase-5 adversarial review DONE (2026-07-22): 6 confirmed (3 blocker/3 major) + 2 minor, ALL
> fixed + guarded.** Blockers: (1) `GraphScene.sync` rebound stale cards on `load_file` id-reuse →
> now reconciles by record identity; (2) the document dropped zones/groups + zone **back-edges** on
> load→save → now preserved verbatim (`_zones`/`_groups`/`_back_edges`, `has_unedited_structure`
> warns); (3) delete-mid-wire-drag crashed the link-search slot → drag cancelled on delete + anchor
> re-checked. Majors: right-click no longer ends a drag; the G8 re-seed re-announces on a changed
> source key (was once-only); **`io.load`/`view.viewer` moved to a Qt-free `nodelab_v2/ops.py`**
> (`ensure_ops`/`headless_engine`) so a saved GUI graph runs headless (verified PySide6-free).

---

## 5. Remaining — Phase 6 / 7 + skills

- ~~**Phase 6**~~ ✓ — inspection & export wired to the pull engine: Viewer (Voxel heatmap +
  Point/Label overlays), Spreadsheet (per-domain structure tables), CSV/Parquet/Arrow export.
  *Remaining (later):* Track-trajectory overlay, the QRhi GPU viewer path if 60fps needs it.
- ~~**Phase 7**~~ ✓ **DONE 2026-07-22 — scoped coexistence (user-chosen).**
  [`V2.05_phase7_capability_matrix.md`](V2.05_phase7_capability_matrix.md) = the parity
  assessment. **Finding: v2 is NOT at full parity** (~30 v1 nodes unported — DIC/DVC, granule,
  cell-tracker, DL seg, interaction/workflow), so a hard retirement was rejected. Instead:
  `run.py` launches **v2 by default**, `--legacy` → v1; a v2 Help→"Capability vs legacy" note +
  README document both; the `pipeline_kit` parity gate stays green (v1 untouched). A full v1
  removal is deferred until §2 gaps close or are declared obsolete (V2.05 §5 exit criterion).
- ~~**Skills** — rewrite for v2~~ ✓ **DONE 2026-07-21.** Added **`wire-node-v2`** (concepts:
  Dataset/domains, SocketSpec/ModeSpec, the `DimMode` lever + `available_in` variant sockets,
  `Granularity`/`kernel_axes`, `meta_transform`, `ctx.calib`/`to_pixels_v2`, the two-hash memo,
  transfer/bridges/C2) and **`build-node-v2`** (procedure: grill → `register_node` compute →
  metadata+footprint gate → `nodegraph.selftest` verify gate). The v1 `wire-node`/`build-node`
  were re-scoped to legacy `pipeline_kit`-only triggers with a pointer to the v2 variants, so
  "add a node" now routes to v2.

---

## 6. Open decisions still to settle (grill before committing)

0. ~~**Simulation-zone determinism × memo** (V2.00 §16)~~ — **RESOLVED 2026-07-22 (user).**
   *Unroll + revision-fold* keying + *debug-verify + impure escape hatch* determinism policy.
   Implemented in `nodegraph/zones.py` (Phase 4a). See §3 / running log §0.
1. **asyncio↔Qt bridge** for the pull scheduler — qasync vs hand-rolled QThreadPool + epoch registry.
   The cooperative cancel-token layer does the real cancellation, so the asyncio shell may be optional.
   Decide when building **G7**.
2. ~~**Fusion reducers**~~ — **RESOLVED/DONE 2026-07-21** (`sigma_clip` MAD-robust + `trimmed_mean`
   in `reducers.py`, whole-domain). Weighted-mean fusion deferred (needs a weights input).
3. **TensorStore vs {b2nd + hand-rolled scheduler}** — pick ONE store. Leaning b2nd (keystone
   confirmed TILED + planar blocks); revisit only if the async/coarse-view layer needs TensorStore.
4. **Multi-layer keying UX** — Label/Point/Track are multi-instance by source layer; nodes consuming
   them need a clean "which mask / which point set / which tracking" selector (in-body dropdown;
   default = most-recently-added). Design in the GUI (§4). **HALF DONE 2026-07-28
   ([`V2.10`](V2.10_param_socket_contract.md)):** every consuming node now HAS a source-layer
   socket — 18 nodes read a layer-name param that had no socket at all, so the whole
   segmentation chain was pinned to one hardcoded name and a graph could not carry two masks.
   What remains is the picker itself: the sockets are **free text**, not a list of the layers
   actually present on the incoming edge. **Correction (2026-07-29):** V2.10 first claimed this
   was "purely a GUI change" because `propagate_meta` already carried the layer catalog. It does
   **not** — `MetaEnvelope.layers` is declared but never populated (the pass accumulates
   `domains` only), exactly as this doc's §4 G-notes said. The picker therefore needed the
   edit-time layer catalog BUILT first. **✅ DONE 2026-07-29
   ([`V2.11`](V2.11_layer_picker.md)):** `SocketSpec.layer_in`/`layer_out` +
   `NodeSpec.extra_layers` + `MetaEnvelope.layer_names` populated by `propagate_meta`
   (pass-through, drop by axis delta, add this node's writes); `document.layer_choices`
   follows the PRIMARY Dataset edge; the inspector renders an **editable** combo
   (suggestions + free text, because a couple of producers name layers the edit-time pass
   cannot predict). Also fixed a pre-existing bug it uncovered: a wheel over an unfocused
   `QComboBox` silently rewrote and pinned a param — now `_NoWheelCombo`, which covers the
   Mode dropdowns too. **This closes §6.4.**
5. **Field IR depth** — grow beyond arithmetic/comparison: a `Sample`/`Nearest` field op over the
   structure bridges; how expressive before it becomes a mini-language (V2.00 §16).
6. **Migration reality** — greenfield means old `.nd2s_pipeline.json` files aren't opened by v2;
   confirm no important saved pipelines are stranded.

---

## 7. Directive (V2.03) follow-ups still open

The mechanisms exist; these wire them all the way through:
- **`meta_transform` on axis-changing nodes** — only `channel.select` declares one so far. Z-Project /
  Rescale / Stack / Stitch nodes (§2) must declare theirs (helpers in `metadata.py`:
  `resample`/`z_project`/`stack_time`/`channel_select`/`crop`/`stitch`).
- **Edit-time MetaEnvelope pass in the GUI** — `metadata.propagate_meta` is built + tested but not yet
  driving live widget re-seed (G8).
- **Structured calibration reads** — `CALIBRATION_KEYS` + typed `ReadContext` accessors exist; the
  hard-error guard for un-contexted reads is C7.
- **Per-channel derive** — C8.

---

## 8. Conventions, gotchas, pointers

**Node-port convention** (`nodes.py`): a node = `define_node(...)` (registered in `NODES`) + a
`compute(ctx) -> Dataset` in `COMPUTES`. `ctx.calib(key)` reads calibration (recorded → memo-fenced);
`to_pixels_v2` converts units (incl. `um_axial` ÷ z_step); a realized image wraps into
`ArrayProvider`; structure results attach via `Dataset.with_structure(table)` (columns → per-domain
`AttributeLayer`s keyed by source layer). The 2D/3D lever is a `DimMode()` (header-rendered), NOT a
socket; it folds into `recipe_hash` via params so 2D/3D memoize separately.

**Memo invariant:** identity is the monotonic `revision`, never `id()`, never a content hash for
lookup. `recipe_hash` = op + params (incl. mode/dim state) + upstream recipe_hashes + upstream
revisions (structural); declared reads validated on a hit; `output_fingerprint` (content) drives
cutoff/dedup and **includes the image provider's `fingerprint()`** (a Dataset differing only by image
must not collide — a bug the catalog pipeline caught).

**Offscreen-Qt gotchas** (for `_nodelab_v2_shot.py`; none affect the on-display app): the offscreen
QPA platform loads **zero system fonts** (text → tofu) — register Windows TTFs before building the
window; Qt **crashes on offscreen teardown** (exit 5) *after* a successful render — `os._exit(0)` to
bypass; `deleteLater()` widgets **ghost** in an offscreen grab unless you `setParent(None)` first.

**How the user works** (norms): grill one-question-at-a-time before big design; present before locking
(keep V2.00 canonical, add addenda); adversarially verify research/findings and mark unverified;
metadata-intelligent params (unit/derive) on everything spatial/temporal; Qt-free core; **no `pip
install` unprompted**; re-check any survey-sourced dependency (versions are as-of 2026-07-20) before it
becomes a real dependency.

**Key files:** design — `V2.00` (locked), `V2.03` (directive addendum), `V2.02` (Phase-2a spec),
`V2.01` (data-I/O research), `node_backend_reference.md` (backend menu), `V1.90` (v1 GUI plan). Code —
`nodegraph/` (18), `nodelab_v2/` (9), `scripts/_pipeline_kit_parity.py`,
`scripts/_bench_provider_granularity.py`, `scripts/_nodelab_v2_shot.py`. Running log —
`CONTEXT_nodegraph_v2_handoff.md`. Metadata-intelligence (V1.91, reused): vendored
`nd2studios/pipeline_kit/metadata_adapt.py`.
