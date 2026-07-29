# CONTEXT — nodegraph v2 handoff (fresh-chat brief) · 2026-07-23

> **⚠️ UPDATE 2026-07-29 — NodeLab v1 IS REMOVED; the parity gate below is GONE.**
> `nodelab/`, `nd2studios/` (the vendored `pipeline_kit` backend), both v1 scripts, and the
> legacy `build-node`/`wire-node` skills were deleted; `run.py --legacy` no longer exists.
> **Wherever this document says to run `scripts/_pipeline_kit_parity.py --check` or that
> "parity stays green" / "parity clean", skip it** — the only gates now are
> `PYTHONUTF8=1 python -m nodegraph.selftest` and `scripts/_nodelab_v2_phase5_probe.py`.
> Everything §3.D calls "still legacy-only" was **declared obsolete by the user**, not
> ported. `nodelab_v2/ingest.py` no longer reads v1's ND2 metadata parser — it was vendored
> verbatim to `nodelab_v2/nd2_meta.py`. Full record: **`V2.05_phase7_capability_matrix.md`
> §6**. Everything else below remains accurate.

> **Read this first.** Self-contained state + what's next for the **nodegraph v2** effort
> (a greenfield, Blender-geometry-nodes-style node system for the ND2Studios microscope
> image-analysis app). Supersedes `CONTEXT_v2_handoff_2026-07-22.md` (still accurate for
> the deeper history). Depth lives in the sibling docs (§7); this is the orientation +
> forward map.
>
> **UPDATE 2026-07-27 (later³) — GUI launches on a BLANK welcome canvas (no demo graph).**
> `run.py` used to open on the 8-node demo chain with the Viewer eating ~70% of the centre.
> Now: `MainWindow` builds **no** nodes, the central splitter opens with the **Viewer folded
> away** (`setSizes([0, 1200])`, pane 0 made collapsible) so the canvas owns the whole centre,
> and the new `nodelab_v2/welcome.py` **`WelcomeCard`** sits centred on it — dashed "+" affordance,
> *Start your graph*, the three gestures that place a node (palette double-click/drag, `Ctrl+L`
> load, double-click-to-preview + `Ctrl+Space`), and buttons **Load image… / Browse nodes /
> Example graph**. It shows exactly while `doc.nodes` is empty (so File → New re-welcomes) and
> **forwards a palette drop that lands on it** to the canvas beneath (`op_dropped` → the window's
> normal drop handler) — otherwise the card would swallow the drop it is asking for. The first pull
> that returns an image unfolds the Viewer to the old 2.5:1 split (`window.VIEWER_SPLIT`, via
> `_open_viewer`, which only acts on a folded pane so a user-sized one is left alone). The demo
> chain became **`MainWindow.build_demo()`** (public, clears first, fits the view) — reachable from
> the card's *Example graph* button and driven by the probe / `_nodelab_v2_shot.py` (new
> `--welcome` flag shoots the launch state instead). Two supporting fixes: `GraphView.fit_all` no
> longer fits an EMPTY scene (that zoomed into a ~140 px box — giant grid dots on a blank canvas;
> it resets to 1:1 on the origin), and the canvas hides its scrollbars (it pans by dragging, and
> `_grow_scene_rect` made the bars meaningless). New `PalettePanel.focus_search`. **Verified:** GUI
> probe ALL PASS with a new launch block (blank canvas, centred card, 1:1 zoom, drop passthrough
> places a node, File → New re-welcomes, `build_demo()` fills 8 nodes); a real windowed run reports
> `launch | nodes: 0 | welcome: True | split: [0, 789] | zoom: 1.0` → *Example graph* → pull →
> `split: [565, 224]`.
>
> **UPDATE 2026-07-27 (later²) — GUI: MAXIMIZED node canvas + mini-map Viewer + click-to-preview.**
> The centre still splits Viewer-over-canvas by default, but the canvas now carries a painted
> **maximize toggle** in its top-right corner (`scene.GraphView._max_btn`, a `minimap.HudButton`;
> also View → *Maximize node canvas* / **Ctrl+Space**, `Esc` to leave). Maximizing hands the whole
> centre to the graph and **re-homes the very same `ViewerPanel`** (reparented, never a copy — so
> channels/LUT/playback/overlays carry across) into the new **`nodelab_v2/minimap.py`
> `MiniMapOverlay`**: a bordered, HUD-styled frame pinned to the canvas' **top-left** corner
> (accent corner brackets, live-state dot, elided "id · label" header, dock button; drag the
> header to move — it re-anchors to the nearest corner so window resizes keep it put — drag the
> bottom-right grip to resize, `set_frame_size` remembers the wanted size and re-clamps to the
> canvas). While maximized, **clicking any node previews it live**: selection → a
> `FOLLOW_DELAY_MS`=170 ms debounce (a marquee across a chain = ONE pull) → `pull_node`; the shown
> card wears an **accent spine + ● tag** (`GraphScene.set_viewed` → `NodeItem.set_viewed`, survives
> `sync()`), and the mini-map dot tracks idle/busy/live/error. View → *Preview clicked node* exposes
> the same follow behaviour when docked (forced on while maximized, restored on dock).
> `ViewerPanel.set_compact(on, force=…)` trims the strip for the mini-map (histograms + LUT tool
> buttons + fps spinners fold away; cursors/channel toggles/overlays/status stay; overlay labels
> shrink to P/L/T; image minimum 200² → 120×90) and the window re-applies it `force=True` after each
> pull, since `_rebuild_channels` builds fresh (visible) LUT columns. Two viewer-surface fixes fell
> out: `_ImageView` now hides its scrollbars and **re-fits on resize** unless the user has zoomed or
> panned (docking ↔ mini-map re-frames instead of cropping), and `glview` hardened for reparenting —
> `_release_gl` (via `QOpenGLContext.aboutToBeDestroyed`) frees textures/buffers/program with the
> dying context current and `initializeGL` rebuilds + replays `_last_planes`.
>
> **Separately fixed (user hit it live, 2026-07-27):** dropping a wire onto a socket crashed —
> `GraphScene._end_wire` runs `_cancel_temp()` (which clears `_drag_fixed`) *before* resolving the
> drop, then called `_endpoints(sock)`, which read the just-cleared anchor → `AttributeError:
> 'NoneType' object has no attribute 'io'` on **every** connect made with the mouse. `_endpoints`
> now takes the anchor as an argument (`_end_wire` passes the captured `fixed`) and both it and
> `_validity` are None-safe. No probe covered the socket-drop path — every wiring test went through
> `document.connect` — so the G1 section gained one that drives `begin_wire`/`_update_temp`/
> `_end_wire` directly (connect, detach-and-re-drag from a connected input, invalid drop = no-op,
> no dangling drag state). **Verified:** GUI
> phase5 probe ALL PASS with a new E9 block (re-home, top-left placement, compact fold, one-pull
> debounce, viewed-card marker, re-anchor, dock-back at the old split size) + a third screenshot
> `nodelab_v2_phase5_maximized.png`; a real windowed run on the **GPU path** keeps the same
> `GLImageView` (no `gl_failed`, image intact across maximize → scrub → dock). Files:
> `nodelab_v2/minimap.py` (new), `scene.py`, `window.py`, `viewer.py`, `node_item.py`, `glview.py`,
> `scripts/_nodelab_v2_phase5_probe.py`.
>
> **UPDATE 2026-07-27 (later²) — `track.objects` PORTED (§3.C cleared; the last big v1
> kernel gap).** The 5.3k-line vendored `track_objects` kernel is now a real node — the
> **richer sibling of `track.link`**, exposing all **five** interchangeable linkers
> (centroid Hungarian · SerialTrack topology PTV · Cell-Tracker topology / fingerprint /
> mask-overlap IoU) behind a `method` Mode, with a `target` Mode (Label **or** Point
> members) mirroring `track.link`. Deps were all already present (numba, pandas, scipy) —
> the kernel imports in 1.4 s and every method runs for real.
>
> **Why a separate node, not a mode on `track.link`:** the kernel imports numba + pandas at
> module scope, while `nodegraph.tracking` is deliberately dep-free — folding it in would
> make the built-in linker unimportable without numba. Both nodes read the same upstream
> layers and emit the same membership shape, so they are drop-in alternatives.
>
> **Input contract:** v2 Label tables already carry every key the kernel needs — `y`/`x` →
> `centroid_y_px`/`centroid_x_px`, `area` → `area_px`, all in px (`structure._label_table`).
> Grouping keys `(f"c{c}z{z}", m)` — a **compound channel key** — so every `(m,c,z)` is an
> independent tracking run off the kernel's one shared id counter. That is what makes a
> 2D-only kernel correct on a z-stack: planes are never linked to each other. 3D
> (`z_kind="subpixel"`) is **refused** (pointing at `track.link` point mode); dimensionality
> is inherited from the members' `z_kind` per §7b, never a DimMode lever.
>
> **Output (user-locked):** the Track table (bypassing `TrackMembership.to_table`'s hardcoded
> 3-key literal to carry `track_length`/`m`/`c` — the viewer's gate is a superset test) **plus
> a `track_id` write-back column on the member layer** (0 = untracked, matching the bridges'
> `drop_nonpositive` rule). The write-back is the load-bearing half: **nothing in v2 consumes
> `Domain.TRACK`** (the bridges need a live `TrackMembership` object no node can rebuild from
> a Dataset), so without it the result would be viewer/CSV-only. Results are renumbered
> through a new public `tracking.build_membership` → contiguous `1..K` first-appearance ids,
> rows sorted `(track_id,t,member_id)` — byte-identical conventions to `track.link`.
>
> **Determinism (memo §10):** measured — every linker is repeat-stable, and row order changes
> only the id *numbering*, never the induced partition. Two guards make it total anyway: a
> canonical `lexsort` row order `(m,c,z,t,id)` + the renumber. `st_use_prev_results` is
> **pinned False** (its POD-GPR warm start draws unseeded from numpy's global RNG).
>
> **Adversarial review (5 lenses → 8 findings verified by refutation) found 4 real defects,
> all fixed + regression-guarded.** All were one class — *a live-looking GUI control the
> selected kernel path silently ignores*, which the node's own docstring charter forbids:
> (1) `max_distance` was exposed under `overlap`, which matches purely by mask IoU and never
> receives it → socket now gated to the four linkers that consume it (and calibration is no
> longer read there, dropping a phantom memo fence); (2) `max_size_diff_frac` is a *gate* for
> the centroid linker but self-neutralises on zero areas, so on Point members (no `area`
> column, ever) it silently let the Hungarian solver swap the identities it was set to keep
> apart → hidden on Point + refused if explicitly set; (3) kernel **gotcha #8** — every linker
> early-returns on a `(m,c,z)` group spanning <2 frames *before* the `min_track_length`
> post-pass, so at `min_track_length=1` those rows kept `track_id=None` and got the 0
> "untracked" sentinel; a member's fate depended on whether an *unrelated* groupmate existed
> at another frame → such groups are now seeded as singletons (provable no-op at the default
> ≥2); (4) `ct_max_gap=0` is floored at 1 by the overlap path only → refused (the clean fix
> edits the **vendored** kernel; the al-dic `gridxy_roi_range` precedent licenses version-compat
> fixes, not semantic preferences, so the node refuses instead and the vendoring stays verbatim).
>
> Files: `nodegraph/nodes.py` (+`_TRACK_OBJECT_METHODS`, `_compute_track_objects`, registration),
> `nodegraph/tracking.py` (+public `build_membership`), `nodegraph/selftest.py` (+`test_track_objects`).
> Catalog **53→54**; selftest **50→51 groups**; parity clean; GUI phase5 probe ALL PASS.
> **Next item:** the v1→v2 `.nd2s_pipeline.json` importer (nothing built — the only remaining
> item that unblocks existing v1 users), the DIC mesh-refinement nodes (optional), or the
> §3.D legacy-only leftovers (`logic:if_else`, enhancement extras, workflow IO).
>
> **UPDATE 2026-07-27 (later) — DIC (`analysis.dic_correlate`) WIRED + VERIFIED (al-dic 0.7.1 present).**
> The last dep-gated stub is now a real compute (`_compute_dic_correlate` in `nodes.py`),
> the **2D image sibling of DVC**: it owns the m/t/z loop + reference pairing (a
> `reference_mode` lever {fixed_frame | previous_frame} + `reference_frame`, an optional
> external `reference` Dataset, an optional `roi` Voxel-mask layer from `analysis.roi_mask`),
> reads calibration for `voxel_size_um=(px,px)`, and calls the vendored kernel
> `nodegraph.kernels.dic_correlate.run_pyaldic_pair` per ref/def PLANE pair → a **Point**
> displacement field (grid centers, disp µm, reusing DVC's `_dvc_rows`; DIC's `DVCResult`
> has strain/qfactor = None). 2D-only (use `dvc_field` for 3D). Provenance stamped as
> `dic_reference_mode`/`dic_reference_frame` (distinct from `dvc_*` so a DIC field is NOT fed
> to `accumulate_field`, which is ALDVC-only).
>
> **al-dic turned out to be INSTALLED (0.7.1)** on this machine (pulled in with the GUI's
> PySide6) — so DIC RUNS for real, and surfaced one integration gap now FIXED: al-dic ≥0.7
> requires `para.gridxy_roi_range` set EXPLICITLY when `run_aldic()` is called directly (it
> defaults to a zero-size box → "No grid points generated"). The vendored adapter never set it
> (older al-dic auto-derived it), so `nodegraph/kernels/dic_correlate.py:_build_dicpara` now
> sets it to the full image extent `(gridx=(0,W), gridy=(0,H))` — the solver insets by
> `winsize//2` and applies `use_masks` itself (a documented, version-compat deviation from the
> verbatim vendoring). **Verified end-to-end:** `test_catalog_dic` runs a real IC-GN+ADMM
> correlation on a 128² speckle pair and recovers a planted (1,3)px shift in µm (self-pair ≈ 0);
> when al-dic is ABSENT the node still raises a friendly ImportError (the other branch). Cost:
> the first al-dic call JIT-compiles numba (~5-6 s, one-time/process) — it's the only al-dic
> solve in the suite (the kernel-ports group skips its DIC call when al-dic is present).
> Catalog count unchanged (stub → real); **`_gated_compute` removed** (DIC was its only user).
> Selftest **50 groups**; parity clean; GUI probe ALL PASS. Records: this banner + memory index.
>
> **UPDATE 2026-07-27 — remaining C1 slivers DONE (§3.F); only computed-provider pyramids
> left open.** Three §6b follow-ups shipped, all in the nodegraph core (disjoint from the
> concurrent GPU-viewer work): (1) **`util.stack` T→1 tree-reduce** — generalized
> `ZReduceProvider`→`_AxisReduceProvider` (reduce axis z|t) + new `TReduceProvider`; monoid
> combiners fold the T series per tile, median/sigma_clip/trimmed_mean gather the t-column;
> byte-identical to eager. (2) **Lazy units** (eager-stat/lazy-apply) — `enhance.normalize`
> (plane→`MapComputeProvider`, volume→`VolumeComputeProvider`, series→eager (lo,hi) per (m,c)
> + lazy per-plane apply), `align.drift` (eager per-frame FFT estimate + lazy per-plane shift),
> `util.resample` (new **`PlaneRealizeProvider`** — geometry-changing per-unit lazy realize,
> only touched planes/volumes resize). (3) **Kernel-param field gate** (§6b Fork B) —
> `SocketSpec.kernel_param` flag (radius/σ sockets carry it) + `_map_image` drops TILEABLE→
> plane unit when a `kernel_param` socket is wired to a **non-Const** Field. Files:
> `nodegraph/streaming.py` (+`_AxisReduceProvider`/`TReduceProvider`/`PlaneRealizeProvider`),
> `nodes.py` (stack/normalize/drift/resample rewired + `_kernel_field_varies` gate in
> `_map_image` + kernel_param flags), `registry.py` (`kernel_param`), `__init__.py` exports.
> Selftest **+`streaming slivers`**; parity clean; GUI probe ALL PASS. Record:
> **`V2.04_c1_streaming_eval.md` §6b addendum (2026-07-27)**. **Computed-provider pyramids
> DEFERRED (user decision 2026-07-27) — the entire C1 §6b sliver list is now closed.**
> **Next item:** `track_objects` (a richer alternative to `track.link`), the DIC mesh-refinement
> nodes (`dic_mesh_refinement` kernel, optional), or remaining GUI refinements (QRhi/GPU viewer
> landed via a concurrent session; v1→v2 importer still open).
>
> **UPDATE 2026-07-26 (later still) — Memo GC DONE (§3.F C1 sliver closed).** Closes the
> last-flagged C1 memory hazard: the persistent memo would pin one full raster per iteration
> of a high-T unrolled zone forever. `Memo(budget_bytes=…)` is now a byte-budget **LRU** —
> retained realized bytes (`memo.payload_bytes`: a Dataset's in-memory `ArrayProvider` image
> + attribute arrays; lazy/disk providers = 0) over budget evict LRU entries; accounting is
> keyed by the **unique deduped blob** (`fp → refcount`), so a shared blob frees only at its
> last ref. Eviction is **correctness-safe** (only forces a later recompute — deterministic
> output fp, identity on the monotonic `revision`), so **no pinning**; `_last_fp` (Salsa
> cutoff) is never GC'd. `budget_bytes=None` (headless default) = byte-identical pre-GC
> behavior; `Engine(memo_bytes=…)` plumbs it; the **GUI persistent memo caps at 1 GiB**
> (`nodelab_v2.runner.MEMO_BUDGET_BYTES`). Re-pull note: an evicted ancestor recomputes with
> a fresh revision → the descendant chain recomputes (honest memory/recompute trade); a
> budget that fits never evicts and re-pulls fully memoized. Files: `nodegraph/memo.py`
> (+`payload_bytes`, LRU/refcount), `provider.py` (`ArrayProvider.nbytes`), `engine.py`
> (`memo_bytes`), `nodelab_v2/runner.py`. Selftest **46→47 groups** (+`memo GC`); parity
> clean; GUI probe ALL PASS. Record: **`V2.04_c1_streaming_eval.md` §6b addendum**.
> **Next item:** `util.stack` tree-reduce / lazy resample-normalize-drift units (remaining
> C1 slivers), `track_objects`, or the remaining GUI refinements.
>
> **UPDATE 2026-07-24 — DVC is now DONE (§3.A / §6 resolved).** The gating decision was
> grilled + locked and both nodes shipped: **`analysis.dvc_field`** (ALDVC displacement+
> strain → a POINT subset-grid field; `reference_mode` lever fixed_frame/previous_frame +
> an OPTIONAL external `reference` Dataset input socket) and **`transform.rasterize_field`**
> (Point field → Voxel layers). Design record: **`V2.06_dvc_field.md`**. Catalog **47→49**;
> selftest **40→42 groups**; parity clean; GUI phase5 probe ALL PASS. Also fixed a latent
> `graph.py` `dataset_preds` edge-order bug (two-Dataset-socket nodes could take the wrong
> input's calibration) + 3 stale viewer refs in the GUI probe.
>
> **UPDATE 2026-07-24 (later) — DVC cumulative accumulation + warm-start DONE (§3.A closed).**
> After a grill settling the algorithm question (cumulative-from-incremental is a real
> **Lagrangian composition**, distinct from `fixed_frame` direct correlation — *not* a
> display transform), two nodes/changes shipped: **`analysis.accumulate_field`** (wraps the
> vendored `accumulate_incremental`/`compute_strain` = the core of `build_accumulated_results`;
> composes `previous_frame` increments → cumulative disp+strain Point series, same schema;
> **inherits the reference config from the upstream DVC node's stamped provenance** — no
> redundant control, refuses fixed_frame/no-provenance) and **cross-frame warm-start**
> (`u0_seed` + a `newFFTSearch` toggle) on `analysis.dvc_field` (v1 parity). Catalog **49→50**;
> selftest still **42 groups** (DVC group extended); parity clean; GUI probe ALL PASS. Design
> record: **`V2.06_dvc_field.md` §4**; new **`wire-node-v2` §7b** (provenance inheritance).
>
> **UPDATE 2026-07-24 (later still) — §7b applied catalog-wide + C8 DONE.** (a) The §7b
> directive went into the **core**: `Dataset.with_structure` auto-preserves each table's
> `z_kind` → `__struct_zkind__` (read via `structure_zkind`), so every structure producer
> self-describes dim; `transform.rasterize_field` **lost its DimMode lever** and inherits
> dim from the field's `z_kind` (fixes a default silent-corruption bug — a 2D per-plane
> field on a z>1 image was read as 3D); `measure` carries `z_kind` forward (clobber guard).
> (b) **C8 — per-channel derive (`ctx.channel`)**: `EvalContext.channel(c).param(name)`
> resolves an `emission_nm`-derived param for channel c (override → per-c derive → default,
> auto memo-fenced); `detect.spots` now resolves radii per channel, `enhance.deconvolve`
> resolves emission/NA per channel (and honors overrides). Selftest **43 groups** green;
> parity clean; GUI probe ALL PASS. Records: `V2.06 §6`, `wire-node-v2 §7 + §7b`,
> `CONTEXT_v2_remaining` (C8 ✓).
>
> **UPDATE 2026-07-24 (fast CCL / C6 follow-up DONE).** `structure.label_components` now
> uses `scipy.ndimage.label` + a raster-canonical relabel (`_canonical_relabel`) + `bincount`
> areas + `center_of_mass` centroids — **byte-identical** to the pure-numpy flood-fill (kept
> as `_label_components_flood`: the reference + scipy-absent fallback). ~40× faster at 512²
> (1.17 s → 0.028 s); the ~191 s cliff at 6554² is gone. Selftest `structure` group asserts
> scipy≡flood across 2D/3D connectivities (**44 groups** now); parity clean; GUI probe ALL
> PASS; bench `scripts/_bench_ccl_watershed.py` updated.
>
> **UPDATE 2026-07-25 — GENERAL-NODE directive + rename sweep + granule chain DONE
> (`V2.07_general_nodes_granule_chain.md`).** Standing user directive: **nodes are general
> primitives, not workflow-specific** (granule-ness = default params/docs). (a) Rename sweep
> (behavior identical): `detect.beads→detect.particles`, `granule_boundary→boundary_band`
> (any labels), `granule_cluster→cluster_points`, `dic_roi_mask→roi_mask`; kept true
> algorithm names (dvc_field, dic_correlate, stardist_nuclei). (b) **Mesh decision:** a mesh
> adds face/topology no domain has, but downstream is raster/label + there's no 3-D viewer →
> keep existing domains, defer a MESH domain. (c) New **`analysis.tessellate_volume`** (fused
> granule_tessellate+granule_volume_mask; mesh internal): labeled Points → Voxel **Label**
> raster + per-label geometry table (centroid + analytic volume_um3/density/n_points/
> surface_area); `merge_tol=0` (general), 3D-only, chains into `boundary_band`/`measure`;
> `cluster_points` upstream still sklearn-gated but the node is testable with synthetic
> labels. Selftest **45 groups**; parity clean; GUI ALL PASS; catalog **52**.
>
> **UPDATE 2026-07-26 (later⁵) — GUI modern-look pass + header removals.** A cohesive
> visual refresh: new shared **`theme.controls_qss()`** (rounded buttons w/ hover/pressed,
> modern accent-filled checkbox indicators, focus-ring inputs/combos, thin rounded
> scrollbars, tooltips) appended in every panel's `restyle()`; modernized window chrome in
> `window._window_qss` (rounded menu items/menus, slim dock titles, rounded tab bar,
> splitter handles, status bar). **Viewer header REMOVED** (`viewer.py`): no more title/
> overlay row — the image now fills the reclaimed top space; the **Points/Labels/Tracks
> overlay checkboxes moved into the channel-strip** (right of the C toggles) so all display
> controls sit in one strip below the image; the viewed-node name folded into the bottom
> status readout (`_title` gone — `show_result`/`show_error` write it into `_status`).
> **Console headers REMOVED** (`console.py`+`window.py`): both the redundant `QDockWidget`
> "Console" title strip (`setTitleBarWidget(QWidget())` + `NoDockWidgetFeatures`) AND the
> internal "Console" label are gone — only a slim right-aligned Copy all/Clear strip remains
> above a flush log; the dock shrank (118 px) so the **node canvas gains that vertical
> space**. GUI phase5 probe ALL PASS (viewer API preserved: `_status`/`_sliders`/`_overlay`/
> `_label_overlay`/`_track_overlay`/`_repaint` intact; status keeps "pulled in …"); core
> gates untouched (48 selftest / parity clean). Files: `theme.py`, `window.py`, `viewer.py`,
> `console.py`, `palette.py`, `spreadsheet.py`, `inspector.py`.
>
> **UPDATE 2026-07-26 (later⁴) — GUI Group creation DONE (§3.E).** Instance-based "Make
> Group" (Blender-style): `document.make_group(node_ids, name)` collapses a selected
> **linear** sub-chain (exactly 1 external Dataset input + 1 output; sources refused — their
> seed would be buried) into a `nodegraph.groups.Group` DEFINITION (body bounded by
> `group.input`/`group.output`) + a single `group:<name>` **instance node** wired to the
> same frontier; `ungroup(inst)` reverses it (restores the interior with fresh ids,
> reconnects the frontier, drops the def). **Run/propagate expand the instance** —
> `to_graph(materialize=True)` now calls `groups.expand` before channel-tap materialization,
> so the engine/memo/meta-pass need no group awareness; propagate patches each instance's
> envelope from its body OUTPUT (`grp.out%inst`) so downstream **domain rails read through
> the opaque instance** (verified: label node shows VOX through the group). GUI: instance
> renders as a purple GROUP card (synthetic single Dataset in/out via `document.input_specs`/
> `output_specs` special-casing; inspector shows a group note); Graph → "Group selection…"
> (Ctrl+G) / "Ungroup" (Ctrl+Shift+G); serialized as instance-node + separate group def
> (round-trips; `has_unedited_structure` no longer flags groups). Validated by a trial
> `expand` before commit (atomic rollback). Files: `document.py`, `node_item.py`, `theme.py`
> (`group` category color), `inspector.py`, `window.py`. Selftest 48 (unchanged — headless
> `groups` already covered); parity clean; GUI phase5 probe ALL PASS (+ Groups check).
> **§3.E GUI-refinement list now fully cleared** except the optional QRhi path + a v1→v2
> importer. **Next item:** `track_objects`, remaining v1 kernel gaps, or QRhi.
>
> **UPDATE 2026-07-26 (later still) — GUI reroute dots + labelled frames DONE (§3.E).**
> Two canvas-organization refinements. **(a) Labelled frames** (GUI-only, like node
> positions — never enter the run graph): a `FrameRecord` in `document.py` + a new
> `nodelab_v2/frame_item.py` `FrameItem` that paints a tinted, titled rounded rect BEHIND
> its member nodes, auto-sizes to enclose them (reflows on every node move via
> `scene.reroute`), and drags all members together; Graph → "Frame selection…" (Ctrl+J);
> serialized in the `ui` extras (round-trips); a frame always has ≥1 member (emptied ⇒
> auto-removed); deleting a frame keeps its nodes. **(b) Reroute** = a REAL
> `rr.reroute` pass-through node (identity compute; TILEABLE, no meta_transform; hidden
> from palette via the `rr.` prefix) rendered COMPACT (a 22 px dot with edge sockets, via
> a NodeItem reroute branch), created by **double-clicking a wire** (splices into it;
> refuses value/field wires without dropping the wire). Selftest **47→48 groups**
> (+`rr.reroute` identity/compose test); catalog +1; parity clean; GUI phase5 probe ALL
> PASS (+ Frames + Reroute checks). Files: `document.py`, `frame_item.py` (new), `scene.py`,
> `node_item.py`, `theme.py` (`RR_SIZE`), `window.py`, `nodegraph/nodes.py`,
> `nodegraph/selftest.py`. **Next item:** Group-creation UI (mirrors Repeat-zone creation),
> QRhi path, or `track_objects`/remaining v1 gaps.
>
> **UPDATE 2026-07-26 (later) — GUI Track-trajectory overlay DONE (§3.E).** The viewer
> gained a **"Tracks"** overlay (third checkbox beside Points/Labels): it joins each Track
> layer's `(track_id, t, member_id)` rows to member positions on the Point/Label domain
> (by `id`; Point/Label id-spaces overlap, so per Track layer it picks the member layer
> whose id set best covers the members), then draws one golden-angle-hued polyline per
> track through its members' `(y,x)` ordered by t (whole trail, projected over z, filtered
> to the viewed M) with per-t vertex dots and an **enlarged dot on the vertex whose t ==
> the viewed T** (so scrubbing/playing T highlights each track's current position). Pure
> `viewer.py` addition (`_tracks_here`/`_member_layers`/`_track_color` + a `_repaint`
> block); no engine/core change. Verified: standalone join probe (Point + Label members,
> M-filter, current-T vertex, distinct hues) + a **permanent check folded into the phase5
> GUI gate** (`_nodelab_v2_phase5_probe.py`) + a visual render. GUI probe ALL PASS; core
> gates untouched (still 46 selftest groups; parity no drift). **Next item:** reroute dots
> + labelled frames, Group-creation UI, or `track_objects`/C1 slivers.
>
> **UPDATE 2026-07-26 — `cluster_points` WIRED; granule chain fully live (V2.07 §5).**
> scikit-learn was **already installed (1.9.0)** — the "ABSENT" note below was stale. The
> gated `cluster_points` stub is now the **real** GaussianMixture/KMeans+BIC compute (ports
> `granule_cluster`): unlabeled 3-D Points → a per-point cluster-id column (µm-scaled fit,
> per-(m,t,c), calib fenced, z_kind kept). **`detect.particles → cluster_points →
> tessellate_volume → boundary_band` now runs end-to-end.** Only `dic_correlate` (al-dic)
> stays gated. Selftest **46 groups** (+`cluster points`); parity clean; GUI ALL PASS.
> **Next item:** `track_objects`, C1 slivers, or GUI refinements.

---

## 0. Where we are — one paragraph

**The v2 build sequence (Phases 1–7) is COMPLETE for the workflow it covers, and a first
wave of v1 analysis-kernel ports has landed.** The headless engine (`nodegraph/`) is a
lazy-pull, two-hash-memo, per-tile **streaming** engine (C1) with a **47-node**
metadata-intelligent catalog (2D/3D lever, domain transfer, structure bridges, zones/
groups, tracking), real ND2 ingest, JSON serialization, and per-tile field eval. The GUI
(`nodelab_v2/`) is a working editor on the Qt-side QThreadPool bridge: interactive wiring,
palette, a **permanent multi-channel Viewer** (colour composite, M/T/Z sliders + playback,
Label/Point overlays), Spreadsheet + CSV/Arrow export, a **Console**, light theme,
splice/collapse, and Repeat-zone creation. **Phase 7 was resolved as scoped coexistence**
(`run.py` launches v2 by default, `--legacy` runs v1; nothing deleted). Most recently,
**6 v1 analysis kernels were ported to the catalog + 2 registered as dep-gated stubs**
(bead detection, histogram threshold, registration, granule boundary, StarDist nuclei, DIC
ROI mask). The biggest open item is a **design decision for DVC** (`aldvc_field`); the
rest is porting the remaining v1 capability gaps + GUI refinements.

---

## 1. Gates — must stay green (run these first in any session)

```bash
PYTHONUTF8=1 python -m nodegraph.selftest          # → 40 groups, "ALL NODEGRAPH SELF-TESTS PASSED"
python scripts/_pipeline_kit_parity.py --check     # → "no drift" (v1 pipeline_kit untouched)
```
- **`PYTHONUTF8=1` on Windows** — some `[ok]` lines carry `↔`/`σ`/`µ` glyphs the cp1252
  console chokes on (cosmetic, not a failure).
- Catalog = **47 node types**. `python run.py` boots NodeLab **v2** (default);
  `python run.py --legacy` boots NodeLab **v1** (the vendored pipeline_kit backend).
- **GUI offscreen verify:** `PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png`
  (drives wiring rules, H11, mute, save/load, a real off-thread pull, splice, collapse,
  theme, zone-wrap, spreadsheet, export). `scripts/_nodelab_v2_shot.py out.png` = a render.
  **Offscreen gotchas:** register Windows TTFs (else tofu), `os._exit(0)` to bypass the
  exit-5 teardown crash, `setParent(None)` before an offscreen grab.
- Heavier / not in the fast gate: `scripts/_ingest_nd2_smoke.py` (real ND2),
  `scripts/_bench_provider_granularity.py` (keystone).

**Env:** Python 3.13.1; numpy 2.4.2, scipy 1.17.1, scikit-image 0.26.0, blosc2 4.9.1,
zarr 3.2.1, pyarrow 23.0.1, nd2 0.11.2, PySide6 (+PyQt6). **NEW/confirmed present (as of
2026-07-23):** numba, pandas, opencv (cv2), cupy, **tensorflow + stardist + csbdeep**
(the StarDist model `2D_versatile_fluo` loads). **scikit-learn 1.9.0 PRESENT** (was listed
absent; confirmed 2026-07-26 → `cluster_points` wired). **al-dic 0.7.1 PRESENT** (2026-07-27;
pyALDIC IC-GN+ADMM DIC solver — `dic_correlate` fully WIRED + verified end-to-end; needed the
`gridxy_roi_range` compat fix in its kernel for al-dic ≥0.7; https://github.com/zachtong/pyALDIC).
**ABSENT (gate kernels):** cellpose,
bart, tensorstore, PySAP. **Never `pip install` unprompted; re-verify any backend API
in-env before writing against it.**

---

## 2. What's BUILT (orientation only — don't rebuild)

**`nodegraph/` engine (24 core modules, Qt-free):** domains, reducers (+fusion +
`PartialReducer` tree-reduce), revision, dataset, transfer (+multi-hop `execute_bridge_plan`),
sockets, registry (`DimMode` lever, `Granularity`, `available_in` variant sockets,
`meta_transform`), graph, metadata, provider (Synthetic / in-mem + **disk** B2nd /
ArrayProvider), memo (two-hash), engine (lazy pull + ReadContext + granularity routing +
`strict_reads`), **streaming** (C1: `TileCache`, `MapComputeProvider`/`VolumeComputeProvider`/
`ZReduceProvider`/`WindowView`, `stream_fp`, `realize`), structure (CCL / seeded-watershed /
columnar tables / `TrackMembership`), bridges, field (windowed per-tile eval), boundary,
zones (unroll + revision-fold + per-frame-T), groups (nestable expand), tracking, serialize,
**nodes** (the 47-node catalog), selftest. Plus **`nodegraph/kernels/`** (15 vendored
pure-compute v1 kernels, byte-verbatim + their `.md` contracts — §4).

**Catalog (47):** the original 39 (`channel.select`; the enhancement suite; `analysis.*`
threshold/label/measure/edt/watershed/multiotsu/threshold_local/extract_boundary;
`detect.spots`; `util.{zproject,crop,resample,stack}`; `align.drift`;
`transform.transfer_domain`; `track.link`; zone/group boundaries) **+ 8 new (2026-07-23):**
`detect.beads`, `analysis.histogram_threshold`, `registration.stabilize`,
`analysis.granule_boundary`, `detect.stardist_nuclei`, `analysis.dic_roi_mask`, and the
dep-gated stubs `analysis.granule_cluster` (sklearn) + `analysis.dic_correlate` (al-dic).

**C1 streaming eval (V2.04, LOCKED + implemented):** computes return lazy chained
providers; overlap-recompute halos (probe-verified); engine-owned byte-budget tile/field
cache keyed by staleness-proof flat fingerprints; PartialReducer tree-reduce; windowed
per-tile field eval. Adversarially reviewed (9 confirmed fixes, R1–R9 guards).

**GUI (`nodelab_v2/`, 18 modules):** `document.py` (Qt-free `GraphDocument`: the editing
model, validated wiring, `propagate_meta` re-seed, mute/collapse bypass, save/load,
Repeat-zone creation), `runner.py` (`EngineRunner`: **G7 QThreadPool + epoch registry**,
persistent Memo, `io.load` source resolution + synthetic fallback, per-channel plane
render), `scene.py`/`node_item.py`/`edge_item.py` (document-mirroring canvas, drag-connect,
splice, detach-redrag, link-search), `palette.py`, **`viewer.py`** (permanent panel:
multi-channel emission-colour composite, M/T/Z sliders + play/pause + fps, per-channel
toggles, Label/Point overlays, scroll-zoom/drag-pan), `spreadsheet.py` + `export.py`
(CSV/Parquet/Arrow), **`console.py`** (log dock), `inspector.py`, `theme.py` (dark+light),
`ops.py` (Qt-free `io.load`/`view.viewer` + `headless_engine`), `ingest.py`, `window.py`
(QSplitter: Viewer over canvas; File/Run/Graph/View/Help menus). *(Note: viewer/window/
console/scene/inspector/runner/theme evolved 2026-07-23 beyond the Phase-5/6 baseline —
the offscreen probe + boot check pass; re-run the probe after touching them.)*

**Phase 7 = scoped coexistence** (`V2.05_phase7_capability_matrix.md`): v2 default, v1 on
`--legacy`, pipeline_kit untouched. Full v1-vs-v2 capability matrix + the remaining gap list
live there.

**Roadmap marks:** C1–C8 done (C1 streaming, C2 multi-hop, C3 tracking, C4 ND2 ingest, C5
provider version, C6 CCL benchmark, C7 strict reads; C8 per-channel derive = the
`_channel_emission` path). Phases 1–7 done. Skills: **`build-node-v2` / `wire-node-v2`**
(active), `wire-node`/`build-node` (legacy v1), `grilling`.

---

## 3. What REMAINS — prioritized

**A. DVC (`aldvc_field`) — ✅ FULLY DONE 2026-07-24 (see top banner + `V2.06_dvc_field.md`).**
The original open framing is kept below for context; both forks were locked and shipped.
The former remaining sliver — `build_accumulated_results` (cumulative Lagrangian
composition of `previous_frame` increments) — is **now wired** as `analysis.accumulate_field`
(+ cross-frame warm-start on `dvc_field`). Nothing left open on DVC.

**A′ (original, now resolved). DVC (`aldvc_field`) — the biggest gap; needs a DESIGN DECISION first (grill).**
The kernel is vendored (`nodegraph/kernels/aldvc_field.py`, imports OK) and produces a
per-voxel **displacement vector field + strain** from a **ref+def volume pair**. Two forks
to settle with the user before wiring: (i) **how DVC consumes ref/def** — consecutive
timepoints (t=0 ref, t→t+1 def)? a `reference_frame` param? two Dataset inputs? — and (ii)
**how to represent a voxel vector field in v2** — three Voxel scalar layers `disp_z/disp_y/
disp_x` (µm) + strain component layers, or a new field/vector representation. Once locked,
the port mirrors the existing kernel nodes (loop m/t, `voxel_size_um=(z_step_um,
pixel_size_um,pixel_size_um)`, squeeze a singleton-Z volume to 2-D per the `.md`). This is
the recommended next big item.

**B. Granule chain completion — ✅ DONE 2026-07-25 (see top banner + `V2.07`).** Shipped as
the GENERAL **`analysis.tessellate_volume`** (fused `granule_tessellate` + `granule_volume_mask`;
mesh stays internal — not a v2 domain): labeled Points → a Voxel **Label** raster + a per-label
geometry table (centroid + analytic volume_um3/density/n_points/surface_area). `merge_tol=0`
(honors input labels — general), 3D-only, chains into the general `boundary_band`/`measure`.
Verified with synthetic Points+labels. Upstream `analysis.cluster_points` (ex-`granule_cluster`)
is still **sklearn-gated**, so the live particles→cluster hand-off waits on scikit-learn, but the
node is complete. Mesh domain deferred (V2.07 §2).

**C. Remaining v1 kernel ports — ✅ `track_objects` DONE 2026-07-27 (see top banner).**
Shipped as **`track.objects`**: all five linkers behind a `method` Mode, Label **or** Point
members, per-`(m,c,z)` independent runs, a Track table + a `track_id` write-back on the member
layer, `track.link`-identical membership conventions. Of the supplied set only `checkpoint`
stays unported — a workflow-IO concern (§D), not an analysis kernel. The `.md` contracts +
vendored `.py` for every kernel remain in `nodegraph/kernels/`.

**D. Not in the supplied kernel set — still legacy-only** (run on `--legacy`): cell-tracker
spatial maps, DIC mesh refinement/register distinct nodes, enhancement extras (bleach/
blob-subtract/spatial-flatness/temporal-fold), interaction/workflow IO (mask3d/review/
exclude/dismiss/pause/prism/send_to_results/save_data/export/checkpoint), `logic:if_else`.
See `V2.05` §2 for the full mapping.

**E. GUI refinements (Phase 6+).** ~~Track-trajectory viewer overlay (needs a membership↔
position join across t)~~ **✓ DONE 2026-07-26** (viewer "Tracks" checkbox — per-track hued
polylines joined member→position by id, current-T vertex highlighted; phase5-probe covered).
Still open: the optional **QRhi** GPU viewer path (only if the CPU composite
can't hold 60 fps, with a `maxTextureSize()` tiling guard — a 6554² plane is not one GL
texture); ~~reroute dots + labelled frames~~ **✓ DONE 2026-07-26** (reroute = hidden
`rr.reroute` pass-through, compact dot, double-click-a-wire to insert; frames = GUI-only
`FrameItem` grouping, Ctrl+J, `ui`-serialized); ~~**Group creation** in the GUI~~ **✓ DONE
2026-07-26** (`make_group`/`ungroup` = instance-node + stored `Group` def, expanded at
run/propagate via `groups.expand`, Ctrl+G / Ctrl+Shift+G); a v1→v2 `.nd2s_pipeline.json`
importer (greenfield — none built).

**F. C1 slivers (V2.04 §6b, deferred):** ~~`util.stack` tree-reduce~~ **✅ DONE 2026-07-27**
(`TReduceProvider`); ~~resample/normalize/drift lazy units~~ **✅ DONE 2026-07-27** (normalize
plane/volume/series + drift eager-estimate/lazy-apply via `MapComputeProvider`/`VolumeComputeProvider`;
resample via new `PlaneRealizeProvider`); ~~the kernel-param field gate~~ **✅ DONE 2026-07-27**
(`SocketSpec.kernel_param` + `_map_image` non-Const-field → plane-unit downgrade); ~~a Memo GC~~
**✅ DONE 2026-07-26** (byte-budget LRU `Memo(budget_bytes=…)`, per-blob refcount, correctness-safe
eviction, `_last_fp` kept; GUI persistent memo caps at 1 GiB via `Engine(memo_bytes=…)`; selftest
group `memo GC`; V2.04 §6b addendum). **Computed-provider pyramids: DEFERRED**
(user decision 2026-07-27) — `StreamProvider.levels=1` stays; the Viewer stride-decimates full-res.
Downsample-then-compute is wrong for non-linear ops; compute-then-downsample buys no savings; the
GPU-viewer + prefetch path already covers viewer smoothness. **This closes the entire C1 §6b sliver
list.** Slivers selftest group: `streaming slivers` (V2.04 §6b addendum 2026-07-27).

---

## 4. Adding a node — the workflow (used for the kernel ports)

Use the **`build-node-v2`** skill (procedure) + **`wire-node-v2`** (concepts). In short: a
node = `register_node(compute, op_key=..., **spec)` in `nodegraph/nodes.py`. Declare per-dim
`granularity`/`kernel_axes`; unit/derive on every spatial/temporal param; a `meta_transform`
if it changes axes/calibration; read calibration via `ctx.calib` (recorded → memo-fenced),
convert with `to_pixels_v2`; wrap a realized array in `ArrayProvider`, a Voxel layer via
`with_layer(Domain.VOXEL, ...)`, structure via `with_structure(table)`. Add an end-to-end
selftest pull (finite/correct; 2D vs 3D distinct recipe hashes; axis-changing payload axes ==
`engine.env(node).axes`; the right calibration key in `dict(engine.entry(node).reads)`).

**Porting a v1 kernel specifically** (the pattern the 6 new nodes follow): the kernel lives
in `nodegraph/kernels/<module>.py`, **lazily imported inside the compute** (keeps the engine
importable without the heavy dep); the `nodegraph/kernels/<module>.md` is the authoritative
contract — **read it before wiring**. The node owns the m/t/c loop; the kernel acts on one
frame/volume. **Load-bearing gotchas (recurring):**
- **`voxel_size_um` is ALWAYS `(dz, dy, dx)` slowest-first** = `(z_step_um, pixel_size_um,
  pixel_size_um)`. A `(dx,dy,dz)` swap silently corrupts anisotropy + µm columns.
- Bead detect's **2D fallback puts z in column 0** (=0), y,x in cols 1,2 — take `pts[:,1:3]`
  and stamp the true plane index.
- Histogram threshold is **2D-only, needs INTEGER input** (the node casts rint+clip→uint16
  and **raises** on a normalized [0,1] float) and `voxel_size=(dy,dx)` a **2-tuple** (it
  multiplies ALL elements → a 3-tuple folds z into area).
- Registration shifts are **`(row,col)=(y,x)`**; estimate on ref_z=z//2 of ref_channel, then
  apply the SAME bundle to every c,z (register-once/apply-all preserves colocalization).
- Granule kernels flip **`(z,y,x)` voxel → `(x,y,z)` micron** internally, driven by
  `voxel_size_um` — do NOT pre-convert points to µm.
- A **dep-gated** kernel registers a stub via `_gated_compute(kernel, package)` (nodes.py) —
  a real node the compute of which raises a clear `ImportError` until the dep is installed.

---

## 5. Working norms (observed) + gotchas

- **Grill before big design; present before locking.** For big autonomous decisions the user
  reserves (e.g. Phase-7 retirement, C1 forks, the DVC representation), **ASK** (one question
  at a time, with a recommendation). Keep `V2.00` canonical + add dated addenda (V2.01–V2.05).
- **Rigor + adversarial verification.** After building, run a review workflow (fan-out lenses
  → per-finding adversarial verify) and fix confirmed findings with regression guards. It has
  repeatedly caught real bugs the happy-path missed (C1 staleness, GUI data-loss, kernel-port
  hardening). **Fallbacks that have worked when a workflow stalls/session-limits:** self-review
  with direct `python` probes covering the same lenses; the classifier/agents can flap — retry,
  or continue with read-only work. Don't block on a stuck workflow.
- **Parallel subagents only when non-conflicting** — each owns ONE new file / reads one doc;
  the orchestrator does the shared-file edits (`nodes.py` registration, `selftest.py` wiring)
  sequentially. (The 14 kernel contracts were extracted this way in parallel.)
- **Test/demo fixtures MUST use fake op_keys** (`test.*`/`io.*`/`eng.*`) — a fixture
  `define_node`-ing a real op_key clobbers it in the global `NODES` registry → order-dependent
  "passes alone, fails in suite". Also: a test must NOT poke `runner._providers[("synthetic",)]`
  (corrupts the live io.load source resolution) — use a throwaway `EngineRunner` + fake keys.
- **Memo/streaming invariants:** identity is the monotonic `revision`, never `id()`, never a
  content hash for lookup. A streaming provider fp MUST fold declared calibration reads +
  field expr hashes or the tile cache serves stale tiles. Provider fingerprints are FLAT digest
  strings (nested tuples blow up `_canon` on deep unrolled chains). Construct a streaming
  provider LAST in a compute (reads recorded after construction don't enter its fp).
- **Never `pip install` unprompted; re-verify a backend's signature in-env before writing.**

---

## 6. THE decision that gates the most work

**DVC (`aldvc_field`) representation + ref/def pairing** (§3.A) — grill the user before
wiring: (i) how the node consumes the ref+def volume pair, (ii) how a per-voxel displacement
vector field is represented in v2 (3 Voxel scalar layers vs a new type). Recommendation to
open with: consecutive-timepoint pairing (t→t+1) with three `disp_{z,y,x}` Voxel µm layers +
strain layers — but present, don't lock. Everything else in §3 is a mechanical port or a GUI
refinement that needs no cross-cutting decision.

---

## 7. Key files

- **Design (canonical + dated addenda):** `V2.00_nodegraph_blender_revamp.md` (11 locked
  decisions), `V2.03_directive_metadata_toggle.md` (per-edge metadata + 2D/3D lever),
  `V2.04_c1_streaming_eval.md` (C1, LOCKED+implemented), `V2.05_phase7_capability_matrix.md`
  (Phase-7 coexistence + the v1↔v2 capability matrix + remaining-gap list),
  `V2.02_phase2a_spec.md`, `V2.01_data_io_research.md`, `../Architecture/node_backend_reference.md`.
- **Prior handoffs:** `CONTEXT_v2_handoff_2026-07-22.md`, `CONTEXT_v2_remaining.md` (per-area
  ✓/deferred map), `CONTEXT_nodegraph_v2_handoff.md` (dated running build log).
- **Code:** `nodegraph/` (24 core + `kernels/` 15 vendored), `nodelab_v2/` (18 GUI + ingest/ops),
  `run.py` (v2 default | `--legacy`), `scripts/_*.py` (parity, `python -m nodegraph.selftest`,
  benchmarks, ingest smoke, `_nodelab_v2_phase5_probe.py`, `_nodelab_v2_shot.py`).
- **Kernel contracts (read before porting one):** `nodegraph/kernels/<module>.md` — the
  authoritative integration contract for each of the 14 supplied kernels (entry point, array
  shapes, units, gotchas, provenance).
- **Skills:** `.claude/skills/{build-node-v2, wire-node-v2}` (active v2), `{build-node, wire-node}`
  (legacy v1), `grilling`.
