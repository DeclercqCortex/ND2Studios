# ND2Studios Architecture

**Version:** V1.23

## System Overview

ND2Studios is a PySide6 desktop application for the McGhee Lab. It loads
Nikon ND2 files (and TIFF stacks), surfaces their metadata, lets the user
build a linear processing recipe with a trial/accept/reject pattern, and
writes three kinds of output: per-channel Z-projection TIFF stacks,
multi-channel RGB composite TIFFs, and time-lapse movies (MP4 / GIF) with
configurable scale bar / timestamp / channel-label overlays.

It is intentionally narrower than CellTracker: there is no segmentation,
no tracking, and no cell-level analysis. ND2Studios is a *sibling* of
CellTracker — code that turned out to be reusable was copied verbatim
(ND2 I/O, enhancement plugins, image viewer, scale bar) rather than
imported, so the two packages can evolve independently.

## High-level layout

```
ND2Studios/
├── run.py                   # launcher
├── nd2studios/              # the package
│   ├── __main__.py          # Qt setup + theme + MainWindow
│   ├── core/                # app state and chrome
│   ├── widgets/             # Qt-aware reusable widgets
│   ├── backend/             # pure Python data layer (no Qt imports)
│   ├── plugins/enhancement/ # registered enhancement plugins
│   ├── workers/             # QThread workers
│   └── pages/               # the three GUI pages
├── CodeLog/                 # plans, changelog, architecture, refs
├── Research/                # literature reviews per method
├── hpc/                     # HPC integration (unused in V1.0)
└── scripts/                 # version_push.py
```

## Module Breakdown

### `nd2studios/__main__.py` — entry point
Sets `QT_FONT_DPI`, `SSL_CERT_FILE` (via `certifi`), instantiates
`QApplication` with the Fusion style + Dracula stylesheet, force-imports
`nd2studios.plugins.enhancement.builtin`,
`nd2studios.backend.analysis.nuclei_segmentation`,
`nd2studios.backend.analysis.tear_detection`,
`nd2studios.backend.analysis.histogram_threshold_pipeline`,
`nd2studios.backend.analysis.spots_pipeline`, and
`nd2studios.backend.analysis.manual_mask` so both the plugin registry
and the analysis pipeline registry are fully populated before any page is
built, opens `MainWindow`.

### `core/`

| File | Purpose |
|---|---|
| `settings.py` | App constants (window dims, sidebar widths, animation duration, drop-shadow params), Dracula palette, `PAGES` list, `STATUS_ORDER`, `PAGE_PREREQS` |
| `theme.py` | Single QSS stylesheet adapted from CellTracker + PyDracula. Chrome rules (`#bgApp`, `#titleBarWidget`, `#navBtn`, `#sessionBtn`, `#toggleBtn`) coexist with form/button/list/dialog rules ported from CellTracker |
| `main_window.py` | Frameless `QMainWindow`. Builds: outer container with 10 px margin (room for drop shadow) → `#bgApp` card with `QGraphicsDropShadowEffect` → custom title bar (drag, double-click maximize, Min/Max/Close) → body `QSplitter(Qt.Horizontal)` of sidebar + content area (draggable handle). Sidebar collapses 60↔240 px via `QPropertyAnimation(InOutQuart)` on `minimumWidth`/`maximumWidth`; after expansion the max-width cap is removed so the splitter handle can grow the sidebar past 240 px. `_on_sidebar_anim_done()` calls `_reset_active_viewer_zoom()` to fit the image after every toggle. Content area has top bar (`titleLabel` + `StatusIndicator`), `QStackedWidget`, bottom bar (version label + `QProgressBar` + status text). Edge resize via 4 `CustomGrip`s. Navigation guards consult `Settings.PAGE_PREREQS` |
| `experiment_manager.py` | `ND2StudiosRecord` carries configs (`import`, `recipe`, `export`), metadata snapshot (`nd2_metadata`), `channel_display` (`{name: {enabled, color}}`), the recipe (`List[(plugin_name, params)]`) and `recipe_normalized` flag, frame counts + pixel size, plus *non-serialized* in-memory caches: `_raw_channels`, `_processed_channels`, `_frame_timestamps`. `ND2StudiosManager` is a `QObject` exposing `active_changed` / `status_changed`. `save_session()` writes JSON manifest + `_arrays.npz` companion (`raw_*`, `proc_*`, `frame_timestamps`); `load_session()` is the inverse |
| `plugin_registry.py` | `ParamSpec` (declarative parameter), `PluginBase` (decorator-based registry), `EnhancementPlugin` base for `(T,H,W)` array transforms |
| `analysis_registry.py` (V1.19) | `AnalysisResult` dataclass (`label_masks`, `measurements`, `summary`), `AnalysisPipeline` ABC with separate `_registry` (distinct from `PluginBase`). Pipelines take full multi-frame channel dicts and return structured results rather than transformed arrays |

### `widgets/` — Qt-aware reusable building blocks

| File | Purpose |
|---|---|
| `common.py` | `MplCanvas` (Dracula matplotlib figure), `ParamEditor` (auto-form from `List[ParamSpec]`), `StatusIndicator` (colored status badge) |
| `image_viewer.py` | `ImageCanvas` (zoom/pan), `ZoomToolbar`, `ImageViewer` (single- or multi-channel RGB compositing with per-channel color LUTs and enable/disable). Also exports `CHANNEL_COLORS`, `frame_to_uint8`, `apply_lut_color`, `composite_channels`. V1.30: `ImageCanvas` gained mutually-exclusive draw modes (`set_draw_mode("rect"|"ellipse"|"polygon"|None)`) and a `shape_drawn(mode, vertices)` signal; rectangle/ellipse are rubber-band drag, polygon samples vertices along a freehand path and auto-closes on release. The in-progress shape paints in cyan dashed lines to distinguish it from the yellow crop rubber-band. |
| `multi_axis_viewer.py` (V1.1, restructured V1.2, V1.17, hook V1.20) | `ChannelChip` (compact toggle + color combo only — LUTs moved to the sidebar in V1.2); `MultiAxisViewer` with a `QSplitter(Qt.Horizontal)` root: left column (image canvas + zoom toolbar + M/T/Z sliders + chip strip) and right column (`LutSidebar`) — both panes are draggable. Each M/T/Z axis row includes a `QPushButton` play/pause toggle and a `QDoubleSpinBox` FPS control; per-axis `QTimer`s advance the slider on each tick (wraps to 0). `LutSidebar.collapse_changed` triggers `canvas.reset_zoom()`. M/T/Z sliders auto-hide when an axis is degenerate. Reads from a `LazyND2Volume` (M/Z scrolling, takes `stage_xy_um` for the tile overlay) or a flat `Dict[name, (T,H,W)]` (recipe / processed view). V1.17: slider changes check `FrameCache` first (instant display on hit); on a miss a 50/80 ms debounce timer fires `_do_refresh()`. After each frame compose, `PrefetchManager.request_neighbors()` queues ±5 T neighbors. Histogram samples cached per `(m, z_mode)` to skip resample on M revisits. V1.20: `set_frame_post_process(fn)` hook — `fn(rgb_uint8, t, m) -> rgb_uint8` applied after compositing before `canvas.set_image()`; pass `None` to clear |
| `lut_histogram.py` (V1.1) | `LutHistogramWidget` — log-scaled intensity histogram with draggable lo/hi handles, gamma slider, Auto / Reset buttons. Samples ≤ 32 frames so it stays cheap on long timeseries. Module-level `apply_lut(frame, lo, hi, gamma) -> uint8` |
| `lut_sidebar.py` (V1.2, expanded V1.3) | `LutSidebar` — collapsible right-side panel hosting two independently-collapsible sections: (1) a **Tiles** section with a `TileLayoutWidget` in navigate mode, and (2) a **LUTs** section with one `LutHistogramWidget` per channel. Sidebar width 300 / 28 px collapsed; animated on both `minimumWidth` and `maximumWidth` (animating both prevents `QHBoxLayout` from fighting the animation). Section headers (`_SectionHeader`) have ▾/▸ toggles. Tiles section auto-hides when ≤ 1 tile. Signals: `channel_contrast_changed(name, lo, hi, gamma)`, `tile_navigate_requested(m)`, `collapse_changed(bool)`. `update_swatch(name, rgb)` keeps the swatch in sync when the chip strip's color combo changes |
| `tile_layout.py` (V1.3, auto-sized V1.6) | `TileLayoutWidget` — interactive multipoint layout with two modes (`"navigate"` / `"select"`), `QPainter` rendering, hit-testing in cached widget rects, hover preview, tile-index labels, and an expand-to-modal `⤢` button. V1.6: `_adapt_minimum_size()` runs after every `set_tile_layout()` and sets `setMinimumSize(n_cols × 50 + margins, n_rows × 50 + header + margins)` so the host's `QScrollArea` can scroll when the natural size exceeds the viewport. Three signals: `navigate_requested(m)`, `selection_changed(set[int])`, `expand_requested`. `TileLayoutDialog` (V1.6: wraps the widget in its own `QScrollArea`) is the modal "expand" wrapper — auto-closes after click in navigate mode, stays open in select mode |
| `scale_bar.py` | Matplotlib scale-bar rendering helpers (used by `MplCanvas` consumers) |
| `video_player.py` | Play/Pause/FPS playback controls |
| `custom_grips.py` | Per-edge frameless-window resize widgets. Lean reimplementation of PyDracula's `CustomGrip` |
| `export_preview_dialog.py` (V1.32) | `ExportPreviewDialog(channels, colors, enabled, pixel_size_um, frame_timestamps_s, lut_settings, movie_options, title)` — modal preview shown after the user confirms movie or image-sequence export settings. Embeds an `ImageCanvas` + `ZoomToolbar` with a T scrubber, a ▶ Play / ⏸ Pause toggle, and an in-dialog FPS spin box; a second `QTimer` advances T at the chosen FPS so the user sees the movie actually play back with brightness / contrast / saturation / hue / fade applied live. Manually grabbing the T slider pauses playback. Exposes the five adjustment sliders (each a `_LabeledSlider`) and a "Show overlays in preview" toggle. A 30 ms single-shot debounce coalesces slider drags before recompositing; the play-tick path renders directly to keep the frame rate honest. `adjustments()` returns the user-chosen `ImageAdjustments` once the dialog is accepted; the dialog itself never runs the export. `closeEvent`/`done()` stop the playback timer so it can't outlive the dialog |

### `backend/` — pure-Python data layer (no Qt imports)

| File | Purpose |
|---|---|
| `nd2_loader.py` | `ND2Metadata` dataclass, `read_nd2_metadata`, `read_nd2_metadata_extended` (returns dict with full metadata: T/Z/C/P, pixel size, z step, channel names + colors + emission/excitation + exposures, objective + camera + microscope, binning, frame timestamps, **per-M stage XY/Z** — tries `f.experiment` XYPosLoop first, falls back to `frame_metadata()` per-M; `stage_layout_source` includes method suffix e.g. `"stage_xy:experiment"`, loops), `load_nd2_timeseries`, `LazyND2Channel` (numpy-protocol on-demand frame proxy with `__getitem__`, `materialize`, `crop`), `load_nd2_timeseries_lazy`, `z_project` |
| `nd2_volume.py` (V1.1) | `LazyND2Volume` exposing `(M, T, Z, H, W)`. `get_frame(c, m, t, z, z_mode, z_start, z_end)` returns a single (H, W) array on demand. `to_lazy_channel(c, m, z_mode, z_index, ...)` collapses to a V1.0 `LazyND2Channel` so the recipe pipeline keeps the `(T, H, W)` contract |
| `tiff_loader.py` (V1.17, multi-Z multi-C support V1.30) | `get_tiff_info`, `load_tiff_stack`, `LazyTIFFChannel` (lazy `(T,H,W)` view; `n_pages_per_t`/`page_within_t` params enable per-channel access; V1.17: caches `tifffile.TiffFile` handle on first read via `_ensure_open()`; V1.30: new `z_stride` param so Z accumulator reads `base_page + z * z_stride` — `z_stride=n_c` for ImageJ TZCYX hyperstacks, `z_stride=1` for single-channel multi-Z), `load_tiff_stack_lazy`, `read_imagej_tiff_metadata`, `load_imagej_tiff_channels(z_projection="max")` (V1.30: passes `n_z`/`z_stride=n_c`/`z_projection` so multi-channel multi-Z files actually project Z), `_SingleFileTIFFView` (V1.30: `get_frame` reads pages directly for multi-Z files honoring `z`/`z_mode`; `to_lazy_channel` builds proxies that honor `z_mode`/`z_index`/Z range; caches a `tifffile.TiffFile` for direct page reads), `LazyMultiFileTIFFVolume` (V1.30: `to_lazy_channel` propagates `z_mode`/`z_index`/`z_*`/`t_*` through to the underlying view), `assign_dimensions`, `extract_2d_timeseries` |
| `frame_cache.py` (V1.17) | `FrameCache`: thread-safe LRU cache keyed by `(c, m, t, z, z_mode)` storing normalized `(H, W)` frames. `get`/`put`/`contains`/`clear` all lock-protected. Default budget 300 MB; evicts oldest entries on overflow |
| `normalization.py` | `normalize_frame_means` (frame-mean intensity normalization) |
| `recipes.py` | `build_recipe`, `save_recipe`, `load_recipe` for `.nd2s_recipe.json` (kind = `nd2studios.recipe`) |
| `exporters/tiff_exporter.py` | `export_tiff_stack(stack, filepath, bit_depth, pixel_size_um, progress_cb)` (single-channel `(T,H,W)` or `(T,Z,H,W)`, used by label-mask export); `export_tiff_hyperstack(channels, enabled, filepath, bit_depth, pixel_size_um, progress_cb)` (V1.29 — multi-channel, writes a single `(T,Z,C,H,W)` ImageJ TZCYX hyperstack with `Labels=[<channel names>]`, `unit=um`, `spacing`, resolution; same file construction as `export_stitched_tiff`). BigTIFF auto-switch |
| `exporters/composite_exporter.py` | `export_rgb_composite_tiff(channels, colors, enabled, filepath, …, image_adjustments)`, `_composite_frame(…, image_adjustments)`, `CHANNEL_COLORS`. V1.32: `ImageAdjustments` dataclass (`brightness`, `contrast`, `saturation`, `hue`, `fade`); vectorised `apply_image_adjustments(rgb, adj)` for contrast / saturation / hue / fade on the final RGB; `_apply_brightness_to_gray(gray, brightness)` applies brightness as **multiplicative gain** (`gain = 1 + brightness/100`) inside the composite loop *before* the channel colour LUT is mixed in. The gain semantic keeps background pixels near zero (only signal scales visibly — additive offset would lift the noise floor as much as the peaks) and the per-channel placement keeps a red-only channel pure red rather than washing toward white. `ImageAdjustments.is_identity_post_composite()` lets the composite pass skip the post-RGB pass when only brightness is set |
| `exporters/movie_exporter.py` | `MovieOptions` dataclass + `export_movie(channels, colors, enabled, filepath, options, pixel_size_um, frame_timestamps_s, lut_settings, image_adjustments, progress_cb, status_cb)`. Uses `imageio` (libx264 for MP4, pillow for GIF). PIL-rendered overlays for scale bar, timestamp, channel labels |
| `exporters/image_sequence_exporter.py` (V1.32) | `format_frame_name(basename, t, m, z, n_t, n_m, n_z, extension)` — pure naming helper that only includes axis suffixes (`_T01`, `_M02`, `_Z03`) when the corresponding axis has more than one frame. `ImageSequenceRequest` dataclass; `export_image_sequence(request, progress_cb, status_cb)` writes one PNG per frame via PIL. Two iteration paths: in-memory channels (T-only, recipe-applied) or `LazyND2Volume` (M × T × Z, raw — no recipe). Reuses `_composite_frame` and `_draw_overlays` so the colour pipeline matches movie exports byte-for-byte |
| `analysis/nuclei_segmentation.py` (V1.19) | `NucleiSegmentationPipeline` — Cellpose 3 per-frame 2D segmentation. Optional dep (`pip install cellpose`); raises `ImportError` with install message when absent. Uses `skimage.measure.regionprops` for per-object measurements. Registered via `@AnalysisPipeline.register` |
| `analysis/tear_detection.py` (V1.19) | `TearDetectionPipeline` — classical dark/homogeneous region detection. Stage 1: tissue mask via Otsu on log-intensity + morphological cleanup (4× downsample). Stage 2: rank-normalised homogeneity score from local intensity (`uniform_filter`), local variance (`generic_filter`), local entropy (`skimage.filters.rank.entropy`). Stage 3: connected-component extraction with area filters. Stage 4: optional weak-stain disambiguation via counterstain channel. No DL dependencies. Measurements include `class` (`tear`/`weak_stain`/`unknown`), `homogeneity_score`, `solidity`, `eccentricity` in addition to the standard fields |
| `analysis/histogram_threshold_pipeline.py` (V1.20) | `HistogramThresholdPipeline` — adapter that registers `histothresh/` into the `AnalysisPipeline` registry. Iterates T frames, calls `HistogramThresholdSegmenter.run(frame)` per slice, stacks `(T,H,W)` int32 label arrays, and assembles `AnalysisResult`. 16 `ParamSpec` parameters exposed in the Analysis tab UI. Sets `bit_depth_strict=False` so TIFF inputs that overflow the declared range warn rather than crash |
| `analysis/histothresh/` (V1.20) | Qt-free subpackage implementing the histogram threshold segmenter. `validation.py`: `validate_bit_depth()`, `infer_bit_depth()`, `BitDepthError` — rejects float inputs and values overflowing the declared bit-depth max. `histogram.py`: `Histogram` dataclass with `percentile()` / `cumulative()` helpers; `compute_histogram()` bins the full LUT range (one bin per integer value); four `suggest_threshold_*()` helpers (Otsu, Triangle, Minimum, Multi-Otsu). `thresholds.py`: `threshold_single()` (below/above/between/outside), `threshold_hysteresis()` (inverts image for `below` mode to reuse scikit-image's high-pass hysteresis), `threshold_percentile()`, `threshold_relative()`. `morphology.py`: `apply_spatial_constraints()` (opening → closing → hole-fill → small-object removal), `homogeneity_gate()`. `config.py`: `ThresholdConfig` plain dataclass with `__post_init__` validation of method/direction param combinations. `identifier.py`: `HistogramThresholdSegmenter` orchestrator returning `SegmentationResult` (mask, int32 labels, regions list, histogram, threshold_used dict, provenance dict) |
| `results_engine.py` (V1.22) | `compute_measurements(label_masks, channels, metadata, m_index)` → `List[Dict]` with area, centroids (px / µm / stage-absolute µm), perimeter, eccentricity, solidity, bbox, per-channel mean/std intensity. Uses `skimage.measure.regionprops`; no Qt. Also: `export_overlay_frames()` (uint8 RGB + cyan mask overlay, `imageio`), `export_label_masks_tiff()` (int32 stacks, `tifffile`) |
| `template.py` (V1.23) | `save_template()` / `write_template()` / `load_template()` for `.nd2st.json` pipeline templates (import, recipe, analysis, results configs without data). `TEMPLATE_EXTENSION = ".nd2st.json"` |
| `analysis/spots_pipeline.py` (V1.21) | `BrightDarkSpotsPipeline` — adapter that registers `spots/` into the `AnalysisPipeline` registry. Iterates T frames, calls `BrightDarkSpotsSegmenter.run(frame)` per slice, stacks `(T,H,W)` int32 label arrays, and assembles `AnalysisResult` with extra measurement columns (`diameter_px`, `contrast_score`, `circularity`, `polarity`). 11 `ParamSpec` parameters. `intensity_percentile=0.0` maps to gate disabled. Sets `bit_depth_strict=False` |
| `analysis/spots/` (V1.21) | Qt-free subpackage implementing the GA3-style scale-space spot detector. `validation.py`: re-exports `histothresh.validation` helpers + `validate_diameter()`. `scale_space.py`: `resolve_sigma()` (FWHM-matched sigma, Marr-Hildreth ratio), `log_response()` (σ²-normalised LoG, positive for bright), `dog_response()` (DoG band-pass, positive for bright). `detection.py`: `find_extrema()` (peak_local_max on normalised response, intensity-gate suppression, contrast gate), `h_transform_seeds()` (h-maxima / h-minima). `symmetry.py`: `SYMMETRY_FLOOR` dict, `circularity()`, `filter_by_symmetry()` (GA3 All/More/Medium/Less bins). `grow.py`: `grow_seeds_dilation()` (disk morphology), `grow_seeds_watershed()` (watershed-from-marker; negated image for bright). `config.py`: `SpotsConfig` dataclass with `__post_init__` validation. `identifier.py`: `BrightDarkSpotsSegmenter` orchestrator (12-step: validate → sigma → DoG/LoG → normalise → intensity mask → peak detection → label rasterisation → symmetry gate → grow → re-label → regionprops → provenance); `SpotsResult` dataclass |
| `exporters/stitch_exporter.py` (V1.1, layout rewritten V1.14, export format V1.15, metadata aligned V1.29, Z-preserving V1.30) | `StitchLayout` dataclass; `compute_tile_layout(stage_xy_um, pixel_size_um, tile_h, tile_w, m_indices)` converts stage XY µm to pixel offsets directly: `offset_x = round((sx - min_x) / pixel_size_um)`, `offset_y = round((max_y - sy) / pixel_size_um)` (Y flipped). Canvas = bounding box of all tile corners. Physical gaps between non-adjacent tiles appear as empty pixels. `source = "physical"`. Grid fallback when no stage XY data. Helper functions `_cluster_axis`, `_assign_index`, `_serpentine_layout` retained but not in the main path. `stitch_one_frame(tile_frames, layout, dtype)`, `export_stitched_tiff(volume, layout, m_indices, channel_indices, channel_colors, filepath, ...)` writes a `(T, Z, C, H, W)` ImageJ TZCYX hyperstack with `Labels=[<channel names>]`, `unit=um`, `spacing=pixel_size_um`, resolution. `z_mode="none"` on a multi-Z file keeps every Z plane (`Z = volume.n_zslices`); projection modes and single-Z files collapse to `Z=1`. File construction matches the Export page's `export_tiff_hyperstack` exactly so the two writers produce identical headers; re-importable into ND2Studios as a multi-channel multi-Z TIFF |

### `plugins/enhancement/builtin.py`
19 enhancement plugins (Normalize, CLAHE, GaussianBlur, MedianFilter,
BackgroundSubtract, GammaCorrection, BleachCorrection,
TemporalFoldCorrection, SpatialFlatness, TopHat, DoG, UnsharpMask,
BilateralDenoise, MorphGradient, LocalContrast, BlobSubtract,
NLMDenoise, WaveletDenoise, TVDenoise) registered via
`@PluginBase.register`. Each declares parameters via `ParamSpec` so
`ParamEditor` can auto-generate a form for it.

### `workers/`

| File | Purpose |
|---|---|
| `base_worker.py` | `BaseWorker(QThread)` with `progress(int)`, `status(str)`, `finished(object)`, `error(str)` signals and `cancel()` |
| `load_worker.py` | `LoadWorker` — opens an ND2 (extended metadata + lazy channels) or TIFF (single-channel TYX) file off the GUI thread |
| `recipe_worker.py` | `RecipeWorker` — applies an ordered recipe to each channel; lazy proxies are materialized once per channel; optional frame-mean normalization first |
| `export_worker.py` | `ExportRequest` dataclass + `ExportWorker` dispatching to `tiff_stack`, `tiff_zstack`, `rgb_composite`, `movie`, or `image_sequence` (V1.32) mode. `ExportRequest` carries an optional `image_adjustments: ImageAdjustments` shared across modes, plus `basename` / `iterate_volume` / `z_mode` / `z_view_index` for image-sequence jobs |
| `stitch_worker.py` (V1.1) | `StitchRequest` + `StitchWorker` wrapping `export_stitched_tiff` with progress / status signals |
| `analysis_worker.py` (V1.19) | `AnalysisWorker(BaseWorker)` — runs any `AnalysisPipeline.run()` in a background thread; passes `progress_cb` and `cancelled_cb` (Qt-free lambda) |
| `batch_worker.py` (V1.23) | `BatchWorker(BaseWorker)` — sequential per-file pipeline runner. Extra signal `file_done(int, int)`. For each file: calls `LoadWorker.run_task()` → `RecipeWorker.run_task()` → `pipeline.run()` → `results_engine.compute_measurements()` synchronously within the thread. Errors per file are caught and reported without aborting. Writes `batch_results.csv` to output dir at end |
| `prefetch_worker.py` (V1.17) | `PrefetchManager(QThread)` — single background thread that fills a `FrameCache` with ±5 nearest-neighbor T (or Z) frames. Opens its own `LazyND2Volume` handle inside `run()`. Queue is replaced on each `request_neighbors()` call. Emits `frame_ready(c, m, t, z, z_mode)` via queued signal. Only wired up for the ND2 volume path in `set_volume()` |

### `pages/`

| File | Page | Responsibilities |
|---|---|---|
| `import_page.py` | Import | File browse, extended metadata table, **read-only** per-channel info rows, Z-mode + frame-stride config, multi-axis preview (M/T/Z sliders + per-channel toggle/color/LUT), Confirm Import (sets status `imported`), Stitch M… button → `StitchDialog` |
| `recipe_page.py` | Recipe | Plugin picker, parameter editor, trial/accept/reject pipeline, save/load `.nd2s_recipe.json`, side-by-side raw vs processed `MultiAxisViewer`s (T scrolling only on this page) |
| `export_page.py` | Export | Source selector (raw/processed), four tabs (TIFF / Composite / Movie / Image Sequence) each running the appropriate `ExportWorker` job. V1.32: clicking "Export Movie…" or "Preview & Export Image Sequence…" opens an `ExportPreviewDialog` (T scrubber + brightness / contrast / saturation / hue / fade sliders) before the worker is started. The Image Sequence tab adds a base-name line edit (auto-populated from the loaded filename) and an "Iterate all M / Z positions" checkbox (enabled only when the file actually has multi-M or unprojected multi-Z); filenames are `{basename}_T01[_M02][_Z03].png` |
| `analysis_page.py` (V1.19, layout V1.20) | Analysis | Horizontal splitter: left panel (pipeline selector + `ParamEditor` + run row + results widget), right panel (`MultiAxisViewer`). Viewer wired to exp on `on_activated()` so the file is immediately visible for parameter tweaking. After `AnalysisWorker` completes, label masks are overlaid via `MultiAxisViewer.set_frame_post_process()` (updates per T/M slider move). Results widget (summary + per-frame table + export buttons) hidden until first analysis completes or session restored |
| `results_page.py` (V1.22) | Results | Pipeline selector (populated from `exp.analysis_results` keys), "Compute Measurements" button triggers `results_engine.compute_measurements()` synchronously on the GUI thread. Measurements shown in a `QTableView` backed by a custom `QAbstractTableModel` with `QSortFilterProxyModel` for column sorting. Summary panel: n_objects, n_frames, mean/std area_um², dynamic per-channel mean intensity rows. Export row: CSV (`csv.DictWriter`), Label Masks TIFF (`export_label_masks_tiff`), Overlay Images TIFF/JPG (`export_overlay_frames`). Empty-state label shown until "Compute" is clicked |
| `batch_page.py` (V1.23) | Batch | Template path + browse/save controls; file queue (`QListWidget`, add files / add folder / clear / remove-selected); output directory picker; export-images checkbox + format combo; Run/Cancel buttons; progress bar + file counter + status label. Kicks off `BatchWorker`; on `finished` offers to open the output folder |
| `stitch_dialog.py` (V1.1) | dialog | Layout preview (matplotlib), per-tile / per-channel selectors, Z mode / Z slice picker, output path picker, kicks off `StitchWorker`. Launched from the Import page |

## Data Flow

```
ND2 / TIFF on disk
   │
   │  ImportPage._on_browse        → single file → LoadWorker
   │  ImportPage._on_reconstruct   → ReconstructDialog (V1.28)
   │     ↳ user picks N files + chain_axis ∈ {T, M, Z, C}
   │     ↳ LoadWorker(filepaths=…, chain_axis=…) → LazyMultiFile{ND2,TIFF}Volume
   │       (axis-agnostic; bisect-routed; each file contributes its
   │        native chain-axis size)
   ▼
ND2StudiosRecord._raw_volume     (LazyND2Volume | LazyMultiFileND2Volume | LazyMultiFileTIFFVolume)
ND2StudiosRecord._raw_channels   (Dict[name, LazyND2Channel | MultiFileLazyChannel | LazyTIFFChannel])    ← initial M=0
ND2StudiosRecord.nd2_metadata    (full extended metadata dict)
   │
   │  Viewer scrolls M/T/Z; user toggles channels; user drags LUT.
   │  ImportPage Confirm → rebuilds _raw_channels via
   │  LazyND2Volume.all_channels_as_lazy(m, z_mode, z_index)
   ▼
ND2StudiosRecord._raw_channels   (Dict[name, LazyND2Channel])    ← chosen M/Z
   │
   │  RecipePage crop drag/click → _apply_crop()
   ▼
ND2StudiosRecord._original_raw_channels  (Dict[name, lazy|ndarray])  ← pre-crop snapshot
ND2StudiosRecord._raw_channels           (Dict[name, lazy|ndarray])  ← cropped view
ND2StudiosRecord.crop_rect               (x, y, w, h) | None
   │  Re-crop always applies to _original_raw_channels (non-destructive).
   │  Reset Crop restores _raw_channels ← _original_raw_channels.
   │  Snapshot persists on the record; survives tab navigation for all sources.
   │
   │  RecipePage trial/accept → RecipeWorker
   ▼
ND2StudiosRecord._processed_channels  (Dict[str, ndarray])
ND2StudiosRecord.recipe              (List[(plugin_name, params)])
   │
   │  ExportPage tab → ExportWorker → backend/exporters/
   ▼
.tif (single multi-channel TZCYX ImageJ hyperstack, V1.29)
.tif (RGB composite) | .mp4 / .gif on disk

Side branch (Import page → Stitch M…):
   _raw_volume → StitchDialog → StitchWorker → export_stitched_tiff()
   → multi-page TIFF (single-channel or RGB composite). The original
   ND2 is never touched.

Side branch (Analysis page — V1.19):
   _processed_channels (or _raw_channels) → AnalysisPage._on_run()
   → materialized ndarray → AnalysisWorker → AnalysisPipeline.run()
   → AnalysisResult {label_masks, measurements, summary}
   → exp.analysis_results[pipeline_name] = result
   → page overlay renderer + summary table
   → "Export Label Masks" → export_tiff_stack(..., bit_depth="passthrough")
   → "Export Measurements" → csv.DictWriter → .csv
```

## External Dependencies

| Package | Purpose | Notes |
|---|---|---|
| PySide6 | GUI framework | Qt 6 bindings |
| numpy, scipy, pandas | numerical core | core array ops |
| nd2 | ND2 I/O with full metadata | Talley Lambert's library |
| tifffile | TIFF stack export | per-page write, BigTIFF |
| scikit-image, opencv-python, PyWavelets | image processing | used by enhancement plugins |
| matplotlib | plots, scale bars | embedded via `MplCanvas` |
| imageio + imageio-ffmpeg | MP4 / GIF export | libx264 codec for MP4 |
| Pillow | overlay rendering | scale bar / timestamp / labels |
| certifi | SSL bundle | optional, for plugins that pull online resources |

| cellpose (optional, V1.19) | Nuclei segmentation | `pip install cellpose`; detected at runtime via `importlib.util.find_spec`. Absent = friendly error dialog, not a crash |

ND2Studios deliberately does **not** depend on TensorFlow, StarDist,
csbdeep, or any tracking library — those belong in CellTracker.
Cellpose is an optional analysis dependency; the rest of the app runs
without it.

## Design Decisions

1. **Sibling, not fork.** ND2Studios copies backend code from
   CellTracker rather than importing it. Keeps the two packages
   independent at the cost of a small duplication tax. If duplication
   passes ~30%, extract a shared `mcghee_lab_imaging` package.
2. **Backend purity.** Every module under `nd2studios/backend/` and
   `nd2studios/plugins/` is Qt-free. They take data + params + an
   optional `progress_cb`; nothing else. This makes them trivially
   testable and trivially callable from a future headless mode.
3. **`LazyND2Channel` everywhere on import.** ND2 files are routinely
   tens of GB. The viewer pulls one frame at a time. Materialization
   only happens when a worker needs a contiguous array (recipe
   application, export).
4. **Trial / Accept / Reject.** The committed recipe is the source of
   truth. A trial step runs `committed + trial` against the raw data
   and parks the result on `_processed_channels`; Accept appends the
   trial step to the recipe; Reject reverts by re-running just the
   committed recipe.
5. **PyDracula chrome ported into Python.** No `.ui` files. The
   collapsible animated sidebar with text labels and the custom
   frameless title bar are reimplemented as plain `QWidget`s, matching
   CellTracker's authoring style.
6. **Single dark theme.** Theme switching is out of scope for V1.0. The
   QSS is structured (palette via `Settings.*` constants, `#bgApp`
   wrapper) so a light theme can be added later without restructuring.
7. **`.nd2s` session = JSON manifest + `_arrays.npz`.** Mirrors
   CellTracker's `.cta`. Channels and frame timestamps are stored as
   compressed NPZ; everything else is human-readable JSON.
8. **Movies via `imageio`.** Avoids OpenCV's licensing complications
   for video encoding. `imageio-ffmpeg` is a binary distribution that
   ships its own `ffmpeg`.

## Known limitations / V1.1+ candidates

- TIFF loader only handles single-channel TYX. Multi-channel TIFFs
  need a `dim_order` picker in the Import page.
- No autosave. CellTracker's autosave is keyed to expensive detection
  runs; our recipes are fast enough that explicit Save/Load suffices.
- Theme switching is intentionally absent.
- Headless / HPC mode is wired into the directory tree (`hpc/`,
  `scripts/`) but not exposed on the GUI; `recipes.py` is already
  designed to be applied from a headless runner.
- Channel-aware multi-channel TIFF input (most multi-channel TIFFs
  follow `TCYX` or `TZCYX`).
