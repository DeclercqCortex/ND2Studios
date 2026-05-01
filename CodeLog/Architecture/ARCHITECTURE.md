# ND2Studios Architecture

**Version:** V1.1

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
`nd2studios.plugins.enhancement.builtin` so the plugin registry is
populated before the Recipe page is built, opens `MainWindow`.

### `core/`

| File | Purpose |
|---|---|
| `settings.py` | App constants (window dims, sidebar widths, animation duration, drop-shadow params), Dracula palette, `PAGES` list, `STATUS_ORDER`, `PAGE_PREREQS` |
| `theme.py` | Single QSS stylesheet adapted from CellTracker + PyDracula. Chrome rules (`#bgApp`, `#titleBarWidget`, `#navBtn`, `#sessionBtn`, `#toggleBtn`) coexist with form/button/list/dialog rules ported from CellTracker |
| `main_window.py` | Frameless `QMainWindow`. Builds: outer container with 10 px margin (room for drop shadow) → `#bgApp` card with `QGraphicsDropShadowEffect` → custom title bar (drag, double-click maximize, Min/Max/Close) → body row of sidebar + content area. Sidebar collapses 60↔240 px via `QPropertyAnimation(InOutQuart)` on `minimumWidth`/`maximumWidth`. Content area has top bar (`titleLabel` + `StatusIndicator`), `QStackedWidget`, bottom bar (version label + `QProgressBar` + status text). Edge resize via 4 `CustomGrip`s. Navigation guards consult `Settings.PAGE_PREREQS` |
| `experiment_manager.py` | `ND2StudiosRecord` carries configs (`import`, `recipe`, `export`), metadata snapshot (`nd2_metadata`), `channel_display` (`{name: {enabled, color}}`), the recipe (`List[(plugin_name, params)]`) and `recipe_normalized` flag, frame counts + pixel size, plus *non-serialized* in-memory caches: `_raw_channels`, `_processed_channels`, `_frame_timestamps`. `ND2StudiosManager` is a `QObject` exposing `active_changed` / `status_changed`. `save_session()` writes JSON manifest + `_arrays.npz` companion (`raw_*`, `proc_*`, `frame_timestamps`); `load_session()` is the inverse |
| `plugin_registry.py` | `ParamSpec` (declarative parameter), `PluginBase` (decorator-based registry), `EnhancementPlugin` base for `(T,H,W)` array transforms |

### `widgets/` — Qt-aware reusable building blocks

| File | Purpose |
|---|---|
| `common.py` | `MplCanvas` (Dracula matplotlib figure), `ParamEditor` (auto-form from `List[ParamSpec]`), `StatusIndicator` (colored status badge) |
| `image_viewer.py` | `ImageCanvas` (zoom/pan), `ZoomToolbar`, `ImageViewer` (single- or multi-channel RGB compositing with per-channel color LUTs and enable/disable). Also exports `CHANNEL_COLORS`, `frame_to_uint8`, `apply_lut_color`, `composite_channels` |
| `multi_axis_viewer.py` (V1.1) | `ChannelControlRow` (enable / color / embedded `LutHistogramWidget`); `MultiAxisViewer` with M/T/Z sliders auto-hiding when an axis is degenerate. Reads from a `LazyND2Volume` (M/Z scrolling) or a flat `Dict[name, (T,H,W)]` (recipe / processed view). Composites with manual `apply_lut(frame, lo, hi, gamma)` per channel |
| `lut_histogram.py` (V1.1) | `LutHistogramWidget` — log-scaled intensity histogram with draggable lo/hi handles, gamma slider, Auto / Reset buttons. Samples ≤ 32 frames so it stays cheap on long timeseries. Module-level `apply_lut(frame, lo, hi, gamma) -> uint8` |
| `scale_bar.py` | Matplotlib scale-bar rendering helpers (used by `MplCanvas` consumers) |
| `video_player.py` | Play/Pause/FPS playback controls |
| `custom_grips.py` | Per-edge frameless-window resize widgets. Lean reimplementation of PyDracula's `CustomGrip` |

### `backend/` — pure-Python data layer (no Qt imports)

| File | Purpose |
|---|---|
| `nd2_loader.py` | `ND2Metadata` dataclass, `read_nd2_metadata`, `read_nd2_metadata_extended` (returns dict with full metadata: T/Z/C/P, pixel size, z step, channel names + colors + emission/excitation + exposures, objective + camera + microscope, binning, frame timestamps, stage XY/Z, loops), `load_nd2_timeseries`, `LazyND2Channel` (numpy-protocol on-demand frame proxy with `__getitem__`, `materialize`, `crop`), `load_nd2_timeseries_lazy`, `z_project` |
| `nd2_volume.py` (V1.1) | `LazyND2Volume` exposing `(M, T, Z, H, W)`. `get_frame(c, m, t, z, z_mode, z_start, z_end)` returns a single (H, W) array on demand. `to_lazy_channel(c, m, z_mode, z_index, ...)` collapses to a V1.0 `LazyND2Channel` so the recipe pipeline keeps the `(T, H, W)` contract |
| `tiff_loader.py` | `get_tiff_info`, `load_tiff_stack`, `LazyTIFFChannel`, `load_tiff_stack_lazy`, `assign_dimensions`, `extract_2d_timeseries` |
| `normalization.py` | `normalize_frame_means` (frame-mean intensity normalization) |
| `recipes.py` | `build_recipe`, `save_recipe`, `load_recipe` for `.nd2s_recipe.json` (kind = `nd2studios.recipe`) |
| `exporters/tiff_exporter.py` | `export_tiff_stack(stack, filepath, bit_depth, pixel_size_um, progress_cb)`, BigTIFF auto-switch |
| `exporters/composite_exporter.py` | `export_rgb_composite_tiff(channels, colors, enabled, filepath, …)`, `_composite_frame`, `CHANNEL_COLORS` |
| `exporters/movie_exporter.py` | `MovieOptions` dataclass + `export_movie(channels, colors, enabled, filepath, options, pixel_size_um, frame_timestamps_s, progress_cb)`. Uses `imageio` (libx264 for MP4, pillow for GIF). PIL-rendered overlays for scale bar, timestamp, channel labels |
| `exporters/stitch_exporter.py` (V1.1) | `StitchLayout` dataclass, `compute_tile_layout(stage_xy_um, pixel_size_um, tile_h, tile_w, m_indices)` (stage-XY-based tile placement with grid fallback), `stitch_one_frame(tile_frames, layout, dtype)`, `export_stitched_tiff(volume, layout, m_indices, channel_indices, channel_colors, filepath, ...)` writing single-channel grayscale or multi-channel RGB composite TIFF |

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
| `export_worker.py` | `ExportRequest` dataclass + `ExportWorker` dispatching to `tiff_stack`, `rgb_composite`, or `movie` mode |
| `stitch_worker.py` (V1.1) | `StitchRequest` + `StitchWorker` wrapping `export_stitched_tiff` with progress / status signals |

### `pages/`

| File | Page | Responsibilities |
|---|---|---|
| `import_page.py` | Import | File browse, extended metadata table, **read-only** per-channel info rows, Z-mode + frame-stride config, multi-axis preview (M/T/Z sliders + per-channel toggle/color/LUT), Confirm Import (sets status `imported`), Stitch M… button → `StitchDialog` |
| `recipe_page.py` | Recipe | Plugin picker, parameter editor, trial/accept/reject pipeline, save/load `.nd2s_recipe.json`, side-by-side raw vs processed `MultiAxisViewer`s (T scrolling only on this page) |
| `export_page.py` | Export | Source selector (raw/processed), three tabs (TIFF / Composite / Movie) each running the appropriate `ExportWorker` job |
| `stitch_dialog.py` (V1.1) | dialog | Layout preview (matplotlib), per-tile / per-channel selectors, Z mode / Z slice picker, output path picker, kicks off `StitchWorker`. Launched from the Import page |

## Data Flow

```
ND2 / TIFF on disk
   │
   │  ImportPage._on_browse → LoadWorker
   ▼
ND2StudiosRecord._raw_volume     (LazyND2Volume — V1.1)
ND2StudiosRecord._raw_channels   (Dict[name, LazyND2Channel])    ← initial M=0
ND2StudiosRecord.nd2_metadata    (full extended metadata dict)
   │
   │  Viewer scrolls M/T/Z; user toggles channels; user drags LUT.
   │  ImportPage Confirm → rebuilds _raw_channels via
   │  LazyND2Volume.all_channels_as_lazy(m, z_mode, z_index)
   ▼
ND2StudiosRecord._raw_channels   (Dict[name, LazyND2Channel])    ← chosen M/Z
   │
   │  RecipePage trial/accept → RecipeWorker
   ▼
ND2StudiosRecord._processed_channels  (Dict[str, ndarray])
ND2StudiosRecord.recipe              (List[(plugin_name, params)])
   │
   │  ExportPage tab → ExportWorker → backend/exporters/
   ▼
.tif (per-channel) | .tif (RGB composite) | .mp4 / .gif on disk

Side branch (Import page → Stitch M…):
   _raw_volume → StitchDialog → StitchWorker → export_stitched_tiff()
   → multi-page TIFF (single-channel or RGB composite). The original
   ND2 is never touched.
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

ND2Studios deliberately does **not** depend on TensorFlow, StarDist,
csbdeep, or any segmentation/tracking library — those belong in
CellTracker.

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
