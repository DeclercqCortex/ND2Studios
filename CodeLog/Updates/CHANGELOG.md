# Changelog

All notable changes to ND2Studios will be documented in this file.

Format: [Keep a Changelog](https://keepachangelog.com/)

## [V1.1] - 2026-04-30

### Added

#### M / T / Z scrolling in the viewer
- **`backend/nd2_volume.py`**: new `LazyND2Volume` class exposing
  `(M, T, Z, H, W)` lazy reads against an ND2 file. Holds a single
  `nd2.ND2File` handle internally; `get_frame(c, m, t, z, z_mode,
  z_start, z_end)` returns one (H, W) array per call. Z projection is
  applied per-frame on demand.
- `LazyND2Volume.to_lazy_channel(c, m, z_mode, z_index, ...)` collapses
  the volume to a V1.0-compatible `LazyND2Channel` so the recipe /
  export pipeline keeps its `(T, H, W)` contract — plugins did not
  need to change. `all_channels_as_lazy()` is a convenience that
  returns an `OrderedDict` of channel-name → lazy proxy.
- **`workers/load_worker.py`**: also returns `volume: LazyND2Volume`
  in the loaded payload so the Import page can wire up the new viewer.

#### Live channel toggle / color / LUT in the viewer
- **`widgets/multi_axis_viewer.py`**: new `MultiAxisViewer` widget
  with M / T / Z sliders (auto-hidden when an axis is degenerate)
  and a per-channel control strip (`ChannelControlRow`) holding an
  enable checkbox, color combobox, and embedded LUT histogram. Any
  change recomposes the displayed image instantly. The viewer
  accepts either a `LazyND2Volume` (M/Z navigation enabled) or a
  flat `Dict[name, (T, H, W)]` (recipe / processed view).
- Image rendering goes through `apply_lut(frame, lo, hi, gamma)` per
  channel rather than the old auto-percentile path.

#### Manual LUT histogram per channel
- **`widgets/lut_histogram.py`**: new `LutHistogramWidget` showing a
  log-scaled intensity histogram with two draggable vertical handles
  (`lo`, `hi`), a `γ` slider (0.20..5.00), and `Auto` / `Reset`
  buttons. Histogram is computed by sampling at most 32 frames so it
  stays cheap on large timeseries. Emits
  `contrast_changed(lo, hi, gamma)`. Exports a module-level
  `apply_lut(frame, lo, hi, gamma) -> uint8` helper used by the
  multi-axis viewer.

#### M-position stitching
- **`backend/exporters/stitch_exporter.py`**: `StitchLayout` dataclass,
  `compute_tile_layout()` (places tiles using ND2 stage XY positions
  divided by the file's pixel size; falls back to row-major grid when
  positions are missing or all identical),
  `stitch_one_frame(tile_frames, layout)` for one timepoint, and
  `export_stitched_tiff(volume, layout, m_indices, channel_indices,
  channel_colors, filepath, z_mode, z_index, rgb, pixel_size_um,
  progress_cb)` for the full timelapse. Single-channel output writes
  a grayscale TIFF; multi-channel writes an RGB composite using
  per-channel percentile contrast.
- **`workers/stitch_worker.py`**: `StitchRequest` + `StitchWorker`
  wrapping `export_stitched_tiff` with progress signals.
- **`pages/stitch_dialog.py`**: `StitchDialog` launched from a new
  "Stitch M…" button on the Import page. Shows a live layout preview
  (matplotlib canvas with tile rectangles), per-tile checkboxes,
  per-channel toggle + color combo, Z mode picker, and an output
  path picker. Cancels cleanly; never modifies the source file.

### Changed
- **`core/experiment_manager.py`**: `ND2StudiosRecord` gains
  `m_index`, `z_view_mode`, `z_view_index` (round-tripped via JSON),
  `n_multipoints`, `n_zslices`, and a non-serialized `_raw_volume`
  field. `channel_display` entries now also carry `lut_lo`, `lut_hi`,
  `lut_gamma` so contrast settings persist across save/load.
- **`pages/import_page.py`**: replaced the `ChannelRow` widget with
  the read-only `ChannelInfoRow` (exposure / em / ex display only).
  The right-side preview is now a `MultiAxisViewer`. Added a
  "Stitch M…" button next to "Confirm Import" — enabled when the
  loaded file has more than one M position. Confirm Import now
  rebuilds per-channel proxies via
  `LazyND2Volume.all_channels_as_lazy(m, z_mode, z_index)` so the
  pipeline operates on whatever (M, Z mode) the user is viewing.
- **`pages/recipe_page.py`**: both side-by-side viewers are now
  `MultiAxisViewer`s. Removed the `_channel_view_state()` helper —
  the new viewer pulls channel display state directly from the
  experiment record.

### Documentation
- V1.1 plan: `CodeLog/ClaudesPlan/V1.1_mz_scrolling_lut_stitching.md`.
- Architecture updated to describe the new modules and data flow.

## [V1.0] - 2026-04-30

### Added

#### Project scaffold
- VSClaude project layout (`CodeLog/`, `Research/`, `hpc/`, `scripts/`)
- `CLAUDE.md` customized for ND2Studios with project conventions, key
  technical decisions, and dependencies
- V1.0 plan document at `CodeLog/ClaudesPlan/V1.0_initial_scaffold.md`
- Top-level `run.py` launcher that prepends the project root to
  `sys.path` and calls `nd2studios.__main__.main()`

#### Core (`nd2studios/core/`)
- `settings.py`: Dracula palette, app dimensions, animated sidebar
  config (60↔240 px, 280 ms `InOutQuart`), 3-page list (`import`,
  `recipe`, `export`), status ladder (`new → imported → preprocessed →
  ready_to_export`) and per-page prerequisites
- `theme.py`: Dracula dark stylesheet adapted from CellTracker, with
  additional rules for `#bgApp` rounded card, `#titleBarWidget` custom
  frameless title bar, `#navBtn` collapsible sidebar buttons with text
  labels, `#sessionBtn`, and `#toggleBtn` (hamburger)
- `main_window.py`: frameless `QMainWindow` with:
  - PyDracula-style custom title bar (drag-to-move, double-click to
    maximize, Min/Max/Close buttons)
  - 4 `CustomGrip` resize handles + a `QSizeGrip` corner
  - Animated collapsible sidebar via `QPropertyAnimation` on
    `minimumWidth` and `maximumWidth`
  - Drop shadow on `#bgApp` (blur 17, alpha 150)
  - Sidebar navigation buttons in a `QButtonGroup` (exclusive)
  - Session buttons (New / Save / Load) at the sidebar bottom
  - Top bar (page title + `StatusIndicator`) and bottom bar (version +
    `QProgressBar` + status text)
  - `QStackedWidget` page switching with prerequisite-gated navigation
- `experiment_manager.py`: `ND2StudiosRecord` (configs, metadata, recipe,
  channel display, raw + processed channel caches, frame timestamps)
  and `ND2StudiosManager` (`active_changed`, `status_changed` signals,
  `.nd2s` JSON manifest + `_arrays.npz` companion save/load)
- `plugin_registry.py`: `PluginBase` with `@register` decorator,
  `ParamSpec` for declarative parameter UIs, `EnhancementPlugin` base
  (input/output `(T, H, W)` numpy)

#### Backend (`nd2studios/backend/`)
- `nd2_loader.py`: copied from CellTracker — `ND2Metadata` dataclass,
  `read_nd2_metadata`, `load_nd2_timeseries`, `LazyND2Channel` (numpy
  protocol proxy with `shape`, `dtype`, `__getitem__`, `materialize`,
  `crop`), `load_nd2_timeseries_lazy`, `z_project`. Extended with
  `read_nd2_metadata_extended()` returning a dict with: dim_order,
  sizes, dtype, height, width, T/Z/C/P counts, pixel size, z step,
  voxel size, channel names + colors + emission/excitation + exposures,
  objective name/magnification/NA/immersion, binning, camera name,
  microscope name, per-frame timestamps, per-frame stage XY/Z, loops
- `tiff_loader.py`: copied from CellTracker — `LazyTIFFChannel`,
  `load_tiff_stack_lazy`, `assign_dimensions`, `extract_2d_timeseries`
- `normalization.py`: copied from CellTracker — frame-mean normalization
- `recipes.py`: copied from CellTracker — recipe save/load with the
  extension changed to `.nd2s_recipe.json` and the kind tag set to
  `nd2studios.recipe`
- `exporters/tiff_exporter.py`: `export_tiff_stack(stack, filepath,
  bit_depth, pixel_size_um, progress_cb)` with passthrough / uint8 /
  uint16 modes (using 0.5–99.5 percentile bounds), TIFF resolution
  tags, BigTIFF when > 3.9 GB
- `exporters/composite_exporter.py`:
  `CHANNEL_COLORS`, `_composite_frame()` (additive RGB blending),
  `export_rgb_composite_tiff()`
- `exporters/movie_exporter.py`: `MovieOptions` dataclass (FPS, scale
  bar config, timestamp config, channel-label config), `export_movie()`
  using `imageio` for MP4 (libx264) and GIF; PIL-drawn overlays —
  scale bar (configurable corner / length in µm / color / thickness),
  timestamp (corner / color / synthetic-dt fallback when ND2 timestamps
  are unavailable), channel labels with color swatches

#### Plugins (`nd2studios/plugins/enhancement/`)
- `builtin.py`: 19 enhancement plugins copied verbatim from CellTracker
  (Normalize, CLAHE, GaussianBlur, MedianFilter, BackgroundSubtract,
  GammaCorrection, BleachCorrection, TemporalFoldCorrection,
  SpatialFlatness, TopHat, DoG, UnsharpMask, BilateralDenoise,
  MorphGradient, LocalContrast, BlobSubtract, NLMDenoise,
  WaveletDenoise, TVDenoise) — registered via `@PluginBase.register`

#### Widgets (`nd2studios/widgets/`)
- `common.py`: `MplCanvas` (Dracula-styled matplotlib figure with
  per-axis dark theming), `ParamEditor` (auto-generates a form from
  `List[ParamSpec]` with float/int/bool/choice/str support and a
  `params_changed(dict)` signal), `StatusIndicator` (colored badge with
  the ND2Studios 5-status palette: new / imported / preprocessed /
  ready_to_export / error)
- `image_viewer.py`: copied from CellTracker — `CHANNEL_COLORS`,
  `frame_to_uint8`, `apply_lut_color`, `composite_channels`,
  `ImageCanvas` (zoom/pan), `ZoomToolbar`, `ImageViewer` (multi-channel
  RGB compositing, T slider, auto-contrast)
- `scale_bar.py`: copied from CellTracker — matplotlib scale bar
  rendering helpers
- `video_player.py`: copied from CellTracker — Play/Pause/FPS playback
  controls
- `custom_grips.py`: PySide6 frameless-window edge resize handles —
  reimplemented from PyDracula's `CustomGrip` as a leaner
  `QWidget`-based class with per-edge cursor, transparent background,
  and `_resize_parent(delta)` that respects `minimumWidth`/`Height`

#### Workers (`nd2studios/workers/`)
- `base_worker.py`: `BaseWorker(QThread)` with `progress(int)`,
  `status(str)`, `finished(object)`, `error(str)` signals and `cancel()`
- `load_worker.py`: `LoadWorker(filepath, z_projection, z_start, z_end,
  t_start, t_end, t_stride)` returning a dict with metadata, lazy
  channels, channel names, and frame timestamps; supports `.nd2`
  (extended metadata) and `.tif`/`.tiff` (single-channel TYX)
- `recipe_worker.py`: `RecipeWorker(channels, recipe, normalized)`
  applies the ordered list of `(plugin_name, params)` plugins to each
  channel; lazy proxies are materialized once per channel; optional
  frame-mean normalization runs first
- `export_worker.py`: `ExportRequest` dataclass + `ExportWorker`
  dispatching to `tiff_stack` / `rgb_composite` / `movie` modes; per
  enabled channel for TIFF mode

#### Pages (`nd2studios/pages/`)
- `import_page.py`: file browse, full extended-metadata table (file,
  dimensions, frame size, dtype, pixel size, z step, objective, camera,
  microscope, binning, acquisition span, mean dt, stage XY range,
  loops), scrollable channel-row list (enable + color combo + per-channel
  exposure / emission / excitation), Z-projection mode (max/mean/min/none),
  frame stride, image preview via multi-channel `ImageViewer`, Confirm
  Import button
- `recipe_page.py`: plugin combo + description label, auto-generated
  parameter form, frame-mean normalization checkbox, Trial / Accept /
  Reject / Remove Last / Clear All buttons, recipe list view, Save / Load
  recipe (`.nd2s_recipe.json`), side-by-side raw vs processed previews
- `export_page.py`: source selector (raw / processed), three tabs:
  TIFF stack (bit depth picker), RGB composite (one click), Movie
  (FPS, format, scale bar, timestamp, channel labels — with per-corner
  position pickers); each tab kicks off `ExportWorker`

### Documentation
- `CodeLog/Architecture/ARCHITECTURE.md` populated with module
  breakdown, data flow, dependencies, and design decisions
- This changelog
