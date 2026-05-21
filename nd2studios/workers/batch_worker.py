"""
BatchWorker (V1.23) — process multiple files through a saved pipeline template.

For each file: load → apply recipe → run analysis → compute measurements.
Optionally exports per-file overlay images.
All per-file measurement rows are accumulated and written to a single
aggregate CSV at the end.

Extra signal:
    file_done(int, int)  — emitted with (completed_count, total) after each file.
"""
from __future__ import annotations

import csv
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Signal

from nd2studios.workers.base_worker import BaseWorker


class BatchWorker(BaseWorker):
    """Sequential per-file pipeline runner."""

    file_done = Signal(int, int)   # (completed, total)

    def __init__(
        self,
        filepaths: List[str],
        template: Dict[str, Any],
        output_dir: str,
        export_images: bool = False,
        image_format: str = "tiff",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.filepaths = filepaths
        self.template = template
        self.output_dir = output_dir
        self.export_images = export_images
        self.image_format = image_format

    # ── Main task ─────────────────────────────────────────────────────────────

    def run_task(self) -> str:
        from nd2studios.backend.results_engine import (
            compute_measurements, export_overlay_frames,
        )
        from nd2studios.core.analysis_registry import AnalysisPipeline
        from nd2studios.workers.load_worker import LoadWorker
        from nd2studios.workers.recipe_worker import RecipeWorker

        pipeline_cfg = self.template.get("pipeline", {})
        import_cfg: Dict[str, Any] = pipeline_cfg.get("import", {})
        recipe_raw: list = pipeline_cfg.get("recipe", [])
        recipe: List[Tuple[str, Dict]] = [
            (s["name"], dict(s.get("params", {}))) for s in recipe_raw
        ]
        recipe_normalized: bool = bool(pipeline_cfg.get("recipe_normalized", False))
        analysis_cfg: Dict[str, Any] = pipeline_cfg.get("analysis", {})
        pipeline_name: str = analysis_cfg.get("pipeline_name", "")
        analysis_params: Dict[str, Any] = dict(analysis_cfg.get("params", {}))

        pipeline_cls = AnalysisPipeline.get_pipeline(pipeline_name)

        os.makedirs(self.output_dir, exist_ok=True)

        all_rows: List[Dict[str, Any]] = []
        total = len(self.filepaths)

        for file_idx, filepath in enumerate(self.filepaths):
            if self.cancelled:
                break

            basename = os.path.splitext(os.path.basename(filepath))[0]
            self.set_status(f"[{file_idx + 1}/{total}] Loading {basename}…")

            try:
                rows = self._process_one(
                    filepath=filepath,
                    basename=basename,
                    file_idx=file_idx,
                    total=total,
                    import_cfg=import_cfg,
                    recipe=recipe,
                    recipe_normalized=recipe_normalized,
                    pipeline_cls=pipeline_cls,
                    pipeline_name=pipeline_name,
                    analysis_params=analysis_params,
                    compute_measurements=compute_measurements,
                    export_overlay_frames=export_overlay_frames,
                )
                all_rows.extend(rows)
            except Exception as exc:
                self.set_status(
                    f"[{file_idx + 1}/{total}] ERROR {basename}: {exc}"
                )

            self.file_done.emit(file_idx + 1, total)
            self.set_progress(int((file_idx + 1) / max(total, 1) * 100))

        # Write aggregate CSV
        csv_path = os.path.join(self.output_dir, "batch_results.csv")
        if all_rows:
            fieldnames = list(all_rows[0].keys())
            # Ensure source_file is the first column
            if "source_file" in fieldnames:
                fieldnames.remove("source_file")
                fieldnames.insert(0, "source_file")
            with open(csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(
                    f, fieldnames=fieldnames, extrasaction="ignore"
                )
                writer.writeheader()
                writer.writerows(all_rows)

        self.set_status(
            f"Done. {len(all_rows)} total objects from {total} file(s)."
        )
        return csv_path

    # ── Per-file processing ───────────────────────────────────────────────────

    def _process_one(
        self,
        filepath: str,
        basename: str,
        file_idx: int,
        total: int,
        import_cfg: Dict[str, Any],
        recipe: List[Tuple[str, Dict]],
        recipe_normalized: bool,
        pipeline_cls: Optional[Any],
        pipeline_name: str,
        analysis_params: Dict[str, Any],
        compute_measurements,
        export_overlay_frames,
    ) -> List[Dict[str, Any]]:
        from nd2studios.workers.load_worker import LoadWorker
        from nd2studios.workers.recipe_worker import RecipeWorker

        z_mode: str = import_cfg.get("z_mode", "max")
        z_index: int = int(import_cfg.get("z_index", 0))
        t_stride: int = int(import_cfg.get("frame_stride", 1))

        # 1. Load (M=0 channels + metadata)
        load_worker = LoadWorker(
            filepath,
            z_projection=z_mode,
            t_stride=t_stride,
        )
        load_result = load_worker.run_task()
        channels_m0: Dict[str, Any] = load_result.get("channels", {})
        metadata: Dict[str, Any] = load_result.get("metadata", {})

        # Materialise lazy proxies for M=0
        channels_m0 = {
            k: (
                v.materialize() if hasattr(v, "materialize") and callable(v.materialize)
                else np.asarray(v)
            )
            for k, v in channels_m0.items()
        }

        if self.cancelled:
            return []

        # Determine M positions — probe volume for multi-M ND2 files.
        m_positions = [0]
        raw_volume = None
        if filepath.lower().endswith(".nd2"):
            try:
                from nd2studios.backend.nd2_volume import LazyND2Volume
                raw_volume = LazyND2Volume(filepath)
                if raw_volume.n_multipoints > 1:
                    m_positions = list(range(raw_volume.n_multipoints))
            except Exception:
                pass

        multi_m = len(m_positions) > 1
        file_frac = 1.0 / max(total, 1)
        base_pct = int(file_idx * file_frac * 100)

        all_rows: List[Dict[str, Any]] = []

        for m in m_positions:
            if self.cancelled:
                break

            # Build channels for this M position.
            if raw_volume is not None and multi_m:
                channels: Dict[str, np.ndarray] = {}
                for c_idx, ch_name in enumerate(raw_volume.channel_names):
                    arr = raw_volume.to_lazy_channel(
                        c_idx, m=m, z_mode=z_mode, z_index=z_index
                    ).materialize()
                    if t_stride > 1 and arr.ndim == 3:
                        arr = arr[::t_stride]
                    channels[ch_name] = arr
            else:
                channels = dict(channels_m0)

            # 2. Apply recipe
            if recipe and not self.cancelled:
                self.set_status(
                    f"[{file_idx + 1}/{total}] Recipe M{m}: {basename}…"
                    if multi_m else f"[{file_idx + 1}/{total}] Recipe: {basename}…"
                )
                recipe_worker = RecipeWorker(channels, recipe, recipe_normalized)
                channels = recipe_worker.run_task()

            if self.cancelled:
                break

            # 3. Run analysis pipeline
            if pipeline_cls is None:
                continue

            self.set_status(
                f"[{file_idx + 1}/{total}] {pipeline_name} M{m}: {basename}…"
                if multi_m else f"[{file_idx + 1}/{total}] {pipeline_name}: {basename}…"
            )
            pipeline = pipeline_cls()
            analysis_result = pipeline.run(
                channels,
                metadata,
                analysis_params,
                progress_cb=lambda p: self.set_progress(base_pct + int(p * file_frac)),
                cancelled_cb=lambda: self.cancelled,
            )

            if self.cancelled:
                break

            # 4. Compute measurements
            rows = compute_measurements(
                analysis_result.label_masks,
                channels,
                metadata,
                m_index=m,
            )
            for row in rows:
                row["source_file"] = os.path.basename(filepath)
                if multi_m:
                    row["m_position"] = m
            all_rows.extend(rows)

            # 5. Optional image export
            if self.export_images and not self.cancelled:
                self.set_status(
                    f"[{file_idx + 1}/{total}] Exporting images M{m}: {basename}…"
                    if multi_m else f"[{file_idx + 1}/{total}] Exporting images: {basename}…"
                )
                img_subdir = f"{basename}_M{m:02d}" if multi_m else basename
                img_dir = os.path.join(self.output_dir, img_subdir)
                os.makedirs(img_dir, exist_ok=True)
                try:
                    export_overlay_frames(
                        channels=channels,
                        label_masks=analysis_result.label_masks,
                        metadata=metadata,
                        output_dir=img_dir,
                        fmt=self.image_format,
                    )
                except Exception:
                    pass  # image export failure should not abort the batch

        return all_rows
