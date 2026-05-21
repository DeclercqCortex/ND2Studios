"""QThread worker for analysis pipelines."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np

from nd2studios.core.analysis_registry import AnalysisPipeline, AnalysisResult
from nd2studios.workers.base_worker import BaseWorker


class AnalysisWorker(BaseWorker):
    """Runs an AnalysisPipeline in a background thread.

    Emits ``finished(AnalysisResult)`` on success, ``error(str)`` on failure.
    Supports cancellation via ``cancel()`` — the pipeline checks ``cancelled_cb``
    between frames.
    """

    def __init__(
        self,
        pipeline: AnalysisPipeline,
        channels: Dict[str, np.ndarray],
        metadata: Dict[str, Any],
        params: Dict[str, Any],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.pipeline = pipeline
        self.channels = channels
        self.metadata = metadata
        self.params = params

    def run_task(self) -> AnalysisResult:
        return self.pipeline.run(
            self.channels,
            self.metadata,
            self.params,
            progress_cb=self.set_progress,
            cancelled_cb=lambda: self.cancelled,
        )
