"""Base QThread worker with standard signals — copied from CellTracker."""
from __future__ import annotations

import traceback

from PySide6.QtCore import QThread, Signal


class BaseWorker(QThread):
    """Background worker with progress / status / finished / error signals.

    Subclasses override :meth:`run_task` and use :meth:`set_progress`
    and :meth:`set_status` to report progress.
    """
    progress = Signal(int)        # 0-100
    status = Signal(str)          # status text
    finished = Signal(object)     # result object
    error = Signal(str)           # error message

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def set_progress(self, value: int) -> None:
        self.progress.emit(value)

    def set_status(self, text: str) -> None:
        self.status.emit(text)

    def run(self) -> None:
        try:
            result = self.run_task()
            if not self._cancelled:
                self.finished.emit(result)
        except Exception as e:
            self.error.emit(f"{type(e).__name__}: {e}\n{traceback.format_exc()}")

    def run_task(self):
        raise NotImplementedError
