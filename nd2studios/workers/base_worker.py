"""Base QThread worker with standard signals — copied from CellTracker."""
from __future__ import annotations

import traceback
from typing import ClassVar, List

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

    # Class-level registry keeps a Python reference to every running worker so
    # Python's GC cannot destroy the QThread wrapper while the OS thread is
    # still executing — the source of "QThread: Destroyed while thread is still
    # running" warnings.
    _running: ClassVar[List[BaseWorker]] = []

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cancelled = False
        BaseWorker._running.append(self)

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
        finally:
            try:
                BaseWorker._running.remove(self)
            except ValueError:
                pass

    def run_task(self):
        raise NotImplementedError
