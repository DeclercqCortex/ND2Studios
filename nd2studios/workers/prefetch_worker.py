"""
Background frame prefetcher for ``MultiAxisViewer``.

``PrefetchManager`` is a single ``QThread`` that reads neighboring frames
into a ``FrameCache`` ahead of slider movement.  It opens its **own**
``LazyND2Volume`` handle (via a factory callable) so it never touches the
main-thread volume's ``nd2.ND2File`` handle — nd2 file handles are not
thread-safe.

Usage::

    def _factory(fp=volume.filepath):
        from nd2studios.backend.nd2_volume import LazyND2Volume
        return LazyND2Volume(fp)

    mgr = PrefetchManager(_factory, cache, channels=list(range(n_channels)))
    mgr.frame_ready.connect(self._on_prefetch_ready)
    mgr.start()

    # After each frame render:
    mgr.request_neighbors(m, t, z,
                          t_range=(0, n_t-1), z_range=(0, n_z-1),
                          z_mode="max", n=5)

    # On teardown:
    mgr.stop()
"""
from __future__ import annotations

import threading
from collections import deque
from typing import Callable, List

import numpy as np
from PySide6.QtCore import QThread, Signal


class PrefetchManager(QThread):
    """Single background thread that fills a ``FrameCache`` with neighbors.

    Signals
    -------
    frame_ready(c, m, t, z, z_mode):
        Emitted after a frame is stored in the cache.  Delivered to the
        main thread via Qt's automatic queued-connection mechanism.
    """

    frame_ready = Signal(int, int, int, int, str)

    def __init__(
        self,
        reader_factory: Callable,
        cache,
        channels: List[int],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._reader_factory = reader_factory
        self._cache = cache
        self._channels = list(channels)

        self._queue: deque = deque()
        self._queue_lock = threading.Lock()
        self._wakeup = threading.Event()
        self._stop_flag = threading.Event()

        self.setObjectName("PrefetchManager")

    # ── public API ──────────────────────────────────────────────────────────

    def request_neighbors(
        self,
        m: int,
        t: int,
        z: int,
        t_range: tuple,
        z_range: tuple,
        z_mode: str = "max",
        n: int = 5,
    ) -> None:
        """Replace the prefetch queue with ±n neighbors of (m, t, z).

        Uses T-axis neighbors when multiple timepoints exist; falls back
        to Z-axis neighbors for single-timepoint volumes.  Nearest
        neighbors are queued first.
        """
        t_min, t_max = t_range
        z_min, z_max = z_range

        candidates: list = []
        if t_max > t_min:
            for delta in range(1, n + 1):
                for sign in (1, -1):
                    nt = t + sign * delta
                    if t_min <= nt <= t_max:
                        candidates.append((m, nt, z, z_mode))
        else:
            for delta in range(1, n + 1):
                for sign in (1, -1):
                    nz = z + sign * delta
                    if z_min <= nz <= z_max:
                        candidates.append((m, t, nz, z_mode))

        tasks: list = []
        for pm, pt, pz, pzm in candidates:
            for c in self._channels:
                key = (c, pm, pt, pz, pzm)
                if not self._cache.contains(key):
                    tasks.append(key)

        with self._queue_lock:
            self._queue.clear()
            self._queue.extend(tasks)
        self._wakeup.set()

    def cancel_all(self) -> None:
        """Discard all pending prefetch work."""
        with self._queue_lock:
            self._queue.clear()

    def stop(self) -> None:
        """Signal the thread to stop and wait up to 2 s for it to exit."""
        self._stop_flag.set()
        self._wakeup.set()
        self.wait(2000)

    # ── QThread.run ─────────────────────────────────────────────────────────

    def run(self) -> None:
        reader = self._reader_factory()
        try:
            while not self._stop_flag.is_set():
                self._wakeup.wait()
                self._wakeup.clear()

                while not self._stop_flag.is_set():
                    with self._queue_lock:
                        if not self._queue:
                            break
                        key = self._queue.popleft()

                    if self._cache.contains(key):
                        continue

                    c, m, t, z, z_mode = key
                    try:
                        frame = reader.get_frame(
                            c=c, m=m, t=t, z=z, z_mode=z_mode,
                        )
                        if frame is None:
                            continue
                        arr = np.asarray(frame)
                        if arr.ndim > 2:
                            arr = arr.squeeze()
                        if arr.ndim == 2:
                            self._cache.put(key, arr)
                            self.frame_ready.emit(c, m, t, z, z_mode)
                    except Exception:
                        pass
        finally:
            if hasattr(reader, "close"):
                reader.close()
