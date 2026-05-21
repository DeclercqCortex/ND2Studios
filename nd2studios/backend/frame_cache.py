"""
Thread-safe LRU frame cache for ``MultiAxisViewer``.

Stores normalized 2D (H, W) numpy arrays keyed by
``(channel_idx, m, t, z, z_mode)`` with a byte-budget eviction policy.
Both the main UI thread and the background ``PrefetchManager`` thread
access this cache; every public method is protected by a ``threading.Lock``.
"""
from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Optional

import numpy as np


class FrameCache:
    """LRU frame cache with a configurable memory budget.

    Parameters
    ----------
    max_bytes:
        Maximum total frame data to hold (default 300 MB).
    """

    def __init__(self, max_bytes: int = 300 * 1024 ** 2) -> None:
        self._max_bytes = max_bytes
        self._cache: OrderedDict[tuple, np.ndarray] = OrderedDict()
        self._sizes: dict[tuple, int] = {}
        self._total_bytes: int = 0
        self._lock = threading.Lock()

    # ── public API ──────────────────────────────────────────────────────────

    def get(self, key: tuple) -> Optional[np.ndarray]:
        """Return the cached frame for *key*, or ``None`` on a miss.

        Promotes the entry to most-recently-used on a hit.
        """
        with self._lock:
            if key not in self._cache:
                return None
            self._cache.move_to_end(key)
            return self._cache[key]

    def put(self, key: tuple, frame: np.ndarray) -> None:
        """Store *frame* under *key*, evicting LRU entries as needed."""
        nbytes = frame.nbytes
        with self._lock:
            if key in self._cache:
                self._total_bytes -= self._sizes[key]
                del self._cache[key]
            self._cache[key] = frame
            self._sizes[key] = nbytes
            self._total_bytes += nbytes
            while self._total_bytes > self._max_bytes and self._cache:
                oldest, _ = next(iter(self._cache.items()))
                self._total_bytes -= self._sizes.pop(oldest)
                del self._cache[oldest]

    def contains(self, key: tuple) -> bool:
        """Return ``True`` if *key* is cached (no LRU promotion)."""
        with self._lock:
            return key in self._cache

    def clear(self) -> None:
        """Evict all entries."""
        with self._lock:
            self._cache.clear()
            self._sizes.clear()
            self._total_bytes = 0
