"""
RecipeWorker — apply an ordered list of enhancement plugins to a
multi-channel timeseries in a background thread.

Returns:
    {channel_name: (T, H, W) np.ndarray}
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from nd2studios.core.plugin_registry import PluginBase
from nd2studios.workers.base_worker import BaseWorker


class RecipeWorker(BaseWorker):
    """Apply a recipe (ordered (plugin_name, params) pairs) to each channel."""

    def __init__(
        self,
        channels: Dict[str, Any],
        recipe: List[Tuple[str, Dict[str, Any]]],
        normalized: bool = False,
        parent=None,
    ):
        super().__init__(parent)
        self.channels = channels
        self.recipe = recipe
        self.normalized = normalized

    def run_task(self) -> Dict[str, np.ndarray]:
        from nd2studios.backend.normalization import normalize_timeseries

        results: Dict[str, np.ndarray] = {}
        n_channels = len(self.channels)
        n_steps = len(self.recipe)
        # +1 phase per channel for the materialization loop, since reading
        # 35×(11264×6144) uint16 = ~4.5 GiB takes longer than most plugin
        # steps and used to look like a freeze (no progress, no cancel).
        total_phases = max(
            1,
            n_channels * (n_steps + (1 if self.normalized else 0) + 1),
        )

        for ch_idx, (ch_name, data) in enumerate(self.channels.items()):
            if self.cancelled:
                return results

            # Lazy proxies must be materialized for plugins that mutate.
            # Stream frame-by-frame so the GUI shows progress and the
            # Cancel button can interrupt long reads from disk. Falls
            # back to one-shot ``data.materialize()`` / ``np.asarray()``
            # when ``data`` isn't indexable per-frame.
            current = self._materialize_with_progress(
                ch_idx, ch_name, data, total_phases,
            )
            if current is None:  # cancelled mid-load
                return results

            if self.normalized:
                self.set_status(f"Channel {ch_name}: frame-mean normalization")
                try:
                    current = normalize_timeseries(current)
                except Exception:
                    # Normalization is best-effort; don't fail the whole recipe.
                    pass

            for step_idx, (plugin_name, params) in enumerate(self.recipe):
                if self.cancelled:
                    return results
                self.set_status(f"Channel {ch_name}: {plugin_name}")
                plugin_cls = PluginBase.get_plugin("enhancement", plugin_name)
                if plugin_cls is None:
                    # Skip unknown plugins; never crash the worker.
                    continue
                plugin = plugin_cls()
                current = plugin.execute(current, params, progress_cb=None)

                # +1 covers the materialization phase counted in total_phases.
                done = (ch_idx * (n_steps + 1 + (1 if self.normalized else 0))
                        + 1 + (1 if self.normalized else 0) + step_idx + 1)
                self.set_progress(int(done / total_phases * 100))

            results[ch_name] = current

        self.set_progress(100)
        return results

    def _materialize_with_progress(
        self, ch_idx: int, ch_name: str, data,
        total_phases: int,
    ) -> "np.ndarray | None":
        """Read a lazy proxy frame-by-frame so the GUI shows progress.

        Reports per-frame status updates so users can distinguish a
        slow disk read from a hang.  Returns ``None`` if the worker
        was cancelled mid-load.
        """
        # Per-frame path: data has a usable ``shape[0]`` and supports
        # ``data[t]`` indexing. Covers LazyND2Channel,
        # MultiFileLazyChannel, LazyTIFFChannel, and plain ndarrays.
        n_steps = len(self.recipe)
        per_ch_phases = n_steps + 1 + (1 if self.normalized else 0)
        shape = getattr(data, "shape", None)
        if (shape is not None and len(shape) >= 1
                and hasattr(data, "__getitem__")):
            try:
                n_t = int(shape[0])
            except Exception:
                n_t = 0
            if n_t > 0:
                size_mb = 0
                try:
                    if len(shape) >= 3:
                        size_mb = (n_t * int(shape[-2]) * int(shape[-1])
                                   * np.dtype(getattr(data, "dtype",
                                                       np.uint16)).itemsize
                                   ) // (1024 * 1024)
                except Exception:
                    size_mb = 0
                hint = f" (~{size_mb} MiB)" if size_mb else ""
                self.set_status(
                    f"Channel {ch_name}: reading {n_t} frames{hint}…"
                )
                frames = []
                for t in range(n_t):
                    if self.cancelled:
                        return None
                    try:
                        f = np.asarray(data[t])
                    except Exception:
                        # If indexing fails fall back to one-shot below.
                        frames = None
                        break
                    if f.ndim > 2:
                        f = f.squeeze()
                    frames.append(f)
                    if t % 4 == 0 or t == n_t - 1:
                        # Reserve the first 1/per_ch_phases slot per
                        # channel for materialization progress.
                        sub = (t + 1) / n_t
                        done = ch_idx * per_ch_phases + sub
                        self.set_progress(
                            int(done / total_phases * 100)
                        )
                        self.set_status(
                            f"Channel {ch_name}: read {t + 1}/{n_t}"
                        )
                if frames:
                    try:
                        return np.stack(frames, axis=0)
                    except Exception:
                        # If frames have inconsistent shape (shouldn't
                        # happen with our lazy proxies), fall through.
                        pass

        # One-shot fallback for proxies that can't be frame-indexed.
        self.set_status(f"Channel {ch_name}: loading frames…")
        if hasattr(data, "materialize") and callable(getattr(data, "materialize")):
            return data.materialize()
        return np.asarray(data).copy()
