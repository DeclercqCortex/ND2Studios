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
        total = max(1, n_channels * (n_steps + (1 if self.normalized else 0)))

        for ch_idx, (ch_name, data) in enumerate(self.channels.items()):
            if self.cancelled:
                return results

            # Lazy proxies must be materialized for plugins that mutate.
            self.set_status(f"Channel {ch_name}: loading frames…")
            if hasattr(data, "materialize") and callable(getattr(data, "materialize")):
                current = data.materialize()
            else:
                current = np.asarray(data).copy()

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

                done = ch_idx * (n_steps + 1) + step_idx + 1 + (1 if self.normalized else 0)
                self.set_progress(int(done / total * 100))

            results[ch_name] = current

        self.set_progress(100)
        return results
