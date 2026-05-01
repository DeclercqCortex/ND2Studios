"""
Central state manager for ND2Studios.

`ND2StudiosRecord` holds everything about a session: configs, metadata,
the in-memory channel arrays, and the current recipe. `ND2StudiosManager`
holds the active record and emits change signals.

Sessions serialize to .nd2s = JSON manifest + companion NPZ for arrays.
"""
from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import QObject, Signal


SESSION_EXTENSION = ".nd2s"


@dataclass
class ND2StudiosRecord:
    """Complete state for one ND2Studios session.

    Status ladder (mirrors `Settings.STATUS_ORDER`):
        new → imported → preprocessed → ready_to_export
    """

    # Identity
    exp_id: str = ""
    name: str = "Untitled"
    timestamp: str = ""
    status: str = "new"

    # Configs (JSON-serializable)
    import_config: Dict[str, Any] = field(default_factory=dict)
    recipe_config: Dict[str, Any] = field(default_factory=dict)
    export_config: Dict[str, Any] = field(default_factory=dict)

    # Recipe: ordered list of (plugin_name, params_dict) pairs.
    # Lives on the record so it round-trips through save/load and can be
    # exported separately as a portable .nd2s_recipe.json.
    recipe: List[Tuple[str, Dict[str, Any]]] = field(default_factory=list)
    recipe_normalized: bool = False

    # ND2 metadata snapshot (serializable subset of ND2Metadata)
    nd2_metadata: Dict[str, Any] = field(default_factory=dict)

    # Channel display config: {channel_name: {"enabled": bool, "color": str,
    #                                         "lut_lo": float, "lut_hi": float,
    #                                         "lut_gamma": float}}
    channel_display: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # V1.1 viewer-state — survives session save/load.
    m_index: int = 0           # currently displayed M position
    z_view_mode: str = "max"   # max / mean / min / none
    z_view_index: int = 0      # which Z slice when z_view_mode == "none"

    # Frame info
    n_frames: int = 0
    frame_height: int = 0
    frame_width: int = 0
    n_multipoints: int = 1
    n_zslices: int = 1
    pixel_size_um: float = 1.0
    fps: float = 10.0  # default movie export FPS

    # ── In-memory caches (NOT JSON-serialized; arrays go to companion NPZ) ──
    # Raw channels right after import (post Z-projection). Each value is
    # either a numpy array or a `LazyND2Channel` proxy.
    _raw_channels: Optional[Dict[str, Any]] = field(default=None, repr=False)
    # Channels after the recipe has been applied. None if no recipe yet.
    _processed_channels: Optional[Dict[str, np.ndarray]] = field(default=None, repr=False)
    # Per-frame timestamps (seconds since experiment start), if present in ND2.
    _frame_timestamps: Optional[np.ndarray] = field(default=None, repr=False)
    # V1.1: a LazyND2Volume for M/Z scrolling (rebuilt from filepath on load).
    _raw_volume: Optional[Any] = field(default=None, repr=False)

    def to_dict(self) -> dict:
        """Serialize the JSON-safe portion of the record."""
        return {
            "exp_id": self.exp_id,
            "name": self.name,
            "timestamp": self.timestamp,
            "status": self.status,
            "import_config": self.import_config,
            "recipe_config": self.recipe_config,
            "export_config": self.export_config,
            "recipe": [
                {"name": n, "params": p} for (n, p) in self.recipe
            ],
            "recipe_normalized": self.recipe_normalized,
            "nd2_metadata": self.nd2_metadata,
            "channel_display": self.channel_display,
            "m_index": self.m_index,
            "z_view_mode": self.z_view_mode,
            "z_view_index": self.z_view_index,
            "n_frames": self.n_frames,
            "frame_height": self.frame_height,
            "frame_width": self.frame_width,
            "n_multipoints": self.n_multipoints,
            "n_zslices": self.n_zslices,
            "pixel_size_um": self.pixel_size_um,
            "fps": self.fps,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ND2StudiosRecord":
        rec = cls()
        for key, val in d.items():
            if key == "recipe":
                rec.recipe = [(item["name"], dict(item.get("params", {}))) for item in val]
            elif hasattr(rec, key) and not key.startswith("_"):
                setattr(rec, key, val)
        return rec


class ND2StudiosManager(QObject):
    """Manages the active session record and emits change signals."""

    active_changed = Signal()
    status_changed = Signal(str)

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._active: Optional[ND2StudiosRecord] = None

    @property
    def active(self) -> Optional[ND2StudiosRecord]:
        return self._active

    def new_experiment(self, name: str = "Untitled") -> ND2StudiosRecord:
        rec = ND2StudiosRecord(
            exp_id=str(uuid.uuid4())[:8],
            name=name,
            timestamp=datetime.now().isoformat(timespec="seconds"),
            status="new",
        )
        self._active = rec
        self.active_changed.emit()
        return rec

    def set_status(self, status: str) -> None:
        if self._active is not None:
            self._active.status = status
            self.status_changed.emit(status)

    # ── Persistence ────────────────────────────────────────────────
    def save_session(self, filepath: str) -> None:
        """Save the active session to a `.nd2s` file plus companion NPZ.

        The JSON manifest is human-readable and the NPZ holds the heavy
        arrays (raw + processed channel data, frame timestamps).
        """
        if self._active is None:
            return

        # Force the requested extension.
        if not filepath.lower().endswith(SESSION_EXTENSION):
            filepath = filepath + SESSION_EXTENSION

        base = os.path.splitext(filepath)[0]
        manifest = self._active.to_dict()

        arrays: Dict[str, np.ndarray] = {}

        # Materialize lazy channels before saving.
        raw = self._active._raw_channels or {}
        for ch_name, ch_data in raw.items():
            arrays[f"raw_{ch_name}"] = _materialize(ch_data)
        if raw:
            manifest["_raw_channel_names"] = list(raw.keys())

        proc = self._active._processed_channels or {}
        for ch_name, ch_data in proc.items():
            arrays[f"proc_{ch_name}"] = np.asarray(ch_data)
        if proc:
            manifest["_processed_channel_names"] = list(proc.keys())

        if self._active._frame_timestamps is not None:
            arrays["frame_timestamps"] = np.asarray(self._active._frame_timestamps)

        if arrays:
            np.savez_compressed(base + "_arrays.npz", **arrays)
            manifest["_arrays_file"] = os.path.basename(base + "_arrays.npz")

        with open(filepath, "w") as f:
            json.dump(manifest, f, indent=2)

    def load_session(self, filepath: str) -> ND2StudiosRecord:
        """Load a session from `.nd2s` (and the NPZ next to it)."""
        with open(filepath, "r") as f:
            manifest = json.load(f)

        rec = ND2StudiosRecord.from_dict(manifest)
        dirpath = os.path.dirname(filepath)

        arrays_file = manifest.get("_arrays_file")
        if arrays_file:
            npz_path = os.path.join(dirpath, arrays_file)
            if os.path.exists(npz_path):
                data = np.load(npz_path)

                raw_names: List[str] = manifest.get("_raw_channel_names", [])
                if raw_names:
                    rec._raw_channels = {
                        n: data[f"raw_{n}"] for n in raw_names if f"raw_{n}" in data
                    }

                proc_names: List[str] = manifest.get("_processed_channel_names", [])
                if proc_names:
                    rec._processed_channels = {
                        n: data[f"proc_{n}"] for n in proc_names if f"proc_{n}" in data
                    }

                if "frame_timestamps" in data:
                    rec._frame_timestamps = np.asarray(data["frame_timestamps"])

        self._active = rec
        self.active_changed.emit()
        return rec


def _materialize(channel_data: Any) -> np.ndarray:
    """Convert a LazyND2Channel proxy or array-like into a contiguous ndarray."""
    materialize = getattr(channel_data, "materialize", None)
    if callable(materialize):
        return materialize()
    return np.asarray(channel_data)
