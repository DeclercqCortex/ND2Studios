"""Checkpoint freeze/resume CACHE — pure serialization, NO analysis math.

WHERE THE REAL MATH LIVES
    There is NONE. This node carries no analysis kernel. It is a freeze/resume
    cache: it serializes an already-computed, in-memory checkpoint store to disk
    and rebuilds it on load. All the actual computation (segmentation, tracking,
    measurements) happens in *other* nodes upstream; the Checkpoint node merely
    persists their frozen output so a later Run can resume from it.

PROVENANCE (branch Version-1.45)
    - nd2studios/pipeline_graph/checkpoint_io.py  (save/load logic, verbatim)
    - nd2studios/core/analysis_registry.py        (AnalysisResult dataclass —
      MINIMAL vendored copy: only the fields the (de)serialization touches)

VENDORED VERBATIM; imports nothing from nd2studios; caller owns all prep.

CONTRACT
    save_checkpoints(dir_path, store, signature=None) -> int
    load_checkpoints(dir_path) -> (store, signature)

    Serializes an in-memory checkpoint store {node_id: {"hash", "data"}} to a
    <pipeline>.checkpoints/ directory (one compressed NPZ of label masks per
    checkpoint node + an index.json manifest) and rebuilds it. Returns ({}, None)
    on an absent / foreign / newer cache.

DROPPED UI/REGISTRY-ONLY MEMBERS
    From analysis_registry.py, the AnalysisResult copy keeps ONLY the fields the
    checkpoint (de)serialization reads/writes: label_masks, secondary_label_masks,
    overlay_color, overlay_alpha, overlay_outline, secondary_overlay_color,
    secondary_overlay_alpha, volumetric_voxel_counts. The compute/UI-only
    `measurements` and `summary` fields and the entire AnalysisPipeline base
    class (abstract run/get_params/register — pure registry + UI plumbing) are
    NOT on the checkpoint (de)serialization path and were dropped.

CHANGES FROM SOURCE (permitted edits only)
    - Removed `from nd2studios.core.analysis_registry import AnalysisResult`
      (AnalysisResult is now defined in-file).
    - In checkpoints_dir_for(), removed the lazy
      `from nd2studios.pipeline_graph.io import PIPELINE_EXTENSION` and inlined
      the constant value ".nd2s_pipeline.json" as PIPELINE_EXTENSION.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class AnalysisResult:
    """Minimal vendored copy — only the fields checkpoint (de)serialization touches.

    A value in ``label_masks`` may be an in-RAM ndarray OR a disk-backed lazy
    reader; both support ``[t]`` per-frame indexing and ``.shape``.
    """
    label_masks: Dict[str, Any] = field(default_factory=dict)
    overlay_color: Optional[tuple] = None
    overlay_alpha: float = 0.45
    overlay_outline: bool = False
    secondary_label_masks: Dict[str, np.ndarray] = field(default_factory=dict)
    secondary_overlay_color: Optional[tuple] = None
    secondary_overlay_alpha: float = 0.3
    volumetric_voxel_counts: Optional[Dict[str, Dict[Tuple[int, int], int]]] = None


PIPELINE_EXTENSION = ".nd2s_pipeline.json"
CHECKPOINT_CACHE_KIND = "nd2studios.checkpoints"
CHECKPOINT_CACHE_VERSION = 1
_INDEX_NAME = "index.json"


def _safe(name: str) -> str:
    """Filesystem-safe token from a node id (ids are already ``node-<hex>``)."""
    return "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(name))


def _json_default(o: Any) -> Any:
    """Coerce numpy scalars / arrays / sets so measurement rows serialize."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


def checkpoints_dir_for(pipeline_path: str) -> str:
    """Companion cache directory for a pipeline file path.

    ``…/foo.nd2s_pipeline.json`` → ``…/foo.checkpoints``.
    """
    if pipeline_path.endswith(PIPELINE_EXTENSION):
        base = pipeline_path[:-len(PIPELINE_EXTENSION)]
    else:
        base = os.path.splitext(pipeline_path)[0]
    return base + ".checkpoints"


def save_checkpoints(
    dir_path: str,
    store: Dict[str, Dict[str, Any]],
    signature: Optional[Dict[str, Any]] = None,
) -> int:
    """Write ``store`` (the page's ``_checkpoint_store``) to ``dir_path``.

    ``store`` is ``{node_id: {"hash": str, "data": <snapshot>}}`` where the
    snapshot is the dict built by ``PipelinesPage._run_checkpoint``. Label-mask
    arrays (which may be lazy disk-backed readers pointing at a session scratch
    that won't survive) are materialized and stored compressed. Returns the
    number of checkpoints written.
    """
    if os.path.isdir(dir_path):
        shutil.rmtree(dir_path, ignore_errors=True)
    os.makedirs(dir_path, exist_ok=True)

    checkpoints: Dict[str, Any] = {}
    for node_id, entry in (store or {}).items():
        snap = entry.get("data") or {}
        arrays: Dict[str, np.ndarray] = {}
        counter = [0]

        def _stash(arr: Any) -> Optional[str]:
            try:
                a = np.asarray(arr)
            except Exception:  # noqa: BLE001 — skip a mask that won't materialize
                return None
            key = f"a{counter[0]}"
            counter[0] += 1
            arrays[key] = a
            return key

        results_by_m = snap.get("results_by_m") or {}
        res_list = []
        # Preserve object identity → m so ctx_result can be re-pointed on load.
        ctx_result = snap.get("ctx_result")
        ctx_m: Optional[int] = None
        for m, res in sorted(results_by_m.items()):
            if res is ctx_result:
                ctx_m = int(m)
            ch_entries = []
            for ch, mask in (getattr(res, "label_masks", {}) or {}).items():
                key = _stash(mask)
                if key is not None:
                    ch_entries.append({"channel": ch, "key": key})
            sec_entries = []
            for nm, mask in (getattr(res, "secondary_label_masks", {}) or {}).items():
                key = _stash(mask)
                if key is not None:
                    sec_entries.append({"name": nm, "key": key})
            vox = getattr(res, "volumetric_voxel_counts", None)
            vox_json = None
            if vox:
                vox_json = {
                    ch: [[int(f), int(l), int(c)] for (f, l), c in d.items()]
                    for ch, d in vox.items()
                }
            oc = getattr(res, "overlay_color", None)
            soc = getattr(res, "secondary_overlay_color", None)
            res_list.append({
                "m": int(m),
                "channels": ch_entries,
                "secondary": sec_entries,
                "overlay_color": list(oc) if oc is not None else None,
                "overlay_alpha": float(getattr(res, "overlay_alpha", 0.45)),
                "overlay_outline": bool(getattr(res, "overlay_outline", False)),
                "secondary_overlay_color": list(soc) if soc is not None else None,
                "secondary_overlay_alpha": float(
                    getattr(res, "secondary_overlay_alpha", 0.3)),
                "voxel_counts": vox_json,
            })

        cmap = snap.get("track_colormap")
        crop = snap.get("results_crop")
        checkpoints[node_id] = {
            "hash": entry.get("hash"),
            "results_crop": list(crop) if crop else None,
            "results_by_m": res_list,
            "ctx_result_m": ctx_m,
            "all_rows": snap.get("all_rows") or [],
            "results_rows": snap.get("results_rows") or [],
            "ctx_rows": snap.get("ctx_rows") or [],
            "track_colormap": ({str(k): list(v) for k, v in cmap.items()}
                               if cmap else None),
            "track_long_ids": sorted(int(x) for x in
                                     (snap.get("track_long_ids") or set())),
            "track_overlay_rows": snap.get("track_overlay_rows") or [],
        }
        if arrays:
            np.savez_compressed(
                os.path.join(dir_path, f"{_safe(node_id)}.npz"), **arrays)

    index = {
        "kind": CHECKPOINT_CACHE_KIND,
        "version": CHECKPOINT_CACHE_VERSION,
        "signature": signature or {},
        "checkpoints": checkpoints,
    }
    with open(os.path.join(dir_path, _INDEX_NAME), "w", encoding="utf-8") as f:
        json.dump(index, f, default=_json_default)
    return len(checkpoints)


def load_checkpoints(
    dir_path: str,
) -> Tuple[Dict[str, Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Read a checkpoint cache written by :func:`save_checkpoints`.

    Returns ``(store, signature)`` in the same shape the page's
    ``_checkpoint_store`` uses (so it can be assigned directly), or ``({}, None)``
    when the directory is absent / not a checkpoint cache / newer than this build.
    """
    index_path = os.path.join(dir_path, _INDEX_NAME)
    if not os.path.isfile(index_path):
        return {}, None
    with open(index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    if not isinstance(index, dict) or index.get("kind") != CHECKPOINT_CACHE_KIND:
        return {}, None
    if int(index.get("version", 0)) > CHECKPOINT_CACHE_VERSION:
        return {}, None

    signature = index.get("signature") or None
    store: Dict[str, Dict[str, Any]] = {}
    for node_id, cp in (index.get("checkpoints") or {}).items():
        npz_path = os.path.join(dir_path, f"{_safe(node_id)}.npz")
        arrays: Dict[str, np.ndarray] = {}
        if os.path.isfile(npz_path):
            with np.load(npz_path) as z:
                arrays = {k: z[k] for k in z.files}

        results_by_m: Dict[int, AnalysisResult] = {}
        for r in cp.get("results_by_m") or []:
            label_masks = {}
            for ce in r.get("channels") or []:
                arr = arrays.get(ce.get("key"))
                if arr is not None:
                    label_masks[ce["channel"]] = arr
            secondary = {}
            for se in r.get("secondary") or []:
                arr = arrays.get(se.get("key"))
                if arr is not None:
                    secondary[se["name"]] = arr
            vox = None
            if r.get("voxel_counts"):
                vox = {
                    ch: {(int(f), int(l)): int(c) for f, l, c in lst}
                    for ch, lst in r["voxel_counts"].items()
                }
            oc = r.get("overlay_color")
            soc = r.get("secondary_overlay_color")
            res = AnalysisResult(
                label_masks=label_masks,
                secondary_label_masks=secondary,
                overlay_color=tuple(oc) if oc else None,
                overlay_alpha=float(r.get("overlay_alpha", 0.45)),
                overlay_outline=bool(r.get("overlay_outline", False)),
                secondary_overlay_color=tuple(soc) if soc else None,
                secondary_overlay_alpha=float(r.get("secondary_overlay_alpha", 0.3)),
                volumetric_voxel_counts=vox,
            )
            results_by_m[int(r["m"])] = res

        ctx_m = cp.get("ctx_result_m")
        ctx_result = (results_by_m.get(int(ctx_m)) if ctx_m is not None
                      else next(iter(results_by_m.values()), None))
        cmap = cp.get("track_colormap")
        crop = cp.get("results_crop")
        snap = {
            "results_by_m": results_by_m,
            "all_rows": cp.get("all_rows") or [],
            "results_rows": cp.get("results_rows") or [],
            "ctx_rows": cp.get("ctx_rows") or [],
            "ctx_result": ctx_result,
            "ctx_results_by_m": dict(results_by_m),
            "results_crop": tuple(crop) if crop else None,
            "track_colormap": ({int(k): tuple(v) for k, v in cmap.items()}
                               if cmap else None),
            "track_long_ids": set(int(x) for x in
                                  (cp.get("track_long_ids") or [])),
            "track_overlay_rows": cp.get("track_overlay_rows") or [],
        }
        store[node_id] = {"hash": cp.get("hash"), "data": snap}
    return store, signature
