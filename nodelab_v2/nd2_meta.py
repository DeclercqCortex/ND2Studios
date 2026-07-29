"""ND2 extended-metadata reader — the optics/calibration parse behind v2 ingest.

Vendored **verbatim** (2026-07-29) from the retired v1 backend's
``nd2studios/backend/nd2_loader.py`` when NodeLab v1 was removed. It is the one piece of v1
the v2 app layer depended on: :mod:`nodelab_v2.ingest` calls
:func:`read_nd2_metadata_extended` for a real microscope file's calibration + per-channel
display metadata (pixel/z size, channel names, emission/excitation λ, native colour,
exposure, objective, binning, stage positions, frame timestamps).

**Do not re-derive this.** The ``nd2`` SDK exposes these fields inconsistently across
versions and file vintages, so every read goes through :func:`_safe` and the structure is
probed defensively. It was proven against the lab's real files; a "cleaner" rewrite loses
that. Qt-free; ``nd2`` is imported lazily inside the function.

Only the two functions v2 actually calls came across — the v1 ``ND2Metadata`` dataclass,
the pixel loaders, the Z-projection helper, and the JSON sidecar cache stayed behind with
v1 (v2 reads pixels via :func:`nodelab_v2.ingest.read_nd2` / ``nd2.to_dask()`` and caches
through the engine memo instead).
"""
from __future__ import annotations

from typing import Dict, List, Tuple


def _safe(get, default=None):
    """Run a getter that may explode on missing fields and return `default`."""
    try:
        return get()
    except Exception:
        return default


def read_nd2_metadata_extended(filepath):
    """Return a dict of all the metadata fields ND2Studios surfaces.

    Always-present keys (best-effort, may be empty):
        filepath, dim_order, sizes, dtype, height, width,
        n_timepoints, n_zslices, n_channels, n_multipoints,
        pixel_size_um, z_step_um, voxel_size_um,
        channel_names, channel_colors, channel_emission_nm,
        channel_excitation_nm, channel_exposure_ms,
        objective_name, objective_magnification, objective_na,
        objective_immersion, binning_x, binning_y,
        camera_name, microscope_name,
        acquisition_start, frame_timestamps_s,
        stage_xy_um, stage_z_um, loops
    """
    import nd2

    out = {"filepath": filepath}
    with nd2.ND2File(filepath) as f:
        sizes = dict(f.sizes)
        out["sizes"] = sizes
        out["dim_order"] = list(sizes.keys())
        out["dtype"] = str(f.dtype)
        out["height"] = sizes.get("Y", 0)
        out["width"] = sizes.get("X", 0)
        out["n_timepoints"] = sizes.get("T", 1)
        out["n_zslices"] = sizes.get("Z", 1)
        out["n_channels"] = sizes.get("C", 1)
        out["n_multipoints"] = sizes.get("P", sizes.get("M", 1))

        vox = _safe(f.voxel_size)
        if vox is not None:
            out["pixel_size_um"] = float(getattr(vox, "x", 1.0))
            out["z_step_um"] = float(getattr(vox, "z", 1.0))
            out["voxel_size_um"] = (
                float(getattr(vox, "z", 1.0)),
                float(getattr(vox, "y", out["pixel_size_um"])),
                float(getattr(vox, "x", out["pixel_size_um"])),
            )
        else:
            out["pixel_size_um"] = 1.0
            out["z_step_um"] = 1.0
            out["voxel_size_um"] = (1.0, 1.0, 1.0)

        meta = _safe(lambda: f.metadata)
        channels = _safe(lambda: list(meta.channels), default=[]) or []
        names = []
        colors = []
        emission = []
        excitation = []
        exposure_ms = []
        for ch in channels:
            ch_inner = getattr(ch, "channel", ch)
            names.append(str(getattr(ch_inner, "name", "Ch")))
            colors.append(_safe(lambda c=ch_inner: int(getattr(c, "colorRGB", None))))
            emission.append(_safe(lambda c=ch_inner: float(getattr(c, "emissionLambdaNm", None))))
            excitation.append(_safe(lambda c=ch_inner: float(getattr(c, "excitationLambdaNm", None))))
            exp = (
                _safe(lambda c=ch: float(getattr(c, "exposureTimeMs", None)))
                or _safe(lambda c=ch_inner: float(getattr(c, "exposureTimeMs", None)))
            )
            exposure_ms.append(exp)
        if not names:
            names = [f"Ch{i}" for i in range(out["n_channels"])]
        out["channel_names"] = names
        out["channel_colors"] = colors
        out["channel_emission_nm"] = emission
        out["channel_excitation_nm"] = excitation
        out["channel_exposure_ms"] = exposure_ms

        microscope = _safe(lambda: meta.channels[0].microscope) if channels else None
        if microscope is not None:
            out["objective_name"] = str(_safe(lambda: microscope.objectiveName) or "")
            out["objective_magnification"] = _safe(lambda: float(microscope.objectiveMagnification))
            out["objective_na"] = _safe(lambda: float(microscope.objectiveNumericalAperture))
            out["objective_immersion"] = str(_safe(lambda: microscope.immersionRefractiveIndex) or "")
        else:
            out["objective_name"] = ""
            out["objective_magnification"] = None
            out["objective_na"] = None
            out["objective_immersion"] = ""

        instrument = _safe(lambda: meta.channels[0].volume) if channels else None
        out["binning_x"] = _safe(lambda: int(instrument.cameraTransformationMatrix.binning))
        out["binning_y"] = out.get("binning_x")
        camera = _safe(lambda: meta.channels[0].instrument) if channels else None
        out["camera_name"] = str(_safe(lambda: camera.cameraName) or "")
        out["microscope_name"] = str(_safe(lambda: microscope.systemName) or "") if microscope else ""

        seq_dims = [d for d in out["dim_order"] if d not in ("Y", "X")]
        m_axis = "P" if "P" in seq_dims else ("M" if "M" in seq_dims else None)
        n_m = sizes.get(m_axis, 1) if m_axis else 1
        n_t = out["n_timepoints"]

        def _flat(coords: Dict[str, int]) -> int:
            idx = 0
            for d in seq_dims:
                idx = idx * sizes.get(d, 1) + int(coords.get(d, 0))
            return idx

        def _read_xy_from_experiment() -> List[Tuple[float, float]]:
            """Read planned stage XY positions from f.experiment (XYPosLoop)."""
            pts: List[Tuple[float, float]] = []
            for loop in (_safe(lambda: f.experiment, default=[]) or []):
                ltype = str(_safe(lambda lp=loop: lp.type) or "")
                if "XYPos" not in ltype:
                    continue
                points = (
                    _safe(lambda lp=loop: list(lp.parameters.points), default=[]) or []
                )
                for pt in points:
                    sx = _safe(lambda p=pt: float(p.stagePositionUm.x))
                    sy = _safe(lambda p=pt: float(p.stagePositionUm.y))
                    if sx is not None and sy is not None:
                        pts.append((sx, sy))
                if pts:
                    return pts
            return []

        # 1) Per-M stage positions — try f.experiment XYPosLoop first, then
        #    fall back to frame_metadata() per-M (which requires correct flat-
        #    index arithmetic). The experiment loop is the authoritative planned
        #    positions; frame_metadata is the actual per-frame readback.
        stage_xy: List[Tuple[float, float]] = []
        stage_z: List[float] = []

        stage_xy = _read_xy_from_experiment()
        xy_source_method = "experiment" if stage_xy else "frame_metadata"

        if not stage_xy and m_axis is not None:
            for m in range(n_m):
                coords = {m_axis: m, "T": 0, "Z": 0, "C": 0}
                seq_idx = _flat(coords)
                fm = _safe(lambda i=seq_idx: f.frame_metadata(i))
                if fm is None:
                    continue
                pos = (
                    _safe(lambda meta=fm: meta.channels[0].position) or
                    _safe(lambda meta=fm: meta.position)
                )
                if pos is None:
                    continue
                sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
                sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
                sz = _safe(lambda p=pos: float(p.stagePositionUm.z))
                if sx is not None and sy is not None:
                    stage_xy.append((sx, sy))
                if sz is not None:
                    stage_z.append(sz)
        elif not stage_xy:
            # Single-M file — record one nominal position from frame 0.
            fm0 = _safe(lambda: f.frame_metadata(0))
            if fm0 is not None:
                pos = (
                    _safe(lambda meta=fm0: meta.channels[0].position) or
                    _safe(lambda meta=fm0: meta.position)
                )
                if pos is not None:
                    sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
                    sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
                    if sx is not None and sy is not None:
                        stage_xy.append((sx, sy))

        n_xy_from_stage = len(stage_xy)

        # 2) Per-T frame timestamps.
        frame_ts: List[float] = []
        for t in range(n_t):
            coords = {"T": t, "Z": 0, "C": 0}
            if m_axis is not None:
                coords[m_axis] = 0
            seq_idx = _flat(coords)
            fm = _safe(lambda i=seq_idx: f.frame_metadata(i))
            if fm is None:
                continue
            ts = (
                _safe(lambda meta=fm: float(meta.channels[0].time.relativeTimeMs)) or
                _safe(lambda meta=fm: float(meta.relativeTimeMs))
            )
            if ts is not None:
                frame_ts.append(ts / 1000.0)

        if n_xy_from_stage == n_m and n_m > 0:
            stage_layout_source = f"stage_xy:{xy_source_method}"
        elif n_xy_from_stage > 0:
            stage_layout_source = f"partial:{n_xy_from_stage}/{n_m}:{xy_source_method}"
        else:
            stage_layout_source = "missing"

        out["frame_timestamps_s"] = frame_ts
        out["stage_xy_um"] = stage_xy
        out["stage_z_um"] = stage_z
        out["stage_layout_source"] = stage_layout_source
        out["acquisition_start"] = ""
        out["loops"] = [
            {
                "type": str(_safe(lambda lp=lp: lp.type) or ""),
                "count": int(_safe(lambda lp=lp: lp.count) or 0),
            }
            for lp in (_safe(lambda: f.experiment, default=[]) or [])
        ]

    return out


__all__ = ["read_nd2_metadata_extended"]
