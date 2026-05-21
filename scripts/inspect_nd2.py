#!/usr/bin/env python3
"""
ND2 metadata diagnostic.

Prints everything ND2Studios uses to build the tile layout, plus a few
extra sanity numbers, so we can see what's actually in the file when
the layout looks wrong on a real acquisition.

Usage (Windows, from the project root):

    python scripts/inspect_nd2.py "D:\\McGhee Lab\\TF_ELISA_2025_Data\\ERK_Scouting\\20260305_223016_664\\Well2_Channelp53-GFP,ERK-mRuby2,H2B-iRFP670,TD_Seq0000.nd2"

If you don't pass a path, the script searches the current directory
for the first .nd2 file.

What it prints
--------------
1. ``f.sizes`` and the dim order — tells us what axes the file has and
   in what order, which determines the flat-index math.
2. Voxel size (pixel size in µm + Z step).
3. The first 5 channel names + per-channel exposure / em / ex.
4. **Per-M stage XY positions** — all of them, plus a short summary of
   uniqueness, X spread, Y spread, smallest / largest gap.
5. Inferred grid: how many distinct columns and rows the bimodal
   clustering finds, and what tolerance it picked.
6. The full ``compute_tile_layout`` result: canvas size, source flag,
   first few offsets.

Paste the output back to the chat and we can see where things go wrong.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def _safe(get, default=None):
    try:
        return get()
    except Exception:
        return default


def _find_nd2_in_cwd() -> str | None:
    here = Path.cwd()
    for p in here.glob("*.nd2"):
        return str(p)
    for p in here.rglob("*.nd2"):
        return str(p)
    return None


def main() -> int:
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        found = _find_nd2_in_cwd()
        if not found:
            print("usage: python scripts/inspect_nd2.py <file.nd2>")
            return 2
        path = found

    if not os.path.exists(path):
        print(f"Path does not exist: {path}")
        return 2

    # Make the project's nd2_loader / stitch_exporter importable.
    project_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(project_root))

    print("=" * 72)
    print(f"File: {path}")
    print(f"Size on disk: {os.path.getsize(path) / (1024 ** 3):.2f} GiB")
    print("=" * 72)

    import nd2
    import numpy as np

    with nd2.ND2File(path) as f:
        sizes = dict(f.sizes)
        dim_order = list(sizes.keys())
        print()
        print("--- file shape ---")
        print(f"sizes        : {sizes}")
        print(f"dim_order    : {dim_order}")
        print(f"dtype        : {f.dtype}")
        print(f"H × W (px)   : {sizes.get('Y')} × {sizes.get('X')}")

        vox = _safe(f.voxel_size)
        if vox is not None:
            print(f"pixel size µm: x={getattr(vox, 'x', '?'):.4f}  "
                  f"y={getattr(vox, 'y', '?'):.4f}  z={getattr(vox, 'z', '?'):.4f}")
        else:
            print("pixel size µm: <not available>")

        # Tile size in µm — determines the cluster-tolerance cap.
        try:
            px = float(getattr(vox, "x", 1.0))
            tile_um_x = sizes.get('X', 0) * px
            tile_um_y = sizes.get('Y', 0) * px
            print(f"tile size µm : {tile_um_x:.1f} × {tile_um_y:.1f}")
        except Exception:
            pass

        print()
        print("--- channels ---")
        try:
            channels = list(f.metadata.channels) or []
        except Exception:
            channels = []
        for i, ch in enumerate(channels[:5]):
            name = str(_safe(lambda c=ch: c.channel.name) or f"Ch{i}")
            em = _safe(lambda c=ch: float(c.channel.emissionLambdaNm))
            ex = _safe(lambda c=ch: float(c.channel.excitationLambdaNm))
            exp_ms = (
                _safe(lambda c=ch: float(c.exposureTimeMs))
                or _safe(lambda c=ch: float(c.channel.exposureTimeMs))
            )
            print(f"  C{i}: {name!r:30}  em={em}  ex={ex}  exp_ms={exp_ms}")
        if len(channels) > 5:
            print(f"  …and {len(channels) - 5} more")

        # --- Per-M stage XY (this is the heart of the layout problem) ---
        print()
        print("--- per-M stage XY ---")

        seq_dims = [d for d in dim_order if d not in ("Y", "X")]
        m_axis = "P" if "P" in seq_dims else ("M" if "M" in seq_dims else None)
        n_m = sizes.get(m_axis, 1) if m_axis else 1
        print(f"M axis: {m_axis!r}  (n_multipoints = {n_m})")

        def flat_index(coords):
            idx = 0
            for d in seq_dims:
                idx = idx * sizes.get(d, 1) + int(coords.get(d, 0))
            return idx

        # Pull (x, y) for every M.
        xys = []
        first_failure = None
        for m in range(n_m):
            coords = {"T": 0, "Z": 0, "C": 0}
            if m_axis is not None:
                coords[m_axis] = m
            i = flat_index(coords)
            fm = _safe(lambda idx=i: f.frame_metadata(idx))
            if fm is None:
                if first_failure is None:
                    first_failure = (m, i)
                xys.append(None)
                continue
            pos = (
                _safe(lambda meta=fm: meta.channels[0].position) or
                _safe(lambda meta=fm: meta.position)
            )
            if pos is None:
                if first_failure is None:
                    first_failure = (m, i)
                xys.append(None)
                continue
            sx = _safe(lambda p=pos: float(p.stagePositionUm.x))
            sy = _safe(lambda p=pos: float(p.stagePositionUm.y))
            xys.append((sx, sy) if (sx is not None and sy is not None) else None)

        valid = [t for t in xys if t is not None]
        print(f"valid positions: {len(valid)} / {n_m}")
        if first_failure is not None:
            m, i = first_failure
            print(f"first failure  : M={m}  flat_index={i}  "
                  f"(maybe an out-of-range index?)")

        if valid:
            xs = np.array([t[0] for t in valid])
            ys = np.array([t[1] for t in valid])
            print(f"X range µm     : {xs.min():.2f} … {xs.max():.2f}  "
                  f"(spread {xs.max() - xs.min():.2f})")
            print(f"Y range µm     : {ys.min():.2f} … {ys.max():.2f}  "
                  f"(spread {ys.max() - ys.min():.2f})")
            ux = sorted(set(round(x, 1) for x in xs))
            uy = sorted(set(round(y, 1) for y in ys))
            print(f"unique X (round to 0.1µm): {len(ux)}  "
                  f"first 8: {[round(v, 2) for v in ux[:8]]}")
            print(f"unique Y (round to 0.1µm): {len(uy)}  "
                  f"first 8: {[round(v, 2) for v in uy[:8]]}")

            # Print all positions with their M index.
            print()
            print("M, x_µm, y_µm:")
            for m, t in enumerate(xys):
                if t is None:
                    print(f"  M={m:>3}  <missing>")
                else:
                    print(f"  M={m:>3}  x={t[0]:>10.2f}  y={t[1]:>10.2f}")

            # Compute clustering with the V1.6 algorithm.
            print()
            print("--- V1.6 clustering result ---")
            try:
                from nd2studios.backend.exporters.stitch_exporter import (
                    _cluster_axis, compute_tile_layout,
                )
                cap_x = (sizes.get('X', 0) * float(getattr(vox, 'x', 1.0))) * 0.5
                cap_y = (sizes.get('Y', 0) * float(getattr(vox, 'y', 1.0))) * 0.5
                col_centroids = _cluster_axis(xs, max_tolerance=cap_x)
                row_centroids = _cluster_axis(ys, max_tolerance=cap_y)
                print(f"col cap µm     : {cap_x:.2f}")
                print(f"row cap µm     : {cap_y:.2f}")
                print(f"col centroids  : {[round(c, 2) for c in col_centroids]}  "
                      f"({len(col_centroids)} columns)")
                print(f"row centroids  : {[round(r, 2) for r in row_centroids]}  "
                      f"({len(row_centroids)} rows)")

                # Show pairwise gap statistics (helps diagnose bimodality).
                xs_sorted = np.sort(np.unique(np.round(xs, 3)))
                ys_sorted = np.sort(np.unique(np.round(ys, 3)))
                if len(xs_sorted) >= 2:
                    dx = np.diff(xs_sorted)
                    print(f"X unique-gaps  : min={dx.min():.2f}  median={np.median(dx):.2f}  "
                          f"max={dx.max():.2f}  count={len(dx)}")
                if len(ys_sorted) >= 2:
                    dy = np.diff(ys_sorted)
                    print(f"Y unique-gaps  : min={dy.min():.2f}  median={np.median(dy):.2f}  "
                          f"max={dy.max():.2f}  count={len(dy)}")

                tile_h = sizes.get('Y', 1)
                tile_w = sizes.get('X', 1)
                px_um = float(getattr(vox, 'x', 1.0))
                layout = compute_tile_layout(
                    [t for t in xys if t is not None],
                    pixel_size_um=px_um,
                    tile_h=tile_h, tile_w=tile_w,
                )
                n_canvas_cols = layout.canvas_w // max(layout.tile_w, 1)
                n_canvas_rows = layout.canvas_h // max(layout.tile_h, 1)
                n_unique = len(set(layout.offsets))
                print()
                print(f"layout source  : {layout.source}")
                print(f"layout canvas  : {n_canvas_cols} cols × {n_canvas_rows} rows")
                print(f"unique offsets : {n_unique}  (expected {len(valid)})")
                print(f"first 5 offsets: {layout.offsets[:5]}")
            except Exception as e:
                import traceback
                print(f"compute_tile_layout failed: {e}")
                traceback.print_exc()

    print()
    print("=" * 72)
    print("Done. Paste this whole output back to the chat.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
