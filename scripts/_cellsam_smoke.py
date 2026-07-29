"""CellSAM end-to-end smoke check — does `analysis.segment` (method=cellsam) really run?

    python scripts/_cellsam_smoke.py                       # synthetic cells
    python scripts/_cellsam_smoke.py --image plane.tif     # your own 2-D plane
    python scripts/_cellsam_smoke.py --model-path C:/w/cellsam_general.pt
    python scripts/_cellsam_smoke.py --bbox 0.25 --tile    # knobs + tiled inference

Checks, in order, and stops at the first thing that is missing:

1. the ``cellSAM`` package and its deps are importable;
2. the weights are reachable — either already cached in ``~/.deepcell/models``, or
   downloadable, which needs ``DEEPCELL_ACCESS_TOKEN`` (https://users.deepcell.org, the
   models are licensed for **non-commercial academic use**). ``--model-path`` skips both;
3. the model loads (once — the kernel caches a process singleton);
4. the REAL node runs through the REAL Engine and produces a Voxel label raster + a Label
   table with globally-unique ids.

Prints timings, because CPU inference is ~12 s per image (the paper's own benchmark) and
scales with cell count: CellSAM's mask decoder runs once per detected cell, unlike
Cellpose's single pass. The token is never printed.
"""
from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402


def synthetic(h: int = 256, w: int = 256, n: int = 12) -> np.ndarray:
    """A deterministic field of blurred elliptical "cells" on a dim background."""
    yy, xx = np.mgrid[0:h, 0:w]
    img = np.full((h, w), 40.0)
    # a fixed lattice with a reproducible jitter — no RNG seed to argue about
    k = 0
    for gy in range(3):
        for gx in range(4):
            if k >= n:
                break
            cy, cx = 45 + gy * 80 + (gx % 2) * 9, 40 + gx * 60 + (gy % 2) * 7
            ry, rx = 20 + (k % 3) * 3, 15 + (k % 4) * 3
            body = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            img[body <= 1.0] = 210.0 - (k % 5) * 12          # cell interior
            img[(body > 1.0) & (body <= 1.25)] = 255.0       # a brighter rim
            k += 1
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(img, 1.2).astype(np.uint16)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", default="", help="a 2-D image (tif/png) instead of synthetic")
    ap.add_argument("--model-path", default="", help="local CellSAM .pt (skips the download)")
    ap.add_argument("--model", default="cellsam_general",
                    help="cellsam_general | cellsam_extra")
    ap.add_argument("--bbox", type=float, default=0.4, help="bbox_threshold (0-1)")
    ap.add_argument("--tile", action="store_true", help="tiled inference (large FOVs)")
    ap.add_argument("--tile-size", type=int, default=512)
    ap.add_argument("--tile-overlap", type=int, default=56)
    ap.add_argument("--min-area", type=float, default=0.0, help="µm² size floor")
    ap.add_argument("--pixel-size", type=float, default=0.325, help="µm per pixel")
    a = ap.parse_args(argv)

    def die(msg: str) -> int:
        print("\n[FAIL] " + msg)
        return 1

    # ── 1. the package ────────────────────────────────────────────────────────
    from nodegraph.kernels import cellsam_segment as CS
    print("== 1. package ==")
    if not CS.cellsam_available():
        return die("cellSAM is not importable.\n"
                   "    pip install git+https://github.com/vanvalenlab/cellSAM.git")
    import cellSAM
    print(f"   cellSAM {getattr(cellSAM, '__version__', '?')}  device -> "
          f"{CS.resolve_device()}   (override with {CS.DEVICE_ENV}=cpu|cuda|auto)")

    # ── 2. the weights ────────────────────────────────────────────────────────
    from pathlib import Path
    print("== 2. weights ==")
    if a.model_path:
        if not Path(a.model_path).is_file():
            return die(f"--model-path {a.model_path!r} is not a file")
        print(f"   local checkpoint: {a.model_path}  (no token needed)")
    else:
        cache = Path.home() / ".deepcell" / "models"
        cached = sorted(cache.glob("cellsam_v*/*.pt"))
        tok = os.environ.get("DEEPCELL_ACCESS_TOKEN") or ""
        if cached:
            print("   already cached:", ", ".join(str(p.name) for p in cached))
        elif not tok:
            return die(
                "no cached weights and DEEPCELL_ACCESS_TOKEN is not set (in THIS process).\n"
                "    1. sign in at https://users.deepcell.org/login/ and create a token\n"
                "       (the models are licensed for NON-COMMERCIAL ACADEMIC use)\n"
                "    2. paste the token with NO quotes-and-brackets around it:\n"
                '         $env:DEEPCELL_ACCESS_TOKEN = "abc123..."   <- this session, now\n'
                '         setx DEEPCELL_ACCESS_TOKEN "abc123..."     <- persist it\n'
                "       `setx` writes the registry and does NOT touch the shell you type it\n"
                "       in, so do BOTH (or reopen PowerShell after setx).\n"
                "    3. re-run this script — the first load downloads 1.7 GB to\n"
                "       ~/.deepcell/models and is cached (md5-checked) forever after.\n"
                "    ...or pass --model-path <a local cellsam .pt> to skip the download.")
        else:
            # A pasted placeholder is the likeliest failure and it otherwise surfaces as an
            # opaque 403 a gigabyte later, so name it here.
            bad = []
            if tok[0] in "<\"'" or tok[-1] in ">\"'":
                bad.append("it is wrapped in <> or quotes — paste the token bare")
            if tok != tok.strip():
                bad.append("it has leading/trailing whitespace")
            if bad:
                return die("DEEPCELL_ACCESS_TOKEN looks malformed: "
                           + "; ".join(bad)
                           + f"\n    (length {len(tok)}; the value is never printed)")
            print(f"   token set ({len(tok)} chars); first load downloads 1.7 GB to "
                  f"~/.deepcell/models")

    # ── 3. the model ──────────────────────────────────────────────────────────
    print("== 3. model load ==")
    t0 = time.time()
    try:
        CS.get_cellsam_model(a.model, model_path=a.model_path)
    except Exception as exc:                       # noqa: BLE001 — this is the report
        return die(f"{type(exc).__name__}: {exc}")
    print(f"   loaded in {time.time() - t0:.1f}s (cached for the rest of the process)")

    # ── 4. the node, through the real Engine ──────────────────────────────────
    print("== 4. analysis.segment (method=cellsam) ==")
    from nodegraph.dataset import AxisSizes, Dataset
    from nodegraph.domains import Domain
    from nodegraph.engine import Engine
    from nodegraph.graph import Graph, NodeInstance
    from nodegraph.metadata import MetaEnvelope
    from nodegraph.nodes import COMPUTES
    from nodegraph.provider import ArrayProvider
    from nodegraph.registry import OutDataset, define_node

    if a.image:
        if a.image.lower().endswith((".tif", ".tiff")):
            import tifffile
            plane = np.asarray(tifffile.imread(a.image))
        else:
            from skimage.io import imread
            plane = np.asarray(imread(a.image))
        while plane.ndim > 2:                      # take the first plane of anything deeper
            plane = plane[0]
        print(f"   image: {a.image}  {plane.shape} {plane.dtype}")
    else:
        plane = synthetic()
        print(f"   image: synthetic {plane.shape} {plane.dtype}, 12 planted cells")

    h, w = plane.shape
    ax = AxisSizes(m=1, t=1, z=1, c=1, y=h, x=w)
    optics = {"pixel_size_um": a.pixel_size, "bit_depth": 16}
    ds = Dataset(axes=ax, metadata=dict(optics)).with_image(
        ArrayProvider(plane.reshape(1, 1, 1, 1, h, w)))
    define_node("io.cssmoke", "Seed", outputs=[OutDataset()])
    g = Graph()
    g.add(NodeInstance("S", "io.cssmoke"))
    g.add(NodeInstance("N", "analysis.segment",
                       modes={"dim": "2D", "method": "cellsam"},
                       params={"bbox_threshold": a.bbox, "cellsam_model": a.model,
                               "model_path": a.model_path, "tile": bool(a.tile),
                               "tile_size": a.tile_size, "tile_overlap": a.tile_overlap,
                               "min_area": a.min_area}))
    g.connect("S", "N")
    eng = Engine(g, computes=COMPUTES, seeds={"S": ds},
                 meta_seeds={"S": MetaEnvelope(axes=ax, metadata=optics)})
    t0 = time.time()
    try:
        out = eng.pull("N")
    except Exception as exc:                       # noqa: BLE001 — this is the report
        return die(f"{type(exc).__name__}: {exc}")
    dt = time.time() - t0

    raster = out.get(Domain.VOXEL, "labels")
    ids = out.get(Domain.LABEL, "id", layer="labels")
    area = out.get(Domain.LABEL, "area", layer="labels")
    n = int(np.asarray(raster.values).max()) if raster is not None else 0
    print(f"   segmented in {dt:.1f}s  ->  {n} cell(s)")
    if area is not None and len(area.values):
        px2 = a.pixel_size ** 2
        areas = np.asarray(area.values)
        print(f"   area px²: min {areas.min()} / median {int(np.median(areas))} / "
              f"max {areas.max()}   (µm²: {areas.min()*px2:.1f} / "
              f"{np.median(areas)*px2:.1f} / {areas.max()*px2:.1f})")
    print(f"   provenance: segment_method={out.metadata.get('segment_method')!r} "
          f"segment_model={out.metadata.get('segment_model')!r}")

    if n == 0:
        print("\n[WARN] the model ran but found nothing. On real data try a lower "
              "--bbox (0.2-0.3) for out-of-distribution images; on the synthetic "
              "fixture this means the weights loaded but are not behaving.")
        return 2
    if ids is not None:
        got = sorted(np.asarray(ids.values).tolist())      # AttributeLayer -> its column
        assert got == list(range(1, n + 1)), \
            f"ids must be contiguous 1..K, got {got[:5]}...{got[-3:]}"
    print("\n[PASS] CellSAM is wired end to end: model -> node -> label raster + table.")
    print("       In the GUI: drop a Segmentation node, set method=cellsam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
