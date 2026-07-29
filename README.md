# NodeLab — ND2Studios node editor

NodeLab is a **Blender-geometry-nodes-style node editor** for microscopy image analysis:
build acquire → enhance → segment → measure → track pipelines by wiring nodes on a
pannable, zoomable canvas, tune each node's parameters, and pull results through a lazy
streaming engine.

```
pip install -r requirements.txt
python run.py
```

**Docs:** [MANUAL.md](MANUAL.md) — how to use it (window, workflows, all 54 nodes, shortcuts,
troubleshooting). [CodeLog/Architecture/ENGINEERING_NOTES.md](CodeLog/Architecture/ENGINEERING_NOTES.md)
— how it works (data model, node contracts, domains/bridges, metadata intelligence, engine,
memo, streaming, GUI seam, invariants).

## What makes it different

- **Metadata-intelligent nodes.** Every spatial/temporal parameter carries a unit and a
  *derivation* from the image's own calibration — a radius in µm becomes the right pixel
  count for *this* objective, and a deconvolution PSF is derived from the emission
  wavelength and NA per channel. Override any of them and the override sticks.
- **One 2D/3D lever.** A node declares its data-access footprint per dimensionality, so
  the same graph runs plane-wise or truly volumetric; z==1 data greys the lever out.
- **Lazy per-tile streaming eval.** Computes return chained lazy providers, not realized
  6-D arrays — with overlap-recompute halos, a byte-budget tile/field cache, and
  tree-reduced projections. A 6554² plane never lands in RAM whole unless something needs it.
- **Two-hash memoization.** A structural recipe hash plus a content fingerprint means an
  unrelated edit recomputes only the invalidated chain, with a byte-budget LRU GC.
- **Domains, not just images.** Voxel, Point, Label, Track, Mesh, Frame and more, with
  generated transfer lattices and structure bridges between them.
- **Zones and groups.** Repeat/Simulation zones (including per-frame-T) and nestable node
  groups, unrolled into the flat graph so the engine needs no special cases.

## Layout

```
ND2Studios_Blender/
├── run.py                  # launcher
├── requirements.txt
├── nodegraph/              # the ENGINE (Qt-free)
│   ├── domains.py …        # domains, dataset, transfer, bridges, structure, field
│   ├── engine.py           # lazy pull + ReadContext + granularity routing
│   ├── streaming.py        # per-tile streaming providers + tile/field cache
│   ├── memo.py             # two-hash memo + byte-budget LRU
│   ├── registry.py         # NodeSpec, the 2D/3D DimMode lever, Granularity
│   ├── nodes.py            # the node catalog
│   ├── zones.py groups.py  # unroll / expand
│   ├── serialize.py        # *.nd2graph.json
│   ├── selftest.py         # `python -m nodegraph.selftest` — the core gate
│   └── kernels/            # vendored pure-compute analysis kernels + .md contracts
├── nodelab_v2/             # the GUI (PySide6; Qt lives only here)
│   ├── document.py         # Qt-free editing model: wiring rules, metadata propagation
│   ├── runner.py           # QThreadPool + epoch-registry engine bridge
│   ├── scene.py node_item.py edge_item.py frame_item.py minimap.py
│   ├── viewer.py glview.py overlays.py   # multi-channel viewer, GPU path, overlays
│   ├── spreadsheet.py export.py          # tables + CSV / Parquet / Arrow
│   ├── ingest.py nd2_meta.py             # ND2 / TIFF → engine + calibration
│   └── window.py app.py theme.py …
├── CodeLog/                # design records, handoffs, changelog
└── scripts/                # verification probes + benchmarks
```

## Gates

```
PYTHONUTF8=1 python -m nodegraph.selftest                      # headless core
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png  # driven GUI probe
```

`PYTHONUTF8=1` on Windows only because some `[ok]` lines carry `µ`/`σ`/`↔` glyphs a cp1252
console chokes on. Heavier, not in the fast gate: `scripts/_ingest_nd2_smoke.py` (real ND2),
`scripts/_bench_provider_granularity.py` (the storage-layout keystone benchmark).

## Adding a node

Read the **`wire-node-v2`** skill (concepts) then follow **`build-node-v2`** (procedure).
A node is `register_node(compute, op_key=..., **spec)` in [nodegraph/nodes.py](nodegraph/nodes.py):
declare a per-dim `granularity`/`kernel_axes` footprint, put a unit + derivation on every
spatial parameter, add a `meta_transform` if it changes axes or calibration, read
calibration through `ctx.calib` so it folds into the memo, and land an end-to-end pull in
[nodegraph/selftest.py](nodegraph/selftest.py).

## Requirements

Python 3.13, PySide6, numpy, scipy, scikit-image, blosc2, zarr, pyarrow, nd2, tifffile.
Optional, feature-gated at import: scikit-learn (point clustering), al-dic (2D DIC),
tensorflow + stardist and cellSAM + torch (the learned Segmentation methods),
numba + pandas (the 5-method tracker), cupy.

**CellSAM** needs its weights fetched once per machine (~1.7 GB, non-commercial academic
licence, so they cannot ship here):

```bash
pip install git+https://github.com/vanvalenlab/cellSAM.git
python scripts/setup_cellsam.py     # prompts for a users.deepcell.org token, verifies md5
python scripts/_cellsam_smoke.py    # end-to-end check
```

Loading is offline afterwards. If your machine intercepts HTTPS (corporate proxy /
antivirus), the script detects it and points you at `pip install truststore`. See
[MANUAL.md §1](MANUAL.md#1-install--launch).

## History

A first-generation editor (`nodelab` on a vendored `nd2studios`/`pipeline_kit` backend)
was removed on 2026-07-29 after its remaining unported nodes were declared obsolete;
[CodeLog/ClaudesPlan/V2.05_phase7_capability_matrix.md](CodeLog/ClaudesPlan/V2.05_phase7_capability_matrix.md)
§6 records exactly what went with it. The `_v2` in `nodelab_v2/` is a leftover of that split.
