# `cellsam_segment` — CellSAM (Segmentation node) integration contract

## 1. Purpose + where the real math lives

Segment cells in a **single, already-prepared 2-D plane** with CellSAM and return a
contiguous `int32` label image of the same `(H, W)` shape. Serves the `cellsam` method of
the `analysis.segment` node.

**The real math is EXTERNAL.** CellSAM is a Segment Anything ViT-B backbone whose features
feed two heads: **CellFinder**, an Anchor-DETR *set-prediction* detector that emits one
bounding box per cell (chosen precisely because it is NMS-free, which matters when cells
are densely packed), and a fine-tuned SAM mask decoder that turns those boxes into
instance masks. In other words CellSAM automates SAM's prompting, which is what makes it
usable without a human clicking every cell (Marks, Israel *et al.*, "CellSAM: a foundation
model for cell segmentation", *Nature Methods* 22:2585–2593, 2025;
<https://github.com/vanvalenlab/cellSAM>). All of that lives in the external `cellSAM` +
`segment_anything` + `torch` packages and the downloaded `cellsam_general` /
`cellsam_extra` weights.

The **in-repo glue** vendored here is only:

1. a **process-singleton** model loader keyed on `(model, model_path, device)`, so the
   checkpoint is read once per process rather than once per plane;
2. device resolution (`auto`/`cpu`/`cuda`) from an environment knob;
3. one `segment_cellular_image` call per plane — or `cellSAM.wsi.segment_wsi` when tiling;
4. normalization of four upstream quirks (§6);
5. `relabel_contiguous`: a bincount + LUT renumber to `1..K`, so the caller can offset ids
   into globally-unique ones.

This kernel is **2-D-per-plane**. All looping (T / Z / multipoint / channel), file I/O,
crop / downsample / enhancement, physical-unit size filtering and table building are the
caller's responsibility — `nodegraph/nodes.py::_compute_segment` owns them.

**Why not `cellsam_pipeline`.** It is the documented one-shot entry point but calls
`get_model()` on *every* invocation, so per-plane use re-reads the checkpoint for every
frame of a series. It also min-max normalizes in place, and its
`low_contrast_enhancement=True` branch is broken upstream
(`cellSAM.utils.enhance_low_contrast` assigns `model.bbox_threshold` with no `model` in
scope → `NameError`). This kernel drives the two lower-level entry points instead.

---

## 2. Entry points

```python
def segment_plane(
    image: np.ndarray,                 # (H, W)
    model=None,                        # loaded CellSAM; None -> the singleton
    *,
    bbox_threshold: float = 0.4,
    normalize: bool = True,
    postprocess: bool = False,
    remove_boundaries: bool = False,
    model_name: str = "cellsam_general",
    model_path: str = "",
    device: str = "",                  # "" -> $NODELAB_CELLSAM_DEVICE -> "auto"
    tile: bool = False,
    tile_size: int = 512,
    overlap: int = 56,
    iou_threshold: float = 0.5,
) -> np.ndarray: ...                   # (H, W) int32, ids 1..K, 0 = background

def get_cellsam_model(model: str = "cellsam_general", *, model_path: str = "",
                      device: str = "") -> "torch.nn.Module": ...

def resolve_device(requested: str = "") -> str: ...     # -> "cpu" | "cuda"

def relabel_contiguous(labels: np.ndarray) -> np.ndarray: ...   # pure numpy

# Availability probe (no import side effects):
def cellsam_available() -> bool: ...
```

`get_cellsam_model` returns a module-global singleton. `relabel_contiguous` is standalone
and needs numpy only. `cellsam_available()` is a `find_spec` probe — it says nothing about
whether the **weights** have been downloaded, which is a separate failure mode.

---

## 3. Inputs

### `segment_plane`

| name | Python type | shape | dtype | axis order | units | required? / default | meaning & constraints |
|------|-------------|-------|-------|------------|-------|---------------------|-----------------------|
| image | `np.ndarray` | `(H, W)` | any numeric | `(row=Y, col=X)` | intensity (a.u.) | **required** | One 2-D plane — one channel, one Z, one T. A non-2-D array raises `ValueError` (the caller owns the loop). Placed in CellSAM's whole-cell channel slot upstream (§6). |
| model | `nn.Module` or `None` | — | — | — | — | default `None` | Pass a loaded model to reuse it; `None` loads/uses the singleton. |

### `get_cellsam_model`

| name | type | default | meaning |
|------|------|---------|---------|
| model | `str` | `"cellsam_general"` | `cellsam_general` (the published generalist — use it to reproduce the paper) or `cellsam_extra` (extra training data; recommended for domains outside the paper). Ignored when `model_path` is set. |
| model_path | `str` | `""` | Local `.pt` checkpoint → `cellSAM.get_local_model`. Skips the download **and** the API token. |
| device | `str` | `""` | `auto` \| `cpu` \| `cuda`; `""` falls back to `$NODELAB_CELLSAM_DEVICE`, then `auto`. |

---

## 4. Parameters

| name | type | default | valid range / choices | semantics (effect on output) |
|------|------|---------|-----------------------|------------------------------|
| bbox_threshold | `float` | `0.4` | `[0, 1]` | CellFinder box-confidence cut — **the** precision/recall knob. Lower it for out-of-distribution images, raise it for cleaner data. CellSAM then blends it with a per-image k-means split of the box confidences (`0.66·T + 0.33·T_cluster`), which is the paper's dynamic `T_box`; so the effective cut is data-adaptive around this value. |
| normalize | `bool` | `True` | — | CellSAM's own preprocessing: 99.9-percentile clip, per-channel rescale to `[0,1]`, CLAHE with kernel 128 — the pipeline described in the paper's Methods. Leave on unless the caller already matched it. |
| postprocess | `bool` | `False` | — | Upstream morphological cleanup (open/close with `disk(2)`, dilate/erode `disk(10)`, σ=3 Gaussian, re-threshold). Upstream calls it "recommended for noisy images". See the §8 crash. |
| remove_boundaries | `bool` | `False` | — | Erode a one-pixel gap between touching cells (`subtract_boundaries`). |
| tile | `bool` | `False` | — | Segment in overlapping blocks and stitch by IoU (`cellSAM.wsi.segment_wsi`) instead of one pass. Needed for large FOVs; upstream suggests tiling above roughly 3000 cells per image. |
| tile_size | `int` | `512` | `>= 64`, practically `[256, 2048]` | Block edge, **pixels**. Smaller for dense images. |
| overlap | `int` | `56` | `[1, tile_size-1]` | Block overlap, **pixels**; must be wide enough to contain a typical cell. Passed as upstream's `iou_depth` as well — upstream requires `iou_depth <= overlap`, so the same value is both legal and maximal. |
| iou_threshold | `float` | `0.5` | `[0, 1]` | IoU above which two blocks' labels merge into one cell. Not exposed as a node socket; the upstream default. |

Tiling parameters are in **pixels, deliberately**: CellSAM resizes every tile to 1024²
internally, so tile geometry is a property of the model's input space and of memory, not a
physical extent. They are declared `unit="px"` on the node rather than hidden as constants.

---

## 5. Output

`segment_plane` → `np.ndarray`

| field | shape | dtype | axis order | units | meaning |
|-------|-------|-------|------------|-------|---------|
| labels | same as `image` `(H, W)` | `int32` | `(Y, X)` | label ids | `0` = background; each cell a unique id, **contiguous `1..K`** in ascending original-id order. A blank plane returns all zeros. |

`relabel_contiguous` → same shape, `int32`, ids renumbered `1..K` (background preserved).

The embedding and bounding boxes that `segment_cellular_image` also returns are
**discarded** — nothing downstream consumes them, and the embedding is a large array that
would otherwise be carried through the memo.

---

## 6. Conventions & GOTCHAS (the real integration risk)

- **Axis order is `(H, W)` = `(row=Y, col=X)`.** Strictly 2-D input; `segment_plane` raises
  on anything else rather than guessing which axis is Z, T or C.
- **Channel slots: a single plane becomes the WHOLE-CELL channel.** CellSAM was trained on
  3-channel input ordered `(blank, nuclear, whole-cell)`, and
  `cellSAM.utils.format_image_shape` right-aligns whatever it is handed
  (`out[:, :, -C:] = img`). A 1-channel plane therefore lands in the whole-cell slot —
  which is exactly how the paper handles nuclear-only datasets ("We moved the green channel
  to blue for nuclear-only datasets … to keep the blue channel always occupied", Methods →
  Dataset construction). This kernel passes the plane through as-is and lets upstream place
  it; it does **not** re-implement the padding.
- **Multi-channel (nuclear + membrane) fusion is NOT exposed by the node.** It is a real
  CellSAM capability and the right mode for tissue/multiplexed data, but a fused
  segmentation has no single `c` to file its Label rows under, so exposing it would need a
  Label-domain channel-key decision. Select the marker channel upstream instead.
- **`model` is a REQUIRED positional argument** of `segment_cellular_image`, even though the
  project README shows `segment_cellular_image(img, device='cuda')`. The README call raises
  `TypeError`; this kernel always passes a loaded model.
- **The no-cells path is broken upstream, in two layers — and it is the one every z-stack
  hits.** (a) `segment_cellular_image` guards with `if preds is None`, but `CellSAM.predict`
  returns the 4-tuple `(None, None, None, None)` when no box survives the confidence/IoU
  filter (`sam_inference.py`), so **the guard never fires** (upstream issue #98). Execution
  unpacks the tuple and calls `fill_holes_and_remove_small_masks(None)` →
  `AttributeError: 'NoneType' object has no attribute 'ndim'`. A blank plane, an empty FOV,
  or the dark end slices of a stack would crash the entire pull. `segment_plane` absorbs
  **exactly** that AttributeError (only when the message names `NoneType`) and returns an
  empty plane — repairing a broken guard, not inventing behaviour: "no cells" is an empty
  segmentation, which is what upstream's own dead branch was written to return.
  (b) That dead branch is itself mis-shaped: `np.zeros(img.shape[1:])` on the
  `(1, 3, H, W)` tensor gives `(3, H, W)`, not `(H, W)`. `segment_plane` collapses any 3-D
  return too, so an upstream fix for (a) lands safely. **The tiled path is immune to both**:
  `cellSAM.wsi.segment_chunk` wraps every block in `try/except Exception` and substitutes
  zeros.
- **`fill_holes_and_remove_small_masks` always runs inside upstream** with `min_size=25`
  **pixels**, and mutates its argument in place. So a hard 25-px floor is applied before any
  caller-side physical-unit filter ever sees the labels, and upstream's ids are already
  renumbered. Do not rely on the pre-filter numbering.
- **The model singleton is module-global and NOT thread-safe** — same caveat as
  `stardist_segment`. Two threads asking for different models race on the globals. Load a
  per-call model and pass it in if the caller multithreads.
- **Device is an environment knob (`NODELAB_CELLSAM_DEVICE`), not a parameter.** Because the
  model is a process singleton, a per-graph device control would silently stop taking effect
  after the first load and would make a memo key ambiguous (identical recipe hash, different
  device). Same reasoning as `NODELAB_STARDIST_CPU`. An explicit `cuda` with no CUDA device
  is a hard error, never a silent CPU fallback.
- **CPU-only torch works.** `cellSAM/modelconfig.yaml` declares `device: cuda`, but
  `AnchorDETR.build_inference` never reads `args.device` and `CellSAM.predict` takes its
  device from `next(self.parameters()).device`. The model stays wherever it was loaded.
- **Expect ~12 s per image on CPU** (the paper's own benchmark: <1 s on GPU, ~12 s on CPU,
  scaling roughly linearly with cell count because the mask decoder runs once per detection
  — unlike Cellpose's single pass). Plan the node's progress reporting accordingly.
- **Weights are licensed for non-commercial academic use** and need a DeepCell API token.

---

## 7. Dependencies

| package | why | when imported |
|---------|-----|---------------|
| `numpy` | arrays, bincount/LUT relabel | **import-time** (top level) |
| `cellSAM` | `get_model` / `get_local_model` / `segment_cellular_image` | **lazy** — inside `_require_cellsam`, gated by `importlib.util.find_spec` |
| `torch` | the ViT + SAM decoder; CUDA probe | **lazy** — inside `resolve_device` (skipped entirely for `device="cpu"`) and transitively by `cellSAM` |
| `segment_anything`, `torchvision`, `kornia`, `pyyaml`, `requests`, `tqdm`, `scikit-learn` | hard deps of `cellSAM` itself | with `cellSAM` |
| `dask`, `dask-image`, `scikit-learn` | `cellSAM.wsi.segment_wsi` | **lazy** — only when `tile=True` |

```
pip install git+https://github.com/vanvalenlab/cellSAM.git
export DEEPCELL_ACCESS_TOKEN=<token from https://users.deepcell.org>
```

`cellSAM` is **not on PyPI** — it installs from git, and `segment_anything` is itself a git
dependency. Weights download to `$HOME/.deepcell/models/cellsam_v1.2/` on first
`get_model()`. `import cellsam_segment` and `relabel_contiguous` work with numpy alone.

---

## 8. Failure modes / edge cases

- **`cellSAM` not installed** → friendly `ImportError` with the install + token hint, raised
  only when a model is loaded or a plane segmented (import still succeeds).
- **Weights missing / no token / no network** → `ImportError` wrapping the upstream error,
  with a **weights**-specific hint (deliberately not the install hint — the package is
  importable by then, and blaming the install sends you chasing a phantom). `model_path` is
  the offline escape hatch.
- **TLS interception breaks the download, not the package** (hit on this machine,
  2026-07-29). Antivirus or corporate HTTPS scanning re-signs every connection with its own
  root — here `CN=Norton Web/Mail Shield Root … generated by Norton Antivirus for SSL/TLS
  scanning`. Two distinct walls follow, and neither is a cellSAM bug:
  1. `requests` uses certifi, which does not contain that root → `CERTIFICATE_VERIFY_FAILED:
     unable to get local issuer certificate`. Adding the root to a merged bundle
     (`SSL_CERT_FILE`) is **not enough on Python 3.13**, which verifies strictly and then
     rejects it with `Basic Constraints of CA cert not marked critical` — the AV root is not
     RFC 5280-clean. (`pip` is unaffected because modern pip already uses the OS trust store
     via `truststore`.)
  2. `curl.exe`/schannel *does* trust it, but the synthetic cert publishes no CRL/OCSP, so
     revocation checking fails with `CRYPT_E_NO_REVOCATION_CHECK` → needs `--ssl-no-revoke`
     (the same class of fix as this repo's `git config http.sslBackend schannel`).

  **The reliable route is to make the download offline-shaped**, because `get_model` skips
  the network entirely once the version directory exists:
  ```powershell
  # 1. presigned URL via the API (PowerShell's .NET stack accepts the AV root)
  $j = (Invoke-WebRequest 'https://users.deepcell.org/api/getData/' -Method POST `
        -Headers @{'X-Api-Key'=$env:DEEPCELL_ACCESS_TOKEN} `
        -Body @{s3_key='models/cellsam-models_v1.2.tar.gz'} -UseBasicParsing).Content | ConvertFrom-Json
  # 2. fetch + verify: transport revocation is skipped, CONTENT is md5-verified
  curl.exe -L --ssl-no-revoke -C - -o "$HOME/.deepcell/models/cellsam-models_v1.2.tar.gz" $j.url
  # expected md5 f41e6899c49fc8ce12f77a8d25594604   (cellSAM/_auth.py::_model_versions)
  # 3. tar -xzf it into ~/.deepcell/models so cellsam_v1.2/cellsam_general.pt exists
  ```
  Or `pip install truststore` and call `truststore.inject_into_ssl()` before importing
  cellSAM, which routes verification through the OS store and fixes every Python HTTPS
  client in the process (including StarDist's model fetcher, which hits the same wall).
- **`device="cuda"` with no CUDA** → `RuntimeError` naming the torch build (deliberately not
  a silent CPU fallback).
- **Non-2-D `image`** → `ValueError`.
- **Blank / featureless plane** → valid empty result, `labels.max() == 0`. Reached by
  absorbing the upstream `AttributeError` described in §6, NOT by upstream's own
  (unreachable) empty branch. Any OTHER `AttributeError` is re-raised.
- **`postprocess=True` on a prediction with no non-zero label** → upstream
  `postprocess_predictions` ends in `np.max(new_masks, axis=0)` over a list built from
  `np.unique(...)[1:]`; that list is empty and numpy raises `ValueError: zero-size array`.
  **Not swallowed** — a crash from a documented upstream edge is more honest than a silently
  blank plane. Leave `postprocess` off unless the images are noisy.
- **`tile=True` without dask-image / scikit-learn** → `ImportError` naming `cellSAM.wsi`.
- **A returned mask whose shape does not match the plane** → `ValueError` rather than a
  guessed alignment.
- **`tile_size`/`overlap` out of range** → clamped (`tile_size >= 64`,
  `1 <= overlap <= tile_size-1`) so upstream's `iou_depth > depth` assertion cannot fire.

---

## 9. Minimal runnable example

```python
import numpy as np
from nodegraph.kernels import cellsam_segment as cs

# --- pure-numpy path (no heavy deps needed) ---
lab = np.array([[0, 7, 7, 4],
                [0, 7, 9, 9]], dtype=np.int32)
out = cs.relabel_contiguous(lab)          # ascending ORIGINAL id: 4->1, 7->2, 9->3
assert out.tolist() == [[0, 2, 2, 1], [0, 2, 3, 3]] and out.dtype == np.int32
assert cs.resolve_device("cpu") == "cpu"  # does not even import torch

# --- full CellSAM path (needs cellSAM + torch + weights) ---
if cs.cellsam_available():
    img = np.load("sample_imgs/YeaZ_pred.npy")          # any (H, W) plane
    labels = cs.segment_plane(img, bbox_threshold=0.4)   # (H, W) int32, ids 1..K
    print(labels.max(), "cells")
```

The glue is exercised end-to-end without weights by injecting a stub `cellSAM` into
`sys.modules` — see `nodegraph.selftest::test_segment` (`_stub_cellsam`), which is how the
singleton reuse, the kwarg forwarding, **both** no-cells failures (the real
`(None, None, None, None)` AttributeError and the mis-shaped `(3, H, W)` return) and the
contiguous relabel are all covered in the fast gate.

---

## 10. Pipeline wiring (the `analysis.segment` node)

Upstream, the node does all prep: the engine's lazy provider yields one prepared
`(H, W)` plane per `(m, t, z, c)` (after any crop / resample / channel select / enhancement
in the graph), and the model is loaded **once per pull**, before the loop.

Downstream, the node owns everything physical and structural — none of it belongs here:

- the shared postprocess: per-label hole filling, then the size filter in **µm² (2D) /
  µm³ (3D)** converted with `pixel_size_um` / `z_step_um`, then a contiguous relabel;
- the global id offset that makes ids unique across every `(m,t,z,c)` unit;
- the `Domain.VOXEL` label raster and the `Domain.LABEL` table
  (`id,m,t,c,area,z,y,x`, `z_kind="plane_index"`);
- the `segment_method` / `segment_model` provenance stamp.

Because the node refuses its 3D lever for this method, the kernel is only ever called on
one plane at a time. Fusing 2-D slices into true 3-D objects is a separate algorithm
(u-Segment3D, used for Fig. 3d of the paper) and would be its own node.

---

## 11. Provenance

**Not vendored from ND2Studios v1.45** — CellSAM postdates that branch and has no v1
counterpart. This file is new in-repo glue written against `cellSAM` at `master`
(`__version__ = "0.0.dev1"`, model version `1.2`), in the same "external wrapper" class as
`stardist_segment` and `dic_correlate`: the kernel holds no algorithm, only the adapter and
the model cache.

| symbol | origin |
|--------|--------|
| `get_cellsam_model`, `_cellsam_model` / `_cellsam_key` singleton | new; the shape of `stardist_segment.get_stardist_model` |
| `cellsam_available`, `_require_cellsam`, `_INSTALL_HINT` | new; the shape of `dic_correlate.al_dic_available` / `_require_al_dic` |
| `resolve_device`, `DEVICE_ENV` | new; the reasoning of the `NODELAB_STARDIST_CPU` env knob (`nodes.py`) |
| `relabel_contiguous` | new; the bincount+LUT shape of `stardist_segment.filter_and_relabel`, minus the filtering (physical units live in the node) |
| `segment_plane` | new adapter around `cellSAM.model.segment_cellular_image` / `cellSAM.wsi.segment_wsi` |

**Upstream behaviour deliberately NOT reproduced:** `cellsam_pipeline` (reloads the model
per call; §1), `enhance_low_contrast` (upstream `NameError`; §1), the returned image
embedding and bounding boxes (§5), and multi-channel `(blank, nuclear, whole-cell)` fusion
(§6 — needs a Label-domain channel-key decision first).

**DEVIATION note (2026-07-29, cellSAM `master` compat):** two upstream returns are
normalized rather than passed through — the no-cells failure is absorbed into an empty
plane (both the real `AttributeError` from upstream issue #98 and the mis-shaped
`(3, H, W)` return, §6), and every result is renumbered contiguously so the caller's
per-unit id offset stays correct. Both are documented above because a future upstream fix
would make them no-ops, not because they are optional.
