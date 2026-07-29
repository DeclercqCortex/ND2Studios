---
name: build-node-v2
description: >-
  The step-by-step procedure for BUILDING a new node or MODIFYING an existing one in the
  nodegraph v2 engine (the greenfield rebuild under `nodegraph/`). Invoke whenever you are
  about to add a v2 node type, change a node's sockets/modes/compute, declare a Granularity
  or meta_transform, or edit `nodegraph/nodes.py`. It grills the design, then writes, then
  runs a hard metadata/footprint gate and the nodegraph.selftest verify gate. Concepts live
  in wire-node-v2. This is the ONLY node-building skill (the legacy v1 `build-node` was
  deleted with v1 on 2026-07-29). Triggers:
  "build a node", "add a node", "new enhancement/analysis node", "modify this node", "change
  the node's params/ports", "add a 2D/3D lever", "make the node metadata-aware", "nodegraph node", "add a socket", "layer socket",
  "socket contract", "param has no socket".
---

# Building / modifying a nodegraph v2 node — the procedure

Do-this-in-this-order. Every v2 node is metadata-intelligent, declares its per-dim
data-access footprint honestly, and passes `nodegraph.selftest`. Concepts (Dataset,
domains, sockets, the 2D/3D lever, Granularity, meta_transform, the memo, bridges) are
in the **`wire-node-v2`** skill — this skill assumes them and tells you what to *do*.

Order: **§0 Grill → §1 Build → §2 Metadata + footprint gate → §3 (if modifying) Modify
safely → §4 Verify gate.** Do not skip §0 or §4.

---

## §0 — Grill the design first

Run the **`/grilling`** interview (one question at a time, each with your recommendation,
looking up facts in the repo yourself). Walk this decision tree in dependency order:

1. **Kind / category.** enhancement (Image→Image) · analysis (→ Voxel mask / Label /
   Point structure) · utility (axis-changing) · registration · logic · source. → fixes
   the `category` and the `op_key` prefix (`enhance.`/`analysis.`/`util.`/`align.`/…).
2. **Data contract.** What Dataset does it consume/produce? Image→image, image→structure
   (Label/Point/Track), or a Voxel layer (mask/distance field)? Does it change `AxisSizes`
   or calibration? → decides `meta_transform` (§ below) and whether it's axis-changing.
3. **2D/3D.** Is it dimension-agnostic (pointwise → no lever), 2D/3D-capable (→ `DimMode()`
   + per-dim `granularity`/`kernel_axes`), or genuinely one-mode? Is any 3D backend a
   **stack-of-2D** (→ `supports_true_3d=False`, `three_d_fallback="stack_of_2d"`, 3D
   granularity `WHOLE_PLANE`)?
4. **Sockets & params.** Enumerate every tunable: type, default, and **physical unit**
   (`um`/`um_axial`/`nm`/`s`/`px`/`""`). **For each one also settle: does the compute read
   it, and can the USER reach it?** Every param the compute reads needs a socket (the
   engine does not default-fill params, so a socket-less param is pinned forever and
   invisible in the GUI), and every socket must be read by some path. If it names an
   attribute **layer**, decide its direction and domain now — `layer_in=<Domain>` to select
   an existing layer, `layer_out=(<Domain>,…)` to name one this node creates (a tuple: one
   name can land in two domains). See `wire-node-v2` §4b/§4c — this is a hard gate below. Which have an optical basis (spot size, blur σ,
   min area) → a `derive`. Anisotropic ones get a 3D-only `_z` companion socket
   (`available_in={"dim": frozenset({"3D"})}`, `unit="um_axial"`). **If the node has a
   non-dim `Mode`, gate every param only one branch reads** — `available_in` keys on any
   mode name, and a socket the chosen method never consumes must be hidden, not accepted
   and discarded (`wire-node-v2` §5b; check the kernel per branch, and leave ungated
   anything live in all branches or whose liveness depends on a wire).
5. **Footprint.** `Granularity` per dim (TILEABLE / WHOLE_PLANE / WHOLE_VOLUME /
   WHOLE_SERIES / MULTI_VIEW) + `kernel_axes`. Must match what the compute reads.
6. **Backend.** Which scipy/skimage (or other) call? **Re-verify it exists + its
   signature in THIS env** before writing (`python -c "import inspect, skimage; ..."`) —
   versions drift; the survey in `node_backend_reference.md` is as-of 2026-07-20.
7. **Performance / numba.** Is the hot path one big compiled call (scipy/skimage/cv2/FFT/
   vectorized numpy) → **leave it, numba can't beat it**; or a Python loop of many small
   array ops / a sequential non-vectorizable algorithm → **numba `njit(cache=True)`
   candidate**. Decide *before* writing the compute; see `wire-node-v2` §12 for the
   decision table and the mandatory `cache=True` / spawn-warmup / no-scipy-in-kernel
   rules. Most nodes need **no** numba — only flag it when profiling (or an obvious O(n²)
   Python loop) says so.
8. **Identity.** The exact `op_key` (frozen). Confirm it does not collide with an existing
   node or a selftest fixture.

Write the resolved spec into the compute's docstring — it is the record.

---

## §1 — Build (write it in `nodegraph/nodes.py`)

One `register_node(compute, **spec)` per node. Templates (match the existing catalog):

### Enhancement filter with a 2D/3D lever (the common case)

```python
def _compute_median(ctx: EvalContext) -> Dataset:
    from scipy.ndimage import median_filter          # lazy import
    ds = ctx.inputs[0]
    wy = _win(_radius_px(ctx, "radius", 0.3))         # µm → px via ctx.calib(pixel_size_um)
    wz = _win(_radius_z_px(ctx, "radius", 0.3)) if ctx.is_volume else wy
    return _map_image(                                # 2D per-plane / 3D per-volume router
        ctx, ds,
        plane_fn=lambda a: median_filter(a, size=(wy, wy)),
        volume_fn=lambda v: median_filter(v, size=(wz, wy, wy)))

register_node(
    _compute_median, op_key="enhance.median", label="Median", category="enhancement",
    inputs=[InDataset(), *_InRadius()], outputs=[OutDataset()], modes=[DimMode()],
    granularity=_DIM_GRAN, kernel_axes=_DIM_KAX,
    description="Edge-preserving median; 2D per-plane vs anisotropic 3D window.")
```

- `_map_image(ctx, ds, plane_fn, volume_fn)` routes on `ctx.is_volume`; a WHOLE_VOLUME
  node MUST pass a `volume_fn`. `_InRadius(name, label, default)` = a lateral µm socket +
  its 3D-only `um_axial` companion. `_DIM_GRAN`/`_DIM_KAX` are the shared per-dim maps.
- Read calibration **only** via `ctx.calib(...)`; convert with `to_pixels_v2(...)`.
  Read a param via `ctx.params.get(name, fallback)`; mode state via
  `ctx.params.get("__modes__", {}).get("op", default)`.
- A realized array wraps into `ds.with_image(ArrayProvider(out6d))` (shape
  `(M,T,Z,C,Y,X)`).

### Analysis → structure (Voxel mask / Label / Point)

```python
def _compute_threshold(ctx):        # → a Voxel mask layer
    ...
    return ds.with_layer(Domain.VOXEL, ctx.params.get("name", "mask"), mask6)

def _compute_label(ctx):            # → Label raster + StructureTable (global-unique ids)
    ...  # label_components per plane/volume, offset ids, merge columns
    return ds.with_layer(Domain.VOXEL, layer, raster).with_structure(merged_table)
```
Structure tables carry the invariant `id,m,t,c,z,y,x` schema; attach with
`Dataset.with_structure`. Detection → `point_table(...)`; watershed →
`seeded_watershed(...)` + a region-props table.

### Axis-changing node (declare a meta_transform, sync payload to it)

```python
def _compute_zproject(ctx):
    ...  # reduce Z; out shape (M,T,1,C,Y,X)
    projected = ds.with_image(ArrayProvider(out)).reshaped_axes(replace(ax, z=1))
    return projected.with_metadata(z_step_um=None, z_collapsed=True)   # lockstep

register_node(_compute_zproject, op_key="util.zproject", ..., category="utility",
    granularity=Granularity.WHOLE_VOLUME, meta_transform=_meta_z_project)  # from metadata.py
```
For a **relative** calibration change (Resample scales pixel size), do NOT re-derive in
the compute — `ctx.calib` returns the **post-transform** env value; SYNC the payload to
it, or you double-count. Match the meta_transform's size/bounds math exactly.

Add force-import coverage automatically: `register_node` runs at import of
`nodegraph.nodes`, so a new node appears once the module is imported (no `__init__`
edit needed). Backends are **lazily imported inside the compute** so the core stays
scipy/skimage-free.

---

## §2 — Metadata + footprint gate (HARD — blocks "done")

- [ ] **Every spatial/temporal param has a `unit`** (`um`/`um_axial`/`nm`/`s`/`px`).
      No bare-pixel spatial constant. A 3D anisotropic extent has an axial `_z` socket
      (`um_axial`, `available_in={"dim":{"3D"}}`).
- [ ] **Optically derivable params have a `derive`** over the envelope symbols, each
      `or`-guarded. Dimensionless params (a 0–1 ratio, a mode) carry `unit=""`.
- [ ] **The compute converts via `to_pixels_v2` and reads via `ctx.calib`** — never a
      raw pixel constant, never `ctx.inputs[i].metadata[calib_key]`.
- [ ] **`granularity`/`kernel_axes` match the compute.** A `WHOLE_VOLUME` node reads a
      volume and supplies `volume_fn`; a stack-of-2D node declares 3D `WHOLE_PLANE` +
      `supports_true_3d=False`. 2D and 3D must produce **distinct recipe hashes** (the
      lever folds into params).
- [ ] **Axis-changing node:** `meta_transform` declared; the compute's payload axes AND
      calibration equal the meta_transform's prediction for every param combination
      (one-sided / out-of-range included); no double-counted relative change.
- [ ] **Value-scale change restamped** (`wire-node-v2` §7c). Ask: *does this node change
      what its numbers mean?* A reducer that SUMS widens `bit_depth`
      (`metadata.bit_depth_after_sum`); an output that is no longer integer counts drops it
      (`meta_transform=value_rescaled`); redistributing inside the same range restamps
      nothing — verify the BACKEND's real output range rather than assuming (skimage's
      `equalize_adapthist` returns `[0,1]`). Both halves in lockstep: the `meta_transform`
      for edit-time, the payload `with_metadata` in the compute, synced from
      `ctx.calib("bit_depth")` — never re-derived.
- [ ] **Raw-count consumer reads the depth.** A param in the image's own intensity units
      (a fixed threshold, a LUT range) resolves from `ctx.calib("bit_depth")` — via a
      `derive` for its default — and behaves sanely when the key is ABSENT (post-Normalize
      data). Never hardcode 16.

- [ ] **THE SOCKET CONTRACT** (`wire-node-v2` §4b — all five clauses are enforced by
      `selftest::test_param_socket_contract`, which fails the build):
      **(1)** every param the compute reads has a socket — no exceptions beyond a
      documented `_PARAM_NO_SOCKET_OK` alias or a *proven* machine-set param;
      **(2)** every socket is read on some path — otherwise `available_in`-gate it, refuse
      it explicitly, or delete it (a live-looking control the kernel ignores is
      charter-forbidden);
      **(3)** `available_in` names real modes AND real values;
      **(4)** a layer-name socket declares `layer_in`/`layer_out` with a domain that is
      consistent with `reads_domains`/`adds_domains` (the `reads` half is exempt when the
      socket is mode-gated);
      **(5)** the default is declared **once**, in the `SocketSpec` — read layer params
      with **`ctx.layer("name")`**, never `ctx.params.get("name", "name")`, so the compute
      and `propagate_meta`'s edit-time prediction cannot drift.
- [ ] **A layer this node CREATES is announced.** If a `layer_out` socket cannot describe
      it — a literal name with no socket, a name derived from another param, or a write
      into the layer a *read* socket names — add `NodeSpec.extra_layers(params, modes)`.
      It runs inside `propagate_meta` on every keystroke, so it must be **total** (never
      raise). Without it the GUI's layer picker will not offer your node's output
      downstream.

If a spatial param would ship in raw pixels: STOP — give it `unit`/`derive` or justify
scale-invariance in the docstring.

---

## §3 — Modifying an existing node (memo + saved-file safety)

- **Never rename/repurpose an `op_key`** (saved-file contract). New keys are free.
- **Adding a socket/mode** is safe; the lever/mode state and new params fold into the
  recipe hash → old memo entries simply miss and recompute.
- **Changing `granularity`/`kernel_axes`/`meta_transform`** changes routing/prediction —
  re-verify the payload↔meta_transform agreement and the per-dim hashes.
- **Changing a `derive`/`unit`** doesn't touch saved graphs but changes runtime
  interpretation — audit the compute's `to_pixels_v2` call and the GUI re-seed.
- **Adding a socket for a param the compute already read** is the common repair, and it
  is safe: the value was previously pinned to the compute's inline fallback, so declaring
  the socket with that SAME value as its `default` changes nothing for existing graphs.
  Then delete the inline fallback in favour of `ctx.layer(...)` (contract clause 5).
- **Changing a param into a Mode** (closed enumeration → header picker) is NOT
  transparent: the engine hands the compute the *resolved* mode state, so "unset" and
  "explicitly the default" are indistinguishable and a params fallback cannot be honoured.
  **Refuse the legacy param with a message** rather than silently running on the defaults —
  `transform.transfer_domain` is the worked example.
- A test/demo fixture must **never** `define_node` a real `op_key` — it clobbers the
  node in the global `NODES` registry and causes order-dependent selftest failures.

---

## §4 — Verify gate (run ALL; fix red before "done")

```bash
# 1. the headless core + catalog selftest — the v2 node gate (add coverage for your node)
PYTHONUTF8=1 python -m nodegraph.selftest            # must end "ALL NODEGRAPH SELF-TESTS PASSED"

# 2. the driven GUI probe — sockets/inspector/pull still behave (skip only for a pure
#    docstring edit). Was preceded here by a v1 `pipeline_kit` parity gate; v1 was
#    removed 2026-07-29 (V2.05 §6), so that gate no longer exists.
PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py out.png   # "ALL PHASE-5 GUI PROBES PASSED"

# 3. force-import: your node registers cleanly
PYTHONUTF8=1 python -c "import nodegraph.nodes; from nodegraph.registry import NODES; \
  assert 'enhance.median' in NODES, 'not registered'; print('registered OK')"
```

**Add a selftest case** (in `nodegraph/selftest.py`, registered in `main()`) that pulls
your node end-to-end through the `Engine` and asserts:
- output finite + correct geometry, in both 2D and 3D (if it has a lever);
- 2D vs 3D (and each mode) produce **distinct** `entry(node).recipe_hash`;
- an axis-changing node: `engine.env(node).axes` (from the meta_transform) **equals** the
  pulled payload's `axes` (and calibration) across ordinary + edge-case params;
- structure output lands on the expected domain/layer with the invariant schema;
- a metadata-dependent node re-reads the right calibration key (assert it appears in
  `dict(engine.entry(node).reads)` so the memo fences on it).

Build the test image with a deterministic pattern (avoid RNG); guard the group with
`if not _HAVE_SKIMAGE: _ok(...); return` when it needs scipy/skimage.

### Done checklist
- [ ] §0 grilled; resolved spec in the compute docstring; backend signature re-verified in-env.
- [ ] §2 gate passes (units/derive; footprint matches compute; meta_transform agrees).
- [ ] `op_key` frozen + collision-free (not squatted by a fixture).
- [ ] **Socket contract green** — every param reachable, every socket read,
      layer sockets declared + domain-consistent, defaults declared once
      (`ctx.layer`), and any un-socketable output layer registered via
      `extra_layers`.
- [ ] **If a numba kernel was added:** it's a module-level `@nb.njit(cache=True)` pure
      numeric helper (no `ctx`/`Dataset`/scipy inside), and the numba↔numpy equivalence
      was checked on a fixture (see `scripts/_bench_nms_numba.py` for the pattern).
- [ ] §4 green: `nodegraph.selftest` (with new coverage), v1 parity clean, force-import.

---

**See also — `wire-node-v2`:** §1 the Blender↔v2 model · **§4b the socket contract +
§4c layer-name sockets** · §5 the 2D/3D lever + variant sockets · §6 Granularity · §7 metadata intelligence (`unit`/`derive`, `ctx.calib`,
`to_pixels_v2`) · §8 `meta_transform` · §9 structure/transfer/bridges · §10 the memo
invariant.
