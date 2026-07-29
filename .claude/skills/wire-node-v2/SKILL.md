---
name: wire-node-v2
description: >-
  How to define, wire, and metadata-adapt a pipeline node in the nodegraph v2 engine
  (the greenfield, Blender-geometry-nodes rebuild under `nodegraph/`). Read this BEFORE
  adding any v2 node type, changing a node's sockets/modes, touching the 2D/3D lever,
  declaring a data-access footprint (Granularity/kernel_axes), adding a meta_transform,
  or wiring domain transfer / structure bridges. This is the ONLY node-concepts reference
  (the legacy v1 `wire-node` was deleted with v1 on 2026-07-29). Triggers: "add a node", "new pipeline node",
  "nodegraph node", "wire a node", "node sockets", "2D/3D lever", "DimMode", "Granularity",
  "meta_transform", "domain transfer", "structure bridge", "metadata-intelligent param",
  "socket contract", "layer_in", "layer_out", "layer picker", "ctx.layer",
  "unreachable param", "dead socket".
---

# Wiring a node in nodegraph v2 (NodeLab v2)

The v2 engine is a **Blender geometry-nodes engine adapted to microscope image
analysis**, rebuilt greenfield under [`nodegraph/`](../../../nodegraph/) (Qt-free;
GUI lives only in `nodelab_v2/`). Nodes are pure, typed functions of a lazy
**Dataset**; a two-hash memo + lazy pull scheduler decides what actually computes.

> **This is the concepts/contracts reference.** To actually build or modify a v2
> node, use the **`build-node-v2`** skill — it runs the procedure (grill → write →
> footprint/metadata gate → verify gate) and links back here.

---

## 1. Mental model — Blender → microscopy (v2 mapping)

| Blender geometry nodes | nodegraph v2 | Where |
|---|---|---|
| Geometry flowing through the tree | one lazy **`Dataset`** (image provider + attribute layers) | [`dataset.py`](../../../nodegraph/dataset.py) |
| Typed sockets; implicit conversions | `SocketType` {DATASET, FLOAT, INT, BOOL, VECTOR, COLOR, STRING} + `can_connect` | [`sockets.py`](../../../nodegraph/sockets.py) |
| Named-attribute layer over geometry | **AttributeLayer**s keyed `(domain, layer, name)` over domains | `dataset.py` |
| Attribute **domains** (point/edge/face) | **domains**: lattice (Global/Frame/Timepoint/Channel/Voxel/…) + structure (Label/Point/Track) | [`domains.py`](../../../nodegraph/domains.py) |
| Field / value inputs adapting to the mesh | **value sockets** carrying `unit`/`derive` (metadata-intelligence) + `is_field` | [`registry.py`](../../../nodegraph/registry.py) |
| A modifier's evaluation dimension | the **2D/3D lever** (`DimMode`, a header ModeSpec) | `registry.py` |
| Viewer node / lazy eval | **lazy pull** — nothing computes until `Engine.pull(node)` | [`engine.py`](../../../nodegraph/engine.py) |
| Node = pure function of inputs + params | Same. No Qt, no global state in a compute body | `nodegraph/` is Qt-free |

**The one rule:** a node declares typed sockets + unit-tagged params + its
data-access footprint; the engine does the routing, the px↔µm math, the memoization,
and the domain transfers. A node never hardcodes a pixel constant and never invents
graph state.

---

## 2. The seam — where a v2 node lives

A node type = a **`NodeSpec`** (registered in the global `NODES` registry) + a
**`compute(ctx) -> Dataset`** function (recorded in `COMPUTES`, keyed by `op_key`).
Both are created by one call in [`nodes.py`](../../../nodegraph/nodes.py):

```python
register_node(compute_fn, op_key="enhance.median", label="Median",
              category="enhancement", inputs=[...], outputs=[OutDataset()],
              modes=[DimMode()], granularity=..., kernel_axes=..., meta_transform=...)
```

`register_node` (nodes.py) = `define_node(**spec)` (registry.py, builds + registers the
`NodeSpec`) + `COMPUTES[op_key] = compute_fn`. The `Engine` looks up the compute by
`op_key` and runs it inside the pull/memo/ReadContext harness. Importing `nodegraph.nodes`
registers every node (its module-level `register_node` calls fire).

---

## 3. Anatomy of a v2 node (the pieces)

1. **`op_key`** — the frozen string identity (`enhance.gaussian`, `analysis.label`,
   `util.crop`). A saved-file contract; never rename. **Test/demo fixtures MUST use a
   fake op_key** (`test.*`) or they clobber a real node in the global registry.
2. **`NodeSpec`** ([registry.py](../../../nodegraph/registry.py)) — `op_key`, `label`,
   `category`, `inputs`/`outputs` (`SocketSpec`s), `modes` (`ModeSpec`s), `description`,
   and the v2 declarations: `granularity`, `kernel_axes`, `meta_transform`,
   `supports_2d`/`supports_true_3d`/`three_d_fallback`.
3. **Sockets** (`SocketSpec`) — `InDataset()`/`OutDataset()` for the main wire;
   `InFloat/InInt/InBool/InVector/InColor/InString` for value/field inputs. Value
   sockets carry `unit`, `derive`, `is_field`, `default`, `dims` (VECTOR arity),
   `available_in` (mode-gated presence).
4. **Modes** (`ModeSpec`, via `Mode(...)` / `DimMode()`) — in-body dropdowns; may
   reconfigure sockets. Fold into the recipe hash (as `params["__modes__"]`).
5. **`compute(ctx)`** — reads `ctx.inputs`, `ctx.calib`/`ctx.meta`, `ctx.params`
   (incl. `__modes__`), `ctx.granularity`/`ctx.is_volume`, and returns a new `Dataset`.

---

## 4. Typed sockets & wiring (`sockets.can_connect`)

- **DATASET↔DATASET** is the main wire. `InDataset(multi=True)` accepts multiple.
- **Value/field sockets** convert implicitly per the conversion table
  (`can_convert`): e.g. FLOAT→INT, scalar→VECTOR (broadcast).
- **VECTOR** arity is **widen-only** (2→3 pads the axial 0; 3→2 rejected).
- A socket may be **field-driven** (`is_field=True`) — a value that varies over a
  domain, evaluated by the field IR ([`field.py`](../../../nodegraph/field.py)).
- Cycles are rejected outside zones (`graph.topo_order`); the engine walks inputs in
  **canonical socket order** so a multi-input node's args land in declared slots
  regardless of edge-connect order.

Choose output by what downstream consumes: a transform → `OutDataset()`; the Dataset
carries new attribute layers / structure tables added by the compute.


### 4b. THE SOCKET CONTRACT — one law for every socket (V2.10/V2.11)

**Read this before adding or editing any socket.** These rules are not style; each one is
enforced by `nodegraph.selftest::test_param_socket_contract`, which fails the build. They
exist because socket declarations and compute code drifted apart repeatedly — a 2026-07-28
sweep found **18 of 55 nodes** with a defect of this class.

**The root hazard: the engine does NOT filter params against the socket list.** It hands
the compute `{**node.params, "__modes__": …}` — params are *overrides*, never
default-filled. So a compute can happily read a param no socket declares: it works
headlessly (a test just passes it) and is **unreachable in the GUI**, which builds its
widgets from `NodeSpec.inputs`. No functional test can see it — the node returns the right
answer for the only value it can ever have. This is a *structural* defect, so the contract
is checked structurally.

**The five clauses:**

1. **Every param the compute reads has a socket.** If you write `ctx.params.get("x")`,
   declare an `x` socket. The only exemptions are documented in `_PARAM_NO_SOCKET_OK`
   (currently one back-compat alias) and *machine-set* params written by the graph builder
   rather than a user (`channel.select`'s `channels`, written by the per-channel tap
   materializer). Prove machine-set by finding the writer; do not assume.
2. **Every socket the node declares is read.** A control that does nothing is
   charter-forbidden (the `track.objects` review). The remedies, in order:
   **(a) gate it** with `available_in` so it only appears in modes that consume it;
   **(b) refuse** — raise if it is explicitly set on a path that ignores it; **(c) delete**.
   Prefer (a). Use (b) when the condition is not a rectangle of mode values —
   `transform.transfer_domain` refuses a non-default `reducer` on a pure broadcast because
   "src drops an axis vs dst" cannot be expressed in `available_in`.
3. **`available_in` must name real modes and real values.** A typo silently hides the
   socket in every state.
4. **A layer-name socket declares its direction and domain** (§4c).
5. **A default is declared exactly ONCE — in the `SocketSpec`.**

**Clause 5 is the subtle one.** Because params are raw overrides, computes used to repeat
their socket's default inline: `ctx.params.get("mask", "mask")`. That is a *second* copy —
and `propagate_meta` needs the same name at edit time to predict the layer catalog, making
a *third*. Nothing kept them in step. Read layer params through **`ctx.layer("mask")`**,
which resolves override → `SocketSpec.default` via the one shared
`registry.layer_value`, the same function `propagate_meta` calls. Never re-spell the
default in the compute.

### 4c. Layer-name sockets — `layer_in` / `layer_out` (V2.11)

A STRING socket naming an attribute **layer** is not free text; declare which way it points:

```python
InString("mask",  "Mask layer",   field=False, default="mask",
         layer_in=Domain.VOXEL)              # SELECTS an existing layer → GUI picker
InString("name",  "Output layer", field=False, default="labels",
         layer_out=(Domain.VOXEL, Domain.LABEL))   # NAMES a layer this node CREATES
```

* `layer_out` is a **tuple** because one name can land in two domains — `analysis.label`
  emits both a Voxel raster *and* a Label table called `labels`.
* `layer_in_mode="from_domain"` instead of `layer_in` when the domain is a **mode value**
  (only `transform.transfer_domain`).
* **Domain consistency is enforced:** `layer_in` domain ⊆ `reads_domains`, and `layer_out`
  domains ⊆ `adds_domains`. The `reads_domains` half is **exempt when the socket is
  mode-gated** — a conditional requirement, and `reads_domains` has no per-mode form (why
  `tessellate` / `track.link` legitimately declare it empty).
* Producers no socket can describe use **`NodeSpec.extra_layers(params, modes)`**: literal
  names with no socket (`align.drift`'s `drift_y`/`drift_x`), names derived from another
  param (`extract_boundary` → `f"{labels}_boundary"`), or writes into the layer a *read*
  socket names (`measure`). It runs on every keystroke — **it must never raise**.

**What the declarations buy.** `metadata.propagate_meta` builds `MetaEnvelope.layer_names`
— the `(domain, name)` catalog for each edge — and the inspector turns a `layer_in` socket
into an **editable combo** offering exactly those names (`document.layer_choices`, which
follows the *primary* Dataset edge only, so a `reference`/`raw` second input never leaks in).

**Two traps if you touch the catalog itself:**

* **Do not key anything on `LayerKey`.** It is `(domain, layer, name)`, but the user-facing
  name is in a *different slot per family*: `with_layer` leaves `layer=None`, so a lattice
  layer is `(VOXEL, None, "mask")`; `with_structure` files each COLUMN as
  `(POINT, "spots", "y")`. Keying on slot 1 collapses every Voxel layer to `(VOXEL, None)`.
  `layer_names` stores the user-facing name per domain to sidestep this.
* **The catalog is NOT monotone.** `reshaped_axes(drop_stale=True)` discards lattice layers
  whose shape no longer matches, so an axis-changing node (crop/resample/zproject/stack/
  channel.select) *removes* layers. The drop is derived centrally from the axis delta — you
  do not declare it — but do not assume "layers only accumulate".

---

## 5. The 2D/3D lever + variant sockets (V2.03 directive B)

Every 2D/3D-capable node gets a **header lever** — `DimMode()`, a distinguished
`ModeSpec` with `role="dim_lever"`, `presentation="header"`, and a metadata-adaptive
default (`derive="'3D' if (n_z or 1) > 1 else '2D'"` — z>1 ⇒ 3D). It is a **Mode, not a
socket**, so it folds into the recipe hash as an in-body option → 2D and 3D memoize
distinctly, and serialization is unchanged.

- **Variant sockets:** `SocketSpec.available_in={"dim": frozenset({"3D"})}` marks a
  socket present only in that mode state (e.g. an anisotropic axial `sigma_z`,
  `unit="um_axial"`, 3D-only). `NodeSpec.active_inputs(state)` / `active_sockets(state)`
  resolve the live set. The paired-float pattern: a lateral param + a 3D-only `_z`
  companion that defaults to the lateral value.
- **`supports_true_3d` / `three_d_fallback`:** declare honesty — a genuinely
  volumetric backend sets `supports_true_3d=True`; a **stack-of-2D** backend (e.g.
  `denoise_wavelet`, `denoise_bilateral`) sets `supports_true_3d=False`,
  `three_d_fallback="stack_of_2d"`, and keeps its 3D `granularity` at `WHOLE_PLANE` so
  it stays on the per-plane path (H15). Never silently loop 2D over Z while claiming 3D.
- **z==1 guard (GUI):** the lever greys 3D when incoming z==1; a serialized 3D-on-z==1
  is a validation error (H11).
- **Not every node has a lever.** Pointwise/dimension-agnostic ops (gamma, threshold)
  omit it (`kernel_axes=frozenset()`). A **scope** choice (Normalize: plane/volume/series)
  is a plain `Mode`, **not** the dim lever (H24).

### 5b. `available_in` gates ANY mode, not just `dim` (2026-07-28)

**A socket the selected mode never reads must be HIDDEN, not shown and discarded.** The
key of `available_in` is any mode name — `{"method": frozenset({"fixed"})}`,
`{"boundary": frozenset({"alpha_shape"})}` — and a socket may gate on several modes at
once (`analysis.histogram_threshold` gates each threshold on **method × direction**, so
1–2 live fields show instead of all 8). When adding a node with a non-dim `Mode`, read
the kernel and gate every param that only one branch consumes; a param that is live in
*every* branch stays ungated (`detect.particles`: `min_size` filters blobs in both LoG
and components — leave it alone; `analysis.boundary_band`: `band_voxels` is dilation
iterations *and* the `edt` fallback threshold, so only `band_um` is gated).

Two things this does **not** change: the engine still passes every declared param to the
compute (gating is edit-time only, so a compute's `ctx.params.get(name, default)`
fallback must stay correct), and the mode already folds into the recipe hash, so hiding
a socket never alters a memo key.

`available_in` can only see **mode state**, not whether an optional socket is wired — so
a param whose liveness depends on a wire (DVC/DIC `reference_frame`: dead under
`previous_frame`, but live again when an external `reference` Dataset is connected)
stays ungated on purpose. Don't gate those; hiding a control that still has an effect is
the same bug in reverse.

Both halves must be verified: `NodeSpec.active_inputs(state)` in `nodegraph.selftest`,
**and** the GUI — the node card re-lays-out via `scene.sync → item.refresh`, and the
inspector rebuilds via `_set_mode`'s deferred `_rebuild` (covered by the "mode gate"
check in `scripts/_nodelab_v2_phase5_probe.py`).

---

## 6. Data-access footprint — `Granularity` + `kernel_axes`

Declares what the node must consume *whole* vs per-element; gates tiling/halo/memo in
the scheduler and selects the provider read path (engine granularity routing).

| `Granularity` | Meaning | Provider read |
|---|---|---|
| `TILEABLE` | pointwise / small-stencil 2D | tile / region at one z |
| `WHOLE_PLANE` | a full (Y,X) plane per (m,t,z,c) | region at one z |
| `WHOLE_VOLUME` | a full (Z,Y,X) volume per (m,t,c) | `get_region_volume` |
| `WHOLE_SERIES` | the full T series per (m,c) | across-T reads |
| `MULTI_VIEW` | several M (stitching / fusion) | across-M reads |

- For a lever node, pass a **per-dim map**: `granularity={"2D": TILEABLE, "3D": WHOLE_VOLUME}`,
  `kernel_axes={"2D": frozenset({"y","x"}), "3D": frozenset({"z","y","x"})}` — resolved
  by mode state via `spec.resolve_granularity(state)` / `resolve_kernel_axes(state)`.
- In the compute, `ctx.is_volume` (== resolved `WHOLE_VOLUME`) routes the 2D vs 3D path.
  The shared helper `_map_image(ctx, ds, plane_fn, volume_fn)` (nodes.py) applies a
  per-plane fn in 2D and a per-volume fn in 3D and re-wraps the result as an
  `ArrayProvider`.
- **Declare the footprint honestly** — it must match what the compute actually reads.
  A `WHOLE_VOLUME` node must supply a `volume_fn`; a stack-of-2D node keeps 3D at
  `WHOLE_PLANE`.

---

## 7. Metadata intelligence — `unit` / `derive` on value sockets

Every spatial/temporal number is authored in **physical units**; the framework
resolves it to px/frames from the *incoming edge's* metadata (per-edge, V2.03 §A).

- `unit` ∈ `{"px","um","um_axial","um2","um3","nm","s",""}`. `um_axial` divides by
  `z_step_um` (anisotropic axial sampling); `um`/`nm` divide by `pixel_size_um`.
- `derive` is a pure-arithmetic expression over the **envelope symbol table**
  (`metadata.envelope_symbols`): `pixel_size_um, z_step_um, dt_s, emission_nm, na, mag,
  n_m/n_t/n_z/n_c, z_collapsed, is_3d`. Guard every symbol with `or <fallback>`
  (e.g. `"0.61*(emission_nm or 520)/(na or 1.4)/1000"` → diffraction-limited µm).
- **In the compute**, convert with `to_pixels_v2(value, unit, pixel_size_um=..., z_step_um=...)`
  and read calibration through **`ctx.calib(key)`** (validated against `CALIBRATION_KEYS`;
  a typo is a hard error) — the read is *recorded* and the memo re-validates it on a hit,
  so a metadata change the node actually read invalidates exactly that node.
- The compute reads params with an **inline fallback default**
  (`ctx.params.get("sigma", 0.5)`) — the socket `derive`/`default` seed the GUI widget;
  headless the compute's fallback applies.
- **Per-channel derive — `ctx.channel(c)` (C8/H12).** A **c-iterating** node (its `for c`
  loop processes each channel: Deconvolve's PSF, Spot Detection's radii) must resolve any
  `emission_nm`-derived param **per channel** — channel c's λ, not a collapsed channel-0
  value. Use **`ctx.channel(c).param(name)`**: a user override (param set) wins as one value
  for all channels; else the socket's `derive` is re-evaluated with channel c's optics
  (`envelope_symbols(ctx.env, c)`), memo-fenced automatically; else the socket default.
  `ctx.channel(c).emission_nm()` gives just that channel's λ. Resolve **eagerly** in the loop
  (before returning a lazy provider — a late read trips the frozen-ReadContext guard). This
  also makes `derive` work headless (not only as a GUI seed) for these nodes.
- **Never read `ctx.inputs[i].metadata[calib_key]` directly** — that bypasses the memo
  fence. In debug (`Engine(strict_reads=True)`) it is a hard error (C7).

### 7b. Provenance inheritance — a node stamps context a downstream node reads

When a downstream node's behaviour is **determined by how an upstream node ran** — not
by a value the user should re-type — the upstream node should **stamp that config** and
the downstream node should **inherit it**, instead of exposing a redundant/conflictable
control. This is metadata intelligence at the *structural* (non-calibration) level.

**The built-in generic case — structure dimensionality (`z_kind`).** A structure table's
`z_kind` (`"plane_index"` = 2D per-plane, `"subpixel"` = 3D) is lost when
`with_structure` explodes the table into per-column `AttributeLayer`s. So `with_structure`
**automatically preserves it** into a namespaced `__struct_zkind__` metadata map (keyed by
domain+layer) — **every structure-producing node self-describes its dimensionality for
free, no per-node code**. A consumer inherits it with **`ds.structure_zkind(domain, layer)`**
(→ `"plane_index"`/`"subpixel"`/`None`). This is why `transform.rasterize_field` has **no
`DimMode` lever**: it routes 2D-per-plane vs 3D-volumetric off the *field's own* `z_kind`,
so a 2D per-plane field on a `z>1` image is never misread as a 3D grid (its plane-index
`z` treated as a coordinate). `analysis.accumulate_field` inherits its dim the same way.

**The general pattern (for config beyond dim):**
- **Stamp on the output** with `ds.with_metadata(key=value, ...)` — the sanctioned write
  path. Use a **namespaced** non-calibration key (e.g. `dvc_reference_mode`,
  `dvc_strain_type`) so it never collides with `CALIBRATION_KEYS`. Example: `analysis.dvc_field`
  stamps `dvc_reference_mode` / `dvc_strain_*` alongside its Point output (dim is NOT
  stamped here — it rides the generic `z_kind`).
- **Inherit in the consumer** by reading `ctx.inputs[0].metadata.get("key")` **directly**
  (or `ds.structure_zkind(...)` for dim). Allowed *because the key is not a calibration
  key* — `_StrictCalibMetadata` passes non-calibration reads through (only `CALIBRATION_KEYS`
  trip C7). Memo-safe without recording: the marker rides the upstream payload, so a change
  upstream bumps the upstream **revision**, which already folds into the consumer's
  `recipe_hash` (and `output_fingerprint` folds `__struct_zkind__` via `metadata`).
- **Derive, don't ask.** A consumer that inherits its dim/reference/measure from the
  provenance needs **no `DimMode` lever and no reference param** — a lever could disagree
  with the data and silently corrupt it. Fix a plain `granularity`/`kernel_axes` (a superset
  like `frozenset({"z","y","x"})`) instead; the compute loops the units internally.
- **Refuse the wrong input.** Inherited provenance also lets the consumer **validate**:
  `analysis.accumulate_field` raises if the marker is absent (not a DVC field) or says
  `fixed_frame` (already cumulative — nothing to accumulate), and re-stamps its own output
  `dvc_reference_mode="cumulative"` so a second accumulate is refused too.
- Contrast with **calibration** provenance: pixel/z/dt still flow through the envelope and
  are read via `ctx.calib` (§7) — only *structural config that isn't a physical unit* uses
  this stamp-and-inherit pattern.

### 7c. Calibration describes the CURRENT data, not the file (2026-07-28)

**Every node reads its INPUT envelope, so calibration is always the most up-to-date value
— which means a node that changes what its numbers MEAN must restamp the affected key in
lockstep.** This is the same law as §8 (an axis-changing node restamps pixel size); it
just also applies to keys that are not geometry. Read it as: *the envelope is a running
description of the data on this wire, never a record of the acquisition.*

**`bit_depth` is the intensity-domain instance.** It is a calibration key (ND2
`bitsPerComponentSignificant` — most files are **12**-bit, not 16), so:

- A node that **widens the value scale** restamps it. Summing `n` samples of a `b`-bit
  signal needs `b + ceil(log2 n)` bits → `metadata.bit_depth_after_sum(env, n)`, used by
  `z_project`/`stack_time` for their `sum` combiners only (mean/median/max/min stay inside
  the input range). A 12-bit series summed over T=8 becomes 15-bit *for every node after
  it*, and chains (`sum` → `sum` → 17-bit).
- A node whose output **is no longer integer counts** DROPS it: `metadata.value_rescaled`
  is the ready-made meta_transform (percentile `enhance.normalize` → `[0,1]` floats).
  Absent `bit_depth` is the honest signal "no declared integer scale", and consumers must
  handle it: `analysis.threshold`'s fixed level derives mid-range from the depth and falls
  back to `0.5` without one; `analysis.histogram_threshold` refuses `[0,1]` input outright.
- A node that only **redistributes** intensities inside the same range does NOT restamp —
  `enhance.clahe` rescales its equalized result back to the input's `[min,max]`, and γ
  (`(a/mx)**g * mx`) preserves the plane max. Check the backend's actual output range
  before deciding (skimage's `equalize_adapthist` returns `[0,1]`; the node's own
  post-scaling is what saves it).

**Both halves, as always.** Declare the `meta_transform` so *edit-time* propagation
predicts the new value (the GUI's widget re-seed and every `derive` read it before any
pull), and stamp the payload in the compute so the pulled Dataset agrees. And per §8, do
NOT re-derive a relative change in the compute — `ctx.calib` already returns the
**post-transform** value, so sync the payload to it (`bd_out = ctx.calib("bit_depth")`)
or a re-pull would widen twice.

**Consumers read it through `ctx.calib("bit_depth")`**, which memo-fences the read — so
re-ingesting the same file at another depth, or flipping an upstream combiner to `sum`,
invalidates exactly the nodes that cared.

### 7d. The optional `raw` socket — measure on unenhanced pixels (2026-07-28)

Enhancement is for *finding* objects; the numbers you report should come from the pixels
the camera recorded. A node that MEASURES intensities therefore takes an optional second
Dataset input, `_InRaw()` (`nodes.py`), resolved by **`_intensity_provider(ctx, ds)` →
`(provider, is_raw)`**. Wired: those pixels are measured. Unwired: its own image, exactly
as before. On `analysis.measure` (all stats) and `analysis.histogram_threshold` (the region
table's intensity columns, re-measured via the vendored `_measure_regions`).

Four rules make it safe, and they generalize to any "second Dataset input" node:

- **Override MEASUREMENT only, never segmentation.** The mask, the label raster and every
  threshold stay on the main input — the chain the user tuned. So geometry and calibration
  remain one consistent story and only the *reported* numbers change. A node that wants to
  *threshold* raw pixels needs no lever: wire the raw Dataset into its main input.
- **Declare it AFTER `data`.** `graph.dataset_preds` sorts by declared socket position, so
  `data` stays `dpreds[0]` = the calibration/domain env source no matter which edge the
  user wired first. Never let an auxiliary input become the env.
- **Refuse a geometry mismatch.** The two are read voxel-for-voxel, so a cropped /
  resampled / z-projected / channel-tapped raw would report neighbouring objects'
  intensities — silent corruption. Compare `provider.axes` and raise, naming both shapes
  and the fix. (`AxisSizes` is a frozen dataclass: `==` works, but it is NOT iterable —
  format it field by field.)
- **The memo needs nothing.** A new predecessor folds into `recipe_hash` automatically, so
  wiring or unwiring `raw` re-keys the node on its own.

Why not a Mode toggle: a lever cannot say *which* raw Dataset, and a hidden "read the
source" path would break the "pure function of its inputs" law the memo depends on. An
explicit socket is visible in the graph, diffable, and serializes for free. Same shape as
`analysis.dvc_field`/`dic_correlate`'s optional `reference` input.

---

## 8. Axis-changing nodes — `meta_transform` (V2.03 §A2)

A node that changes `AxisSizes` or calibration (Z-Project, Crop, Resample, Stack,
Stitch, Channel Split/Merge) MUST declare a **`meta_transform`** — a function in
[`metadata.py`](../../../nodegraph/metadata.py) (`z_project`/`crop`/`resample`/
`stack_time`/`channel_select`/`stitch`) that maps the incoming `MetaEnvelope` to the
outgoing one **touching no pixels**. The edit-time `propagate_meta` pass runs it so the
GUI's widget re-seed, the lever default, and the memo `OutputHeader.axes` are correct
before any pull.

**The payload MUST agree with the meta_transform prediction:**
- Update axes+calibration *in lockstep* in the compute — `reshaped_axes(new_axes)` +
  `with_metadata(...)` (a `None` value removes a key, e.g. z-project drops `z_step_um`).
- **Do not double-count a relative calibration change.** `ctx.calib` reads the node's
  **post-transform** envelope (the meta_transform already ran in `propagate_meta`). So a
  resample must **sync** the payload pixel size to the env's post value, *not* re-divide
  by the scale (that would apply the scale twice). Mirror the meta_transform; let it be
  the single source of truth. Match `span()`/bounds semantics between the compute and the
  meta_transform exactly (one-sided/out-of-range inputs included) or header≠payload.

---

## 9. Structure domains & transfer/bridges

- **Producing structure** (Label/Point/Track): build a `StructureTable`
  ([structure.py](../../../nodegraph/structure.py)) with the invariant
  `id,m,t,c,z,y,x` schema (`z` never NaN; `z_kind` = "plane_index" 2D / "subpixel" 3D)
  and attach via **`Dataset.with_structure(table)`** (columns → per-domain
  `AttributeLayer`s keyed by source `layer`). CCL → `label_components`; watershed →
  `seeded_watershed` + a region-props table; detection → `point_table`.
- **Voxel layers** (masks, distance fields) are lattice attributes:
  `Dataset.with_layer(Domain.VOXEL, name, values6d)` (shape validated against axes).
- **Domain transfer** ([transfer.py](../../../nodegraph/transfer.py)): lattice↔lattice is
  *generated* (reduce over dropped axes / broadcast) and executes via `execute_transfer`;
  structure hops use **explicit bridges** ([bridges.py](../../../nodegraph/bridges.py):
  `voxel_to_label`, `label_to_voxel`, `voxel_to_point`, `point_to_voxel`, `points_in_label`,
  `containing_label`, `gather_by_track`, `broadcast_track`, …). An **indirect route**
  (Voxel→Track = Voxel→Label→Track) is planned by `plan_transfer` and chained per
  frame/volume by **`execute_bridge_plan`** (C2) with the structure inputs (label raster /
  points / track membership).
- **Boundary** ([boundary.py](../../../nodegraph/boundary.py)): the constructive
  Point↔Label spine (`fill_boundary`/`extract_boundary`) — an explicit data-layer op,
  **not** an auto-routed bridge; the input geometry is the dimensionality authority.

---

## 10. The memo invariant (why identity is `revision`, not `id()`)

- **`recipe_hash`** (lookup): op + params (incl. `__modes__` mode/lever state) + upstream
  recipe_hashes + upstream **revisions**. A cheap proxy — never hashes pixels.
- **Declared reads** (`ctx.calib`) are re-validated on a hit — a metadata change the node
  read invalidates it; one it didn't read does not.
- **`output_fingerprint`** (content) drives cutoff/dedup and **includes the image
  provider's `fingerprint()`** (two datasets differing only by image must not collide).
- **Source identity:** a source node folds its provider's `.version` (C5) into the key so
  two providers with different data don't collide on one cached payload.
- Layer content identity is the monotonic **`revision`** — never `id()`, never a content
  hash for lookup.

---

## 11. Verify gate (see build-node-v2 §4 for the full procedure)

```bash
python -m nodegraph.selftest                          # all groups green (core + catalog)
python scripts/_nodelab_v2_phase5_probe.py out.png    # the driven GUI probe
```
On Windows, prefix `PYTHONUTF8=1` (some `[ok]` lines carry `↔`/`σ` glyphs). A v2 node's
gate is `nodegraph.selftest` — add an end-to-end pull through the `Engine` that asserts:
the footprint resolves per dim, 2D vs 3D produce distinct recipe hashes, an axis-changing
node's payload axes/calibration equal its `meta_transform` prediction, and structure output
lands on the right domain/layer.

---

## 12. Compute performance — numba vs numpy/scipy (when to reach for it)

A compute body is fast **by default** when it bottoms out in one vectorized numpy op
or a scipy/skimage/cv2 call — those are already compiled C/Fortran/MKL and numba
**cannot beat them**. Reach for numba **only** when the hot cost is a *Python loop of
many small array ops* or a *sequential, data-dependent algorithm that won't vectorize*.

**The decision rule** — ask of the hot path: *is wall-time dominated by one big compiled
call, or by a Python loop doing thousands of tiny ops?*

| The compute's hot path… | Use | Why |
|---|---|---|
| one big `ndimage.*` / `skimage.*` / `cv2.*` / `np.fft` / vectorized numpy | **numpy/scipy** | already C/MKL; numba ties or loses **and** adds compile latency |
| element-wise math numpy expands into N temporaries (`a*b + c*d - e`) | **numba** `njit` | fuses one pass, no temp arrays (memory-bandwidth bound) |
| a Python `for` over 10³–10⁶ items, each a few small numpy calls | **numba** `njit` | erases per-call dispatch + alloc overhead (the real win: 10–100×) |
| sequential/data-dependent (greedy NMS, region-grow, per-subset solve, ODE step) | **numba** `njit` | can't vectorize; numpy forces slow-Python or wasteful over-compute |
| the above **and** embarrassingly parallel per element | **numba** `njit(parallel=True)` + `prange` | frees the GIL, uses all cores in-process |

**Codebase idiom** (match it — see `kernels/bead_detect.py`, `kernels/track_objects.py`):
`import numba as nb`; module-level `@nb.njit(cache=True)` kernels with explicit dtypes;
`@nb.njit(parallel=True, cache=True)` + `nb.prange` for the parallel ones. Keep the
kernel a **pure numeric helper** (arrays in, arrays out) — no `ctx`, no `Dataset`, no
scipy calls inside `njit` (numba can't call them; keep `map_coordinates`/`label`/FFT
outside the kernel and pass the arrays in).

**Non-negotiable gotchas for THIS engine** (streaming + spawn):
- **`cache=True` is mandatory.** v2 does per-tile streaming eval and the DVC path fans
  across processes with `ProcessPoolExecutor` — Windows uses **spawn**, so every worker
  is a fresh interpreter that re-JITs on first call (~0.3–2 s each). `cache=True` writes
  the compiled artifact to disk so workers/tiles/re-runs load instead of recompiling.
  Without it numba can go **net-negative** on short interactive pulls.
- **First-ever call still pays one cold compile.** Fine for a long analysis solve; weigh
  it for a tiny per-tile pointwise op (there, a numba port may not be worth it at all —
  vectorized numpy already wins those).
- **Confirm the numba cache dir is writable** in the packaged/frozen app, or every run
  recompiles cold.
- **Don't stack redundant parallelism.** If the node already fans across processes
  (DVC IC-GN), `parallel=True` inside the kernel competes with the pool — prefer a
  serial `njit` kernel per worker, or `prange` only on the non-fanned path.

Backends stay **lazily imported inside the compute** (§2); a numba kernel is a
module-level helper, so importing it at module top is fine (numba is a declared dep) —
but never let a kernel's presence pull scipy/skimage to module top.
