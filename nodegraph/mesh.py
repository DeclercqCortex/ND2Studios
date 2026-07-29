"""The **Mesh** domain — boundary surfaces as flat CSR strata (nodegraph v2, V2.08).

A mesh is the one piece of geometry no other v2 domain can hold: its irreducible
content is **topology** (which vertices form which face). Vertices alone are Points, a
filled region is a Voxel Label, and "which vertices belong to which region" is a
Label-over-Points grouping — all of those already exist. Faces do not.

Why this shape
--------------
Blender splits mesh data across POINT / EDGE / FACE / CORNER domains precisely because
a mesh has several different array **lengths**. v2 gets one new domain, so the differing
lengths ride the second axis the store already provides: ``LayerKey = (Domain, layer,
name)``. A mesh named ``L`` therefore occupies **three layer buckets**, each internally
uniform-length::

    (MESH, "L",      *)  → K   rows, one per mesh ELEMENT (one closed surface)
    (MESH, "L/vert", *)  → Nv  rows, the vertex pool
    (MESH, "L/face", *)  → Nf  rows, the face pool (triangles)

Each bucket is a legal :class:`~nodegraph.structure.StructureTable` carrying the
invariant ``id,m,t,c,z,y,x`` coordinate schema, so ``StructureTable.n`` stays meaningful,
``to_arrow`` works per bucket, and the Spreadsheet renders three clean tables instead of
one ragged one. **Every array is flat and fixed-width** (int64 / float64): no object
dtype, no ragged column, no 2-D array, no live ``scipy.spatial.Delaunay``. That is a
correctness requirement, not tidiness — :func:`nodegraph.memo._canon` hashes an ndarray
via ``ascontiguousarray(...).tobytes()``, which for object dtype hashes *pointer bytes*,
so identical meshes would hash differently (permanent memo miss) and distinct meshes
could collide via a reused CPython pointer slot (a wrong payload served). ``_canon`` now
refuses object dtype outright, which is what makes this contract enforceable.

Element → pool addressing is **CSR**: each element row carries ``vert_start/vert_count``
and ``face_start/face_count``. ``start``+``count`` rather than a Blender-style ``(K+1)``
offsets array specifically so *every* element column is exactly length K — a stray K+1
column would break the bucket's uniformity. Blender's offsets are one line away:
``np.concatenate([[0], np.cumsum(vert_count)])``.

Coordinate space
----------------
Vertices are **voxel ``(z,y,x)``**, not world ``(x,y,z)`` µm, deliberately against the
vendored kernels' own ``vertices_um`` convention. Every other v2 structure table's
``z,y,x`` are voxel coords and ``COORD_COLUMNS`` carries no unit tag, so a µm mesh would
be the only geometry in the store whose *meaning* depends on calibration that is not
captured in the layer — a recalibrated upstream would silently reinterpret it. Keeping
voxels puts the µm↔voxel flip inside computes behind ``ctx.calib``, where the memo fence
already lives, and makes a vertex row Point-schema-compatible (so the Viewer's existing
point machinery can draw mesh vertices). Producers convert on the way in; consumers
convert on the way out.

Triangles only, for now
-----------------------
Every mesh producer in this repo emits ``(Nf,3)`` triangles (alpha-shape boundary faces,
``ConvexHull.simplices``, ``marching_cubes``), so a face row stores ``v0,v1,v2`` directly
and there is no CORNER stratum. The layer sub-key naming keeps that additive: n-gon
support later means a new ``L/corner`` bucket plus a marker, not a migration of what is
written here.

Qt-free; numpy only (no scipy).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from nodegraph.domains import Domain
from nodegraph.memo import digest
from nodegraph.structure import StructureTable

# The layer sub-key separator + the strata that hang off a mesh's base name. Verified
# safe: nothing in ``nodegraph/`` or ``nodelab_v2/`` splits, prefixes or path-parses a
# ``.layer`` string — they are fully opaque — so ``/`` is available as a reserved
# separator. ``validate`` refuses a base name that contains it (see ``_check_name``).
MESH_SEP = "/"
MESH_STRATA: Tuple[str, ...] = ("vert", "face")

#: metadata key holding the per-mesh provenance stamp (``wire-node-v2`` §7b). Namespaced
#: and non-calibration, so ``strict_reads`` passes a consumer's read straight through.
MESH_PROVENANCE_KEY = "__mesh_provenance__"


def mesh_layer(name: str, part: Optional[str] = None) -> str:
    """The store ``layer`` key for a mesh's element bucket (``part=None``) or one of its
    strata (``part="vert"``/``"face"``). The single implementation of the convention."""
    if part is None:
        return name
    if part not in MESH_STRATA:
        raise ValueError(f"unknown mesh stratum {part!r} (expected one of {MESH_STRATA})")
    return f"{name}{MESH_SEP}{part}"


def mesh_part(layer: str) -> Tuple[str, Optional[str]]:
    """Inverse of :func:`mesh_layer` — ``("L/vert") -> ("L", "vert")``, ``("L") -> ("L", None)``.
    An unrecognized suffix is treated as part of the base name (not an error), so a
    non-mesh layer that happens to contain ``/`` round-trips unchanged."""
    base, sep, part = layer.rpartition(MESH_SEP)
    if sep and part in MESH_STRATA:
        return base, part
    return layer, None


def mesh_names(dataset: Any) -> Tuple[str, ...]:
    """The base names of every mesh in ``dataset``, sorted — a mesh is present iff its
    element bucket is (the strata alone are not a mesh)."""
    names = {a.layer for a in dataset.layers_on(Domain.MESH)
             if a.layer is not None and mesh_part(a.layer)[1] is None}
    return tuple(sorted(names))


# ── the producer-side value type ──────────────────────────────────────────────────

@dataclass
class MeshElement:
    """One closed surface, as a producer hands it over — **element-local** faces and
    **voxel ``(z,y,x)``** vertices. :func:`build_mesh_tables` does all concatenation,
    global-index rebasing and CSR bookkeeping, so no node open-codes it."""

    m: int
    t: int
    c: int
    src_label: int                        # the ORIGINAL upstream cluster / region id
    verts_zyx: np.ndarray                 # (Nv, 3) float, voxel (z,y,x)
    faces: np.ndarray                     # (Nf, 3) int, ELEMENT-LOCAL vertex indices
    centroid_zyx: Tuple[float, float, float]
    volume_um3: float = 0.0
    surface_area_um2: float = 0.0
    density: float = 0.0
    n_points: int = 0

    def __post_init__(self) -> None:
        v = np.asarray(self.verts_zyx, dtype=float).reshape(-1, 3)
        f = np.asarray(self.faces, dtype=np.int64).reshape(-1, 3)
        if len(f) and (f.min() < 0 or f.max() >= len(v)):
            raise ValueError(
                f"element {self.src_label}: face index out of range for {len(v)} vertices")
        self.verts_zyx = v
        self.faces = f


# ── the three-bucket table set ────────────────────────────────────────────────────

@dataclass(frozen=True)
class MeshTables:
    """The three strata of one mesh instance. Build with :func:`build_mesh_tables`,
    attach with :func:`with_mesh`, read back with :func:`read_mesh`."""

    element: StructureTable
    vertex: StructureTable
    face: StructureTable
    layer: str = "mesh"

    @property
    def n_elements(self) -> int:
        return self.element.n

    def content_hash(self) -> str:
        """Content hash over all three strata in a fixed order. Deterministic *because*
        every column is flat and fixed-width — the assertion that closes the object-dtype
        trap (two independently built identical meshes must hash equal)."""
        return digest("mesh", self.layer, self.element.content_hash(),
                      self.vertex.content_hash(), self.face.content_hash())

    def validate(self) -> "MeshTables":
        """Assert every CSR / foreign-key / dtype invariant, or raise ``ValueError``.

        This is the **entire** enforcement mechanism: ``Dataset.with_attribute`` gates its
        only shape check on ``is_lattice``, so a non-lattice layer gets *zero* store-level
        validation (two attribute layers of different lengths coexist on one layer key
        without complaint). :func:`with_mesh` therefore calls this unconditionally, and is
        the only sanctioned write path — never call ``Dataset.with_structure`` on a mesh
        bucket directly.
        """
        _check_name(self.layer)
        el, vt, fc = self.element, self.vertex, self.face
        for tbl, what in ((el, "element"), (vt, "vertex"), (fc, "face")):
            for col, values in tbl.columns.items():
                a = np.asarray(values)
                if a.dtype == object:
                    raise ValueError(
                        f"mesh {self.layer!r} {what} column {col!r} is object dtype — "
                        "mesh columns must be flat fixed-width arrays (object arrays hash "
                        "by pointer, which corrupts the memo)")
                if a.ndim != 1:
                    raise ValueError(
                        f"mesh {self.layer!r} {what} column {col!r} is {a.ndim}-D; "
                        "mesh columns must be 1-D")
            n = tbl.n
            for col, values in tbl.columns.items():
                if len(np.asarray(values)) != n:
                    raise ValueError(
                        f"mesh {self.layer!r} {what} column {col!r} has length "
                        f"{len(np.asarray(values))}, expected {n} (bucket must be uniform)")
        k, nv, nf = el.n, vt.n, fc.n

        for req in ("id", "m", "t", "c", "z", "y", "x", "element_uid",
                    "vert_start", "vert_count", "face_start", "face_count"):
            if req not in el.columns:
                raise ValueError(f"mesh {self.layer!r} element bucket missing {req!r}")
        for req in ("id", "m", "t", "c", "z", "y", "x", "element"):
            if req not in vt.columns:
                raise ValueError(f"mesh {self.layer!r} vertex bucket missing {req!r}")
        for req in ("id", "m", "t", "c", "element", "v0", "v1", "v2"):
            if req not in fc.columns:
                raise ValueError(f"mesh {self.layer!r} face bucket missing {req!r}")

        uid = np.asarray(el.columns["element_uid"], dtype=np.int64)
        if not np.array_equal(uid, np.arange(k, dtype=np.int64)):
            raise ValueError(
                f"mesh {self.layer!r}: element_uid must be 0..K-1 in row order (it is the "
                "foreign key the vertex/face buckets point at)")
        vs = np.asarray(el.columns["vert_start"], dtype=np.int64)
        vc = np.asarray(el.columns["vert_count"], dtype=np.int64)
        fs = np.asarray(el.columns["face_start"], dtype=np.int64)
        f_c = np.asarray(el.columns["face_count"], dtype=np.int64)
        _check_csr(self.layer, "vertex", vs, vc, nv)
        _check_csr(self.layer, "face", fs, f_c, nf)

        # dense per-(m,t,c) ids — the ids the rasterizer paints and the Label table joins on
        em = np.asarray(el.columns["m"], dtype=np.int64)
        et = np.asarray(el.columns["t"], dtype=np.int64)
        ec = np.asarray(el.columns["c"], dtype=np.int64)
        eid = np.asarray(el.columns["id"], dtype=np.int64)
        for key in {(int(a), int(b), int(d)) for a, b, d in zip(em, et, ec)}:
            sel = (em == key[0]) & (et == key[1]) & (ec == key[2])
            want = np.arange(1, int(sel.sum()) + 1, dtype=np.int64)
            if not np.array_equal(np.sort(eid[sel]), want):
                raise ValueError(
                    f"mesh {self.layer!r}: element ids at (m,t,c)={key} are "
                    f"{sorted(eid[sel].tolist())}, expected dense 1..{int(sel.sum())}")

        if k and nv:
            vown = np.asarray(vt.columns["element"], dtype=np.int64)
            if vown.min() < 0 or vown.max() >= k:
                raise ValueError(f"mesh {self.layer!r}: vertex.element out of range")
            # every vertex must sit inside its owner's CSR slice
            if not np.all((np.arange(nv) >= vs[vown]) & (np.arange(nv) < vs[vown] + vc[vown])):
                raise ValueError(
                    f"mesh {self.layer!r}: vertex rows are not contiguous within their "
                    "owning element's vert_start/vert_count slice")
        if nf:
            fown = np.asarray(fc.columns["element"], dtype=np.int64)
            if k and (fown.min() < 0 or fown.max() >= k):
                raise ValueError(f"mesh {self.layer!r}: face.element out of range")
            tri = np.column_stack([np.asarray(fc.columns[f"v{i}"], dtype=np.int64)
                                   for i in range(3)])
            if tri.min() < 0 or tri.max() >= max(nv, 1):
                raise ValueError(
                    f"mesh {self.layer!r}: face vertex index out of range [0,{nv})")
            # face vertices are GLOBAL indices and must lie in the owner's vertex slice
            lo = vs[fown][:, None]
            hi = (vs[fown] + vc[fown])[:, None]
            if not np.all((tri >= lo) & (tri < hi)):
                raise ValueError(
                    f"mesh {self.layer!r}: a face references a vertex outside its own "
                    "element (v0/v1/v2 are GLOBAL indices into the vertex pool)")
        return self


def _check_name(name: str) -> None:
    if not name:
        raise ValueError("a mesh needs a non-empty layer name")
    if MESH_SEP in name:
        raise ValueError(
            f"mesh name {name!r} may not contain {MESH_SEP!r} — it is reserved as the "
            f"stratum separator (a layer named 'foo/vert' would collide with mesh 'foo')")


def _check_csr(layer: str, what: str, start: np.ndarray, count: np.ndarray,
               total: int) -> None:
    if np.any(count < 0):
        raise ValueError(f"mesh {layer!r}: negative {what} count")
    if int(count.sum()) != total:
        raise ValueError(
            f"mesh {layer!r}: {what} counts sum to {int(count.sum())} but the pool holds "
            f"{total} rows")
    expected = np.concatenate([[0], np.cumsum(count)[:-1]]) if len(count) else count
    if not np.array_equal(start, expected.astype(np.int64)):
        raise ValueError(
            f"mesh {layer!r}: {what}_start is not the contiguous prefix sum of "
            f"{what}_count (slices must partition the pool in row order)")


# ── build ─────────────────────────────────────────────────────────────────────────

def build_mesh_tables(elements: Sequence[MeshElement], *, layer: str = "mesh",
                      z_kind: str = "subpixel") -> MeshTables:
    """Concatenate ``elements`` into the three CSR strata. Assigns ``element_uid`` in row
    order, ``id`` densely ``1..K`` **per (m,t,c)** (matching ``_label_table`` and the
    raster paint, so mesh element ↔ raster region ↔ Label row are id-identical), and
    rebases each element's local faces onto the global vertex pool.

    Elements are sorted by ``(m,t,c,src_label)`` so the output is deterministic
    regardless of the producer's iteration order.
    """
    _check_name(layer)
    els = sorted(elements, key=lambda e: (int(e.m), int(e.t), int(e.c), int(e.src_label)))
    k = len(els)

    e_id = np.zeros(k, dtype=np.int64)
    seen: Dict[Tuple[int, int, int], int] = {}
    for i, e in enumerate(els):
        key = (int(e.m), int(e.t), int(e.c))
        seen[key] = seen.get(key, 0) + 1
        e_id[i] = seen[key]

    vcount = np.array([len(e.verts_zyx) for e in els], dtype=np.int64)
    fcount = np.array([len(e.faces) for e in els], dtype=np.int64)
    vstart = np.concatenate([[0], np.cumsum(vcount)[:-1]]).astype(np.int64) if k else vcount
    fstart = np.concatenate([[0], np.cumsum(fcount)[:-1]]).astype(np.int64) if k else fcount

    def ecol(fn, dtype) -> np.ndarray:
        return np.array([fn(e) for e in els], dtype=dtype) if k else np.zeros(0, dtype=dtype)

    element = StructureTable(Domain.MESH, {
        "id": e_id,
        "element_uid": np.arange(k, dtype=np.int64),
        "m": ecol(lambda e: int(e.m), np.int64),
        "t": ecol(lambda e: int(e.t), np.int64),
        "c": ecol(lambda e: int(e.c), np.int64),
        "z": ecol(lambda e: float(e.centroid_zyx[0]), float),
        "y": ecol(lambda e: float(e.centroid_zyx[1]), float),
        "x": ecol(lambda e: float(e.centroid_zyx[2]), float),
        "src_label": ecol(lambda e: int(e.src_label), np.int64),
        "vert_start": vstart, "vert_count": vcount,
        "face_start": fstart, "face_count": fcount,
        "volume_um3": ecol(lambda e: float(e.volume_um3), float),
        "surface_area_um2": ecol(lambda e: float(e.surface_area_um2), float),
        "density": ecol(lambda e: float(e.density), float),
        "n_points": ecol(lambda e: int(e.n_points), np.int64),
        "closed": np.zeros(k, dtype=np.int64),      # filled in below
    }, layer=mesh_layer(layer), z_kind=z_kind)

    verts = (np.vstack([e.verts_zyx for e in els]) if k and vcount.sum()
             else np.zeros((0, 3), dtype=float))
    vown = np.repeat(np.arange(k, dtype=np.int64), vcount) if k else np.zeros(0, np.int64)
    vertex = StructureTable(Domain.MESH, {
        "id": np.arange(len(verts), dtype=np.int64),
        "element": vown,
        "m": np.asarray(element.columns["m"])[vown] if k else np.zeros(0, np.int64),
        "t": np.asarray(element.columns["t"])[vown] if k else np.zeros(0, np.int64),
        "c": np.asarray(element.columns["c"])[vown] if k else np.zeros(0, np.int64),
        "z": verts[:, 0], "y": verts[:, 1], "x": verts[:, 2],
    }, layer=mesh_layer(layer, "vert"), z_kind=z_kind)

    tri = (np.vstack([e.faces + vstart[i] for i, e in enumerate(els) if len(e.faces)])
           if k and fcount.sum() else np.zeros((0, 3), dtype=np.int64))
    fown = np.repeat(np.arange(k, dtype=np.int64), fcount) if k else np.zeros(0, np.int64)
    face = StructureTable(Domain.MESH, {
        "id": np.arange(len(tri), dtype=np.int64),
        "element": fown,
        "m": np.asarray(element.columns["m"])[fown] if k else np.zeros(0, np.int64),
        "t": np.asarray(element.columns["t"])[fown] if k else np.zeros(0, np.int64),
        "c": np.asarray(element.columns["c"])[fown] if k else np.zeros(0, np.int64),
        "v0": tri[:, 0], "v1": tri[:, 1], "v2": tri[:, 2],
    }, layer=mesh_layer(layer, "face"), z_kind=z_kind)

    # ``closed`` is derived, never author-supplied: it is what the rasterizer branches on.
    closed = np.zeros(k, dtype=np.int64)
    for i in range(k):
        f = tri[fstart[i]:fstart[i] + fcount[i]]
        closed[i] = 1 if faces_are_closed(f) else 0
    cols = dict(element.columns)
    cols["closed"] = closed
    element = StructureTable(Domain.MESH, cols, layer=element.layer, z_kind=z_kind)

    return MeshTables(element, vertex, face, layer=layer).validate()


def faces_are_closed(faces: np.ndarray) -> bool:
    """True if ``faces`` (Nf,3) is watertight — every undirected edge shared by exactly
    two triangles. Orientation-independent (edges are sorted), which matters because the
    vendored producers store raw unsorted vertex triples, so winding is arbitrary."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) < 4:
        return False
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    e = np.sort(e, axis=1)
    _, counts = np.unique(e, axis=0, return_counts=True)
    return bool(np.all(counts == 2))


def enclosed_volume_um3(verts_zyx: np.ndarray, faces: np.ndarray,
                        voxel_size_um: Sequence[float]) -> float:
    """Enclosed volume (µm³) of a closed triangle mesh by the divergence theorem,
    ``|Σ a·(b×c)| / 6``. ``abs`` because the vendored producers' winding is arbitrary
    (a globally consistent but inward orientation would just flip the sign). 0 with no
    faces. Meaningless on an open surface — check :func:`faces_are_closed` first."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return 0.0
    dz, dy, dx = (float(v) for v in voxel_size_um)
    v = np.asarray(verts_zyx, dtype=float).reshape(-1, 3) * np.array([dz, dy, dx])
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)


def surface_area_um2(verts_zyx: np.ndarray, faces: np.ndarray,
                     voxel_size_um: Sequence[float]) -> float:
    """Triangle-mesh surface area (µm²) from voxel-space vertices. 0 with no faces
    (a Voronoi/degenerate boundary can legitimately carry ``(0,3)`` faces)."""
    f = np.asarray(faces, dtype=np.int64).reshape(-1, 3)
    if len(f) == 0:
        return 0.0
    dz, dy, dx = (float(v) for v in voxel_size_um)
    v = np.asarray(verts_zyx, dtype=float).reshape(-1, 3) * np.array([dz, dy, dx])
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    return float(0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum())


# ── attach / read ─────────────────────────────────────────────────────────────────

def with_mesh(dataset: Any, tables: MeshTables, *,
              provenance: Optional[Dict[str, Any]] = None) -> Any:
    """Attach ``tables`` to ``dataset`` as three structure buckets (+ an optional
    provenance stamp) and return the new Dataset. **The only sanctioned mesh write path**
    — it calls :meth:`MeshTables.validate` first, which is the sole enforcement of the
    CSR invariants the store itself will not check.

    ``provenance`` is stamped under :data:`MESH_PROVENANCE_KEY` keyed by mesh name — the
    ``wire-node-v2`` §7b stamp-and-inherit pattern. It lets the rasterizer *derive* its
    interior test instead of exposing a lever the user could set to disagree with the data.
    """
    tables.validate()
    ds = dataset
    for tbl in (tables.element, tables.vertex, tables.face):
        ds = ds.with_structure(tbl)
    if provenance:
        pmap = dict(ds.metadata.get(MESH_PROVENANCE_KEY, {}))
        pmap[tables.layer] = dict(provenance)
        ds = ds.with_metadata(**{MESH_PROVENANCE_KEY: pmap})
    return ds


def mesh_provenance(dataset: Any, layer: str) -> Dict[str, Any]:
    """The provenance a producer stamped for mesh ``layer`` (``{}`` if none). Reading it
    off the input payload's metadata is memo-safe: the marker rides the payload, so an
    upstream change bumps the upstream revision, which already folds into the consumer's
    recipe hash — and it is a non-calibration key, so ``strict_reads`` passes it through."""
    return dict(dataset.metadata.get(MESH_PROVENANCE_KEY, {}).get(layer, {}))


def read_mesh(dataset: Any, layer: str = "mesh") -> MeshTables:
    """Reassemble :class:`MeshTables` for mesh ``layer`` from the attribute store."""
    _check_name(layer)

    def bucket(part: Optional[str]) -> StructureTable:
        key = mesh_layer(layer, part)
        cols = {a.name: np.asarray(a.values)
                for a in dataset.layers_on(Domain.MESH) if a.layer == key}
        if not cols:
            raise ValueError(
                f"no mesh {layer!r} on this Dataset (missing bucket {key!r}) — "
                "wire a tessellation node upstream")
        return StructureTable(Domain.MESH, cols, layer=key,
                              z_kind=(dataset.structure_zkind(Domain.MESH, key)
                                      or "subpixel"))

    return MeshTables(bucket(None), bucket("vert"), bucket("face"), layer=layer)


def mesh_element(tables: MeshTables, row: int) -> Tuple[np.ndarray, np.ndarray]:
    """Slice one element out as ``(verts_zyx (Nv_e,3) float, faces (Nf_e,3) int)`` with
    faces rebased to **element-local** indices (what every mesh kernel wants)."""
    el = tables.element.columns
    vs = int(np.asarray(el["vert_start"])[row])
    vc = int(np.asarray(el["vert_count"])[row])
    fs = int(np.asarray(el["face_start"])[row])
    fc = int(np.asarray(el["face_count"])[row])
    v = tables.vertex.columns
    verts = np.column_stack([np.asarray(v["z"], dtype=float)[vs:vs + vc],
                             np.asarray(v["y"], dtype=float)[vs:vs + vc],
                             np.asarray(v["x"], dtype=float)[vs:vs + vc]])
    f = tables.face.columns
    faces = np.column_stack([np.asarray(f[f"v{i}"], dtype=np.int64)[fs:fs + fc]
                             for i in range(3)]) - vs
    return verts, faces.reshape(-1, 3)


def mesh_rows(tables: MeshTables) -> List[Dict[str, Any]]:
    """The element bucket as a list of plain per-row dicts (iteration convenience)."""
    cols = tables.element.columns
    return [{k: np.asarray(v)[i] for k, v in cols.items()}
            for i in range(tables.element.n)]


__all__ = [
    "MESH_SEP", "MESH_STRATA", "MESH_PROVENANCE_KEY",
    "mesh_layer", "mesh_part", "mesh_names",
    "MeshElement", "MeshTables", "build_mesh_tables",
    "faces_are_closed", "surface_area_um2", "enclosed_volume_um3",
    "with_mesh", "mesh_provenance", "read_mesh", "mesh_element", "mesh_rows",
]
