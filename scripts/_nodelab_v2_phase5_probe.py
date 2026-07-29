"""Offscreen end-to-end probe of the Phase-5 GUI slice (G1/G2/G4/G6/G7/G8/G10).

    PYTHONUTF8=1 python scripts/_nodelab_v2_phase5_probe.py [out.png]

Boots the real MainWindow offscreen and drives the same code paths the mouse would:
document wiring rules (validity, cycle rejection, non-multi replace), link-search
compatibility, save/load round-trip incl. canvas positions, a REAL EngineRunner pull
on the synthetic source through to viewer pixels, the H11 lever guard, mute
pass-through, and an inspector param edit re-seeding a downstream ƒmd pill (G8).
Asserts + printed checkmarks; exits via os._exit (offscreen teardown crash gotcha).
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtCore import QPointF            # noqa: E402
from PySide6.QtGui import QFontDatabase       # noqa: E402
from PySide6.QtWidgets import QApplication    # noqa: E402


def _load_fonts() -> None:
    for name in ("segoeui.ttf", "consola.ttf", "arial.ttf", "seguisb.ttf"):
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)


def _ok(msg: str) -> None:
    print(f"[ok] {msg}")


def main(argv) -> int:
    out = argv[1] if len(argv) > 1 else "nodelab_v2_phase5.png"
    app = QApplication(sys.argv[:1])
    _load_fonts()

    from nodegraph.dataset import AxisSizes
    from nodegraph.metadata import MetaEnvelope
    from nodelab_v2 import theme as T
    from nodelab_v2.scene import compatible_ops
    from nodelab_v2.window import MainWindow

    win = MainWindow()
    win.resize(1500, 880)
    win.show()
    _seen_fail = []
    win.runner.failed.connect(
        lambda nid, tr: _seen_fail.append((nid, tr.strip().splitlines()[-1])))
    app.processEvents()

    # ── launch state: a blank canvas with the welcome card (no demo graph) ─────
    assert not win.doc.nodes, "the app must open on an empty canvas"
    assert win.welcome.isVisible() and win.welcome.parent() is win.view
    wg, vw, vh = win.welcome.geometry(), win.view.width(), win.view.height()
    assert abs(wg.center().x() - vw / 2) <= 2 and abs(wg.center().y() - vh / 2) <= 2
    assert abs(win.view.transform().m11() - 1.0) < 1e-9   # 1:1, not zoomed into nothing
    win.welcome.op_dropped.emit("enhance.gamma", QPointF(40.0, 40.0))   # drop passthrough
    app.processEvents()
    assert len(win.doc.nodes) == 1 and not win.welcome.isVisible()
    win.file_new()
    app.processEvents()
    assert not win.doc.nodes and win.welcome.isVisible()   # File → New re-welcomes
    _ok("Launch: blank canvas + centred welcome card (drop passthrough places a node)")

    win.build_demo()                       # everything below drives the example graph
    app.processEvents()
    assert len(win.doc.nodes) == 8 and not win.welcome.isVisible()
    doc = win.doc

    # ── G1: wiring rules through the document (what socket drags execute) ──────
    ok, _ = doc.can_connect("n2", "out", "n4", "data")
    assert ok, "dataset→dataset should connect"
    ok, why = doc.can_connect("n4", "out", "n2", "data")
    assert not ok and "cycle" in why, f"cycle must be rejected (got {why!r})"
    ok, _ = doc.can_connect("n2", "out", "n2", "data")
    assert not ok, "self-loop must be rejected"
    # non-multi replace: n4.data currently fed by n3 — reconnect from n2 replaces it
    before = doc.edge_into("n4", "data")
    assert before == ("n3", "out", "n4", "data")
    removed = doc.connect("n2", "out", "n4", "data")
    assert removed == [("n3", "out", "n4", "data")]
    assert doc.edge_into("n4", "data") == ("n2", "out", "n4", "data")
    doc.connect("n3", "out", "n4", "data")          # restore the demo chain
    _ok("G1: validity, cycle/self-loop rejection, non-multi replace")

    # link-drag search offers compatible ops for a Dataset output
    n2_out = win.scene.node_items["n2"].socket("out", "out")
    entries = compatible_ops(n2_out.spec, "out")
    labels = {spec.op_key for spec, _s in entries}
    assert "enhance.gaussian" in labels and "view.viewer" in labels
    assert not any(op.startswith(("zone.", "group.", "test.")) for op in labels)
    _ok(f"G1: link-drag search offers {len(entries)} compatible ops (fixtures hidden)")

    # G1 (crash regression 2026-07-27): DROPPING a wire on a socket — the real mouse
    # path, which no probe covered. `_end_wire` tears the drag down (`_cancel_temp`
    # clears `_drag_fixed`) BEFORE resolving the drop, so the anchor must be passed
    # explicitly; reading it back from state raised AttributeError on every connect.
    from PySide6.QtCore import QPoint as _QPoint
    from nodelab_v2.document import GraphDocument as _GD0
    from nodelab_v2.scene import GraphScene as _GS0
    wdoc = _GD0()
    wsc = _GS0(wdoc)
    wdoc.add_node("io.load", node_id="wa", x=0, y=0)
    wdoc.add_node("enhance.gamma", node_id="wb", x=300, y=0)
    wdoc.add_node("enhance.gamma", node_id="wc", x=600, y=0)
    a_out = wsc.node_items["wa"].socket("out", "image")
    b_in = wsc.node_items["wb"].socket("in", "data")
    wsc.begin_wire(a_out, a_out.anchor())
    assert wsc._drag_fixed is a_out and wsc._temp_wire is not None
    wsc._update_temp(b_in.anchor())                       # hover highlights the target
    assert b_in.highlight is True
    wsc._end_wire(b_in.anchor(), _QPoint(0, 0))           # ← used to raise
    assert ("wa", "image", "wb", "data") in wdoc.edges, wdoc.edges
    assert wsc._drag_fixed is None and wsc._temp_wire is None and b_in.highlight is None
    # dragging FROM a connected non-multi input detaches and re-drags from the source,
    # so dropping it on another node's input moves the wire there
    c_in = wsc.node_items["wc"].socket("in", "data")
    wsc.begin_wire(b_in, b_in.anchor())
    assert wsc._drag_fixed is a_out                       # re-anchored to the source end
    wsc._end_wire(c_in.anchor(), _QPoint(0, 0))
    assert ("wa", "image", "wc", "data") in wdoc.edges
    assert ("wa", "image", "wb", "data") not in wdoc.edges
    # an invalid drop (self-loop) connects nothing and still leaves no dangling drag
    wsc.begin_wire(a_out, a_out.anchor())
    wsc._end_wire(wsc.node_items["wa"].socket("out", "image").anchor(), _QPoint(0, 0))
    assert wsc._drag_fixed is None and len(wdoc.edges) == 1
    _ok("G1: wire DROP on a socket connects (+ detach/re-drag, invalid drop is a no-op)")

    # ── G8: live derive re-seed — the ƒmd pill follows the propagated envelope ──
    g_item = win.scene.node_items["n3"]                    # gaussian (σ derive-less
    d_item = win.scene.node_items["n7"]                    # deconvolve: na derives
    seed_env = MetaEnvelope(axes=AxisSizes(m=1, t=1, z=5, c=2, y=512, x=512),
                            metadata={"pixel_size_um": 0.1, "z_step_um": 0.3,
                                      "objective_na": 1.4,
                                      "channel_emission_nm": [520.0, 640.0]})
    doc.set_meta_seed("n1", seed_env)
    na_sock = d_item.spec.input("na")
    assert d_item.resolved(na_sock) == 1.4, d_item.resolved(na_sock)
    doc.set_meta_seed("n1", MetaEnvelope(axes=seed_env.axes,
                                         metadata={**seed_env.metadata,
                                                   "objective_na": 0.45}))
    assert d_item.resolved(na_sock) == 0.45      # pill re-seeded live (G8)
    # the INSPECTOR's auto boxes re-seed too (n7 is the selected node at startup)
    boxes = {s.name: b for _n, s, b in win.inspector._auto_boxes}
    assert "na" in boxes and abs(boxes["na"].value() - 0.45) < 1e-9, \
        {k: b.value() for k, b in boxes.items()}
    _ok("G8: ƒmd pill + inspector auto box re-seed from the envelope (1.4 → 0.45)")

    # ── H11 lever guard: z==1 disables 3D and flags a locked-3D node invalid ────
    assert not g_item.z_is_one() and g_item._switch.allow_3d
    doc.set_meta_seed("n1", MetaEnvelope(axes=AxisSizes(m=1, t=1, z=1, c=2,
                                                        y=512, x=512),
                                         metadata=dict(seed_env.metadata)))
    assert g_item.z_is_one() and not g_item._switch.allow_3d
    assert g_item.dim == "3D" and g_item.dim_invalid()      # demo starts 3D → red badge
    doc.set_meta_seed("n1", seed_env)                        # restore z=5
    assert g_item._switch.allow_3d and not g_item.dim_invalid()
    _ok("H11: z==1 greys the 3D lever + red-badges a locked-3D node; unknown ≠ 1")

    # ── mode-gated params: the CARD and the INSPECTOR both follow the live mode ──
    # n4 is analysis.threshold, whose `threshold` socket is `fixed`-only (a histogram
    # method derives its own cut). Changing the mode through the real inspector combo
    # must drop the socket row from the card AND rebuild the form — the inspector used
    # to keep painting the previous method's params (only `refresh_derived` ran).
    from PySide6.QtWidgets import QComboBox as _QCombo, QLabel as _QLabel
    t_item = win.scene.node_items["n4"]
    win.scene.clearSelection()
    t_item.setSelected(True)
    app.processEvents()
    assert win.inspector._node is t_item

    def _insp_params() -> set:
        return {w.text() for w in win.inspector.findChildren(_QLabel)}

    def _card_ins(item) -> set:
        return {k[1] for k in item._sockets if k[0] == "in"}

    m_combo = next(cb for cb in win.inspector.findChildren(_QCombo)
                   if "otsu" in [cb.itemText(i) for i in range(cb.count())])
    assert t_item.state()["method"] == "fixed"
    assert "threshold" in _card_ins(t_item) and "threshold" in _insp_params()
    m_combo.setCurrentText("otsu")            # the real signal path (currentTextChanged)
    app.processEvents()                       # the rebuild is deferred by one turn
    assert t_item.state()["method"] == "otsu"
    assert "threshold" not in _card_ins(t_item), "the card must drop the dead socket"
    assert "threshold" not in _insp_params(), "the inspector must rebuild off the mode"
    # …and the combo is still live after the rebuild that replaced it
    m_combo = next(cb for cb in win.inspector.findChildren(_QCombo)
                   if "otsu" in [cb.itemText(i) for i in range(cb.count())])
    m_combo.setCurrentText("fixed")
    app.processEvents()
    assert "threshold" in _card_ins(t_item) and "threshold" in _insp_params()
    _ok("mode gate: changing `method` re-resolves the active sockets on the card AND "
        "rebuilds the inspector form (analysis.threshold: fixed-only `threshold`)")

    # ── G3: mute pass-through in the run graph ──────────────────────────────────
    doc.set_muted("n4", True)
    g = doc.to_graph(for_run=True)
    assert "n4" not in {e.dst for e in g.edges} and "n4" not in {e.src for e in g.edges}
    assert any(e.src == "n3" and e.dst == "n5" for e in g.edges)   # bypassed around
    doc.set_muted("n4", False)
    _ok("G3: muted node is bypassed (n3 → n5) in the run graph")

    # ── G6: save / load round-trip incl. canvas positions ──────────────────────
    tmp = os.path.join(tempfile.mkdtemp(prefix="nd2graph_"), "t.nd2graph.json")
    doc.set_pos("n3", 123.0, 456.0)
    doc.save_file(tmp)
    with open(tmp, encoding="utf-8") as f:
        raw = json.load(f)
    assert raw["format_version"] == "2.0" and "ui" in raw
    n_nodes, n_edges = len(doc.nodes), len(doc.edges)
    doc.load_file(tmp)
    assert len(doc.nodes) == n_nodes and len(doc.edges) == n_edges
    assert (doc.nodes["n3"].x, doc.nodes["n3"].y) == (123.0, 456.0)
    # headless loader reads the same file (GUI extras ignored)
    from nodegraph.serialize import from_dict
    g2, _z, _gr = from_dict(raw)
    assert set(g2.nodes) == set(doc.nodes)
    _ok("G6: save/load round-trip (positions kept; headless loader reads the file)")

    # ── G7 + G4: a real pull on the synthetic source through to viewer pixels ──
    done = {}
    win.runner.finished.connect(lambda nid, *a: done.setdefault("id", nid))
    win.runner.failed.connect(lambda nid, tr: done.setdefault("err", tr))
    win.pull_node("n3")                                     # gaussian (3D on z=5)
    t0 = time.time()
    while not done and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert done.get("err") is None, f"pull failed:\n{done.get('err')}"
    assert done.get("id") == "n3"
    pm = win.viewer._view._item.pixmap()   # viewer is now a QGraphicsView (was a QLabel)
    assert pm is not None and not pm.isNull() and pm.width() > 100
    assert "pulled in" in win.viewer._status.text()
    _ok(f"G7+G4: engine pull off the UI thread → viewer shows {pm.width()}"
        f"×{pm.height()} px ({win.viewer._status.text().split('·')[-1].strip()})")

    # memo persistence: an immediate re-pull is a cache hit (fast)
    done.clear()
    t0 = time.time()
    win.pull_node("n3")
    while not done and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    repull = time.time() - t0
    assert done.get("id") == "n3" and repull < 5.0
    _ok(f"G7: re-pull hits the persistent memo ({repull*1000:.0f} ms)")

    # ── G2: palette content ─────────────────────────────────────────────────────
    win.palette.refill("gauss")
    tree = win.palette._tree
    found = []
    for i in range(tree.topLevelItemCount()):
        head = tree.topLevelItem(i)
        for j in range(head.childCount()):
            found.append(head.child(j).text(0))
    assert any("Gaussian" in t for t in found), found
    win.palette.refill("")
    _ok("G2: palette search filters the registry")

    # ── review regressions (Phase-5 impl review, 2026-07-22) ──────────────────
    from nodegraph.graph import Edge, Graph, NodeInstance
    from nodegraph.zones import Zone
    from nodelab_v2.document import GraphDocument
    from nodelab_v2.ops import ensure_ops, headless_engine

    # R1 (BLOCKER): load_dict reusing ids rebinds NodeItems to the NEW records
    #   (no stale op_key/spec/position). Standalone scene so the demo stays intact.
    from nodelab_v2.scene import GraphScene as _GS
    rdoc = GraphDocument()
    rdoc.add_node("channel.select", node_id="n1", x=10, y=10)
    rscene = _GS(rdoc)
    old_n1 = rscene.node_items["n1"]
    assert old_n1.op_key == "channel.select"
    rdoc.load_dict({
        "format_version": "2.0",
        "graph": {"nodes": [{"id": "n1", "op_key": "enhance.gaussian",
                             "params": {}, "modes": {}}], "edges": []},
        "ui": {"nodes": {"n1": {"x": 999.0, "y": 888.0, "muted": False}}},
    })
    new_n1 = rscene.node_items["n1"]
    assert new_n1 is not old_n1, "stale NodeItem kept after reload"
    assert new_n1.rec is rdoc.nodes["n1"] and new_n1.op_key == "enhance.gaussian"
    assert (new_n1.pos().x(), new_n1.pos().y()) == (999.0, 888.0)
    _ok("R1: reload rebinds cards to fresh records (op/spec/pos correct, no stale)")

    # R5 (BLOCKER): a file with a zone round-trips through the document unharmed
    zdoc = GraphDocument()
    zg = {
        "format_version": "2.0",
        "graph": {"nodes": [
            {"id": "s", "op_key": "enhance.gamma", "params": {}, "modes": {}},
            {"id": "ri", "op_key": "zone.repeat_in", "params": {}, "modes": {}},
            {"id": "ro", "op_key": "zone.repeat_out", "params": {}, "modes": {}}],
            "edges": [{"src": "ri", "dst": "ro", "src_socket": "out",
                       "dst_socket": "data", "kind": "forward"},
                      {"src": "ro", "dst": "ri", "src_socket": "out",
                       "dst_socket": "data", "kind": "back"}]},
        "zones": [{"id": "z1", "kind": "repeat", "in_id": "ri", "out_id": "ro",
                   "body": [], "iterations": 3, "impure": False}],
    }
    zdoc.load_dict(zg)
    assert zdoc.has_unedited_structure
    round_tripped = zdoc.to_dict()
    assert len(round_tripped["zones"]) == 1 and round_tripped["zones"][0]["id"] == "z1"
    assert round_tripped["zones"][0]["iterations"] == 3
    # the back-edge survived too
    assert any(e["kind"] == "back" for e in round_tripped["graph"]["edges"])
    _ok("R5: zones + back-edges preserved verbatim across GUI load→save")

    # R6 (MAJOR): a GUI-authored graph runs HEADLESS (no PySide6) via ops.headless_engine
    ensure_ops()
    from nodegraph.provider import SyntheticProvider
    from nodegraph.dataset import Dataset as _DS
    hg = Graph()
    hg.add(NodeInstance("src", "io.load"))
    hg.add(NodeInstance("v", "view.viewer"))
    hg.connect("src", "v")
    sp = SyntheticProvider(AxisSizes(m=1, t=1, z=1, c=1, y=32, x=32), tile=16)
    seed = _DS(axes=sp.axes, metadata={"pixel_size_um": 0.1}).with_image(sp)
    heng = headless_engine(hg, seeds={"src": seed},
                           meta_seeds={"src": MetaEnvelope(axes=sp.axes)})
    out_ds = heng.pull("v")                        # view.viewer compute is in COMPUTES
    assert out_ds.image is not None
    _ok("R6: view.viewer compute is Qt-free → GUI graph runs headless")

    # R2 (MAJOR): re-seed re-announces when a node's resolved source key CHANGES
    #   (the old code announced once per node-id and then froze forever). Use a
    #   THROWAWAY runner + fake keys so poking its cache can't corrupt the live
    #   window's real ("synthetic",) source.
    from nodelab_v2.runner import EngineRunner as _ER
    r = _ER(GraphDocument())
    r._node_source_key["nX"] = ("fake-a",)
    r._providers[("fake-a",)] = (object(), MetaEnvelope())
    assert dict(r._fresh_envs())  # first resolution announces
    assert not r._fresh_envs()    # unchanged key: no re-announce
    r._node_source_key["nX"] = ("fake-b",)                  # user typed a real path
    r._providers[("fake-b",)] = (object(), MetaEnvelope(axes=AxisSizes(z=9)))
    fresh = dict(r._fresh_envs())
    assert "nX" in fresh and fresh["nX"].axes.z == 9, "re-seed did not re-announce"
    _ok("R2: source re-seed re-announces on a changed source key (not once-only)")

    # ── G5 spreadsheet + Point overlay (Phase-6 inspection) ───────────────────
    from nodegraph.dataset import Dataset as _DS2
    from nodegraph.domains import Domain as _Dom
    from nodegraph.provider import ArrayProvider
    from nodelab_v2.spreadsheet import structure_tables
    ax5 = AxisSizes(m=1, t=1, z=3, c=1, y=64, x=64)
    sds = _DS2(axes=ax5, metadata={"pixel_size_um": 0.1}).with_image(
        ArrayProvider(np.zeros((1, 1, 3, 1, 64, 64))))
    # a Point structure (2 spots on z=1) + a Label table
    for name, vals in {"id": [1, 2], "m": [0, 0], "t": [0, 0], "c": [0, 0],
                       "z": [1.0, 1.0], "y": [10.0, 40.0], "x": [20.0, 50.0],
                       "intensity": [7.5, 9.0]}.items():
        sds = sds.with_layer(_Dom.POINT, name, np.array(vals), layer="spots")
    for name, vals in {"id": [1], "m": [0], "t": [0], "c": [0], "z": [0.0],
                       "y": [5.0], "x": [5.0], "area": [42.0]}.items():
        sds = sds.with_layer(_Dom.LABEL, name, np.array(vals), layer="labels")

    tables = structure_tables(sds)
    assert ("point", "spots") in tables and ("label", "labels") in tables
    assert set(tables[("point", "spots")]) >= {"id", "y", "x", "intensity"}
    win.sheet.show_dataset("spots-test", sds)
    # select the point table (combo is sorted, so label comes first)
    pt_idx = next(i for i in range(win.sheet._pick.count())
                  if win.sheet._pick.itemData(i) == ("point", "spots"))
    win.sheet._pick.setCurrentIndex(pt_idx)
    assert win.sheet._table.rowCount() == 2               # 2 points
    hdrs = [win.sheet._table.horizontalHeaderItem(i).text()
            for i in range(win.sheet._table.columnCount())]
    assert hdrs[0] == "id" and "intensity" in hdrs        # coord columns first
    _ok(f"G5: spreadsheet groups {len(tables)} structure tables (point+label), "
        f"coord columns first")

    # the viewer overlays the 2 z==1 points and NONE at z==0
    win.viewer._dataset = sds
    win.viewer._axes = ax5
    win.viewer._plane = np.zeros((64, 64))
    from PySide6.QtGui import QPixmap as _QPix
    win.viewer._base_pix = _QPix(64, 64)
    win.viewer._sliders["z"].setMaximum(1)   # frame controls are sliders now (were spins)
    win.viewer._sliders["z"].setValue(1)
    assert len(win.viewer._points_here()) == 2
    win.viewer._sliders["z"].setValue(0)
    assert len(win.viewer._points_here()) == 0            # points are on z==1 only
    _ok("G5/G4: Point overlay filters to the viewed (m,t,z,c) plane")

    # ── Track-trajectory overlay (Phase-6 refinement) ─────────────────────────
    from nodegraph.structure import TrackMembership as _TM
    ax6 = AxisSizes(m=1, t=3, z=1, c=1, y=64, x=64)
    tds = _DS2(axes=ax6).with_image(ArrayProvider(np.zeros((1, 3, 1, 1, 64, 64))))
    # two point tracks over 3 timepoints (ids globally unique, as detect.spots emits)
    for name, vals in {"id": [1, 2, 3, 4, 5, 6], "m": [0]*6, "t": [0, 0, 1, 1, 2, 2],
                       "c": [0]*6, "z": [0.0]*6, "y": [10., 40., 15., 40., 20., 40.],
                       "x": [10., 10., 15., 25., 20., 40.]}.items():
        tds = tds.with_layer(_Dom.POINT, name, np.array(vals), layer="spots")
    tds = tds.with_structure(_TM(track_id=[1, 1, 1, 2, 2, 2], t=[0, 1, 2, 0, 1, 2],
                                 member_id=[1, 3, 5, 2, 4, 6],
                                 member_domain=_Dom.POINT).to_table(layer="spots"))
    win.viewer._dataset = tds
    win.viewer._axes = ax6
    win.viewer._ref_plane = np.zeros((64, 64))
    win.viewer._base_pix = _QPix(64, 64)
    win.viewer._sliders["m"].setMaximum(0)
    win.viewer._sliders["z"].setMaximum(0); win.viewer._sliders["z"].setValue(0)
    win.viewer._sliders["t"].setMaximum(2); win.viewer._sliders["t"].setValue(1)
    trajs = win.viewer._tracks_here()
    assert len(trajs) == 2, f"expected 2 trajectories, got {len(trajs)}"
    by_id = {tid: (path, cur, ts) for tid, path, cur, ts in trajs}
    assert by_id[1][0] == [(10., 10.), (15., 15.), (20., 20.)], by_id[1][0]  # ordered by t
    assert by_id[1][1] == 1 and by_id[2][1] == 1, "current-t vertex ≠ viewed T"
    assert by_id[1][2] == [0, 1, 2], by_id[1][2]           # per-vertex t (trail modes)
    win.viewer._sliders["t"].setValue(2)
    assert all(cur == 2 for _tid, _p, cur, _ts in win.viewer._tracks_here())  # follows T
    assert win.viewer._track_color(1).name() != win.viewer._track_color(2).name()
    win.viewer._repaint()                                  # draws without raising
    _ok("Track overlay: membership↔position join, ordered by t, current-T vertex tracks T")

    # ── Overlay system overhaul (2026-07-28) ──────────────────────────────────
    from nodelab_v2 import overlays as _OV
    vw = win.viewer

    # O1: one Overlays button replaces the three checkboxes; the popup has a tab per
    # domain, and the reserved ones (voxels/mesh) are visibly inert rather than fake-live.
    assert vw._ovl_btn.text() == "◈ Overlays"
    vw.open_overlay_dialog()
    dlg = vw._ovl_dialog
    assert dlg is not None and dlg.isVisible()
    assert [dlg.tabs.tabText(i) for i in range(dlg.tabs.count())] == \
        ["Points", "Labels", "Tracks", "Voxels", "Mesh"], "overlay tabs"
    assert all(_OV.TAB_BY_KEY[t].implemented for t in ("points", "labels", "tracks"))
    # every tab's control set is fully wired (a spec ⇒ a widget), and a tab the renderer
    # does NOT draw is inert rather than fake-live. Driven off TAB_INFO.implemented so
    # promoting a reserved domain to a real one needs no probe edit.
    reserved = [t for t in _OV.TABS if not _OV.TAB_BY_KEY[t].implemented]
    for tab in _OV.TABS:
        for spec in _OV.FIELDS[tab]:
            assert (tab, spec.key) in dlg._widgets, (tab, spec.key)
            if tab in reserved:
                assert not dlg._rows[(tab, spec.key)][1].isEnabled(), (tab, spec.key)
    for tab in reserved:
        assert not dlg._enables[tab].isEnabled(), tab
    _ok(f"O1: Overlays popup — {dlg.tabs.count()} domain tabs, "
        f"{sum(len(_OV.FIELDS[t]) for t in _OV.TABS)} spec-driven controls, "
        f"reserved tabs inert ({', '.join(reserved) or 'none'})")

    # O2: dependent rows go dead when the mode ignores them (no live-looking control
    # the renderer never reads — the same charter the node catalog holds itself to)
    dlg._set("points", "color_mode", "single")
    assert dlg._rows[("points", "color")][1].isEnabled()
    dlg._set("points", "color_mode", "per_point")
    assert not dlg._rows[("points", "color")][1].isEnabled()
    assert dlg._rows[("points", "sat")][1].isEnabled()
    dlg._set("tracks", "trail", "all")
    assert not dlg._rows[("tracks", "window")][1].isEnabled()
    dlg._set("tracks", "trail", "window")
    assert dlg._rows[("tracks", "window")][1].isEnabled()
    _ok("O2: enable_if dependencies grey out the rows the chosen mode ignores")

    # O3: a settings edit repaints without re-pulling, and toggling a domain off is the
    # old checkbox behaviour through the settings (one source of truth)
    assert vw._view.overlay_cb is not None, "no overlay callback on the image surface"
    rev0 = vw._ovl_rev
    dlg._set("labels", "width", 4.0)
    assert vw._ovl_rev > rev0 and vw.overlays.labels.width == 4.0
    assert vw.overlay_enabled("tracks")
    vw.set_overlay_enabled("tracks", False)
    assert not vw.overlays.tracks.enabled
    assert len(vw._tracks_here()) == 2       # geometry still joinable while hidden
    vw.set_overlay_enabled("tracks", True)
    _ok("O3: a settings edit bumps the overlay revision and repaints (no re-pull)")

    # O4: zoom-invariance — the SAME settings drawn at two different zooms must put the
    # same number of ink pixels on screen (sizes are screen px, not image px). Render the
    # renderer directly through two mappings differing only in scale.
    from PySide6.QtGui import QImage as _QImg, QPainter as _QPnt, QColor as _QCol
    from PySide6.QtCore import QPointF as _QPt

    def _ink(scale, cx, cy):
        img = _QImg(240, 240, _QImg.Format_ARGB32)
        img.fill(_QCol(0, 0, 0))
        lab = np.zeros((24, 24), np.int32)
        lab[6:14, 6:14] = 1
        fr = _OV.OverlayFrame(
            map_pt=lambda x, y: _QPt(120 + (x - cx) * scale, 120 + (y - cy) * scale),
            plane_wh=(24, 24), label_plane=lab,
            points=[_OV.PointMark(10.0, 10.0, 1, 0)])
        st = _OV.OverlaySettings()
        st.labels.style = "outline"       # fills DO scale (a fill is the region)
        st.tracks.enabled = False
        p = _QPnt(img)
        _OV.OverlayRenderer().paint(p, st, fr)
        p.end()
        arr = np.frombuffer(img.constBits(), np.uint8).reshape(240, 240, 4)
        return int(np.count_nonzero(arr[..., :3].max(axis=2) > 40))

    ink_out, ink_in = _ink(6.0, 10.0, 10.0), _ink(24.0, 10.0, 10.0)
    # a 4× zoom on the same region: the outline gets LONGER (more of it is on screen) but
    # never THICKER, and the point glyph is pixel-identical — so the ink cannot grow 16×
    assert ink_in < ink_out * 6, f"overlay ink grew with zoom: {ink_out} → {ink_in}"
    ren_a, ren_b = _OV.OverlayRenderer(), _OV.OverlayRenderer()
    g_a, size_a = ren_a.glyph(_OV.PointsOverlay(), _OV.qcolor("#ffc83c"), 1.0)
    g_b, size_b = ren_b.glyph(_OV.PointsOverlay(), _OV.qcolor("#ffc83c"), 1.0)
    assert (size_a, g_a.size()) == (size_b, g_b.size())    # zoom-independent glyph size
    _ok(f"O4: overlay sizes are SCREEN px — 4× zoom kept the ink bounded "
        f"({ink_out}→{ink_in}, not 16×) and the glyph identical")

    # O5: the requested point look — a golden star with a bright centre pixel and a
    # brightness gradient down each arm
    ren = _OV.OverlayRenderer()
    ps = _OV.PointsOverlay(spread=3, unit_px=9.0)
    pm, gsize = ren.glyph(ps, _OV.qcolor(ps.color), 1.0)
    gim = pm.toImage()
    ctr = int(gsize / 2)
    arm = [_QCol(gim.pixel(ctr + int(k * 9), ctr)).red() for k in range(4)]
    assert arm[0] > arm[1] > arm[2] > arm[3] > 0, f"arm gradient not monotonic: {arm}"
    assert _QCol(gim.pixel(ctr, ctr)).green() > _QCol(gim.pixel(ctr + 9, ctr)).green()
    assert len(_OV._DIRS["star"]) == 8 and len(_OV._DIRS["cross"]) == 4
    _ok(f"O5: golden star — bright centre + gradient arms {arm} over 8 directions")

    # O6: per-item colouring is deterministic, distinct, and shared by labels/points/
    # tracks (so "each label a different colour" and "each track a different colour" are
    # literally the same rule), and a label's outline and fill agree
    cols = [_OV.distinct_color(i).name() for i in range(1, 25)]
    assert len(set(cols)) == 24, "per-item palette collided"
    assert _OV.distinct_color(7).name() == cols[6]         # stable across calls (over T)
    lab2 = np.zeros((8, 8), np.int32)
    lab2[1:4, 1:4] = 3
    fill = ren._fill_image(_OV.LabelsOverlay(fill_opacity=100), lab2)
    assert _QCol(fill.pixel(2, 2)).name() == _OV.distinct_color(3).name(), "fill≠outline hue"
    _ok("O6: one golden-angle palette for labels/points/tracks; fill hue == outline hue")

    # O7: "spread to other tabs" copies by ROLE and reports every move
    st = _OV.OverlaySettings()
    st.labels.opacity = 55
    st.labels.width = 3.5
    st.labels.color_mode = "single"
    notes = _OV.spread_settings(st, "labels")
    assert st.tracks.opacity == 55 and st.points.opacity == 55
    assert st.tracks.width == 3.5 and st.points.thickness == 3.5   # role, not key name
    assert st.tracks.color_mode == "single" and st.points.color_mode == "single"
    assert st.labels.style == "both" and st.tracks.trail == "all"   # domain-only untouched
    assert notes and all(":" in n for n in notes)
    assert not _OV.spread_settings(st, "labels")                    # idempotent
    _ok(f"O7: spread copied {len(notes)} role-matched settings, left domain-only alone")

    # O8: persistence — round-trip a file, merge a PARTIAL dict, survive a bad one
    ovl_dir = tempfile.mkdtemp(prefix="nd2ovl_")
    ovl_path = os.path.join(ovl_dir, f"look{_OV.FILE_SUFFIX}")
    st.points.shape = "ring"
    st.points.spread = 5
    _OV.write_json(pathlib.Path(ovl_path), st)
    back = _OV.OverlaySettings()
    back.update_from_dict(_OV.read_json(pathlib.Path(ovl_path)))
    assert back.to_dict() == st.to_dict(), "overlay settings did not round-trip"
    part = _OV.OverlaySettings()
    changed = part.update_from_dict({"overlays": {"labels": {"width": 7.0,
                                                            "bogus_key": 1}}})
    assert changed == ["labels.width"] and part.labels.width == 7.0
    assert part.points.shape == "star"          # a partial file leaves the rest alone
    env_path = os.path.join(ovl_dir, "env.json")
    _OV.write_json(pathlib.Path(env_path), st)
    os.environ[_OV.ENV_VAR] = env_path
    try:
        loaded, sources = _OV.load_defaults()
        assert loaded.points.shape == "ring" and sources and "env.json" in sources[0]
    finally:
        os.environ.pop(_OV.ENV_VAR, None)
    _ok(f"O8: settings round-trip, partial merge ({changed}), and "
        f"{_OV.ENV_VAR} override load")

    # O9: the dialog's Load/Save/default buttons write real files
    dlg._settings.update_from_dict(st.to_dict())
    dlg._settings.labels.width = 6.5            # a marker the points-tab reset must keep
    dlg.reload()
    user_path = os.path.join(ovl_dir, "user.json")
    dlg._write(pathlib.Path(user_path), "test")
    assert os.path.isfile(user_path)
    assert dlg.current_tab() == "points" and dlg._settings.points.shape == "ring"
    dlg._reset_tab()                            # current tab (points) back to built-in
    assert dlg._settings.points.shape == "star", "reset tab did not restore defaults"
    assert dlg._settings.labels.width == 6.5, "reset tab touched another tab"
    dlg._reset_all()
    assert dlg._settings.to_dict() == _OV.OverlaySettings().to_dict()
    dlg.close()
    _ok("O9: dialog save / reset-tab / reset-all act on the live settings")

    # O10: the MESH overlay (V2.08) — a 3-D surface has no single 2-D picture, so it is
    # drawn as its CROSS-SECTION at the viewed Z. Two 8-voxel cubes must each yield one
    # closed loop with exactly the cube's extent on a mid-plane, nothing at all off the
    # mesh, and the vertex style must be empty at a cube's mid-plane (no vertices there)
    # yet populated on a corner plane.
    import nodegraph.mesh as _MSH
    from nodegraph.dataset import Dataset as _MDS
    from nodegraph.provider import ArrayProvider as _MAP
    from PySide6.QtCore import QRectF as _QRect

    def _cube_mesh(z0, y0, x0, s):
        V = np.array([[z0 + dz, y0 + dy, x0 + dx]
                      for dz in (0., s) for dy in (0., s) for dx in (0., s)], float)
        quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1),
                 (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
        Fs = []
        for a, b, c, d in quads:
            Fs += [[a, b, c], [a, c, d]]
        return V, np.array(Fs, np.int64)

    _mels = []
    for _i, (_yo, _xo) in enumerate([(6., 6.), (6., 26.)]):
        _V, _F = _cube_mesh(4., _yo, _xo, 8.)
        _mels.append(_MSH.MeshElement(m=0, t=0, c=0, src_label=_i, verts_zyx=_V, faces=_F,
                                      centroid_zyx=(8., _yo + 4., _xo + 4.),
                                      volume_um3=1., surface_area_um2=1., density=1.,
                                      n_points=8))
    _mtb = _MSH.build_mesh_tables(_mels, layer="mesh")
    _max = AxisSizes(m=1, t=1, z=16, c=1, y=48, x=48)
    _mds = _MSH.with_mesh(
        _MDS(axes=_max, metadata={"pixel_size_um": 0.2, "z_step_um": 0.5})
        .with_image(_MAP(np.zeros((1, 1, 16, 1, 48, 48), np.uint16))),
        _mtb, provenance={"boundary": "convex_hull"})
    assert _OV.TAB_BY_KEY["mesh"].implemented, "the mesh tab must be live now"
    vw.show_result("MESH", {0: np.zeros((48, 48), np.uint16)}, _max, 0.01, dataset=_mds)
    assert vw.overlay_enabled("mesh")
    vw._sliders["z"].setValue(8)
    vw._geo_key = None
    vw._ensure_geometry()
    _secs = vw._geo_mesh
    assert len(_secs) == 2, f"expected 2 cross-sections, got {len(_secs)}"
    assert [s.object_id for s in _secs] == [1, 2]
    _ext = []
    for _s in _secs:
        assert all(_s.closed), "a cube's cross-section must close"
        _pts = np.array(_s.loops[0])
        _ext.append((_pts[:, 0].min(), _pts[:, 0].max(),
                     _pts[:, 1].min(), _pts[:, 1].max()))
    assert _ext[0] == (6.0, 14.0, 6.0, 14.0), _ext[0]
    assert _ext[1] == (6.0, 14.0, 26.0, 34.0), _ext[1]
    vw._sliders["z"].setValue(0)                     # off the mesh entirely
    vw._geo_key = None
    vw._ensure_geometry()
    assert not vw._geo_mesh, "the mesh must vanish on a plane it does not cross"
    # every style must put real ink on screen. Painted through an EXPLICIT mapping (as O4
    # does) rather than the live widget transform, so the assertion is about the renderer
    # and not about where the panel happens to be scrolled.
    def _mesh_ink(style, zv):
        vw.overlays.mesh.style = style
        vw._sliders["z"].setValue(zv)
        vw._geo_key = None
        vw._ensure_geometry()
        secs = vw._geo_mesh
        im = _QImg(200, 200, _QImg.Format_ARGB32)
        im.fill(_QCol(0, 0, 0))
        fr = _OV.OverlayFrame(
            map_pt=lambda x, y: _QPt(x * 4.0, y * 4.0), plane_wh=(48, 48), mesh=secs)
        st_m = _OV.OverlaySettings()
        st_m.points.enabled = st_m.labels.enabled = st_m.tracks.enabled = False
        st_m.mesh = vw.overlays.mesh
        p = _QPnt(im)
        _OV.OverlayRenderer().paint(p, st_m, fr)
        p.end()
        arr = np.frombuffer(im.constBits(), np.uint8).reshape(200, 200, 4)
        return len(secs), int(np.count_nonzero(arr[..., :3].max(axis=2) > 40))

    _fill_ink = None
    for _style, _zv, _want in (("wireframe", 8, 2), ("surface", 8, 2),
                               ("points", 8, 0), ("points", 4, 2)):
        _n, _ink = _mesh_ink(_style, _zv)
        assert _n == _want, (_style, _zv, _n)
        if _want:
            assert _ink > 0, f"mesh style {_style} drew nothing"
        if _style == "surface":
            _fill_ink = _ink
    # a filled cross-section must cover strictly more than its outline alone
    _, _outline_ink = _mesh_ink("wireframe", 8)
    assert _fill_ink > _outline_ink, (_fill_ink, _outline_ink)
    vw.overlays.mesh.style = "wireframe"
    _pv = _QImg(240, 160, _QImg.Format_ARGB32)
    _pp = _QPnt(_pv)
    _OV.render_preview(_pp, _QRect(0, 0, 240, 160), vw.overlays, "mesh")
    _pp.end()
    vw.set_overlay_enabled("mesh", False)
    vw._geo_key = None
    vw._ensure_geometry()
    assert not vw._geo_mesh, "a disabled overlay must extract no geometry"
    vw.set_overlay_enabled("mesh", True)
    _ok("O10: MESH overlay draws the Z cross-section (2 closed loops at the exact cube "
        "extents, empty off-plane), all 3 styles paint + preview, on/off honoured")

    # ── Phase-remainder features (2026-07-22) ─────────────────────────────────
    from nodelab_v2.export import export_dataset
    from nodelab_v2.document import GraphDocument as _GD

    # E1: CSV export of the crafted point+label Dataset
    csv_path = os.path.join(tempfile.mkdtemp(prefix="nd2exp_"), "t.csv")
    n_rows = export_dataset(sds, csv_path)
    with open(csv_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    assert n_rows == 3 and lines[0].startswith("domain,layer,")   # 2 points + 1 label
    assert any("point" in ln for ln in lines) and any("label" in ln for ln in lines)
    _ok(f"Export: CSV wrote {n_rows} rows (long form, domain/layer columns)")

    # E1b: Parquet export (pyarrow present)
    try:
        pq_path = os.path.join(os.path.dirname(csv_path), "t.parquet")
        export_dataset(sds, pq_path)
        import pyarrow.parquet as _pq
        assert _pq.read_table(pq_path).num_rows == 3
        _ok("Export: Parquet round-trips (pyarrow)")
    except ImportError:
        _ok("Export: Parquet SKIPPED (pyarrow absent)")

    # E2: splice-on-wire inserts a node into an existing link
    sdoc = _GD()
    sdoc.add_node("io.load", node_id="a", x=0, y=0)
    sdoc.add_node("enhance.gaussian", node_id="b", x=200, y=0)
    sdoc.connect("a", "image", "b", "data")
    ss = _GS(sdoc)
    ok_sp = ss.splice_onto(sdoc.add_node("enhance.gamma", node_id="g", x=100, y=0).id,
                           ("a", "image", "b", "data"))
    assert ok_sp
    assert sdoc.edge_into("b", "data") == ("g", "out", "b", "data")
    assert ("a", "image", "g", "data") in sdoc.edges         # a → g → b
    assert ("a", "image", "b", "data") not in sdoc.edges     # original edge replaced
    # a Dataset node spliced onto a VALUE-socket wire must be REFUSED without dropping
    # the wire (the second connect, into a value input, is invalid). Construct the
    # value wire directly (the catalog has no value-output node to form one naturally).
    sdoc.edges.append(("a", "image", "b", "sigma"))          # dataset-out → value-in
    spliced = ss.splice_onto(
        sdoc.add_node("enhance.gamma", node_id="g2", x=50, y=200).id,
        ("a", "image", "b", "sigma"))
    assert not spliced and ("a", "image", "b", "sigma") in sdoc.edges   # not dropped
    _ok("Splice: inserts on Dataset wires; refuses (no dropped wire) on a value wire")

    # E3: collapse toggles a compact layout (fewer socket rows, shorter card)
    gi = win.scene.node_items["n3"]
    tall = gi._height
    win.doc.set_collapsed("n3", True)
    assert win.scene.node_items["n3"]._height < tall
    assert win.scene.node_items["n3"].rec.collapsed
    win.doc.set_collapsed("n3", False)
    assert win.scene.node_items["n3"]._height == tall
    _ok("Collapse: compact layout toggles + relayouts (C key path)")

    # E4: light theme rebinds tokens + restyles without error
    dark_bg = T.BG.name()
    win.set_theme("light")
    assert T.MODE == "light" and T.BG.name() != dark_bg
    assert T.INK.lightness() < T.BG.lightness()      # dark ink on light bg
    win.set_theme("dark")
    assert T.MODE == "dark" and T.BG.name() == dark_bg
    _ok("G9: light/dark theme toggle rebinds palette + restyles panels")

    # E5: Repeat-zone creation wraps a linear chain (validated via unroll)
    zdoc2 = _GD()
    zdoc2.add_node("io.load", node_id="src", x=0, y=0)
    zdoc2.add_node("enhance.gamma", node_id="body", x=200, y=0)
    zdoc2.add_node("analysis.threshold", node_id="sink", x=400, y=0)
    zdoc2.connect("src", "image", "body", "data")
    zdoc2.connect("body", "out", "sink", "data")
    zid = zdoc2.wrap_repeat_zone(["body"], iterations=4)
    assert zid and len(zdoc2._zones) == 1 and zdoc2._zones[0].iterations == 4
    assert zdoc2._zones[0].kind == "repeat"
    assert any(op == "zone.repeat_in" for op in
               (r.op_key for r in zdoc2.nodes.values()))
    assert len(zdoc2._back_edges) == 1                # the Out→In feedback edge
    from nodegraph.zones import unroll as _unr
    _unr(zdoc2.to_graph(for_run=False), zdoc2._zones)  # unrolls cleanly
    # a re-save round-trips the zone + back-edge
    assert zdoc2.has_unedited_structure
    rt = zdoc2.to_dict()
    assert len(rt["zones"]) == 1
    # ambiguous selection refuses cleanly (two outputs)
    try:
        zdoc2.wrap_repeat_zone(["src", "body", "sink"])
        raise SystemExit("wrap should have refused a node already in a zone")
    except ValueError:
        pass
    _ok("Zones: Repeat-zone wrap (validated via unroll, round-trips, refuses bad sel)")

    # E6: labelled frames (GUI-only canvas grouping) — create, enclose, follow, persist
    from nodelab_v2.frame_item import FrameItem as _FI
    fdoc = _GD()
    fsc = _GS(fdoc)
    fdoc.add_node("enhance.gamma", node_id="fa", x=100, y=100)
    fdoc.add_node("enhance.gamma", node_id="fb", x=400, y=250)
    fdoc.add_node("enhance.gamma", node_id="fc", x=900, y=100)
    frec = fdoc.add_frame("Preprocess", ["fa", "fb"])
    fit = fsc.frame_items[frec.id]
    assert isinstance(fit, _FI) and fit.zValue() < 0        # behind the nodes
    fr_rect = fit.mapToScene(fit.boundingRect()).boundingRect()
    for nid in ("fa", "fb"):
        ni = fsc.node_items[nid]
        assert fr_rect.contains(ni.mapToScene(ni.boundingRect()).boundingRect())
    nc = fsc.node_items["fc"]
    assert not fr_rect.contains(nc.mapToScene(nc.boundingRect()).boundingRect())
    w0 = fit.mapToScene(fit.boundingRect()).boundingRect().width()
    fsc.node_items["fb"].setPos(700, 520)                   # move a member
    assert fit.mapToScene(fit.boundingRect()).boundingRect().width() > w0  # frame follows
    # save/load round-trip via the ui extras
    fd = fdoc.to_dict()
    assert frec.id in fd["ui"]["frames"] and fd["ui"]["frames"][frec.id]["title"] == "Preprocess"
    fdoc2 = _GD(); fdoc2.load_dict(fd)
    assert fdoc2.frames[frec.id].members == ["fa", "fb"]
    # deleting a member prunes; emptying auto-removes; deleting a frame keeps its nodes
    fdoc.remove_node("fa")
    assert fdoc.frames[frec.id].members == ["fb"]
    fdoc.remove_node("fb")
    assert frec.id not in fdoc.frames and frec.id not in fsc.frame_items
    keep = fdoc.add_frame("Keep", ["fc"])
    fdoc.remove_frame(keep.id)
    assert keep.id not in fdoc.frames and "fc" in fdoc.nodes
    _ok("Frames: create/enclose/follow-move, save-load, prune, delete keeps nodes")

    # E7: reroute — hidden pass-through node, spliced into a wire, renders compact
    rrdoc = _GD()
    rrsc = _GS(rrdoc)
    rrdoc.add_node("enhance.gamma", node_id="ra", x=0, y=0)
    rrdoc.add_node("enhance.gamma", node_id="rb", x=400, y=0)
    rrdoc.connect("ra", "out", "rb", "data")
    from nodelab_v2.scene import visible_specs as _visible_specs
    assert not any(s.op_key == "rr.reroute" for s in _visible_specs())  # hidden from palette
    rr = rrdoc.add_node("rr.reroute", x=200, y=0)
    assert rrsc.splice_onto(rr.id, ("ra", "out", "rb", "data"))         # a → reroute → b
    assert ("ra", "out", rr.id, "data") in rrdoc.edges
    assert rrdoc.edge_into("rb", "data") == (rr.id, "out", "rb", "data")
    rit = rrsc.node_items[rr.id]
    # card_rect is the geometry; boundingRect adds the glow repaint margin on top
    assert rit._is_reroute and rit.card_rect().width() == T.RR_SIZE      # compact dot
    assert rit.boundingRect().width() == T.RR_SIZE + 2 * rit.GLOW_M
    assert rit.socket("in", "data") is not None and rit.socket("out", "out") is not None
    _ok("Reroute: hidden pass-through, splices into a Dataset wire, renders compact")

    # E8: group creation — collapse a linear sub-chain into a reusable group instance
    from nodegraph.groups import group_name_of as _gname
    gdoc = _GD()
    gsc = _GS(gdoc)
    gdoc.add_node("io.load", node_id="gs", x=0, y=0)
    gdoc.add_node("enhance.gamma", node_id="ga", x=200, y=0)
    gdoc.add_node("enhance.gaussian", node_id="gb", x=400, y=0)
    gdoc.add_node("analysis.threshold", node_id="gc", x=600, y=0)
    gdoc.add_node("analysis.label", node_id="gd", x=800, y=0)
    gdoc.connect("gs", "image", "ga", "data")
    gdoc.connect("ga", "out", "gb", "data")
    gdoc.connect("gb", "out", "gc", "data")
    gdoc.connect("gc", "out", "gd", "data")
    ginst = gdoc.make_group(["ga", "gb", "gc"], name="Preprocess")
    assert gdoc.nodes[ginst].op_key == "group:Preprocess"
    assert {"ga", "gb", "gc"}.isdisjoint(gdoc.nodes)         # members left the parent
    assert ("gs", "image", ginst, "data") in gdoc.edges and (ginst, "out", "gd", "data") in gdoc.edges
    gitem = gsc.node_items[ginst]                            # renders as a group card
    assert gitem._is_group and gitem.socket("in", "data") and gitem.socket("out", "out")
    # the run/propagate graph inlines the body (no residual group:* nodes)
    gexp = gdoc.to_graph(for_run=True, materialize=True)
    assert not any(_gname(n.op_key) for n in gexp.nodes.values())
    assert any(n.op_key == "enhance.gaussian" for n in gexp.nodes.values())
    # save/load round-trips the instance + definition (groups are GUI-manageable now)
    gd_dict = gdoc.to_dict()
    assert len(gd_dict["groups"]) == 1 and not gdoc.has_unedited_structure
    _GD().load_dict(gd_dict)                                 # loads without error
    # ungroup restores the interior + reconnects the frontier
    assert gdoc.ungroup(ginst) and ginst not in gdoc.nodes and not gdoc._groups
    assert gdoc.edge_into("gd", "data") is not None          # chain reconnected to sink
    assert sum(1 for n in gdoc.nodes.values() if n.op_key == "enhance.gaussian") == 1
    # guardrail: grouping a source (buries its seed) is refused
    try:
        gdoc.make_group(["gs"], "X"); raise SystemExit("grouping a source should raise")
    except ValueError:
        pass
    _ok("Groups: make_group→instance, run-expand, save-load, ungroup, source guardrail")

    # E8b: deleting a node — all three affordances, plus dissolve (delete + heal) ──
    ddoc = _GD()
    dsc = _GS(ddoc)
    for i, op in enumerate(("io.load", "enhance.gamma", "enhance.gaussian",
                            "analysis.threshold")):
        ddoc.add_node(op, node_id=f"d{i}", x=220 * i, y=0)
    ddoc.connect("d0", "image", "d1", "data")
    ddoc.connect("d1", "out", "d2", "data")
    ddoc.connect("d2", "out", "d3", "data")
    deleted: list = []
    dsc.nodes_deleted.connect(lambda ids: deleted.extend(ids))
    # (1) the hover ✕ badge on the card
    ditem = dsc.node_items["d3"]
    assert not ditem._close.isVisible()                  # hidden until hovered
    from PySide6.QtCore import QEvent as _QEvent, Qt
    from PySide6.QtGui import QKeyEvent
    from PySide6.QtWidgets import QGraphicsSceneHoverEvent as _Hov
    _h = _Hov(_QEvent.GraphicsSceneHoverEnter); _h.setPos(QPointF(5, 5))
    ditem.hoverEnterEvent(_h)
    assert ditem._close.isVisible()                      # …then it offers itself
    ditem._close.clicked.emit()
    assert "d3" not in ddoc.nodes and deleted == ["d3"]
    # (2) Delete key through the scene (what the canvas keyboard does)
    dsc.clearSelection()
    dsc.node_items["d2"].setSelected(True)
    dsc.keyPressEvent(QKeyEvent(_QEvent.KeyPress, Qt.Key_Delete, Qt.NoModifier))
    assert "d2" not in ddoc.nodes
    # (3) the right-click context menu (built, then its Delete action triggered)
    from PySide6.QtWidgets import QMenu as _QMenu
    dsc.clearSelection()
    dmenu = _QMenu()
    dsc._fill_node_menu(dmenu, dsc.node_items["d1"])
    texts = [a.text() for a in dmenu.actions()]
    assert any(t.startswith("Delete node") for t in texts), texts
    assert any("Dissolve" in t for t in texts), texts
    next(a for a in dmenu.actions() if a.text().startswith("Delete node")).trigger()
    assert "d1" not in ddoc.nodes and ddoc.nodes and set(ddoc.nodes) == {"d0"}
    # (4) dissolve heals the chain: src → [mid] → dst becomes src → dst
    hdoc = _GD()
    hsc = _GS(hdoc)
    hdoc.add_node("io.load", node_id="h0", x=0, y=0)
    hdoc.add_node("enhance.gamma", node_id="h1", x=200, y=0)
    hdoc.add_node("analysis.threshold", node_id="h2", x=400, y=0)
    hdoc.add_node("analysis.label", node_id="h3", x=600, y=0)
    hdoc.connect("h0", "image", "h1", "data")
    hdoc.connect("h1", "out", "h2", "data")
    hdoc.connect("h2", "out", "h3", "data")
    assert hsc.dissolve_node("h1")
    assert "h1" not in hdoc.nodes
    assert hdoc.edge_into("h2", "data") == ("h0", "image", "h2", "data")   # healed
    assert hsc.dissolve_node("h0")                        # a SOURCE just leaves a gap
    assert hdoc.edge_into("h2", "data") is None and "h2" in hdoc.nodes
    # the window's Edit action works no matter where the keyboard focus is
    win.viewer.setFocus()
    win.scene.clearSelection()
    win.scene.node_items["n6"].setSelected(True)
    win._sync_edit_actions()
    assert win._del_act.isEnabled() and win._dissolve_act.isEnabled()
    win._del_act.trigger()
    assert "n6" not in win.doc.nodes
    win.scene.clearSelection()
    win._sync_edit_actions()
    assert not win._del_act.isEnabled()                   # nothing selected → greyed out
    win.build_demo()                                      # restore the example graph
    app.processEvents()
    _ok("Delete: hover ✕ badge, Del key, context menu, Edit action (focus-proof); "
        "dissolve deletes a mid-chain node and reconnects the wire through it")

    # E8c: per-node progress — plan → queued → running/cached → done, on the cards ──
    events: list = []
    win.runner.node_progress.connect(
        lambda ev, nid, info: events.append((ev, nid, info.get("fraction"))))
    plans: list = []
    win.runner.plan.connect(lambda t, ids: plans.append((t, sorted(ids))))
    # edit n4's param first, so THIS node is guaranteed to recompute while its upstream
    # is served from the session's persistent memo (earlier probes already pulled it) —
    # which is exactly the mixed recompute/cached picture the cards must show.
    win.doc.nodes["n4"].params["threshold"] = 0.37
    win.doc.touch()
    app.processEvents()
    win.pull_node("n4")                       # Load → Select → Gaussian → Threshold
    t0 = time.time()
    while win.runner._busy and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert plans and plans[-1][0] == "n4"
    assert set(plans[-1][1]) == {"n1", "n2", "n3", "n4"}, plans[-1]
    kinds = [(ev, nid) for ev, nid, _f in events]
    assert ("start", "n4") in kinds and ("done", "n4") in kinds, kinds
    n3_last = max(i for i, k in enumerate(kinds) if k[1] == "n3")
    assert n3_last < kinds.index(("start", "n4")), kinds   # upstream settles first
    assert kinds[n3_last][0] in ("done", "cached"), kinds[n3_last]
    # analysis.threshold is EAGER (per-plane) so it reports real fractions; the last one
    # always lands on 1.0 (the runner never throttles the final update)
    fr = [f for ev, nid, f in events if ev == "progress" and nid == "n4"]
    assert fr and fr[-1] == 1.0 and all(0.0 <= f <= 1.0 for f in fr), fr
    items = win.scene.node_items
    assert items["n4"].run_state() == "done" and items["n4"]._run_text().endswith(
        ("ms", "s")), items["n4"]._run_text()
    # the card itself stays graphic (header rail + status dot) — the wall time rides in
    # its tooltip, so the numbers are one hover away instead of printed on the canvas
    assert items["n4"].toolTip().endswith(("ms", "s")), items["n4"].toolTip()
    assert "n4" in items["n4"].toolTip()
    assert items["n3"].run_state() in ("done", "cached")
    assert items["n7"].run_state() == ""          # not in this pull → no stale state
    assert not items["n7"].toolTip()
    assert not any(i.run_state() in ("queued", "running", "decoding")
                   for i in items.values())      # everything settled
    assert not any(e.flow for e in win.scene.edge_items)   # nothing in flight → no flow
    # a re-pull of the same graph is all memo hits → 'cached' cards, no recompute
    events.clear()
    win.pull_node("n4")
    t0 = time.time()
    while win.runner._busy and time.time() - t0 < 180:
        app.processEvents()
        time.sleep(0.01)
    app.processEvents()
    assert any(ev == "cached" for ev, _n, _f in events), events
    assert items["n3"].run_state() == "cached"
    # ONE shared scene timer animates both the working cards (pulsing dot / sweeping
    # rail) and the flowing wires — and only while something needs animating
    assert not win.scene._anim.isActive()
    win.scene._set_state("n3", "running")         # no fraction → indeterminate sweep
    assert items["n3"].is_running() and win.scene._anim.isActive()
    flowing = [e.model_edge for e in win.scene.edge_items if e.flow]
    assert flowing, "a pull in flight must flow the wires out of produced nodes"
    assert all(e[0] in win.scene._run for e in flowing), flowing
    ph, eph = items["n3"]._phase, win.scene.edge_items[0]._phase
    win.scene._tick_progress()
    assert items["n3"]._phase != ph
    assert any(e._phase != eph for e in win.scene.edge_items if e.flow)
    win.scene.clear_run_states()
    assert not win.scene._anim.isActive() and items["n3"].run_state() == ""
    assert not any(e.flow for e in win.scene.edge_items)
    # the status-bar LED is the footer twin of the card dot: pulses busy, settles idle
    assert win._led_state == "idle" and not win._led_timer.isActive()
    win._set_led("busy")
    assert win._led_timer.isActive() and win._led_on
    win._led_tick()
    assert not win._led_on                        # off-beat of the pulse
    win._set_led("idle")
    assert not win._led_timer.isActive() and win._led_on
    _ok("Progress: plan→queued, per-node start/done/cached (rail + dot on the card, wall "
        "time in its tooltip), eager fractions, flowing wires, one shared timer")

    # E9: maximized canvas + mini-map Viewer + click-to-preview ────────────────
    docked_sizes = win._center.sizes()
    assert win._center.count() == 2 and win._center.widget(0) is win.viewer
    win.set_maximized(True)
    for _ in range(3):
        app.processEvents()                            # splitter re-layout + reposition
    # the SAME viewer widget moved into the overlay (not a copy) and left the splitter
    assert win.viewer.parent() is win.minimap and win.minimap.content is win.viewer
    assert win._center.count() == 1 and win._center.widget(0) is win.view
    assert win.minimap.isVisible() and win.minimap.parent() is win.view
    assert win.view.is_maximized() and win._max_act.isChecked()
    # pinned to the canvas' TOP-LEFT corner, inside it
    mg = win.minimap.geometry()
    assert mg.left() < win.view.width() / 2 and mg.top() < win.view.height() / 2
    assert mg.right() < win.view.width() and mg.bottom() < win.view.height()
    # compact layout: LUT + fps controls give way to the image; the panel can shrink
    assert win.viewer.compact
    assert not any(h.isVisible() for h in win.viewer._hists.values())
    assert not win.viewer._fps_spins["t"].isVisible()
    assert win.viewer._ovl_btn.text() == "◈"           # glyph only while compact
    assert win.viewer.minimumSizeHint().width() <= win.minimap.MIN_W
    _ok(f"Maximize: canvas owns the centre; Viewer re-homed into a "
        f"{mg.width()}×{mg.height()} mini-map at ({mg.left()},{mg.top()})")

    # click-to-preview: selecting a node (what a click does) pulls it into the mini-map
    assert win._follow_act.isChecked()                 # forced on while maximized
    pulled = []
    win.runner.started.connect(lambda nid: pulled.append(nid))
    win.scene.clearSelection()
    win.scene.node_items["n5"].setSelected(True)       # "click" the label node
    t0 = time.time()
    while "n5" not in pulled and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    assert "n5" in pulled, f"click did not preview the node (pulled={pulled})"
    assert win._viewed == "n5" and win.scene.viewed_id == "n5"
    assert win.scene.node_items["n5"]._viewed          # accent spine marks the card
    assert not win.scene.node_items["n3"]._viewed
    assert "n5" in win.minimap._title
    t0 = time.time()
    while win.minimap.state == "busy" and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    assert win.minimap.state == "live", win.minimap.state
    _ok("Mini-map: a node click pulls it live (debounced), card + header follow")

    # a marquee across several nodes queues ONE pull (the debounce), not one per node
    n_before = len(pulled)
    win.scene.clearSelection()
    for nid in ("n2", "n3", "n4"):
        win.scene.node_items[nid].setSelected(True)
        app.processEvents()
    t0 = time.time()
    while len(pulled) == n_before and time.time() - t0 < 60:
        app.processEvents()
        time.sleep(0.005)
    time.sleep(0.3)
    app.processEvents()
    assert len(pulled) - n_before == 1, f"debounce queued {len(pulled)-n_before} pulls"
    assert pulled[-1] in ("n2", "n3", "n4"), pulled[-1]
    _ok(f"Mini-map: a multi-node selection debounces to a single pull ({pulled[-1]})")

    # the mini-map moves/resizes and re-anchors, then docks back unharmed
    win.minimap.set_frame_size(300, 240)
    win.minimap.move(24, 20)
    win.minimap._reanchor()
    win.view.resize(win.view.width() - 120, win.view.height())
    app.processEvents()
    assert win.minimap.geometry().topLeft().x() == 24  # top-left anchor survives resize
    assert (win.minimap.width(), win.minimap.height()) == (300, 240)
    win.set_maximized(False)
    app.processEvents()
    assert win._center.count() == 2 and win._center.widget(0) is win.viewer
    assert win.viewer.isVisible() and not win.minimap.isVisible()
    assert not win.viewer.compact
    assert all(h.isVisible() for h in win.viewer._hists.values())
    assert win.viewer._ovl_btn.text() == "◈ Overlays"
    assert not win.view.is_maximized() and not win._max_act.isChecked()
    assert win._center.sizes() == docked_sizes, (win._center.sizes(), docked_sizes)
    _ok("Restore: Viewer docks back at its old split size, full controls returned")

    # ── G10 + screenshots (dark + light + maximized) ─────────────────────────
    app.processEvents()
    win.view.fit_all()
    app.processEvents()
    assert win.grab().save(out)
    _ok(f"screenshot (dark) {out}")

    win.set_maximized(True)
    win.pull_node("n3")
    t0 = time.time()
    while win.minimap.state == "busy" and time.time() - t0 < 120:
        app.processEvents()
        time.sleep(0.01)
    win.view.fit_all()
    for _ in range(3):
        app.processEvents()
    max_out = out.replace(".png", "_maximized.png")
    assert win.grab().save(max_out)
    _ok(f"screenshot (maximized + mini-map) {max_out}")
    win.set_maximized(False)
    app.processEvents()
    win.set_theme("light")
    for _ in range(2):
        app.processEvents()
    light_out = out.replace(".png", "_light.png")
    assert win.grab().save(light_out)
    _ok(f"screenshot (light) {light_out}")

    for _ in range(4):
        app.processEvents()
    assert not _seen_fail, f"unexpected runner failures: {_seen_fail}"
    _ok("no spurious runner failures across the session")

    # ── P1: the layer picker (V2.11) ─────────────────────────────────────────
    # A source-layer socket offers the layers actually present on the incoming edge
    # instead of making the user retype a name. Editable, not a closed list: a couple of
    # producers name layers the edit-time pass cannot predict.
    from PySide6.QtCore import QPoint as _QPoint
    from PySide6.QtGui import QWheelEvent
    from nodelab_v2.document import GraphDocument as _PDoc
    from nodelab_v2.inspector import InspectorPanel as _PInsp, _NoWheelCombo
    from nodelab_v2.node_item import NodeItem as _PItem

    pdoc = _PDoc()
    pdoc.add_node("io.load", node_id="PS")
    pdoc.meta_seeds["PS"] = MetaEnvelope(axes=AxisSizes(m=1, t=1, z=1, c=1, y=16, x=16))
    pdoc.add_node("analysis.threshold", node_id="PT", params={"name": "m2"})
    pdoc.add_node("analysis.label", node_id="PL", params={"name": "regions"})
    # V2.12: `analysis.watershed` is now the `watershed` METHOD of analysis.segment, and
    # its `mask` socket is gated on that method — so the mode must be set for the picker to
    # exist at all, which is itself the check that `available_in` gating reaches the form.
    pdoc.add_node("analysis.segment", node_id="PW", modes={"method": "watershed"})
    pdoc.connect("PS", "image", "PT", "data")
    pdoc.connect("PT", "out", "PL", "data")
    pdoc.connect("PL", "out", "PW", "data")

    assert pdoc.layer_choices("PW", pdoc.nodes["PW"].spec().input("mask")) \
        == ["m2", "regions"], "picker must offer the upstream Voxel layers"
    assert "labels" not in pdoc.layer_choices(
        "PW", pdoc.nodes["PW"].spec().input("mask")), "never offer a node its own output"
    # the method gates BOTH halves of the form: the `mask` socket and the `level` Mode
    _prec, _pspec = pdoc.nodes["PW"], pdoc.nodes["PW"].spec()
    _sock_names = lambda: {s.name for s in pdoc.input_specs("PW")}
    _mode_names = lambda: {m.name for m in _pspec.active_modes(_prec.state())}
    assert "mask" in _sock_names() and "level" in _mode_names()
    _prec.modes["method"] = "cellsam"
    assert "mask" not in _sock_names(), \
        "a socket the chosen method never reads must vanish from the form"
    assert "level" not in _mode_names(), \
        "and so must a MODE the chosen method never reads (V2.12 ModeSpec.available_in)"
    _prec.modes["method"] = "watershed"
    assert "mask" in _sock_names() and "level" in _mode_names(), "gating is reversible"

    pinsp = _PInsp()
    pitem = _PItem(pdoc.nodes["PW"], pdoc)
    pinsp.set_node(pitem)
    _pick = next((c for c in pinsp.findChildren(_NoWheelCombo) if c.isEditable()
                  and [c.itemText(i) for i in range(c.count())] == ["m2", "regions"]), None)
    assert _pick is not None, "no populated layer picker in the inspector"
    # The Segmentation node's foreground socket defaults to EMPTY — unset means "segment
    # the image", and naming a layer means "split THAT foreground instead".
    assert _pick.currentText() == "", "picker starts at the socket default"
    _pick.setCurrentText("regions")
    _pick.activated.emit(_pick.findText("regions"))
    assert pdoc.nodes["PW"].params.get("mask") == "regions", "picking commits"
    _pick.setEditText("typed_by_hand")
    _pick.lineEdit().editingFinished.emit()
    assert pdoc.nodes["PW"].params.get("mask") == "typed_by_hand", \
        "free text must still commit — an unpredictable layer name is never blocked"

    # Wheel safety. A plain QComboBox CHANGES VALUE and eats the event on an unfocused
    # wheel notch (measured on PySide6 6.10.2), and the inspector is a fixed-height
    # scroll area, so scrolling past a combo used to silently rewrite and pin a param.
    pdoc.nodes["PW"].params["mask"] = "regions"
    _pick.setCurrentText("regions")
    _pick.clearFocus()
    for _ in range(3):
        _wev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), _QPoint(0, -120), _QPoint(0, -120),
                           Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        app.sendEvent(_pick, _wev)
        assert not _wev.isAccepted(), "an unfocused combo must let the panel scroll"
    assert pdoc.nodes["PW"].params.get("mask") == "regions", \
        "a wheel over an unfocused picker must not change the value"
    _mode = next((c for c in pinsp.findChildren(_NoWheelCombo)
                  if not c.isEditable()), None)
    if _mode is not None:
        _before = dict(pdoc.nodes["PW"].modes)
        _mode.clearFocus()
        _wev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), _QPoint(0, -120), _QPoint(0, -120),
                           Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
        app.sendEvent(_mode, _wev)
        assert pdoc.nodes["PW"].modes == _before, \
            "the same guard must cover the Mode dropdowns (this bug pre-dated the picker)"
    pinsp.set_node(None)
    pinsp.setParent(None)
    _ok("P1 layer picker: offers the upstream layers, commits by pick AND by free text, "
        "never offers a node its own output, and a wheel over an unfocused combo is inert "
        "(fixes a pre-existing Mode-dropdown bug too)")

    print("\nALL PHASE-5 GUI PROBES PASSED")
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
