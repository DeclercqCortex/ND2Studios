"""Main window for NodeLab v2 (Phase 5).

Chrome (G10): menu bar (File / Run / View), status bar, dockable Palette (G2),
Properties inspector, and Viewer (G4). File actions (G6) round-trip
``*.nd2graph.json`` through the document (headless
:mod:`nodegraph.serialize` + a ``ui`` extras object). Run actions (G7) submit pulls to
the :class:`~nodelab_v2.runner.EngineRunner` — double-click any node (or press F5 on a
selection) to view its output; edits invalidate in-flight results by epoch.

**Maximized canvas (2026-07-27).** The centre normally splits Viewer-over-canvas. The
canvas' top-right ⛶ button (also View → *Maximize node canvas*, ``Ctrl+Space``) hands
the whole centre to the graph and moves the *same* ViewerPanel into the
:class:`~nodelab_v2.minimap.MiniMapOverlay` — a bordered mini-map in the canvas'
top-left corner that previews whatever node you click, live (:data:`FOLLOW_DELAY_MS`
debounce; the previewed card wears an accent spine). ``Esc``, the mini-map's dock
button, or a header double-click puts the Viewer back in the splitter at its old size.
"""
from __future__ import annotations

from typing import Optional

import nodegraph.nodes  # noqa: F401 — registers the node catalog into NODES
from PySide6.QtCore import QPointF, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QDockWidget, QFileDialog, QInputDialog, QLabel, QMainWindow, QMessageBox, QSplitter,
    QWidget,
)

from nodelab_v2 import theme as T
from nodelab_v2.document import GraphDocument
from nodelab_v2.inspector import InspectorPanel
from nodelab_v2.minimap import MiniMapOverlay
from nodelab_v2.node_item import NodeItem
from nodelab_v2.palette import PalettePanel
from nodelab_v2.runner import EngineRunner, ensure_gui_ops
from nodelab_v2.scene import GraphScene, GraphView
from nodelab_v2.spreadsheet import SpreadsheetPanel
from nodelab_v2.viewer import ViewerPanel
from nodelab_v2.welcome import WelcomeCard

#: how long a selection must settle before the mini-map pulls it (ms). Long enough that
#: rubber-banding / arrowing through a chain doesn't queue a pull per node, short enough
#: that a single click feels immediate.
FOLLOW_DELAY_MS = 170

#: pulse period of the status-bar run LED (ms) — the footer twin of a card's status dot.
LED_PULSE_MS = 360

#: central splitter sizes once there is an image to look at (Viewer-dominant, ~2.5:1).
#: The app LAUNCHES with the Viewer collapsed instead — nothing has been pulled yet, so
#: the blank welcome canvas gets the whole centre; the first pull opens the Viewer.
VIEWER_SPLIT = [860, 340]

def _window_qss() -> str:
    return f"""
QMainWindow {{ background:{T.BG.name()}; }}
QMainWindow::separator {{ background:{T.BORDER.name()}; width:2px; height:2px; }}
QMenuBar {{ background:{T.BODY.name()}; color:{T.INK.name()};
  border-bottom:1px solid {T.BORDER.name()}; padding:2px 4px; }}
QMenuBar::item {{ background:transparent; padding:5px 11px; border-radius:6px;
  margin:0 1px; }}
QMenuBar::item:selected {{ background:{T.PANEL_HI.name()}; }}
QMenuBar::item:pressed {{ background:{T.ACCENT_DIM.name()}; }}
QMenu {{ background:{T.PANEL.name()}; color:{T.INK.name()};
  border:1px solid {T.BORDER.name()}; border-radius:9px; padding:5px; }}
QMenu::item {{ padding:6px 24px 6px 14px; border-radius:6px; }}
QMenu::item:selected {{ background:{T.ACCENT_DIM.name()}; }}
QMenu::separator {{ height:1px; background:{T.BORDER.name()}; margin:5px 10px; }}
QStatusBar {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
  border-top:1px solid {T.BORDER.name()}; }}
QStatusBar::item {{ border:0; }}
QDockWidget {{ color:{T.MUTED.name()}; font-size:10px; font-weight:800;
  titlebar-close-icon:none; titlebar-normal-icon:none; }}
QDockWidget::title {{ background:{T.BODY.name()}; color:{T.MUTED.name()};
  padding:6px 13px; border-bottom:1px solid {T.BORDER.name()}; }}
QSplitter::handle {{ background:{T.BG.name()}; }}
QSplitter::handle:hover {{ background:{T.BORDER.name()}; }}
QSplitter::handle:vertical {{ height:3px; }}
QSplitter::handle:horizontal {{ width:3px; }}
QTabWidget::pane {{ border:0; }}
QTabBar::tab {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
  padding:6px 15px; border:1px solid {T.BORDER.name()}; border-bottom:0;
  border-top-left-radius:7px; border-top-right-radius:7px; margin-right:2px; }}
QTabBar::tab:selected {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
QTabBar::tab:hover {{ background:{T.PANEL_HI.name()}; }}
""" + T.controls_qss()

FILE_FILTER = "nd2graph (*.nd2graph.json);;All files (*)"


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        ensure_gui_ops()
        self.setWindowTitle("NodeLab v2 — nodegraph canvas")
        self.setStyleSheet(_window_qss())

        self.doc = GraphDocument()
        self.scene = GraphScene(self.doc)
        self.view = GraphView(self.scene)
        self.runner = EngineRunner(self.doc)
        self._viewed: Optional[str] = None

        # centre: a DOMINANT Viewer on top over the node canvas, split vertically. With
        # the console removed, the whole window height is split here, so the image gets
        # the room to be seen well (~72% viewer / ~28% canvas) — but only once there IS
        # an image: the app launches with the Viewer collapsed so the blank welcome
        # canvas owns the centre, and the first pull opens the split (_open_viewer).
        self.viewer = ViewerPanel()
        center = QSplitter(Qt.Vertical)
        center.addWidget(self.viewer)
        center.addWidget(self.view)
        center.setStretchFactor(0, 5)     # viewer grows much faster
        center.setStretchFactor(1, 2)     # canvas keeps a usable slice
        center.setCollapsible(0, True)    # …and may be folded away entirely
        center.setCollapsible(1, False)
        center.setSizes([0, 1200])        # launch: all canvas, no image yet
        self._center = center
        self._center_sizes = list(VIEWER_SPLIT)
        self.setCentralWidget(center)

        # the canvas opens EMPTY: this card invites the first node and steps aside as
        # soon as the document has one (File → New brings it back).
        self.welcome = WelcomeCard(self.view)
        self.welcome.load_image_requested.connect(self.file_load_source)
        self.welcome.browse_nodes_requested.connect(self.focus_palette)
        self.welcome.example_requested.connect(self.build_demo)
        self.welcome.op_dropped.connect(self._on_op_dropped)

        # maximized canvas (Ctrl+Space / the ⛶ button): the Viewer moves OUT of the
        # splitter and into this HUD frame over the canvas' top-left corner, where it
        # keeps following whatever node you click.
        self.minimap = MiniMapOverlay(self.view)
        self._maximized = False
        self._follow_pending: Optional[str] = None
        self._follow_timer = QTimer(self)
        self._follow_timer.setSingleShot(True)
        self._follow_timer.setInterval(FOLLOW_DELAY_MS)
        self._follow_timer.timeout.connect(self._follow_pull)

        # docks
        self.palette = PalettePanel(on_add=self._add_at_center)
        pd = QDockWidget("Nodes", self)
        pd.setWidget(self.palette)
        pd.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.LeftDockWidgetArea, pd)

        # right sidebar: Properties + Spreadsheet (tabbed)
        self.inspector = InspectorPanel()
        idock = QDockWidget("Properties", self)
        idock.setWidget(self.inspector)
        idock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, idock)

        self.sheet = SpreadsheetPanel()
        sdock = QDockWidget("Spreadsheet", self)
        sdock.setWidget(self.sheet)
        sdock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable)
        self.addDockWidget(Qt.RightDockWidgetArea, sdock)
        self.tabifyDockWidget(idock, sdock)
        idock.raise_()

        # Console removed — the reclaimed bottom-dock space goes to the central splitter
        # (Viewer + node canvas). Run/error messages surface in the status bar, and a
        # failed pull shows its trace in the Viewer (status line + tooltip).

        self._build_menus()
        # run LED (the same language as the cards' status dot): a bead in the status bar
        # that pulses while the engine works. It is a PERMANENT widget — a temporary
        # showMessage() hides normal status-bar widgets, and this must stay visible.
        self._led = QLabel("●")
        self._led_state = "idle"
        self._led_on = True
        self._led_timer = QTimer(self)
        self._led_timer.setInterval(LED_PULSE_MS)
        self._led_timer.timeout.connect(self._led_tick)
        self.statusBar().addPermanentWidget(self._led)
        self._set_led("idle")
        self.statusBar().showMessage("ready")

        # signals
        self.scene.selectionChanged.connect(self._on_selection)
        self.scene.node_activated.connect(self.pull_node)
        self.view.op_dropped.connect(self._on_op_dropped)
        self.doc.on_change(self._on_doc_changed)
        self.doc.on_change(self.inspector.refresh_derived)   # G8 live ƒmd re-seed
        self.scene.pull_requested.connect(self.pull_node)
        self.scene.nodes_deleted.connect(self._on_nodes_deleted)
        self.runner.started.connect(
            lambda nid: (self.statusBar().showMessage(f"pulling {nid}…"),
                         self.viewer.show_running(nid),
                         self._set_led("busy"),
                         self.minimap.set_state("busy")))
        self.runner.finished.connect(self._on_run_finished)
        self.runner.plane_ready.connect(self._on_plane_ready)
        self.runner.failed.connect(self._on_run_failed)
        # per-node progress: the cards show queued → running → done/cached/error, and the
        # status bar counts the nodes that actually ran.
        self.runner.plan.connect(self._on_run_plan)
        self.runner.node_progress.connect(self._on_node_progress)
        self.viewer.request_changed.connect(self._on_view_request)
        self.view.maximize_toggled.connect(self.set_maximized)
        self.minimap.restore_requested.connect(lambda: self.set_maximized(False))
        self.doc.on_change(self._sync_welcome)

        self._sync_welcome()          # open on a blank, welcoming canvas

    # ── chrome (G10) ─────────────────────────────────────────────────────────
    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("&File")
        for text, seq, fn in (
                ("&New", QKeySequence.New, self.file_new),
                ("&Open…", QKeySequence.Open, self.file_open),
                ("&Save", QKeySequence.Save, self.file_save),
                ("Save &As…", QKeySequence.SaveAs, self.file_save_as)):
            act = QAction(text, self)
            act.setShortcut(seq)
            act.triggered.connect(fn)
            m_file.addAction(act)

        m_file.addSeparator()
        load = QAction("&Load ND2/TIFF file…", self)
        load.setShortcut("Ctrl+L")
        load.triggered.connect(self.file_load_source)
        m_file.addAction(load)

        m_file.addSeparator()
        exp = QAction("&Export table…", self)
        exp.setShortcut("Ctrl+E")
        exp.triggered.connect(self.file_export)
        m_file.addAction(exp)

        # ── Edit: the discoverable home of delete (the canvas also has the hover ✕ badge
        # and a right-click menu). The shortcuts are scoped to the CANVAS
        # (WidgetWithChildrenShortcut on the view) so pressing Del while editing a value
        # in the inspector still edits text instead of deleting the node behind it.
        m_edit = self.menuBar().addMenu("&Edit")
        self._del_act = QAction("&Delete selected", self)
        self._del_act.setShortcuts([QKeySequence(Qt.Key_Delete),
                                    QKeySequence(Qt.Key_Backspace)])
        self._del_act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self._del_act.setToolTip("Delete the selected nodes / wires / frames")
        self._del_act.triggered.connect(self.delete_selection)
        self.view.addAction(self._del_act)
        m_edit.addAction(self._del_act)
        self._dissolve_act = QAction("Dis&solve (delete, keep the chain)", self)
        self._dissolve_act.setShortcut("Ctrl+X")
        self._dissolve_act.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        self._dissolve_act.setToolTip(
            "Delete the selected node(s) and reconnect each one's input to whatever it fed")
        self._dissolve_act.triggered.connect(self.scene.dissolve_selection)
        self.view.addAction(self._dissolve_act)
        m_edit.addAction(self._dissolve_act)
        m_edit.addSeparator()
        sel_all = QAction("Select &all nodes", self)
        sel_all.setShortcut("Ctrl+A")
        sel_all.setShortcutContext(Qt.WidgetWithChildrenShortcut)
        sel_all.triggered.connect(self.select_all_nodes)
        self.view.addAction(sel_all)
        m_edit.addAction(sel_all)
        m_edit.aboutToShow.connect(self._sync_edit_actions)

        m_run = self.menuBar().addMenu("&Run")
        pull = QAction("&Pull selected node", self)
        pull.setShortcut("F5")
        pull.triggered.connect(self.pull_selected)
        m_run.addAction(pull)
        again = QAction("Pull &viewed again", self)
        again.setShortcut("Shift+F5")
        again.triggered.connect(lambda: self._viewed and self.pull_node(self._viewed))
        m_run.addAction(again)

        m_graph = self.menuBar().addMenu("&Graph")
        wrap = QAction("&Wrap selection in Repeat zone…", self)
        wrap.triggered.connect(self.wrap_repeat)
        m_graph.addAction(wrap)
        frame = QAction("&Frame selection…", self)
        frame.setShortcut("Ctrl+J")
        frame.triggered.connect(self.frame_selection)
        m_graph.addAction(frame)
        m_graph.addSeparator()
        grp = QAction("&Group selection…", self)
        grp.setShortcut("Ctrl+G")
        grp.triggered.connect(self.group_selection)
        m_graph.addAction(grp)
        ungrp = QAction("&Ungroup", self)
        ungrp.setShortcut("Ctrl+Shift+G")
        ungrp.triggered.connect(self.ungroup_selection)
        m_graph.addAction(ungrp)

        m_view = self.menuBar().addMenu("&View")
        fit = QAction("&Fit graph", self)
        fit.setShortcut("Home")
        fit.triggered.connect(self.view.fit_all)
        m_view.addAction(fit)
        self._max_act = QAction("&Maximize node canvas", self)
        self._max_act.setCheckable(True)
        self._max_act.setShortcut("Ctrl+Space")
        self._max_act.setToolTip("Give the whole centre to the graph; the Viewer becomes "
                                 "a mini-map in the canvas' top-left corner")
        self._max_act.toggled.connect(self.set_maximized)
        m_view.addAction(self._max_act)
        self._follow_act = QAction("&Preview clicked node", self)
        self._follow_act.setCheckable(True)
        self._follow_act.setToolTip("Pull and show a node as soon as you click it "
                                    "(always on while the canvas is maximized)")
        m_view.addAction(self._follow_act)
        m_view.addSeparator()
        ovl = QAction("&Overlays…", self)
        ovl.setShortcut("Ctrl+Shift+O")
        ovl.setToolTip("Configure the Point / Label / Track overlays — size, opacity, "
                       "look, colour — and save the look as a default")
        ovl.triggered.connect(self.viewer.open_overlay_dialog)
        m_view.addAction(ovl)
        m_view.addSeparator()
        self._light = QAction("&Light theme", self)
        self._light.setCheckable(True)
        self._light.toggled.connect(
            lambda on: self.set_theme("light" if on else "dark"))
        m_view.addAction(self._light)

        m_help = self.menuBar().addMenu("&Help")
        cap = QAction("&Capabilities…", self)
        cap.triggered.connect(self._show_capabilities)
        m_help.addAction(cap)

    # ── run LED ──────────────────────────────────────────────────────────────
    def _set_led(self, state: str) -> None:
        """``idle`` | ``busy`` | ``error`` — busy pulses, the others sit steady."""
        self._led_state = state
        self._led.setToolTip({"idle": "engine idle", "busy": "engine working",
                              "error": "last pull failed"}[state])
        if state == "busy":
            if not self._led_timer.isActive():
                self._led_on = True
                self._led_timer.start()
        else:
            self._led_timer.stop()
            self._led_on = True
        self._paint_led()

    def _paint_led(self) -> None:
        col = {"idle": T.MUTED, "busy": T.ACCENT, "error": T.ERROR}[self._led_state]
        # QSS takes rgba() for the off-beat; QColor.name() would drop the alpha
        rgba = f"rgba({col.red()},{col.green()},{col.blue()},{1.0 if self._led_on else 0.3})"
        self._led.setStyleSheet(f"color:{rgba};font-size:13px;padding:0 6px 0 2px;")

    def _led_tick(self) -> None:
        self._led_on = not self._led_on
        self._paint_led()

    # ── edit ─────────────────────────────────────────────────────────────────
    def _sync_edit_actions(self) -> None:
        """Grey out the delete entries when there is nothing selected, and name what
        they would act on (so the menu itself teaches the shortcut)."""
        nodes = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        n_any = len(self.scene.selectedItems())
        self._del_act.setEnabled(bool(n_any))
        self._del_act.setText("&Delete selected" if n_any != 1 else
                              f"&Delete {self._sel_label()}")
        self._dissolve_act.setEnabled(bool(nodes))

    def _sel_label(self) -> str:
        for it in self.scene.selectedItems():
            if isinstance(it, NodeItem):
                spec = it.rec.spec()
                return f"'{spec.label if spec else it.op_key}'"
        return "selection"

    def delete_selection(self) -> None:
        """Edit → Delete: remove the selected nodes/wires/frames. Reports what went, so a
        deletion is never silent (the canvas is busy — a card vanishing off-screen would
        otherwise be invisible)."""
        n_nodes = sum(1 for i in self.scene.selectedItems() if isinstance(i, NodeItem))
        n_all = len(self.scene.selectedItems())
        if not n_all:
            self.statusBar().showMessage(
                "nothing selected — click a node (or wire) first, then Del")
            return
        self.scene.delete_selection()
        self.statusBar().showMessage(
            f"deleted {n_nodes} node(s)" if n_nodes else f"deleted {n_all} item(s)")

    def select_all_nodes(self) -> None:
        for item in self.scene.node_items.values():
            item.setSelected(True)

    def _on_nodes_deleted(self, node_ids) -> None:
        ids = list(node_ids)
        self.statusBar().showMessage(
            f"deleted {', '.join(ids)}" if len(ids) <= 4 else f"deleted {len(ids)} nodes")

    def _show_capabilities(self) -> None:
        """What this editor covers. (Was the Phase-7 "v2 vs legacy v1" note; NodeLab v1
        was removed 2026-07-29 — see `V2.05_phase7_capability_matrix.md` §6.)"""
        QMessageBox.information(
            self, "NodeLab — capabilities",
            "NodeLab runs on the nodegraph engine: a lazy-pull, per-tile streaming, "
            "two-hash-memoized graph with a metadata-intelligent node catalog.\n\n"
            "Covered:\n"
            "  • Enhancement / denoise suite + metadata-derived deconvolution\n"
            "  • Thresholding (global / local / multi-Otsu / histogram methods), "
            "connected components, watershed, EDT, boundaries\n"
            "  • Detection: spots, particles, StarDist nuclei\n"
            "  • Correlation: DIC (pyALDIC) and DVC (ALDVC) + cumulative "
            "accumulation and field rasterization\n"
            "  • Structure: point clustering, tessellation / mesh, measurements, "
            "domain transfer\n"
            "  • Tracking: frame linking + the 5-method tracker, with track overlays\n"
            "  • The 2D/3D lever, zones, groups, reroutes and frames\n"
            "  • A live multi-channel Viewer, Spreadsheet, and CSV / Parquet / Arrow "
            "export\n\n"
            "Retired with NodeLab v1 (declared obsolete): cell-tracker spatial maps, "
            "DIC mesh refinement, the bleach / blob-subtract / spatial-flatness / "
            "temporal-fold enhancements, Cellpose nuclei, the interactive workflow-IO "
            "nodes (3D mask drawing, review, exclude, pause, prism, save data, "
            "checkpoints) and if/else.\n\n"
            "See CodeLog/ClaudesPlan/V2.05_phase7_capability_matrix.md.")

    def set_theme(self, mode: str) -> None:
        """Rebind the palette (G9) and re-apply it everywhere: node cards repaint from
        the tokens, QSS panels restyle, the canvas background updates."""
        T.apply(mode)
        self.setStyleSheet(_window_qss())
        for panel in (self.palette, self.inspector, self.viewer, self.sheet,
                      self.minimap, self.welcome, self.view):
            panel.restyle()
        self._paint_led()          # the LED colors come from the tokens, not from QSS
        self.view.setBackgroundBrush(T.BG)
        self.scene.update()
        self.view.viewport().update()

    # ── maximized canvas + mini-map (2026-07-27) ──────────────────────────────
    def set_maximized(self, on: bool) -> None:
        """Toggle the maximized node canvas.

        **On** — the Viewer leaves the splitter (so the canvas owns the whole centre)
        and is re-homed into the :class:`~nodelab_v2.minimap.MiniMapOverlay` pinned to
        the canvas' top-left corner, trimmed to its compact layout, with click-to-preview
        forced on so the mini-map follows the node you're working on.
        **Off** — the Viewer goes back into the splitter at its previous size and
        click-to-preview returns to whatever the user had chosen.

        The same live ViewerPanel widget is moved (never a second copy), so channels,
        LUT, playback and overlays carry straight across."""
        on = bool(on)
        if on == self._maximized:
            return
        self._maximized = on
        if on:
            self._center_sizes = self._center.sizes()
            self.viewer.set_compact(True)
            self.minimap.attach(self.viewer)
            self.minimap.reposition()
            self.minimap.show()
            self.minimap.raise_()
            self._follow_before_max = self._follow_act.isChecked()
            self._follow_act.setChecked(True)
            # the splitter re-lays out on the next turn — re-anchor once the canvas has
            # actually grown into the freed space
            QTimer.singleShot(0, self.minimap.reposition)
        else:
            self.minimap.detach()
            self.minimap.hide()
            self.viewer.set_compact(False)
            self._center.insertWidget(0, self.viewer)
            self.viewer.show()
            back = self._center_sizes or list(VIEWER_SPLIT)
            self._center.setSizes(back if back[0] >= 80 else list(VIEWER_SPLIT))
            self._follow_act.setChecked(getattr(self, "_follow_before_max", False))
        # keep both entry points (canvas button + View menu) in sync, no signal loop
        self.view.set_maximized(on)
        self._max_act.blockSignals(True)
        self._max_act.setChecked(on)
        self._max_act.blockSignals(False)
        self._sync_minimap_title()
        self.statusBar().showMessage(
            "canvas maximized — click any node to preview it in the mini-map (Esc to "
            "dock the Viewer back)" if on else "Viewer docked")

    def _open_viewer(self) -> None:
        """Make sure the Viewer pane is actually on screen before showing a result — it
        launches folded away (blank canvas, nothing pulled) and a user can fold it back
        by dragging the splitter. Only acts when it is folded; a pane the user has
        already sized is left exactly as it is."""
        if self._maximized:
            self._center_sizes = list(VIEWER_SPLIT)   # applies when the Viewer docks back
            return
        if self._center.sizes()[0] < 80:
            self._center.setSizes(list(VIEWER_SPLIT))

    def _sync_minimap_title(self) -> None:
        """The mini-map header names what it is showing (id · node label)."""
        nid = self._viewed
        rec = self.doc.nodes.get(nid) if nid else None
        if rec is None:
            self.minimap.set_title("viewer · click a node")
            return
        spec = rec.spec()
        self.minimap.set_title(f"{nid} · {spec.label if spec else rec.op_key}")

    def _follow_pull(self) -> None:
        """Debounced click-to-preview: pull the node the selection settled on."""
        nid, self._follow_pending = self._follow_pending, None
        if nid and nid in self.doc.nodes and nid != self._viewed:
            self.pull_node(nid)

    # ── document plumbing ─────────────────────────────────────────────────────
    def _on_doc_changed(self) -> None:
        self.runner.invalidate()          # an edit supersedes in-flight results
        if self._viewed is not None and self._viewed not in self.doc.nodes:
            self._viewed = None           # the previewed node was deleted/reloaded away
            self.scene.set_viewed(None)
            self.minimap.set_state("idle")
            self._sync_minimap_title()
        name = self.doc.path or "untitled"
        self.statusBar().showMessage(f"{name} — rev {self.doc.revision}")

    def _on_selection(self) -> None:
        try:
            sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        except RuntimeError:
            return          # scene torn down (app closing) — the C++ object is gone
        self.inspector.set_node(sel[0] if sel else None)
        # click-to-preview (always on while maximized): debounce, so dragging a marquee
        # across a chain queues ONE pull — the node the selection settled on.
        if sel and self._follow_act.isChecked():
            self._follow_pending = sel[0].node_id
            self._follow_timer.start()

    def _add_at_center(self, op_key: str) -> None:
        c = self.view.mapToScene(self.view.viewport().rect().center())
        self.doc.add_node(op_key, x=c.x() - 100, y=c.y() - 40)

    def wrap_repeat(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            QMessageBox.information(self, "Wrap in Repeat zone",
                                    "Select a linear sub-chain (one dataset input, one "
                                    "output) to wrap.")
            return
        n, ok = QInputDialog.getInt(self, "Repeat zone", "Iterations:", 3, 1, 100000)
        if not ok:
            return
        try:
            self.doc.wrap_repeat_zone(sel, iterations=n)
        except ValueError as exc:
            QMessageBox.warning(self, "Can't wrap", str(exc))
            return
        self.statusBar().showMessage(f"wrapped {len(sel)} node(s) in a Repeat×{n} zone")

    def group_selection(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            QMessageBox.information(
                self, "Group selection",
                "Select a linear sub-chain (one dataset input, one output) to collapse "
                "into a reusable group node. Sources must stay outside the group.")
            return
        name, ok = QInputDialog.getText(self, "Make group", "Group name:", text="Group")
        if not ok:
            return
        try:
            inst = self.doc.make_group(sel, name or "Group")
        except ValueError as exc:
            QMessageBox.warning(self, "Can't group", str(exc))
            return
        item = self.scene.node_items.get(inst)
        if item is not None:
            self.scene.clearSelection()
            item.setSelected(True)
        self.statusBar().showMessage(f"grouped {len(sel)} node(s) into '{name or 'Group'}'")

    def ungroup_selection(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems()
               if isinstance(i, NodeItem) and i._is_group]
        if not sel:
            QMessageBox.information(self, "Ungroup",
                                   "Select a group node to inline its contents back into "
                                   "the canvas.")
            return
        n = sum(1 for nid in sel if self.doc.ungroup(nid))
        self.statusBar().showMessage(f"ungrouped {n} group(s)")

    def frame_selection(self) -> None:
        sel = [i.node_id for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if not sel:
            QMessageBox.information(self, "Frame selection",
                                    "Select one or more nodes to enclose in a frame.")
            return
        title, ok = QInputDialog.getText(self, "Frame", "Frame label:", text="Frame")
        if not ok:
            return
        self.doc.add_frame(title or "Frame", sel)
        self.statusBar().showMessage(f"framed {len(sel)} node(s)")

    def _on_op_dropped(self, op_key: str, pos: QPointF) -> None:
        # splice-on-wire: if the node is dropped on a link, insert it into that wire
        e = self.scene.edge_at(pos)
        edge_tuple = e.model_edge if e is not None else None
        rec = self.doc.add_node(op_key, x=pos.x() - 20, y=pos.y() - 20)
        if edge_tuple is not None:
            self.scene.splice_onto(rec.id, edge_tuple)

    # ── run (G7) ─────────────────────────────────────────────────────────────
    def pull_selected(self) -> None:
        sel = [i for i in self.scene.selectedItems() if isinstance(i, NodeItem)]
        if sel:
            self.pull_node(sel[0].node_id)

    def pull_node(self, node_id: str) -> None:
        self._viewed = node_id
        self.scene.set_viewed(node_id)       # accent spine on the card being shown
        self._sync_minimap_title()
        self.runner.pull(node_id, self.viewer.coords(), self.viewer.channels())

    def _on_view_request(self) -> None:
        if self._viewed is not None:
            # coords/channel-only change → runner serves from the decoded-plane cache
            # (no graph snapshot / engine re-pull) when the graph is unchanged.
            self.runner.request_plane(self._viewed, self.viewer.coords(),
                                      self.viewer.channels())

    def _on_run_finished(self, node_id, payload, plane, axes, seconds) -> None:
        if plane:
            self._open_viewer()       # there is something to see now — unfold the pane
        self.viewer.show_result(node_id, plane, axes, seconds, dataset=payload)
        if self._maximized:
            # a new axes shape rebuilds the channel/LUT controls, and fresh widgets are
            # visible — re-fold them so the mini-map keeps its compact strip
            self.viewer.set_compact(True, force=True)
        self.sheet.show_dataset(node_id, payload)
        self.minimap.set_state("live")
        self._set_led("idle")
        # settle the cards: nothing is queued now, and the pulled node wears the run's
        # wall time (its own compute may have been microseconds — the wait was the read)
        self.scene.finish_run(node_id, seconds=seconds)
        computed = getattr(self, "_run_computed", 0)
        cached = getattr(self, "_run_cached", 0)
        detail = f" ({computed} computed, {cached} cached)" if (computed or cached) else ""
        self.statusBar().showMessage(f"{node_id} pulled in {seconds:.2f}s{detail}")

    # ── per-node progress (G7 + 2026-07-28) ──────────────────────────────────
    def _on_run_plan(self, target: str, node_ids) -> None:
        self._run_target = target
        self._run_plan = [n for n in node_ids if n in self.doc.nodes]
        self._run_computed = 0
        self._run_cached = 0
        self.scene.set_run_plan(target, self._run_plan)

    def _on_node_progress(self, event: str, node_id: str, info: dict) -> None:
        self.scene.on_node_progress(event, node_id, info)
        if event == "done":
            self._run_computed = getattr(self, "_run_computed", 0) + 1
        elif event == "cached":
            self._run_cached = getattr(self, "_run_cached", 0) + 1
        if event in ("start", "progress", "decode"):
            total = len(getattr(self, "_run_plan", ()) or ())
            ran = getattr(self, "_run_computed", 0) + getattr(self, "_run_cached", 0)
            frac = info.get("fraction")
            where = ("reading planes" if event == "decode"
                     else f"{node_id}{(' · ' + str(info.get('note'))) if info.get('note') else ''}")
            pct = f" {int(round(frac * 100))}%" if isinstance(frac, float) else ""
            self.statusBar().showMessage(
                f"pulling {getattr(self, '_run_target', node_id)} — "
                f"{min(ran + 1, total) if total else 1}/{total or 1} · {where}{pct}")

    def _on_plane_ready(self, node_id, planes, axes, seconds) -> None:
        # fast-path display update (scrub/play): no dataset re-delivery, no spreadsheet
        # refresh — only the viewer's frame changes.
        self.viewer.show_planes(node_id, planes, axes, seconds)

    def _on_run_failed(self, node_id, trace) -> None:
        # console removed: the Viewer shows the failing line (status) + full trace (tooltip)
        # …and the card that raised keeps its red 'error' state (the engine's error event
        # named it, which is more precise than the pulled node).
        self.scene.finish_run(node_id, failed=True)
        self.viewer.show_error(node_id, trace)
        self._set_led("error")
        self.minimap.set_state("error")
        last = [ln for ln in trace.strip().splitlines() if ln.strip()]
        self.statusBar().showMessage(f"{node_id} FAILED — {last[-1] if last else 'see Viewer'}")

    # ── file (G6) ────────────────────────────────────────────────────────────
    def file_new(self) -> None:
        self.doc.clear()
        self._viewed = None
        self.scene.set_viewed(None)
        self.minimap.set_state("idle")
        self._sync_minimap_title()

    def file_load_source(self) -> None:
        """File → Load ND2/TIFF file…: pick an image, read its metadata (no pixels), and
        drop a pre-loaded ``io.load`` source node titled with the file name and carrying
        one output socket per channel (+ a combined 'All channels'). Pixels ingest lazily
        on the first pull; the seeded envelope makes the sockets/colors appear at once."""
        import os
        from nodelab_v2.document import CHANNELS_KEY, TITLE_KEY
        from nodegraph.metadata import MetaEnvelope

        path, _f = QFileDialog.getOpenFileName(
            self, "Load ND2 / TIFF file", "",
            "Microscopy images (*.nd2 *.tif *.tiff);;ND2 (*.nd2);;"
            "TIFF (*.tif *.tiff);;All files (*)")
        if not path:
            return
        try:
            from nodelab_v2.ingest import read_meta_only
            axes, calib, disp = read_meta_only(path)
        except Exception as exc:  # noqa: BLE001 — any reader failure → a friendly dialog
            QMessageBox.warning(self, "Load failed",
                                f"Could not read:\n  {path}\n\n{exc}")
            return
        names = disp.get("channel_names") or [f"Ch{i}" for i in range(axes.c)]
        emis = disp.get("channel_emission_nm")
        colors = disp.get("channel_colors")

        def _native(i):
            # only accept an already-unpacked [r,g,b] triple; ND2's packed-int colorRGB
            # has an ambiguous byte order, so we fall back to the emission tint for those.
            if isinstance(colors, (list, tuple)) and i < len(colors):
                col = colors[i]
                if isinstance(col, (list, tuple)) and len(col) == 3:
                    return [int(v) for v in col]
            return None

        chans = [{
            "name": names[i] if i < len(names) else f"Ch{i}",
            "emission_nm": (emis[i] if isinstance(emis, (list, tuple)) and i < len(emis)
                            else None),
            "color": _native(i),
        } for i in range(axes.c)]

        c = self.view.mapToScene(self.view.viewport().rect().center())
        rec = self.doc.add_node(
            "io.load", x=c.x() - 107, y=c.y() - 40,
            params={"path": path, TITLE_KEY: os.path.basename(path),
                    CHANNELS_KEY: chans})
        self.doc.set_meta_seed(rec.id, MetaEnvelope(axes=axes, metadata=dict(calib)))
        item = self.scene.node_items.get(rec.id)
        if item is not None:
            self.scene.clearSelection()
            item.setSelected(True)
        self.statusBar().showMessage(
            f"loaded {os.path.basename(path)} — {axes.c} channel(s)")

    def file_open(self) -> None:
        path, _f = QFileDialog.getOpenFileName(self, "Open graph", "", FILE_FILTER)
        if not path:
            return
        try:
            self.doc.load_file(path)
        except (ValueError, OSError) as exc:
            QMessageBox.warning(self, "Open failed", str(exc))
            return
        self.view.fit_all()
        if self.doc.has_unedited_structure:
            QMessageBox.information(
                self, "Zones / groups preserved",
                "This file contains zones or node groups that this build can't edit "
                "yet. They are shown as their member nodes and preserved unchanged on "
                "save — editing/creating them in the GUI is a later phase.")

    def file_save(self) -> None:
        if not self.doc.path:
            self.file_save_as()
            return
        self.doc.save_file(self.doc.path)
        self.statusBar().showMessage(f"saved {self.doc.path}")

    def file_save_as(self) -> None:
        path, _f = QFileDialog.getSaveFileName(self, "Save graph", "graph.nd2graph.json",
                                               FILE_FILTER)
        if not path:
            return
        self.doc.save_file(path)
        self.statusBar().showMessage(f"saved {path}")

    def file_export(self) -> None:
        from nodelab_v2.export import export_dataset
        ds = self.sheet.dataset
        if ds is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Pull a node with Label/Point/Track structures first "
                                    "(the Spreadsheet tab shows what's exportable).")
            return
        nid = self.sheet.node_id or "table"
        path, _f = QFileDialog.getSaveFileName(
            self, "Export structure tables", f"{nid}.csv",
            "CSV (*.csv);;Parquet (*.parquet);;Arrow IPC (*.arrow)")
        if not path:
            return
        try:
            n = export_dataset(ds, path)
        except (ValueError, OSError, ImportError) as exc:
            QMessageBox.warning(self, "Export failed", str(exc))
            return
        self.statusBar().showMessage(f"exported {n} rows → {path}")

    # ── empty canvas / example content ────────────────────────────────────────
    def _sync_welcome(self) -> None:
        """Show the welcome card exactly while the canvas holds no nodes."""
        self.welcome.setVisible(not self.doc.nodes)
        if self.welcome.isVisible():
            self.welcome.raise_()
            self.minimap.raise_()          # the mini-map still owns its corner

    def focus_palette(self) -> None:
        """Put the cursor in the Nodes palette search (the welcome card's shortcut to
        placing a first node)."""
        w = self.palette.parentWidget()
        while w is not None and not isinstance(w, QDockWidget):
            w = w.parentWidget()
        if w is not None:
            w.show()
            w.raise_()
        self.palette.focus_search()

    def build_demo(self) -> None:
        """Replace the canvas with the small example chain (Load → Select → enhance →
        threshold → label → measure, plus a deconvolve → Viewer branch). Reachable from
        the welcome card; also what the GUI probe drives."""
        self.doc.clear()
        d = self.doc
        specs = [
            ("n1", "io.load", {}, 30, 150),
            ("n2", "channel.select", {}, 290, 150),
            ("n3", "enhance.gaussian", {"dim": "3D"}, 548, 90),
            ("n4", "analysis.threshold", {}, 806, 90),
            ("n5", "analysis.label", {}, 1064, 90),
            ("n6", "analysis.measure", {}, 1330, 150),
            ("n7", "enhance.deconvolve", {"dim": "3D"}, 548, 430),
            ("n8", "view.viewer", {}, 806, 430),
        ]
        for nid, op, modes, x, y in specs:
            d.add_node(op, node_id=nid, modes=modes, x=x, y=y)
        for s, ss, dd, ds in [
                ("n1", "image", "n2", "data"), ("n2", "out", "n3", "data"),
                ("n3", "out", "n4", "data"), ("n4", "out", "n5", "data"),
                ("n5", "out", "n6", "data"), ("n2", "out", "n7", "data"),
                ("n7", "out", "n8", "data")]:
            d.connect(s, ss, dd, ds)
        self.scene.node_items["n7"].setSelected(True)
        self.view.fit_all()
        self.statusBar().showMessage("example graph loaded — double-click a node to view it")


__all__ = ["MainWindow"]
