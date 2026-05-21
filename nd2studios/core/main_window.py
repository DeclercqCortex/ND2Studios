"""
ND2Studios main window.

Layout (PyDracula-derived):

    ┌────────────────────────────────────────────────┐
    │  custom title bar (drag, min/max/close)       │
    ├──────┬─────────────────────────────────────────┤
    │ side │  top bar (page title + status badge)   │
    │ bar  ├─────────────────────────────────────────┤
    │      │                                         │
    │ ☰    │  page stack (QStackedWidget)            │
    │ 📂   │                                         │
    │ 🧪   │                                         │
    │ 💾   │                                         │
    │      │                                         │
    │ New  │                                         │
    │ Save ├─────────────────────────────────────────┤
    │ Load │  bottom bar (progress + status text)   │
    └──────┴─────────────────────────────────────────┘

Sidebar collapses 60↔240 px with `QPropertyAnimation`. Frameless window
has `CustomGrip` resize handles on all four edges and a drop shadow on
the `#bgApp` background frame.
"""
from __future__ import annotations

import os
from typing import Dict, Optional

from PySide6.QtCore import (
    QEasingCurve, QEvent, QPropertyAnimation, QSize, Qt, QTimer,
)
from PySide6.QtGui import QColor, QMouseEvent
from PySide6.QtWidgets import (
    QButtonGroup, QFileDialog, QGraphicsDropShadowEffect, QHBoxLayout,
    QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QSizeGrip, QSplitter, QStackedWidget, QVBoxLayout, QWidget,
)

from nd2studios.core.experiment_manager import (
    ND2StudiosManager, SESSION_EXTENSION,
)
from nd2studios.core.settings import Settings
from nd2studios.widgets.common import StatusIndicator
from nd2studios.widgets.custom_grips import CustomGrip


class MainWindow(QMainWindow):
    """Frameless main window with collapsible sidebar and three pages."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{Settings.APP_NAME} v{Settings.APP_VERSION}")
        self.setMinimumSize(Settings.MIN_WIDTH, Settings.MIN_HEIGHT)
        self.resize(1440, 900)

        # Frameless + translucent so the rounded #bgApp shows through cleanly.
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        # State
        self.exp_manager = ND2StudiosManager(self)
        self.exp_manager.active_changed.connect(self._on_experiment_changed)
        self.exp_manager.status_changed.connect(self._on_status_changed)

        self.pages: Dict[str, QWidget] = {}
        self._current_page_key: Optional[str] = None
        self._sidebar_animation: Optional[QPropertyAnimation] = None
        self._sidebar_expanded: bool = True
        self._drag_pos = None
        self._is_maximized = False

        self._build_ui()
        self._install_grips()

        # Start with a blank session.
        self.exp_manager.new_experiment("Untitled")

    # ── UI construction ────────────────────────────────────────────
    def _build_ui(self) -> None:
        # Outer container so we can put a drop shadow on the inner #bgApp.
        outer = QWidget(self)
        self.setCentralWidget(outer)
        outer_layout = QVBoxLayout(outer)
        outer_layout.setContentsMargins(10, 10, 10, 10)  # room for drop shadow
        outer_layout.setSpacing(0)

        bg_app = QWidget()
        bg_app.setObjectName("bgApp")
        outer_layout.addWidget(bg_app)
        self._bg_app = bg_app

        # Drop shadow on the rounded card.
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(Settings.SHADOW_BLUR_RADIUS)
        shadow.setXOffset(Settings.SHADOW_OFFSET_X)
        shadow.setYOffset(Settings.SHADOW_OFFSET_Y)
        shadow.setColor(QColor(0, 0, 0, Settings.SHADOW_ALPHA))
        bg_app.setGraphicsEffect(shadow)

        bg_layout = QVBoxLayout(bg_app)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setSpacing(0)

        # ── Custom title bar ──
        bg_layout.addWidget(self._build_title_bar())

        # ── Body: sidebar + content area ──
        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        self._body_splitter = QSplitter(Qt.Horizontal)
        self._body_splitter.setChildrenCollapsible(False)
        self._body_splitter.addWidget(self._build_sidebar())
        self._body_splitter.addWidget(self._build_content_area())
        self._body_splitter.setStretchFactor(0, 0)
        self._body_splitter.setStretchFactor(1, 1)
        body_layout.addWidget(self._body_splitter, stretch=1)

        bg_layout.addWidget(body, stretch=1)

    def _build_title_bar(self) -> QWidget:
        bar = QWidget()
        bar.setObjectName("titleBarWidget")
        bar.setFixedHeight(Settings.TITLE_BAR_HEIGHT)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        app_label = QLabel(Settings.APP_NAME)
        app_label.setObjectName("titleBarApp")
        layout.addWidget(app_label)

        info_label = QLabel(Settings.APP_DESCRIPTION)
        info_label.setObjectName("titleBarInfo")
        layout.addWidget(info_label)
        self._title_info_label = info_label

        layout.addStretch(1)

        # Min / Max / Close
        self._btn_min = QPushButton("—")
        self._btn_min.setObjectName("titleBarBtn")
        self._btn_min.setToolTip("Minimize")
        self._btn_min.clicked.connect(self.showMinimized)

        self._btn_max = QPushButton("□")
        self._btn_max.setObjectName("titleBarBtn")
        self._btn_max.setToolTip("Maximize / Restore")
        self._btn_max.clicked.connect(self._toggle_max_restore)

        self._btn_close = QPushButton("✕")
        self._btn_close.setObjectName("titleBarCloseBtn")
        self._btn_close.setToolTip("Close")
        self._btn_close.clicked.connect(self.close)

        for b in (self._btn_min, self._btn_max, self._btn_close):
            layout.addWidget(b)

        # The whole title bar acts as a drag handle.
        bar.mousePressEvent = self._title_mouse_press
        bar.mouseMoveEvent = self._title_mouse_move
        bar.mouseDoubleClickEvent = self._title_mouse_double_click
        return bar

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setObjectName("leftMenuBg")
        sidebar.setMinimumWidth(Settings.SIDEBAR_COLLAPSED_WIDTH)
        self._sidebar = sidebar

        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Toggle (hamburger) at the top.
        self._toggle_btn = QPushButton("☰")
        self._toggle_btn.setObjectName("toggleBtn")
        self._toggle_btn.setToolTip("Collapse / Expand sidebar")
        self._toggle_btn.clicked.connect(self._toggle_sidebar)
        layout.addWidget(self._toggle_btn)

        # Nav buttons (one per page).
        self._nav_group = QButtonGroup(self)
        self._nav_group.setExclusive(True)
        self._nav_buttons: Dict[str, QPushButton] = {}

        for key, icon, title, tooltip in Settings.PAGES:
            btn = QPushButton(f"  {icon}    {title}")
            btn.setObjectName("navBtn")
            btn.setCheckable(True)
            btn.setToolTip(tooltip)
            btn.clicked.connect(lambda _checked, k=key: self._navigate(k))
            self._nav_group.addButton(btn)
            self._nav_buttons[key] = btn
            layout.addWidget(btn)

        layout.addStretch(1)

        # Session controls at the bottom.
        for label, slot, tooltip in [
            ("  📄  New",     self._new_session,    "Start a new empty session"),
            ("  💾  Save",    self._save_session,   "Save session to a .nd2s file"),
            ("  📂  Load",    self._load_session,   "Load a saved .nd2s session"),
        ]:
            btn = QPushButton(label)
            btn.setObjectName("sessionBtn")
            btn.setToolTip(tooltip)
            btn.clicked.connect(slot)
            layout.addWidget(btn)

        return sidebar

    def _build_content_area(self) -> QWidget:
        content = QWidget()
        content.setObjectName("contentArea")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # Top bar: page title + status indicator.
        top_bar = QWidget()
        top_bar.setObjectName("topBar")
        top_bar.setFixedHeight(48)
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(16, 0, 16, 0)

        self._title_label = QLabel(Settings.PAGES[0][2])
        self._title_label.setObjectName("titleLabel")
        top_layout.addWidget(self._title_label)
        top_layout.addStretch(1)

        self._status_indicator = StatusIndicator()
        top_layout.addWidget(self._status_indicator)

        layout.addWidget(top_bar)

        # Page stack — pages are imported lazily here so that core/ does
        # not have a hard cycle with pages/ at import time.
        from nd2studios.pages.import_page import ImportPage
        from nd2studios.pages.recipe_page import RecipePage
        from nd2studios.pages.export_page import ExportPage
        from nd2studios.pages.analysis_page import AnalysisPage
        from nd2studios.pages.results_page import ResultsPage
        from nd2studios.pages.batch_page import BatchPage

        page_classes = {
            "import": ImportPage,
            "recipe": RecipePage,
            "export": ExportPage,
            "analysis": AnalysisPage,
            "results": ResultsPage,
            "batch": BatchPage,
        }
        self._stack = QStackedWidget()
        for key, _icon, _title, _tooltip in Settings.PAGES:
            cls = page_classes.get(key)
            page = cls(self) if cls is not None else self._stub(key)
            self.pages[key] = page
            self._stack.addWidget(page)

        layout.addWidget(self._stack, stretch=1)

        # Bottom bar: version + progress bar + status text.
        bottom_bar = QWidget()
        bottom_bar.setObjectName("bottomBar")
        bottom_bar.setFixedHeight(Settings.BOTTOM_BAR_HEIGHT)
        bb_layout = QHBoxLayout(bottom_bar)
        bb_layout.setContentsMargins(16, 0, 16, 0)

        version_label = QLabel(f"{Settings.APP_NAME} v{Settings.APP_VERSION}")
        version_label.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        bb_layout.addWidget(version_label)
        bb_layout.addStretch(1)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedWidth(220)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setValue(0)
        self._progress_bar.setVisible(False)
        bb_layout.addWidget(self._progress_bar)

        self._status_text = QLabel("Ready")
        self._status_text.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 8pt;")
        bb_layout.addWidget(self._status_text)

        layout.addWidget(bottom_bar)

        # Pick the first nav button.
        first_key = Settings.PAGES[0][0]
        self._nav_buttons[first_key].setChecked(True)
        self._current_page_key = first_key

        return content

    def _stub(self, key: str) -> QWidget:
        """Fallback page if a real page class fails to import (dev safety net)."""
        w = QWidget()
        layout = QVBoxLayout(w)
        lbl = QLabel(f"{key.title()} — coming soon")
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lbl.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 14pt;")
        layout.addWidget(lbl)
        return w

    # ── Frameless window: title bar interactions ───────────────────
    def _title_mouse_press(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint()
            event.accept()

    def _title_mouse_move(self, event: QMouseEvent) -> None:
        if self._drag_pos is None or event.buttons() != Qt.LeftButton:
            return
        # If maximized, restore first (mimics PyDracula).
        if self._is_maximized:
            self._toggle_max_restore()
            self._drag_pos = event.globalPosition().toPoint()
            return
        delta = event.globalPosition().toPoint() - self._drag_pos
        self.move(self.pos() + delta)
        self._drag_pos = event.globalPosition().toPoint()
        event.accept()

    def _title_mouse_double_click(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            QTimer.singleShot(100, self._toggle_max_restore)

    def _toggle_max_restore(self) -> None:
        if self._is_maximized:
            self.showNormal()
            self._is_maximized = False
            self._btn_max.setText("□")
            for grip in self._grips:
                grip.show()
        else:
            self.showMaximized()
            self._is_maximized = True
            self._btn_max.setText("❐")
            for grip in self._grips:
                grip.hide()

    def _install_grips(self) -> None:
        # Edge grips for resizing the frameless window.
        self._grips = [
            CustomGrip(self, Qt.LeftEdge),
            CustomGrip(self, Qt.RightEdge),
            CustomGrip(self, Qt.TopEdge),
            CustomGrip(self, Qt.BottomEdge),
        ]
        self._size_grip = QSizeGrip(self)
        self._size_grip.setFixedSize(Settings.GRIP_SIZE, Settings.GRIP_SIZE)
        self._reposition_grips()

    def _reposition_grips(self) -> None:
        if not hasattr(self, "_grips"):
            return
        g = Settings.GRIP_SIZE
        w, h = self.width(), self.height()
        # Edges: hug the outer 10 px margin we left on `outer`.
        self._grips[0].setGeometry(0, g, g, h - 2 * g)            # left
        self._grips[1].setGeometry(w - g, g, g, h - 2 * g)        # right
        self._grips[2].setGeometry(0, 0, w, g)                    # top
        self._grips[3].setGeometry(0, h - g, w, g)                # bottom
        self._size_grip.move(w - g, h - g)

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        super().resizeEvent(event)
        self._reposition_grips()

    # ── Sidebar collapse/expand ────────────────────────────────────
    def _toggle_sidebar(self) -> None:
        start = self._sidebar.width()
        end = (
            Settings.SIDEBAR_COLLAPSED_WIDTH if self._sidebar_expanded
            else Settings.SIDEBAR_EXPANDED_WIDTH
        )
        self._sidebar_expanded = not self._sidebar_expanded

        anim = QPropertyAnimation(self._sidebar, b"minimumWidth")
        anim.setDuration(Settings.SIDEBAR_ANIMATION_MS)
        anim.setStartValue(start)
        anim.setEndValue(end)
        anim.setEasingCurve(QEasingCurve.InOutQuart)

        anim2 = QPropertyAnimation(self._sidebar, b"maximumWidth")
        anim2.setDuration(Settings.SIDEBAR_ANIMATION_MS)
        anim2.setStartValue(start)
        anim2.setEndValue(end)
        anim2.setEasingCurve(QEasingCurve.InOutQuart)

        anim.finished.connect(self._on_sidebar_anim_done)
        anim.start()
        anim2.start()
        # Keep refs alive until they finish.
        self._sidebar_animation = anim
        self._sidebar_animation_2 = anim2

    def _on_sidebar_anim_done(self) -> None:
        if self._sidebar_expanded:
            # Remove upper bound so the splitter handle can grow the sidebar past 240 px.
            self._sidebar.setMaximumWidth(16777215)
        self._reset_active_viewer_zoom()

    def _reset_active_viewer_zoom(self) -> None:
        page = self.pages.get(self._current_page_key)
        viewer = getattr(page, "viewer", None)
        if viewer and hasattr(viewer, "canvas"):
            viewer.canvas.reset_zoom()

    # ── Navigation ─────────────────────────────────────────────────
    def _navigate(self, page_key: str) -> None:
        prereq = Settings.PAGE_PREREQS.get(page_key)
        if prereq is not None and self.exp_manager.active is not None:
            required, msg = prereq
            order = Settings.STATUS_ORDER
            if order.index(self.exp_manager.active.status) < order.index(required):
                QMessageBox.information(self, "Step Required", msg)
                # Re-check the previous nav button.
                if self._current_page_key:
                    self._nav_buttons[self._current_page_key].setChecked(True)
                return

        # Save state from outgoing page if it implements `save_to_experiment`.
        if (self._current_page_key
                and self.exp_manager.active is not None):
            old_page = self.pages.get(self._current_page_key)
            if hasattr(old_page, "save_to_experiment"):
                old_page.save_to_experiment(self.exp_manager.active)

        # Switch.
        keys = [p[0] for p in Settings.PAGES]
        idx = keys.index(page_key)
        self._stack.setCurrentIndex(idx)
        self._current_page_key = page_key
        self._title_label.setText(Settings.PAGES[idx][2])
        self._nav_buttons[page_key].setChecked(True)

        # Notify incoming page.
        new_page = self.pages.get(page_key)
        if hasattr(new_page, "on_activated"):
            new_page.on_activated()

    def _on_experiment_changed(self) -> None:
        exp = self.exp_manager.active
        if exp is None:
            return
        self._status_indicator.set_status(exp.status)
        # Push state into every page that wants it.
        for page in self.pages.values():
            if hasattr(page, "load_from_experiment"):
                page.load_from_experiment(exp)

    def _on_status_changed(self, status: str) -> None:
        self._status_indicator.set_status(status)

    # ── Public hooks the pages call ────────────────────────────────
    def set_progress(self, value: int, text: str = "") -> None:
        self._progress_bar.setVisible(value > 0)
        self._progress_bar.setValue(value)
        if text:
            self._status_text.setText(text)

    def set_status_text(self, text: str) -> None:
        self._status_text.setText(text)

    # ── Session controls ───────────────────────────────────────────
    def _new_session(self) -> None:
        self.exp_manager.new_experiment("Untitled")
        self.set_status_text("New session")

    def _save_session(self) -> None:
        if self.exp_manager.active is None:
            return
        # Let pages flush their state first.
        for page in self.pages.values():
            if hasattr(page, "save_to_experiment"):
                page.save_to_experiment(self.exp_manager.active)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Session", "",
            f"ND2Studios Session (*{SESSION_EXTENSION})",
        )
        if path:
            self.exp_manager.save_session(path)
            self.set_status_text(f"Saved: {os.path.basename(path)}")

    def _load_session(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Session", "",
            f"ND2Studios Session (*{SESSION_EXTENSION})",
        )
        if path:
            self.exp_manager.load_session(path)
            self.set_status_text(f"Loaded: {os.path.basename(path)}")
