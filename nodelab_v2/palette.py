"""Registry-driven node palette (G2) — searchable, grouped by category; double-click
adds at the view center, or drag a row onto the canvas (mime
``application/x-nd2studios-op``, accepted by :class:`~nodelab_v2.scene.GraphView`)."""
from __future__ import annotations

from collections import defaultdict
from typing import Callable, Optional

from PySide6.QtCore import QMimeData, Qt
from PySide6.QtGui import QDrag
from PySide6.QtWidgets import (
    QLineEdit, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget,
)

from nodelab_v2 import theme as T
from nodelab_v2.scene import visible_specs


class _PaletteTree(QTreeWidget):
    def __init__(self) -> None:
        super().__init__()
        self.setHeaderHidden(True)
        self.setDragEnabled(True)
        self.setIndentation(12)

    def startDrag(self, _actions) -> None:
        it = self.currentItem()
        op = it.data(0, Qt.UserRole) if it is not None else None
        if not op:
            return
        mime = QMimeData()
        mime.setData("application/x-nd2studios-op", op.encode("utf-8"))
        drag = QDrag(self)
        drag.setMimeData(mime)
        drag.exec(Qt.CopyAction)


class PalettePanel(QWidget):
    """The dockable palette. ``on_add(op_key)`` fires on double-click/Enter."""

    def __init__(self, on_add: Callable[[str], None]) -> None:
        super().__init__()
        self._on_add = on_add
        self.restyle()
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)
        self._search = QLineEdit()
        self._search.setPlaceholderText("Search nodes…  (drag onto the canvas)")
        self._tree = _PaletteTree()
        lay.addWidget(self._search)
        lay.addWidget(self._tree)
        self._search.textChanged.connect(self.refill)
        self._tree.itemDoubleClicked.connect(self._add_current)
        self.refill("")

    def focus_search(self) -> None:
        """Select the search box (the welcome card's 'Browse nodes' lands here)."""
        self._search.setFocus(Qt.OtherFocusReason)
        self._search.selectAll()

    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QTreeWidget {{ background:{T.PANEL.name()}; border:0; outline:0; }}
            QTreeWidget::item {{ padding:4px 4px; border-radius:5px; }}
            QTreeWidget::item:selected {{ background:{T.ACCENT_DIM.name()};
                color:{T.INK.name()}; }}
            QTreeWidget::item:hover {{ background:{T.PANEL_HI.name()}; }}
        """ + T.controls_qss())

    def refill(self, text: str = "") -> None:
        t = (text or "").lower()
        self._tree.clear()
        by_cat = defaultdict(list)
        for spec in visible_specs():
            if t and t not in spec.label.lower() and t not in spec.op_key.lower():
                continue
            by_cat[spec.category].append(spec)
        for cat in sorted(by_cat):
            head = QTreeWidgetItem([cat.upper()])
            head.setFlags(Qt.ItemIsEnabled)
            head.setForeground(0, T.MUTED)
            self._tree.addTopLevelItem(head)
            for spec in sorted(by_cat[cat], key=lambda s: s.label):
                row = QTreeWidgetItem([spec.label])
                row.setData(0, Qt.UserRole, spec.op_key)
                row.setToolTip(0, f"{spec.op_key}\n{spec.description}")
                head.addChild(row)
            head.setExpanded(True)

    def _add_current(self, item: Optional[QTreeWidgetItem] = None, _col: int = 0) -> None:
        it = item or self._tree.currentItem()
        op = it.data(0, Qt.UserRole) if it is not None else None
        if op:
            self._on_add(op)


__all__ = ["PalettePanel"]
