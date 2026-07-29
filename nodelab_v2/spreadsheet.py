"""Spreadsheet panel (G5 / V2.00 §10) — the per-domain structure table.

A pulled :class:`~nodegraph.dataset.Dataset` carries its detected structures as
element-indexed :class:`~nodegraph.dataset.AttributeLayer`\\ s on the Label / Point /
Track domains, keyed by source layer (``Dataset.with_structure``). This panel groups
those into tables — one per ``(domain, layer)`` — and shows the selected one as rows
(elements, ordered by id) × columns (attributes, coordinate columns first per
:data:`nodegraph.structure.COORD_COLUMNS`). The export path (Phase 6) reads from here.

Lattice/Voxel attributes are per-voxel rasters (image-like) — shown in the Viewer, not
tabulated here. Qt; reads only the numpy columns off the Dataset.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox, QHBoxLayout, QLabel, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from nodegraph.domains import is_structure
from nodegraph.structure import COORD_COLUMNS
from nodelab_v2 import theme as T

_COORD_ORDER = {name: i for i, name in enumerate(COORD_COLUMNS)}


def structure_tables(dataset) -> Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]]:
    """Group a Dataset's structure-domain attributes into ``{(domain, layer):
    {col_name: values}}`` — one entry per detected Label/Point/Track instance."""
    tables: Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]] = {}
    if dataset is None or not hasattr(dataset, "attributes"):
        return tables
    for (domain, layer, name), attr in dataset.attributes.items():
        if not is_structure(domain):
            continue
        tables.setdefault((domain.value, layer), {})[name] = attr.values
    return tables


def _ordered_columns(cols: Dict[str, np.ndarray]) -> List[str]:
    """Coordinate columns first (canonical order), then the rest alphabetically."""
    return sorted(cols, key=lambda n: (_COORD_ORDER.get(n, len(_COORD_ORDER)), n))


class SpreadsheetPanel(QWidget):
    """Per-domain structure table for the viewed node's Dataset."""

    def __init__(self) -> None:
        super().__init__()
        self.restyle()
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)
        head = QHBoxLayout()
        self._pick = QComboBox()
        self._pick.currentIndexChanged.connect(self._show_current)
        head.addWidget(QLabel("Domain"))
        head.addWidget(self._pick, 1)
        v.addLayout(head)
        self._table = QTableWidget()
        self._table.setAlternatingRowColors(False)
        self._table.setEditTriggers(QTableWidget.NoEditTriggers)
        v.addWidget(self._table, 1)
        self._status = QLabel("")
        self._status.setProperty("role", "muted")
        v.addWidget(self._status)
        self._tables: Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]] = {}
        self.dataset = None                       # the last shown Dataset (for export)
        self.node_id: Optional[str] = None

    def restyle(self) -> None:
        self.setStyleSheet(f"""
            QWidget {{ background:{T.PANEL.name()}; color:{T.INK.name()}; }}
            QLabel[role="muted"] {{ color:{T.MUTED.name()}; }}
            QTableWidget {{ background:{T.BG.name()}; color:{T.INK.name()};
                gridline-color:{T.BORDER.name()}; border:1px solid {T.BORDER.name()};
                border-radius:8px; font-family:{T.MONO}; outline:0; }}
            QHeaderView::section {{ background:{T.BODY.name()}; color:{T.INK_2.name()};
                border:0; border-right:1px solid {T.BORDER.name()};
                border-bottom:1px solid {T.BORDER.name()}; padding:4px 7px;
                font-weight:600; }}
            QTableCornerButton::section {{ background:{T.BODY.name()};
                border:0; border-bottom:1px solid {T.BORDER.name()}; }}
            QTableWidget::item {{ padding:2px 4px; }}
            QTableWidget::item:selected {{ background:{T.ACCENT_DIM.name()};
                color:{T.INK.name()}; }}
        """ + T.controls_qss())

    def show_dataset(self, node_id: str, dataset) -> None:
        self.dataset = dataset
        self.node_id = node_id
        self._tables = structure_tables(dataset)
        self._pick.blockSignals(True)
        self._pick.clear()
        for (dom, layer) in sorted(self._tables):
            self._pick.addItem(f"{dom}" + (f" · {layer}" if layer else ""),
                               (dom, layer))
        self._pick.blockSignals(False)
        if self._tables:
            self._pick.setCurrentIndex(0)
            self._show_current()
        else:
            self._table.clear()
            self._table.setRowCount(0)
            self._table.setColumnCount(0)
            self._status.setText(f"{node_id}: no structure tables on this output")

    def _show_current(self, *_a) -> None:
        key = self._pick.currentData()
        cols = self._tables.get(key) if key is not None else None
        if not cols:
            return
        names = _ordered_columns(cols)
        n = max((len(v) for v in cols.values()), default=0)
        self._table.clear()
        self._table.setColumnCount(len(names))
        self._table.setRowCount(n)
        self._table.setHorizontalHeaderLabels(names)
        for c, name in enumerate(names):
            vals = cols[name]
            for r in range(n):
                val = vals[r] if r < len(vals) else None
                if isinstance(val, (float, np.floating)):
                    txt = f"{float(val):.4g}"
                elif val is None:
                    txt = ""
                else:
                    txt = str(val)
                it = QTableWidgetItem(txt)
                it.setTextAlignment(Qt.AlignVCenter | Qt.AlignRight)
                self._table.setItem(r, c, it)
        self._table.resizeColumnsToContents()
        dom, layer = key
        self._status.setText(f"{dom}" + (f" · {layer}" if layer else "")
                             + f" — {n} elements × {len(names)} attributes")


__all__ = ["SpreadsheetPanel", "structure_tables"]
