"""Structure-table export (Phase 6 / V2.00 §10 — "the export path reads from here").

Takes the per-``(domain, layer)`` tables that :func:`nodelab_v2.spreadsheet.structure_tables`
produces off a pulled Dataset and writes them to disk. Qt-free (a caller supplies the
path); numpy + stdlib for CSV, pyarrow **lazily imported** for Arrow/Parquet so the
core has no hard pyarrow dependency.

* ``.csv`` — one file; when a Dataset has several structure tables the domain/layer is
  written as leading ``domain``/``layer`` columns so everything lands in one sheet.
* ``.parquet`` / ``.arrow`` — a single table (same long form), via pyarrow.

Column order follows :func:`nodelab_v2.spreadsheet` (coordinate columns first).
"""
from __future__ import annotations

import csv
from typing import Dict, List, Optional, Tuple

import numpy as np

from nodelab_v2.spreadsheet import _ordered_columns, structure_tables

Tables = Dict[Tuple[str, Optional[str]], Dict[str, np.ndarray]]


def _long_rows(tables: Tables) -> Tuple[List[str], List[list]]:
    """Flatten all tables into (header, rows) long form: leading ``domain``/``layer``
    columns then the union of attribute columns (coordinate columns first)."""
    attr_names: List[str] = []
    for cols in tables.values():
        for n in _ordered_columns(cols):
            if n not in attr_names:
                attr_names.append(n)
    # keep coordinate columns first across the union
    attr_names = _ordered_columns({n: None for n in attr_names})
    # the injected leading columns must not collide with an attribute column literally
    # named "domain"/"layer" (real structure tables never do, but stay robust): prefix
    # with "_" until unique.
    dom_key, lay_key = "domain", "layer"
    while dom_key in attr_names:
        dom_key = "_" + dom_key
    while lay_key in attr_names:
        lay_key = "_" + lay_key
    header = [dom_key, lay_key, *attr_names]
    rows: List[list] = []
    for (dom, layer), cols in sorted(tables.items()):
        n = max((len(v) for v in cols.values()), default=0)
        for r in range(n):
            row = [dom, layer if layer is not None else ""]
            for name in attr_names:
                # None (not "") for a missing cell: csv renders it empty, and Arrow
                # keeps a numeric column numeric (a "" would force str and fail typing)
                v = cols[name][r] if name in cols and r < len(cols[name]) else None
                row.append(v.item() if isinstance(v, np.generic) else v)
            rows.append(row)
    return header, rows


def export_csv(tables: Tables, path: str) -> int:
    """Write the tables as one long-form CSV; returns the row count."""
    header, rows = _long_rows(tables)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return len(rows)


def export_arrow(tables: Tables, path: str) -> int:
    """Write the tables as a single Parquet/Arrow file (pyarrow lazily imported)."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    header, rows = _long_rows(tables)
    columns = {name: [row[i] for row in rows] for i, name in enumerate(header)}
    table = pa.table(columns)
    if path.endswith(".arrow"):
        with pa.OSFile(path, "wb") as sink:
            with pa.ipc.new_file(sink, table.schema) as writer:
                writer.write_table(table)
    else:
        pq.write_table(table, path)
    return len(rows)


def export_dataset(dataset, path: str) -> int:
    """Export a pulled Dataset's structure tables to ``path`` (extension picks the
    format). Raises :class:`ValueError` if the Dataset has no structures to export."""
    tables = structure_tables(dataset)
    if not tables:
        raise ValueError("no structure tables on this output to export")
    lower = path.lower()
    if lower.endswith((".parquet", ".arrow")):
        return export_arrow(tables, path)
    return export_csv(tables, path)


__all__ = ["export_dataset", "export_csv", "export_arrow", "structure_tables"]
