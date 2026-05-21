"""V1.13 row-flush + center-aligned layout — synthetic verification.

Runs the four representative cases from the V1.13 plan:
  - Well3 hex-staggered diamond (61 tiles, expected 7 × 13 with row
    widths [2,3,5,5,6,6,7,6,6,5,5,3,2])
  - Well2 (17 unique cells × 4 visits, expected 4 × 17)
  - Diagonal 9-tile scan (expected 9 × 1)
  - Dense regular 4 × 17 mosaic (expected 4 × 17)

Pass/fail is printed for each.
"""
from __future__ import annotations

import os
import sys
from collections import Counter

# Make the package importable when run directly.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from nd2studios.backend.exporters.stitch_exporter import compute_tile_layout


def case(label, ok):
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {label}")
    return ok


# ----- Synthetic Well3: hex-staggered diamond -----
# Per the plan, row widths bottom→top stage Y are
# [2, 3, 5, 5, 6, 6, 7, 6, 6, 5, 5, 3, 2] (61 tiles).
# Within each row, X step = 2 tile widths (hex stagger). Rows are
# evenly spaced in Y by 1 tile height.
TILE_H = 1024
TILE_W = 1024
PIXEL_UM = 0.65
TILE_UM_X = TILE_W * PIXEL_UM   # 665.6 µm
TILE_UM_Y = TILE_H * PIXEL_UM   # 665.6 µm

ROW_WIDTHS_BOTTOM_UP = [2, 3, 5, 5, 6, 6, 7, 6, 6, 5, 5, 3, 2]
EXPECTED_ROW_WIDTHS_TOP_DOWN = list(reversed(ROW_WIDTHS_BOTTOM_UP))

X_STEP = 2 * TILE_UM_X
Y_STEP = TILE_UM_Y

stage_xy = []
m_indices = []
m_counter = 0
# Build bottom-up so row 0 (lowest Y) is the bottom row.
for r_bottom_up, w in enumerate(ROW_WIDTHS_BOTTOM_UP):
    y = r_bottom_up * Y_STEP
    # Center-staggered X positions, starting at a row-dependent offset
    # so the diamond shape is symmetric around X=0.
    leading = (max(ROW_WIDTHS_BOTTOM_UP) - w) // 2
    for k in range(w):
        x = (leading + k) * X_STEP
        stage_xy.append((x, y))
        m_indices.append(m_counter)
        m_counter += 1
N_WELL3 = len(stage_xy)

print("Case 1 — Well3 hex-staggered diamond")
print(f"  built {N_WELL3} tiles across "
      f"{len(ROW_WIDTHS_BOTTOM_UP)} rows")
layout = compute_tile_layout(
    stage_xy_um=stage_xy,
    pixel_size_um=PIXEL_UM,
    tile_h=TILE_H, tile_w=TILE_W,
    m_indices=m_indices,
)
print(f"  source = {layout.source!r}")
print(f"  canvas = {layout.canvas_w} x {layout.canvas_h}")
print(f"  n_phys_cols x n_phys_rows = "
      f"{layout.n_phys_cols} x {layout.n_phys_rows}")

ok = True
ok &= case("source is 'row_flush'", layout.source == "row_flush")
ok &= case("7 cols x 13 rows", layout.n_phys_cols == 7
           and layout.n_phys_rows == 13)

# Group offsets back into rows by their y coordinate.
rows_by_y: dict[int, list[int]] = {}
for i, (oy, ox) in enumerate(layout.offsets):
    rows_by_y.setdefault(oy, []).append(ox)
row_widths_top_down = [
    len(rows_by_y[oy]) for oy in sorted(rows_by_y)
]
ok &= case(
    f"row widths top-down match {EXPECTED_ROW_WIDTHS_TOP_DOWN}",
    row_widths_top_down == EXPECTED_ROW_WIDTHS_TOP_DOWN,
)

# Within each row, columns must be flush (consecutive integers
# starting at the leading offset).
all_flush = True
all_centered = True
max_per_row = max(row_widths_top_down)
for oy, oxs in rows_by_y.items():
    cols = sorted(ox // TILE_W for ox in oxs)
    if cols != list(range(cols[0], cols[0] + len(cols))):
        all_flush = False
    expected_lead = (max_per_row - len(cols)) // 2
    if cols[0] != expected_lead:
        all_centered = False
ok &= case("each row is flush (consecutive cols)", all_flush)
ok &= case("each row is centered", all_centered)

ok &= case("61 unique offsets",
           len(set(layout.offsets)) == 61)
case_well3 = ok

# ----- Synthetic Well2: 17 cells × 4 visits -----
print("\nCase 2 — Well2 (17 cells × 4 visits, snake scan)")
stage_xy2 = []
m_indices2 = []
m_counter = 0
for r in range(17):
    y = r * Y_STEP
    for visit in range(4):
        # Same XY for each visit — visits should tiebreak by M.
        x = visit * (TILE_UM_X)  # 4 columns spaced by 1 tile width
        stage_xy2.append((x, y))
        m_indices2.append(m_counter)
        m_counter += 1

layout2 = compute_tile_layout(
    stage_xy_um=stage_xy2,
    pixel_size_um=PIXEL_UM,
    tile_h=TILE_H, tile_w=TILE_W,
    m_indices=m_indices2,
)
print(f"  source = {layout2.source!r}")
print(f"  n_phys_cols x n_phys_rows = "
      f"{layout2.n_phys_cols} x {layout2.n_phys_rows}")

ok2 = True
ok2 &= case("source is 'row_flush'", layout2.source == "row_flush")
ok2 &= case("4 cols x 17 rows",
            layout2.n_phys_cols == 4 and layout2.n_phys_rows == 17)
ok2 &= case("68 offsets total", len(layout2.offsets) == 68)
case_well2 = ok2

# ----- Diagonal 9-tile scan -----
print("\nCase 3 — Diagonal 9-tile scan (M=0..8 at increasing X *and* Y)")
stage_xy3 = [(i * X_STEP, i * Y_STEP) for i in range(9)]
m_indices3 = list(range(9))
layout3 = compute_tile_layout(
    stage_xy_um=stage_xy3,
    pixel_size_um=PIXEL_UM,
    tile_h=TILE_H, tile_w=TILE_W,
    m_indices=m_indices3,
)
print(f"  source = {layout3.source!r}")
print(f"  n_phys_cols x n_phys_rows = "
      f"{layout3.n_phys_cols} x {layout3.n_phys_rows}")
ok3 = True
ok3 &= case("source is 'row_flush'", layout3.source == "row_flush")
ok3 &= case("1 col x 9 rows",
            layout3.n_phys_cols == 1 and layout3.n_phys_rows == 9)
ok3 &= case("9 distinct row offsets",
            len({oy for (oy, _) in layout3.offsets}) == 9)
case_diag = ok3

# ----- Dense regular 4 × 17 mosaic -----
print("\nCase 4 — Dense regular 4 × 17 mosaic (every cell populated)")
stage_xy4 = []
m_indices4 = []
m_counter = 0
for r in range(17):
    for c in range(4):
        stage_xy4.append((c * TILE_UM_X, r * Y_STEP))
        m_indices4.append(m_counter)
        m_counter += 1
layout4 = compute_tile_layout(
    stage_xy_um=stage_xy4,
    pixel_size_um=PIXEL_UM,
    tile_h=TILE_H, tile_w=TILE_W,
    m_indices=m_indices4,
)
print(f"  source = {layout4.source!r}")
print(f"  n_phys_cols x n_phys_rows = "
      f"{layout4.n_phys_cols} x {layout4.n_phys_rows}")
ok4 = True
ok4 &= case("source is 'row_flush'", layout4.source == "row_flush")
ok4 &= case("4 cols x 17 rows",
            layout4.n_phys_cols == 4 and layout4.n_phys_rows == 17)
ok4 &= case("68 distinct offsets",
            len(set(layout4.offsets)) == 68)
case_dense = ok4

# ----- Summary -----
print("\n" + "=" * 50)
all_pass = all([case_well3, case_well2, case_diag, case_dense])
print(f"Well3 diamond: {'PASS' if case_well3 else 'FAIL'}")
print(f"Well2 4×17:    {'PASS' if case_well2 else 'FAIL'}")
print(f"Diagonal 9×1:  {'PASS' if case_diag else 'FAIL'}")
print(f"Dense 4×17:    {'PASS' if case_dense else 'FAIL'}")
print("=" * 50)
sys.exit(0 if all_pass else 1)
