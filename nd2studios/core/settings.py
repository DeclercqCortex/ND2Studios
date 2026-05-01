"""
Application constants, Dracula palette, page list, sidebar/titlebar config.

All UI dimensions and colors live here so theming and layout tweaks happen
in one place.
"""
from __future__ import annotations


class Settings:
    APP_NAME = "ND2Studios"
    APP_VERSION = "1.0.0"
    APP_DESCRIPTION = "ND2 file processing and export — McGhee Lab"

    # Window
    MIN_WIDTH = 1280
    MIN_HEIGHT = 820

    # Sidebar dimensions (PyDracula-style collapsible)
    SIDEBAR_COLLAPSED_WIDTH = 60
    SIDEBAR_EXPANDED_WIDTH = 240
    SIDEBAR_ANIMATION_MS = 280  # InOutQuart easing

    # Top title bar
    TITLE_BAR_HEIGHT = 40
    BOTTOM_BAR_HEIGHT = 28

    # Drop shadow on the app background frame (PyDracula chrome)
    SHADOW_BLUR_RADIUS = 17
    SHADOW_OFFSET_X = 0
    SHADOW_OFFSET_Y = 0
    SHADOW_ALPHA = 150  # 0–255

    # Custom title bar drag-resize grip thickness
    GRIP_SIZE = 10

    # ── Dracula palette (matches CellTracker + Modern_GUI_PyDracula) ──
    BG_PRIMARY = "#282a36"      # main background
    BG_SECONDARY = "#21252b"    # sidebar / top + bottom bars
    BG_TERTIARY = "#2c313c"     # active nav button background
    BG_HOVER = "#343b48"        # hover state
    FG_PRIMARY = "#f8f8f2"      # default text
    FG_SECONDARY = "#b0b0b0"    # subdued text
    ACCENT_PURPLE = "#bd93f9"   # primary accent (CTA, focus, default button)
    ACCENT_PINK = "#ff79c6"     # secondary accent (gradients, highlights)
    ACCENT_GREEN = "#50fa7b"    # success
    ACCENT_CYAN = "#8be9fd"     # info
    ACCENT_ORANGE = "#ffb86c"   # warning
    ACCENT_RED = "#ff5555"      # danger
    ACCENT_YELLOW = "#f1fa8c"
    BORDER_COLOR = "#44475a"

    # ── Page list ──────────────────────────────────────────────────
    # Tuple format: (key, icon_unicode, label, tooltip)
    PAGES = [
        ("import", "📂", "Import", "Load an ND2 or TIFF file and inspect metadata."),
        ("recipe", "🧪", "Recipe", "Build a processing pipeline (trial / accept / reject)."),
        ("export", "💾", "Export", "Export TIFF stacks, RGB composites, and movies."),
    ]

    # Status ladder. Forward navigation can require a minimum status.
    STATUS_ORDER = ["new", "imported", "preprocessed", "ready_to_export"]

    # Per-page minimum status to enter (None = always accessible)
    PAGE_PREREQS = {
        "import": None,
        "recipe": ("imported", "Import a file first (Page 1)."),
        "export": ("imported", "Import a file first (Page 1)."),
    }
