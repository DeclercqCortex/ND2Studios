"""Palette + metrics for NodeLab v2 (dark + light; mirrors the approved mockup).

The NodeLab identity — the cyan accent + amber-2D / cyan-3D switch — is preserved in
both modes. The color TOKENS (``BG``, ``PANEL``, ``INK``, …) are module globals so node
cards read them at paint time; :func:`apply` rebinds them to the dark or light palette
and repainting/restyling picks the change up (G9). Socket-type colors come straight
from :data:`nodegraph.sockets.SOCKET_COLOR` (theme-independent — the data model owns
them). A widget with a cached QSS string re-reads the tokens via its own ``restyle()``.
"""
from __future__ import annotations

from PySide6.QtGui import QColor

from nodegraph.domains import Domain, domain_color
from nodegraph.sockets import SOCKET_COLOR, SocketType


def c(hex_: str) -> QColor:
    return QColor(hex_)


# ── palettes ────────────────────────────────────────────────────────────────────
_DARK = {
    "BG": "#14171c", "GRID_DOT": "#2b323c", "PANEL": "#232a33", "PANEL_HI": "#2a323d",
    "BODY": "#1e242c", "BORDER": "#39424f", "BORDER_HI": "#4c5867",
    "INK": "#c9d2de", "INK_2": "#93a0b1", "MUTED": "#6f7c8d",
    "ACCENT": "#6fe3ff", "ACCENT_INK": "#06222b", "ACCENT_DIM": "#284a56",
    "DIM2D": "#e0a13a", "DIM2D_INK": "#2a1c03", "WIRE": "#5fd06a", "ERROR": "#e05a5a",
}
_LIGHT = {
    "BG": "#eef1f5", "GRID_DOT": "#d2d9e2", "PANEL": "#ffffff", "PANEL_HI": "#f4f7fa",
    "BODY": "#eff2f7", "BORDER": "#c7d0db", "BORDER_HI": "#a9b4c2",
    "INK": "#1b232e", "INK_2": "#48555f", "MUTED": "#7a8794",
    "ACCENT": "#0f8fb2", "ACCENT_INK": "#e6feff", "ACCENT_DIM": "#bce4ef",
    "DIM2D": "#b9770c", "DIM2D_INK": "#fff2d8", "WIRE": "#2c9a37", "ERROR": "#c23b3b",
}
_PALETTES = {"dark": _DARK, "light": _LIGHT}

#: the color-token names (kept in sync between the two palettes).
_TOKENS = tuple(_DARK)

MODE = "dark"

# module-level color globals (rebound by apply); declared here so imports resolve.
BG = GRID_DOT = PANEL = PANEL_HI = BODY = BORDER = BORDER_HI = None       # type: ignore
INK = INK_2 = MUTED = ACCENT = ACCENT_INK = ACCENT_DIM = None             # type: ignore
DIM2D = DIM2D_INK = WIRE = ERROR = None                                   # type: ignore


def apply(mode: str) -> None:
    """Rebind the color-token globals to the ``"dark"`` or ``"light"`` palette. Node
    cards repaint from the new tokens; QSS widgets call their own ``restyle()``."""
    global MODE
    MODE = mode if mode in _PALETTES else "dark"
    g = globals()
    for name in _TOKENS:
        g[name] = c(_PALETTES[MODE][name])


apply("dark")

# socket dot colors, straight from the headless model (theme-independent)
SOCKET = {t: c(SOCKET_COLOR[t]) for t in SocketType}

# domain rail/wire colors, straight from the headless model (theme-independent) — the
# socket domain-rail chips + the wire tint that communicates which domain(s) transfer.
# Goes through ``domain_color()`` (which falls back to Global grey), NOT ``DOMAIN_COLOR[d]``:
# this is the only site in the GUI that iterates the whole ``Domain`` enum, so a subscript
# here turned a new domain member without a color entry into a KeyError at *import* time,
# which took the entire NodeLab window down before it could draw anything.
DOMAIN = {d: c(domain_color(d)) for d in Domain}


def domain_qcolor(domain: Domain) -> QColor:
    return DOMAIN.get(domain, MUTED)

# granularity chip colors (by read-footprint "cost") — read on both themes via alpha
GRAN = {
    "tileable": c("#3aa64a"), "whole_plane": c("#c58f1e"),
    "whole_volume": c("#c85a3a"), "whole_series": c("#a24aa4"),
    "multi_view": c("#3a80c0"),
}

# category header strip colors
CATEGORY = {
    "enhancement": c("#c98a5a"), "analysis": c("#5a8ac9"), "channel": c("#4bb8c0"),
    "io": c("#8a93a1"), "detection": c("#c264a0"), "general": c("#8a93a1"),
    "group": c("#9b7bd4"),
}


def emission_qcolor(nm) -> QColor:
    """An approximate sRGB for a visible emission wavelength ``nm`` (Bruton's
    piecewise map, gamma 0.8). Used to tint a channel by the colour closest to its
    emission spectrum. ``None`` / non-finite / out-of-visible (e.g. a transmitted-light
    channel) → a neutral grey so the channel is still legible."""
    try:
        w = float(nm)
    except (TypeError, ValueError):
        return QColor(196, 200, 208)
    if not (380.0 <= w <= 780.0):
        # near-IR still reads as deep red; anything else is greyscale
        if 780.0 < w <= 900.0:
            r, g, b = 0.35, 0.0, 0.0
        else:
            return QColor(196, 200, 208)
    elif w < 440:
        r, g, b = -(w - 440) / 60.0, 0.0, 1.0
    elif w < 490:
        r, g, b = 0.0, (w - 440) / 50.0, 1.0
    elif w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20.0
    elif w < 580:
        r, g, b = (w - 510) / 70.0, 1.0, 0.0
    elif w < 645:
        r, g, b = 1.0, -(w - 645) / 65.0, 0.0
    else:
        r, g, b = 1.0, 0.0, 0.0
    # intensity roll-off at the spectrum edges
    if w < 420:
        f = 0.3 + 0.7 * (w - 380) / 40.0
    elif w > 700:
        f = 0.3 + 0.7 * (780 - w) / 80.0
    else:
        f = 1.0
    g_ = 0.8
    to8 = lambda c: 0 if c <= 0 else min(255, int(round(255 * (c * f) ** g_)))
    return QColor(to8(r), to8(g), to8(b))


def category_color(name: str) -> QColor:
    return CATEGORY.get(name, CATEGORY["general"])


def gran_color(name: str) -> QColor:
    return GRAN.get(name, MUTED)


def menu_qss() -> str:
    """Styling for a **parentless** popup :class:`QMenu` (the canvas context menu). The
    window stylesheet can't reach it — a popup created without a parent widget is not in
    the window's widget tree — so it carries its own copy of the same look."""
    return f"""
    QMenu {{ background:{PANEL.name()}; color:{INK.name()};
        border:1px solid {BORDER.name()}; border-radius:9px; padding:5px; }}
    QMenu::item {{ padding:6px 26px 6px 14px; border-radius:6px; }}
    QMenu::item:selected {{ background:{ACCENT_DIM.name()}; }}
    QMenu::item:disabled {{ color:{MUTED.name()}; font-weight:800; }}
    QMenu::separator {{ height:1px; background:{BORDER.name()}; margin:5px 10px; }}
    QMenu::indicator {{ width:13px; height:13px; margin-left:6px; border-radius:3px;
        border:1px solid {BORDER_HI.name()}; background:{BODY.name()}; }}
    QMenu::indicator:checked {{ background:{ACCENT.name()};
        border-color:{ACCENT.name()}; }}
    """


def controls_qss() -> str:
    """Shared modern styling for the common interactive controls — buttons, checkboxes,
    combo/line/spin edits, list/table views, and thin rounded scrollbars. Panels append
    this in their ``restyle()`` so every dock reads as one system, and it re-reads the
    live tokens on each call so it tracks the light/dark switch (G9)."""
    return f"""
    QPushButton {{ background:{BODY.name()}; color:{INK.name()};
        border:1px solid {BORDER.name()}; border-radius:7px;
        padding:5px 13px; font-weight:600; }}
    QPushButton:hover {{ background:{PANEL_HI.name()}; border-color:{BORDER_HI.name()}; }}
    QPushButton:pressed {{ background:{ACCENT_DIM.name()}; border-color:{ACCENT.name()}; }}
    QPushButton:disabled {{ color:{MUTED.name()}; background:{BODY.name()}; }}
    QCheckBox {{ color:{INK_2.name()}; spacing:6px; font-size:12px; }}
    QCheckBox::indicator {{ width:15px; height:15px; border-radius:4px;
        border:1px solid {BORDER_HI.name()}; background:{BODY.name()}; }}
    QCheckBox::indicator:hover {{ border-color:{ACCENT.name()}; }}
    QCheckBox::indicator:checked {{ background:{ACCENT.name()};
        border-color:{ACCENT.name()}; }}
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{ background:{BODY.name()};
        color:{INK.name()}; border:1px solid {BORDER.name()}; border-radius:6px;
        padding:4px 8px; selection-background-color:{ACCENT_DIM.name()}; }}
    QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color:{BORDER_HI.name()}; }}
    QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color:{ACCENT.name()}; }}
    QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled {{ color:{MUTED.name()}; background:{BG.name()};
        border-color:{BORDER.name()}; }}
    QCheckBox:disabled {{ color:{MUTED.name()}; }}
    QCheckBox::indicator:disabled {{ border-color:{BORDER.name()};
        background:{BG.name()}; }}
    QComboBox::drop-down {{ border:0; width:16px; }}
    QComboBox QAbstractItemView {{ background:{PANEL.name()}; color:{INK.name()};
        border:1px solid {BORDER.name()}; border-radius:6px; outline:0;
        selection-background-color:{ACCENT_DIM.name()};
        selection-color:{INK.name()}; padding:2px; }}
    QListWidget, QTreeWidget {{ background:{BG.name()}; color:{INK.name()};
        border:1px solid {BORDER.name()}; border-radius:7px; outline:0; padding:2px; }}
    QListWidget::item {{ border-radius:5px; padding:3px 6px; }}
    QListWidget::item:selected {{ background:{ACCENT_DIM.name()}; color:{INK.name()}; }}
    QListWidget::item:hover {{ background:{PANEL_HI.name()}; }}
    QScrollBar:vertical {{ background:transparent; width:10px; margin:2px; }}
    QScrollBar::handle:vertical {{ background:{BORDER_HI.name()}; border-radius:4px;
        min-height:28px; }}
    QScrollBar::handle:vertical:hover {{ background:{MUTED.name()}; }}
    QScrollBar:horizontal {{ background:transparent; height:10px; margin:2px; }}
    QScrollBar::handle:horizontal {{ background:{BORDER_HI.name()}; border-radius:4px;
        min-width:28px; }}
    QScrollBar::handle:horizontal:hover {{ background:{MUTED.name()}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width:0; height:0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background:transparent; }}
    QToolTip {{ background:{PANEL.name()}; color:{INK.name()};
        border:1px solid {BORDER.name()}; padding:4px 7px; }}
    """


def mix(a: QColor, b: QColor, t: float) -> QColor:
    """Linear blend ``a*(1-t) + b*t``."""
    return QColor(round(a.red() * (1 - t) + b.red() * t),
                  round(a.green() * (1 - t) + b.green() * t),
                  round(a.blue() * (1 - t) + b.blue() * t))


def alpha(col: QColor, a: int) -> QColor:
    q = QColor(col)
    q.setAlpha(a)
    return q


# ── metrics ───────────────────────────────────────────────────────────────────
NODE_W = 214
RR_SIZE = 22          # a reroute node's compact dot (width == height)
HEADER_H = 36
ROW_H = 26
GRAN_H = 28
PAD_TOP = 8
PAD_BOTTOM = 10
RADIUS = 7
PROG_H = 2            # the per-node progress rail riding the header's bottom edge
DOT_R = 3.2           # the header status dot (running pulse / done / cached / error)
CLOSE_BTN = 15        # the hover ✕ delete button in a card's header
SANS = "Segoe UI"
MONO = "Consolas"
