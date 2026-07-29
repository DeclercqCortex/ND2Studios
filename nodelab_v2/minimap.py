"""The mini-map overlay — the Viewer's HUD home while the node canvas is maximized.

Maximizing the canvas (the ⛶ button in its top-right corner, ``Ctrl+Space``, or
View → *Maximize node canvas*) hands the whole centre to the graph and re-homes the
:class:`~nodelab_v2.viewer.ViewerPanel` into this frame: a small bordered window pinned
to the canvas's **top-left corner**, like a game mini-map, that keeps showing whatever
node you click — live.

The frame is chrome only; it never touches image state. It hosts the **real** Viewer
widget (reparented, not a copy), so channels, LUT, playback and overlays keep working
exactly as they do docked — :meth:`~nodelab_v2.viewer.ViewerPanel.set_compact` only
trims the control strip to fit. Drag the header to move it (it re-anchors to the
nearest corner, so a window resize keeps it where you put it), drag the bottom-right
grip to resize, and double-click the header — or press the dock button — to put the
Viewer back into the splitter.

Everything here is painted rather than icon-fonted (:class:`HudButton`, the corner
brackets, the grip), so the overlay renders identically on every platform and follows
the light/dark tokens through :meth:`MiniMapOverlay.restyle`.
"""
from __future__ import annotations

from typing import Optional, Tuple

from PySide6.QtCore import QEvent, QPoint, QPointF, QRect, QRectF, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPen
from PySide6.QtWidgets import (
    QAbstractButton, QLabel, QSizePolicy, QVBoxLayout, QWidget,
)

from nodelab_v2 import theme as T


class ElidedLabel(QLabel):
    """A QLabel that **elides** instead of forcing its full text width onto the layout.

    A plain QLabel's minimum size hint is its whole string, which would stop the
    mini-map from ever shrinking below the length of the Viewer's status line. This one
    reports no minimum width and paints an elided copy, keeping the full text in
    :meth:`text` (callers/tests read it) and in the tooltip.
    """

    def __init__(self, text: str = "", parent: Optional[QWidget] = None,
                 mode=Qt.ElideRight) -> None:
        super().__init__(text, parent)
        self._mode = mode
        self._col: Optional[QColor] = None
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)

    def setText(self, text: str) -> None:          # noqa: N802 — Qt override
        super().setText(text)
        self.setToolTip(text or "")

    def set_color(self, col: QColor) -> None:
        """Paint colour. Set explicitly (not via QSS) because this label paints itself,
        so the style sheet's ``color:`` never reaches the painter."""
        self._col = QColor(col)
        self.update()

    def paintEvent(self, _e) -> None:              # noqa: N802 — Qt override
        p = QPainter(self)
        fm = QFontMetrics(self.font())
        txt = fm.elidedText(self.text(), self._mode, max(0, self.width()))
        p.setPen(self._col or self.palette().color(self.foregroundRole()))
        p.drawText(self.rect(), self.alignment() | Qt.AlignVCenter, txt)
        p.end()


class HudButton(QAbstractButton):
    """A small painted HUD button — the canvas's maximize toggle and the mini-map's dock
    button. The glyph is drawn with QPainter (``kind``: ``"maximize"`` / ``"restore"`` /
    ``"dock"``) rather than taken from an icon font, so it never depends on a platform
    symbol font and it re-reads the theme tokens on every repaint."""

    def __init__(self, kind: str = "maximize", parent: Optional[QWidget] = None,
                 size: int = 26) -> None:
        super().__init__(parent)
        self._kind = kind
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setFixedSize(size, size)

    def set_kind(self, kind: str) -> None:
        if kind != self._kind:
            self._kind = kind
            self.update()

    @property
    def kind(self) -> str:
        return self._kind

    def enterEvent(self, e) -> None:               # noqa: N802 — Qt override
        self.update()
        super().enterEvent(e)

    def leaveEvent(self, e) -> None:               # noqa: N802 — Qt override
        self.update()
        super().leaveEvent(e)

    def paintEvent(self, _e) -> None:              # noqa: N802 — Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(0.5, 0.5, self.width() - 1.0, self.height() - 1.0)
        hot = self.underMouse()
        on = self.isDown() or (self.isCheckable() and self.isChecked())
        p.setPen(QPen(T.BORDER_HI if (hot or on) else T.alpha(T.BORDER, 190), 1))
        p.setBrush(T.ACCENT_DIM if on else (T.PANEL_HI if hot else T.alpha(T.PANEL, 235)))
        p.drawRoundedRect(r, 7, 7)

        col = T.ACCENT if (on or hot) else T.INK_2
        p.setPen(QPen(col, 1.7, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        p.setBrush(Qt.NoBrush)
        c = r.center()
        box = QRectF(c.x() - 6.5, c.y() - 5.5, 13.0, 11.0)
        if self._kind == "dock":
            # a pane with a filled top band — "put the Viewer back above the canvas"
            p.drawRoundedRect(box, 2.0, 2.0)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawRoundedRect(QRectF(box.left() + 1.2, box.top() + 1.2,
                                     box.width() - 2.4, 3.6), 1.2, 1.2)
            p.end()
            return
        arm = 3.6
        inward = self._kind == "restore"
        for (x, dx) in ((box.left(), 1.0), (box.right(), -1.0)):
            for (y, dy) in ((box.top(), 1.0), (box.bottom(), -1.0)):
                if inward:                          # brackets face the centre
                    x0, y0 = x + dx * arm, y + dy * arm
                    p.drawLine(QPointF(x0, y0), QPointF(x0 - dx * arm, y0))
                    p.drawLine(QPointF(x0, y0), QPointF(x0, y0 - dy * arm))
                else:                               # brackets open outward (fullscreen)
                    p.drawLine(QPointF(x, y), QPointF(x + dx * arm, y))
                    p.drawLine(QPointF(x, y), QPointF(x, y + dy * arm))
        p.end()


class MiniMapOverlay(QWidget):
    """A floating, draggable, resizable HUD frame hosting one content widget over its
    parent (the node canvas). See the module docstring for the interaction model."""

    HEADER_H = 23                 # the drag strip along the top (painted, not a widget)
    EDGE = 5                      # painted frame margin around the content
    GRIP = 16                     # bottom-right resize hot zone
    MIN_W, MIN_H = 220, 170
    DEFAULT_SIZE = (410, 330)

    #: live-dot colours by state — what the mini-map is doing right now
    _STATE_TOKEN = {"idle": "MUTED", "busy": "DIM2D", "live": "WIRE", "error": "ERROR"}

    restore_requested = Signal()

    def __init__(self, parent: QWidget) -> None:
        super().__init__(parent)
        self.setObjectName("minimap")
        self.setMouseTracking(True)
        self.setAutoFillBackground(False)
        self._content: Optional[QWidget] = None
        self._title = "viewer"
        self._state = "idle"
        self._anchor: Tuple[str, str] = ("left", "top")     # the corner it sticks to
        self._offset = QPoint(14, 14)                       # distance from that corner
        self._want = QSize(*self.DEFAULT_SIZE)              # size the user asked for
        self._drag: Optional[str] = None                    # 'move' | 'resize'
        self._drag_at = QPoint()
        self._drag_geo = QRect()

        self._lay = QVBoxLayout(self)
        self._lay.setContentsMargins(self.EDGE + 1, self.HEADER_H,
                                     self.EDGE + 1, self.EDGE + 1)
        self._lay.setSpacing(0)

        self._btn = HudButton("dock", self, size=19)
        self._btn.setToolTip("Dock the Viewer back above the canvas (Esc)")
        self._btn.clicked.connect(self.restore_requested.emit)

        self.resize(*self.DEFAULT_SIZE)
        self.hide()
        parent.installEventFilter(self)
        self.restyle()

    # ── content ────────────────────────────────────────────────────────────────
    def attach(self, widget: QWidget) -> None:
        """Re-home ``widget`` (the live ViewerPanel) into the frame."""
        if self._content is widget:
            return
        self.detach()
        self._content = widget
        self._lay.addWidget(widget, 1)
        widget.show()

    def detach(self) -> Optional[QWidget]:
        """Release the content widget (still parented here, hidden) so the caller can
        re-insert it into the splitter. Deliberately does NOT ``setParent(None)``: a
        momentary top-level would destroy and rebuild the Viewer's GL context."""
        w = self._content
        if w is not None:
            self._lay.removeWidget(w)
            w.hide()
            self._content = None
        return w

    @property
    def content(self) -> Optional[QWidget]:
        return self._content

    # ── header state ───────────────────────────────────────────────────────────
    def set_title(self, text: str) -> None:
        if text != self._title:
            self._title = text or "viewer"
            self.update()

    def set_state(self, state: str) -> None:
        """``idle`` / ``busy`` (pulling) / ``live`` (showing a result) / ``error``."""
        if state != self._state:
            self._state = state if state in self._STATE_TOKEN else "idle"
            self.update()

    @property
    def state(self) -> str:
        return self._state

    def _state_color(self) -> QColor:
        return getattr(T, self._STATE_TOKEN.get(self._state, "MUTED"))

    # ── geometry: anchored to the nearest corner of the parent ──────────────────
    def eventFilter(self, obj, ev) -> bool:        # noqa: N802 — Qt override
        if obj is self.parentWidget() and ev.type() == QEvent.Resize:
            self.reposition()
        return False

    def set_frame_size(self, w: int, h: int) -> None:
        """Set the frame's wanted size (what a grip drag records). It is re-clamped to
        the canvas on every :meth:`reposition`, so a cramped canvas squeezes the frame
        without forgetting the size to go back to."""
        self._want = QSize(max(self.MIN_W, int(w)), max(self.MIN_H, int(h)))
        self.reposition()

    def reposition(self) -> None:
        """Re-apply the stored corner anchor + offset, with the wanted size clamped
        inside the parent (so growing the canvas restores the full frame)."""
        par = self.parentWidget()
        if par is None:
            return
        w = max(self.MIN_W, min(self._want.width(), max(self.MIN_W, par.width() - 8)))
        h = max(self.MIN_H, min(self._want.height(), max(self.MIN_H, par.height() - 8)))
        if (w, h) != (self.width(), self.height()):
            self.resize(w, h)
        x = (self._offset.x() if self._anchor[0] == "left"
             else par.width() - w - self._offset.x())
        y = (self._offset.y() if self._anchor[1] == "top"
             else par.height() - h - self._offset.y())
        self.move(max(0, min(int(x), max(0, par.width() - w))),
                  max(0, min(int(y), max(0, par.height() - h))))

    def _reanchor(self) -> None:
        """Record which corner the frame now belongs to (its centre decides) and the
        offsets from it, so the next parent resize keeps it visually in place."""
        par = self.parentWidget()
        if par is None:
            return
        g = self.geometry()
        cx, cy = g.center().x(), g.center().y()
        horiz = "left" if cx <= par.width() / 2 else "right"
        vert = "top" if cy <= par.height() / 2 else "bottom"
        self._anchor = (horiz, vert)
        self._offset = QPoint(
            g.left() if horiz == "left" else par.width() - g.right() - 1,
            g.top() if vert == "top" else par.height() - g.bottom() - 1)

    def _grip_rect(self) -> QRectF:
        return QRectF(self.width() - self.GRIP, self.height() - self.GRIP,
                      self.GRIP, self.GRIP)

    def _header_rect(self) -> QRectF:
        return QRectF(0, 0, self.width(), self.HEADER_H)

    def resizeEvent(self, e) -> None:              # noqa: N802 — Qt override
        super().resizeEvent(e)
        self._btn.move(self.width() - self.EDGE - self._btn.width() - 1, self.EDGE - 2)

    # ── move / resize interaction ──────────────────────────────────────────────
    def mousePressEvent(self, e) -> None:          # noqa: N802 — Qt override
        pos = e.position().toPoint()
        if e.button() == Qt.LeftButton and self._grip_rect().contains(pos):
            self._drag = "resize"
        elif e.button() == Qt.LeftButton and self._header_rect().contains(pos):
            self._drag = "move"
            self.setCursor(Qt.ClosedHandCursor)
        else:
            super().mousePressEvent(e)
            return
        self._drag_at = e.globalPosition().toPoint()
        self._drag_geo = self.geometry()
        e.accept()

    def mouseMoveEvent(self, e) -> None:           # noqa: N802 — Qt override
        par = self.parentWidget()
        if self._drag is None or par is None:
            pos = e.position().toPoint()
            self.setCursor(Qt.SizeFDiagCursor if self._grip_rect().contains(pos)
                           else (Qt.OpenHandCursor if self._header_rect().contains(pos)
                                 else Qt.ArrowCursor))
            super().mouseMoveEvent(e)
            return
        d = e.globalPosition().toPoint() - self._drag_at
        g = QRect(self._drag_geo)
        if self._drag == "move":
            x = max(0, min(g.left() + d.x(), par.width() - g.width()))
            y = max(0, min(g.top() + d.y(), par.height() - g.height()))
            self.move(x, y)
        else:
            w = max(self.MIN_W, min(g.width() + d.x(), par.width() - g.left()))
            h = max(self.MIN_H, min(g.height() + d.y(), par.height() - g.top()))
            self.resize(w, h)
            self._want = QSize(w, h)
        self._reanchor()
        e.accept()

    def mouseReleaseEvent(self, e) -> None:        # noqa: N802 — Qt override
        if self._drag is not None:
            self._drag = None
            self.unsetCursor()
            self._reanchor()
            e.accept()
            return
        super().mouseReleaseEvent(e)

    def mouseDoubleClickEvent(self, e) -> None:    # noqa: N802 — Qt override
        if self._header_rect().contains(e.position().toPoint()):
            self.restore_requested.emit()
            e.accept()
            return
        super().mouseDoubleClickEvent(e)

    # ── paint: the mini-map chrome ─────────────────────────────────────────────
    def paintEvent(self, _e) -> None:              # noqa: N802 — Qt override
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing, True)
        r = QRectF(1, 1, self.width() - 2, self.height() - 2)
        p.setPen(Qt.NoPen)
        p.setBrush(T.BODY)
        p.drawRoundedRect(r, 10, 10)
        p.setPen(QPen(T.BORDER_HI, 2))
        p.setBrush(Qt.NoBrush)
        p.drawRoundedRect(r, 10, 10)
        # accent corner brackets — the "HUD / mini-map" read
        p.setPen(QPen(T.alpha(T.ACCENT, 205), 2, Qt.SolidLine, Qt.FlatCap))
        arm = 15.0
        for x, dx in ((r.left() + 1, 1.0), (r.right() - 1, -1.0)):
            for y, dy in ((r.top() + 1, 1.0), (r.bottom() - 1, -1.0)):
                p.drawLine(QPointF(x + dx * 8, y), QPointF(x + dx * (8 + arm), y))
                p.drawLine(QPointF(x, y + dy * 8), QPointF(x, y + dy * (8 + arm)))

        # header: live dot + the previewed node, elided against the dock button
        cy = self.EDGE + 6.0
        p.setPen(Qt.NoPen)
        p.setBrush(self._state_color())
        p.drawEllipse(QPointF(self.EDGE + 8.0, cy), 3.6, 3.6)
        f = QFont(T.SANS, 7)
        f.setBold(True)
        f.setCapitalization(QFont.AllUppercase)
        f.setLetterSpacing(QFont.PercentageSpacing, 108)
        p.setFont(f)
        p.setPen(T.INK_2)
        left = self.EDGE + 16.0
        right = float(self._btn.x() - 5)
        if right > left:
            txt = QFontMetrics(f).elidedText(self._title, Qt.ElideRight,
                                             int(right - left))
            p.drawText(QRectF(left, self.EDGE - 3, right - left, self.HEADER_H - 6),
                       Qt.AlignVCenter | Qt.AlignLeft, txt)

        # resize grip — three diagonal ticks in the bottom-right corner
        p.setPen(QPen(T.alpha(T.MUTED, 210), 1.4))
        g = self._grip_rect().adjusted(3, 3, -3, -3)
        for k in (0.0, 4.5, 9.0):
            p.drawLine(QPointF(g.right() - k, g.bottom()),
                       QPointF(g.right(), g.bottom() - k))
        p.end()

    # ── styling ────────────────────────────────────────────────────────────────
    def restyle(self) -> None:
        self.update()
        self._btn.update()


__all__ = ["MiniMapOverlay", "HudButton", "ElidedLabel"]
