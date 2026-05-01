"""
``LutHistogramWidget`` — per-channel manual contrast control.

A compact widget that shows a histogram of pixel intensities sampled
from a (T, H, W) timeseries and lets the user drag two vertical
handles to set the displayed contrast bounds (``lo``, ``hi``). A small
gamma slider adjusts the curve. ``Auto`` snaps to the 0.5–99.5
percentile bounds; ``Reset`` snaps to the dtype's natural full range.

The widget emits ``contrast_changed(lo: float, hi: float, gamma: float)``
on every interactive change. The viewer subscribes and recomposes the
displayed image immediately.

Performance note: histograms are computed by sampling at most
``MAX_SAMPLE_FRAMES`` frames from the timeseries — never the whole
stack, which can be tens of GB. Sampling happens lazily and is cheap
to recompute.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout, QWidget,
)

from nd2studios.core.settings import Settings
from nd2studios.widgets.common import MplCanvas


MAX_SAMPLE_FRAMES = 32
NUM_HIST_BINS = 256


class LutHistogramWidget(QWidget):
    """Histogram + min/max + gamma for a single channel."""

    contrast_changed = Signal(float, float, float)  # lo, hi, gamma

    def __init__(self, channel_name: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.channel_name = channel_name
        self._lo: float = 0.0
        self._hi: float = 65535.0
        self._gamma: float = 1.0
        self._dtype_max: float = 65535.0
        self._hist_counts: Optional[np.ndarray] = None
        self._hist_edges: Optional[np.ndarray] = None
        self._dragging: Optional[str] = None  # 'lo' or 'hi'

        self._build_ui()

    # ── UI ──
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        # Tiny histogram canvas. Height kept small so a row of these
        # under the viewer doesn't dominate the screen.
        self.canvas = MplCanvas(self, width=3.5, height=0.9, dpi=90)
        self.canvas.setMinimumHeight(80)
        self.canvas.setMaximumHeight(110)
        self.ax = self.canvas.add_subplot(111)
        self.ax.set_yticks([])
        self.ax.set_xticks([])
        for s in self.ax.spines.values():
            s.set_color(Settings.BORDER_COLOR)
        layout.addWidget(self.canvas)

        # Wire matplotlib mouse events for dragging the handles.
        self.canvas.mpl_connect("button_press_event", self._on_press)
        self.canvas.mpl_connect("motion_notify_event", self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)

        # Numeric + gamma row.
        row = QHBoxLayout()
        row.setSpacing(4)
        row.setContentsMargins(2, 0, 2, 0)

        self.spin_lo = QDoubleSpinBox()
        self.spin_lo.setRange(0.0, 1e9)
        self.spin_lo.setDecimals(0)
        self.spin_lo.setMaximumWidth(80)
        self.spin_lo.valueChanged.connect(self._on_spin_changed)
        row.addWidget(self.spin_lo)

        row.addWidget(QLabel("…"))

        self.spin_hi = QDoubleSpinBox()
        self.spin_hi.setRange(0.0, 1e9)
        self.spin_hi.setDecimals(0)
        self.spin_hi.setMaximumWidth(80)
        self.spin_hi.valueChanged.connect(self._on_spin_changed)
        row.addWidget(self.spin_hi)

        row.addWidget(QLabel("γ"))
        self.slider_gamma = QSlider(Qt.Orientation.Horizontal)
        self.slider_gamma.setRange(20, 500)  # 0.20 .. 5.00 with /100
        self.slider_gamma.setValue(100)
        self.slider_gamma.setFixedWidth(80)
        self.slider_gamma.valueChanged.connect(self._on_gamma_changed)
        row.addWidget(self.slider_gamma)

        self.btn_auto = QPushButton("Auto")
        self.btn_auto.setFixedWidth(48)
        self.btn_auto.clicked.connect(self._on_auto)
        row.addWidget(self.btn_auto)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.setFixedWidth(54)
        self.btn_reset.clicked.connect(self._on_reset)
        row.addWidget(self.btn_reset)

        row.addStretch(1)
        layout.addLayout(row)

    # ── Public API ──
    def set_data(self, timeseries: "np.ndarray | object",
                 dtype: Optional[np.dtype] = None) -> None:
        """Compute the histogram from a sampled subset of the timeseries.

        ``timeseries`` can be an ndarray or any object supporting
        ``__len__`` and ``__getitem__`` (i.e. ``LazyND2Channel``).
        """
        if timeseries is None:
            return
        try:
            n = len(timeseries)
        except TypeError:
            n = getattr(timeseries, "shape", (0,))[0]
        if n == 0:
            return

        # Sample at most MAX_SAMPLE_FRAMES evenly across T.
        n_sample = min(MAX_SAMPLE_FRAMES, n)
        idxs = np.linspace(0, n - 1, n_sample, dtype=int)
        samples = []
        for i in idxs:
            try:
                samples.append(np.asarray(timeseries[int(i)]).ravel())
            except Exception:
                continue
        if not samples:
            return
        flat = np.concatenate(samples)

        # Determine the natural dtype range (for the Reset button).
        sample_dtype = dtype if dtype is not None else flat.dtype
        if np.issubdtype(sample_dtype, np.integer):
            info = np.iinfo(sample_dtype)
            self._dtype_max = float(info.max)
        else:
            self._dtype_max = float(flat.max())

        self.spin_lo.setRange(0.0, max(1.0, self._dtype_max))
        self.spin_hi.setRange(0.0, max(1.0, self._dtype_max))

        # Histogram: log scale on counts because most fluorescence images
        # are dominated by background pixels.
        edges = np.linspace(0.0, max(1.0, float(flat.max())), NUM_HIST_BINS + 1)
        counts, _ = np.histogram(flat, bins=edges)
        self._hist_counts = np.log1p(counts.astype(np.float32))
        self._hist_edges = edges

        # Initialize lo/hi to the 0.5..99.5 percentile if not yet set.
        if self._lo == 0.0 and self._hi == self._dtype_max:
            lo, hi = np.percentile(flat, [0.5, 99.5])
            self._lo = float(lo)
            self._hi = float(max(hi, lo + 1))

        self._sync_spinboxes()
        self._redraw()

    def set_contrast(self, lo: float, hi: float, gamma: float) -> None:
        """Programmatic set (e.g. on session load) without re-emitting."""
        self._lo = float(lo)
        self._hi = float(hi)
        self._gamma = float(gamma)
        self._sync_spinboxes()
        self.slider_gamma.blockSignals(True)
        self.slider_gamma.setValue(int(round(self._gamma * 100)))
        self.slider_gamma.blockSignals(False)
        self._redraw()

    def get_contrast(self) -> Tuple[float, float, float]:
        return self._lo, self._hi, self._gamma

    # ── Drawing ──
    def _redraw(self) -> None:
        if self._hist_counts is None:
            return
        ax = self.ax
        ax.clear()
        ax.set_yticks([])
        ax.set_xticks([])
        for s in ax.spines.values():
            s.set_color(Settings.BORDER_COLOR)

        edges = self._hist_edges
        centers = 0.5 * (edges[:-1] + edges[1:])
        ax.fill_between(centers, 0, self._hist_counts,
                        color=Settings.BORDER_COLOR, alpha=0.7)
        ax.set_xlim(edges[0], edges[-1])
        # Vertical handles.
        ax.axvline(self._lo, color=Settings.ACCENT_CYAN, linewidth=1.4)
        ax.axvline(self._hi, color=Settings.ACCENT_PINK, linewidth=1.4)
        # Gamma curve overlay (in [lo, hi] mapped to [0, max_count]).
        if self._hi > self._lo:
            xs = np.linspace(self._lo, self._hi, 64)
            ys_norm = ((xs - self._lo) / (self._hi - self._lo)) ** (1.0 / self._gamma)
            ys = ys_norm * float(self._hist_counts.max())
            ax.plot(xs, ys, color=Settings.ACCENT_PURPLE, linewidth=1.2, alpha=0.9)
        self.canvas.draw_idle()

    # ── Mouse handlers (drag the lo/hi vertical lines) ──
    def _on_press(self, event):
        if event.inaxes is not self.ax or event.xdata is None:
            return
        x = event.xdata
        # Pick whichever line is closer.
        d_lo = abs(x - self._lo)
        d_hi = abs(x - self._hi)
        self._dragging = "lo" if d_lo <= d_hi else "hi"
        self._move_to(x)

    def _on_motion(self, event):
        if self._dragging is None or event.inaxes is not self.ax:
            return
        if event.xdata is None:
            return
        self._move_to(event.xdata)

    def _on_release(self, _event):
        self._dragging = None

    def _move_to(self, x: float) -> None:
        if self._dragging == "lo":
            self._lo = max(0.0, min(x, self._hi - 1.0))
        elif self._dragging == "hi":
            self._hi = max(self._lo + 1.0, x)
        self._sync_spinboxes()
        self._redraw()
        self._emit()

    # ── Buttons / spinboxes ──
    def _on_spin_changed(self, _v: float) -> None:
        lo = float(self.spin_lo.value())
        hi = float(self.spin_hi.value())
        if hi <= lo:
            hi = lo + 1.0
            self.spin_hi.blockSignals(True)
            self.spin_hi.setValue(hi)
            self.spin_hi.blockSignals(False)
        self._lo, self._hi = lo, hi
        self._redraw()
        self._emit()

    def _on_gamma_changed(self, raw: int) -> None:
        self._gamma = max(0.05, raw / 100.0)
        self._redraw()
        self._emit()

    def _on_auto(self) -> None:
        if self._hist_counts is None:
            return
        # Recompute percentile bounds from the histogram CDF.
        counts = np.expm1(self._hist_counts)  # invert log1p
        cdf = np.cumsum(counts) / max(counts.sum(), 1)
        edges = self._hist_edges
        lo_idx = int(np.searchsorted(cdf, 0.005))
        hi_idx = int(np.searchsorted(cdf, 0.995))
        lo_idx = max(0, min(lo_idx, len(edges) - 2))
        hi_idx = max(lo_idx + 1, min(hi_idx, len(edges) - 1))
        self._lo = float(edges[lo_idx])
        self._hi = float(edges[hi_idx])
        self._sync_spinboxes()
        self._redraw()
        self._emit()

    def _on_reset(self) -> None:
        self._lo = 0.0
        self._hi = self._dtype_max
        self._gamma = 1.0
        self.slider_gamma.blockSignals(True)
        self.slider_gamma.setValue(100)
        self.slider_gamma.blockSignals(False)
        self._sync_spinboxes()
        self._redraw()
        self._emit()

    # ── Helpers ──
    def _sync_spinboxes(self) -> None:
        self.spin_lo.blockSignals(True)
        self.spin_lo.setValue(self._lo)
        self.spin_lo.blockSignals(False)
        self.spin_hi.blockSignals(True)
        self.spin_hi.setValue(self._hi)
        self.spin_hi.blockSignals(False)

    def _emit(self) -> None:
        self.contrast_changed.emit(self._lo, self._hi, self._gamma)


def apply_lut(frame: np.ndarray, lo: float, hi: float, gamma: float) -> np.ndarray:
    """Map ``frame`` to uint8 [0, 255] using the LUT bounds + gamma.

    Used by the multi-axis viewer to compose a frame with manual
    contrast settings instead of the auto-percentile path.
    """
    f = frame.astype(np.float32)
    if hi <= lo:
        hi = lo + 1.0
    f = (f - lo) / (hi - lo)
    f = np.clip(f, 0.0, 1.0)
    if abs(gamma - 1.0) > 1e-3:
        f = f ** (1.0 / max(gamma, 0.05))
    return (f * 255).astype(np.uint8)
