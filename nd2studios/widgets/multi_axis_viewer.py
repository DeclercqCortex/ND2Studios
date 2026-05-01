"""
``MultiAxisViewer`` — V1.1 viewer with M, T, Z sliders, per-channel
toggle/color/LUT, and additive RGB compositing.

The viewer reads frames from a ``LazyND2Volume`` (when M/Z scrolling
is active) *or* from a dict of pre-collapsed `(T, H, W)` channel
arrays (when the user has accepted a fixed M and Z mode and is
working downstream). Both paths converge on the same compositor.

Differences from the V1.0 :class:`ImageViewer`:

* M and Z sliders alongside T (M slider hides when M==1, Z when Z==1
  *or* the user has chosen a Z projection mode that collapses the axis).
* Per-channel control row: enable checkbox, color combo, embedded
  :class:`LutHistogramWidget`. Toggling any of these recomposes
  immediately — no Confirm step.
* Manual contrast (LUT) replaces auto-percentile by default; ``Auto``
  inside each LUT widget snaps back to percentile bounds.

Compositing math is identical to V1.0: per-channel ``apply_lut`` →
multiply by per-channel RGB color → additive sum → clip to uint8.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFrame, QHBoxLayout, QLabel, QSlider,
    QVBoxLayout, QWidget,
)

from nd2studios.backend.nd2_volume import LazyND2Volume
from nd2studios.core.settings import Settings
from nd2studios.widgets.image_viewer import (
    CHANNEL_COLORS, ImageCanvas, ZoomToolbar,
)
from nd2studios.widgets.lut_histogram import LutHistogramWidget, apply_lut


class ChannelControlRow(QWidget):
    """One row per channel: enable / color / LUT histogram."""

    state_changed = Signal()  # toggle or color
    contrast_changed = Signal(float, float, float)  # forwarded from LUT

    def __init__(self, name: str, color_default: str = "gray",
                 enabled: bool = True, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.name = name
        outer = QHBoxLayout(self)
        outer.setContentsMargins(2, 2, 2, 2)
        outer.setSpacing(8)

        self.cb = QCheckBox(name)
        self.cb.setChecked(enabled)
        self.cb.setMinimumWidth(120)
        self.cb.stateChanged.connect(lambda _s: self.state_changed.emit())
        outer.addWidget(self.cb)

        self.combo_color = QComboBox()
        self.combo_color.addItems(list(CHANNEL_COLORS.keys()))
        if color_default in CHANNEL_COLORS:
            self.combo_color.setCurrentText(color_default)
        self.combo_color.setFixedWidth(96)
        self.combo_color.currentTextChanged.connect(
            lambda _t: self.state_changed.emit())
        outer.addWidget(self.combo_color)

        self.lut = LutHistogramWidget(name)
        self.lut.contrast_changed.connect(
            lambda lo, hi, g: self.contrast_changed.emit(lo, hi, g))
        outer.addWidget(self.lut, stretch=1)

    @property
    def enabled(self) -> bool:
        return self.cb.isChecked()

    @property
    def color_name(self) -> str:
        return self.combo_color.currentText()

    @property
    def color_rgb(self) -> Tuple[int, int, int]:
        return CHANNEL_COLORS.get(self.color_name, (255, 255, 255))


class MultiAxisViewer(QWidget):
    """Image viewer with M, T, Z sliders + per-channel control strip."""

    # Emitted when the user navigates so the page can keep state.
    coords_changed = Signal(int, int, int)   # m, t, z
    channels_changed = Signal()              # any channel toggle/color/LUT

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._volume: Optional[LazyND2Volume] = None
        self._channels: Dict[str, Any] = {}   # used when no volume; (T,H,W) per channel
        self._channel_rows: List[ChannelControlRow] = []
        self._z_mode: str = "max"
        self._z_index: int = 0

        # Coordinate state.
        self._m: int = 0
        self._t: int = 0
        self._z: int = 0

        self._build_ui()

    # ── UI ──
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        # Image canvas + zoom toolbar (reusing V1.0 widgets).
        self.canvas = ImageCanvas()
        layout.addWidget(self.canvas, stretch=1)

        zoom_row = QHBoxLayout()
        zoom_row.setContentsMargins(4, 0, 4, 0)
        self.zoom_toolbar = ZoomToolbar(self.canvas)
        zoom_row.addWidget(self.zoom_toolbar)
        zoom_row.addStretch(1)
        layout.addLayout(zoom_row)

        # M / T / Z sliders.
        slider_box = QFrame()
        slider_box.setObjectName("contentArea")
        slider_layout = QVBoxLayout(slider_box)
        slider_layout.setContentsMargins(8, 4, 8, 4)
        slider_layout.setSpacing(2)

        self._m_row, self.m_slider, self.m_label = self._make_axis_row("M")
        self._t_row, self.t_slider, self.t_label = self._make_axis_row("T")
        self._z_row, self.z_slider, self.z_label = self._make_axis_row("Z")
        self.m_slider.valueChanged.connect(self._on_m_changed)
        self.t_slider.valueChanged.connect(self._on_t_changed)
        self.z_slider.valueChanged.connect(self._on_z_changed)
        slider_layout.addLayout(self._m_row)
        slider_layout.addLayout(self._t_row)
        slider_layout.addLayout(self._z_row)
        layout.addWidget(slider_box)

        # Per-channel control strip — dynamically populated.
        self._channels_box = QFrame()
        self._channels_box.setObjectName("contentArea")
        self._channels_layout = QVBoxLayout(self._channels_box)
        self._channels_layout.setContentsMargins(4, 4, 4, 4)
        self._channels_layout.setSpacing(2)
        layout.addWidget(self._channels_box)

    def _make_axis_row(self, label: str):
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)
        lbl = QLabel(label + ":")
        lbl.setFixedWidth(20)
        row.addWidget(lbl)
        slider = QSlider(Qt.Orientation.Horizontal)
        slider.setRange(0, 0)
        row.addWidget(slider, stretch=1)
        info = QLabel("0/0")
        info.setMinimumWidth(60)
        row.addWidget(info)
        return row, slider, info

    # ── Population ──
    def set_volume(self, volume: Optional[LazyND2Volume],
                   channel_display: Optional[Dict[str, Dict[str, Any]]] = None,
                   z_mode: str = "max", z_index: int = 0,
                   m: int = 0, t: int = 0, z: int = 0) -> None:
        """Wire up to a LazyND2Volume. Use this on the Import page so M/Z
        scrolling reads frames directly from disk."""
        self._volume = volume
        self._channels = {}
        self._z_mode = z_mode
        self._z_index = z_index
        self._m = m
        self._t = t
        self._z = z

        if volume is None:
            self._populate_channel_rows([], channel_display or {})
            return

        # Configure sliders.
        self.m_slider.blockSignals(True)
        self.m_slider.setRange(0, max(0, volume.n_multipoints - 1))
        self.m_slider.setValue(int(self._m))
        self.m_slider.blockSignals(False)
        self._m_row.itemAt(0).widget().setVisible(volume.n_multipoints > 1)
        self.m_slider.setVisible(volume.n_multipoints > 1)
        self.m_label.setVisible(volume.n_multipoints > 1)

        self.t_slider.blockSignals(True)
        self.t_slider.setRange(0, max(0, volume.n_timepoints - 1))
        self.t_slider.setValue(int(self._t))
        self.t_slider.blockSignals(False)

        # Z slider only useful when not projecting.
        z_visible = volume.n_zslices > 1 and z_mode == "none"
        self.z_slider.blockSignals(True)
        self.z_slider.setRange(0, max(0, volume.n_zslices - 1))
        self.z_slider.setValue(int(self._z))
        self.z_slider.blockSignals(False)
        self._z_row.itemAt(0).widget().setVisible(z_visible)
        self.z_slider.setVisible(z_visible)
        self.z_label.setVisible(z_visible)

        self._update_axis_labels()

        self._populate_channel_rows(volume.channel_names, channel_display or {})
        # Compute per-channel histograms from a sample at the current M.
        self._populate_lut_samples_from_volume()
        self._refresh()

    def set_channels(self, channels: Dict[str, Any],
                     channel_display: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Wire up to a dict of (T, H, W) channel arrays. Use this on the
        Recipe page where M is fixed and Z is collapsed."""
        self._volume = None
        self._channels = dict(channels)
        if not channels:
            self._populate_channel_rows([], channel_display or {})
            return
        sample = next(iter(channels.values()))
        n = sample.shape[0]

        # Hide M and Z sliders.
        for w in (self._m_row.itemAt(0).widget(), self.m_slider, self.m_label):
            w.setVisible(False)
        for w in (self._z_row.itemAt(0).widget(), self.z_slider, self.z_label):
            w.setVisible(False)

        self.t_slider.blockSignals(True)
        self.t_slider.setRange(0, max(0, n - 1))
        self.t_slider.setValue(0)
        self.t_slider.blockSignals(False)
        self._t = 0
        self._update_axis_labels()

        self._populate_channel_rows(list(channels.keys()), channel_display or {})
        # Histograms from each channel's data.
        for row in self._channel_rows:
            data = channels.get(row.name)
            if data is not None:
                row.lut.set_data(data)
        self._refresh()

    def _populate_channel_rows(self, names: List[str],
                               display: Dict[str, Dict[str, Any]]) -> None:
        # Clear.
        for row in self._channel_rows:
            row.setParent(None)
            row.deleteLater()
        self._channel_rows.clear()

        cycle = ["green", "red", "cyan", "magenta", "yellow", "blue", "orange"]
        for i, name in enumerate(names):
            cd = display.get(name, {})
            row = ChannelControlRow(
                name,
                color_default=cd.get("color", cycle[i % len(cycle)]),
                enabled=bool(cd.get("enabled", True)),
            )
            row.state_changed.connect(self._on_channel_state)
            row.contrast_changed.connect(lambda *_: self._on_channel_state())
            self._channel_rows.append(row)
            self._channels_layout.addWidget(row)

    def _populate_lut_samples_from_volume(self) -> None:
        """Sample frames from the active M position to seed each LUT."""
        if self._volume is None:
            return
        for c, name in enumerate(self._volume.channel_names):
            row = next((r for r in self._channel_rows if r.name == name), None)
            if row is None:
                continue
            # Cheap sampler: 8 frames evenly spaced in T at the current M.
            n_t = self._volume.n_timepoints
            n_sample = min(8, n_t)
            idxs = np.linspace(0, max(0, n_t - 1), max(1, n_sample), dtype=int)
            samples: List[np.ndarray] = []
            last_shape: Optional[Tuple[int, int]] = None
            for t in idxs:
                try:
                    f = self._volume.get_frame(c=c, m=self._m, t=int(t),
                                                z=self._z, z_mode=self._z_mode)
                    samples.append(f.ravel())
                    last_shape = f.shape
                except Exception:
                    continue
            if not samples or last_shape is None:
                continue
            row.lut.set_data(_SingleFrameSeries(samples, last_shape),
                             dtype=self._volume.dtype)

    # ── Slider handlers ──
    def _on_m_changed(self, v: int) -> None:
        self._m = int(v)
        self._update_axis_labels()
        # Re-seed LUT samples for the new M.
        self._populate_lut_samples_from_volume()
        self._refresh()
        self.coords_changed.emit(self._m, self._t, self._z)

    def _on_t_changed(self, v: int) -> None:
        self._t = int(v)
        self._update_axis_labels()
        self._refresh()
        self.coords_changed.emit(self._m, self._t, self._z)

    def _on_z_changed(self, v: int) -> None:
        self._z = int(v)
        self._update_axis_labels()
        self._refresh()
        self.coords_changed.emit(self._m, self._t, self._z)

    def _on_channel_state(self) -> None:
        self._refresh()
        self.channels_changed.emit()

    # ── Drawing ──
    def _refresh(self) -> None:
        composite = self._compose_current_frame()
        if composite is None:
            return
        self.canvas.set_image(composite)

    def _compose_current_frame(self) -> Optional[np.ndarray]:
        # Multi-source: prefer volume (M/Z aware) when available.
        sources: List[Tuple[ChannelControlRow, np.ndarray]] = []
        if self._volume is not None:
            for c_idx, name in enumerate(self._volume.channel_names):
                row = next((r for r in self._channel_rows if r.name == name), None)
                if row is None or not row.enabled:
                    continue
                try:
                    frame = self._volume.get_frame(
                        c=c_idx, m=self._m, t=self._t, z=self._z,
                        z_mode=self._z_mode,
                    )
                except Exception:
                    continue
                sources.append((row, frame))
        else:
            for row in self._channel_rows:
                if not row.enabled:
                    continue
                data = self._channels.get(row.name)
                if data is None:
                    continue
                try:
                    frame = np.asarray(data[self._t])
                except Exception:
                    continue
                sources.append((row, frame))

        if not sources:
            return None

        sample = sources[0][1]
        h, w = sample.shape
        composite = np.zeros((h, w, 3), dtype=np.float32)
        for row, frame in sources:
            lo, hi, gamma = row.lut.get_contrast()
            mapped = apply_lut(frame, lo, hi, gamma).astype(np.float32)
            r, g, b = row.color_rgb
            composite[..., 0] += mapped * (r / 255.0)
            composite[..., 1] += mapped * (g / 255.0)
            composite[..., 2] += mapped * (b / 255.0)
        return np.clip(composite, 0, 255).astype(np.uint8)

    def _update_axis_labels(self) -> None:
        if self._volume is not None:
            self.m_label.setText(f"{self._m + 1}/{self._volume.n_multipoints}")
            self.t_label.setText(f"{self._t + 1}/{self._volume.n_timepoints}")
            self.z_label.setText(f"{self._z + 1}/{self._volume.n_zslices}")
        else:
            n_t = self.t_slider.maximum() + 1 if self.t_slider.maximum() >= 0 else 0
            self.t_label.setText(f"{self._t + 1}/{n_t}")

    # ── Read-out ──
    def channel_state(self) -> Dict[str, Dict[str, Any]]:
        """Return {name: {enabled, color, lut_lo, lut_hi, lut_gamma}}."""
        out: Dict[str, Dict[str, Any]] = {}
        for row in self._channel_rows:
            lo, hi, g = row.lut.get_contrast()
            out[row.name] = {
                "enabled": row.enabled,
                "color": row.color_name,
                "lut_lo": lo, "lut_hi": hi, "lut_gamma": g,
            }
        return out

    def coords(self) -> Tuple[int, int, int]:
        return self._m, self._t, self._z

    def apply_channel_state(self, state: Dict[str, Dict[str, Any]]) -> None:
        """Restore channel rows from a saved state map (round-trips
        through the experiment record)."""
        for row in self._channel_rows:
            cfg = state.get(row.name)
            if not cfg:
                continue
            row.cb.blockSignals(True)
            row.cb.setChecked(bool(cfg.get("enabled", True)))
            row.cb.blockSignals(False)
            color = cfg.get("color")
            if color and color in CHANNEL_COLORS:
                row.combo_color.blockSignals(True)
                row.combo_color.setCurrentText(color)
                row.combo_color.blockSignals(False)
            if "lut_lo" in cfg and "lut_hi" in cfg:
                row.lut.set_contrast(
                    float(cfg["lut_lo"]),
                    float(cfg["lut_hi"]),
                    float(cfg.get("lut_gamma", 1.0)),
                )
        self._refresh()


class _SingleFrameSeries:
    """Tiny shim that matches the protocol LutHistogramWidget.set_data
    expects (``__len__`` + ``__getitem__``) for a list of 1D sample
    arrays. Lets us feed pre-flattened pixel samples without copying
    them into a fake 3D stack."""
    def __init__(self, samples: List[np.ndarray], frame_shape):
        self._samples = samples
        self._frame_shape = frame_shape

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> np.ndarray:
        return self._samples[int(idx)]
