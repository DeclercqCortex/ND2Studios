"""
``ExportPreviewDialog`` — modal preview shown after the user confirms
movie or image-sequence export settings.

The dialog renders the composited frame (channels → LUT → channel
colour → image adjustments → overlays) at a chosen T index, lets the
user scrub through T, and exposes brightness / contrast / saturation /
hue / fade sliders that update the preview live.

The dialog returns:

* :meth:`exec` — ``QDialog.DialogCode.Accepted`` / ``Rejected``.
* :meth:`adjustments` — the :class:`ImageAdjustments` chosen by the user.

The actual export is run by the caller (``ExportPage``) once the dialog
is accepted; the dialog itself only renders previews and collects the
adjustment values. This keeps the dialog reusable for both movie and
image-sequence exports.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout,
    QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QSplitter, QVBoxLayout, QWidget,
)

from nd2studios.backend.exporters.composite_exporter import (
    ImageAdjustments, _composite_frame,
)
from nd2studios.backend.exporters.movie_exporter import (
    MovieOptions, _draw_overlays,
)
from nd2studios.core.settings import Settings
from nd2studios.widgets.image_viewer import ImageCanvas, ZoomToolbar


class _LabeledSlider(QWidget):
    """A QSlider with a left-side title and a right-side live value label."""

    def __init__(self, title: str, lo: int, hi: int, default: int,
                 suffix: str = "", parent: Optional[QWidget] = None):
        super().__init__(parent)
        row = QHBoxLayout(self)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(8)

        self.lbl = QLabel(title)
        self.lbl.setMinimumWidth(80)
        row.addWidget(self.lbl)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(lo, hi)
        self.slider.setValue(default)
        row.addWidget(self.slider, stretch=1)

        self._suffix = suffix
        self.val = QLabel(self._format(default))
        self.val.setMinimumWidth(48)
        self.val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.val.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        row.addWidget(self.val)

        self.slider.valueChanged.connect(
            lambda v: self.val.setText(self._format(v))
        )

    def _format(self, v: int) -> str:
        return f"{v}{self._suffix}"

    def value(self) -> int:
        return int(self.slider.value())

    def reset(self, v: int) -> None:
        self.slider.setValue(int(v))


class ExportPreviewDialog(QDialog):
    """Modal preview window for movie / image-sequence exports.

    Parameters
    ----------
    channels : ``{name: (T, H, W)}`` arrays to preview. T can be 1.
    colors : ``{name: (R, G, B)}``.
    enabled : ``{name: bool}``.
    pixel_size_um : drives the scale-bar overlay.
    frame_timestamps_s : per-frame ND2 timestamps for the timestamp overlay.
    lut_settings : ``{name: (lo, hi, gamma)}`` — passed through to the
        composite pipeline so the preview matches the viewer.
    movie_options : initial overlay configuration. The dialog does **not**
        edit these (they live on the movie tab); it only renders with them.
    title : window title — distinguishes the movie vs image-sequence dialog.
    """

    def __init__(
        self,
        channels: Dict[str, np.ndarray],
        colors: Dict[str, Tuple[int, int, int]],
        enabled: Dict[str, bool],
        pixel_size_um: float = 1.0,
        frame_timestamps_s: Optional[np.ndarray] = None,
        lut_settings: Optional[Dict[str, Tuple[float, float, float]]] = None,
        movie_options: Optional[MovieOptions] = None,
        title: str = "Export Preview",
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.resize(1000, 720)

        self._channels = channels
        self._colors = colors
        self._enabled = enabled
        self._pixel_size_um = float(pixel_size_um)
        self._frame_timestamps_s = (
            np.asarray(frame_timestamps_s)
            if frame_timestamps_s is not None else None
        )
        self._lut_settings = lut_settings or {}
        self._opts = movie_options or MovieOptions()

        if not channels:
            raise ValueError("ExportPreviewDialog: no channels to preview")
        sample = next(iter(channels.values()))
        if sample.ndim != 3:
            raise ValueError(
                f"ExportPreviewDialog: channels must be (T,H,W) — got {sample.shape}"
            )
        self._n_t = int(sample.shape[0])

        # Debounce timer for slider drags — full composites can be heavy.
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(30)
        self._render_timer.timeout.connect(self._render_preview)

        # Playback timer — ticks at the current preview FPS and advances T.
        self._play_timer = QTimer(self)
        self._play_timer.setInterval(int(1000 / max(0.5, float(self._opts.fps or 10.0))))
        self._play_timer.timeout.connect(self._on_play_tick)
        self._user_scrubbing = False     # set while the user drags the T slider

        self._build_ui()
        self._render_preview()

    # ── UI ──
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(6)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.setChildrenCollapsible(False)

        # Left: canvas + T scrubber + zoom toolbar.
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.setSpacing(4)
        self.canvas = ImageCanvas()
        ll.addWidget(self.canvas, stretch=1)
        ll.addWidget(ZoomToolbar(self.canvas))

        t_row = QHBoxLayout()
        t_row.setContentsMargins(0, 0, 0, 0)
        t_row.setSpacing(6)
        self.btn_play = QPushButton("▶ Play")
        self.btn_play.setCheckable(True)
        self.btn_play.setFixedWidth(78)
        self.btn_play.setEnabled(self._n_t > 1)
        self.btn_play.toggled.connect(self._on_play_toggled)
        t_row.addWidget(self.btn_play)
        self.lbl_t = QLabel("T")
        t_row.addWidget(self.lbl_t)
        self.slider_t = QSlider(Qt.Orientation.Horizontal)
        self.slider_t.setRange(0, max(0, self._n_t - 1))
        self.slider_t.setValue(0)
        self.slider_t.setEnabled(self._n_t > 1)
        self.slider_t.sliderPressed.connect(self._on_slider_pressed)
        self.slider_t.sliderReleased.connect(self._on_slider_released)
        self.slider_t.valueChanged.connect(self._on_t_changed)
        t_row.addWidget(self.slider_t, stretch=1)
        self.spin_t = QSpinBox()
        self.spin_t.setRange(1, max(1, self._n_t))
        self.spin_t.setValue(1)
        self.spin_t.setEnabled(self._n_t > 1)
        self.spin_t.valueChanged.connect(
            lambda v: self.slider_t.setValue(int(v) - 1)
        )
        t_row.addWidget(self.spin_t)
        self.lbl_t_total = QLabel(f"/ {self._n_t}")
        self.lbl_t_total.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        t_row.addWidget(self.lbl_t_total)
        self.spin_fps = QDoubleSpinBox()
        self.spin_fps.setRange(0.5, 60.0)
        self.spin_fps.setSingleStep(0.5)
        self.spin_fps.setSuffix(" fps")
        self.spin_fps.setValue(float(self._opts.fps or 10.0))
        self.spin_fps.setEnabled(self._n_t > 1)
        self.spin_fps.valueChanged.connect(self._on_fps_changed)
        t_row.addWidget(self.spin_fps)
        ll.addLayout(t_row)
        split.addWidget(left)

        # Right: adjustment sliders.
        right = QWidget()
        right.setMinimumWidth(280)
        right.setMaximumWidth(360)
        rl = QVBoxLayout(right)
        rl.setContentsMargins(4, 4, 4, 4)
        rl.setSpacing(8)

        grp = QGroupBox("Image adjustments")
        gl = QVBoxLayout(grp)
        gl.setSpacing(4)

        self.s_brightness = _LabeledSlider("Brightness", -100, 100, 0)
        self.s_contrast = _LabeledSlider("Contrast", -100, 100, 0)
        self.s_saturation = _LabeledSlider("Saturation", -100, 100, 0)
        self.s_hue = _LabeledSlider("Hue", -180, 180, 0, suffix="°")
        self.s_fade = _LabeledSlider("Fade", 0, 100, 0, suffix="%")
        for s in (self.s_brightness, self.s_contrast, self.s_saturation,
                  self.s_hue, self.s_fade):
            s.slider.valueChanged.connect(self._schedule_render)
            gl.addWidget(s)

        btn_reset = QPushButton("Reset adjustments")
        btn_reset.clicked.connect(self._reset_adjustments)
        gl.addWidget(btn_reset)
        rl.addWidget(grp)

        # Overlay toggle — handy when the user wants to see the raw frame
        # without the scale bar / timestamp / labels obscuring things.
        self.cb_show_overlays = QCheckBox("Show overlays in preview")
        self.cb_show_overlays.setChecked(True)
        self.cb_show_overlays.stateChanged.connect(self._schedule_render)
        rl.addWidget(self.cb_show_overlays)

        rl.addStretch(1)

        info = QLabel(
            "Drag sliders to preview; click OK to apply these settings "
            "to the export."
        )
        info.setWordWrap(True)
        info.setStyleSheet(f"color: {Settings.FG_SECONDARY}; font: 9pt;")
        rl.addWidget(info)

        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        outer.addWidget(split, stretch=1)

        bb = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        bb.button(QDialogButtonBox.StandardButton.Ok).setText("Export")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        outer.addWidget(bb)

    # ── Slots ──
    def _on_t_changed(self, idx: int) -> None:
        self.spin_t.blockSignals(True)
        self.spin_t.setValue(int(idx) + 1)
        self.spin_t.blockSignals(False)
        self._schedule_render()

    def _on_slider_pressed(self) -> None:
        # Pause playback while the user is dragging the T scrubber.
        self._user_scrubbing = True
        if self.btn_play.isChecked():
            self.btn_play.setChecked(False)

    def _on_slider_released(self) -> None:
        self._user_scrubbing = False

    def _on_play_toggled(self, on: bool) -> None:
        if on:
            self.btn_play.setText("⏸ Pause")
            self._play_timer.start()
        else:
            self.btn_play.setText("▶ Play")
            self._play_timer.stop()

    def _on_fps_changed(self, fps: float) -> None:
        fps = max(0.5, float(fps))
        self._play_timer.setInterval(int(1000.0 / fps))

    def _on_play_tick(self) -> None:
        if self._user_scrubbing or self._n_t <= 1:
            return
        nxt = (int(self.slider_t.value()) + 1) % self._n_t
        # Block the signal so the debounce timer doesn't add latency — we
        # render directly here to keep playback smooth.
        self.slider_t.blockSignals(True)
        self.slider_t.setValue(nxt)
        self.spin_t.blockSignals(True)
        self.spin_t.setValue(nxt + 1)
        self.spin_t.blockSignals(False)
        self.slider_t.blockSignals(False)
        self._render_preview()

    def _schedule_render(self) -> None:
        self._render_timer.start()

    def _reset_adjustments(self) -> None:
        for s in (self.s_brightness, self.s_contrast, self.s_saturation,
                  self.s_hue, self.s_fade):
            s.reset(0)
        self._render_preview()

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        self._play_timer.stop()
        super().closeEvent(event)

    def done(self, code: int) -> None:
        self._play_timer.stop()
        super().done(int(code))

    def _render_preview(self) -> None:
        t = int(self.slider_t.value())
        frame_dict = {name: arr[t] for name, arr in self._channels.items()}
        rgb = _composite_frame(
            frame_dict, self._colors, self._enabled,
            self._lut_settings or None,
            image_adjustments=self.adjustments(),
        )
        if self.cb_show_overlays.isChecked():
            rgb = _draw_overlays(
                rgb, t_index=t, opts=self._opts,
                pixel_size_um=self._pixel_size_um,
                channel_colors=self._colors,
                channel_enabled=self._enabled,
                channel_names=list(self._channels.keys()),
                frame_timestamps=self._frame_timestamps_s,
            )
        self.canvas.set_image(np.ascontiguousarray(rgb))

    # ── Public API ──
    def adjustments(self) -> ImageAdjustments:
        return ImageAdjustments(
            brightness=float(self.s_brightness.value()),
            contrast=float(self.s_contrast.value()),
            saturation=float(self.s_saturation.value()),
            hue=float(self.s_hue.value()),
            fade=float(self.s_fade.value()),
        )
