"""
Reusable playback controls: Play/Pause, frame slider, FPS spinbox.
Connect to an ImageViewer's frame_changed signal for synchronized playback.
"""
from __future__ import annotations

from PySide6.QtWidgets import (
    QWidget, QHBoxLayout, QPushButton, QSlider, QLabel, QSpinBox,
)
from PySide6.QtCore import Qt, Signal, QTimer

from nd2studios.core.settings import Settings


class VideoPlayer(QWidget):
    """
    Playback controls widget. Emits frame_changed(int) as it plays.
    Connect this to an ImageViewer.set_frame_index() for synced playback.
    """

    frame_changed = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(30)
        self.btn_prev.clicked.connect(self._step_back)
        layout.addWidget(self.btn_prev)

        self.btn_play = QPushButton("Play")
        self.btn_play.setFixedWidth(60)
        self.btn_play.clicked.connect(self.toggle_play)
        layout.addWidget(self.btn_play)

        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(30)
        self.btn_next.clicked.connect(self._step_forward)
        layout.addWidget(self.btn_next)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 0)
        self.slider.valueChanged.connect(self._on_slider)
        layout.addWidget(self.slider, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(65)
        self.frame_label.setStyleSheet(f"color: {Settings.FG_SECONDARY};")
        layout.addWidget(self.frame_label)

        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(10)
        self.fps_spin.setPrefix("FPS: ")
        self.fps_spin.setFixedWidth(90)
        self.fps_spin.valueChanged.connect(self._update_timer)
        layout.addWidget(self.fps_spin)

        # Timer
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._playing = False
        self._current = 0
        self._max = 0

    def set_range(self, max_frame: int):
        """Set the total number of frames."""
        self._max = max_frame
        self.slider.setRange(0, max(0, max_frame))
        self._update_label()

    def set_frame(self, t: int):
        """Set current frame without emitting signal."""
        self._current = t
        self.slider.blockSignals(True)
        self.slider.setValue(t)
        self.slider.blockSignals(False)
        self._update_label()

    @property
    def current_frame(self) -> int:
        return self._current

    @property
    def playing(self) -> bool:
        return self._playing

    def toggle_play(self):
        if self._playing:
            self.pause()
        else:
            self.play()

    def play(self):
        self._playing = True
        self.btn_play.setText("Pause")
        interval = max(1, int(1000 / self.fps_spin.value()))
        self._timer.start(interval)

    def pause(self):
        self._playing = False
        self.btn_play.setText("Play")
        self._timer.stop()

    def _advance(self):
        nxt = self._current + 1
        if nxt > self._max:
            nxt = 0
        self._set_and_emit(nxt)

    def _step_forward(self):
        if self._current < self._max:
            self._set_and_emit(self._current + 1)

    def _step_back(self):
        if self._current > 0:
            self._set_and_emit(self._current - 1)

    def _on_slider(self, val: int):
        self._current = val
        self._update_label()
        self.frame_changed.emit(val)

    def _set_and_emit(self, t: int):
        self._current = t
        self.slider.blockSignals(True)
        self.slider.setValue(t)
        self.slider.blockSignals(False)
        self._update_label()
        self.frame_changed.emit(t)

    def _update_label(self):
        self.frame_label.setText(f"{self._current} / {self._max}")

    def _update_timer(self):
        if self._playing:
            interval = max(1, int(1000 / self.fps_spin.value()))
            self._timer.setInterval(interval)
