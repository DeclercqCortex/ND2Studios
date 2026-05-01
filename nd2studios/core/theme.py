"""
Dracula-based dark theme stylesheet for the ND2Studios GUI.

Adapted from CellTracker's `core/theme.py` with additional rules for the
PyDracula-derived chrome: animated collapsible sidebar with text labels,
custom frameless title bar, drop shadow on the bgApp frame.
"""
from __future__ import annotations


STYLESHEET = """
/* GLOBAL */
QWidget {
    color: #f8f8f2;
    font: 10pt "Helvetica Neue";
    background-color: transparent;
}
QMainWindow {
    background-color: transparent;
}
QToolTip {
    color: #ffffff;
    background-color: rgba(33, 37, 43, 220);
    border: none;
    border-left: 2px solid #bd93f9;
    padding: 6px 10px;
    font: 9pt "Helvetica Neue";
}

/* APP BACKGROUND (the rounded card behind everything when frameless) */
#bgApp {
    background-color: #282a36;
    border: 1px solid #44475a;
    border-radius: 0px;
}

/* FRAMES & PANELS */
#leftMenuBg {
    background-color: #21252b;
    border-right: 1px solid #44475a;
}
#contentArea {
    background-color: #282a36;
}
#topBar {
    background-color: #21252b;
    border-bottom: 1px solid #44475a;
}
#bottomBar {
    background-color: #21252b;
    border-top: 1px solid #44475a;
}

/* CUSTOM TITLE BAR */
#titleBarWidget {
    background-color: #21252b;
    border-bottom: 1px solid #44475a;
}
QLabel#titleBarApp {
    color: #bd93f9;
    font: bold 11pt "Helvetica Neue";
    padding-left: 12px;
}
QLabel#titleBarInfo {
    color: #b0b0b0;
    font: 9pt "Helvetica Neue";
    padding-left: 8px;
}
QPushButton#titleBarBtn {
    background-color: transparent;
    color: #b0b0b0;
    border: none;
    border-radius: 0;
    padding: 0;
    min-width: 36px;
    max-width: 36px;
    min-height: 28px;
    max-height: 28px;
    font: 12pt "Helvetica Neue";
}
QPushButton#titleBarBtn:hover {
    background-color: #343b48;
    color: #f8f8f2;
}
QPushButton#titleBarCloseBtn {
    background-color: transparent;
    color: #b0b0b0;
    border: none;
    border-radius: 0;
    padding: 0;
    min-width: 36px;
    max-width: 36px;
    min-height: 28px;
    max-height: 28px;
    font: 12pt "Helvetica Neue";
}
QPushButton#titleBarCloseBtn:hover {
    background-color: #ff5555;
    color: #ffffff;
}

/* LABELS */
QLabel {
    color: #f8f8f2;
    padding: 0px;
}
QLabel#titleLabel {
    font: bold 14pt "Helvetica Neue";
    color: #bd93f9;
}
QLabel#sectionHeader {
    font: bold 11pt "Helvetica Neue";
    color: #ff79c6;
    padding: 4px 0px;
}
QLabel#accentLabel {
    color: #8be9fd;
}

/* BUTTONS */
QPushButton {
    background-color: #44475a;
    color: #f8f8f2;
    border: none;
    border-radius: 5px;
    padding: 6px 16px;
    font: 10pt "Helvetica Neue";
    min-height: 28px;
}
QPushButton:hover {
    background-color: #5a5e72;
}
QPushButton:pressed {
    background-color: #bd93f9;
    color: #282a36;
}
QPushButton:disabled {
    background-color: #363944;
    color: #666;
}
QPushButton#primaryBtn {
    background-color: #bd93f9;
    color: #282a36;
    font-weight: bold;
}
QPushButton#primaryBtn:hover {
    background-color: #caa4ff;
    color: #282a36;
}
QPushButton#primaryBtn:pressed {
    background-color: #a57ad9;
    color: #282a36;
}
QPushButton#primaryBtn:disabled {
    background-color: #4d4466;
    color: #8a8299;
}
QPushButton#successBtn {
    background-color: #50fa7b;
    color: #282a36;
    font-weight: bold;
}
QPushButton#successBtn:hover { background-color: #6dffa0; color: #282a36; }
QPushButton#successBtn:pressed { background-color: #3edb65; color: #282a36; }
QPushButton#successBtn:disabled { background-color: #3a5a42; color: #88a292; }
QPushButton#dangerBtn {
    background-color: #ff5555;
    color: #f8f8f2;
    font-weight: bold;
}
QPushButton#dangerBtn:hover { background-color: #ff7777; color: #f8f8f2; }
QPushButton#dangerBtn:pressed { background-color: #d94545; color: #f8f8f2; }
QPushButton#dangerBtn:disabled { background-color: #663a3a; color: #a28282; }

/* SIDEBAR NAV BUTTONS (collapsible with text labels — PyDracula style) */
QPushButton#navBtn {
    background-color: transparent;
    color: #b0b0b0;
    border: none;
    border-left: 4px solid transparent;
    border-radius: 0px;
    padding: 0 16px 0 16px;
    text-align: left;
    min-height: 44px;
    max-height: 44px;
    font: 10pt "Helvetica Neue";
}
QPushButton#navBtn:hover {
    background-color: #343b48;
    color: #f8f8f2;
}
QPushButton#navBtn:checked {
    border-left: 4px solid #bd93f9;
    background-color: #2c313c;
    color: #f8f8f2;
}

/* Sidebar bottom session buttons */
QPushButton#sessionBtn {
    background-color: #21252b;
    color: #f8f8f2;
    border: none;
    border-radius: 0;
    padding: 6px 16px;
    text-align: left;
    min-height: 32px;
    max-height: 32px;
    font: 9pt "Helvetica Neue";
}
QPushButton#sessionBtn:hover {
    background-color: #343b48;
}
QPushButton#sessionBtn:pressed {
    background-color: #2c313c;
}

/* Toggle / hamburger button at the top of the sidebar */
QPushButton#toggleBtn {
    background-color: #21252b;
    color: #b0b0b0;
    border: none;
    border-radius: 0;
    padding: 0;
    min-height: 44px;
    max-height: 44px;
    font: 14pt "Helvetica Neue";
    text-align: center;
}
QPushButton#toggleBtn:hover {
    background-color: #343b48;
    color: #f8f8f2;
}

/* INPUTS */
QLineEdit, QSpinBox, QDoubleSpinBox {
    background-color: #343b48;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
    selection-background-color: #bd93f9;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {
    border: 1px solid #bd93f9;
}
QComboBox {
    background-color: #343b48;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    padding: 4px 8px;
    min-height: 26px;
}
QComboBox:hover { border: 1px solid #bd93f9; }
QComboBox::drop-down { border: none; width: 24px; }
QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #bd93f9;
    margin-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #343b48;
    color: #f8f8f2;
    border: 1px solid #44475a;
    selection-background-color: #bd93f9;
    selection-color: #282a36;
    outline: none;
    padding: 2px;
}
QComboBox QAbstractItemView::item { padding: 4px 8px; min-height: 22px; }
QCheckBox { color: #f8f8f2; spacing: 6px; }
QCheckBox::indicator {
    width: 18px; height: 18px;
    border-radius: 3px;
    border: 1px solid #44475a;
    background-color: #343b48;
}
QCheckBox::indicator:checked {
    background-color: #bd93f9;
    border: 1px solid #bd93f9;
}

/* SLIDERS */
QSlider::groove:horizontal { height: 6px; background: #44475a; border-radius: 3px; }
QSlider::handle:horizontal {
    background: #bd93f9; width: 16px; height: 16px;
    margin: -5px 0; border-radius: 8px;
}
QSlider::sub-page:horizontal { background: #bd93f9; border-radius: 3px; }

/* PROGRESS BAR */
QProgressBar {
    background-color: #343b48;
    border: none;
    border-radius: 4px;
    text-align: center;
    color: #f8f8f2;
    font: bold 9pt "Helvetica Neue";
    min-height: 18px;
}
QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #bd93f9, stop:1 #ff79c6);
    border-radius: 4px;
}

/* SCROLL BARS */
QScrollBar:vertical {
    background: #21252b; width: 10px; border: none; border-radius: 5px;
}
QScrollBar::handle:vertical {
    background: #44475a; min-height: 30px; border-radius: 5px;
}
QScrollBar::handle:vertical:hover { background: #5a5e72; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
QScrollBar:horizontal {
    background: #21252b; height: 10px; border: none; border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background: #44475a; min-width: 30px; border-radius: 5px;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }

/* TAB WIDGET */
QTabWidget::pane {
    border: 1px solid #44475a;
    border-radius: 5px;
    background-color: #282a36;
}
QTabBar::tab {
    background-color: #343b48;
    color: #b0b0b0;
    border: none;
    padding: 8px 16px;
    margin-right: 2px;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
}
QTabBar::tab:selected {
    background-color: #282a36;
    color: #bd93f9;
    border-bottom: 2px solid #bd93f9;
}
QTabBar::tab:hover:!selected { background-color: #3a3f4b; color: #f8f8f2; }

/* GROUP BOX */
QGroupBox {
    border: 1px solid #44475a;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font: bold 10pt "Helvetica Neue";
    color: #8be9fd;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    color: #8be9fd;
}

/* LISTS */
QListWidget, QTreeWidget, QTableWidget {
    background-color: #21252b;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    outline: none;
}
QListWidget::item { padding: 6px 8px; border-radius: 3px; }
QListWidget::item:selected { background-color: #44475a; color: #bd93f9; }
QListWidget::item:hover:!selected { background-color: #343b48; }
QHeaderView::section {
    background-color: #343b48;
    color: #8be9fd;
    border: none;
    border-right: 1px solid #44475a;
    padding: 6px;
    font: bold 9pt "Helvetica Neue";
}

/* TEXT EDIT */
QTextEdit, QPlainTextEdit {
    background-color: #21252b;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 5px;
    padding: 4px;
    font: 9pt "Consolas";
}

/* SPLITTER */
QSplitter::handle { background-color: #44475a; }
QSplitter::handle:horizontal { width: 3px; }
QSplitter::handle:vertical { height: 3px; }

/* STATUS INDICATOR */
#statusIndicator {
    font: bold 9pt "Helvetica Neue";
    padding: 4px 12px;
    border-radius: 4px;
}

/* DIALOGS */
QDialog {
    background-color: #282a36;
    color: #f8f8f2;
    border: 1px solid #44475a;
    border-radius: 8px;
}
QMessageBox { background-color: #282a36; color: #f8f8f2; min-width: 420px; }
QMessageBox QLabel {
    color: #f8f8f2;
    font: 10pt "Helvetica Neue";
    min-width: 360px;
    padding: 12px 4px;
    line-height: 140%;
}
QMessageBox QLabel#qt_msgbox_informativelabel {
    color: #b0b0b0;
    font: 9pt "Helvetica Neue";
    padding-top: 4px;
}
QMessageBox QDialogButtonBox {
    padding: 8px 12px 12px 12px;
    background-color: #282a36;
}
QMessageBox QPushButton,
QDialog QDialogButtonBox QPushButton {
    min-width: 92px;
    min-height: 30px;
    padding: 6px 18px;
    background-color: #44475a;
    color: #f8f8f2;
    border: 1px solid #5a5e72;
    border-radius: 5px;
    font: 10pt "Helvetica Neue";
}
QMessageBox QPushButton:hover,
QDialog QDialogButtonBox QPushButton:hover {
    background-color: #5a5e72;
    color: #f8f8f2;
    border: 1px solid #bd93f9;
}
QMessageBox QPushButton:pressed,
QDialog QDialogButtonBox QPushButton:pressed {
    background-color: #bd93f9;
    color: #282a36;
    border: 1px solid #bd93f9;
}
QMessageBox QPushButton:default,
QDialog QDialogButtonBox QPushButton:default {
    background-color: #bd93f9;
    color: #282a36;
    border: 1px solid #bd93f9;
    font-weight: bold;
}
QFileDialog { background-color: #282a36; color: #f8f8f2; }
QFileDialog QLabel { color: #f8f8f2; }
QFileDialog QLineEdit { background-color: #343b48; color: #f8f8f2; }
QFileDialog QListView, QFileDialog QTreeView {
    background-color: #21252b; color: #f8f8f2;
}
QFileDialog QPushButton {
    min-width: 80px; min-height: 28px;
    background-color: #44475a; color: #f8f8f2;
    border: 1px solid #5a5e72; border-radius: 5px; padding: 4px 14px;
}
QFileDialog QPushButton:hover { background-color: #5a5e72; color: #f8f8f2; }
"""
