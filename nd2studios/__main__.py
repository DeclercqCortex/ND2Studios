"""
ND2Studios entry point.

Sets up the Qt application, applies the dark theme, opens the main window.
"""
from __future__ import annotations

import os
import sys


def _setup_environment() -> None:
    """Set environment variables that must be in place before Qt imports."""
    # PyDracula's high-DPI workaround.
    os.environ.setdefault("QT_FONT_DPI", "96")

    # SSL: some plugins (e.g. those that fetch reference data online) need
    # certifi's bundle. Set it eagerly; harmless if unused.
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except ImportError:
        pass


def main() -> None:
    _setup_environment()

    from PySide6.QtCore import Qt
    from PySide6.QtGui import QGuiApplication
    from PySide6.QtWidgets import QApplication

    # Enable high-DPI pixmaps for the image viewer.
    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QGuiApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    if hasattr(Qt, "AA_UseHighDpiPixmaps"):
        QGuiApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    from nd2studios.core.settings import Settings
    from nd2studios.core.theme import STYLESHEET
    from nd2studios.core.main_window import MainWindow

    # Force-import the enhancement plugin module so its decorator-based
    # registration runs at startup (otherwise the Recipe page sees an
    # empty registry).
    import nd2studios.plugins.enhancement.builtin  # noqa: F401

    app = QApplication(sys.argv)
    app.setApplicationName(Settings.APP_NAME)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
