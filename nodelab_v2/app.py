"""Launch NodeLab v2."""
from __future__ import annotations

import sys


def run() -> int:
    import os
    from PySide6.QtWidgets import QApplication
    from nodelab_v2.window import MainWindow

    # The Viewer's GPU surface (raw upload + LUT/compositing in a shader → instantaneous
    # contrast) is the default; request a 3.3-core context for the whole app BEFORE the
    # QApplication. Set NODELAB_GL=0 to force the CPU path (no GL context requested).
    if os.environ.get("NODELAB_GL", "1") not in ("0", "false", "no"):
        from PySide6.QtGui import QSurfaceFormat
        from nodelab_v2.glview import default_surface_format
        QSurfaceFormat.setDefaultFormat(default_surface_format())
    app = QApplication.instance() or QApplication(sys.argv)
    win = MainWindow()
    win.resize(1600, 1000)          # generous default so the Viewer opens large
    win.showMaximized()             # …and fill the screen on launch
    win.view.fit_all()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
