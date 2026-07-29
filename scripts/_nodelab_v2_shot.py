"""Offscreen boot of NodeLab v2 → PNG (headless fidelity check).

    python scripts/_nodelab_v2_shot.py [out.png] [--welcome]

Renders the example canvas without a display (QT_QPA_PLATFORM=offscreen) so the GUI can
be reviewed as an image during development. ``--welcome`` shoots the launch state
instead: the blank canvas with its welcome card.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PySide6.QtWidgets import QApplication  # noqa: E402
from PySide6.QtGui import QFontDatabase      # noqa: E402
from nodelab_v2.window import MainWindow      # noqa: E402


def _load_fonts() -> None:
    """The offscreen QPA platform loads no system fonts, so register the actual
    Windows TTFs by path — purely so the verification PNG shows real glyphs (the
    on-display app already has these families)."""
    for name in ("segoeui.ttf", "consola.ttf", "arial.ttf", "seguisb.ttf"):
        path = os.path.join(os.environ.get("WINDIR", r"C:\Windows"), "Fonts", name)
        if os.path.exists(path):
            QFontDatabase.addApplicationFont(path)


def main(argv) -> int:
    out = argv[1] if len(argv) > 1 else "nodelab_v2_shot.png"
    app = QApplication(sys.argv[:1])
    _load_fonts()
    win = MainWindow()
    win.resize(1400, 840)
    win.show()
    if "--welcome" not in argv:
        win.build_demo()          # the app opens blank now; the shot wants the example
    for _ in range(3):
        app.processEvents()
    win.view.fit_all()
    for _ in range(3):
        app.processEvents()
    pix = win.grab()
    ok = pix.save(out)
    print(f"saved {out}  {pix.width()}x{pix.height()}  ok={ok}")
    sys.stdout.flush()
    os._exit(0 if ok else 1)     # skip Qt's offscreen teardown (crashes on exit ≠ render failure)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
