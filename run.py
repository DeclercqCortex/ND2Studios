#!/usr/bin/env python3
"""
ND2Studios launcher.

Runs the GUI with the project root on PYTHONPATH so the `nd2studios`
package imports work regardless of where this script is invoked from.

Usage:
    python3 run.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# Make sure the project root (the directory containing this file) is on
# sys.path. Without this, `python3 path/to/run.py` from another working
# directory fails to import `nd2studios`.
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from nd2studios.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
