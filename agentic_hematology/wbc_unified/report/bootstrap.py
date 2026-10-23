"""Shared sys.path setup for report/scripts entry points."""
from __future__ import annotations

import sys
from pathlib import Path


REPORT_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = REPORT_ROOT.parent
REPO_ROOT = PROJECT_ROOT.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"
CV_ROOT = PROJECT_ROOT / "cv"


def setup_paths() -> None:
    for p in (REPORT_ROOT, PROJECT_ROOT, REPO_ROOT.parent):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)
