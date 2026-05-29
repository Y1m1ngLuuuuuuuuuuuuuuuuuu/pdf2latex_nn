#!/usr/bin/env python3
"""Compatibility wrapper for PageFurniture Context Phase1 diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.audit.validate_page_furniture_context_phase1_mineru_evidence import main


if __name__ == "__main__":
    raise SystemExit(main())
