#!/usr/bin/env python3
"""Compatibility-safe entrypoint for the current canonical E2E comparison.

The historical ``run_m05_e2e_comparison.py`` script now also points at the
current model/generator defaults, but this wrapper gives new experiments a
clearer name while preserving older report scripts.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.run_m05_e2e_comparison import main


if __name__ == "__main__":
    raise SystemExit(main())
