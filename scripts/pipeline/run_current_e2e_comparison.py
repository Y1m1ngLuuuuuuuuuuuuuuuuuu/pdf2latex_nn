#!/usr/bin/env python3
"""Compatibility-safe entrypoint for the current canonical E2E comparison.

The current paper-facing E2E path is layout-aware and full-v7-first, but does
not use learned GNN relation logits by default.  The historical GNN E2E scripts
remain available for explicit ablations.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.run_layout_aware_reconstruction import main


if __name__ == "__main__":
    raise SystemExit(main())
