#!/usr/bin/env python3
"""Evaluate gold/prediction comparison-structure JSON files."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.structure_metrics import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
