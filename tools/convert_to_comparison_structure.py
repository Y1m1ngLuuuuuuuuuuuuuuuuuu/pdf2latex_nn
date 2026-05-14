#!/usr/bin/env python3
"""Auto-detect LaTeX/Markdown and convert to comparison JSON."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.comparison_structure import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
