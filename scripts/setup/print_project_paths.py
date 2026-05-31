#!/usr/bin/env python3
"""Print resolved PDF2LaTeX project paths without running experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config.project_paths import describe_paths


def main() -> None:
    print(json.dumps(describe_paths(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
