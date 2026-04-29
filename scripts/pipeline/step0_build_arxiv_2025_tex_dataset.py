"""Entry point for building the arXiv 2025 TeX compile-success dataset."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.arxiv_source_dataset import main


if __name__ == "__main__":
    raise SystemExit(main())
