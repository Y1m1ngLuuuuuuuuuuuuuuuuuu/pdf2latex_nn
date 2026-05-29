#!/usr/bin/env python3
"""Compatibility entrypoint for deterministic FrontMatterExtractor Phase0 audits."""

from __future__ import annotations

from tools.audit.validate_frontmatter_extractor_phase0 import main


if __name__ == "__main__":
    raise SystemExit(main())
