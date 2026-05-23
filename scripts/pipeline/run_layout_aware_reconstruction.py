#!/usr/bin/env python3
"""Canonical layout-aware E2E reconstruction entrypoint.

This is the default paper-facing reconstruction path after the GNN relation
influence audit.  It keeps the full-v7-first renderer stack:

    full v7 JSON + graph/v7 bridge
      -> deterministic heading stack / decoder safety rules
      -> RenderTreeIR
      -> OriginalLikeIRLatexRenderer

No learned GNN relation logits are loaded by default.  The historical GNN E2E
entrypoints remain available for ablations and diagnostics, but they are no
longer the default reconstruction path.

For compatibility with older orchestration scripts, legacy model arguments
such as ``--checkpoint``, ``--tau-merge``, and ``--tau-parent`` are accepted and
ignored here.  Passing them does not activate GNN inference.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.pipeline.run_rules_only_e2e_comparison import main as rules_only_main  # noqa: E402


LEGACY_VALUE_ARGS = {
    "--checkpoint",
    "--tau-merge",
    "--tau-parent",
}


def _strip_legacy_gnn_args(argv: list[str]) -> list[str]:
    """Drop accepted legacy GNN args before forwarding to rules-only parser."""

    cleaned: list[str] = [argv[0]]
    index = 1
    while index < len(argv):
        item = argv[index]
        if item in LEGACY_VALUE_ARGS:
            index += 2
            continue
        if any(item.startswith(prefix + "=") for prefix in LEGACY_VALUE_ARGS):
            index += 1
            continue
        cleaned.append(item)
        index += 1
    return cleaned


def main() -> int:
    original_argv = sys.argv
    sys.argv = _strip_legacy_gnn_args(sys.argv)
    try:
        return rules_only_main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
