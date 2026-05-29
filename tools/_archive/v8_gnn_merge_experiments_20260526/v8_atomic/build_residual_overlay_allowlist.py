#!/usr/bin/env python3
"""Build a narrow allowlist for residual-targeted learned MERGE overlay.

The allowlist is derived from ``diagnose_residual_missing_merges.py`` output.
It is intentionally conservative: by default it includes only:

  1. residual gaps whose tail/head already look mergeable;
  2. a small number of large-gap residuals per document, excluding cases already
     classified as math/formula, heading/run-in, citation/reference, or other
     non-continuation patterns.

The output is consumed by ``project_predictions_to_v8.py`` via
``--residual-overlay-allowlist``.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_PRIMARY_REASONS = {"residual_even_though_tail_head_looks_mergeable"}
DEFAULT_LARGE_GAP_REASONS = {"large_generated_gap_or_intervening_blocks"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residual-diagnostic-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--primary-reasons", nargs="*", default=sorted(DEFAULT_PRIMARY_REASONS))
    parser.add_argument("--large-gap-reasons", nargs="*", default=sorted(DEFAULT_LARGE_GAP_REASONS))
    parser.add_argument("--max-large-gap-per-doc", type=int, default=3)
    parser.add_argument("--max-targets-per-doc", type=int, default=20)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    primary_reasons = set(args.primary_reasons)
    large_gap_reasons = set(args.large_gap_reasons)
    doc_dirs = sorted(p for p in args.residual_diagnostic_dir.iterdir() if p.is_dir())
    items: list[dict[str, Any]] = []
    reason_counter: Counter[str] = Counter()
    for doc_dir in doc_dirs:
        path = doc_dir / "residual_missing_merges.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        targets = select_targets(
            payload,
            primary_reasons=primary_reasons,
            large_gap_reasons=large_gap_reasons,
            max_large_gap_per_doc=args.max_large_gap_per_doc,
            max_targets_per_doc=args.max_targets_per_doc,
        )
        if not targets:
            continue
        reason_counter.update(target["reason"] for target in targets)
        items.append({"doc_id": payload["doc_id"], "targets": targets})
    out = {
        "schema_version": "v8_residual_overlay_allowlist_v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_residual_diagnostic_dir": str(args.residual_diagnostic_dir),
        "primary_reasons": sorted(primary_reasons),
        "large_gap_reasons": sorted(large_gap_reasons),
        "max_large_gap_per_doc": args.max_large_gap_per_doc,
        "max_targets_per_doc": args.max_targets_per_doc,
        "doc_count": len(items),
        "target_count": sum(len(item["targets"]) for item in items),
        "reason_counts": dict(reason_counter),
        "items": items,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


def select_targets(
    payload: dict[str, Any],
    *,
    primary_reasons: set[str],
    large_gap_reasons: set[str],
    max_large_gap_per_doc: int,
    max_targets_per_doc: int,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    large_gap_count = 0
    for residual in payload.get("residual_missing_merges") or []:
        source = residual.get("source") or {}
        for gap_index, gap in enumerate(residual.get("gaps") or []):
            reason = str(gap.get("suspected_reason") or "")
            include = False
            if reason in primary_reasons:
                include = True
            elif reason in large_gap_reasons and large_gap_count < max_large_gap_per_doc:
                include = True
                large_gap_count += 1
            if not include:
                continue
            targets.append(
                {
                    "target_id": f"{source.get('block_id', 'source_unknown')}::gap{gap_index}",
                    "source_block_id": source.get("block_id"),
                    "source_line": source.get("line"),
                    "source_preview": source.get("preview"),
                    "reason": reason,
                    "left_generated_id": gap.get("left_generated_id"),
                    "right_generated_id": gap.get("right_generated_id"),
                    "left_tail": gap.get("left_tail"),
                    "right_head": gap.get("right_head"),
                    "index_gap": gap.get("index_gap"),
                    "left_open_ended": gap.get("left_open_ended"),
                    "right_continuation_like": gap.get("right_continuation_like"),
                }
            )
            if len(targets) >= max_targets_per_doc:
                return targets
    return targets


if __name__ == "__main__":
    raise SystemExit(main())
