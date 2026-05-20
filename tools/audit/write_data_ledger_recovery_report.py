#!/usr/bin/env python3
"""Write a recovery report for AutoDL data layout and expansion inputs."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any


ROOT = Path("/root/autodl-tmp/pdf2latex_nn")
OUT = ROOT / "data/09_eval_reports/data_ledger_recovery_20260520"


MANIFESTS = {
    "clean_trainable_1829": ROOT / "data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json",
    "floatproxy_labeled_1829": ROOT / "data/00_manifests/v7_floatproxy_adapter_20260516_205926_labeled.json",
    "floatproxy_rebuilt_1857": ROOT / "data/00_manifests/v7_floatproxy_adapter_20260516_205926_rebuilt.json",
    "bad_expansion_36": ROOT / "data/00_manifests/v7_clean_expand5000_localmodels_20260518_153355.json",
    "from_rawmineru_strict_audit_smoke14": ROOT / "data/00_manifests/v7_clean_expand5000_from_rawmineru_20260520_smoke100.json",
    "from_rawmineru_historical_gate_smoke34": ROOT / "data/00_manifests/v7_clean_expand5000_from_rawmineru_20260520_smoke100_gate30_60.json",
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = {
        "manifests": {name: manifest_summary(path) for name, path in MANIFESTS.items()},
        "current_run_path_audit": load_summary(ROOT / "data/09_eval_reports/expansion_path_audit_20260519/current_run_content_path_audit.json"),
        "alternative_v7": load_summary(OUT / "alternative_v7_content_candidates.json"),
        "mineru_dir_probe": load_summary(ROOT / "data/09_eval_reports/expansion_path_audit_20260519/REMOTE_DATA_LAYOUT_PROBE.json"),
        "stopped_debug_run": stopped_debug_run_summary(),
        "code_changes": {
            "historical_quality_defaults": "build_mini_dataset.py and build_v7_dataset_staged.py default to the historical trainable-set gate: max_orphan_ratio=0.30 and max_unmapped_tex_ratio=0.60.",
            "resolver_cli": "build_v7_dataset_staged.py accepts --mineru-output-dirs, --prefer-valid-v7-content, --require-v7-schema-fields, schema coverage thresholds, --allow-stale-v7-content, and --force-refresh-v7-conversion.",
            "bounded_lookup": "content_resolver enumerates only <root>/<doc_id>/auto and <root>/<doc_id> direct candidates; it no longer recursively scans data/02_mineru_outputs per document.",
        },
    }
    (OUT / "CANONICAL_DATA_LEDGER.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2), encoding="utf-8")
    (OUT / "DATA_LEDGER_RECOVERY_REPORT.md").write_text(markdown(ledger), encoding="utf-8")
    print(json.dumps({"report": str(OUT / "DATA_LEDGER_RECOVERY_REPORT.md")}, ensure_ascii=False, indent=2))
    return 0


def manifest_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    docs = payload.get("documents", payload if isinstance(payload, list) else [])
    content_roots = Counter(root_after(doc.get("content_json") or "", "02_mineru_outputs") for doc in docs)
    graph_roots = Counter(root_after(doc.get("graph_path") or "", "06_graph_features") for doc in docs)
    return {
        "path": str(path),
        "exists": True,
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
        "success_count": payload.get("success_count") if isinstance(payload, dict) else None,
        "document_count": len(docs),
        "content_roots": dict(content_roots.most_common(10)),
        "graph_roots": dict(graph_roots.most_common(10)),
        "first_document": first_doc(docs),
    }


def first_doc(docs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not docs:
        return None
    doc = docs[0]
    return {
        "document_id": doc.get("document_id"),
        "content_json": doc.get("content_json"),
        "graph_path": doc.get("graph_path"),
        "orphan_ratio": doc.get("orphan_ratio"),
        "candidate_edge_recall": doc.get("candidate_edge_recall"),
    }


def root_after(path_text: str, marker: str) -> str:
    if not path_text:
        return "missing"
    parts = PurePosixPath(path_text).parts
    if marker not in parts:
        return "other"
    index = parts.index(marker)
    if index + 1 >= len(parts):
        return "missing_after_marker"
    return parts[index + 1]


def load_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "summary" in payload:
        return {"path": str(path), "exists": True, **payload["summary"]}
    if isinstance(payload, dict):
        return {"path": str(path), "exists": True, "summary": payload.get("summary", payload)}
    return {"path": str(path), "exists": True, "type": type(payload).__name__}


def stopped_debug_run_summary() -> dict[str, Any]:
    snap = OUT / "gate30_60_snapshot_before_stop"
    before = read_text(snap / "processes_before_stop.txt")
    after = read_text(snap / "processes_after_sigterm.txt")
    return {
        "snapshot_dir": str(snap),
        "snapshot_exists": snap.exists(),
        "sigterm_sent_to_pgid": 705243,
        "processes_before_stop": before.strip().splitlines(),
        "processes_after_sigterm": after.strip().splitlines(),
        "stopped": bool(before.strip()) and not bool(after.strip()),
    }


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def markdown(ledger: dict[str, Any]) -> str:
    manifests = ledger["manifests"]
    path_audit = ledger["current_run_path_audit"].get("summary", ledger["current_run_path_audit"])
    alternative = ledger["alternative_v7"]
    stopped = ledger["stopped_debug_run"]
    lines = [
        "# Data Ledger Recovery Report",
        "",
        "## What Is Actually Clean",
        "",
        "The current clean training set is `v7_floatproxy_adapter_20260516_205926_trainable_recall98.json`.",
        "",
        "| manifest | docs | success_count | content root | graph root | status |",
        "|---|---:|---:|---|---|---|",
    ]
    for key, item in manifests.items():
        content = ", ".join(f"{k}:{v}" for k, v in item.get("content_roots", {}).items())
        graph = ", ".join(f"{k}:{v}" for k, v in item.get("graph_roots", {}).items())
        status = manifest_status(key)
        lines.append(
            f"| `{key}` | {item.get('document_count')} | {item.get('success_count')} | {content} | {graph} | {status} |"
        )
    lines.extend(
        [
            "",
            "The 1829 set is a filtered, trainable subset from the float-proxy adapter family. The upstream rebuild manifest has 1857 docs, so the interrupted/unfinished production left a valid completed subset rather than a full 5000-doc corpus.",
            "",
            "## Root Cause Of The Bad Expansion",
            "",
            f"- Processed docs in 5/18 bad expansion audit: `{path_audit.get('processed_docs_count')}`.",
            f"- Old `mineru_output` count: `{path_audit.get('old_mineru_output_count')}`.",
            f"- Stale schema count: `{path_audit.get('stale_schema_count')}`.",
            "- `mineru_output` is a mixed raw/stale directory and must not be treated as canonical v7 content.",
            f"- Alternative valid v7 content exists for `{alternative.get('stale_docs_with_newer_valid_content')}` stale docs out of the processed audit.",
            f"- Best alternative dirs: `{alternative.get('top_alternative_dirs')}`.",
            "",
            "This means the collapse was primarily a content-path/schema selection bug, not evidence that the PDF/TeX source pool is unusable.",
            "",
            "## Stopped Debug Run",
            "",
            f"- Snapshot: `{stopped.get('snapshot_dir')}`",
            f"- SIGTERM succeeded: `{stopped.get('stopped')}`",
            "- The stopped run used the historical training gate `max_orphan_ratio=0.30` and `max_unmapped_tex_ratio=0.60`. It is still not merged because it was an interrupted smoke/debug run, not because 30/60 is invalid.",
            "",
            "## Canonical Rules Going Forward",
            "",
            "1. Keep `v7_floatproxy_adapter_20260516_205926_trainable_recall98.json` as the only current clean training manifest.",
            "2. Treat `data/02_mineru_outputs/mineru_output` as raw/stale source only. It can seed v7 refresh, but cannot be a final graph/label `content_json` source.",
            "3. Use 30/60 as the historical trainable/relabel gate unless a separate audit explicitly asks for stricter 15/30 diagnostics.",
            "4. New expansion must write a fresh v7 content root and use resolver/preflight before graph build.",
            "5. Do not merge `v7_clean_expand5000_localmodels_20260518_153355`, `v7_clean_expand5000_from_rawmineru_20260520_smoke100`, or `gate30_60` outputs into clean without a same-gate lineage audit.",
            "6. Use `--target-total`, not ambiguous `--target`, for expansion goals.",
            "",
            "## Code Changes Applied",
            "",
        ]
    )
    for value in ledger["code_changes"].values():
        lines.append(f"- {value}")
    lines.extend(
        [
            "",
            "## Next Safe Command Template",
            "",
            "Do not run this until you explicitly approve a drypass. This is a controlled drypass shape, not a full expansion:",
            "",
            "```bash",
            "python -u scripts/pipeline/build_v7_dataset_staged.py \\",
            "  --run-name v7_clean_expand3000_resolvedv7_dry100_20260520 \\",
            "  --require-compiled-accepted \\",
            "  --target-total 3000 \\",
            "  --max-candidates 100 \\",
            "  --skip-mineru-stage \\",
            "  --exclude-manifest data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json \\",
            "  --exclude-manifest data/00_manifests/v7_clean_expand5000_20260518_143035_preflight_failed_exclude.json \\",
            "  --mineru-source-dir data/02_mineru_outputs/mineru_output \\",
            "  --mineru-output-dir data/02_mineru_outputs/v7_clean_expand3000_resolvedv7_20260520_content \\",
            "  --mineru-output-dirs \\",
            "    data/02_mineru_outputs/v7_floatproxy_adapter_20260516_205926_content \\",
            "    data/02_mineru_outputs/v7_registry_adapteraware_20260515_181724_content \\",
            "    data/02_mineru_outputs/mineru_output_v7_tocfilter_2000_labelerfix_20260510 \\",
            "    data/02_mineru_outputs/v7_clean_expand5000_from_rawmineru_20260520_content \\",
            "  --prefer-valid-v7-content \\",
            "  --require-v7-schema-fields \\",
            "  --force-refresh-v7-conversion \\",
            "  --max-orphan-ratio 0.30 \\",
            "  --max-unmapped-tex-ratio 0.60 \\",
            "  --min-candidate-recall 0.98 \\",
            "  --embedding-device cpu \\",
            "  --process-workers 2",
            "```",
            "",
            "Recommendation: run only a 100-doc historical-gate drypass next, with raw/effective metrics reported separately. Do not target total 5000 until the resolved-v7 drypass proves the pass rate is healthy.",
        ]
    )
    return "\n".join(lines) + "\n"


def manifest_status(key: str) -> str:
    if key == "clean_trainable_1829":
        return "canonical clean"
    if key in {"floatproxy_labeled_1829", "floatproxy_rebuilt_1857"}:
        return "source lineage for clean family"
    if key == "bad_expansion_36":
        return "invalid/stale-path expansion"
    if key == "from_rawmineru_strict_audit_smoke14":
        return "stricter 15/30 audit smoke, not merged"
    if key == "from_rawmineru_historical_gate_smoke34":
        return "historical 30/60 gate smoke, not merged"
    return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
