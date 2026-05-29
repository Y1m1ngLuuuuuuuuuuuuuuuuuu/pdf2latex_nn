#!/usr/bin/env python3
"""Finalize the v8 FloatCaptionLayout diagnostic branch.

This script is intentionally report-only. It reads existing selected200
FloatCaptionLayout summaries and writes final decision artifacts. It does not
run generation, compilation, training, MinerU, graph rebuilds, or E2E.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path("data/09_eval_reports/float_caption_layout_20260526")
BASE = ROOT / "v8_same_code_ab_validation"
OUT = ROOT / "finalization"

SUMMARY_PATHS = {
    "Patch1_initial_same_code": BASE / "selected200_same_code_ab_summary.json",
    "Patch2_converter_materialization": BASE / "patch2_same_code_ab" / "selected200_same_code_ab_summary.json",
    "Patch3_canonical_dedupe": BASE / "patch3_same_code_ab" / "selected200_same_code_ab_summary.json",
    "Patch3_1_duplicate_guard": BASE / "patch3_1_duplicate_guard" / "selected200_same_code_ab_summary.json",
}

DECISIONS = {
    "Patch1_initial_same_code": "safe_to_keep_experimental_enabled (experimental only, not production default)",
    "Patch2_converter_materialization": "patch_required",
    "Patch3_canonical_dedupe": "patch_required",
    "Patch3_1_duplicate_guard": "diagnostic_only",
}

BENEFITS = {
    "Patch1_initial_same_code": (
        "Improved float-caption accuracy and macro score; removed caption-as-paragraph; "
        "duplicate count decreased under the then-current metric."
    ),
    "Patch2_converter_materialization": (
        "Converter normalization recovered rendered captions and kept caption-as-paragraph "
        "at zero without wrong-type pairing regression."
    ),
    "Patch3_canonical_dedupe": (
        "Canonical selection reduced noisy caption counting and improved missing captions, "
        "with subfigure false suppression held at zero."
    ),
    "Patch3_1_duplicate_guard": (
        "Duplicate guard reduced flag-off duplicate count to zero and preserved subfigure "
        "safety; caption-as-paragraph stayed zero in flag-on."
    ),
}

BLOCKERS = {
    "Patch1_initial_same_code": (
        "Benefit was small and still experimental; later trace showed many candidates "
        "remained unmaterialized or unmatched."
    ),
    "Patch2_converter_materialization": "Duplicate hard gate failed: duplicate captions increased by 21 in flag-on.",
    "Patch3_canonical_dedupe": (
        "Duplicate hard gate still failed: duplicate captions increased by 9; "
        "macro score slightly declined."
    ),
    "Patch3_1_duplicate_guard": (
        "Final hard gate still failed: duplicate captions increased 0 -> 2, "
        "float-caption accuracy declined, and macro body score declined."
    ),
}

METRIC_KEYS = [
    "float_caption_attachment_accuracy",
    "pred_caption_count",
    "missing_caption_count",
    "duplicate_caption_count",
    "true_duplicate_caption_count",
    "caption_as_paragraph_count",
    "wrong_float_type_pairing_count",
    "generated_structure_validity",
    "macro_structure_score_body",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"missing": True, "path": str(path)}
    return json.loads(path.read_text())


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-12:
            return str(int(round(value)))
        return f"{value:.6f}"
    return str(value)


def build_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, path in SUMMARY_PATHS.items():
        data = load_json(path)
        baseline = data.get("baseline", {}) if isinstance(data, dict) else {}
        experimental = data.get("experimental", {}) if isinstance(data, dict) else {}
        delta = data.get("delta", {}) if isinstance(data, dict) else {}
        row: dict[str, Any] = {
            "patch": name,
            "source_path": str(path),
            "status": "missing" if data.get("missing") else "available",
            "decision": DECISIONS[name],
            "main_benefit": BENEFITS[name],
            "main_blocker": BLOCKERS[name],
        }
        for key in METRIC_KEYS:
            row[f"flag_off_{key}"] = baseline.get(key)
            row[f"flag_on_{key}"] = experimental.get(key)
            row[f"delta_{key}"] = delta.get(key)
        rows.append(row)
    return rows


def write_patch_history(rows: list[dict[str, Any]]) -> None:
    path = OUT / "float_caption_patch_history_summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def patch_history_markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "Patch",
        "Decision",
        "Accuracy off",
        "Accuracy on",
        "Delta",
        "Pred off",
        "Pred on",
        "Missing off",
        "Missing on",
        "Dup off",
        "Dup on",
        "Cap-as-para off",
        "Cap-as-para on",
        "Macro off",
        "Macro on",
    ]
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("| " + " | ".join(["---"] + ["---:" for _ in headers[1:]]) + " |")
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["patch"],
                    row["decision"],
                    fmt(row["flag_off_float_caption_attachment_accuracy"]),
                    fmt(row["flag_on_float_caption_attachment_accuracy"]),
                    fmt(row["delta_float_caption_attachment_accuracy"]),
                    fmt(row["flag_off_pred_caption_count"]),
                    fmt(row["flag_on_pred_caption_count"]),
                    fmt(row["flag_off_missing_caption_count"]),
                    fmt(row["flag_on_missing_caption_count"]),
                    fmt(row["flag_off_duplicate_caption_count"]),
                    fmt(row["flag_on_duplicate_caption_count"]),
                    fmt(row["flag_off_caption_as_paragraph_count"]),
                    fmt(row["flag_on_caption_as_paragraph_count"]),
                    fmt(row["flag_off_macro_structure_score_body"]),
                    fmt(row["flag_on_macro_structure_score_body"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_metric_note(baseline_drift: dict[str, Any], metric_version_status: str, metric_version: str) -> None:
    patch2 = baseline_drift.get("patch2_flag_off", {})
    patch3 = baseline_drift.get("patch3_flag_off", {})
    delta = baseline_drift.get("delta_patch3_minus_patch2", {})
    note = f"""# Float Caption Metric Version Note

## Status

- metric_version_status: `{metric_version_status}`
- caption_metric_version: `{metric_version}`
- old reports before Patch3/Patch3.1 are historical references only.
- future FloatCaptionLayout comparisons must be same-code and same-metric.

## Why The Version Changed

Patch2 -> Patch3/Patch3.1 changed the caption accounting path. The large flag-off
drift is explained by a stricter caption metric/converter path, not by a production
default change.

Observed Patch2 flag-off -> Patch3.1 flag-off drift:

| Metric | Patch2 flag-off | Patch3.1 flag-off | Delta |
| --- | ---: | ---: | ---: |
| float_caption_attachment_accuracy | {fmt(patch2.get("float_caption_attachment_accuracy"))} | {fmt(patch3.get("float_caption_attachment_accuracy"))} | {fmt(delta.get("float_caption_attachment_accuracy"))} |
| pred_caption_count | {fmt(patch2.get("pred_caption_count"))} | {fmt(patch3.get("pred_caption_count"))} | {fmt(delta.get("pred_caption_count"))} |
| missing_caption_count | {fmt(patch2.get("missing_caption_count"))} | {fmt(patch3.get("missing_caption_count"))} | {fmt(delta.get("missing_caption_count"))} |
| duplicate_caption_count | {fmt(patch2.get("duplicate_caption_count"))} | {fmt(patch3.get("duplicate_caption_count"))} | {fmt(delta.get("duplicate_caption_count"))} |
| macro_structure_score_body | {fmt(patch2.get("macro_structure_score_body"))} | {fmt(patch3.get("macro_structure_score_body"))} | {fmt(delta.get("macro_structure_score_body"))} |

Remote drift interpretation:

```json
{json.dumps(baseline_drift.get("interpretation", {}), indent=2, ensure_ascii=False)}
```

## Comparison Rule

Do not compare Patch2 absolute values directly against Patch3/Patch3.1 absolute values.
Compare only within the same run family:

- Patch2 flag-off vs Patch2 flag-on;
- Patch3 flag-off vs Patch3 flag-on;
- Patch3.1 flag-off vs Patch3.1 flag-on.

The old baseline remains useful as historical evidence that the implementation path was
explored, but it is not a locked metric baseline.
"""
    (OUT / "float_caption_metric_version_note.md").write_text(note)


def write_final_json(rows: list[dict[str, Any]], metric_version_status: str, metric_version: str) -> None:
    final_decision = {
        "decision": "diagnostic_only",
        "production_default": "unchanged",
        "experimental_flag_default": "off",
        "no_patch4_planned": True,
        "metric_version_status": metric_version_status,
        "caption_metric_version": metric_version,
        "old_baseline_comparability": "historical_reference_only",
        "v8_only_confirmation": {
            "current_fact_layer": "v8 full observable facts",
            "no_fallback_to_old_v7": True,
            "legacy_field_names": "source_v7_ids / v7_id are provenance names only",
            "mainline": (
                "v8 full observable facts -> v8 atomic/reflow -> deterministic merge + "
                "contentlist merge hint -> RenderTreeIR -> IR renderer"
            ),
        },
        "default_flag_confirmation": {
            "src/reasoning/v8_render_tree.py": "enable_float_caption_layout defaults to False",
            "scripts/pipeline/run_v8_layout_reconstruction.py": (
                "--enable-float-caption-layout-experimental is opt-in"
            ),
        },
        "what_worked": [
            "caption-as-paragraph suppression reached zero in flag-on runs",
            "converter normalization partially recovered rendered captions",
            (
                "algorithm caption initial route worked in a limited way, but remaining "
                "cases need AlgorithmRegion rather than FloatCaptionLayout expansion"
            ),
            (
                "diagnostics, promotion funnel, compile smoke, and trace audit provided "
                "useful failure localization"
            ),
        ],
        "what_blocked_production": [
            "duplicate hard gate failed after every materialization patch family",
            (
                "float-caption attachment accuracy did not improve stably under the "
                "locked same-code metric"
            ),
            "macro body score did not improve in Patch3.1",
            (
                "metadata/crop materialization remained unstable and left many "
                "crop/metadata-only candidates unresolved"
            ),
            (
                "caption metric/counting drift means old absolute values cannot be "
                "used as locked baselines"
            ),
        ],
        "next_recommended_work": [
            "AlgorithmRegion / pseudocode region audit",
            "lower-level float/table layout fact extraction audit",
            "FrontMatterExtractor Phase 0",
            "ROI role dataset audit",
        ],
        "patch_history": rows,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (OUT / "float_caption_final_decision.json").write_text(
        json.dumps(final_decision, indent=2, ensure_ascii=False)
    )


def write_final_report(rows: list[dict[str, Any]], metric_version_status: str, metric_version: str) -> None:
    report = f"""# V8 Float-Caption Finalization And Metric Version Lock

## Final Decision

- FloatCaptionLayout final decision: **diagnostic_only**.
- Production default: **unchanged**.
- Experimental flag default: **off**.
- No Patch4 is planned.
- No training / no MinerU / no relabel / no rebuild / no GNN / no E2E was run in this finalization pass.

The experimental FloatCaptionLayout path remains useful as a diagnostic branch, but it should not enter the default v8 render path.

## v8-only Confirmation

- Current fact layer: v8 full observable facts.
- No fallback to old v7 fact layer.
- Legacy names such as `source_v7_ids` / `v7_id` are provenance names only.
- Current mainline remains:

```text
v8 full observable facts
  -> v8 atomic/reflow
  -> deterministic merge + contentlist merge hint
  -> RenderTreeIR
  -> IR renderer
```

Flag/default check:

- `src/reasoning/v8_render_tree.py`: `enable_float_caption_layout` defaults to `False`.
- `scripts/pipeline/run_v8_layout_reconstruction.py`: `--enable-float-caption-layout-experimental` is opt-in.

## Patch History Summary

{patch_history_markdown_table(rows)}

## What Worked

- Caption-as-paragraph suppression worked consistently: flag-on runs reduced it from 21 to 0.
- Converter normalization partially worked; Patch2 reduced rendered-not-converted style failures and improved missing caption count.
- The first algorithm-caption route showed limited signal, but the remaining misses are mostly candidate/region issues and should move to an AlgorithmRegion pass.
- The diagnostic stack worked: same-code A/B, compile smoke, promotion funnel, trace audit, duplicate delta audit, and baseline drift audit localized failures clearly.

## What Blocked Production

- Duplicate hard gate failed: Patch2 increased duplicate captions by 21, Patch3 by 9, and Patch3.1 still increased duplicates from 0 to 2.
- Float-caption attachment accuracy was not stable: Patch3.1 decreased from 0.562570 to 0.559709.
- Macro body score did not improve in Patch3.1: 0.843734 -> 0.842702.
- Materialization remained unstable: crop/metadata-only and promoted-not-rendered style failures were not cleanly solved.
- Metric/counting drift appeared between Patch2 and Patch3/Patch3.1, so older absolute values cannot be used as locked baselines.

## Metric Version Note

- metric_version_status: `{metric_version_status}`
- caption_metric_version: `{metric_version}`
- Old Patch1/Patch2 absolute values are historical references, not directly comparable to Patch3/Patch3.1 absolute values.
- Future FloatCaptionLayout reports must compare same-code / same-metric flag-off vs flag-on.

The Patch2 -> Patch3/Patch3.1 flag-off drift is explained by stricter caption counting/converter behavior, including panel/synthetic caption handling and canonical caption selection. The remote drift summary recommends `patch3_strict_caption_metric_v1`, and this finalization locks that as the current caption metric version.

## Module Disposition

Keep as diagnostic/supporting tools:

- caption trace audit;
- promotion funnel audit;
- converter normalization tests;
- canonical caption selection diagnostics;
- duplicate/subfigure risk review.

Do not enter default render path:

- FloatCaptionLayout experimental materialization;
- expanded crop/metadata caption promotion;
- placeholder expansion;
- any Patch4-style pairing/materialization expansion.

## Next Recommended Work

1. AlgorithmRegion / pseudocode region audit.
2. Lower-level float/table layout fact extraction audit.
3. FrontMatterExtractor Phase 0.
4. ROI role dataset audit.

## Generated Artifacts

- `float_caption_patch_history_summary.csv`
- `float_caption_metric_version_note.md`
- `float_caption_final_decision.json`
- `FLOAT_CAPTION_LAYOUT_FINAL_DECISION_REPORT.md`
"""
    (OUT / "FLOAT_CAPTION_LAYOUT_FINAL_DECISION_REPORT.md").write_text(report)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_patch_history(rows)

    baseline_drift = load_json(BASE / "patch3_1_duplicate_guard" / "baseline_drift_summary.json")
    metric_version_status = "changed_converter_behavior"
    metric_version = "patch3_strict_caption_metric_v1"

    write_metric_note(baseline_drift, metric_version_status, metric_version)
    write_final_json(rows, metric_version_status, metric_version)
    write_final_report(rows, metric_version_status, metric_version)

    print(
        json.dumps(
            {
                "output_dir": str(OUT),
                "files": [
                    "FLOAT_CAPTION_LAYOUT_FINAL_DECISION_REPORT.md",
                    "float_caption_patch_history_summary.csv",
                    "float_caption_metric_version_note.md",
                    "float_caption_final_decision.json",
                ],
                "decision": "diagnostic_only",
                "metric_version": metric_version,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
