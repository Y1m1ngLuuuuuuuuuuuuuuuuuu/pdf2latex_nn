#!/usr/bin/env python3
"""Run the clean5144 ablation suite with explicit rules-only E2E controls.

The suite is intentionally split into two evidence families:

* edge ablation: trained GNN edge-relation variants on the clean5144 graph set;
* E2E relation-source ablation: final generator output when relations come from
  the M06 GNN checkpoint versus deterministic no-GNN rule baselines.

M06 is expected to be pre-trained.  The runner links that run into the edge
ablation root and trains only the remaining GNN ablations.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MATRIX = Path("configs/ablation_matrix_clean5144_20260520.json")
DEFAULT_MANIFEST = Path("data/00_manifests/clean5144_mainline_20260520/clean5144_canonical_manifest.json")
DEFAULT_MAIN_TRAIN_DIR = Path(
    "data/09_eval_reports/clean5144_mainline_20260520/train/M06_current_main_merge_gate_seed7"
)
DEFAULT_EDGE_ROOT = Path("data/09_eval_reports/clean5144_mainline_20260520/edge_ablation_with_rules_20260520")
DEFAULT_E2E_ROOT = Path("data/09_eval_reports/clean5144_mainline_20260520/e2e_relation_source_ablation_20260520")
DEFAULT_RUN_DIR = Path("data/08_runs")
DEFAULT_REPORT = Path("data/09_eval_reports/clean5144_mainline_20260520/ABLATION_WITH_RULES_20260520_REPORT.md")
DEFAULT_DOC_REPORT = Path("docs/ablation_results_clean5144_with_rules_20260520.md")

TRAIN_EXPERIMENTS = [
    "M05_no_merge_gate",
    "M07_gaussian_edge_feature",
    "A00_old_shared_gat",
    "A01_no_message_passing",
    "A02_no_type_aware_message_mask",
    "F00_no_scibert",
    "F01_no_geometry_layout",
    "F02_no_v7_reading_flow",
    "E00_no_punctuation",
    "E01_no_gutter_overlap",
    "T00_no_ohem",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", default="/root/miniconda3/envs/pdf2latex/bin/python")
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--main-train-dir", type=Path, default=DEFAULT_MAIN_TRAIN_DIR)
    parser.add_argument("--edge-root", type=Path, default=DEFAULT_EDGE_ROOT)
    parser.add_argument("--e2e-root", type=Path, default=DEFAULT_E2E_ROOT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--doc-report", type=Path, default=DEFAULT_DOC_REPORT)
    parser.add_argument("--e2e-limit", type=int, default=20)
    parser.add_argument("--e2e-split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--m06-merge-threshold", type=float, default=0.76)
    parser.add_argument("--m06-parent-threshold", type=float, default=0.79)
    parser.add_argument("--skip-edge-training", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--clean-e2e-output", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-e2e-error", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    os.chdir(REPO_ROOT)
    validate_inputs(args)
    link_main_model(args)

    command_manifest = args.edge_root / "ablation_matrix_clean5144_with_rules_commands.json"
    run_script = args.run_dir / "run_edge_ablation_clean5144_with_rules_20260520.sh"
    if not args.skip_edge_training:
        prepare_edge_ablation(args, run_script=run_script, command_manifest=command_manifest)
        run(["bash", str(run_script)])

    edge_summary_json = args.edge_root / "summary.json"
    edge_summary_csv = args.edge_root / "summary.csv"
    summarize_edge_ablation(args, output_json=edge_summary_json, output_csv=edge_summary_csv)

    e2e_payload: dict[str, Any] = {"skipped": bool(args.skip_e2e), "runs": []}
    if not args.skip_e2e:
        e2e_payload = run_relation_source_e2e(args)

    write_final_reports(args, edge_summary_json=edge_summary_json, edge_summary_csv=edge_summary_csv, e2e_payload=e2e_payload)
    return 0


def validate_inputs(args: argparse.Namespace) -> None:
    for path in (args.matrix, args.manifest, args.main_train_dir / "best_model.pth", args.main_train_dir / "threshold_calibration.json"):
        if not path.exists():
            raise FileNotFoundError(path)


def link_main_model(args: argparse.Namespace) -> None:
    target = args.edge_root / "M06_current_main_merge_gate" / "seed_7"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        return
    try:
        target.symlink_to(args.main_train_dir.resolve(), target_is_directory=True)
    except OSError:
        shutil.copytree(args.main_train_dir, target)


def prepare_edge_ablation(args: argparse.Namespace, *, run_script: Path, command_manifest: Path) -> None:
    run_script.parent.mkdir(parents=True, exist_ok=True)
    command_manifest.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            args.python_bin,
            "scripts/pipeline/prepare_ablation_suite.py",
            "--matrix",
            str(args.matrix),
            "--output-sh",
            str(run_script),
            "--output-json",
            str(command_manifest),
            "--output-root",
            str(args.edge_root),
            "--only",
            ",".join(TRAIN_EXPERIMENTS),
        ]
    )


def summarize_edge_ablation(args: argparse.Namespace, *, output_json: Path, output_csv: Path) -> None:
    run(
        [
            args.python_bin,
            "scripts/pipeline/summarize_ablation_results.py",
            "--root",
            str(args.edge_root),
            "--output-json",
            str(output_json),
            "--output-csv",
            str(output_csv),
        ]
    )


def run_relation_source_e2e(args: argparse.Namespace) -> dict[str, Any]:
    args.e2e_root.mkdir(parents=True, exist_ok=True)
    runs = [
        {
            "name": "M06_gnn_relation_source",
            "kind": "gnn",
            "command": [
                args.python_bin,
                "scripts/pipeline/run_m05_e2e_comparison.py",
                "--manifest",
                str(args.manifest),
                "--checkpoint",
                str(args.main_train_dir / "best_model.pth"),
                "--output-dir",
                str(args.e2e_root / "M06_gnn_relation_source"),
                "--limit",
                str(args.e2e_limit),
                "--split",
                args.e2e_split,
                "--merge-threshold",
                str(args.m06_merge_threshold),
                "--parent-threshold",
                str(args.m06_parent_threshold),
                "--heading-skeleton-mode",
                "stack",
                "--renderer",
                "ir",
            ],
        },
        {
            "name": "R00_rules_only_no_merge",
            "kind": "rules_only",
            "command": [
                args.python_bin,
                "scripts/pipeline/run_rules_only_e2e_comparison.py",
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.e2e_root / "R00_rules_only_no_merge"),
                "--rules-mode",
                "rules_only_no_merge",
                "--limit",
                str(args.e2e_limit),
                "--split",
                args.e2e_split,
            ],
        },
        {
            "name": "R01_rules_only_deterministic_merge",
            "kind": "rules_only",
            "command": [
                args.python_bin,
                "scripts/pipeline/run_rules_only_e2e_comparison.py",
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(args.e2e_root / "R01_rules_only_deterministic_merge"),
                "--rules-mode",
                "rules_only_deterministic_merge",
                "--limit",
                str(args.e2e_limit),
                "--split",
                args.e2e_split,
            ],
        },
    ]
    payload: dict[str, Any] = {"skipped": False, "runs": []}
    for item in runs:
        command = list(item["command"])
        if args.skip_compile:
            command.append("--skip-compile")
        if args.clean_e2e_output:
            command.append("--clean-output-dir")
        status = "completed"
        error = None
        try:
            run(command)
        except subprocess.CalledProcessError as exc:
            status = "failed"
            error = str(exc)
            if not args.continue_on_e2e_error:
                raise
        summary = read_e2e_summary(args.e2e_root / item["name"] / "e2e_comparison_manifest.json")
        payload["runs"].append(
            {
                "name": item["name"],
                "kind": item["kind"],
                "status": status,
                "error": error,
                "output_dir": str(args.e2e_root / item["name"]),
                "summary": summary,
            }
        )
    (args.e2e_root / "relation_source_ablation_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_relation_source_csv(args.e2e_root / "relation_source_ablation_summary.csv", payload)
    return payload


def read_e2e_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return dict(payload.get("summary", {}))


def write_relation_source_csv(path: Path, payload: dict[str, Any]) -> None:
    import csv

    fields = [
        "name",
        "kind",
        "status",
        "documents",
        "macro_structure_score",
        "heading_tree_accuracy",
        "reading_order_accuracy",
        "paragraph_boundary_f1",
        "paragraph_text_coverage_f1",
        "section_attachment_body_no_float_f1",
        "float_caption_attachment_accuracy",
        "reference_section_completeness",
        "generated_structure_validity",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in payload.get("runs", []):
            summary = row.get("summary", {})
            writer.writerow({field: row.get(field, summary.get(field)) for field in fields})


def write_final_reports(
    args: argparse.Namespace,
    *,
    edge_summary_json: Path,
    edge_summary_csv: Path,
    e2e_payload: dict[str, Any],
) -> None:
    edge_payload = json.loads(edge_summary_json.read_text(encoding="utf-8")) if edge_summary_json.exists() else {}
    edge_rows = edge_payload.get("summary", [])
    lines = [
        "# Clean5144 Ablation With Rules-Only E2E Controls",
        "",
        "## Scope",
        "",
        "- Dataset: clean5144 canonical manifest.",
        "- Edge ablation: trained 3-class GNN relation models.",
        "- E2E relation-source ablation: compares M06 GNN relations with no-GNN deterministic rules.",
        "- M06 main checkpoint is reused from the completed clean5144 main training run.",
        "",
        "## Artifacts",
        "",
        f"- Matrix: `{args.matrix}`",
        f"- Edge ablation root: `{args.edge_root}`",
        f"- Edge summary JSON: `{edge_summary_json}`",
        f"- Edge summary CSV: `{edge_summary_csv}`",
        f"- E2E relation-source root: `{args.e2e_root}`",
        "",
        "## Edge Ablation Summary",
        "",
        "| experiment | positive macro F1 | MERGE F1 | PARENT F1 | tau_merge | tau_parent |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in edge_rows:
        lines.append(
            "| {experiment} | {pos} | {merge} | {parent} | {tm} | {tp} |".format(
                experiment=row.get("experiment", ""),
                pos=fmt(row.get("calibrated_test_positive_macro_f1_mean")),
                merge=fmt(row.get("calibrated_test_merge_f1_mean")),
                parent=fmt(row.get("calibrated_test_parent_f1_mean")),
                tm=fmt(row.get("tau_merge_mean")),
                tp=fmt(row.get("tau_parent_mean")),
            )
        )
    lines.extend(
        [
            "",
            "## E2E Relation-Source Ablation",
            "",
            "| relation source | documents | macro | heading | reading order | paragraph coverage | section body/no-float | float-caption | refs | validity |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in e2e_payload.get("runs", []):
        summary = row.get("summary", {})
        lines.append(
            "| {name} | {docs} | {macro} | {heading} | {reading} | {coverage} | {section} | {floatcap} | {refs} | {valid} |".format(
                name=row.get("name", ""),
                docs=summary.get("documents", ""),
                macro=fmt(summary.get("macro_structure_score")),
                heading=fmt(summary.get("heading_tree_accuracy")),
                reading=fmt(summary.get("reading_order_accuracy")),
                coverage=fmt(summary.get("paragraph_text_coverage_f1")),
                section=fmt(summary.get("section_attachment_body_no_float_f1")),
                floatcap=fmt(summary.get("float_caption_attachment_accuracy")),
                refs=fmt(summary.get("reference_section_completeness")),
                valid=fmt(summary.get("generated_structure_validity")),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `A01_no_message_passing` is still a learned edge classifier; it is not a pure no-GNN document reconstruction baseline.",
            "- `R00_rules_only_no_merge` disables learned relation predictions entirely and uses heading-stack/full-v7 rendering only.",
            "- `R01_rules_only_deterministic_merge` adds conservative adjacent text merge edges, still without loading a GNN checkpoint.",
            "- Use the edge table for model-design claims and the relation-source table for final generator dependency claims.",
            "",
        ]
    )
    text = "\n".join(lines)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.doc_report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(text, encoding="utf-8")
    args.doc_report.write_text(text, encoding="utf-8")


def fmt(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def run(command: list[str]) -> None:
    print("[run]", " ".join(command), flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
