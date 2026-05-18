#!/usr/bin/env python3
"""Run the current full evaluation suite.

This orchestrates the paper-facing current model evaluation:

1. current ablation matrix
2. ablation summary
3. current E2E generator evaluation
4. Nougat paired comparison
5. rollup report generation

It is safe to skip stages when reusing existing outputs.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PYTHON = Path("/root/miniconda3/envs/pdf2latex/bin/python")
if not DEFAULT_PYTHON.exists():
    DEFAULT_PYTHON = Path(sys.executable)

DEFAULT_MATRIX = Path("configs/ablation_matrix_current.json")
DEFAULT_ABLATION_ROOT = Path("data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current")
DEFAULT_ABLATION_JSON = Path("data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.json")
DEFAULT_ABLATION_CSV = Path("data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current_summary.csv")
DEFAULT_MANIFEST = Path("data/00_manifests/v7_floatproxy_adapter_20260516_205926_trainable_recall98.json")
DEFAULT_CHECKPOINT = DEFAULT_ABLATION_ROOT / "M06_current_main_merge_gate/seed_7/best_model.pth"
DEFAULT_E2E_DIR = Path("data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615")
DEFAULT_NOUGAT_DIR = Path("data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615")
DEFAULT_ROLLUP_DIR = Path("data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--e2e-output-dir", type=Path, default=DEFAULT_E2E_DIR)
    parser.add_argument("--nougat-output-dir", type=Path, default=DEFAULT_NOUGAT_DIR)
    parser.add_argument("--rollup-output-dir", type=Path, default=DEFAULT_ROLLUP_DIR)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="test")
    parser.add_argument("--skip-ablation", action="store_true")
    parser.add_argument("--skip-e2e", action="store_true")
    parser.add_argument("--skip-nougat", action="store_true")
    parser.add_argument("--skip-rollup", action="store_true")
    parser.add_argument("--reuse-existing-ablation", action="store_true")
    parser.add_argument("--reuse-existing-e2e", action="store_true")
    parser.add_argument("--reuse-existing-nougat", action="store_true")
    parser.add_argument("--nougat-skip-model", action="store_true", help="Reuse existing Nougat .mmd files if present.")
    parser.add_argument("--no-network-turbo", action="store_true")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    os.chdir(PROJECT_ROOT)
    setup_env()

    if not args.no_network_turbo:
        maybe_source_network_turbo()

    if not args.skip_ablation:
        if args.reuse_existing_ablation and DEFAULT_ABLATION_JSON.exists():
            print(f"[suite] reuse ablation summary: {DEFAULT_ABLATION_JSON}", flush=True)
        else:
            run([args.python_bin, "scripts/pipeline/prepare_ablation_suite.py", "--matrix", args.matrix])
            run(["bash", "data/08_runs/run_ablation_matrix_current.sh"])
            run(
                [
                    args.python_bin,
                    "scripts/pipeline/summarize_ablation_results.py",
                    "--root",
                    args.ablation_root,
                    "--output-json",
                    DEFAULT_ABLATION_JSON,
                    "--output-csv",
                    DEFAULT_ABLATION_CSV,
                ]
            )

    if not args.skip_e2e:
        if args.reuse_existing_e2e and (args.e2e_output_dir / "e2e_comparison_manifest.json").exists():
            print(f"[suite] reuse E2E manifest: {args.e2e_output_dir / 'e2e_comparison_manifest.json'}", flush=True)
        else:
            run(
                [
                    args.python_bin,
                    "scripts/pipeline/run_current_e2e_comparison.py",
                    "--manifest",
                    args.manifest,
                    "--checkpoint",
                    args.checkpoint,
                    "--output-dir",
                    args.e2e_output_dir,
                    "--limit",
                    args.limit,
                    "--split",
                    args.split,
                    "--clean-output-dir",
                ]
            )

    if not args.skip_nougat:
        if args.reuse_existing_nougat and (args.nougat_output_dir / "nougat_comparison_manifest.json").exists():
            print(f"[suite] reuse Nougat manifest: {args.nougat_output_dir / 'nougat_comparison_manifest.json'}", flush=True)
        else:
            command = [
                args.python_bin,
                "scripts/pipeline/run_nougat_comparison.py",
                "--manifest",
                args.manifest,
                "--output-dir",
                args.nougat_output_dir,
                "--ours-e2e-manifest",
                args.e2e_output_dir / "e2e_comparison_manifest.json",
                "--limit",
                args.limit,
                "--split",
                args.split,
                "--clean-output-dir",
            ]
            if args.nougat_skip_model:
                command.append("--skip-nougat")
            run(command)

    if not args.skip_rollup:
        run(
            [
                args.python_bin,
                "scripts/pipeline/collect_current_eval_results.py",
                "--ablation-summary",
                DEFAULT_ABLATION_JSON,
                "--e2e-manifest",
                args.e2e_output_dir / "e2e_comparison_manifest.json",
                "--nougat-manifest",
                args.nougat_output_dir / "nougat_comparison_manifest.json",
                "--output-dir",
                args.rollup_output_dir,
            ]
        )
    return 0


def setup_env() -> None:
    defaults = {
        "TOKENIZERS_PARALLELISM": "false",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, value in defaults.items():
        os.environ.setdefault(key, value)


def maybe_source_network_turbo() -> None:
    path = Path("/etc/network_turbo")
    if not path.exists():
        return
    run(["bash", "-lc", "source /etc/network_turbo >/dev/null 2>&1 || true"])


def run(command: list[object]) -> None:
    text = [str(item) for item in command]
    print("[suite] $ " + " ".join(text), flush=True)
    subprocess.run(text, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    raise SystemExit(main())
