#!/usr/bin/env python3
"""Generate reproducible train/calibration commands for the v7 ablation matrix.

This script does not train models and does not mutate graph data.  It turns the
JSON matrix into a shell script so the same experiment set can be launched on
AutoDL after the clean v7 manifest is ready.
"""

from __future__ import annotations

import argparse
import json
import shlex
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix", type=Path, default=REPO_ROOT / "configs/ablation_matrix_v3.json")
    parser.add_argument("--output-sh", type=Path, default=REPO_ROOT / "data/08_runs/run_ablation_matrix_v3.sh")
    parser.add_argument("--output-json", type=Path, default=REPO_ROOT / "data/09_eval_reports/ablation_matrix_v3_commands.json")
    parser.add_argument("--only", default="", help="Comma-separated experiment names to include.")
    parser.add_argument("--output-root", help="Optional override for common.output_root in the matrix.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without writing files.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    if args.output_root:
        matrix.setdefault("common", {})["output_root"] = args.output_root
    selected = {name.strip() for name in str(args.only or "").split(",") if name.strip()}
    commands = build_commands(matrix, selected_names=selected)
    payload = {
        "schema_version": "ablation_command_manifest_v1",
        "matrix": str(args.matrix),
        "commands": commands,
    }
    script_text = render_shell_script(commands)
    if args.dry_run:
        print(script_text)
        return 0
    args.output_sh.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_sh.write_text(script_text, encoding="utf-8")
    args.output_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.output_sh.chmod(0o755)
    print(f"experiments={len(commands)}")
    print(f"wrote {args.output_sh}")
    print(f"wrote {args.output_json}")
    return 0


def build_commands(matrix: dict[str, Any], *, selected_names: set[str]) -> list[dict[str, Any]]:
    common = dict(matrix.get("common", {}))
    calibration = dict(matrix.get("calibration", {}))
    repeat_seeds = common.pop("repeat_seeds", None)
    if repeat_seeds is None:
        repeat_seeds = [common.get("seed", 7)]
    commands: list[dict[str, Any]] = []
    for experiment in matrix.get("experiments", []):
        name = str(experiment["name"])
        if selected_names and name not in selected_names:
            continue
        for seed in repeat_seeds:
            exp_args = dict(common)
            exp_args.update(dict(experiment.get("args", {})))
            exp_args["seed"] = int(seed)
            output_root = Path(str(common["output_root"]))
            output_dir = output_root / name / f"seed_{int(seed)}"
            train_cmd = build_train_command(exp_args, output_dir=output_dir)
            calibrate_cmd = None
            if bool(calibration.get("enabled", False)):
                calibrate_cmd = build_calibration_command(exp_args, calibration, output_dir=output_dir)
            commands.append(
                {
                    "name": name,
                    "seed": int(seed),
                    "family": experiment.get("family", "unspecified"),
                    "purpose": experiment.get("purpose", ""),
                    "output_dir": str(output_dir),
                    "train": train_cmd,
                    "calibrate": calibrate_cmd,
                }
            )
    return commands


def build_train_command(args: dict[str, Any], *, output_dir: Path) -> list[str]:
    command = [
        str(args.get("python_bin", "python")),
        "scripts/pipeline/train_edge_gnn_full.py",
        "--root",
        str(args["root"]),
        "--manifest",
        str(args["manifest"]),
        "--output-dir",
        str(output_dir),
    ]
    arg_map = {
        "epochs": "--epochs",
        "batch_size": "--batch-size",
        "lr": "--lr",
        "weight_decay": "--weight-decay",
        "hidden_dim": "--hidden-dim",
        "heads": "--heads",
        "num_layers": "--num-layers",
        "predictor_hidden_dims": "--predictor-hidden-dims",
        "semantic_hidden_dim": "--semantic-hidden-dim",
        "layout_hidden_dim": "--layout-hidden-dim",
        "dropout": "--dropout",
        "loss": "--loss",
        "class_weights": "--class-weights",
        "class_weight_values": "--class-weight-values",
        "gamma": "--gamma",
        "positive_weight_multiplier": "--positive-weight-multiplier",
        "train_negative_dropout": "--train-negative-dropout",
        "ohem_negative_ratio": "--ohem-negative-ratio",
        "ohem_min_negatives": "--ohem-min-negatives",
        "device": "--device",
        "num_workers": "--num-workers",
        "seed": "--seed",
        "selection_metric": "--selection-metric",
        "edge_feature_mode": "--edge-feature-mode",
        "message_edge_mode": "--message-edge-mode",
        "prediction_architecture": "--prediction-architecture",
        "merge_gate_mode": "--merge-gate-mode",
        "merge_gate_logit": "--merge-gate-logit",
        "gaussian_edge_feature_mode": "--gaussian-edge-feature-mode",
        "gaussian_sigma": "--gaussian-sigma",
        "ablate_node_groups": "--ablate-node-groups",
        "ablate_edge_groups": "--ablate-edge-groups",
        "ablate_edge_fields": "--ablate-edge-fields",
    }
    for key, flag in arg_map.items():
        if key in args and args[key] not in (None, ""):
            command.extend([flag, str(args[key])])
    if "predictor_layer_norm" in args:
        command.append("--predictor-layer-norm" if bool(args["predictor_layer_norm"]) else "--no-predictor-layer-norm")
    return command


def build_calibration_command(common: dict[str, Any], calibration: dict[str, Any], *, output_dir: Path) -> list[str]:
    command = [
        str(common.get("python_bin", "python")),
        "scripts/pipeline/calibrate_edge_thresholds.py",
        "--root",
        str(common["root"]),
        "--manifest",
        str(common["manifest"]),
        "--checkpoint",
        str(output_dir / "best_model.pth"),
        "--output-json",
        str(output_dir / "threshold_calibration.json"),
        "--batch-size",
        str(common.get("batch_size", 8)),
        "--device",
        str(common.get("device", "auto")),
        "--num-workers",
        str(common.get("num_workers", 0)),
        "--seed",
        str(common.get("seed", 7)),
        "--tau-min",
        str(calibration.get("tau_min", 0.05)),
        "--tau-max",
        str(calibration.get("tau_max", 0.95)),
        "--tau-step",
        str(calibration.get("tau_step", 0.01)),
        "--mode",
        str(calibration.get("mode", "threshold_priority")),
    ]
    if calibration.get("min_merge_precision") not in (None, ""):
        command.extend(["--min-merge-precision", str(calibration["min_merge_precision"])])
    if calibration.get("precision_floors") not in (None, ""):
        command.extend(["--precision-floors", str(calibration["precision_floors"])])
    if bool(calibration.get("apply_merge_gates", False)):
        command.append("--apply-merge-gates")
    return command


def render_shell_script(commands: list[dict[str, Any]]) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd \"$(dirname \"$0\")/../..\"",
        "",
        "export TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}",
        "export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}",
        "export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}",
        "export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}",
        "export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}",
        "",
        "echo \"[ablation] started at $(date)\"",
    ]
    for command in commands:
        name = command["name"]
        output_dir = command["output_dir"]
        lines.extend(
            [
                "",
                f"echo \"[ablation] {name}: {command.get('purpose', '')}\"",
                f"mkdir -p {shlex.quote(output_dir)}",
                shell_join(command["train"]) + f" 2>&1 | tee {shlex.quote(str(Path(output_dir) / 'train.log'))}",
            ]
        )
        if command.get("calibrate"):
            lines.append(shell_join(command["calibrate"]) + f" 2>&1 | tee {shlex.quote(str(Path(output_dir) / 'calibrate.log'))}")
    lines.extend(["", "echo \"[ablation] finished at $(date)\"", ""])
    return "\n".join(lines)


def shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


if __name__ == "__main__":
    raise SystemExit(main())
