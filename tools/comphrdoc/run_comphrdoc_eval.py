#!/usr/bin/env python3
"""Run the official CompHRDoc evaluation scripts with local paths."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPO = PROJECT_ROOT / "third_party" / "CompHRDoc"
DEFAULT_PYTHON = PROJECT_ROOT / ".venv_comphrdoc" / "bin" / "python"
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, load_config, read_json, safe_doc_id  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--gt-folder", type=Path, required=True)
    parser.add_argument("--pred-folder", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--comphrdoc-repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["teds", "reading_order", "classify"],
        choices=["teds", "reading_order", "classify"],
    )
    parser.add_argument("--reading-workers", type=int, default=1)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if args.output_dir is None:
        cfg = load_config(args.config)
        run_name = args.run_name or f"{args.pred_folder.name}_official_eval"
        args.output_dir = config_path(cfg, "outputs", "report_root") / run_name
    if args.skip_existing and (args.output_dir / "comphrdoc_eval_summary.json").exists():
        print(f"[comphrdoc] skip existing official eval -> {args.output_dir}")
        return 0
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt_folder = args.gt_folder.resolve()
    pred_folder = args.pred_folder.resolve()
    if args.offset or args.limit:
        gt_folder, pred_folder = make_eval_subsets(args, gt_folder, pred_folder)
    tool_dir = args.comphrdoc_repo / "evaluation" / "hrdoc_tool"
    if not tool_dir.exists():
        raise FileNotFoundError(f"CompHRDoc evaluation tools not found: {tool_dir}")
    python_bin = str(args.python if args.python.exists() else Path(sys.executable))
    runs: list[dict[str, Any]] = []
    for task in args.tasks:
        script = {
            "teds": "teds_eval.py",
            "reading_order": "reading_order_eval.py",
            "classify": "classify_eval.py",
        }[task]
        cmd = [
            python_bin,
            script,
            "--gt_folder",
            str(gt_folder),
            "--pred_folder",
            str(pred_folder),
        ]
        if task == "reading_order":
            cmd += ["--num_workers", str(args.reading_workers)]
        log_path = args.output_dir / f"{task}.log"
        print(f"[comphrdoc][{task}] {' '.join(cmd)}", flush=True)
        proc = subprocess.run(
            cmd,
            cwd=str(tool_dir),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(proc.stdout, encoding="utf-8")
        runs.append(
            {
                "task": task,
                "returncode": proc.returncode,
                "log": str(log_path),
                "command": cmd,
                "metrics": parse_log_metrics(proc.stdout),
            }
        )
        print(proc.stdout[-3000:], flush=True)
        if proc.returncode != 0:
            print(f"[comphrdoc][{task}] failed returncode={proc.returncode}; see {log_path}", file=sys.stderr)
    summary = {
        "gt_folder": str(gt_folder),
        "pred_folder": str(pred_folder),
        "comphrdoc_repo": str(args.comphrdoc_repo),
        "offset": args.offset,
        "limit": args.limit,
        "run_name": args.run_name,
        "runs": runs,
    }
    (args.output_dir / "comphrdoc_eval_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    failed = [run for run in runs if run["returncode"] != 0]
    return 1 if failed else 0


def make_eval_subsets(args: argparse.Namespace, gt_folder: Path, pred_folder: Path) -> tuple[Path, Path]:
    cfg = load_config(args.config)
    manifest = read_json(args.manifest or config_path(cfg, "outputs", "manifest"))
    docs = manifest.get("documents", manifest if isinstance(manifest, list) else [])
    if args.offset:
        docs = docs[args.offset :]
    if args.limit:
        docs = docs[: args.limit]
    subset_root = args.output_dir / "_manifest_slice"
    gt_subset = subset_root / "gold"
    pred_subset = subset_root / "pred"
    if gt_subset.exists():
        shutil.rmtree(gt_subset)
    if pred_subset.exists():
        shutil.rmtree(pred_subset)
    gt_subset.mkdir(parents=True, exist_ok=True)
    pred_subset.mkdir(parents=True, exist_ok=True)
    for doc in docs:
        name = f"{safe_doc_id(str(doc['document_id']))}.json"
        shutil.copy2(gt_folder / name, gt_subset / name)
        shutil.copy2(pred_folder / name, pred_subset / name)
    return gt_subset.resolve(), pred_subset.resolve()


def parse_log_metrics(text: str) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for line in text.splitlines():
        lowered = line.lower()
        if "macro" in lowered or "micro" in lowered or "teds" in lowered or "f1" in lowered:
            metrics.setdefault("interesting_lines", []).append(line.strip())
    return metrics


if __name__ == "__main__":
    raise SystemExit(main())
