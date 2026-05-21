#!/usr/bin/env python3
"""Run our PDF2LaTeX-NN front-end/GNN/decoder on CompHRDoc PDFs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from tools.comphrdoc.common import config_path, load_config, read_json, safe_doc_id, write_json  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/external_eval/comphrdoc_test500.yaml"))
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--offset", type=int, default=0, help="Skip the first N PDF-ready manifest documents.")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tau-merge", type=float)
    parser.add_argument("--tau-parent", type=float)
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--force-front-end", action="store_true")
    parser.add_argument("--extra-arg", action="append", default=[])
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = load_config(args.config)
    manifest_path = args.manifest or config_path(cfg, "outputs", "manifest")
    manifest = read_json(manifest_path)
    docs = manifest.get("documents", manifest if isinstance(manifest, list) else [])
    docs = [doc for doc in docs if doc.get("pdf_path")]
    if args.offset:
        docs = docs[args.offset :]
    if args.limit:
        docs = docs[: args.limit]
    if not docs:
        raise ValueError("No PDF-ready documents in CompHRDoc manifest. Run images_to_pdf.py first.")
    run_name = args.run_name or f"m06_smoke{len(docs)}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
    run_dir = config_path(cfg, "outputs", "ours_ir_root") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (run_dir / "e2e" / "e2e_manifest.json").exists():
        print(f"[comphrdoc] skip existing run -> {run_dir}")
        return 0
    temp_manifest = run_dir / "input_manifest.json"
    write_json(
        temp_manifest,
        {
            "schema_version": "comphrdoc_inference_input_v1",
            "documents": [
                {
                    "document_id": doc["document_id"],
                    "pdf_path": doc["pdf_path"],
                    "gold_json": doc.get("gold_json"),
                    "hrdh_test_json": doc.get("hrdh_test_json"),
                    "num_pages": doc.get("num_pages"),
                }
                for doc in docs
            ],
        },
    )
    checkpoint = args.checkpoint or Path(str(cfg["model"]["checkpoint"]))
    tau_merge = args.tau_merge if args.tau_merge is not None else float(cfg["model"].get("tau_merge", 0.37))
    tau_parent = args.tau_parent if args.tau_parent is not None else float(cfg["model"].get("tau_parent", 0.45))
    command = [
        args.python_bin,
        "scripts/pipeline/run_e2e_inference.py",
        "--manifest",
        str(temp_manifest),
        "--checkpoint",
        str(checkpoint),
        "--output-dir",
        str(run_dir / "e2e"),
        "--limit",
        str(len(docs)),
        "--split",
        "all",
        "--merge-threshold",
        str(tau_merge),
        "--parent-threshold",
        str(tau_parent),
        "--heading-skeleton-mode",
        "stack",
        "--skip-compile",
        "--mineru-output-dir",
        str(run_dir / "mineru_output"),
        "--graph-output-dir",
        str(run_dir / "graphs"),
        "--frontend-work-dir",
        str(run_dir / "frontend_work"),
    ]
    if args.force_front_end:
        command.append("--force-front-end")
    command.extend(args.extra_arg)
    print("[comphrdoc] $ " + " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    write_json(
        run_dir / "run_metadata.json",
        {
            "schema_version": "comphrdoc_ours_inference_run_v1",
            "run_name": run_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "manifest": str(manifest_path),
            "input_manifest": str(temp_manifest),
            "checkpoint": str(checkpoint),
            "tau_merge": tau_merge,
            "tau_parent": tau_parent,
            "e2e_manifest": str(run_dir / "e2e" / "e2e_manifest.json"),
        },
    )
    print(f"[comphrdoc] run complete -> {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
