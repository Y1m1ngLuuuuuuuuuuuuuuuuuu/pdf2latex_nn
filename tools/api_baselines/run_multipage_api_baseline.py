#!/usr/bin/env python3
"""Run a protected API/VLM baseline over full PDFs or multi-page windows."""

from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

from common import parse_doc_ids, read_json, require_api_enabled, safe_name, slice_items, utc_now, write_json
from providers import APIRequest, get_provider


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--windows", type=Path)
    parser.add_argument("--provider", choices=["openai", "gemini", "anthropic", "dashscope", "openai_compatible", "mock"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input-mode", choices=["full_pdf", "image_window", "pdf_window"], required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-output-tokens", type=int)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--doc-ids")
    parser.add_argument("--save-raw-response", action="store_true")
    parser.add_argument("--estimate-only", action="store_true")
    parser.add_argument("--run-name", default="api_run")
    return parser


def output_suffix(provider: str) -> str:
    return ".tex"


def build_prompt(template: str, item: dict[str, object]) -> str:
    return template + "\n\n% INPUT_WINDOW_METADATA\n" + "\n".join(
        [
            f"% doc_id: {item.get('doc_id')}",
            f"% window_id: {item.get('window_id', 'full')}",
            f"% pages: {item.get('pages', [])}",
        ]
    )


def main() -> int:
    args = build_arg_parser().parse_args()
    require_api_enabled(args.provider, args.dry_run or args.estimate_only)
    template = args.prompt_template.read_text(encoding="utf-8")
    if args.input_mode == "full_pdf":
        if not args.manifest:
            raise SystemExit("--manifest is required for full_pdf mode")
        from common import load_manifest_items

        items = load_manifest_items(args.manifest)
        tasks = [
            {
                "doc_id": item["doc_id"],
                "window_id": "full",
                "pages": [],
                "image_paths": [],
                "pdf_path": item.get("pdf_path"),
            }
            for item in items
        ]
    else:
        if not args.windows:
            raise SystemExit("--windows is required for image_window/pdf_window mode")
        tasks = read_json(args.windows).get("items") or []
    tasks = slice_items(tasks, offset=args.offset, limit=args.limit, doc_ids=parse_doc_ids(args.doc_ids), sort_by_doc_id=True)
    run_root = args.output_dir
    window_root = run_root / "window_outputs"
    full_root = run_root / "full_outputs"
    preview_root = run_root / "prompt_previews"
    for directory in (window_root, full_root, preview_root):
        directory.mkdir(parents=True, exist_ok=True)
    estimate = {
        "provider": args.provider,
        "model": args.model,
        "input_mode": args.input_mode,
        "tasks": len(tasks),
        "images": sum(len(task.get("image_paths") or []) for task in tasks),
        "pdfs": sum(1 for task in tasks if task.get("pdf_path")),
        "output_dir": str(run_root),
        "dry_run": args.dry_run,
        "real_api_allowed": args.provider == "mock" or args.dry_run or args.estimate_only,
        "concurrency": args.concurrency,
        "max_retries": args.max_retries,
        "created_at": utc_now(),
    }
    print(estimate)
    if args.estimate_only:
        write_json(run_root / "run_summary.json", estimate)
        return 0
    provider = get_provider(args.provider)
    successes = 0
    failures = 0
    for task in tasks:
        doc_id = str(task["doc_id"])
        window_id = str(task.get("window_id") or "full")
        if args.input_mode == "full_pdf":
            out_dir = full_root
            out_path = out_dir / f"{safe_name(doc_id)}{output_suffix(args.provider)}"
        else:
            out_dir = window_root / safe_name(doc_id)
            out_path = out_dir / f"{safe_name(window_id)}{output_suffix(args.provider)}"
        out_dir.mkdir(parents=True, exist_ok=True)
        if (args.resume or args.skip_existing) and out_path.exists():
            successes += 1
            continue
        prompt = build_prompt(template, task)
        preview_path = preview_root / safe_name(doc_id) / f"{safe_name(window_id)}.txt"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        preview_path.write_text(prompt, encoding="utf-8")
        request_meta = {
            "doc_id": doc_id,
            "window_id": window_id,
            "provider": args.provider,
            "model": args.model,
            "input_mode": args.input_mode,
            "image_paths": task.get("image_paths") or [],
            "pdf_path": task.get("pdf_path"),
            "temperature": args.temperature,
            "max_output_tokens": args.max_output_tokens,
            "dry_run": args.dry_run,
        }
        write_json(out_path.with_suffix(".request_meta.json"), request_meta)
        if args.dry_run:
            out_path.write_text("% DRY_RUN_PROMPT_PREVIEW_ONLY\n", encoding="utf-8")
            successes += 1
            continue
        last_error: dict[str, object] | None = None
        for attempt in range(1, max(1, args.max_retries) + 1):
            try:
                response = provider.generate(
                    APIRequest(
                        prompt=prompt,
                        image_paths=[Path(p) for p in task.get("image_paths") or []],
                        pdf_path=Path(str(task["pdf_path"])) if task.get("pdf_path") else None,
                        model=args.model,
                        temperature=args.temperature,
                        max_output_tokens=args.max_output_tokens,
                        metadata={"doc_id": doc_id, "window_id": window_id, "pages": task.get("pages") or []},
                    )
                )
                out_path.write_text(response.text, encoding="utf-8")
                write_json(out_path.with_suffix(".usage.json"), response.usage)
                if args.save_raw_response:
                    write_json(out_path.with_suffix(".raw_response.json"), response.raw)
                successes += 1
                last_error = None
                break
            except Exception as exc:  # keep batch moving
                last_error = {"attempt": attempt, "error": str(exc), "traceback": traceback.format_exc()}
                if attempt < args.max_retries:
                    time.sleep(min(2**attempt, 30))
        if last_error is not None:
            failures += 1
            write_json(out_path.with_suffix(".error.json"), last_error)
    estimate |= {"successes": successes, "failures": failures, "finished_at": utc_now()}
    write_json(run_root / "run_summary.json", estimate)
    print(f"finished successes={successes} failures={failures} summary={run_root / 'run_summary.json'}")
    return 0 if failures == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
