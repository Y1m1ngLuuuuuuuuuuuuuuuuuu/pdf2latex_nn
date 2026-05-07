#!/usr/bin/env python3
"""Summarize a running v7 batch-build log without touching the job.

The batch builder only writes its final manifest at completion, so this watcher
uses the append-only log plus optional error JSONL to report live progress,
quality-gate failures, and rough ETA.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any


START_RE = re.compile(r"\[mini-dataset\] start id=([^ ]+) success=(\d+)/(\d+)")
SUCCESS_RE = re.compile(
    r"\[mini-dataset\] success id=([^ ]+) success=(\d+)/(\d+) labels=(\{[^}]*\}) orphan_ratio=([0-9.]+)%"
)
SKIP_RE = re.compile(r"\[mini-dataset\] skip id=([^ ]+) error=([^:]+): ([^\r\n]*)")
CANDIDATE_RE = re.compile(r"candidate_count=(\d+) target=(\d+)")
PROGRESS_RE = re.compile(
    r"mini-dataset:\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s+\[([^<]+)<([^,]+),\s*([0-9.]+)s/doc\]"
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, help="Batch log path")
    parser.add_argument("--error-log", type=Path, help="Batch error JSONL path")
    parser.add_argument("--manifest", type=Path, help="Final manifest path, if already written")
    parser.add_argument("--current-dir", type=Path, default=Path("logs"), help="Directory with current_v7_build1000_* pointers")
    parser.add_argument("--interval", type=int, default=0, help="Repeat every N seconds; 0 prints once")
    parser.add_argument("--tail", type=int, default=8, help="Number of recent events to print")
    parser.add_argument("--json-output", type=Path, help="Optional path to write the latest summary JSON")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    while True:
        summary = build_summary(args)
        print_summary(summary, tail=args.tail)
        if args.json_output:
            args.json_output.parent.mkdir(parents=True, exist_ok=True)
            args.json_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    log_path = resolve_path(args.log, args.current_dir / "current_v7_build1000_log.txt")
    error_path = resolve_path(args.error_log, args.current_dir / "current_v7_build1000_errors.txt")
    manifest_path = resolve_path(args.manifest, args.current_dir / "current_v7_build1000_manifest.txt")
    text = log_path.read_text(encoding="utf-8", errors="ignore") if log_path and log_path.exists() else ""

    candidate_match = CANDIDATE_RE.search(text)
    candidate_count = int(candidate_match.group(1)) if candidate_match else None
    target = int(candidate_match.group(2)) if candidate_match else None
    starts = START_RE.findall(text)
    successes = SUCCESS_RE.findall(text)
    skips = SKIP_RE.findall(text)
    progress = PROGRESS_RE.findall(text)
    last_progress = progress[-1] if progress else None
    processed = len(successes) + len(skips)
    started = len(starts)
    success_count = int(successes[-1][1]) if successes else 0
    target = int(successes[-1][2]) if successes else target
    skip_counter = Counter(error_type for _, error_type, _ in skips)
    recent_events = recent_batch_events(text, limit=max(1, args.tail))
    elapsed_text = last_progress[3] if last_progress else None
    seconds_per_doc = float(last_progress[5]) if last_progress else None
    estimated = estimate_remaining(
        candidate_count=candidate_count,
        target=target,
        processed=processed,
        success_count=success_count,
        seconds_per_doc=seconds_per_doc,
    )
    manifest_stats = read_manifest_stats(manifest_path)
    error_stats = read_error_stats(error_path)
    return {
        "schema_version": "v7_batch_watch_v1",
        "log_path": str(log_path) if log_path else None,
        "error_log_path": str(error_path) if error_path else None,
        "manifest_path": str(manifest_path) if manifest_path else None,
        "candidate_count": candidate_count,
        "target": target,
        "started": started,
        "processed": processed,
        "success_count": success_count,
        "skip_count": len(skips),
        "pass_rate": safe_ratio(success_count, processed),
        "scan_progress": safe_ratio(processed, candidate_count),
        "elapsed": elapsed_text,
        "seconds_per_doc": seconds_per_doc,
        "last_start": starts[-1][0] if starts else None,
        "last_success": successes[-1][0] if successes else None,
        "skip_types": dict(skip_counter.most_common()),
        "estimated": estimated,
        "recent_events": recent_events,
        "manifest_stats": manifest_stats,
        "error_stats": error_stats,
    }


def resolve_path(explicit: Path | None, pointer_file: Path) -> Path | None:
    if explicit is not None:
        return explicit
    if pointer_file.exists():
        value = pointer_file.read_text(encoding="utf-8").strip()
        if value:
            return Path(value)
    return None


def estimate_remaining(
    *,
    candidate_count: int | None,
    target: int | None,
    processed: int,
    success_count: int,
    seconds_per_doc: float | None,
) -> dict[str, Any]:
    if candidate_count is None or target is None or seconds_per_doc is None:
        return {}
    remaining_candidates = max(0, candidate_count - processed)
    pass_rate = safe_ratio(success_count, max(1, processed))
    needed_successes = max(0, target - success_count)
    needed_candidates_at_rate = int((needed_successes / pass_rate) + 0.999) if pass_rate > 0 else None
    candidates_to_goal = min(remaining_candidates, needed_candidates_at_rate) if needed_candidates_at_rate is not None else None
    can_reach_target_at_current_rate = (
        success_count + int(remaining_candidates * pass_rate) >= target
        if pass_rate > 0
        else False
    )
    return {
        "remaining_candidates": remaining_candidates,
        "estimated_successes_if_rate_holds": success_count + int(remaining_candidates * pass_rate),
        "can_reach_target_at_current_rate": can_reach_target_at_current_rate,
        "eta_scan_all_seconds": int(remaining_candidates * seconds_per_doc),
        "eta_scan_all_human": human_seconds(remaining_candidates * seconds_per_doc),
        "eta_target_seconds": int(candidates_to_goal * seconds_per_doc) if candidates_to_goal is not None else None,
        "eta_target_human": human_seconds(candidates_to_goal * seconds_per_doc) if candidates_to_goal is not None else None,
    }


def recent_batch_events(text: str, *, limit: int) -> list[str]:
    events = []
    for raw_line in text.replace("\r", "\n").splitlines():
        if "[mini-dataset] start" in raw_line or "[mini-dataset] success" in raw_line or "[mini-dataset] skip" in raw_line:
            events.append(raw_line.strip())
    return events[-limit:]


def read_manifest_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"exists": True, "error": f"{type(exc).__name__}: {exc}"}
    documents = payload.get("documents", []) if isinstance(payload, dict) else []
    return {
        "exists": True,
        "success_count": len(documents),
        "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
    }


def read_error_stats(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {"exists": False}
    counts: Counter[str] = Counter()
    lines = 0
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if not line.strip():
            continue
        lines += 1
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            counts["JSONDecodeError"] += 1
            continue
        counts[str(payload.get("error_type", "unknown"))] += 1
    return {"exists": True, "lines": lines, "error_types": dict(counts.most_common())}


def print_summary(summary: dict[str, Any], *, tail: int) -> None:
    print("v7 batch status")
    print(f"log={summary['log_path']}")
    print(
        f"processed={summary['processed']}/{summary['candidate_count']} "
        f"success={summary['success_count']}/{summary['target']} "
        f"skips={summary['skip_count']} pass_rate={summary['pass_rate']:.2%}"
    )
    print(
        f"elapsed={summary.get('elapsed')} sec_per_doc={summary.get('seconds_per_doc')} "
        f"last_start={summary.get('last_start')} last_success={summary.get('last_success')}"
    )
    estimated = summary.get("estimated") or {}
    if estimated:
        print(
            "eta "
            f"target={estimated.get('eta_target_human')} "
            f"scan_all={estimated.get('eta_scan_all_human')} "
            f"projected_successes={estimated.get('estimated_successes_if_rate_holds')} "
            f"can_reach_target={estimated.get('can_reach_target_at_current_rate')}"
        )
    print(f"skip_types={summary['skip_types']}")
    if tail > 0:
        print("recent_events:")
        for event in summary["recent_events"][-tail:]:
            print(f"  {event[:240]}")


def safe_ratio(numerator: int, denominator: int | None) -> float:
    if not denominator:
        return 0.0
    return numerator / denominator


def human_seconds(value: float | int | None) -> str | None:
    if value is None:
        return None
    seconds = max(0, int(value))
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


if __name__ == "__main__":
    raise SystemExit(main())
