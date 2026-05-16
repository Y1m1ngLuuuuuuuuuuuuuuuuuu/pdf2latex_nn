#!/usr/bin/env python3
"""Download the Nougat checkpoint into an explicit directory.

Nougat's built-in downloader writes to torch hub and keeps large files in
memory before flushing.  For AutoDL runs we keep the checkpoint under
``autodl-tmp`` and stream each file directly to disk.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import requests


BASE_URL = "https://github.com/facebookresearch/nougat/releases/download"
DEFAULT_MODEL_TAG = "0.1.0-small"
CHECKPOINT_FILES = [
    "config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-tag", default=DEFAULT_MODEL_TAG)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    parser.add_argument("--retries", type=int, default=6)
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename in CHECKPOINT_FILES:
        output = args.output_dir / filename
        if output.exists() and output.stat().st_size > 15:
            print(f"skip {filename} size={output.stat().st_size}", flush=True)
            continue
        url = f"{BASE_URL}/{args.model_tag}/{filename}"
        print(f"download {url} -> {output}", flush=True)
        download_file(url, output, timeout=args.timeout, chunk_size=args.chunk_size, retries=args.retries)
    return 0


def download_file(url: str, output: Path, *, timeout: int, chunk_size: int, retries: int) -> None:
    tmp = output.with_suffix(output.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout, allow_redirects=True) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0) or 0)
                seen = 0
                with tmp.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        seen += len(chunk)
                        if total:
                            print(f"  {output.name}: {seen / total:.1%}", flush=True)
            break
        except Exception as exc:  # noqa: BLE001 - network retries are intentional here.
            if tmp.exists():
                tmp.unlink()
            if attempt >= retries:
                raise
            delay = min(120, 5 * attempt)
            print(f"  retry {attempt}/{retries} after {type(exc).__name__}: {exc}; sleep={delay}s", flush=True)
            time.sleep(delay)
    if tmp.stat().st_size <= 15:
        raise RuntimeError(f"Downloaded file is suspiciously small: {tmp}")
    tmp.replace(output)


if __name__ == "__main__":
    raise SystemExit(main())
