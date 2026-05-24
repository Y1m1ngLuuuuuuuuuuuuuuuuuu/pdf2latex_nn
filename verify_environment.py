#!/usr/bin/env python3
"""Project environment smoke checker.

The current default reconstruction path is v8/layout-first.  MinerU/PaddleOCR
and API-provider SDKs are optional extras, so this checker distinguishes the
base environment from heavier server/API profiles.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from importlib import import_module
from typing import Iterable


CORE_MODULES = [
    ("numpy", "NumPy"),
    ("pandas", "Pandas"),
    ("PIL", "Pillow"),
    ("fitz", "PyMuPDF"),
    ("PyPDF2", "PyPDF2"),
    ("pdf2image", "pdf2image"),
    ("cv2", "OpenCV"),
    ("scipy", "SciPy"),
    ("networkx", "NetworkX"),
    ("yaml", "PyYAML"),
    ("requests", "Requests"),
    ("tqdm", "tqdm"),
    ("rapidfuzz", "RapidFuzz"),
    ("TexSoup", "TexSoup"),
    ("pylatexenc", "pylatexenc"),
]

GNN_MODULES = [
    ("torch", "PyTorch"),
    ("torch_geometric", "PyTorch Geometric"),
    ("transformers", "Transformers"),
    ("lightning", "Lightning"),
]

SERVER_OPTIONAL_MODULES = [
    ("boto3", "Boto3"),
    ("ultralytics", "Ultralytics"),
    ("paddleocr", "PaddleOCR"),
    ("paddle", "PaddlePaddle"),
]

API_OPTIONAL_MODULES = [
    ("openai", "OpenAI SDK"),
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=["base", "server", "api", "all"],
        default="base",
        help=(
            "base checks v8 + GNN dependencies. server also checks MinerU/OCR-adjacent "
            "extras. api checks API-baseline SDKs. all checks every optional group."
        ),
    )
    return parser


def module_version(module_name: str) -> str:
    module = import_module(module_name)
    return str(getattr(module, "__version__", "installed"))


def check_modules(modules: Iterable[tuple[str, str]], *, required: bool, title: str) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []
    print(f"\n[{title}]")
    for module_name, label in modules:
        try:
            version = module_version(module_name)
            print(f"  {label}: {version}")
        except Exception as exc:
            message = f"{label} missing: {exc}"
            if required:
                errors.append(message)
                print(f"  {label}: MISSING")
            else:
                warnings.append(message)
                print(f"  {label}: optional missing")
    return errors, warnings


def check_torch_details() -> list[str]:
    warnings: list[str] = []
    try:
        import torch
    except Exception:
        return warnings
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU count: {torch.cuda.device_count()}")
        for idx in range(torch.cuda.device_count()):
            print(f"  GPU {idx}: {torch.cuda.get_device_name(idx)}")
    else:
        warnings.append("CUDA is unavailable; this is normal for laptop CPU/MPS work, slower for GNN training.")
        if hasattr(torch.backends, "mps"):
            print(f"  MPS available: {torch.backends.mps.is_available()}")
    return warnings


def check_system_tools() -> list[str]:
    warnings: list[str] = []
    print("\n[System tools]")
    for tool in ["pdfinfo", "pdftoppm", "pdflatex", "xelatex", "latexmk"]:
        path = shutil.which(tool)
        if path:
            print(f"  {tool}: {path}")
        else:
            warnings.append(f"{tool} not found")
            print(f"  {tool}: optional missing")
    return warnings


def check_python_version() -> list[str]:
    warnings: list[str] = []
    print(f"[Python] {sys.version}")
    if sys.version_info < (3, 10):
        raise RuntimeError("Python >= 3.10 is required")
    if sys.version_info >= (3, 13):
        warnings.append(
            "Python >= 3.13 may not have stable wheels for torch/paddle on every platform; "
            "Python 3.11 is recommended."
        )
    return warnings


def main() -> int:
    args = build_arg_parser().parse_args()
    print("=" * 64)
    print("PDF2LaTeX-NN environment check")
    print("=" * 64)

    errors: list[str] = []
    warnings: list[str] = []

    try:
        warnings.extend(check_python_version())
    except RuntimeError as exc:
        errors.append(str(exc))

    group_errors, group_warnings = check_modules(CORE_MODULES, required=True, title="Core v8 / rendering modules")
    errors.extend(group_errors)
    warnings.extend(group_warnings)

    group_errors, group_warnings = check_modules(GNN_MODULES, required=True, title="GNN / embedding branch modules")
    errors.extend(group_errors)
    warnings.extend(group_warnings)
    warnings.extend(check_torch_details())

    if args.profile in {"server", "all"}:
        group_errors, group_warnings = check_modules(
            SERVER_OPTIONAL_MODULES,
            required=False,
            title="Server / MinerU-adjacent optional modules",
        )
        warnings.extend(group_warnings)

    if args.profile in {"api", "all"}:
        _, group_warnings = check_modules(API_OPTIONAL_MODULES, required=False, title="API baseline optional modules")
        warnings.extend(group_warnings)

    warnings.extend(check_system_tools())

    print("\n" + "=" * 64)
    print("Result")
    print("=" * 64)
    if errors:
        print("\nErrors:")
        for item in errors:
            print(f"  - {item}")
    if warnings:
        print("\nWarnings:")
        for item in warnings:
            print(f"  - {item}")
    if errors:
        print("\nEnvironment has missing required dependencies.")
        return 1
    print("\nRequired dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
