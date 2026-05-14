"""PDF page-layout similarity metrics.

This module compares rendered PDFs geometrically.  It does not try to judge OCR
or exact formula recognition.  The metric focuses on page count, ink bounding
boxes, and horizontal/vertical ink-density profiles.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


def compare_pdf_layouts(
    gold_pdf: Path,
    pred_pdf: Path,
    *,
    dpi: int = 72,
    max_pages: int | None = None,
    ink_threshold: int = 245,
) -> dict[str, Any]:
    gold_pdf = gold_pdf.resolve()
    pred_pdf = pred_pdf.resolve()
    gold_pages = pdf_page_count(gold_pdf)
    pred_pages = pdf_page_count(pred_pdf)
    comparable_pages = min(gold_pages, pred_pages)
    if max_pages is not None:
        comparable_pages = min(comparable_pages, max_pages)

    page_scores: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="pdf2latex_layout_") as tmp:
        tmpdir = Path(tmp)
        for page in range(1, comparable_pages + 1):
            gold_img = render_pdf_page_to_gray(gold_pdf, page, tmpdir / f"gold_{page}", dpi=dpi)
            pred_img = render_pdf_page_to_gray(pred_pdf, page, tmpdir / f"pred_{page}", dpi=dpi)
            page_scores.append(compare_gray_pages(gold_img, pred_img, ink_threshold=ink_threshold) | {"page": page})

    layout_score = average([score["layout_similarity"] for score in page_scores])
    return {
        "gold_pdf": str(gold_pdf),
        "pred_pdf": str(pred_pdf),
        "gold_pages": gold_pages,
        "pred_pages": pred_pages,
        "page_count_match": gold_pages == pred_pages,
        "page_count_score": min(gold_pages, pred_pages) / max(gold_pages, pred_pages, 1),
        "compared_pages": comparable_pages,
        "layout_similarity": layout_score,
        "page_scores": page_scores,
    }


def pdf_page_count(pdf_path: Path) -> int:
    if shutil.which("pdfinfo"):
        result = subprocess.run(["pdfinfo", str(pdf_path)], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        match = re.search(r"^Pages:\s+(\d+)\s*$", result.stdout, re.MULTILINE)
        if match:
            return int(match.group(1))
    try:
        import fitz  # type: ignore

        with fitz.open(str(pdf_path)) as doc:
            return int(doc.page_count)
    except Exception as exc:  # pragma: no cover - depends on optional system packages.
        raise RuntimeError(f"Cannot determine page count for {pdf_path}: {exc}") from exc


def render_pdf_page_to_gray(pdf_path: Path, page: int, prefix: Path, *, dpi: int) -> np.ndarray:
    if shutil.which("pdftoppm"):
        subprocess.run(
            [
                "pdftoppm",
                "-gray",
                "-r",
                str(dpi),
                "-f",
                str(page),
                "-l",
                str(page),
                "-singlefile",
                str(pdf_path),
                str(prefix),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return read_pgm(prefix.with_suffix(".pgm"))
    try:
        import fitz  # type: ignore

        zoom = dpi / 72.0
        with fitz.open(str(pdf_path)) as doc:
            pix = doc.load_page(page - 1).get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
            return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
    except Exception as exc:  # pragma: no cover - depends on optional system packages.
        raise RuntimeError(f"Cannot render {pdf_path} page {page}: {exc}") from exc


def compare_gray_pages(gold_img: np.ndarray, pred_img: np.ndarray, *, ink_threshold: int) -> dict[str, float]:
    gold = ensure_uint8_gray(gold_img)
    pred = resize_nearest(ensure_uint8_gray(pred_img), gold.shape)
    gold_ink = gold < ink_threshold
    pred_ink = pred < ink_threshold
    bbox_score = bbox_iou(ink_bbox(gold_ink), ink_bbox(pred_ink))
    row_score = cosine_similarity(resample_profile(gold_ink.mean(axis=1), 160), resample_profile(pred_ink.mean(axis=1), 160))
    col_score = cosine_similarity(resample_profile(gold_ink.mean(axis=0), 120), resample_profile(pred_ink.mean(axis=0), 120))
    pixel_score = 1.0 - float(np.mean(np.abs(gold.astype(np.float32) - pred.astype(np.float32))) / 255.0)
    pixel_score = clamp(pixel_score)
    layout_similarity = clamp(0.30 * row_score + 0.30 * col_score + 0.25 * bbox_score + 0.15 * pixel_score)
    return {
        "layout_similarity": layout_similarity,
        "row_profile_similarity": clamp(row_score),
        "column_profile_similarity": clamp(col_score),
        "ink_bbox_iou": clamp(bbox_score),
        "pixel_l1_similarity": pixel_score,
    }


def read_pgm(path: Path) -> np.ndarray:
    data = path.read_bytes()
    pos = 0

    def read_token() -> bytes:
        nonlocal pos
        while pos < len(data):
            char = data[pos:pos + 1]
            if char == b"#":
                while pos < len(data) and data[pos:pos + 1] not in {b"\n", b"\r"}:
                    pos += 1
            elif char.isspace():
                pos += 1
            else:
                break
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        return data[start:pos]

    magic = read_token()
    width = int(read_token())
    height = int(read_token())
    maxval = int(read_token())
    while pos < len(data) and data[pos:pos + 1].isspace():
        pos += 1
    if magic == b"P5":
        dtype = np.uint8 if maxval <= 255 else ">u2"
        arr = np.frombuffer(data[pos:], dtype=dtype, count=width * height).reshape(height, width)
        if maxval != 255:
            arr = (arr.astype(np.float32) / maxval * 255).astype(np.uint8)
        return arr.astype(np.uint8, copy=False)
    if magic == b"P2":
        values = np.array([int(token) for token in data[pos:].split()], dtype=np.float32)
        arr = values.reshape(height, width)
        return (arr / maxval * 255).astype(np.uint8)
    raise ValueError(f"Unsupported PGM magic {magic!r} in {path}")


def ensure_uint8_gray(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError(f"Expected grayscale image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = np.clip(arr, 0, 255)
    return arr.astype(np.uint8)


def resize_nearest(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    if image.shape == shape:
        return image
    y_idx = np.linspace(0, image.shape[0] - 1, shape[0]).round().astype(int)
    x_idx = np.linspace(0, image.shape[1] - 1, shape[1]).round().astype(int)
    return image[np.ix_(y_idx, x_idx)]


def ink_bbox(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.where(mask)
    if len(xs) == 0 or len(ys) == 0:
        return None
    height, width = mask.shape
    return (float(xs.min()) / width, float(ys.min()) / height, float(xs.max() + 1) / width, float(ys.max() + 1) / height)


def bbox_iou(a: tuple[float, float, float, float] | None, b: tuple[float, float, float, float] | None) -> float:
    if a is None and b is None:
        return 1.0
    if a is None or b is None:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-8)


def resample_profile(values: np.ndarray, bins: int) -> np.ndarray:
    if values.size == 0:
        return np.zeros((bins,), dtype=np.float32)
    x_old = np.linspace(0.0, 1.0, values.size)
    x_new = np.linspace(0.0, 1.0, bins)
    return np.interp(x_new, x_old, values.astype(np.float32))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 1e-12:
        return 1.0 if float(np.linalg.norm(a) + np.linalg.norm(b)) <= 1e-12 else 0.0
    return float(np.dot(a, b) / denom)


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / len(values))


def clamp(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def write_layout_report(report: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
