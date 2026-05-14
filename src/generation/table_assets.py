"""Table/figure fragment grouping and PDF crop helpers.

MinerU may split one visually wide table into several adjacent ``table`` blocks.
For original-like reconstruction we treat those fragments as one table group and
prefer a single union-bbox crop from the source PDF over brittle HTML-to-tabular
conversion.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Any


def annotate_table_group_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Annotate adjacent same-page table fragments with stable group metadata."""

    annotated = [dict(record) for record in records]
    table_indices = [index for index, record in enumerate(annotated) if canonical_record_type(record) == "table"]
    groups: list[list[int]] = []
    used: set[int] = set()
    for index in sorted(table_indices, key=lambda item: table_group_sort_key(annotated[item], item)):
        if index in used:
            continue
        group = [index]
        used.add(index)
        changed = True
        while changed:
            changed = False
            for candidate in table_indices:
                if candidate in used:
                    continue
                if any(should_group_table_fragments(annotated[candidate], annotated[member]) for member in group):
                    group.append(candidate)
                    used.add(candidate)
                    changed = True
        groups.append(sorted(group, key=lambda item: table_fragment_sort_key(annotated[item], item)))

    for group_number, group in enumerate(groups):
        boxes = [record_bbox(annotated[index]) for index in group]
        boxes = [box for box in boxes if box is not None]
        if not boxes:
            continue
        union = union_bbox(boxes)
        primary = choose_table_group_primary(annotated, group)
        page_idx = int_value(annotated[primary].get("page_idx"), 0)
        group_id = f"table_group_p{page_idx:04d}_{group_number:04d}"
        member_ids = [record_identifier(annotated[index], fallback=index) for index in group]
        caption = first_table_caption((annotated[index] for index in group))
        for member_index, index in enumerate(group):
            metadata = {
                "table_group_id": group_id,
                "table_group_member_ids": member_ids,
                "table_group_member_index": member_index,
                "table_group_size": len(group),
                "table_group_primary": index == primary,
                "table_group_bbox": list(union),
                "table_group_caption": caption,
                "table_group_render_strategy": "union_pdf_crop" if len(group) > 1 else "single_pdf_crop",
            }
            annotated[index].update(metadata)
    return annotated


def should_group_table_fragments(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return True when two table detections are likely fragments of one table."""

    if canonical_record_type(left) != "table" or canonical_record_type(right) != "table":
        return False
    if int_value(left.get("page_idx"), -1) != int_value(right.get("page_idx"), -2):
        return False
    left_box = record_bbox(left)
    right_box = record_bbox(right)
    if left_box is None or right_box is None:
        return False
    if y_overlap_ratio(left_box, right_box) < 0.60:
        return False
    page_width = page_width_from_records(left, right)
    x_gap = bbox_x_gap(left_box, right_box)
    left_caption = table_caption_text(left)
    right_caption = table_caption_text(right)
    # Two independently captioned tables placed side by side should stay split.
    if left_caption and right_caption:
        return normalized_caption_text(left_caption) == normalized_caption_text(right_caption)
    if x_gap > max(35.0, 0.08 * page_width):
        return False
    # A common failure mode is two unrelated column-local tables with similar
    # vertical extent.  They overlap in Y, but there is no shared caption/title
    # evidence.  Keep those split unless the fragments physically touch; a true
    # cross-column table should either have a common caption or be one wide box.
    if not left_caption and not right_caption and are_in_opposite_columns(left_box, right_box, page_width):
        if x_gap > max(12.0, 0.018 * page_width):
            return False
    return True


def are_in_opposite_columns(
    left_box: tuple[float, float, float, float],
    right_box: tuple[float, float, float, float],
    page_width: float,
) -> bool:
    center = page_width / 2.0
    left_center = (left_box[0] + left_box[2]) / 2.0
    right_center = (right_box[0] + right_box[2]) / 2.0
    return (left_center < center < right_center) or (right_center < center < left_center)


def normalized_caption_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").casefold())


def is_wide_visual_record(record: dict[str, Any], *, bbox_keys: tuple[str, ...] = ("bbox",), threshold: float = 0.62) -> bool:
    bbox = first_record_bbox(record, bbox_keys)
    if bbox is None:
        return False
    page_width = float(record.get("page_width") or 1000.0)
    width_ratio = max(bbox[2] - bbox[0], 0.0) / max(page_width, 1.0)
    if width_ratio >= threshold:
        return True
    center = page_width / 2.0
    gutter_margin = 0.045 * page_width
    return bbox[0] < center - gutter_margin and bbox[2] > center + gutter_margin


def ensure_table_pdf_crop(
    record: dict[str, Any],
    *,
    source_pdf: str | Path | None,
    asset_output_dir: str | Path | None,
    asset_latex_prefix: str = "assets",
    padding: float = 3.0,
) -> str | None:
    """Crop the table union bbox from the source PDF and return a LaTeX path."""

    return ensure_pdf_region_crop(
        record,
        source_pdf=source_pdf,
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
        padding=padding,
        kind=None,
        bbox_keys=("table_group_bbox", "bbox"),
        id_keys=("table_group_id", "node_id", "id", "block_id", "table_id", "global_order", "original_index", "mineru_block_idx"),
    )


def ensure_figure_pdf_crop(
    record: dict[str, Any],
    *,
    source_pdf: str | Path | None,
    asset_output_dir: str | Path | None,
    asset_latex_prefix: str = "assets",
    padding: float = 3.0,
) -> str | None:
    """Crop a figure/image bbox from the source PDF and return a LaTeX path."""

    return ensure_pdf_region_crop(
        record,
        source_pdf=source_pdf,
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
        padding=padding,
        kind="figure",
        bbox_keys=("figure_group_bbox", "image_group_bbox", "bbox"),
        id_keys=("figure_group_id", "image_group_id", "node_id", "id", "block_id", "figure_id", "image_id", "global_order", "original_index", "mineru_block_idx"),
    )


def ensure_figure_asset(
    record: dict[str, Any],
    *,
    source_pdf: str | Path | None = None,
    asset_output_dir: str | Path | None = None,
    asset_latex_prefix: str = "assets",
    padding: float = 3.0,
) -> str | None:
    """Return a usable LaTeX image path for a figure.

    Prefer MinerU-provided image assets when present.  If no image asset can be
    resolved, fall back to cropping the figure bbox from the source PDF.  Only
    return ``None`` when neither source is available.
    """

    existing = ensure_existing_figure_asset(
        record,
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
    )
    if existing:
        return existing
    return ensure_figure_pdf_crop(
        record,
        source_pdf=source_pdf,
        asset_output_dir=asset_output_dir,
        asset_latex_prefix=asset_latex_prefix,
        padding=padding,
    )


def ensure_existing_figure_asset(
    record: dict[str, Any],
    *,
    asset_output_dir: str | Path | None = None,
    asset_latex_prefix: str = "assets",
) -> str | None:
    """Resolve and optionally copy an already-extracted MinerU figure asset."""

    asset_path = first_existing_asset_path(record, FIGURE_ASSET_KEYS)
    if asset_path is None:
        return None
    if asset_output_dir:
        output_dir = Path(asset_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        ext = asset_path.suffix or ".png"
        figure_id = safe_asset_stem(str(record_identifier(record, fallback=0, keys=FIGURE_ID_KEYS)))
        output_path = output_dir / f"figure_{figure_id}{ext}"
        if not output_path.exists():
            try:
                shutil.copy2(asset_path, output_path)
            except OSError:
                return None
        return latex_asset_path(output_path, asset_output_dir=output_dir, asset_latex_prefix=asset_latex_prefix)
    return latex_safe_path(asset_path)


def ensure_pdf_region_crop(
    record: dict[str, Any],
    *,
    source_pdf: str | Path | None,
    asset_output_dir: str | Path | None,
    asset_latex_prefix: str = "assets",
    padding: float = 3.0,
    kind: str | None = "region",
    bbox_keys: tuple[str, ...] = ("bbox",),
    id_keys: tuple[str, ...] = ("node_id", "id", "block_id", "global_order", "original_index", "mineru_block_idx"),
) -> str | None:
    """Crop an arbitrary PDF bbox and return a LaTeX-safe relative asset path.

    The input bbox is expected to live in the same normalized coordinate system
    used by v7 content JSON.  Page dimensions are read from the record when
    present and fall back to the normalized 1000 x 1000 contract.
    """

    if not source_pdf or not asset_output_dir:
        return None
    pdf_path = Path(source_pdf)
    output_dir = Path(asset_output_dir)
    bbox = first_record_bbox(record, bbox_keys)
    if bbox is None or not pdf_path.exists():
        return None
    page_idx = int_value(record.get("page_idx"), 0)
    region_id = safe_asset_stem(str(record_identifier(record, fallback=page_idx, keys=id_keys)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_stem = f"{safe_asset_stem(kind)}_{region_id}" if kind else region_id
    output_path = output_dir / f"{output_stem}.png"
    if not output_path.exists():
        try:
            import fitz  # type: ignore

            with fitz.open(pdf_path) as doc:
                if page_idx < 0 or page_idx >= len(doc):
                    return None
                page = doc.load_page(page_idx)
                x0, y0, x1, y1 = bbox
                rect = page.rect
                # v7/MinerU bboxes are normalized to a 1000-wide page by default.
                scale_x = rect.width / max(float(record.get("page_width") or 1000.0), 1.0)
                scale_y = rect.height / max(float(record.get("page_height") or 1000.0), 1.0)
                clip = fitz.Rect(
                    max(0.0, (x0 - padding) * scale_x),
                    max(0.0, (y0 - padding) * scale_y),
                    min(rect.width, (x1 + padding) * scale_x),
                    min(rect.height, (y1 + padding) * scale_y),
                )
                pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), clip=clip, alpha=False)
                pix.save(output_path)
        except Exception:
            return None
    return latex_asset_path(output_path, asset_output_dir=output_dir, asset_latex_prefix=asset_latex_prefix)


FIGURE_ASSET_KEYS = (
    "figure_asset_path",
    "image_asset_path",
    "img_path",
    "image_path",
    "figure_path",
    "asset_path",
)
FIGURE_ID_KEYS = (
    "figure_group_id",
    "image_group_id",
    "node_id",
    "id",
    "block_id",
    "figure_id",
    "image_id",
    "global_order",
    "original_index",
    "mineru_block_idx",
)


def first_existing_asset_path(record: dict[str, Any], keys: tuple[str, ...]) -> Path | None:
    candidates: list[Path] = []
    base_dirs = record_base_dirs(record)
    for key in keys:
        value = record.get(key)
        if not isinstance(value, str) or not value.strip():
            continue
        raw_path = Path(value.strip()).expanduser()
        candidates.append(raw_path)
        if not raw_path.is_absolute():
            candidates.extend(base_dir / raw_path for base_dir in base_dirs)
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        except OSError:
            continue
    return None


def record_base_dirs(record: dict[str, Any]) -> list[Path]:
    bases: list[Path] = []
    for key in ("source_json_dir", "json_dir", "asset_base_dir", "mineru_output_dir"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            bases.append(Path(value.strip()).expanduser())
    for key in ("source_json", "content_json", "json_path", "source_path"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            bases.append(Path(value.strip()).expanduser().parent)
    source_refs = record.get("source_refs")
    if isinstance(source_refs, list):
        for ref in source_refs:
            if not isinstance(ref, dict):
                continue
            value = ref.get("path")
            if isinstance(value, str) and value.strip():
                bases.append(Path(value.strip()).expanduser().parent)
    result: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        key = str(base)
        if key not in seen:
            seen.add(key)
            result.append(base)
    return result


def latex_asset_path(output_path: Path, *, asset_output_dir: Path, asset_latex_prefix: str) -> str:
    name = output_path.name
    prefix = str(asset_latex_prefix or "").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


def latex_safe_path(path: Path) -> str:
    return path.as_posix()


def canonical_record_type(record: dict[str, Any]) -> str:
    raw = str(record.get("canonical_type") or record.get("type") or record.get("raw_type") or "").casefold()
    return "table" if raw == "table" else raw


def table_group_sort_key(record: dict[str, Any], fallback: int) -> tuple[int, float, float, int]:
    box = record_bbox(record) or (0.0, 0.0, 0.0, 0.0)
    return (int_value(record.get("page_idx"), 0), box[1], box[0], fallback)


def table_fragment_sort_key(record: dict[str, Any], fallback: int) -> tuple[int, float, float, int]:
    box = record_bbox(record) or (0.0, 0.0, 0.0, 0.0)
    return (int_value(record.get("page_idx"), 0), box[0], box[1], fallback)


def record_bbox(record: dict[str, Any]) -> tuple[float, float, float, float] | None:
    value = record.get("bbox")
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        return (float(value[0]), float(value[1]), float(value[2]), float(value[3]))
    except (TypeError, ValueError):
        return None


def first_record_bbox(record: dict[str, Any], keys: tuple[str, ...]) -> tuple[float, float, float, float] | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, list):
            bbox = record_bbox({"bbox": value})
            if bbox is not None:
                return bbox
    return None


def union_bbox(boxes: list[tuple[float, float, float, float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def y_overlap_ratio(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    intersection = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    min_height = max(min(a[3] - a[1], b[3] - b[1]), 1e-6)
    return intersection / min_height


def bbox_x_gap(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    return max(max(a[0], b[0]) - min(a[2], b[2]), 0.0)


def page_width_from_records(left: dict[str, Any], right: dict[str, Any]) -> float:
    for record in (left, right):
        value = record.get("page_width")
        if isinstance(value, (int, float)) and value > 0:
            return float(value)
    left_box = record_bbox(left)
    right_box = record_bbox(right)
    if left_box and right_box:
        return max(left_box[2], right_box[2], 1000.0) - min(left_box[0], right_box[0], 0.0)
    return 1000.0


def choose_table_group_primary(records: list[dict[str, Any]], group: list[int]) -> int:
    captioned = [index for index in group if table_caption_text(records[index])]
    if captioned:
        return captioned[-1]
    return min(group, key=lambda index: table_group_sort_key(records[index], index))


def first_table_caption(records: Any) -> str:
    for record in records:
        caption = table_caption_text(record)
        if caption:
            return caption
    return ""


def table_caption_text(record: dict[str, Any]) -> str:
    value = record.get("table_group_caption") or record.get("table_caption")
    if isinstance(value, list):
        return " ".join(str(part).strip() for part in value if str(part).strip()).strip()
    if isinstance(value, str):
        return " ".join(value.split())
    return ""


def record_identifier(
    record: dict[str, Any],
    *,
    fallback: int,
    keys: tuple[str, ...] = ("node_id", "id", "block_id", "table_id", "global_order", "original_index", "mineru_block_idx"),
) -> str:
    for key in keys:
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    return f"region_{fallback}"


def safe_asset_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "table"


def int_value(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except ValueError:
            return default
    return default
