"""Table fragment grouping and PDF crop helpers.

MinerU may split one visually wide table into several adjacent ``table`` blocks.
For original-like reconstruction we treat those fragments as one table group and
prefer a single union-bbox crop from the source PDF over brittle HTML-to-tabular
conversion.
"""

from __future__ import annotations

import re
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
    if x_gap > max(35.0, 0.08 * page_width):
        return False
    # Two independently captioned tables placed side by side should stay split.
    if table_caption_text(left) and table_caption_text(right):
        return False
    return True


def ensure_table_pdf_crop(
    record: dict[str, Any],
    *,
    source_pdf: str | Path | None,
    asset_output_dir: str | Path | None,
    asset_latex_prefix: str = "assets",
    padding: float = 3.0,
) -> str | None:
    """Crop the table union bbox from the source PDF and return a LaTeX path."""

    if not source_pdf or not asset_output_dir:
        return None
    pdf_path = Path(source_pdf)
    output_dir = Path(asset_output_dir)
    bbox = record_bbox({"bbox": record.get("table_group_bbox") or record.get("bbox")})
    if bbox is None or not pdf_path.exists():
        return None
    page_idx = int_value(record.get("page_idx"), 0)
    table_id = safe_asset_stem(str(record.get("table_group_id") or record_identifier(record, fallback=page_idx)))
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{table_id}.png"
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


def latex_asset_path(output_path: Path, *, asset_output_dir: Path, asset_latex_prefix: str) -> str:
    name = output_path.name
    prefix = str(asset_latex_prefix or "").strip().strip("/")
    return f"{prefix}/{name}" if prefix else name


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


def record_identifier(record: dict[str, Any], *, fallback: int) -> str:
    for key in ("node_id", "id", "block_id", "table_id", "global_order", "original_index", "mineru_block_idx"):
        value = record.get(key)
        if value is not None and value != "":
            return str(value)
    return f"table_{fallback}"


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
