#!/usr/bin/env python3
"""Convert our v7/DocumentIR records into CompHRDoc ``hr_json`` format.

This bridge is intentionally conservative.  CompHRDoc evaluates page-object
classes, reading order and hierarchical document structure, while our project
targets layout-aware LaTeX reconstruction.  The converter therefore exposes the
shared block-level structure and marks unsupported fine-grained metadata as
best-effort rather than pretending we can recover every CompHRDoc field.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.adapters.mineru_v7_document_ir import load_v7_document_ir  # noqa: E402
from src.ir import BBox, BlockType, DocumentIR, DocumentNode  # noqa: E402
from src.ir.serialization import read_dataclass_json  # noqa: E402


COMPHRDOC_SCHEMA_VERSION = "comphrdoc_bridge_v0"
CAPTION_RE = re.compile(r"^\s*(?:fig(?:ure)?|tab(?:le)?|algorithm)\s*[\.:]?\s*[\w.\-]+", re.IGNORECASE)
EMAIL_RE = re.compile(r"[\w.+\-]+@[\w.\-]+\.[A-Za-z]{2,}")
NUMBERED_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*|[IVXLCDM]{1,8}|[A-Z]|Appendix\s+[A-Z])[\.)]?\s+",
    re.IGNORECASE,
)
TERMINAL_RE = re.compile(r"[.!?。！？]\s*$")


@dataclass(frozen=True)
class ExportRecord:
    text: str
    box: list[int]
    class_: str
    page: int
    is_meta: bool
    line_id: int
    parent_id: int
    relation: str
    page_object_id: int

    def to_json(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "box": self.box,
            "class": self.class_,
            "page": self.page,
            "is_meta": self.is_meta,
            "line_id": self.line_id,
            "parent_id": self.parent_id,
            "relation": self.relation,
            "page_object_id": self.page_object_id,
        }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--v7-json", type=Path, help="One *_content_list_v7_styles.json file")
    input_group.add_argument("--document-ir", type=Path, help="One DocumentIR JSON file")
    input_group.add_argument("--manifest", type=Path, help="Manifest containing v7/document-ir records")
    parser.add_argument("--out-dir", type=Path, required=True, help="Output folder for CompHRDoc hr_json files")
    parser.add_argument("--pdf", type=Path, help="Optional source PDF for single-file conversion")
    parser.add_argument("--doc-id", help="Optional doc id for single-file conversion")
    parser.add_argument(
        "--allow-unstyled",
        action="store_true",
        help="Allow *_content_list_v7.json without style spans",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail the whole manifest on first bad record instead of skipping it",
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    outputs: list[dict[str, Any]] = []
    for spec in iter_input_specs(args):
        try:
            document = load_document(spec, allow_unstyled=args.allow_unstyled)
            records = export_document_ir(document)
            output_path = args.out_dir / f"{safe_filename(document.doc_id)}.json"
            output_path.write_text(
                json.dumps([record.to_json() for record in records], ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            outputs.append(
                {
                    "doc_id": document.doc_id,
                    "output": str(output_path),
                    "nodes": len(document.nodes),
                    "exported_units": len(records),
                }
            )
            print(f"[comphrdoc] wrote {output_path} units={len(records)}", flush=True)
        except Exception as exc:  # noqa: BLE001 - manifest converter should report all bad records.
            failure = {"input": str(spec), "error": repr(exc)}
            failures.append(failure)
            print(f"[comphrdoc][skip] {failure}", file=sys.stderr, flush=True)
            if args.strict:
                raise

    summary = {
        "schema_version": COMPHRDOC_SCHEMA_VERSION,
        "outputs": outputs,
        "failures": failures,
        "notes": [
            "This bridge targets CompHRDoc HDS/reading-order/classification shared structure.",
            "Fine-grained author/date/mail/affiliation classes are best-effort from v7 metadata and text cues.",
            "Generated units are block-level; CompHRDoc official data is often line-level, so absolute scores must be interpreted with that granularity gap in mind.",
        ],
    }
    # CompHRDoc official scripts treat every entry in gt/pred folders as a
    # document JSON.  Keep conversion metadata *outside* the hr_json folder.
    manifest_path = args.out_dir.parent / f"{args.out_dir.name}_conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[comphrdoc] converted={len(outputs)} failures={len(failures)} manifest={manifest_path}")
    return 0 if outputs else 1


def iter_input_specs(args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    if args.v7_json is not None:
        yield {"v7_json": args.v7_json, "pdf": args.pdf, "doc_id": args.doc_id}
        return
    if args.document_ir is not None:
        yield {"document_ir": args.document_ir, "doc_id": args.doc_id}
        return

    payload = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = payload.get("documents", payload.get("records", payload.get("samples", payload))) if isinstance(payload, dict) else payload
    if not isinstance(records, list):
        raise ValueError(f"Unsupported manifest shape: {args.manifest}")
    for record in records:
        if not isinstance(record, dict):
            continue
        v7_path = first_existing_path(
            record,
            "content_v7_path",
            "content_v7_styles_path",
            "v7_json",
            "v7_path",
            "content_json_path",
            "content_path",
            "styled_content_json",
        )
        document_ir = first_existing_path(record, "document_ir_path", "document_ir", "ir_path")
        pdf = first_existing_path(record, "pdf_path", "source_pdf", "compiled_pdf", "pdf")
        if v7_path is None and document_ir is None:
            continue
        yield {
            "v7_json": v7_path,
            "document_ir": document_ir,
            "pdf": pdf,
            "doc_id": record.get("doc_id") or record.get("paper_id") or record.get("arxiv_id") or record.get("id"),
            "record": record,
        }


def load_document(spec: dict[str, Any], *, allow_unstyled: bool) -> DocumentIR:
    if spec.get("document_ir"):
        return read_dataclass_json(Path(spec["document_ir"]), DocumentIR)
    if spec.get("v7_json"):
        from src.adapters.mineru_v7_document_ir import MinerUV7DocumentIRAdapterConfig

        config = MinerUV7DocumentIRAdapterConfig(require_styles=not allow_unstyled)
        return load_v7_document_ir(
            Path(spec["v7_json"]),
            pdf_path=Path(spec["pdf"]) if spec.get("pdf") else None,
            doc_id=str(spec["doc_id"]) if spec.get("doc_id") else None,
            config=config,
        )
    raise ValueError(f"Cannot load document from spec: {spec}")


def export_document_ir(document: DocumentIR) -> list[ExportRecord]:
    nodes = sorted(document.nodes, key=lambda node: (node.reading_index, node.page_idx, node.node_id))
    records: list[ExportRecord] = []
    line_id_by_node: dict[str, int] = {}
    current_section_line_id: int | None = None
    last_text_line_id: int | None = None
    last_text_page_object_id: int | None = None
    page_object_counter = 0
    float_stack: list[int] = []

    for node in nodes:
        if should_skip_node(node):
            continue
        class_name = class_for_node(node)
        is_meta = is_meta_node(node, class_name)
        line_id = len(records)
        line_id_by_node[node.node_id] = line_id

        parent_id = -1
        relation = "contain"
        page_object_id = page_object_counter

        if class_name in {"title", "author", "mail", "affili", "header", "footer", "footnote"}:
            relation = "meta" if class_name in {"title", "author", "mail", "affili"} else "contain"
            last_text_line_id = None
            last_text_page_object_id = None
        elif class_name == "section":
            parent_id = infer_heading_parent(records, node)
            current_section_line_id = line_id
            last_text_line_id = None
            last_text_page_object_id = None
        elif class_name in {"figure", "table"}:
            parent_id = current_section_line_id if current_section_line_id is not None else -1
            float_stack.append(line_id)
            last_text_line_id = None
            last_text_page_object_id = None
        elif class_name == "caption":
            parent_id = nearest_float_parent(records, node, float_stack)
            if parent_id < 0 and current_section_line_id is not None:
                parent_id = current_section_line_id
            relation = "contain"
            last_text_line_id = None
            last_text_page_object_id = None
        elif class_name == "equation":
            parent_id = last_text_line_id if last_text_line_id is not None else (current_section_line_id if current_section_line_id is not None else -1)
            relation = "contain"
            last_text_line_id = None
            last_text_page_object_id = None
        elif class_name in {"fstline", "paraline"}:
            if is_continuation_text(node, records, last_text_line_id):
                class_name = "paraline"
                parent_id = int(last_text_line_id)
                relation = "connect"
                page_object_id = int(last_text_page_object_id if last_text_page_object_id is not None else page_object_counter)
            else:
                class_name = "fstline"
                parent_id = current_section_line_id if current_section_line_id is not None else -1
                relation = "contain"
                page_object_id = page_object_counter
            last_text_line_id = line_id
            last_text_page_object_id = page_object_id

        records.append(
            ExportRecord(
                text=normalized_record_text(node.text),
                box=box_for_node(node),
                class_=class_name,
                page=int(node.page_idx),
                is_meta=is_meta,
                line_id=line_id,
                parent_id=parent_id,
                relation=relation,
                page_object_id=page_object_id,
            )
        )
        page_object_counter = max(page_object_counter + 1, page_object_id + 1)

    return records


def class_for_node(node: DocumentNode) -> str:
    raw = f"{node.raw_type or ''} {node.metadata.get('layout_role') or ''} {node.metadata.get('layout_layer') or ''}".casefold()
    text = node.text.strip()
    if node.node_type is BlockType.HEADER_FOOTER:
        return "header" if "header" in raw else "footer"
    if node.node_type is BlockType.FOOTNOTE:
        return "footnote"
    if node.node_type is BlockType.FIGURE:
        return "figure"
    if node.node_type is BlockType.TABLE:
        return "table"
    if node.node_type in {BlockType.EQUATION, BlockType.INLINE_MATH}:
        return "equation"
    if is_caption_node(node):
        return "caption"
    if node.node_type is BlockType.TITLE:
        if is_document_title(node):
            return "title"
        return "section"
    if node.node_type is BlockType.REFERENCE:
        return "paraline"
    if looks_like_author(node):
        return "author"
    if EMAIL_RE.search(text):
        return "mail"
    if looks_like_affiliation(node):
        return "affili"
    if node.node_type in {BlockType.TEXT, BlockType.LIST, BlockType.CODE, BlockType.ALGORITHM, BlockType.OTHER}:
        if CAPTION_RE.match(text):
            return "caption"
        if node.node_type is BlockType.ALGORITHM:
            return "table"
        return "fstline"
    return "fstline"


def is_caption_node(node: DocumentNode) -> bool:
    raw = f"{node.raw_type or ''} {node.metadata.get('layout_role') or ''}".casefold()
    return "caption" in raw or CAPTION_RE.match(node.text.strip()) is not None


def is_document_title(node: DocumentNode) -> bool:
    if node.reading_index > 5 or node.page_idx != 0:
        return False
    text = node.text.strip()
    if NUMBERED_HEADING_RE.match(text):
        return False
    if len(text.split()) >= 4:
        return True
    return node.metadata.get("layout_role") in {"document_title", "paper_title"}


def looks_like_author(node: DocumentNode) -> bool:
    if node.page_idx != 0 or node.reading_index > 8:
        return False
    text = node.text.strip()
    if EMAIL_RE.search(text):
        return False
    if any(token in text.casefold() for token in ("university", "institute", "department", "school", "college", "laboratory")):
        return False
    words = [word for word in re.split(r"[\s,;]+", text) if word]
    capitalized = sum(1 for word in words if word[:1].isupper())
    return 2 <= len(words) <= 12 and capitalized >= max(2, len(words) // 2)


def looks_like_affiliation(node: DocumentNode) -> bool:
    if node.page_idx != 0 or node.reading_index > 10:
        return False
    text = node.text.casefold()
    return any(token in text for token in ("university", "institute", "department", "school", "college", "laboratory"))


def is_meta_node(node: DocumentNode, class_name: str) -> bool:
    if class_name in {"title", "author", "mail", "affili", "header", "footer", "footnote"}:
        return True
    layer = str(node.metadata.get("layout_layer") or "").casefold()
    return layer in {"metadata_layer", "noise_layer", "annotation_layer"}


def should_skip_node(node: DocumentNode) -> bool:
    if node.node_type is BlockType.TOC:
        return True
    if node.flags.get("duplicate_shadow") or node.metadata.get("duplicate_shadow"):
        return True
    return not node.text.strip() and node.node_type not in {BlockType.FIGURE, BlockType.TABLE}


def infer_heading_parent(records: list[ExportRecord], node: DocumentNode) -> int:
    level = heading_level(node)
    if level <= 1:
        return -1
    for record in reversed(records):
        if record.class_ != "section":
            continue
        pseudo_node_level = heading_level_from_text(record.text)
        if pseudo_node_level < level:
            return record.line_id
    return -1


def heading_level(node: DocumentNode) -> int:
    value = node.features.get("title_numbering_level")
    if isinstance(value, (int, float)) and int(value) > 0:
        return int(value)
    return heading_level_from_text(node.text)


def heading_level_from_text(text: str) -> int:
    stripped = text.strip()
    match = re.match(r"^\s*\d+(?:\.\d+)+", stripped)
    if match:
        return min(3, match.group(0).count(".") + 1)
    if re.match(r"^\s*(?:[IVXLCDM]+|[A-Z]|Appendix\s+[A-Z])[\.)]?\s+", stripped, re.IGNORECASE):
        return 1
    return 1


def nearest_float_parent(records: list[ExportRecord], node: DocumentNode, float_stack: list[int]) -> int:
    if not float_stack:
        return -1
    node_center = bbox_center(box_for_node(node))
    best_id = -1
    best_score = float("inf")
    for candidate_id in reversed(float_stack[-8:]):
        record = records[candidate_id]
        if record.page != node.page_idx:
            continue
        center = bbox_center(record.box)
        score = abs(center[0] - node_center[0]) + 0.5 * abs(center[1] - node_center[1])
        if score < best_score:
            best_score = score
            best_id = candidate_id
    return best_id


def is_continuation_text(node: DocumentNode, records: list[ExportRecord], last_text_line_id: int | None) -> bool:
    if last_text_line_id is None:
        return False
    previous = records[last_text_line_id]
    if previous.class_ not in {"fstline", "paraline"}:
        return False
    if previous.page != node.page_idx:
        return False
    prev_text = previous.text.strip()
    text = node.text.strip()
    if not prev_text or not text:
        return False
    if node.node_type is BlockType.LIST or re.match(r"^\s*(?:[•\-*○▪]|\d+\.|[a-zA-Z]\.)\s+", text):
        return False
    if not TERMINAL_RE.search(prev_text):
        return True
    prev_box = previous.box
    curr_box = box_for_node(node)
    vertical_gap = max(0, curr_box[1] - prev_box[3])
    return vertical_gap < 18 and abs(curr_box[0] - prev_box[0]) < 20


def box_for_node(node: DocumentNode) -> list[int]:
    if not node.bboxes:
        return [0, 0, 0, 0]
    x0 = min(bbox.x0 for bbox in node.bboxes)
    y0 = min(bbox.y0 for bbox in node.bboxes)
    x1 = max(bbox.x1 for bbox in node.bboxes)
    y1 = max(bbox.y1 for bbox in node.bboxes)
    return [int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))]


def bbox_center(box: list[int] | list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def normalized_record_text(text: str) -> str:
    return " ".join(str(text or "").replace("\r", "\n").split())


def first_existing_path(record: dict[str, Any], *keys: str) -> Path | None:
    for key in keys:
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value)
    return None


def safe_filename(value: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_", "."} else "_" for char in str(value))
    return cleaned or "document"


if __name__ == "__main__":
    raise SystemExit(main())
