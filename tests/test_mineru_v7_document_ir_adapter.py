from __future__ import annotations

import json

from src.adapters import convert_v7_payload_to_document_ir, load_v7_document_ir
from src.generation.table_assets import annotate_table_group_records
from src.ir import BlockType, DocumentIR
from src.ir.serialization import read_dataclass_json, write_json
from src.ir.validators import validate_document_ir


def build_v7_payload() -> dict:
    return {
        "schema_version": "content_v7_columnfix_listmarkers_with_styles",
        "source_format": "mineru_content_list_v2",
        "style_source_pdf": "paper.pdf",
        "items": [
            {
                "global_order": 0,
                "page_idx": 0,
                "mineru_block_idx": 3,
                "type": "title",
                "raw_type": "title",
                "bbox": [100, 20, 900, 80],
                "text": "Demo Paper",
                "layout_layer": "metadata_layer",
                "layout_role": "front_matter",
                "style_baseline_size": 16.0,
                "style_spans": [
                    {
                        "text": "Demo Paper",
                        "font_name": "Times-Bold",
                        "font_size": 16.0,
                        "is_bold": True,
                        "is_italic": False,
                        "is_inline_math": False,
                        "is_inline_code": False,
                        "char_count": 10,
                    }
                ],
                "block": {"type": "title", "content": "Demo Paper", "bbox": [100, 20, 900, 80]},
            },
            {
                "global_order": 1,
                "page_idx": 0,
                "mineru_block_idx": 4,
                "type": "paragraph",
                "raw_type": "paragraph",
                "bbox": [100, 120, 900, 180, 100, 190, 900, 240],
                "text": "Body text [1].",
                "layout_layer": "main_text_flow",
                "layout_role": "body_text",
                "style_baseline_size": 10.0,
                "style_spans": [
                    {
                        "text": "Body text [1].",
                        "font_name": "Times-Roman",
                        "font_size": 10.0,
                        "is_bold": False,
                        "is_italic": False,
                        "is_inline_math": False,
                        "is_inline_code": False,
                        "char_count": 14,
                    }
                ],
            },
            {
                "global_order": 2,
                "page_idx": 1,
                "mineru_block_idx": 1,
                "type": "list",
                "raw_type": "list",
                "list_type": "reference_list",
                "bbox": [100, 100, 900, 160],
                "text": "[1] First paper.",
                "reference_items": [{"text": "[1] First paper."}],
                "style_spans": [],
            },
            {
                "global_order": 3,
                "page_idx": 1,
                "mineru_block_idx": 2,
                "type": "index",
                "canonical_type": "toc",
                "bbox": [100, 200, 900, 260],
                "text": "1 Introduction 3",
                "layout_layer": "metadata_layer",
                "layout_role": "toc_entry",
                "style_spans": [],
            },
        ],
    }


def test_convert_v7_payload_to_document_ir_preserves_core_fields():
    document = convert_v7_payload_to_document_ir(build_v7_payload(), doc_id="demo")

    validate_document_ir(document)
    assert document.doc_id == "demo"
    assert document.coordinate_space.value == "page_normalized_1000"
    assert document.reading_order == [
        "v7_p0000_b000003",
        "v7_p0000_b000004",
        "v7_p0001_b000001",
        "v7_p0001_b000002",
    ]
    assert [page.page_idx for page in document.pages] == [0, 1]
    assert [node.node_type for node in document.nodes] == [
        BlockType.TITLE,
        BlockType.TEXT,
        BlockType.REFERENCE,
        BlockType.TOC,
    ]
    assert len(document.nodes[1].bboxes) == 2
    assert document.nodes[0].spans[0].is_bold is True
    assert document.nodes[2].metadata["reference_items"] == [{"text": "[1] First paper."}]
    assert document.nodes[3].metadata["layout_role"] == "toc_entry"


def test_load_v7_document_ir_from_file_and_round_trip(tmp_path):
    content_path = tmp_path / "demo_content_list_v7_styles.json"
    output_path = tmp_path / "document_ir.json"
    content_path.write_text(json.dumps(build_v7_payload()), encoding="utf-8")

    document = load_v7_document_ir(content_path, doc_id="demo")
    write_json(output_path, document)
    loaded = read_dataclass_json(output_path, DocumentIR)

    validate_document_ir(loaded)
    assert loaded.doc_id == "demo"
    assert loaded.nodes[1].features["style_baseline_size"] == 10.0
    assert loaded.nodes[1].source_refs[0].path == str(content_path)


def test_table_fragments_are_grouped_by_union_bbox_metadata():
    records = annotate_table_group_records(
        [
            {
                "type": "table",
                "page_idx": 0,
                "global_order": 10,
                "bbox": [100, 100, 240, 500],
                "table_body": "<table><tr><td>A</td></tr></table>",
            },
            {
                "type": "table",
                "page_idx": 0,
                "global_order": 11,
                "bbox": [245, 102, 390, 498],
                "table_caption": ["Table 1: Wide result table."],
                "table_body": "<table><tr><td>B</td></tr></table>",
            },
        ]
    )

    assert records[0]["table_group_id"] == records[1]["table_group_id"]
    assert records[0]["table_group_primary"] is False
    assert records[1]["table_group_primary"] is True
    assert records[1]["table_group_bbox"] == [100.0, 100.0, 390.0, 500.0]
    assert records[1]["table_group_caption"] == "Table 1: Wide result table."


def test_table_ir_text_keeps_caption_but_not_cell_body():
    payload = {
        "schema_version": "content_v7_columnfix_listmarkers_with_styles",
        "items": [
            {
                "type": "table",
                "page_idx": 0,
                "global_order": 0,
                "bbox": [100, 100, 900, 300],
                "table_caption": "Table 1: Accuracy results.",
                "table_body": "<table><tr><td>model</td><td>99</td></tr></table>",
                "style_spans": [],
            }
        ],
    }

    document = convert_v7_payload_to_document_ir(payload, doc_id="table-demo")

    assert document.nodes[0].node_type == BlockType.TABLE
    assert document.nodes[0].text == "Table 1: Accuracy results."
    assert "model" not in document.nodes[0].text
    assert document.nodes[0].metadata["table_body"] == "<table><tr><td>model</td><td>99</td></tr></table>"


def test_duplicate_shadow_nodes_are_not_rendered_in_document_ir():
    payload = {
        "schema_version": "content_v7_columnfix_listmarkers_with_styles",
        "items": [
            {
                "type": "paragraph",
                "page_idx": 0,
                "global_order": 0,
                "bbox": [100, 100, 450, 180],
                "text": "All experiments used an NVIDIA RTX 8000 GPU with 48GB memory.",
                "layout_layer": "main_text_flow",
                "style_spans": [],
            },
            {
                "type": "paragraph",
                "page_idx": 1,
                "global_order": 1,
                "bbox": [100, 80, 450, 120],
                "text": "with 48GB memory.",
                "layout_layer": "main_text_flow",
                "style_spans": [],
            },
        ],
    }

    document = convert_v7_payload_to_document_ir(payload, doc_id="dedupe-demo")

    assert [node.text for node in document.nodes] == [
        "All experiments used an NVIDIA RTX 8000 GPU with 48GB memory."
    ]
