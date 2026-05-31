from __future__ import annotations

import json

from tools.baselines.convert_contentlist_direct_to_comparison import contentlist_to_comparison
from tools.baselines.convert_mineru_direct_to_comparison import mineru_middle_to_comparison


def write_json(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return path


def block_types(document):
    return [block.block_type for block in document.blocks]


def test_contentlist_title_heading_paragraph_conversion(tmp_path):
    path = write_json(
        tmp_path / "content_list_v2.json",
        [
            [
                {"type": "title", "content": {"level": 1, "title_content": [{"type": "text", "content": "Paper Title"}]}},
                {"type": "title", "content": {"level": 1, "title_content": [{"type": "text", "content": "Introduction"}]}},
                {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": "Body text."}]}},
            ]
        ],
    )
    document = contentlist_to_comparison(path, doc_id="demo")
    assert document.schema_version == "comparison_structure_v1"
    assert block_types(document) == ["document_title", "heading", "paragraph"]
    assert [block.block_id for block in document.blocks] == ["B_000001", "B_000002", "B_000003"]
    assert document.to_dict()["reading_order"] == ["B_000001", "B_000002", "B_000003"]


def test_contentlist_figure_table_caption_conversion(tmp_path):
    path = write_json(
        tmp_path / "content_list_v2.json",
        [
            [
                {
                    "type": "image",
                    "content": {
                        "image_caption": [{"type": "text", "content": "Figure 1: A curve"}],
                        "image_body": [],
                    },
                },
                {
                    "type": "table",
                    "content": {
                        "table_body": "<table><tr><td>A</td></tr></table>",
                        "table_caption": [{"type": "text", "content": "Table 1: Scores"}],
                    },
                },
            ]
        ],
    )
    document = contentlist_to_comparison(path, doc_id="demo")
    assert block_types(document) == ["figure", "caption", "table", "caption"]
    assert document.blocks[1].parent_id == document.blocks[0].block_id
    assert document.blocks[1].marker == "figure"
    assert document.blocks[3].parent_id == document.blocks[2].block_id
    assert document.blocks[3].marker == "table"


def test_contentlist_reference_requires_source_reference_mark(tmp_path):
    path = write_json(
        tmp_path / "content_list.json",
        [
            {"type": "text", "text": "Table 1 shows results [12].", "page_idx": 0},
            {"type": "ref_text", "text": "[12] A. Author. Title.", "page_idx": 5},
        ],
    )
    document = contentlist_to_comparison(path, doc_id="demo")
    assert block_types(document) == ["paragraph", "reference_item"]


def test_contentlist_list_and_inline_math_conversion(tmp_path):
    path = write_json(
        tmp_path / "content_list_v2.json",
        [
            [
                {
                    "type": "index",
                    "content": {
                        "list_items": [
                            {"content": [{"type": "text", "content": "first"}]},
                            {"content": [{"type": "equation_inline", "content": "x+y"}]},
                        ]
                    },
                }
            ]
        ],
    )
    document = contentlist_to_comparison(path, doc_id="demo")
    assert block_types(document) == ["list_item", "list_item"]
    assert document.blocks[1].text == "[MATH]"


def test_mineru_heading_paragraph_list_conversion(tmp_path):
    path = write_json(
        tmp_path / "middle.json",
        {
            "pdf_info": [
                {
                    "preproc_blocks": [
                        {"type": "title", "level": 1, "lines": [{"spans": [{"content": "MinerU Title"}]}]},
                        {"type": "text", "lines": [{"spans": [{"content": "A paragraph."}]}]},
                        {"type": "index", "items": [{"content": "One"}, {"content": "Two"}]},
                    ]
                }
            ]
        },
    )
    document = mineru_middle_to_comparison(path, doc_id="demo")
    assert block_types(document) == ["document_title", "paragraph", "list_item", "list_item"]


def test_mineru_table_markdown_html_conversion(tmp_path):
    path = write_json(
        tmp_path / "middle.json",
        {
            "pdf_info": [
                {
                    "preproc_blocks": [
                        {"type": "table", "lines": [{"spans": [{"html": "<table><tr><td>A</td></tr></table>"}]}]},
                        {"type": "text", "lines": [{"spans": [{"content": "Table 1: Caption"}]}]},
                    ]
                }
            ]
        },
    )
    document = mineru_middle_to_comparison(path, doc_id="demo")
    assert block_types(document) == ["table", "caption"]
    assert "table" not in document.blocks[0].text.lower()
    assert document.blocks[1].parent_id == document.blocks[0].block_id


def test_output_to_dict_schema_and_missing_optional_fields(tmp_path):
    path = write_json(tmp_path / "content_list.json", [{"type": "text", "text": "Plain body."}])
    payload = contentlist_to_comparison(path, doc_id="demo").to_dict()
    assert payload["schema_version"] == "comparison_structure_v1"
    assert payload["doc_id"] == "demo"
    assert payload["blocks"][0]["block_type"] == "paragraph"

