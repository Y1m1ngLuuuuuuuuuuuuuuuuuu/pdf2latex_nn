from __future__ import annotations

from src.reasoning.front_matter_context_group import front_matter_contexts_from_document_ir_check
from src.reasoning.page_furniture_context_group import contexts_from_document_ir_check
from tools.audit.validate_page_furniture_context_phase1_mineru_evidence import regex_only_furniture_items


def _node(node_id: str, text: str, metadata: dict, *, page_idx: int = 0):
    return {"node_id": node_id, "page_idx": page_idx, "text_preview": text, "metadata": metadata}


def test_content_list_header_becomes_negative_evidence():
    check = {
        "after_page_furniture_nodes": [
            _node(
                "h1",
                "Proceedings Header",
                {
                    "mineru_page_furniture_role": "page_header",
                    "page_furniture_source_layer": "content_list",
                    "page_furniture_confidence": "strong_content_list_role",
                    "should_exclude_from_heading_detection": True,
                    "should_exclude_from_visible_prose_metric": True,
                },
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert context.context_kind == "page_header"
    assert context.confidence_tier == "high"
    assert "heading_detection" in context.negative_masks


def test_content_list_footer_becomes_negative_evidence():
    check = {
        "after_page_furniture_nodes": [
            _node(
                "f1",
                "Footer",
                {
                    "mineru_page_furniture_role": "page_footer",
                    "page_furniture_source_layer": "content_list",
                    "page_furniture_confidence": "strong_content_list_role",
                    "should_exclude_from_body_order": True,
                },
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert context.context_kind == "page_footer"
    assert "body_order" in context.negative_masks


def test_content_list_page_number_becomes_negative_evidence():
    check = {
        "after_page_furniture_nodes": [
            _node(
                "p1",
                "12",
                {
                    "mineru_page_furniture_role": "page_number",
                    "page_furniture_source_layer": "content_list",
                    "page_furniture_confidence": "strong_content_list_role",
                    "should_exclude_from_visible_prose_metric": True,
                },
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert context.context_kind == "page_number"
    assert "visible_prose" in context.negative_masks


def test_page_footnote_is_note_not_ordinary_prose():
    check = {
        "after_page_furniture_nodes": [
            _node(
                "n1",
                "* Equal contribution",
                {
                    "mineru_page_furniture_role": "page_footnote",
                    "page_furniture_source_layer": "content_list",
                    "page_furniture_confidence": "strong_content_list_role",
                    "should_exclude_from_body_order": True,
                },
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert context.context_kind == "page_footnote"
    assert context.reason.startswith("strong page furniture")


def test_model_doc_title_becomes_front_matter_evidence():
    check = {
        "after_model_label_nodes": [
            _node(
                "t1",
                "A Document Title",
                {
                    "model_label": "doc_title",
                    "model_score": 0.99,
                    "model_label_confidence": "strong_model_label",
                    "is_document_title_candidate": True,
                    "front_matter_negative_for_body_heading": True,
                    "title_negative_for_body_heading": True,
                },
            )
        ]
    }
    contexts = contexts_from_document_ir_check("doc", check)
    assert contexts[0].context_kind == "document_title"
    front = front_matter_contexts_from_document_ir_check("doc", check)
    assert front[0].front_matter_role == "document_title_candidate"


def test_model_header_becomes_page_furniture_evidence():
    check = {
        "after_model_label_nodes": [
            _node(
                "mh1",
                "Running Header",
                {
                    "model_label": "header",
                    "model_score": 0.9,
                    "model_label_confidence": "strong_model_label",
                    "mineru_page_furniture_role": "page_header",
                    "should_exclude_from_heading_detection": True,
                },
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert context.context_kind == "page_header"
    assert context.evidence_source in {"mixed", "model_label", "document_ir_negative_mask"}


def test_first_page_document_title_is_not_body_heading():
    check = {
        "after_model_label_nodes": [
            _node(
                "title",
                "A Strong Paper Title",
                {"model_label": "doc_title", "model_label_confidence": "strong_model_label", "title_negative_for_body_heading": True},
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert "title_body_heading" in context.negative_masks


def test_abstract_title_is_not_ordinary_body_heading():
    check = {
        "after_model_label_nodes": [
            _node(
                "abs",
                "Abstract",
                {"is_abstract_title_candidate": True, "abstract_title_negative_for_body_heading": True},
            )
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    assert context.context_kind == "abstract_title_candidate"
    assert "abstract_title_body_heading" in context.negative_masks


def test_ordinary_body_section_heading_near_top_is_preserved():
    check = {"after_page_furniture_nodes": [], "after_model_label_nodes": []}
    assert contexts_from_document_ir_check("doc", check) == []


def test_ordinary_centered_body_heading_preserved_without_evidence():
    item = {"id": "heading", "text": "2 Method", "page_idx": 2, "bbox": [200, 60, 400, 78]}
    assert regex_only_furniture_items([item], set()) == []


def test_regex_only_page_furniture_remains_diagnostic():
    item = {"id": "n", "text": "12", "page_idx": 2, "bbox": [300, 780, 320, 790]}
    assert regex_only_furniture_items([item], set())[0]["id"] == "n"


def test_no_renderer_or_graph_changes_needed():
    check = {
        "after_model_label_nodes": [
            _node("x", "Title", {"model_label": "doc_title", "model_label_confidence": "strong_model_label"})
        ]
    }
    context = contexts_from_document_ir_check("doc", check)[0]
    payload = context.to_dict()
    assert "renderer" not in payload
    assert "graph" not in payload
