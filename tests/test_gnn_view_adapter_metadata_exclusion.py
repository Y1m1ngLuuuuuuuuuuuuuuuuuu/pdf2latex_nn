from copy import deepcopy

from src.perception.gnn_view_adapter import build_gnn_view


def test_missing_layer_author_role_is_excluded_from_gnn_view():
    items = [
        {"type": "paragraph", "text_for_embedding": "Body text.", "layout_role": "body_text"},
        {"type": "paragraph", "text_for_embedding": "A. Author", "layout_role": "author"},
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text."]
    assert view.excluded_items_summary["excluded_by_reason"]["metadata_role:author"] == 1


def test_missing_layer_page_number_type_is_excluded_from_gnn_view():
    items = [
        {"type": "paragraph", "text_for_embedding": "Body text.", "layout_role": "body_text"},
        {"type": "page_number", "text_for_embedding": "12"},
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text."]
    assert view.excluded_items_summary["excluded_by_reason"]["page_furniture:page_number"] == 1


def test_missing_layer_affiliation_canonical_type_is_excluded_from_gnn_view():
    items = [
        {"type": "paragraph", "text_for_embedding": "Body text.", "layout_role": "body_text"},
        {"type": "paragraph", "canonical_type": "affiliation", "text_for_embedding": "Example University"},
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text."]
    assert view.excluded_items_summary["excluded_by_reason"]["metadata_role:affiliation"] == 1


def test_metadata_layer_is_excluded_from_gnn_view():
    items = [
        {"type": "paragraph", "text_for_embedding": "Body text.", "layout_role": "body_text"},
        {"type": "title", "layout_layer": "metadata_layer", "text_for_embedding": "Paper Title"},
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text."]
    assert view.excluded_items_summary["excluded_by_reason"]["metadata:title"] == 1


def test_main_text_title_heading_is_not_excluded_by_raw_title_type():
    items = [
        {
            "type": "title",
            "layout_layer": "main_text_flow",
            "layout_role": "section_heading",
            "text_for_embedding": "1 Introduction",
        }
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["1 Introduction"]


def test_float_objects_still_enter_float_proxy_path():
    items = [
        {
            "type": "table",
            "layout_layer": "float_layer",
            "layout_role": "table",
            "text_for_embedding": "raw cell OCR",
            "table_caption": "Table 1: Results.",
        }
    ]

    view = build_gnn_view(items)

    assert len(view.gnn_items) == 1
    assert view.gnn_items[0]["type"] == "table"
    assert view.gnn_items[0]["gnn_proxy_kind"] == "float_proxy"
    assert view.gnn_items[0]["text_for_embedding"] == "Table 1: Results."


def test_excluded_summary_records_nested_metadata_reason():
    items = [
        {"type": "paragraph", "text_for_embedding": "Body text.", "layout_role": "body_text"},
        {"type": "paragraph", "text_for_embedding": "Running title", "metadata": {"role": "page_header"}},
    ]

    view = build_gnn_view(items)

    assert view.excluded_items_summary["excluded_by_reason"]["page_furniture:page_header"] == 1


def test_adapter_does_not_mutate_full_v7_records():
    items = [
        {"type": "paragraph", "text_for_embedding": "Body text.", "layout_role": "body_text"},
        {"type": "page_number", "text_for_embedding": "12"},
    ]
    original = deepcopy(items)

    build_gnn_view(items)

    assert items == original
