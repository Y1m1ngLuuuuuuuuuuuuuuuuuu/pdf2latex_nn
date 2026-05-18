from types import SimpleNamespace

import pytest

from scripts.pipeline.step5_generate_tex import merge_graph_node_metadata, select_records_for_graph
from src.perception.gnn_view_adapter import GNNViewAdapterConfig, build_gnn_view
from src.reasoning.postprocess import ResolvedNode
from scripts.pipeline.batch_visual_qa_inference import source_indexes_for_resolved_node


def test_gnn_view_adapter_excludes_metadata_noise_annotations_but_keeps_float_proxy_by_default():
    items = [
        {
            "type": "title",
            "page_idx": 0,
            "mineru_block_idx": 1,
            "text_for_embedding": "Paper Title",
            "layout_layer": "metadata_layer",
            "layout_role": "document_title",
        },
        {
            "type": "paragraph",
            "page_idx": 0,
            "mineru_block_idx": 2,
            "text_for_embedding": "Body text.",
            "layout_layer": "main_text_flow",
            "layout_role": "body_text",
        },
        {
            "type": "image",
            "page_idx": 0,
            "mineru_block_idx": 3,
            "text_for_embedding": "Figure 1: Demo.",
            "layout_layer": "float_layer",
            "layout_role": "figure_caption",
        },
        {
            "type": "page_header",
            "page_idx": 1,
            "mineru_block_idx": 4,
            "text_for_embedding": "Paper Title",
            "layout_layer": "noise_layer",
            "layout_role": "header",
        },
        {
            "type": "page_footnote",
            "page_idx": 0,
            "mineru_block_idx": 5,
            "text_for_embedding": "1 A note.",
            "layout_layer": "annotation_layer",
            "layout_role": "footnote",
        },
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text.", "Figure 1: Demo."]
    assert [item.get("gnn_proxy_kind") for item in view.gnn_items] == [None, "float_proxy"]
    assert view.gnn_items[1]["type"] == "figure"
    assert view.gnn_to_v7_index == [1, 2]
    assert view.gnn_to_v7_id == ["v7_p0000_b000002", "v7_p0000_b000003"]
    assert view.v7_index_to_gnn_idx == {1: 0, 2: 1}
    assert view.v7_id_to_gnn_idx["v7_p0000_b000002"] == 0
    assert view.excluded_items_summary["excluded_by_reason"]["metadata:document_title"] == 1
    assert view.excluded_items_summary["excluded_by_reason"]["noise_layer"] == 1
    assert view.excluded_items_summary["excluded_by_reason"]["annotation_layer"] == 1


def test_select_records_for_graph_uses_graph_v7_id_sequence_as_contract():
    items = [
        {
            "type": "title",
            "page_idx": 0,
            "mineru_block_idx": 1,
            "text_for_embedding": "Paper Title",
            "layout_layer": "metadata_layer",
            "layout_role": "document_title",
        },
        {
            "type": "paragraph",
            "page_idx": 0,
            "mineru_block_idx": 2,
            "text_for_embedding": "Body text.",
            "layout_layer": "main_text_flow",
            "layout_role": "body_text",
        },
        {
            "type": "image",
            "page_idx": 0,
            "mineru_block_idx": 3,
            "text_for_embedding": "Figure 1: Demo.",
            "layout_layer": "float_layer",
            "layout_role": "figure_caption",
        },
    ]
    data = SimpleNamespace(
        num_nodes=2,
        micro_fusion_applied=False,
        gnn_to_v7_ids=[["v7_p0000_b000002"], ["v7_p0000_b000003"]],
    )

    selected = select_records_for_graph(items, data)

    assert [item["_v7_node_id"] for item in selected] == ["v7_p0000_b000002", "v7_p0000_b000003"]


def test_select_records_for_graph_fails_fast_on_v7_id_sequence_mismatch():
    items = [
        {
            "type": "paragraph",
            "page_idx": 0,
            "mineru_block_idx": 2,
            "text_for_embedding": "Body text.",
            "layout_layer": "main_text_flow",
            "layout_role": "body_text",
        },
        {
            "type": "image",
            "page_idx": 0,
            "mineru_block_idx": 3,
            "text_for_embedding": "Figure 1: Demo.",
            "layout_layer": "float_layer",
            "layout_role": "figure_caption",
        },
    ]
    data = SimpleNamespace(
        num_nodes=2,
        micro_fusion_applied=False,
        gnn_to_v7_ids=[["v7_p0000_b000003"], ["v7_p0000_b000002"]],
    )

    with pytest.raises(ValueError, match="gnn_to_v7_ids"):
        select_records_for_graph(items, data)


def test_merge_graph_node_metadata_does_not_override_full_v7_text_with_stale_merge_text():
    content_record = {
        "type": "paragraph",
        "text": "The framework introduces a novel episodic training approach that mitigates the effects.",
        "text_for_embedding": "The framework introduces a novel episodic training approach that mitigates the effects.",
        "_v7_node_id": "v7_p0000_b000015",
    }
    graph_record = {
        "text": "The framework introduces a novel episodic training ap-",
        "text_for_embedding": "The framework introduces a novel episodic training ap-",
        "merged_text": "The framework introduces a novel episodic training ap-",
        "merged_records": [{"text": "stale continuation"}],
        "source_node_ids": [15],
        "layout_role": "list_item",
    }

    merged = merge_graph_node_metadata(content_record, graph_record)

    assert merged["text"] == content_record["text"]
    assert merged["text_for_embedding"] == content_record["text_for_embedding"]
    assert "merged_text" not in merged
    assert "merged_records" not in merged
    assert "source_node_ids" not in merged
    assert merged["layout_role"] == "list_item"


def test_gnn_view_adapter_float_proxy_uses_caption_not_table_body():
    items = [
        {
            "type": "paragraph",
            "page_idx": 0,
            "mineru_block_idx": 2,
            "text_for_embedding": "Body text.",
            "layout_layer": "main_text_flow",
            "layout_role": "body_text",
        },
        {
            "type": "table",
            "page_idx": 0,
            "mineru_block_idx": 3,
            "text_for_embedding": "cell1 cell2 cell3 noisy table body",
            "layout_layer": "float_layer",
            "layout_role": "table",
            "table_group_caption": "Table 1: Clean caption.",
        },
    ]

    view = build_gnn_view(items)

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text.", "Table 1: Clean caption."]
    assert view.gnn_items[1]["type"] == "table"
    assert view.gnn_items[1]["gnn_proxy_kind"] == "float_proxy"
    assert view.gnn_to_v7_index == [0, 1]


def test_gnn_view_adapter_can_exclude_floats_for_diagnostics():
    items = [
        {
            "type": "paragraph",
            "page_idx": 0,
            "mineru_block_idx": 2,
            "text_for_embedding": "Body text.",
            "layout_layer": "main_text_flow",
            "layout_role": "body_text",
        },
        {
            "type": "image",
            "page_idx": 0,
            "mineru_block_idx": 3,
            "text_for_embedding": "Figure OCR should not matter.",
            "layout_layer": "float_layer",
            "layout_role": "figure",
        },
    ]

    view = build_gnn_view(items, config=GNNViewAdapterConfig(include_float=False))

    assert [item["text_for_embedding"] for item in view.gnn_items] == ["Body text."]
    assert view.excluded_items_summary["excluded_by_reason"]["float:figure"] == 1


def test_gnn_view_adapter_can_keep_metadata_for_compatibility():
    items = [
        {
            "type": "title",
            "page_idx": 0,
            "mineru_block_idx": 1,
            "text_for_embedding": "Paper Title",
            "layout_layer": "metadata_layer",
            "layout_role": "document_title",
        }
    ]

    view = build_gnn_view(items, config=GNNViewAdapterConfig(include_metadata=True))

    assert len(view.gnn_items) == 1
    assert view.gnn_items[0]["_v7_node_id"] == "v7_p0000_b000001"


def test_relation_bridge_does_not_treat_v7_global_order_as_gnn_index():
    node = ResolvedNode(
        node_id=2,
        record={
            "type": "paragraph",
            "text": "continued body",
            "merged_records": [
                {
                    "text": "wrong old-style index",
                    "global_order": 99,
                    "_gnn_view_index": 3,
                    "_v7_source_index": 120,
                    "_v7_node_id": "v7_p0004_b000120",
                },
                {
                    "text": "legacy record",
                    "global_order": 4,
                },
            ],
        },
        merged_node_ids=[2],
    )

    assert source_indexes_for_resolved_node(node) == [2, 3, 4]
