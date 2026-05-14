from src.perception.gnn_view_adapter import GNNViewAdapterConfig, build_gnn_view
from src.reasoning.postprocess import ResolvedNode
from scripts.pipeline.batch_visual_qa_inference import source_indexes_for_resolved_node


def test_gnn_view_adapter_excludes_metadata_noise_annotations_but_keeps_float_mapping():
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
    assert view.gnn_to_v7_index == [1, 2]
    assert view.gnn_to_v7_id == ["v7_p0000_b000002", "v7_p0000_b000003"]
    assert view.v7_index_to_gnn_idx == {1: 0, 2: 1}
    assert view.v7_id_to_gnn_idx["v7_p0000_b000002"] == 0
    assert view.excluded_items_summary["excluded_by_reason"]["metadata:document_title"] == 1
    assert view.excluded_items_summary["excluded_by_reason"]["noise_layer"] == 1
    assert view.excluded_items_summary["excluded_by_reason"]["annotation_layer"] == 1


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
