from __future__ import annotations

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RenderRole
from src.reasoning.v8_render_tree import build_v8_render_tree, role_for_source_node


def test_v8_numeric_list_item_marks_ordered():
    node = DocumentNode(
        node_id="li0",
        node_type=BlockType.TEXT,
        text="1. Computes embeddings for support and query sets.",
        page_idx=0,
        bboxes=[BBox(100, 100, 500, 120)],
        reading_index=0,
    )
    document = DocumentIR(
        doc_id="v8_ordered_list",
        pages=[PageIR(page_idx=0, width=1000, height=1000, node_ids=["li0"])],
        nodes=[node],
        reading_order=["li0"],
    )

    role, level, attributes = role_for_source_node(node, document=document)

    assert role == RenderRole.LIST_ITEM
    assert level == 0
    assert attributes["ordered"] is True

    tree = build_v8_render_tree(document, document_ir_path="document_ir.json")
    list_node = next(item for item in tree.nodes if item.role == RenderRole.LIST_ITEM)
    assert list_node.attributes["ordered"] is True


def test_v8_bullet_list_item_marks_unordered():
    node = DocumentNode(
        node_id="li0",
        node_type=BlockType.TEXT,
        text="• Simultaneous operation across complementary metric spaces",
        page_idx=0,
        bboxes=[BBox(100, 100, 500, 120)],
        reading_index=0,
    )

    role, level, attributes = role_for_source_node(node)

    assert role == RenderRole.LIST_ITEM
    assert level == 0
    assert attributes["ordered"] is False
