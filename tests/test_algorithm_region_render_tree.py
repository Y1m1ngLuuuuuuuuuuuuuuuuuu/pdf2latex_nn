from __future__ import annotations

from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR
from src.reasoning.v8_render_tree import build_v8_render_tree


def test_algorithm_region_renderer_flag_off_preserves_tree_shape() -> None:
    node = DocumentNode(
        node_id="n_alg",
        node_type=BlockType.ALGORITHM,
        text="Algorithm 1: Train\nreturn model",
        page_idx=0,
        bboxes=[BBox(10, 20, 200, 160)],
        reading_index=0,
        raw_type="code",
        metadata={
            "is_algorithm_subtype": True,
            "algorithm_confidence": "strong_subtype",
            "raw_sub_type": "algorithm",
            "code_caption": ["Algorithm 1: Train"],
            "code_body": "return model",
        },
    )
    document = DocumentIR(doc_id="doc", pages=[PageIR(page_idx=0, width=600, height=800, node_ids=[node.node_id])], nodes=[node])
    off_tree = build_v8_render_tree(document, document_ir_path="document_ir.json")
    on_tree = build_v8_render_tree(document, document_ir_path="document_ir.json", enable_algorithm_region_renderer=True)
    assert len(on_tree.nodes) >= len(off_tree.nodes)
    assert not any(render_node.attributes.get("algorithm_region_phase0") for render_node in off_tree.nodes)
    assert any(render_node.attributes.get("algorithm_region_phase0") for render_node in on_tree.nodes)
    assert on_tree.metadata.get("experimental_algorithm_region_renderer_phase0_enabled") is True

