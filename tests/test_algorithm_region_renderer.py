from __future__ import annotations

from src.generation.latex_helpers import render_algorithm_region_phase0
from src.ir import BBox, BlockType, DocumentIR, DocumentNode, PageIR, RenderRole, RenderTreeIR, RenderTreeNode
from src.reasoning.algorithm_region_renderer import apply_algorithm_region_renderer, build_algorithm_region_sidecars


def _document(node: DocumentNode) -> DocumentIR:
    return DocumentIR(doc_id="doc", pages=[PageIR(page_idx=0, width=600, height=800, node_ids=[node.node_id])], nodes=[node])


def _algorithm_node(**metadata) -> DocumentNode:
    base_metadata = {
        "is_algorithm_subtype": True,
        "algorithm_confidence": "strong_subtype",
        "algorithm_origin": "raw_content_list",
        "raw_sub_type": "algorithm",
        "code_caption": ["Algorithm 1: Train"],
        "code_body": "Input: x_y\nreturn x % y",
    }
    base_metadata.update(metadata)
    return DocumentNode(
        node_id="n_alg",
        node_type=BlockType.ALGORITHM,
        text="Algorithm 1: Train\nInput: x_y\nreturn x % y",
        page_idx=0,
        bboxes=[BBox(10, 20, 200, 160)],
        reading_index=0,
        raw_type="code",
        metadata=base_metadata,
    )


def _tree() -> RenderTreeIR:
    return RenderTreeIR(
        doc_id="doc",
        root_id="root",
        document_ir_path="document_ir.json",
        nodes=[
            RenderTreeNode(render_id="root", role=RenderRole.ROOT, children=["r_n_alg"]),
            RenderTreeNode(render_id="r_n_alg", role=RenderRole.ALGORITHM, source_node_ids=["n_alg"], text="Algorithm 1: Train"),
        ],
    )


def test_strong_subtype_algorithm_region_created() -> None:
    result = build_algorithm_region_sidecars(_document(_algorithm_node()))
    assert len(result.regions) == 1
    assert result.regions[0].algorithm_confidence == "strong_subtype"
    assert result.regions[0].render_policy in {"crop_fallback", "verbatim_fallback"}


def test_ambiguous_or_weak_text_only_not_rendered() -> None:
    document = _document(_algorithm_node(algorithm_confidence="weak_text_only", is_algorithm_subtype=False))
    tree, result = apply_algorithm_region_renderer(document, _tree(), enabled=True)
    assert not result.rendered_regions
    assert all(not node.attributes.get("algorithm_region_phase0") for node in tree.nodes)


def test_algorithm_reference_not_rendered_as_algorithm() -> None:
    node = DocumentNode(
        node_id="n_text",
        node_type=BlockType.TEXT,
        text="Algorithm 1 shows the training procedure.",
        page_idx=0,
        bboxes=[BBox(10, 20, 200, 60)],
        reading_index=0,
    )
    result = build_algorithm_region_sidecars(_document(node))
    assert not result.regions


def test_flag_on_materializes_region_and_consumes_source_node() -> None:
    tree, result = apply_algorithm_region_renderer(_document(_algorithm_node()), _tree(), enabled=True)
    assert result.rendered_regions
    assert any(node.attributes.get("algorithm_region_phase0") for node in tree.nodes)
    root = next(node for node in tree.nodes if node.render_id == "root")
    assert "r_n_alg" not in root.children
    assert result.consumed_nodes


def test_phase0_text_fallback_escapes_special_chars() -> None:
    tex = render_algorithm_region_phase0(caption="Algorithm 1: Train", body="x_y = 1 % mask & score", render_policy="verbatim_fallback")
    assert r"x\_y" in tex
    assert r"\%" in tex
    assert r"\&" in tex
    assert "\\begin{figure}[H]" in tex
