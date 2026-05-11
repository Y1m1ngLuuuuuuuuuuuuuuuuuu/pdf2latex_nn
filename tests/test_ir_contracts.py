from __future__ import annotations

import pytest

from src.ir import (
    BBox,
    BlockType,
    ContractError,
    DocumentIR,
    DocumentNode,
    GraphInput,
    GraphLabels,
    GraphTensorRef,
    PageIR,
    PredictedRelations,
    RelationLabel,
    RenderRole,
    RenderTreeIR,
    RenderTreeNode,
    RendererMode,
    StyleProfile,
)
from src.ir.serialization import read_dataclass_json, write_json
from src.ir.validators import (
    validate_document_ir,
    validate_graph_input,
    validate_graph_labels,
    validate_predicted_relations,
    validate_render_tree,
    validate_style_profile,
)


def test_document_ir_round_trip(tmp_path):
    document = DocumentIR(
        doc_id="2501.00050",
        source_pdf="data/01_raw_pdfs/2501.00050.pdf",
        pages=[PageIR(page_idx=0, width=1000.0, height=1000.0, node_ids=["n0"])],
        nodes=[
            DocumentNode(
                node_id="n0",
                node_type=BlockType.TEXT,
                text="hello",
                page_idx=0,
                bboxes=[BBox(10.0, 20.0, 100.0, 40.0)],
                reading_index=0,
            )
        ],
        reading_order=["n0"],
    )

    path = tmp_path / "document_ir.json"
    write_json(path, document)
    loaded = read_dataclass_json(path, DocumentIR)

    validate_document_ir(loaded)
    assert loaded.nodes[0].node_type is BlockType.TEXT
    assert loaded.nodes[0].bboxes[0].to_list() == [10.0, 20.0, 100.0, 40.0]


def test_document_ir_rejects_missing_reading_order_node():
    document = DocumentIR(
        doc_id="doc",
        pages=[PageIR(page_idx=0, width=1000.0, height=1000.0, node_ids=["n0"])],
        nodes=[
            DocumentNode(
                node_id="n0",
                node_type=BlockType.TEXT,
                text="hello",
                page_idx=0,
                bboxes=[BBox(0.0, 0.0, 10.0, 10.0)],
                reading_index=0,
            )
        ],
        reading_order=[],
    )

    validate_document_ir(document)

    bad = DocumentIR(
        doc_id="doc",
        pages=document.pages,
        nodes=document.nodes,
        reading_order=["n0", "ghost"],
    )
    with pytest.raises(ContractError):
        validate_document_ir(bad)


def test_graph_input_and_labels_contracts():
    graph = GraphInput(
        doc_id="doc",
        document_ir_path="doc_ir.json",
        graph_path="graph.pt",
        node_ids=["n0", "n1"],
        edge_ids=["e0", "e1"],
        x=GraphTensorRef(path="graph.pt", tensor_name="x", shape=[2, 831], dtype="float32"),
        edge_index=GraphTensorRef(path="graph.pt", tensor_name="edge_index", shape=[2, 2], dtype="int64"),
        edge_attr=GraphTensorRef(path="graph.pt", tensor_name="edge_attr", shape=[2, 22], dtype="float32"),
    )
    labels = GraphLabels(
        doc_id="doc",
        graph_input_path="graph_input.json",
        edge_ids=["e0", "e1"],
        y=[RelationLabel.MERGE, RelationLabel.NONE],
    )

    validate_graph_input(graph)
    validate_graph_labels(labels)

    bad_labels = GraphLabels(
        doc_id="doc",
        graph_input_path="graph_input.json",
        edge_ids=["e0"],
        y=[RelationLabel.MERGE, RelationLabel.NONE],
    )
    with pytest.raises(ContractError):
        validate_graph_labels(bad_labels)


def test_predictions_render_tree_and_style_profile_contracts():
    predictions = PredictedRelations(
        doc_id="doc",
        graph_input_path="graph_input.json",
        edge_ids=["e0"],
        predicted_labels=[RelationLabel.PARENT_CHILD],
        probabilities=[[0.1, 0.8, 0.1]],
        threshold_config={"merge": 0.42, "parent_child": 0.53},
    )
    tree = RenderTreeIR(
        doc_id="doc",
        root_id="r0",
        document_ir_path="document_ir.json",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["r1"]),
            RenderTreeNode(render_id="r1", role=RenderRole.PARAGRAPH, source_node_ids=["n0"], text="hello"),
        ],
    )
    profile = StyleProfile(profile_id="original", mode=RendererMode.ORIGINAL_LIKE)

    validate_predicted_relations(predictions)
    validate_render_tree(tree)
    validate_style_profile(profile)


def test_render_tree_rejects_cycles():
    tree = RenderTreeIR(
        doc_id="doc",
        root_id="r0",
        document_ir_path="document_ir.json",
        nodes=[
            RenderTreeNode(render_id="r0", role=RenderRole.ROOT, children=["r1"]),
            RenderTreeNode(render_id="r1", role=RenderRole.PARAGRAPH, children=["r0"]),
        ],
    )

    with pytest.raises(ContractError):
        validate_render_tree(tree)
