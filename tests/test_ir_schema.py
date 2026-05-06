from src.perception.schema import (
    BBox,
    Block,
    BlockType,
    Document,
    EdgeCandidate,
    EdgeRelation,
    FeatureTensorSchema,
    FEATURE_TYPE_VOCAB,
    Page,
    ReferenceItem,
    SCHEMA_VERSION,
)
from src.reasoning.graph_builder import TYPE_VOCAB


def test_feature_schema_v0_tensor_dimensions_are_fixed():
    schema = FeatureTensorSchema()

    assert schema.semantic_dim == 768
    assert schema.type_vocab == [
        "text",
        "title",
        "equation",
        "table",
        "figure",
        "algorithm",
        "list",
        "code",
        "reference",
        "other",
    ]
    assert schema.geometry_fields == ["x_start_local", "y_start_page", "x_end_local", "y_end_page"]
    assert schema.derived_stat_fields == ["macro_position", "aspect_ratio", "text_density"]
    assert schema.style_stat_fields == [
        "baseline_font_size_norm",
        "font_size_vs_doc_body",
        "bold_char_ratio",
        "italic_char_ratio",
        "inline_math_char_ratio",
        "inline_code_char_ratio",
    ]
    assert len(schema.sequence_position_fields) == 16
    assert schema.column_feature_fields == ["column_left", "column_right", "column_full_or_single"]
    assert schema.title_structure_fields == ["relative_font_size", "is_h1_pattern", "is_h2_pattern"]
    assert schema.node_feature_dim == 813
    assert schema.edge_attr_dim == 11
    assert schema.edge_attr_fields == [
        "semantic_cosine",
        "delta_y_gap",
        "delta_x_left",
        "left_alignment",
        "center_distance",
        "font_size_delta",
        "bold_to_regular",
        "line_height_ratio",
        "index_delta",
        "is_next",
        "is_sequential_edge",
    ]


def test_graph_builder_type_vocab_uses_ir_schema_vocab():
    assert TYPE_VOCAB == FEATURE_TYPE_VOCAB


def test_document_ir_can_represent_structured_references():
    reference = Block(
        block_id="b000001",
        global_order=1,
        block_type=BlockType.REFERENCE,
        raw_type="list",
        list_type="reference_list",
        page_idx=0,
        bboxes=[BBox(100.0, 200.0, 500.0, 260.0)],
        text="Author A. Paper title.",
        reference_items=[ReferenceItem(text="Author A. Paper title.", raw_index=0)],
        source_page_idxs=[0],
        source_visual_orders=[1],
        source_original_indexes=[7],
    )
    document = Document(
        document_id="sample",
        source_pdf="sample.pdf",
        pages=[Page(page_idx=0, width=1000.0, height=1000.0, blocks=["b000001"])],
        blocks=[reference],
        edges=[
            EdgeCandidate(
                source_block_id="b000000",
                target_block_id="b000001",
                relation=EdgeRelation.NEXT_READING_ORDER,
            )
        ],
    )

    assert document.schema_version == SCHEMA_VERSION
    assert document.blocks[0].block_type == BlockType.REFERENCE
    assert document.blocks[0].reference_items[0].text == "Author A. Paper title."
