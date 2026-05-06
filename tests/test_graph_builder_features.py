from src.reasoning.graph_builder import (
    TYPE_VOCAB,
    build_candidate_edge_pairs,
    build_column_onehot_matrix,
    build_derived_stats_matrix,
    build_edge_attr_matrix,
    build_geometry_matrix,
    build_logical_center_pairs,
    build_scroll_geometry_matrix,
    build_scroll_layout,
    build_sequence_position_matrix,
    build_sequential_edge_index,
    build_style_stats_matrix,
    build_title_structure_matrix,
    build_type_onehot_matrix,
    canonical_type,
    infer_column_ids,
    infer_document_body_font_size,
    infer_page_frames,
    iter_bbox_chunks,
    text_for_embedding,
)


def has_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def item(text, bbox, page=0, column=0, full=False, merge_count=1):
    chunks = len(bbox) // 4
    return {
        "global_order": 0,
        "type": "paragraph",
        "page_idx": page,
        "bbox": bbox,
        "column_id": column,
        "is_full_width": full,
        "text_for_embedding": text,
        "merge_count": merge_count,
        "source_page_idxs": [page] * chunks,
        "source_visual_orders": list(range(chunks)),
    }


def test_iter_bbox_chunks_supports_appended_bbox_format():
    assert iter_bbox_chunks([1, 2, 3, 4, 5, 6, 7, 8]) == [(1.0, 2.0, 3.0, 4.0), (5.0, 6.0, 7.0, 8.0)]


def test_infer_page_frames_finds_two_columns():
    items = [
        item("l1", [80, 100, 480, 200]),
        item("l2", [90, 220, 470, 300]),
        item("r1", [520, 100, 920, 200], column=1),
        item("r2", [530, 220, 910, 300], column=1),
    ]

    frames = infer_page_frames(items)[0]

    assert frames.left is not None
    assert frames.right is not None
    assert frames.left.x_min == 80
    assert frames.right.x_max == 920


def test_geometry_matrix_uses_first_start_and_last_end_local_coordinates():
    if not has_torch():
        return
    items = [
        item("l1", [80, 100, 480, 200]),
        item("l2", [90, 220, 470, 300]),
        item("r1", [520, 100, 920, 200], column=1),
        item("r2", [530, 220, 910, 300], column=1),
        item("merged", [80, 700, 480, 890, 520, 90, 920, 160], merge_count=2),
    ]

    geom = build_geometry_matrix(items)

    assert tuple(geom.shape) == (5, 4)
    assert [round(float(value), 4) for value in geom[0].tolist()] == [0.0, 0.1, 1.0, 0.2]
    assert round(float(geom[4][0]), 4) == 0.0
    assert round(float(geom[4][2]), 4) == 1.0


def test_scroll_geometry_matrix_unrolls_columns_into_pseudo_y_receipt_roll():
    if not has_torch():
        return
    items = [
        {**item("left top", [80, 100, 480, 200], page=0, column=0), "style_baseline_size": 10.0},
        {**item("left bottom", [80, 300, 480, 400], page=0, column=0), "style_baseline_size": 10.0},
        {**item("right top", [520, 100, 920, 200], page=0, column=1), "style_baseline_size": 10.0},
        {**item("right bottom", [520, 300, 920, 400], page=0, column=1), "style_baseline_size": 10.0},
    ]

    layout = build_scroll_layout(items, reading_order_indices=[0, 1, 2, 3])
    features = build_scroll_geometry_matrix(items, scroll_layout=layout)

    assert tuple(features.shape) == (4, 5)
    assert [round(float(value), 4) for value in features[0].tolist()] == [1.0, 0.4, 10.0, 0.125, 0.0]
    assert round(float(features[2][3]), 4) == 0.625
    assert round(float(features[3][4]), 4) == 1.0
    assert layout.boxes[2] is not None
    assert layout.boxes[2].pseudo_y0 == 500.0


def test_sequential_edges_are_bidirectional_by_default():
    if not has_torch():
        return
    edge_index = build_sequential_edge_index(3)

    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]


def test_sequence_position_matrix_adds_sixteen_dimensional_absolute_order_signal():
    if not has_torch():
        return
    pos = build_sequence_position_matrix([item("a", [0, 0, 10, 10]), item("b", [0, 20, 10, 30])])

    assert tuple(pos.shape) == (2, 16)
    assert float(pos[0][0]) == 0.0
    assert float(pos[0][1]) == 1.0
    assert round(float(pos[1][0]), 4) == 0.8415
    assert round(float(pos[1][1]), 4) == 0.5403


def test_column_onehot_matrix_marks_left_right_and_full_width_blocks():
    if not has_torch():
        return
    items = [
        item("l1", [80, 100, 480, 200]),
        item("l2", [90, 220, 470, 300]),
        item("r1", [520, 100, 920, 200], column=1),
        item("r2", [530, 220, 910, 300], column=1),
        item("full", [80, 20, 920, 80], full=True),
    ]

    column_ids = infer_column_ids(items)
    onehot = build_column_onehot_matrix(items, column_ids=column_ids)

    assert column_ids == [0, 0, 1, 1, 2]
    assert onehot.tolist() == [
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_title_structure_matrix_adds_relative_font_and_heading_patterns():
    if not has_torch():
        return
    items = [
        {
            **item("Body text.", [80, 100, 480, 130]),
            "style_baseline_size": 10.0,
        },
        {
            **item("1. Introduction", [80, 40, 480, 80]),
            "type": "title",
            "style_baseline_size": 15.0,
        },
        {
            **item("2.3 Method", [80, 150, 480, 180]),
            "type": "title",
            "style_baseline_size": 12.0,
        },
        {
            **item("II. Related Work", [80, 190, 480, 220]),
            "type": "title",
            "style_baseline_size": 14.0,
        },
    ]

    features = build_title_structure_matrix(items)

    assert tuple(features.shape) == (4, 3)
    assert [round(float(value), 4) for value in features[0].tolist()] == [1.0, 0.0, 0.0]
    assert [round(float(value), 4) for value in features[1].tolist()] == [1.5, 1.0, 0.0]
    assert [round(float(value), 4) for value in features[2].tolist()] == [1.2, 0.0, 1.0]
    assert [round(float(value), 4) for value in features[3].tolist()] == [1.4, 1.0, 0.0]


def test_candidate_edges_use_dual_view_neighbors():
    items = [
        item("a", [0, 0, 10, 10], page=0),
        item("b", [0, 20, 10, 30], page=0),
        item("c", [20, 0, 30, 10], page=0),
        item("d", [0, 40, 10, 50], page=0),
    ]

    pairs = build_candidate_edge_pairs(items, sequential_window=1, spatial_k=1)
    typed = {(source, target, source_type) for source, target, source_type in pairs}

    assert (0, 1, "sequential_forced") in typed
    assert (1, 3, "sequential_forced") in typed
    assert (3, 2, "sequential_forced") in typed
    assert (2, 1, "spatial_down") in typed
    assert any(source == 0 and target == 2 and source_type in {"spatial_right", "sequential"} for source, target, source_type in pairs)
    assert len({(source, target) for source, target, _ in pairs}) == len(pairs)


def test_edge_attr_matrix_uses_strict_fifteen_dimensional_relation_features():
    if not has_torch():
        return
    import torch

    items = [
        {
            **item("For CI-", [80, 100, 480, 200], page=0, column=0),
            "style_baseline_size": 12.0,
            "style_spans": [{"text": "For CI-", "font_size": 12.0, "is_bold": True, "char_count": 7}],
        },
        {
            **item("continues here.", [80, 220, 480, 300], page=0, column=0),
            "style_baseline_size": 10.0,
            "style_spans": [{"text": "continues here.", "font_size": 10.0, "is_bold": False, "char_count": 15}],
        },
    ]
    semantic = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)
    edge_pairs = [(0, 1, "sequential_forced"), (1, 0, "sequential")]

    edge_attr = build_edge_attr_matrix(items, semantic, edge_pairs=edge_pairs)

    assert tuple(edge_attr.shape) == (2, 15)
    forward = edge_attr[0].tolist()
    reverse = edge_attr[1].tolist()
    assert round(float(forward[0]), 4) == 1.0
    assert round(float(forward[1]), 4) == 0.02
    assert round(float(forward[2]), 4) == 0.0
    assert float(forward[3]) == 1.0
    assert round(float(forward[4]), 4) == 0.11
    assert float(forward[5]) == -2.0
    assert float(forward[6]) == 1.0
    assert round(float(forward[7]), 4) == 0.8
    assert float(forward[8]) == 0.0
    assert float(forward[9]) == 0.0
    assert float(forward[10]) == 1.0
    assert forward[10:15] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert reverse[10:15] == [0.0, 0.0, 0.0, 0.0, 1.0]


def test_edge_attr_marks_y_overlap_and_cross_column_gutter():
    if not has_torch():
        return
    import torch

    items = [
        item("left same line", [80, 100, 480, 200], page=0, column=0),
        item("right same line", [560, 120, 920, 210], page=0, column=1),
    ]
    semantic = torch.tensor([[1.0, 0.0], [1.0, 0.0]], dtype=torch.float32)

    edge_attr = build_edge_attr_matrix(items, semantic, edge_pairs=[(0, 1, "spatial_right")])

    assert round(float(edge_attr[0][8]), 4) == 0.8889
    assert float(edge_attr[0][9]) == 1.0


def test_logical_center_distance_unrolls_double_column_reading_flow():
    if not has_torch():
        return
    import torch

    items = [
        item("left top", [80, 100, 480, 200], page=0, column=0),
        item("left bottom", [80, 300, 480, 400], page=0, column=0),
        item("right top", [520, 100, 920, 200], page=0, column=1),
        item("right bottom", [520, 300, 920, 400], page=0, column=1),
    ]
    semantic = torch.ones((4, 2), dtype=torch.float32)
    edge_attr = build_edge_attr_matrix(items, semantic, edge_pairs=[(1, 2, "sequential_forced")])
    physical_distance = (((720 - 280) ** 2 + (150 - 350) ** 2) ** 0.5) / 1000.0

    assert physical_distance > 0.48
    assert round(float(edge_attr[0][4]), 4) == 0.2


def test_logical_center_pairs_keep_same_column_relative_x_not_absolute_x():
    items = [
        item("left", [80, 100, 480, 200], page=0, column=0),
        item("right", [520, 100, 920, 200], page=0, column=1),
    ]

    centers = build_logical_center_pairs(items)

    assert round(centers[0][0][0], 4) == round(centers[1][0][0], 4)


def test_canonical_type_maps_mineru_names_to_fixed_vocab():
    assert TYPE_VOCAB == ["text", "title", "equation", "table", "figure", "algorithm", "list", "code", "reference", "other"]
    assert canonical_type("paragraph") == "text"
    assert canonical_type("equation_interline") == "equation"
    assert canonical_type("chart") == "figure"
    assert canonical_type("algorithm") == "algorithm"
    assert canonical_type("code") == "code"
    assert canonical_type("reference") == "reference"
    assert canonical_type({"type": "list", "list_type": "reference_list"}) == "reference"


def test_type_onehot_matrix_uses_fixed_vocab():
    if not has_torch():
        return
    items = [
        {"type": "paragraph"},
        {"type": "title"},
        {"type": "equation_interline"},
        {"type": "table"},
        {"type": "chart"},
        {"type": "algorithm"},
        {"type": "list"},
        {"type": "code"},
        {"type": "list", "list_type": "reference_list"},
        {"type": "unknown"},
    ]

    onehot = build_type_onehot_matrix(items)

    assert tuple(onehot.shape) == (10, 10)
    assert onehot.argmax(dim=1).tolist() == list(range(10))


def test_empty_non_text_gets_placeholder_for_bert():
    assert text_for_embedding({"type": "chart", "text_for_embedding": ""}) == "[FIGURE]"
    assert text_for_embedding({"type": "table", "text_for_embedding": ""}) == "[TABLE]"
    assert text_for_embedding({"type": "equation_interline", "text_for_embedding": ""}) == "[EQUATION]"
    assert text_for_embedding({"type": "algorithm", "text_for_embedding": ""}) == "[ALGORITHM]"
    assert text_for_embedding({"type": "reference", "text_for_embedding": "Author A. Paper."}) == "[REFERENCE]"
    assert text_for_embedding({"type": "list", "list_type": "reference_list", "text_for_embedding": "Author A. Paper."}) == "[REFERENCE]"


def test_derived_stats_masks_density_for_non_text_types_and_uses_area_sum():
    if not has_torch():
        return
    items = [
        {
            "type": "paragraph",
            "bbox": [0, 0, 10, 10, 0, 0, 20, 10],
            "text_for_embedding": "abcde",
        },
        {
            "type": "equation_interline",
            "bbox": [0, 0, 10, 10],
            "text_for_embedding": r"\\frac{a}{b}",
        },
        {
            "type": "algorithm",
            "bbox": [0, 0, 10, 10],
            "text_for_embedding": "for i in range",
        },
    ]

    stats = build_derived_stats_matrix(items)

    assert tuple(stats.shape) == (3, 3)
    assert round(float(stats[0][2]), 4) == round(5 / 300, 4)
    assert float(stats[1][2]) == 0.0
    assert float(stats[2][2]) == 0.0


def test_style_stats_matrix_summarizes_pymupdf_style_spans():
    if not has_torch():
        return
    items = [
        {
            **item("body", [0, 0, 10, 10]),
            "style_baseline_size": 10.0,
            "style_spans": [
                {"text": "bo", "font_size": 10.0, "is_bold": True, "is_italic": False, "is_inline_math": False, "is_inline_code": False, "char_count": 2},
                {"text": "dy", "font_size": 10.0, "is_bold": False, "is_italic": True, "is_inline_math": False, "is_inline_code": False, "char_count": 2},
            ],
        },
        {
            **item("title", [0, 20, 10, 30]),
            "type": "title",
            "style_baseline_size": 12.0,
            "style_spans": [
                {"text": "x", "font_size": 12.0, "is_bold": True, "is_italic": False, "is_inline_math": True, "is_inline_code": False, "char_count": 1}
            ],
        },
    ]

    assert infer_document_body_font_size(items) == 10.0
    stats = build_style_stats_matrix(items)

    assert tuple(stats.shape) == (2, 6)
    assert round(float(stats[0][0]), 4) == 0.1
    assert round(float(stats[0][1]), 4) == 0.0
    assert round(float(stats[0][2]), 4) == 0.5
    assert round(float(stats[0][3]), 4) == 0.5
    assert round(float(stats[1][1]), 4) == 0.2
    assert round(float(stats[1][4]), 4) == 1.0
