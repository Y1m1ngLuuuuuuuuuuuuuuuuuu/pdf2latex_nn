from src.reasoning.graph_builder import (
    TYPE_VOCAB,
    build_derived_stats_matrix,
    build_geometry_matrix,
    build_sequential_edge_index,
    build_type_onehot_matrix,
    canonical_type,
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


def test_sequential_edges_are_bidirectional_by_default():
    if not has_torch():
        return
    edge_index = build_sequential_edge_index(3)

    assert edge_index.tolist() == [[0, 1, 1, 2], [1, 0, 2, 1]]


def test_canonical_type_maps_mineru_names_to_fixed_vocab():
    assert TYPE_VOCAB == ["text", "title", "equation", "table", "figure", "algorithm", "list", "code", "reference", "other"]
    assert canonical_type("paragraph") == "text"
    assert canonical_type("equation_interline") == "equation"
    assert canonical_type("chart") == "figure"
    assert canonical_type("algorithm") == "algorithm"
    assert canonical_type("code") == "code"
    assert canonical_type("reference") == "reference"


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
        {"type": "reference"},
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
