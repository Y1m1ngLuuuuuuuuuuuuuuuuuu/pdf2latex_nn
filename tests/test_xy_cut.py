from src.perception.xy_cut import (
    is_before,
    reading_order_ranks,
    rebuild_reading_order,
    sort_node_indices_by_reading_order,
    sort_nodes_by_reading_order,
)


def node(text, bbox, page=0):
    return {"text": text, "bbox": bbox, "page_idx": page}


def test_sort_nodes_by_reading_order_handles_single_double_single_column_mix():
    nodes = [
        node("right top", [520, 95, 920, 150]),
        node("title", [100, 20, 900, 60]),
        node("left tall", [80, 90, 480, 300]),
        node("conclusion", [100, 340, 900, 380]),
        node("right lower", [520, 180, 920, 250]),
    ]

    ordered = sort_nodes_by_reading_order(nodes)

    assert [item["text"] for item in ordered] == [
        "title",
        "left tall",
        "right top",
        "right lower",
        "conclusion",
    ]
    assert sort_node_indices_by_reading_order(nodes) == [1, 2, 0, 4, 3]
    assert reading_order_ranks(nodes) == [2, 0, 1, 4, 3]


def test_state_machine_reads_left_column_fully_before_right_column():
    nodes = [
        node("right top", [560, 100, 920, 160]),
        node("left top", [80, 100, 450, 160]),
        node("right bottom", [560, 200, 920, 260]),
        node("left bottom", [80, 200, 450, 260]),
    ]

    ordered = sort_nodes_by_reading_order(nodes)

    assert [item["text"] for item in ordered] == ["left top", "left bottom", "right top", "right bottom"]
    assert sort_node_indices_by_reading_order(nodes) == [1, 3, 0, 2]


def test_state_machine_flushes_double_column_block_at_full_span_boundaries():
    nodes = [
        node("title", [100, 20, 900, 60]),
        node("right top", [520, 95, 920, 150]),
        node("left tall", [80, 90, 480, 300]),
        node("right lower", [520, 180, 920, 250]),
        node("conclusion", [100, 340, 900, 380]),
    ]

    ordered = rebuild_reading_order(nodes, write_index=False)

    assert [item["text"] for item in ordered] == ["title", "left tall", "right top", "right lower", "conclusion"]


def test_is_before_uses_vertical_and_horizontal_rules_with_tolerance():
    assert is_before(node("top", [0, 0, 100, 20]), node("bottom", [0, 80, 100, 100]))
    assert not is_before(node("bottom", [0, 80, 100, 100]), node("top", [0, 0, 100, 20]))
    assert is_before(node("left", [0, 0, 100, 100]), node("right", [104, 20, 200, 90]))
    assert not is_before(node("right", [104, 20, 200, 90]), node("left", [0, 0, 100, 100]))


def test_rebuild_reading_order_can_overwrite_page_local_indices():
    nodes = [
        node("right", [520, 100, 920, 160]),
        node("left", [80, 100, 450, 160]),
    ]

    ordered = rebuild_reading_order(nodes, write_index=True)

    assert [item["text"] for item in ordered] == ["left", "right"]
    assert [item["index"] for item in ordered] == [0, 1]


def test_sort_nodes_by_reading_order_keeps_pages_before_state_machine_sort():
    nodes = [
        node("page two", [80, 10, 400, 50], page=1),
        node("page one", [80, 900, 400, 950], page=0),
    ]

    assert [item["text"] for item in sort_nodes_by_reading_order(nodes)] == ["page one", "page two"]
