from src.perception.reading_order import build_content_v3, build_content_v4, detect_list_marker, extract_text, sort_content_list_v2


def para(text, bbox):
    return {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": text}]}, "bbox": bbox}


def title(text, bbox):
    return {"type": "title", "content": {"title_content": [{"type": "text", "content": text}], "level": 1}, "bbox": bbox}


def test_extract_text_from_v2_nested_content():
    block = title("Key Observation", [100, 100, 900, 120])

    assert extract_text(block) == "Key Observation"


def test_extract_text_from_reference_list_item_content():
    block = {
        "type": "list",
        "content": {
            "list_type": "reference_list",
            "list_items": [
                {
                    "item_type": "text",
                    "item_content": [
                        {"type": "text", "content": "Author A. Paper title."},
                        {"type": "text", "content": "Journal 2024."},
                    ],
                }
            ],
        },
        "bbox": [100, 100, 500, 160],
    }

    assert extract_text(block) == "Author A. Paper title. Journal 2024."


def test_reference_list_is_preserved_as_reference_type_in_visual_order():
    pages = [
        [
            title("References", [100, 80, 250, 110]),
            {
                "type": "list",
                "content": {
                    "list_type": "reference_list",
                    "list_items": [{"item_content": [{"content": "Author A. Paper title."}]}],
                },
                "bbox": [100, 120, 500, 170],
            },
        ]
    ]

    result = sort_content_list_v2(pages)
    page = result["pages"][0]

    assert [item["type"] for item in page] == ["title", "reference"]
    assert page[1]["raw_type"] == "list"
    assert page[1]["text_for_embedding"] == "Author A. Paper title."


def test_two_column_page_is_sorted_left_column_then_right_column():
    pages = [
        [
            para("right top", [560, 100, 920, 160]),
            para("left top", [80, 100, 450, 160]),
            para("right bottom", [560, 200, 920, 260]),
            para("left bottom", [80, 200, 450, 260]),
        ]
    ]

    result = sort_content_list_v2(pages)
    texts = [item["text_for_embedding"] for item in result["pages"][0]]

    assert texts == ["left top", "left bottom", "right top", "right bottom"]
    assert [item["column_id"] for item in result["pages"][0]] == [0, 0, 1, 1]


def test_full_width_title_stays_before_two_columns_and_auxiliary_is_dropped():
    pages = [
        [
            para("right", [560, 150, 920, 190]),
            {"type": "page_number", "content": {"paragraph_content": [{"content": "1"}]}, "bbox": [490, 950, 510, 970]},
            title("Paper Title", [120, 40, 880, 90]),
            para("left", [80, 150, 450, 190]),
        ]
    ]

    result = sort_content_list_v2(pages)
    page = result["pages"][0]

    assert [item["text_for_embedding"] for item in page] == ["Paper Title", "left", "right"]
    assert page[0]["is_full_width"] is True
    assert result["page_summaries"][0]["dropped_auxiliary_blocks"] == 1


def test_region_between_full_width_blocks_sorts_left_column_before_right_column():
    pages = [
        [
            para("", [80, 534, 482, 579]),
            title("Key Observations", [83, 590, 227, 606]),
            para("right continuation", [516, 535, 870, 550]),
            para("left section body", [80, 611, 482, 890]),
            para("right section body", [514, 558, 916, 811]),
        ]
    ]

    result = sort_content_list_v2(pages)
    page = result["pages"][0]

    assert [item["text_for_embedding"] for item in page] == [
        "Key Observations",
        "left section body",
        "right continuation",
        "right section body",
    ]
    assert result["page_summaries"][0]["dropped_empty_textual_blocks"] == 1


def visual_item(text, page, order, bbox, column, item_type="paragraph"):
    return {
        "page_idx": page,
        "visual_order": order,
        "original_index": order,
        "type": item_type,
        "bbox": bbox,
        "column_id": column,
        "is_full_width": False,
        "is_textual": True,
        "text_for_embedding": text,
        "merge_count": 1,
        "source_page_idxs": [page],
        "source_visual_orders": [order],
        "source_original_indexes": [order],
        "block": {"type": item_type},
    }


def test_content_v3_merges_same_page_column_continuation_without_terminal_punctuation():
    payload = {
        "pages": [
            [
                visual_item("a critical requirement", 5, 0, [80, 611, 482, 890], 0),
                visual_item("for intrusion detection systems.", 5, 1, [514, 535, 916, 550], 1),
            ]
        ]
    }

    result = build_content_v3(payload)
    item = result["items"][0]

    assert len(result["items"]) == 1
    assert item["text_for_embedding"] == "a critical requirement for intrusion detection systems."
    assert item["bbox"] == [80, 611, 482, 890, 514, 535, 916, 550]
    assert item["merge_count"] == 2


def test_content_v3_does_not_merge_when_previous_has_terminal_punctuation():
    payload = {
        "pages": [
            [
                visual_item("This sentence ends.", 0, 0, [80, 611, 482, 890], 0),
                visual_item("New sentence.", 0, 1, [514, 535, 916, 550], 1),
            ]
        ]
    }

    result = build_content_v3(payload)

    assert len(result["items"]) == 2


def test_content_v3_merges_cross_page_continuation():
    payload = {
        "pages": [
            [visual_item("continues across", 0, 0, [80, 820, 482, 930], 0)],
            [visual_item("the next page.", 1, 0, [80, 90, 482, 160], 0)],
        ]
    }

    result = build_content_v3(payload)

    assert len(result["items"]) == 1
    assert result["items"][0]["source_page_idxs"] == [0, 1]


def test_detect_list_marker_variants():
    assert detect_list_marker("1. first")["type"] == "arabic"
    assert detect_list_marker("a) first")["type"] == "alpha"
    assert detect_list_marker("iv. first")["type"] == "roman"
    assert detect_list_marker("• first")["type"] == "bullet"
    assert detect_list_marker("（一） first")["type"] == "paren_cjk"
    assert detect_list_marker("一、first")["type"] == "cjk_comma"
    assert detect_list_marker("plain paragraph") is None


def test_content_v4_list_marker_stays_independent_without_continuation_merge():
    payload = {
        "items": [
            visual_item("1. First item starts", 0, 0, [100, 100, 450, 140], 0),
            visual_item("continues without marker", 0, 1, [102, 145, 450, 180], 0),
            visual_item("2. Second item starts", 0, 2, [100, 190, 450, 230], 0),
        ]
    }

    result = build_content_v4(payload)
    items = result["items"]

    assert len(items) == 3
    assert items[0]["list_item_id"] == "li_00000"
    assert "list_continuation_count" not in items[0]
    assert items[0]["text_for_embedding"] == "1. First item starts"
    assert items[1]["list_item_id"] is None
    assert items[1]["text_for_embedding"] == "continues without marker"
    assert items[2]["list_item_id"] == "li_00001"


def test_content_v4_parent_colon_sets_list_parent_without_merging_parent():
    payload = {
        "items": [
            visual_item("The contributions are:", 0, 0, [80, 100, 450, 140], 0),
            visual_item("1. First contribution", 0, 1, [120, 150, 450, 180], 0),
        ]
    }

    result = build_content_v4(payload)
    items = result["items"]

    assert len(items) == 2
    assert items[1]["list_parent_global_order"] == 0
    assert items[1]["list_level"] == 1


def test_content_v4_merges_hyphenated_paragraph_across_float_block():
    payload = {
        "items": [
            visual_item("For CI-", 5, 0, [514, 818, 916, 890], 1),
            visual_item("Algorithm 1: floating block", 6, 1, [81, 66, 478, 481], 0, item_type="algorithm"),
            visual_item("CEVSE2024 continues here.", 6, 2, [81, 507, 482, 704], 0),
        ]
    }

    result = build_content_v4(payload)
    items = result["items"]

    assert len(items) == 2
    assert items[0]["text_for_embedding"] == "For CICEVSE2024 continues here."
    assert items[0]["bbox"] == [514, 818, 916, 890, 81, 507, 482, 704]
    assert items[0]["source_page_idxs"] == [5, 6]
    assert items[0]["skipped_float_types"] == ["algorithm"]
    assert items[1]["type"] == "algorithm"
