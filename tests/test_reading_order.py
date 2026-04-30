from src.perception.reading_order import build_content_v3, extract_text, sort_content_list_v2


def para(text, bbox):
    return {"type": "paragraph", "content": {"paragraph_content": [{"type": "text", "content": text}]}, "bbox": bbox}


def title(text, bbox):
    return {"type": "title", "content": {"title_content": [{"type": "text", "content": text}], "level": 1}, "bbox": bbox}


def test_extract_text_from_v2_nested_content():
    block = title("Key Observation", [100, 100, 900, 120])

    assert extract_text(block) == "Key Observation"


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


def visual_item(text, page, order, bbox, column):
    return {
        "page_idx": page,
        "visual_order": order,
        "original_index": order,
        "type": "paragraph",
        "bbox": bbox,
        "column_id": column,
        "is_full_width": False,
        "is_textual": True,
        "text_for_embedding": text,
        "block": {"type": "paragraph"},
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
