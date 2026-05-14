from src.perception.reading_order import (
    annotate_duplicate_contained_continuations,
    build_content_v7,
    detect_list_marker,
    document_toc_metadata,
    extract_text,
    filter_graph_content_items,
    fix_columnar_reading_order,
    fuse_micro_nodes,
    refresh_content_v7_layout_metadata,
    sort_content_list_v2,
)


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
                    "list_items": [
                        {"item_content": [{"content": "Author A. Paper title."}]},
                        {"item_content": [{"content": "Author B. Another paper."}]},
                    ],
                },
                "bbox": [100, 120, 500, 170],
            },
        ]
    ]

    result = sort_content_list_v2(pages)
    page = result["pages"][0]

    assert [item["type"] for item in page] == ["title", "reference"]
    assert page[1]["raw_type"] == "list"
    assert page[1]["list_type"] == "reference_list"
    assert page[1]["reference_items"] == ["Author A. Paper title.", "Author B. Another paper."]
    assert page[1]["text_for_embedding"] == "Author A. Paper title. Author B. Another paper."


def test_fix_columnar_reading_order_sorts_half_span_block_left_then_right():
    nodes = [
        {"type": "title", "text_for_embedding": "Paper Title", "bbox": [100, 20, 900, 60]},
        {"type": "paragraph", "text_for_embedding": "left top", "bbox": [80, 100, 450, 150]},
        {"type": "paragraph", "text_for_embedding": "right top", "bbox": [560, 105, 920, 145]},
        {"type": "paragraph", "text_for_embedding": "left bottom", "bbox": [80, 400, 450, 430]},
        {"type": "paragraph", "text_for_embedding": "right bottom", "bbox": [560, 410, 920, 440]},
    ]

    fixed = fix_columnar_reading_order(nodes)

    assert [item["text_for_embedding"] for item in fixed] == [
        "Paper Title",
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    ]
    assert fixed[0]["column_fix_span"] == "FULL_SPAN"
    assert [item["column_fix_column"] for item in fixed[1:]] == ["LEFT_COL", "LEFT_COL", "RIGHT_COL", "RIGHT_COL"]
    assert fixed[0]["layout_band_type"] == "full_span"
    assert fixed[0]["layout_role"] == "heading"
    assert fixed[1]["layout_layer"] == "main_text_flow"
    assert fixed[1]["layout_band_column"] == "left"
    assert fixed[3]["layout_band_column"] == "right"
    assert fixed[1]["layout_band_id"] == fixed[3]["layout_band_id"]


def test_content_v7_marks_mineru_index_as_toc_metadata_and_filters_graph_items():
    pages = [
        [
            title("Contents", [100, 80, 240, 110]),
            {"type": "index", "content": "1 Introduction 3 2 Method 4", "bbox": [100, 120, 900, 500]},
            title("1 Introduction", [100, 540, 300, 570]),
            para("Intro body.", [100, 580, 900, 640]),
        ]
    ]

    result = build_content_v7(pages)
    items = result["items"]

    assert items[0]["layout_role"] == "toc_title"
    assert items[0]["layout_layer"] == "metadata_layer"
    assert items[0]["is_main_flow_candidate"] is False
    assert items[1]["type"] == "index"
    assert items[1]["layout_role"] == "toc_entry"
    assert items[1]["canonical_type"] == "toc"
    assert document_toc_metadata(items)["has_toc"] is True
    assert [item["text_for_embedding"] for item in filter_graph_content_items(items)] == [
        "1 Introduction",
        "Intro body.",
    ]


def test_refresh_content_v7_marks_span_based_run_in_heading():
    payload = {
        "items": [
            {
                "type": "paragraph",
                "raw_type": "paragraph",
                "page_idx": 0,
                "bbox": [200, 198, 795, 335],
                "text": "3.1. Implementation. We used the miniKanren family.",
                "text_for_embedding": "3.1. Implementation. We used the miniKanren family.",
                "style_spans": [
                    {"text": "3.1.", "font_size": 10.0, "is_bold": False},
                    {"text": "Implementation.", "font_size": 10.0, "is_bold": True},
                    {"text": "We used the miniKanren family.", "font_size": 10.0, "is_bold": False},
                ],
            },
            {
                "type": "paragraph",
                "raw_type": "paragraph",
                "page_idx": 0,
                "bbox": [200, 354, 795, 521],
                "text": "i. kNN graph construction using Eqs. (1) and (2).",
                "text_for_embedding": "i. kNN graph construction using Eqs. (1) and (2).",
                "style_spans": [
                    {"text": "i.", "font_size": 10.0, "is_bold": False},
                    {"text": "kNN graph construction", "font_size": 10.0, "is_bold": True},
                    {"text": "using Eqs. (1) and (2).", "font_size": 10.0, "is_bold": False},
                ],
            },
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    items = refreshed["items"]

    assert items[0]["run_in_heading"] is True
    assert items[0]["run_in_heading_number"] == "3.1"
    assert items[0]["run_in_heading_text"] == "Implementation"
    assert items[0]["run_in_heading_body"] == "We used the miniKanren family."
    assert items[0]["heading_level"] == 2
    assert items[0]["layout_role"] == "heading"
    assert "run_in_heading" not in items[1]


def test_refresh_content_v7_does_not_mark_research_sections_as_front_matter_after_body_start():
    payload = {
        "items": [
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [40, 30, 760, 80],
                "text_for_embedding": "A Paper About Code Optimization",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [300, 95, 500, 115],
                "text_for_embedding": "University of Example",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [40, 150, 380, 315],
                "text_for_embedding": "Abstract—This paper studies code optimization.",
            },
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [120, 340, 310, 365],
                "text_for_embedding": "I. INTRODUCTION",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [40, 380, 380, 520],
                "text_for_embedding": "The introduction body starts here.",
            },
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [120, 545, 360, 570],
                "text_for_embedding": "II. RESEARCH DESIGN",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [40, 590, 380, 650],
                "text_for_embedding": "This project adopts a research design canvas.",
            },
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [430, 300, 620, 325],
                "text_for_embedding": "A. Research Team",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [430, 340, 770, 440],
                "text_for_embedding": "The research team comprises the following.",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [430, 455, 770, 545],
                "text_for_embedding": "1) Core researcher: I serve as the principal investigator.",
            },
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [430, 570, 620, 595],
                "text_for_embedding": "B. Research Topic",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [430, 610, 770, 700],
                "text_for_embedding": "The research focuses on trustworthy code optimization.",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [260, 740, 540, 755],
                "text_for_embedding": "Submission to the DECS of CHASE 2025. © with the authors.",
            },
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    by_text = {item["text_for_embedding"]: item for item in refreshed["items"]}

    for text in ["II. RESEARCH DESIGN", "A. Research Team", "B. Research Topic"]:
        assert by_text[text]["layout_layer"] == "main_text_flow"
        assert by_text[text]["layout_role"] == "heading"
        assert by_text[text]["is_main_flow_candidate"] is True
        assert any(probe["reason"] == "weak_affiliation_keyword" for probe in by_text[text]["layout_probes"])

    for text in [
        "The research team comprises the following.",
        "1) Core researcher: I serve as the principal investigator.",
    ]:
        assert by_text[text]["layout_layer"] == "main_text_flow"
        assert by_text[text]["is_main_flow_candidate"] is True

    footer = by_text["Submission to the DECS of CHASE 2025. © with the authors."]
    assert footer["layout_layer"] == "noise_layer"
    assert footer["layout_role"] == "noise"
    assert footer["is_main_flow_candidate"] is False
    assert "Submission to the DECS" not in " ".join(
        item["text_for_embedding"] for item in filter_graph_content_items(refreshed["items"])
    )


def test_refresh_content_v7_keeps_pre_heading_epigraph_source_out_of_body_graph():
    payload = {
        "items": [
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [230, 100, 570, 135],
                "text_for_embedding": "Verbalized Bayesian Persuasion",
            },
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [340, 270, 460, 292],
                "text_for_embedding": "Abstract",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [145, 305, 655, 470],
                "text_for_embedding": "Information design explores how a sender influences receivers.",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [170, 540, 690, 580],
                "text_for_embedding": "You can fool some of the people all of the time, and all of the people some of the time.",
            },
            {
                "type": "title",
                "page_idx": 0,
                "bbox": [185, 660, 355, 690],
                "text_for_embedding": "1 Introduction",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [560, 610, 710, 630],
                "text_for_embedding": "Abraham Lincoln",
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [110, 710, 690, 790],
                "text_for_embedding": "Persuasion plays a significant role in modern economies.",
            },
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    by_text = {item["text_for_embedding"]: item for item in refreshed["items"]}

    source = by_text["Abraham Lincoln"]
    assert source["layout_layer"] == "metadata_layer"
    assert source["layout_role"] == "front_matter"
    assert source["is_main_flow_candidate"] is False
    assert source["front_matter_reason"] == "pre_heading_epigraph_source"


def test_refresh_content_v7_marks_repeated_header_footer_as_noise():
    payload = {
        "items": [
            {"type": "paragraph", "page_idx": 0, "bbox": [200, 20, 500, 38], "text_for_embedding": "Journal of Examples"},
            {"type": "title", "page_idx": 0, "bbox": [80, 120, 320, 145], "text_for_embedding": "1 Introduction"},
            {"type": "paragraph", "page_idx": 0, "bbox": [80, 170, 520, 260], "text_for_embedding": "First page body."},
            {"type": "paragraph", "page_idx": 0, "bbox": [300, 950, 320, 965], "text_for_embedding": "1"},
            {"type": "paragraph", "page_idx": 1, "bbox": [200, 20, 500, 38], "text_for_embedding": "Journal of Examples"},
            {"type": "paragraph", "page_idx": 1, "bbox": [80, 120, 520, 260], "text_for_embedding": "Second page body."},
            {"type": "paragraph", "page_idx": 1, "bbox": [300, 950, 320, 965], "text_for_embedding": "2"},
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    headers = [item for item in refreshed["items"] if item["text_for_embedding"] == "Journal of Examples"]
    page_numbers = [item for item in refreshed["items"] if item["text_for_embedding"] in {"1", "2"}]

    assert {item["layout_layer"] for item in headers} == {"noise_layer"}
    assert {item["layout_role"] for item in headers} == {"header"}
    assert {item["layout_role"] for item in page_numbers} == {"page_number"}
    assert "Journal of Examples" not in " ".join(item["text_for_embedding"] for item in filter_graph_content_items(refreshed["items"]))


def test_refresh_content_v7_preserves_repeated_first_page_title_and_rescues_top_author_blocks():
    payload = {
        "items": [
            {
                "type": "title",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [104, 113, 892, 155],
                "text_for_embedding": "Elastic Representation: Mitigating Spurious Correlations for Group Robustness",
                "style_baseline_size": 18.0,
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [160, 210, 430, 250],
                "text_for_embedding": "Zihan Wang New York University zw3508@nyu.edu",
                "style_baseline_size": 10.0,
            },
            {
                "type": "title",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [104, 300, 180, 322],
                "text_for_embedding": "Abstract",
                "style_baseline_size": 10.0,
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [104, 332, 470, 650],
                "text_for_embedding": "We study group robustness.",
                "style_baseline_size": 9.5,
            },
            {
                "type": "title",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [104, 771, 340, 805],
                "text_for_embedding": "1 INTRODUCTION",
                "style_baseline_size": 12.0,
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [104, 820, 470, 920],
                "text_for_embedding": "The introduction starts here.",
                "style_baseline_size": 9.5,
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [566, 214, 846, 250],
                "text_for_embedding": "Qi Lei New York University ql518@nyu.edu",
                "style_baseline_size": 10.0,
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "page_height": 1000,
                "bbox": [566, 255, 880, 292],
                "text_for_embedding": "Quan Zhang Michigan State University quan.zhang@broad.msu.edu",
                "style_baseline_size": 10.0,
            },
            {
                "type": "paragraph",
                "page_idx": 1,
                "page_height": 1000,
                "bbox": [104, 20, 892, 45],
                "text_for_embedding": "Elastic Representation: Mitigating Spurious Correlations for Group Robustness",
                "style_baseline_size": 8.0,
            },
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    title_records = [
        item
        for item in refreshed["items"]
        if item["text_for_embedding"] == "Elastic Representation: Mitigating Spurious Correlations for Group Robustness"
    ]
    by_text = {item["text_for_embedding"]: item for item in refreshed["items"]}

    first_page_title = next(item for item in title_records if item["page_idx"] == 0)
    running_header = next(item for item in title_records if item["page_idx"] == 1)
    assert first_page_title["layout_layer"] == "metadata_layer"
    assert first_page_title["layout_role"] == "document_title"
    assert running_header["layout_layer"] == "noise_layer"
    assert running_header["layout_role"] == "header"

    for text in [
        "Zihan Wang New York University zw3508@nyu.edu",
        "Qi Lei New York University ql518@nyu.edu",
        "Quan Zhang Michigan State University quan.zhang@broad.msu.edu",
    ]:
        assert by_text[text]["layout_layer"] == "metadata_layer"
        assert by_text[text]["layout_role"] == "affiliation"

    assert by_text["1 INTRODUCTION"]["layout_layer"] == "main_text_flow"
    assert by_text["1 INTRODUCTION"]["layout_role"] == "heading"


def test_refresh_content_v7_marks_bottom_marker_notes_as_annotation_layer():
    payload = {
        "items": [
            {"type": "title", "page_idx": 0, "bbox": [80, 80, 320, 105], "text_for_embedding": "1 Introduction"},
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [80, 140, 520, 260],
                "text_for_embedding": "Main body text.",
                "style_spans": [{"text": "Main body text.", "font_size": 10.0}],
            },
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [80, 900, 520, 930],
                "text_for_embedding": "1 This work was supported by a grant.",
                "style_spans": [{"text": "1 This work was supported by a grant.", "font_size": 8.0}],
            },
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    footnote = next(item for item in refreshed["items"] if item["text_for_embedding"].startswith("1 This work"))

    assert footnote["layout_layer"] == "annotation_layer"
    assert footnote["layout_role"] == "footnote"
    assert footnote["is_main_flow_candidate"] is False
    assert all(not item["text_for_embedding"].startswith("1 This work") for item in filter_graph_content_items(refreshed["items"]))


def test_refresh_content_v7_attaches_float_caption_to_nearby_figure_group():
    payload = {
        "items": [
            {"type": "title", "page_idx": 0, "bbox": [80, 60, 260, 85], "text_for_embedding": "1 Introduction"},
            {"type": "image", "page_idx": 0, "bbox": [90, 140, 390, 320], "text_for_embedding": ""},
            {
                "type": "paragraph",
                "page_idx": 0,
                "bbox": [90, 330, 390, 370],
                "text_for_embedding": "Figure 1: Example architecture.",
            },
            {"type": "paragraph", "page_idx": 0, "bbox": [90, 400, 390, 480], "text_for_embedding": "Body continues."},
        ]
    }

    refreshed = refresh_content_v7_layout_metadata(payload)
    figure = next(item for item in refreshed["items"] if item["type"] == "image")
    caption = next(item for item in refreshed["items"] if item["text_for_embedding"].startswith("Figure 1"))

    assert caption["layout_layer"] == "float_layer"
    assert caption["layout_role"] == "figure_caption"
    assert figure["figure_group_id"] == caption["figure_group_id"]
    assert figure["figure_group_caption"] == "Figure 1: Example architecture."


def test_fix_columnar_reading_order_keeps_center_crossing_short_title_as_full_span_separator():
    nodes = [
        {"type": "paragraph", "text_for_embedding": "left before", "bbox": [80, 100, 450, 150]},
        {"type": "paragraph", "text_for_embedding": "right before", "bbox": [560, 110, 920, 150]},
        {"type": "title", "text_for_embedding": "Centered", "bbox": [430, 200, 570, 220]},
        {"type": "paragraph", "text_for_embedding": "left after", "bbox": [80, 250, 450, 300]},
        {"type": "paragraph", "text_for_embedding": "right after", "bbox": [560, 260, 920, 300]},
    ]

    fixed = fix_columnar_reading_order(nodes)

    assert [item["text_for_embedding"] for item in fixed] == [
        "left before",
        "right before",
        "Centered",
        "left after",
        "right after",
    ]
    assert fixed[2]["column_fix_span"] == "FULL_SPAN"


def test_fix_columnar_reading_order_groups_same_row_figure_fragments_left_to_right():
    nodes = [
        {"type": "image", "text_for_embedding": "(b) Coarse Feature", "bbox": [321, 88, 483, 184]},
        {"type": "image", "text_for_embedding": "(a) Input, Prediction", "bbox": [163, 89, 312, 185]},
        {"type": "paragraph", "text_for_embedding": "left body", "bbox": [75, 285, 470, 465]},
        {"type": "image", "text_for_embedding": "(c) Refiner Feature", "bbox": [486, 88, 648, 184]},
        {
            "type": "image",
            "text_for_embedding": "(d) Refiner Feature Figure 3. Visualization of F2C input feature maps.",
            "bbox": [651, 88, 813, 184],
        },
        {"type": "paragraph", "text_for_embedding": "right body", "bbox": [498, 285, 892, 314]},
    ]

    fixed = fix_columnar_reading_order(nodes)

    assert [item["text_for_embedding"] for item in fixed[:4]] == [
        "(a) Input, Prediction",
        "(b) Coarse Feature",
        "(c) Refiner Feature",
        "(d) Refiner Feature Figure 3. Visualization of F2C input feature maps.",
    ]
    assert fixed[0]["layout_band_type"] == "float_group"
    assert fixed[0]["figure_group_id"] == fixed[3]["figure_group_id"]
    assert [item["figure_group_member_index"] for item in fixed[:4]] == [0, 1, 2, 3]
    assert fixed[3]["figure_group_primary"] is True
    assert fixed[3]["figure_group_caption"].startswith("Figure 3.")
    assert [item["text_for_embedding"] for item in fixed[4:]] == ["left body", "right body"]


def test_fix_columnar_reading_order_flushes_right_float_before_single_column_heading_transition():
    nodes = [
        {"type": "paragraph", "text_for_embedding": "left previous section body", "bbox": [100, 500, 430, 620]},
        {"type": "table", "text_for_embedding": "table body", "bbox": [520, 590, 850, 680]},
        {"type": "title", "text_for_embedding": "5 CONCLUSION", "bbox": [100, 735, 260, 755]},
        {"type": "paragraph", "text_for_embedding": "single column conclusion body", "bbox": [100, 775, 850, 830]},
    ]

    fixed = fix_columnar_reading_order(nodes)

    assert [item["text_for_embedding"] for item in fixed] == [
        "left previous section body",
        "table body",
        "5 CONCLUSION",
        "single column conclusion body",
    ]
    assert fixed[1]["layout_layer"] == "float_layer"
    assert fixed[2]["layout_role"] == "heading"


def test_fix_columnar_reading_order_keeps_plain_double_column_left_then_right_without_single_column_transition():
    nodes = [
        {"type": "paragraph", "text_for_embedding": "left previous section body", "bbox": [100, 500, 430, 620]},
        {"type": "table", "text_for_embedding": "right table stays in right column", "bbox": [520, 590, 850, 680]},
        {"type": "title", "text_for_embedding": "left subsection", "bbox": [100, 735, 260, 755]},
        {"type": "paragraph", "text_for_embedding": "left subsection body", "bbox": [100, 775, 430, 830]},
    ]

    fixed = fix_columnar_reading_order(nodes)

    assert [item["text_for_embedding"] for item in fixed] == [
        "left previous section body",
        "left subsection",
        "left subsection body",
        "right table stays in right column",
    ]


def test_content_v7_reorders_each_page_without_merging_or_rewriting_bbox():
    pages = [
        [
            para("right top", [560, 100, 920, 160]),
            title("Paper Title", [100, 20, 900, 60]),
            para("left top", [80, 100, 450, 160]),
            para("right bottom", [560, 200, 920, 260]),
            para("left bottom", [80, 200, 450, 260]),
        ]
    ]

    result = build_content_v7(pages)
    items = result["items"]

    assert result["schema_version"] == "content_v7_columnfix_listmarkers"
    assert [item["text_for_embedding"] for item in items] == [
        "Paper Title",
        "left top",
        "left bottom",
        "right top",
        "right bottom",
    ]
    assert items[1]["bbox"] == [80, 100, 450, 160]
    assert items[1]["layout_band_global_id"] == items[1]["layout_band_id"]
    assert items[1]["layout_flow_order"] == 1
    assert all("merge_count" not in item for item in items)


def test_content_v7_marks_list_items_without_merging_or_rewriting_bbox():
    pages = [
        [
            para("Body before list.", [80, 100, 450, 140]),
            para("1. First point", [80, 150, 450, 180]),
            para("plain paragraph", [80, 190, 450, 220]),
        ]
    ]

    result = build_content_v7(pages)
    items = result["items"]

    assert result["schema_version"] == "content_v7_columnfix_listmarkers"
    assert items[1]["list_marker"] == {"type": "arabic", "marker": "1."}
    assert items[1]["list_item_id"] == "li_00000"
    assert items[1]["bbox"] == [80, 150, 450, 180]
    assert items[2]["list_item_id"] is None
    assert all("merge_count" not in item for item in items)


def test_duplicate_contained_continuation_is_marked_and_filtered_from_graph_items():
    items = [
        {
            "type": "paragraph",
            "page_idx": 0,
            "global_order": 0,
            "text_for_embedding": "All experiments used an NVIDIA RTX 8000 GPU with 48GB memory.",
            "bbox": [100, 100, 450, 180],
            "layout_layer": "main_text_flow",
        },
        {
            "type": "table",
            "page_idx": 0,
            "global_order": 1,
            "text_for_embedding": "Table 1: Results.",
            "bbox": [100, 190, 900, 360],
            "layout_layer": "float_layer",
        },
        {
            "type": "paragraph",
            "page_idx": 1,
            "global_order": 2,
            "text_for_embedding": "with 48GB memory.",
            "bbox": [100, 80, 450, 120],
            "layout_layer": "main_text_flow",
        },
    ]

    annotated = annotate_duplicate_contained_continuations(items)

    assert annotated[2]["duplicate_shadow"] is True
    assert annotated[2]["no_render"] is True
    assert [item["text_for_embedding"] for item in filter_graph_content_items(annotated)] == [
        "All experiments used an NVIDIA RTX 8000 GPU with 48GB memory.",
        "Table 1: Results.",
    ]


def test_two_column_page_uses_state_machine_left_column_then_right_column():
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


def test_region_between_full_width_blocks_uses_left_then_right_column_state():
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


def test_detect_list_marker_variants():
    assert detect_list_marker("1. first")["type"] == "arabic"
    assert detect_list_marker("a) first")["type"] == "alpha"
    assert detect_list_marker("iv. first")["type"] == "roman"
    assert detect_list_marker("• first")["type"] == "bullet"
    assert detect_list_marker("（一） first")["type"] == "paren_cjk"
    assert detect_list_marker("一、first")["type"] == "cjk_comma"
    assert detect_list_marker("plain paragraph") is None


def test_fuse_micro_nodes_preserves_inline_math_order_on_same_line():
    nodes = [
        visual_item("and", 0, 2, [206, 100, 232, 114], 0, item_type="text"),
        visual_item("The value is", 0, 0, [100, 100, 180, 114], 0, item_type="text"),
        visual_item("x", 0, 1, [184, 99, 196, 115], 0, item_type="inline_math"),
        visual_item("y", 0, 3, [238, 99, 250, 115], 0, item_type="inline_math"),
        visual_item(".", 0, 4, [251, 100, 254, 114], 0, item_type="text"),
    ]

    fused = fuse_micro_nodes(nodes)

    assert len(fused) == 1
    assert fused[0]["type"] == "text"
    assert fused[0]["micro_fused"] is True
    assert fused[0]["text_for_embedding"] == "The value is $x$ and $y$."
    assert fused[0]["bbox"] == [100.0, 99.0, 254.0, 115.0]
    assert fused[0]["source_node_indexes"] == [1, 2, 0, 3, 4]


def test_fuse_micro_nodes_does_not_cross_column_gutter_on_same_y_line():
    nodes = [
        visual_item("left line", 0, 0, [80, 100, 180, 115], 0, item_type="text"),
        visual_item("right line", 0, 1, [560, 100, 660, 115], 1, item_type="text"),
    ]

    fused = fuse_micro_nodes(nodes)

    assert [item["text_for_embedding"] for item in fused] == ["left line", "right line"]
    assert all(not item.get("micro_fused") for item in fused)


def test_fuse_micro_nodes_keeps_structural_nodes_out_of_line_fusion():
    nodes = [
        visual_item("Introduction", 0, 0, [80, 100, 180, 120], 0, item_type="title"),
        visual_item("body", 0, 1, [190, 102, 230, 116], 0, item_type="text"),
        visual_item("z", 0, 2, [234, 101, 244, 117], 0, item_type="inline_math"),
        visual_item("table body", 0, 3, [248, 100, 330, 120], 0, item_type="table"),
    ]

    fused = fuse_micro_nodes(nodes)

    assert [item["type"] for item in fused] == ["title", "text", "table"]
    assert fused[1]["text_for_embedding"] == "body $z$"
