from src.generation.latex_renderer import RenderConfig, render_latex_document
from src.generation.latex_renderer import render_equation as render_generation_equation
from src.reasoning.postprocess import (
    MERGE,
    PARENT_CHILD,
    DecodedEdge,
    ResolvedNode,
    TreeDecoder,
    TreeDecoderConfig,
    build_resolved_tree,
    decode_relations_with_arborescence,
    escape_latex,
    render_equation,
    safe_verbatim_text,
)


def has_torch():
    try:
        import torch  # noqa: F401
    except ModuleNotFoundError:
        return False
    return True


def test_arborescence_decode_relations_drops_parent_cycles():
    if not has_torch():
        return
    import torch

    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.01, 0.96, 0.01, 0.02],
            [0.01, 0.95, 0.01, 0.03],
            [0.01, 0.94, 0.01, 0.04],
        ]
    )

    decoded = decode_relations_with_arborescence(edge_index, scores, threshold=0.5, num_nodes=3)

    assert [(edge.source, edge.target, edge.label) for edge in decoded] == [
        (0, 1, PARENT_CHILD),
        (1, 2, PARENT_CHILD),
    ]


def test_tree_decoder_contracts_merge_nodes_and_repoints_parent_edges():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text_for_embedding": "Introduction"},
        {"type": "paragraph", "text_for_embedding": "Cyber-"},
        {"type": "paragraph", "text_for_embedding": "security, matters."},
    ]
    edge_index = torch.tensor([[1, 0], [2, 2]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.96, 0.01, 0.01, 0.02],
            [0.01, 0.93, 0.01, 0.05],
        ],
        dtype=torch.float32,
    )

    decoder = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5))
    root = decoder.decode(records, edge_index, scores)

    assert len(root.children) == 1
    title = root.children[0]
    assert title.text == "Introduction"
    assert len(title.children) == 1
    assert title.children[0].merged_node_ids == [1, 2]
    assert title.children[0].text == "Cybersecurity, matters."


def test_tree_decoder_refuses_cross_type_merge_contraction():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text_for_embedding": "Introduction"},
        {"type": "paragraph", "text_for_embedding": "Body text."},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.99, 0.0, 0.0, 0.01]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    assert [child.text for child in root.children] == ["Introduction"]
    assert [child.text for child in root.children[0].children] == ["Body text."]
    assert root.children[0].merged_node_ids == [0]
    assert root.children[0].children[0].merged_node_ids == [1]


def test_tree_decoder_rejects_equation_as_parent_and_preserves_following_text():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "equation_interline", "text": "d_{Ch}(x,c_k)"},
        {"type": "paragraph", "text": "Metric space integration: Each distance metric undergoes z-score normalization."},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.0, 0.99, 0.01]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.5, require_parent_argmax=True)).decode(records, edge_index, scores)
    tex = TreeDecoder().render_document(root)

    assert [child.merged_node_ids for child in root.children] == [[0], [1]]
    assert "Metric space integration" in tex


def test_tree_decoder_renders_reference_items_from_merged_records():
    records = [
        {"type": "reference", "reference_items": ["A. First.", "B. Second."]},
        {"type": "reference", "reference_items": ["C. Third."]},
    ]
    root = build_resolved_tree(records, [DecodedEdge(source=0, target=1, label=MERGE, score=0.99)])

    tex = TreeDecoder().render_document(root)

    assert tex.count(r"\bibitem") == 3
    assert "A. First." in tex
    assert "B. Second." in tex
    assert "C. Third." in tex


def test_tree_decoder_groups_reference_list_children_as_one_bibliography():
    records = [
        {"type": "title", "text": "References"},
        {"type": "list", "list_type": "reference_list", "reference_items": ["A. First.", "B. Second."]},
        {"type": "list", "list_type": "reference_list", "reference_items": ["C. Third."]},
    ]
    root = build_resolved_tree(
        records,
        [
            DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.99),
            DecodedEdge(source=0, target=2, label=PARENT_CHILD, score=0.99),
        ],
    )

    tex = TreeDecoder().render_document(root)

    assert tex.count(r"\begin{thebibliography}{99}") == 1
    assert tex.count(r"\end{thebibliography}") == 1
    assert tex.count(r"\bibitem") == 3
    assert r"\begin{itemize}" not in tex


def test_tree_decoder_refuses_parallel_cross_column_merge_contraction():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "paragraph", "text_for_embedding": "Left column bottom.", "bbox": [80, 700, 480, 760]},
        {"type": "paragraph", "text_for_embedding": "Right column bottom.", "bbox": [520, 705, 920, 765]},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.99, 0.0, 0.0, 0.01]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    assert [child.text for child in root.children] == ["Left column bottom.", "Right column bottom."]
    assert [child.merged_node_ids for child in root.children] == [[0], [1]]


def test_tree_decoder_gutter_barrier_does_not_block_cross_page_merge():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "paragraph", "text_for_embedding": "Cross page", "bbox": [80, 700, 480, 760], "page_idx": 0},
        {"type": "paragraph", "text_for_embedding": "continuation.", "bbox": [520, 705, 920, 765], "page_idx": 1},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.99, 0.0, 0.0, 0.01]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    assert len(root.children) == 1
    assert root.children[0].merged_node_ids == [0, 1]


def test_tree_decoder_msa_keeps_forest_roots_under_virtual_root():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "A"},
        {"type": "paragraph", "text": "Child"},
        {"type": "title", "text": "B"},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.01, 0.91, 0.01, 0.07]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.5)).decode(records, edge_index, scores)

    assert [child.node_id for child in root.children] == [0, 2]
    assert [child.node_id for child in root.children[0].children] == [1]


def test_tree_decoder_forces_abstract_title_to_virtual_root():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "Paper Title"},
        {"type": "title", "text": "Abstract"},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.01, 0.99, 0.0, 0.0]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.0)).decode(records, edge_index, scores)

    assert [child.text for child in root.children] == ["Paper Title", "Abstract"]
    assert root.children[0].children == []


def test_tree_decoder_suppresses_text_to_text_parent_edges():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "Introduction"},
        {"type": "paragraph", "text": "First body paragraph."},
        {"type": "paragraph", "text": "Second body paragraph."},
    ]
    edge_index = torch.tensor([[1, 0], [2, 2]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.01, 0.95, 0.01, 0.03],
            [0.01, 0.10, 0.01, 0.88],
        ],
        dtype=torch.float32,
    )

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.0)).decode(records, edge_index, scores)

    title = root.children[0]
    assert title.text == "Introduction"
    assert [child.text for child in title.children] == ["First body paragraph.", "Second body paragraph."]


def test_tree_decoder_deduplicates_semantic_ghost_titles_and_reroutes_edges():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "References."},
        {"type": "title", "text": "REFERENCES"},
        {"type": "paragraph", "text": "A cited work."},
    ]
    edge_index = torch.tensor([[1, 1], [0, 2]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.01, 0.99, 0.0, 0.0],
            [0.01, 0.98, 0.0, 0.01],
        ],
        dtype=torch.float32,
    )

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.0)).decode(records, edge_index, scores)
    tex = TreeDecoder().render_document(root)

    assert [child.text for child in root.children] == ["References."]
    assert [child.text for child in root.children[0].children] == ["A cited work."]
    assert tex.count(r"\section*{References.}") == 1
    assert "REFERENCES" not in tex


def test_tree_decoder_enforces_reference_tail_topology_until_appendix():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "Introduction"},
        {"type": "paragraph", "text": "Body paragraph."},
        {"type": "title", "text": "References"},
        {"type": "paragraph", "text": "[1] First cited work."},
        {"type": "paragraph", "text": "[2] Second cited work."},
        {"type": "title", "text": "Appendix A"},
        {"type": "paragraph", "text": "Appendix body."},
    ]
    edge_index = torch.tensor([[0, 5, 3], [1, 6, 4]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.01, 0.91, 0.01, 0.07],
            [0.01, 0.92, 0.01, 0.06],
            [0.01, 0.99, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.5)).decode(records, edge_index, scores)

    references = next(child for child in root.children if child.text == "References")
    appendix = next(child for child in root.children if child.text == "Appendix A")
    assert [child.text for child in references.children] == [
        "[1] First cited work.",
        "[2] Second cited work.",
    ]
    assert [child.text for child in appendix.children] == ["Appendix body."]


def test_tree_decoder_reference_topology_skips_page_noise_nodes():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "References"},
        {"type": "page_number", "text": "14"},
        {"type": "paragraph", "text": "[1] First cited work."},
    ]
    edge_index = torch.empty((2, 0), dtype=torch.long)
    scores = torch.empty((0, 4), dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.5)).decode(records, edge_index, scores)

    assert [child.text for child in root.children] == ["References", "14"]
    assert [child.text for child in root.children[0].children] == ["[1] First cited work."]


def test_tree_decoder_skeleton_keeps_headings_in_physical_order_over_gnn_parent_edges():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "Introduction", "style_baseline_size": 12, "regime_reading_order": 0},
        {"type": "paragraph", "text": "Intro body.", "style_baseline_size": 10, "regime_reading_order": 1},
        {"type": "title", "text": "Related Work", "style_baseline_size": 12, "regime_reading_order": 2},
        {"type": "title", "text": "Metric-Based Learning", "style_baseline_size": 11, "regime_reading_order": 4},
        {"type": "paragraph", "text": "Metric body.", "style_baseline_size": 10, "regime_reading_order": 5},
    ]
    edge_index = torch.tensor([[3, 3], [2, 1]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.01, 0.99, 0.0, 0.0],
            [0.01, 0.98, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.0)).decode(records, edge_index, scores)

    assert [child.text for child in root.children] == ["Introduction", "Related Work"]
    assert [child.text for child in root.children[0].children] == ["Intro body."]
    assert [child.text for child in root.children[1].children] == ["Metric-Based Learning"]
    assert [child.text for child in root.children[1].children[0].children] == ["Metric body."]


def test_tree_decoder_rejects_merge_across_section_scopes():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "A", "regime_reading_order": 0},
        {"type": "paragraph", "text": "End of A-", "regime_reading_order": 1},
        {"type": "title", "text": "B", "regime_reading_order": 2},
        {"type": "paragraph", "text": "start of B.", "regime_reading_order": 3},
    ]
    edge_index = torch.tensor([[1], [3]], dtype=torch.long)
    scores = torch.tensor([[0.99, 0.0, 0.0, 0.01]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    assert [child.text for child in root.children] == ["A", "B"]
    assert [child.text for child in root.children[0].children] == ["End of A-"]
    assert [child.text for child in root.children[1].children] == ["start of B."]


def test_tree_decoder_dfs_renderer_escapes_text_but_not_equations():
    records = [
        {"type": "title", "text": "Intro_50%"},
        {"type": "paragraph", "text": "A&B_#1"},
        {"type": "equation", "text": r"E = mc^2"},
    ]
    decoded = [
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=0, target=2, label=PARENT_CHILD, score=0.8),
    ]
    root = build_resolved_tree(records, decoded)
    tex = TreeDecoder().render_document(root, title="Demo_100%")

    assert r"\title{Demo\_100\%}" in tex
    assert r"\section{Intro\_50\%}" in tex
    assert r"A\&B\_\#1" in tex
    assert "\\[\nE = mc^2\n\\]" in tex
    assert "mc\\textasciicircum{}2" not in tex


def test_tree_decoder_lifts_matching_first_title_into_document_title():
    records = [
        {"type": "title", "text": "Paper_50%"},
        {"type": "paragraph", "text": "Author A"},
        {"type": "title", "text": "Abstract"},
    ]
    decoded = [DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9)]
    root = build_resolved_tree(records, decoded)
    tex = TreeDecoder().render_document(root, title="Paper_50%")

    assert r"\title{Paper\_50\%}" in tex
    assert r"\section{Paper\_50\%}" not in tex
    assert "Author A" in tex
    assert r"\section*{Abstract}" in tex


def test_tree_decoder_adaptively_renders_numbered_title_levels():
    records = [
        {"type": "title", "text": "1. Introduction"},
        {"type": "title", "text": "1.1 Background"},
        {"type": "title", "text": "1.1.1 Details"},
        {"type": "title", "text": "II. Related Work"},
    ]
    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\section{Introduction}" in tex
    assert r"\subsection{Background}" in tex
    assert r"\subsubsection{Details}" in tex
    assert r"\section{Related Work}" in tex
    assert "1. Introduction" not in tex
    assert "1.1 Background" not in tex
    assert "1.1.1 Details" not in tex
    assert "II. Related Work" not in tex


def test_tree_decoder_renders_abstract_unnumbered_and_bare_numeric_intro_as_section():
    records = [
        {"type": "title", "text": "Paper Title", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 0, "style_baseline_size": 20.0},
        {"type": "paragraph", "text": "Author A", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 1, "style_baseline_size": 12.0},
        {"type": "title", "text": "Abstract", "layout_role": "abstract", "layout_layer": "metadata_layer", "global_order": 2, "style_baseline_size": 11.0},
        {"type": "paragraph", "text": "Abstract body.", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 3, "style_baseline_size": 11.0},
        {"type": "title", "text": "1 Introduction", "layout_role": "heading", "layout_layer": "main_text_flow", "global_order": 4, "style_baseline_size": 17.0, "layout_band_type": "double_column", "layout_band_column": "left"},
        {"type": "paragraph", "text": "Intro body.", "layout_role": "body_text", "layout_layer": "main_text_flow", "global_order": 5, "style_baseline_size": 12.0},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []), title="Paper Title")

    assert r"\section*{Abstract}" in tex
    assert r"\section{Introduction}" in tex
    assert r"\subsection*{Introduction}" not in tex
    assert tex.index(r"\section*{Abstract}") < tex.index("Abstract body.")
    assert tex.index("Abstract body.") < tex.index(r"\section{Introduction}")


def test_tree_decoder_does_not_promote_frontmatter_date_to_section():
    records = [
        {"type": "title", "text": "Paper Title", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 0, "style_baseline_size": 20.0},
        {"type": "title", "text": "25 March 2025", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 1, "style_baseline_size": 12.0},
        {"type": "title", "text": "Abstract", "layout_role": "abstract", "layout_layer": "metadata_layer", "global_order": 2, "style_baseline_size": 11.0},
        {"type": "paragraph", "text": "Abstract body.", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 3, "style_baseline_size": 11.0},
        {"type": "title", "text": "1 Introduction", "layout_role": "heading", "layout_layer": "main_text_flow", "global_order": 4, "style_baseline_size": 17.0},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []), title="Paper Title")

    assert "25 March 2025" in tex
    assert r"\section{March 2025}" not in tex
    assert r"\section{25 March 2025}" not in tex
    assert tex.index("25 March 2025") < tex.index(r"\section*{Abstract}")
    assert r"\section{Introduction}" in tex


def test_tree_decoder_renders_toc_title_as_latex_tableofcontents_and_skips_ocr_entries():
    records = [
        {"type": "title", "text": "Paper Title", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 0},
        {"type": "title", "text": "Contents", "layout_role": "toc_title", "layout_layer": "metadata_layer", "global_order": 1},
        {"type": "index", "text": "1 Introduction 3 2 Method 4", "layout_role": "toc_entry", "layout_layer": "metadata_layer", "global_order": 2},
        {"type": "title", "text": "1 Introduction", "layout_role": "heading", "layout_layer": "main_text_flow", "global_order": 3},
        {"type": "paragraph", "text": "Intro body.", "layout_role": "body_text", "layout_layer": "main_text_flow", "global_order": 4},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []), title="Paper Title")

    assert r"\tableofcontents" in tex
    assert r"\section{Contents}" not in tex
    assert "1 Introduction 3 2 Method 4" not in tex
    assert tex.index(r"\tableofcontents") < tex.index(r"\section{Introduction}")


def test_tree_decoder_inserts_toc_from_graph_document_metadata_after_filtering_toc_nodes():
    records = [
        {"type": "title", "text": "Paper Title", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 0},
        {"type": "title", "text": "1 Introduction", "layout_role": "heading", "layout_layer": "main_text_flow", "global_order": 3},
        {"type": "paragraph", "text": "Intro body.", "layout_role": "body_text", "layout_layer": "main_text_flow", "global_order": 4},
    ]

    tex = TreeDecoder().render_document(
        build_resolved_tree(records, []),
        title="Paper Title",
        document_metadata={"has_toc": True, "toc_order": 1, "toc_page_idx": 1},
    )

    assert r"\tableofcontents" in tex
    assert tex.index(r"\tableofcontents") < tex.index(r"\section{Introduction}")


def test_generation_renderer_does_not_strip_frontmatter_date():
    root = ResolvedNode(
        node_id=-1,
        record={},
        children=[
            ResolvedNode(
                node_id=0,
                record={"type": "title", "text": "25 March 2025", "layout_role": "front_matter", "layout_layer": "metadata_layer"},
                merged_node_ids=[0],
            ),
            ResolvedNode(
                node_id=1,
                record={"type": "title", "text": "1 Introduction", "layout_role": "heading", "layout_layer": "main_text_flow"},
                merged_node_ids=[1],
            ),
        ],
    )

    tex = render_latex_document(root, RenderConfig())

    assert "25 March 2025" in tex
    assert r"\section{March 2025}" not in tex
    assert r"\section{Introduction}" in tex


def test_tree_decoder_treats_numbered_title_with_list_item_role_as_heading():
    records = [
        {"type": "title", "text": "Abstract", "layout_role": "abstract", "layout_layer": "metadata_layer", "global_order": 0, "style_baseline_size": 12.0},
        {"type": "paragraph", "text": "Abstract body.", "layout_role": "front_matter", "layout_layer": "metadata_layer", "global_order": 1, "style_baseline_size": 10.0},
        {"type": "title", "text": "1. Introduction", "layout_role": "list_item", "layout_layer": "main_text_flow", "global_order": 2, "style_baseline_size": 12.0},
        {"type": "paragraph", "text": "Intro body.", "layout_role": "body_text", "layout_layer": "main_text_flow", "global_order": 3, "style_baseline_size": 10.0},
        {"type": "title", "text": "2. Related Work", "layout_role": "list_item", "layout_layer": "main_text_flow", "global_order": 4, "style_baseline_size": 12.0},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\section{Introduction}" in tex
    assert r"\section{Related Work}" in tex
    assert r"\subsection{Introduction}" not in tex
    assert r"\begin{enumerate}" not in tex


def test_tree_decoder_keeps_ieee_alpha_headings_under_roman_section():
    records = [
        {"type": "title", "text": "III. METHODOLOGY", "layout_role": "list_item", "layout_layer": "main_text_flow", "global_order": 0, "style_baseline_size": 12.0},
        {"type": "title", "text": "A. Problem Definition", "layout_role": "list_item", "layout_layer": "main_text_flow", "global_order": 1, "style_baseline_size": 12.0},
        {"type": "paragraph", "text": "Problem body.", "layout_role": "body_text", "layout_layer": "main_text_flow", "global_order": 2, "style_baseline_size": 10.0},
        {"type": "title", "text": "C. Model Architecture", "layout_role": "list_item", "layout_layer": "main_text_flow", "global_order": 3, "style_baseline_size": 12.0},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\section{METHODOLOGY}" in tex
    assert r"\subsection{Problem Definition}" in tex
    assert r"\subsection{Model Architecture}" in tex
    assert r"\section{Model Architecture}" not in tex


def test_tree_decoder_uses_tree_depth_for_unnumbered_title_levels():
    records = [
        {"type": "title", "text": "Methods"},
        {"type": "title", "text": "Model"},
        {"type": "title", "text": "Loss"},
    ]
    decoded = [
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=1, target=2, label=PARENT_CHILD, score=0.9),
    ]
    tex = TreeDecoder().render_document(build_resolved_tree(records, decoded))

    assert r"\section{Methods}" in tex
    assert r"\subsection{Model}" in tex
    assert r"\subsubsection{Loss}" in tex


def test_tree_decoder_renders_inline_math_without_escaping():
    records = [
        {"type": "title", "text": "Math"},
        {"type": "inline_math", "text": r"x_i + y_j"},
    ]
    decoded = [DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9)]
    tex = TreeDecoder().render_document(build_resolved_tree(records, decoded))

    assert r"$x_i + y_j$" in tex
    assert r"x\_i" not in tex


def test_tree_decoder_preserves_inline_formula_segments_in_paragraph_blocks():
    records = [
        {
            "type": "paragraph",
            "text": r"Given x _ { i } and y.",
            "block": {
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Given "},
                        {"type": "equation_inline", "content": r"x _ { i }"},
                        {"type": "text", "content": " and y_1."},
                    ]
                }
            },
        }
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"Given $x _ { i }$ and y\_1." in tex
    assert r"\textbackslash{}" not in tex


def test_tree_decoder_preserves_bare_inline_latex_math_inside_caption_text():
    records = [
        {
            "type": "figure",
            "text": r'''Fig. 3. Access order, ``j'' reactions and " \mathrm { p } ^ { \mathrm { , } \mathrm { , } } the number.''',
        }
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r'``j' in tex
    assert r"$\mathrm { p } ^ { \mathrm { , } \mathrm { , } }$ the number" in tex
    assert r"\textbackslash{}mathrm" not in tex


def test_generation_renderer_preserves_bare_inline_latex_math_inside_caption_text():
    root = type("Root", (), {})()
    root.children = [
        {
            "type": "figure",
            "text": r'''Fig. 3. Access order, ``j'' reactions and " \mathrm { p } ^ { \mathrm { , } \mathrm { , } } the number.''',
        }
    ]

    tex = render_latex_document(root)

    assert r"$\mathrm { p } ^ { \mathrm { , } \mathrm { , } }$ the number" in tex
    assert r"\textbackslash{}mathrm" not in tex


def test_tree_decoder_renders_list_children_as_items_with_raw_equations():
    records = [
        {"type": "list", "text": "Contributions"},
        {"type": "paragraph", "text": "Plain_item"},
        {"type": "equation", "text": r"\theta_i = x_i"},
    ]
    decoded = [
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=0, target=2, label=PARENT_CHILD, score=0.8),
    ]
    tex = TreeDecoder().render_document(build_resolved_tree(records, decoded))

    assert r"\begin{itemize}" in tex
    assert r"\item Plain\_item" in tex
    assert "\\item \\[\n\\theta_i = x_i\n\\]" in tex
    assert r"\theta\_i" not in tex


def test_tree_decoder_wraps_consecutive_bullet_text_siblings_in_itemize():
    records = [
        {"type": "title", "text": "Contributions"},
        {"type": "paragraph", "text": "Lead paragraph."},
        {"type": "paragraph", "text": "• First_item"},
        {"type": "paragraph", "text": "2. Second item"},
        {"type": "paragraph", "text": "After paragraph."},
    ]
    decoded = [
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=0, target=2, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=0, target=3, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=0, target=4, label=PARENT_CHILD, score=0.9),
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, decoded))

    assert "Lead paragraph.\n\n\\begin{itemize}" in tex
    assert tex.count(r"\begin{itemize}") == 1
    assert r"\item First\_item" in tex
    assert r"\item Second item" in tex
    assert "\\end{itemize}\n\nAfter paragraph." in tex
    assert r"\textbullet{} First" not in tex


def test_tree_decoder_refuses_merge_when_target_starts_with_list_marker():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "paragraph", "text": "Lead into list."},
        {"type": "paragraph", "text": "1. First item"},
    ]
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    scores = torch.tensor([[0.99, 0.0, 0.0, 0.01]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    flattened = [node.text for node in root.children]
    assert flattened == ["Lead into list.", "1. First item"]
    assert all(child.merged_node_ids == [idx] for idx, child in enumerate(root.children))


def test_tree_decoder_allows_merge_from_list_marker_to_text_continuation():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "paragraph", "text": "1. First item starts"},
        {"type": "paragraph", "text": "and continues."},
    ]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.99, 0.0, 0.0],
            [0.99, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    assert len(root.children) == 1
    assert root.children[0].text == "1. First item starts and continues."
    assert root.children[0].merged_node_ids == [0, 1]


def test_tree_decoder_refuses_merge_across_intermediate_list_marker():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "equation_interline", "text": "d_{Ch}(x,c_k)"},
        {"type": "paragraph", "text": "4. Wasserstein distance"},
        {"type": "equation_interline", "text": "d_W(x,c_k)"},
    ]
    edge_index = torch.tensor([[0], [2]], dtype=torch.long)
    scores = torch.tensor([[0.99, 0.0, 0.0]], dtype=torch.float32)

    root = TreeDecoder(TreeDecoderConfig(merge_threshold=0.5, parent_threshold=0.5)).decode(records, edge_index, scores)

    assert [child.merged_node_ids for child in root.children] == [[0], [1], [2]]


def test_tree_decoder_groups_numbered_sibling_items_as_enumerate():
    records = [
        {"type": "title", "text": "Steps"},
        {"type": "paragraph", "text": "1. First step"},
        {"type": "paragraph", "text": "2. Second step"},
    ]
    decoded = [
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.9),
        DecodedEdge(source=0, target=2, label=PARENT_CHILD, score=0.9),
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, decoded))

    assert r"\begin{enumerate}" in tex
    assert r"\item First step" in tex
    assert r"\item Second step" in tex
    assert "1. First step" not in tex
    assert r"\begin{itemize}" not in tex


def test_tree_decoder_keeps_display_equations_inside_numbered_list_items():
    records = [
        {"type": "paragraph", "text": "1. Euclidean distance $(d_E)$: Captures geometry"},
        {"type": "equation_interline", "text": "d_E(x,c_k)=\\|f_\\theta(x)-c_k\\|_2"},
        {"type": "paragraph", "text": "2. Cosine distance $(d_C)$: Measures similarity"},
        {"type": "equation_interline", "text": "d_C(x,c_k)=1-\\frac{f_\\theta(x)\\cdot c_k}{\\|f_\\theta(x)\\|\\|c_k\\|}"},
        {"type": "paragraph", "text": "After list."},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert tex.count(r"\begin{enumerate}") == 1
    assert tex.count(r"\end{enumerate}") == 1
    assert tex.count(r"\item") == 2
    assert r"\item Euclidean distance" in tex
    assert r"\item Cosine distance" in tex
    assert "\\[\nd_E(x,c_k)" in tex
    assert tex.index(r"\item Euclidean distance") < tex.index("\\[\nd_E(x,c_k)") < tex.index(r"\item Cosine distance")
    assert "\\end{enumerate}\n\nAfter list." in tex


def test_tree_decoder_native_list_node_strips_marker_and_selects_environment():
    records = [{"type": "list", "list_type": "ordered", "text": "1. Native item"}]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\begin{enumerate}" in tex
    assert r"\item Native item" in tex
    assert "1. Native item" not in tex


def test_tree_decoder_renderer_sorts_root_and_nested_children_by_reading_order():
    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"})
    late = ResolvedNode(node_id=10, record={"type": "paragraph", "text": "Late paragraph."}, merged_node_ids=[10])
    title = ResolvedNode(node_id=2, record={"type": "title", "text": "Early Section"}, merged_node_ids=[2])
    title.children = [
        ResolvedNode(node_id=5, record={"type": "paragraph", "text": "Second child."}, merged_node_ids=[5]),
        ResolvedNode(node_id=3, record={"type": "paragraph", "text": "First child."}, merged_node_ids=[3]),
    ]
    root.children = [late, title]

    tex = TreeDecoder().render_document(root)

    assert tex.index(r"\section{Early Section}") < tex.index("Late paragraph.")
    assert tex.index("First child.") < tex.index("Second child.")


def test_tree_decoder_renderer_uses_state_machine_for_mixed_column_siblings():
    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"})
    right_top = ResolvedNode(node_id=0, record={"type": "paragraph", "text": "Right top.", "bbox": [520, 95, 920, 150]}, merged_node_ids=[0])
    title = ResolvedNode(node_id=1, record={"type": "title", "text": "Title", "bbox": [100, 20, 900, 60]}, merged_node_ids=[1])
    left = ResolvedNode(node_id=2, record={"type": "paragraph", "text": "Left tall.", "bbox": [80, 90, 480, 300]}, merged_node_ids=[2])
    bottom = ResolvedNode(node_id=3, record={"type": "paragraph", "text": "Conclusion.", "bbox": [100, 340, 900, 380]}, merged_node_ids=[3])
    right_lower = ResolvedNode(node_id=4, record={"type": "paragraph", "text": "Right lower.", "bbox": [520, 180, 920, 250]}, merged_node_ids=[4])
    root.children = [right_top, title, left, bottom, right_lower]

    tex = TreeDecoder().render_document(root)

    assert tex.index(r"\section{Title}") < tex.index("Left tall.")
    assert tex.index("Left tall.") < tex.index("Right top.")
    assert tex.index("Right top.") < tex.index("Right lower.")
    assert tex.index("Right lower.") < tex.index("Conclusion.")


def test_tree_decoder_renderer_prefers_explicit_order_over_bbox_for_cross_column_tail():
    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"})
    conclusion = ResolvedNode(
        node_id=117,
        record={"type": "title", "text": "Conclusion", "global_order": 117, "bbox": [232, 715, 331, 731]},
        merged_node_ids=[117],
    )
    conclusion_body = ResolvedNode(
        node_id=119,
        record={"type": "paragraph", "text": "Conclusion body.", "global_order": 119, "bbox": [514, 83, 915, 251]},
        merged_node_ids=[119],
    )
    references = ResolvedNode(
        node_id=120,
        record={"type": "title", "text": "References", "global_order": 120, "bbox": [666, 266, 764, 281]},
        merged_node_ids=[120],
    )
    root.children = [references, conclusion_body, conclusion]

    tex = TreeDecoder().render_document(root)

    assert tex.index(r"\section{Conclusion}") < tex.index("Conclusion body.")
    assert tex.index("Conclusion body.") < tex.index(r"\section*{References}")


def test_tree_decoder_demotes_custom_prefixed_same_style_titles_to_paragraphs():
    records = [
        {"type": "title", "text": "Introduction", "style_baseline_size": 14.0, "global_order": 0},
        {"type": "title", "text": "Q1: What is the probability?", "style_baseline_size": 14.0, "global_order": 1},
        {"type": "title", "text": "Case 1-1: Exactly One Fall", "style_baseline_size": 14.0, "global_order": 2},
        {"type": "title", "text": "1) Voxel Distributions", "style_baseline_size": 14.0, "global_order": 3},
        {"type": "title", "text": "Method", "style_baseline_size": 14.0, "global_order": 4},
        {"type": "paragraph", "text": "Body text.", "style_baseline_size": 10.0, "global_order": 5},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\section{Introduction}" in tex
    assert r"\section{Method}" in tex
    assert r"\paragraph*{Q1: What is the probability?}" in tex
    assert r"\paragraph*{Case 1-1: Exactly One Fall}" in tex
    assert r"\paragraph*{1) Voxel Distributions}" in tex
    assert r"\section{Q1" not in tex
    assert r"\section{Case" not in tex
    assert r"\section{Voxel" not in tex


def test_tree_decoder_demotes_numeric_list_style_title_to_subsection():
    records = [
        {"type": "title", "text": "Generalized Linear Models", "style_baseline_size": 15.0, "layout_role": "heading", "global_order": 0},
        {
            "type": "title",
            "text": r"3. Linear Relationship in Predictors: The linear predictor \eta is modeled as \beta^\top X",
            "style_baseline_size": 14.0,
            "layout_role": "list_item",
            "global_order": 1,
        },
        {"type": "paragraph", "text": "The body continues.", "style_baseline_size": 10.0, "global_order": 2},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\section{Generalized Linear Models}" in tex
    assert r"\subsection{Linear Relationship in Predictors" in tex
    assert r"\section{Linear Relationship in Predictors" not in tex
    assert "3. Linear Relationship" not in tex


def test_tree_decoder_keeps_cross_page_numbered_heading_in_previous_enumerate():
    if not has_torch():
        return
    import torch

    records = [
        {"type": "title", "text": "4 Poisson Regression", "layout_role": "heading", "global_order": 0},
        {"type": "title", "text": "GLMs: Associating Covariates with Risks", "layout_role": "heading", "global_order": 1},
        {"type": "paragraph", "text": "GLMs extend linear regression.", "global_order": 2},
        {"type": "paragraph", "text": "1. Response Variable Belongs to the Exponential Family:", "layout_role": "list_item", "global_order": 3},
        {"type": "paragraph", "text": "2. Model Predicts the Expected Value of Y:", "layout_role": "list_item", "global_order": 4},
        {"type": "page_number", "text": "5", "layout_role": "noise", "global_order": 5},
        {"type": "page_header", "text": "Survival Analysis", "layout_role": "noise", "global_order": 6},
        {
            "type": "title",
            "text": r"3. Linear Relationship in Predictors: The linear predictor \eta is modeled as \beta^\top X",
            "layout_role": "list_item",
            "global_order": 7,
        },
        {"type": "paragraph", "text": "The following examples illustrate GLMs.", "global_order": 8},
    ]
    edge_index = torch.tensor([[1, 1, 7], [3, 4, 8]], dtype=torch.long)
    scores = torch.tensor(
        [
            [0.0, 0.99, 0.01],
            [0.0, 0.99, 0.01],
            [0.0, 0.99, 0.01],
        ],
        dtype=torch.float32,
    )

    root = TreeDecoder(TreeDecoderConfig(parent_threshold=0.5)).decode(records, edge_index, scores)
    tex = TreeDecoder().render_document(root)

    assert tex.count(r"\begin{enumerate}") == 1
    assert tex.count(r"\item") == 3
    assert r"\item Linear Relationship in Predictors" in tex
    assert "The following examples illustrate GLMs." in tex
    assert r"\subsection{Linear Relationship in Predictors" not in tex
    assert tex.index("Model Predicts the Expected Value") < tex.index("Linear Relationship in Predictors")


def test_tree_decoder_keeps_layout_heading_as_subsection_scope_for_following_body():
    if not has_torch():
        return
    import torch

    records = [
        {
            "type": "title",
            "text": "4 Poisson Regression in Survival Analysis Framework",
            "layout_role": "heading",
            "layout_band_type": "full_span",
            "layout_band_column": "full",
            "layout_is_band_boundary": True,
            "global_order": 0,
        },
        {"type": "paragraph", "text": "Opening body.", "global_order": 1},
        {
            "type": "title",
            "text": "The Exponential Family and Its Role in GLMs",
            "layout_role": "heading",
            "layout_band_type": "double_column",
            "layout_band_column": "left",
            "style_spans": [{"text": "The Exponential Family and Its Role in GLMs", "font_size": 10.0, "is_bold": True}],
            "global_order": 2,
        },
        {"type": "paragraph", "text": "A prerequisite paragraph.", "global_order": 3},
        {"type": "equation_interline", "text": "f(y;\\eta)=b(y)", "global_order": 4},
        {"type": "paragraph", "text": "where:", "bbox": [112, 397, 163, 410], "page_idx": 0, "global_order": 5},
        {"type": "paragraph", "text": r"\eta: the natural parameter,", "bbox": [156, 421, 866, 436], "page_idx": 0, "global_order": 6},
        {"type": "paragraph", "text": "T(y): the sufficient statistic,", "bbox": [156, 439, 816, 455], "page_idx": 0, "global_order": 7},
    ]

    root = TreeDecoder().decode(
        records,
        edge_index=torch.empty((2, 0), dtype=torch.long),
        scores=torch.empty((0, 3), dtype=torch.float32),
    )
    tex = TreeDecoder().render_document(root)

    assert r"\section{Poisson Regression in Survival Analysis Framework}" in tex
    assert r"\subsection*{The Exponential Family and Its Role in GLMs}" in tex
    assert tex.index(r"\subsection*{The Exponential Family and Its Role in GLMs}") < tex.index("A prerequisite paragraph.")
    assert tex.index("A prerequisite paragraph.") < tex.index("\\[\nf(y;\\eta)=b(y)\n\\]")
    assert "where:\n\n\\begin{itemize}" in tex
    assert r"\item $\eta$: the natural parameter," in tex
    assert r"\item T(y): the sufficient statistic," in tex


def test_tree_decoder_keeps_consistent_freeform_title_style_structural():
    records = [
        {"type": "title", "text": "Introduction", "style_baseline_size": 14.0, "global_order": 0},
        {"type": "title", "text": "Related Work", "style_baseline_size": 14.0, "global_order": 1},
        {"type": "title", "text": "Conclusion", "style_baseline_size": 14.0, "global_order": 2},
        {"type": "paragraph", "text": "Body text.", "style_baseline_size": 10.0, "global_order": 3},
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\section{Introduction}" in tex
    assert r"\section{Related Work}" in tex
    assert r"\section{Conclusion}" in tex
    assert r"\paragraph*{Related Work}" not in tex


def test_tree_decoder_causality_barrier_drops_reversed_parent_edges():
    nodes = {
        2: ResolvedNode(node_id=2, record={"type": "paragraph", "text": "Earlier child."}, merged_node_ids=[2]),
        5: ResolvedNode(node_id=5, record={"type": "title", "text": "Later hallucinated parent."}, merged_node_ids=[5]),
    }
    edges = [DecodedEdge(source=5, target=2, label=PARENT_CHILD, score=0.99)]

    filtered = TreeDecoder().apply_causality_barrier(nodes, edges)

    assert filtered == []


def test_tree_decoder_causality_barrier_allows_floating_table_targets():
    nodes = {
        5: ResolvedNode(node_id=5, record={"type": "title", "text": "Later parent."}, merged_node_ids=[5]),
        2: ResolvedNode(node_id=2, record={"type": "table", "text": "Floating table."}, merged_node_ids=[2]),
    }
    edges = [DecodedEdge(source=5, target=2, label=PARENT_CHILD, score=0.99)]

    filtered = TreeDecoder().apply_causality_barrier(nodes, edges)

    assert filtered == edges


def test_tree_decoder_renders_algorithm_as_algorithmic_float():
    raw_algorithm = "Input: x_i < 1 % raw\nfor i in S do\nreturn x_i\nend"
    records = [{"type": "algorithm", "text": raw_algorithm}]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\usepackage{float}" in tex
    assert r"\usepackage{algorithm}" in tex
    assert r"\usepackage{algpseudocode}" in tex
    assert r"\begin{algorithmic}[1]" in tex
    assert r"\Require \(\displaystyle x_i < 1 \% raw\)" in tex
    assert r"\For{\texttt{i in S}}" in tex
    assert r"\State \Return \(\displaystyle x_i\)" in tex
    assert r"\EndFor" in tex
    assert r"\begin{verbatim}" not in tex
    assert r"x\_i" not in tex


def test_tree_decoder_algorithmic_converts_unicode_to_math_latex():
    records = [{"type": "algorithm", "text": "return θ ← θ - β∇L"}]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\State \Return \(\displaystyle \theta \gets \theta - \beta\nablaL\)" in tex
    assert "θ" not in tex
    assert "β" not in tex
    assert "∇" not in tex


def test_tree_decoder_algorithmic_keeps_c_like_code_out_of_math_mode():
    raw_algorithm = "wrapper(func, ret_type, n_args, args) { ret_type ret = func(args); return ret; }"
    records = [{"type": "algorithm", "text": raw_algorithm}]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\texttt{wrapper" in tex
    assert r"\{" in tex
    assert r"\}" in tex
    assert r"\(\displaystyle wrapper" not in tex


def test_tree_decoder_restores_line_breaks_for_heuristic_pseudocode_text():
    records = [{"type": "paragraph", "text": "Input: x_i Output: y_i for i in S return y_i end"}]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\Require \(\displaystyle x_i\)" in tex
    assert r"\Ensure \(\displaystyle y_i\)" in tex
    assert r"\For{\texttt{i in S}}" in tex
    assert r"\State \Return \(\displaystyle y_i\)" in tex
    assert r"\EndFor" in tex
    assert r"y\_i" not in tex


def test_tree_decoder_renders_table_as_structured_placeholder():
    records = [
        {
            "type": "table",
            "id": "table_3",
            "bbox": [10, 20.5, 300, 420],
            "text": "Table 1: Performance & Results\nraw table body",
        }
    ]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\begin{table}[H]" in tex
    assert "% [TODO_TABLE_RECONSTRUCT: BBOX=(10, 20.50, 300, 420), ID=table_3]" in tex
    assert r"\caption{Table 1: Performance \& Results}" in tex
    assert r"\begin{verbatim}" not in tex


def test_generation_renderer_uses_node_type_dispatch_for_titles_and_math():
    root = type("Root", (), {})()
    title = {"type": "title", "text": "Method_1", "children": [{"type": "title", "text": "Details_2"}]}
    root.children = [title, {"type": "equation", "text": r"\theta_i = x_i"}, {"type": "inline_math", "text": r"a_b"}]

    tex = render_latex_document(root)

    assert r"\section{Method\_1}" in tex
    assert r"\subsection{Details\_2}" in tex
    assert "\\[\n\\theta_i = x_i\n\\]" in tex
    assert r"$a_b$" in tex
    assert r"\theta\_i" not in tex


def test_generation_renderer_renders_reference_items_from_merged_records():
    root = type("Root", (), {})()
    root.children = [
        {
            "type": "reference",
            "reference_items": ["A. First.", "B. Second."],
            "merged_records": [{"type": "reference", "reference_items": ["C. Third."]}],
        }
    ]

    tex = render_latex_document(root)

    assert tex.count(r"\bibitem") == 3
    assert "C. Third." in tex


def test_generation_renderer_groups_reference_list_children_as_one_bibliography():
    root = type("Root", (), {})()
    root.children = [
        {
            "type": "title",
            "text": "References",
            "children": [
                {"type": "list", "list_type": "reference_list", "reference_items": ["A. First.", "B. Second."]},
                {"type": "list", "list_type": "reference_list", "reference_items": ["C. Third."]},
            ],
        }
    ]

    tex = render_latex_document(root)

    assert tex.count(r"\begin{thebibliography}{99}") == 1
    assert tex.count(r"\bibitem") == 3
    assert r"\begin{itemize}" not in tex


def test_generation_renderer_sorts_root_and_nested_children_by_id_numbers():
    root = type("Root", (), {})()
    root.children = [
        {"id": "node_10", "type": "paragraph", "text": "Late paragraph."},
        {
            "id": "node_2",
            "type": "title",
            "text": "Early Section",
            "children": [
                {"id": "node_5", "type": "paragraph", "text": "Second child."},
                {"id": "node_3", "type": "paragraph", "text": "First child."},
            ],
        },
    ]

    tex = render_latex_document(root)

    assert tex.index(r"\section{Early Section}") < tex.index("Late paragraph.")
    assert tex.index("First child.") < tex.index("Second child.")


def test_generation_renderer_prefers_explicit_order_over_bbox_for_cross_column_tail():
    root = type("Root", (), {})()
    root.children = [
        {"type": "title", "text": "References", "global_order": 120, "bbox": [666, 266, 764, 281]},
        {"type": "paragraph", "text": "Conclusion body.", "global_order": 119, "bbox": [514, 83, 915, 251]},
        {"type": "title", "text": "Conclusion", "global_order": 117, "bbox": [232, 715, 331, 731]},
    ]

    tex = render_latex_document(root)

    assert tex.index(r"\section{Conclusion}") < tex.index("Conclusion body.")
    assert tex.index("Conclusion body.") < tex.index(r"\section*{References}")


def test_generation_renderer_intercepts_algorithm_like_text_before_escaping():
    root = type("Root", (), {})()
    root.children = [{"type": "paragraph", "text": "Algorithm 1: Search Require: x_i if x_i < 1 return x_i end"}]

    tex = render_latex_document(root)

    assert r"\usepackage{algorithm}" in tex
    assert r"\usepackage{algpseudocode}" in tex
    assert r"\caption{Search}" in tex
    assert r"\Require \(\displaystyle x_i\)" in tex
    assert r"\If{\texttt{x\_i < 1}}" in tex
    assert r"\State \Return \(\displaystyle x_i\)" in tex
    assert r"\EndIf" in tex
    assert r"\begin{verbatim}" not in tex


def test_generation_renderer_algorithm_unicode_uses_latex_math():
    root = type("Root", (), {})()
    root.children = [{"type": "algorithm", "text": "return θ ← θ - β∇L"}]

    tex = render_latex_document(root)

    assert r"\State \Return \(\displaystyle \theta \gets \theta - \beta\nablaL\)" in tex
    assert "θ" not in tex


def test_generation_renderer_table_placeholder_uses_bbox_and_caption_slot():
    root = type("Root", (), {})()
    root.children = [
        {
            "type": "table",
            "global_order": 12,
            "bbox": [1, 2, 3, 4],
            "text": "Table 2: Ablation #1\nignored cells",
        }
    ]

    tex = render_latex_document(root)

    assert "% [TODO_TABLE_RECONSTRUCT: BBOX=(1, 2, 3, 4), ID=table_12]" in tex
    assert r"\caption{Table 2: Ablation \#1}" in tex
    assert "ignored cells" not in tex


def test_generation_renderer_wraps_mixed_bullet_sibling_runs():
    root = type("Root", (), {})()
    root.children = [
        {"type": "paragraph", "text": "Before."},
        {"type": "paragraph", "text": "- Alpha_item"},
        {"type": "paragraph", "text": "b. Beta item"},
        {"type": "paragraph", "text": "After."},
    ]

    tex = render_latex_document(root)

    assert "Before.\n\n\\begin{itemize}" in tex
    assert tex.count(r"\begin{itemize}") == 1
    assert r"\item Alpha\_item" in tex
    assert r"\item Beta item" in tex
    assert "\\end{itemize}\n\nAfter." in tex


def test_generation_renderer_preserves_structured_inline_formula_segments():
    root = type("Root", (), {})()
    root.children = [
        {
            "type": "paragraph",
            "text": r"Given x _ { i }.",
            "block": {
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "Given "},
                        {"type": "equation_inline", "content": r"x _ { i }"},
                        {"type": "text", "content": "."},
                    ]
                }
            },
        }
    ]

    tex = render_latex_document(root)

    assert r"Given $x _ { i }$." in tex
    assert r"x \_ \{ i \}" not in tex


def test_generation_renderer_repairs_contextual_inline_math_ocr_operator():
    root = type("Root", (), {})()
    root.children = [
        {
            "type": "paragraph",
            "text": r"The linear predictor η is modeled \arcsin = \beta ^ { \top } X",
            "block": {
                "content": {
                    "paragraph_content": [
                        {"type": "text", "content": "The linear predictor η is modeled "},
                        {"type": "equation_inline", "content": r"\arcsin = \beta ^ { \top } X"},
                    ]
                }
            },
        }
    ]

    tex = render_latex_document(root)

    assert r"The linear predictor \ensuremath{\eta} is modeled as: $\eta = \beta ^ { \top } X$" in tex
    assert r"\arcsin =" not in tex


def test_tree_decoder_does_not_swallow_unindented_text_after_numbered_item():
    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
    root.children = [
        ResolvedNode(
            node_id=1,
            record={
                "type": "paragraph",
                "text": "3. Linear Relationship in Predictors: The predictor is modeled.",
                "_render_as_list_item": True,
                "bbox": [100, 100, 600, 125],
                "page_idx": 0,
            },
            merged_node_ids=[1],
        ),
        ResolvedNode(
            node_id=2,
            record={
                "type": "paragraph",
                "text": "The following examples illustrate GLMs.",
                "bbox": [100, 145, 600, 170],
                "page_idx": 0,
            },
            merged_node_ids=[2],
        ),
    ]

    tex = TreeDecoder().render_document(root)

    assert "\\item Linear Relationship in Predictors: The predictor is modeled.\n\\end{enumerate}" in tex
    assert "\\end{enumerate}\n\nThe following examples illustrate GLMs." in tex


def test_tree_decoder_keeps_formula_explanation_inside_numbered_item():
    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
    root.children = [
        ResolvedNode(
            node_id=1,
            record={"type": "paragraph", "text": "1. Ensuring Non-Negativity:", "_render_as_list_item": True},
            merged_node_ids=[1],
        ),
        ResolvedNode(
            node_id=2,
            record={"type": "equation", "text": r"h(t|X)=h_0(t)+\\beta^T X"},
            merged_node_ids=[2],
        ),
        ResolvedNode(
            node_id=3,
            record={"type": "paragraph", "text": "which can result in invalid negative hazard rates."},
            merged_node_ids=[3],
        ),
        ResolvedNode(
            node_id=4,
            record={"type": "paragraph", "text": "2. Multiplicative Nature:", "_render_as_list_item": True},
            merged_node_ids=[4],
        ),
    ]

    tex = TreeDecoder().render_document(root)

    assert tex.count(r"\begin{enumerate}") == 1
    assert "which can result in invalid negative hazard rates." in tex
    assert tex.index(r"h(t|X)=h_0(t)+\\beta^T X") < tex.index("which can result")
    assert tex.count(r"\item") == 2


def test_algorithm_control_conditions_are_rendered_as_safe_text():
    records = [{
        "type": "algorithm",
        "text": "Algorithm 1: Demo\nif l < rul then continue ▷ prune node\nfor training uses \\thetat from server",
    }]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\If{\texttt{" in tex
    assert r"\For{\texttt{" in tex
    assert "▷" not in tex
    assert r"\thetat" not in tex


def test_escape_latex_covers_reserved_characters():
    assert escape_latex(r"a_b & 50% #1") == r"a\_b \& 50\% \#1"


def test_escape_latex_maps_unicode_math_and_falls_back_to_ascii():
    assert escape_latex("ϵ γ ≤ ∈ • café") == (
        r"\ensuremath{\epsilon} \ensuremath{\gamma} \ensuremath{\leq} "
        r"\ensuremath{\in} \textbullet{} cafe"
    )


def test_tree_decoder_wraps_inner_math_environments():
    records = [{"type": "equation", "text": r"\begin{array}{r}x_1\\x_2\end{array}"}]
    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert "\\[\n\\begin{array}{r}x_1\\\\x_2\\end{array}\n\\]" in tex


def test_render_equation_splits_multiple_tags_into_align():
    raw = r"a=b\tag{1} c=d\tag{2}"

    for renderer in (render_equation, render_generation_equation):
        tex = renderer(raw)
        assert tex.startswith("\\begin{align}")
        assert r"a=b \tag{1}" in tex
        assert r"c=d \tag{2}" in tex
        assert tex.count(r"\tag") == 2
        assert "\\[" not in tex


def test_tree_decoder_suppresses_long_redundant_ocr_continuation():
    root = ResolvedNode(node_id=-1, record={"type": "root", "text": "ROOT"}, merged_node_ids=[])
    root.children = [
        ResolvedNode(
            node_id=1,
            record={
                "type": "paragraph",
                "text": "Large Language Models have achieved remarkable success across a wide range of tasks such as text generation and sentiment analysis.",
            },
            merged_node_ids=[1],
        ),
        ResolvedNode(
            node_id=2,
            record={
                "type": "paragraph",
                "text": "across a wide range of tasks such as text generation and sentiment analysis.",
            },
            merged_node_ids=[2],
        ),
    ]

    tex = TreeDecoder().render_document(root)

    assert tex.count("across a wide range of tasks") == 1


def test_safe_verbatim_text_removes_raw_unicode_math():
    safe = safe_verbatim_text("theta θ ± café")

    assert "θ" not in safe
    assert "±" not in safe
    assert r"\ensuremath{\theta}" in safe
    assert r"\ensuremath{\pm}" in safe
    assert "cafe" in safe


def test_build_resolved_tree_merges_text_and_renderer_emits_tex_document():
    records = [
        {"type": "title", "text": "Introduction"},
        {"type": "paragraph", "text": "Cyber-"},
        {"type": "paragraph", "text": "security matters."},
    ]
    decoded = [
        DecodedEdge(source=1, target=2, label=MERGE, score=0.99),
        DecodedEdge(source=0, target=1, label=PARENT_CHILD, score=0.95),
    ]

    root = build_resolved_tree(records, decoded)
    tex = render_latex_document(root, RenderConfig(title="Demo"))

    assert r"\documentclass{article}" in tex
    assert r"\providecommand{\mathbfcal}[1]{\mathbf{\mathcal{#1}}}" in tex
    assert r"\section{Introduction}" in tex
    assert "Cybersecurity matters." in tex
    assert r"\usepackage{amsmath}" in tex
