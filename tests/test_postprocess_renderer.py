from src.generation.latex_renderer import RenderConfig, render_latex_document
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
    assert tex.count(r"\section{References.}") == 1
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
    assert r"\section{Abstract}" in tex


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
    assert tex.index("Conclusion body.") < tex.index(r"\section{References}")


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
    assert r"\For{i in S}" in tex
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


def test_tree_decoder_restores_line_breaks_for_heuristic_pseudocode_text():
    records = [{"type": "paragraph", "text": "Input: x_i Output: y_i for i in S return y_i end"}]

    tex = TreeDecoder().render_document(build_resolved_tree(records, []))

    assert r"\Require \(\displaystyle x_i\)" in tex
    assert r"\Ensure \(\displaystyle y_i\)" in tex
    assert r"\For{i in S}" in tex
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
    assert tex.index("Conclusion body.") < tex.index(r"\section{References}")


def test_generation_renderer_intercepts_algorithm_like_text_before_escaping():
    root = type("Root", (), {})()
    root.children = [{"type": "paragraph", "text": "Algorithm 1: Search Require: x_i if x_i < 1 return x_i end"}]

    tex = render_latex_document(root)

    assert r"\usepackage{algorithm}" in tex
    assert r"\usepackage{algpseudocode}" in tex
    assert r"\caption{Search}" in tex
    assert r"\Require \(\displaystyle x_i\)" in tex
    assert r"\If{\(\displaystyle x_i < 1\)}" in tex
    assert r"\State \Return \(\displaystyle x_i\)" in tex
    assert r"\EndIf" in tex
    assert r"\begin{verbatim}" not in tex
    assert r"x\_i" not in tex


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
