from src.generation.latex_renderer import RenderConfig, render_latex_document
from src.reasoning.postprocess import (
    MERGE,
    PARENT_CHILD,
    DecodedEdge,
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

    assert [child.text for child in root.children] == ["Introduction", "Body text."]
    assert all(child.merged_node_ids == [idx] for idx, child in enumerate(root.children))


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
    assert [child.text for child in title.children] == ["Second body paragraph."]
    assert root.children[1].text == "First body paragraph."


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
