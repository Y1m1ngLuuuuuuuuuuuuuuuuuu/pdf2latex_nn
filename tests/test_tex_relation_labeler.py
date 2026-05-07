from src.reasoning.tex_ast_builder import build_tex_ast, tex_nodes_by_id
from src.reasoning.tex_relation_labeler import TexRelationLabel, label_pdf_edges, label_tex_relation


SAMPLE_TEX = r"""
\section{Method}
Intro paragraph.

\begin{itemize}
  \item Apple
  \item Banana
  \item Cherry
\end{itemize}

\section{Results}
Another paragraph.
"""


def node_by_text(nodes, text):
    for node in nodes.values():
        if node.get("text") == text:
            return node
    raise AssertionError(f"missing node text: {text}")


def nodes_by_type(nodes, node_type):
    return [node for node in nodes.values() if node.get("node_type") == node_type]


def test_tex_ast_builder_records_paths_parent_ids_and_child_indexes():
    payload = build_tex_ast(SAMPLE_TEX, document_id="sample")
    nodes = tex_nodes_by_id(payload)

    method = node_by_text(nodes, "Method")
    itemize = node_by_text(nodes, "itemize")
    items = nodes_by_type(nodes, "item")

    assert payload["schema_version"] == "tex_ast_path_v0"
    assert method["path"] == ["ROOT", method["tex_id"]]
    assert itemize["parent_id"] == method["tex_id"]
    assert itemize["path"] == ["ROOT", method["tex_id"], itemize["tex_id"]]
    assert [item["parent_id"] for item in items] == [itemize["tex_id"]] * 3
    assert [item["child_index"] for item in items] == [0, 1, 2]


def test_label_tex_relation_uses_path_encoding():
    payload = build_tex_ast(SAMPLE_TEX)
    nodes = tex_nodes_by_id(payload)
    itemize = node_by_text(nodes, "itemize")
    items = nodes_by_type(nodes, "item")
    method = node_by_text(nodes, "Method")
    results = node_by_text(nodes, "Results")

    assert label_tex_relation(items[0]["tex_id"], items[0]["tex_id"], nodes) == TexRelationLabel.MERGE
    assert label_tex_relation(itemize["tex_id"], items[0]["tex_id"], nodes) == TexRelationLabel.PARENT_CHILD
    assert label_tex_relation(items[0]["tex_id"], itemize["tex_id"], nodes) == TexRelationLabel.NONE
    assert label_tex_relation(items[0]["tex_id"], items[1]["tex_id"], nodes) == TexRelationLabel.NONE
    assert label_tex_relation(items[0]["tex_id"], items[2]["tex_id"], nodes) == TexRelationLabel.NONE
    assert label_tex_relation(method["tex_id"], results["tex_id"], nodes) == TexRelationLabel.NONE


def test_label_tex_relation_can_opt_into_legacy_undirected_parent_child():
    payload = build_tex_ast(SAMPLE_TEX)
    nodes = tex_nodes_by_id(payload)
    itemize = node_by_text(nodes, "itemize")
    items = nodes_by_type(nodes, "item")

    assert (
        label_tex_relation(items[0]["tex_id"], itemize["tex_id"], nodes, directed_parent_child=False)
        == TexRelationLabel.PARENT_CHILD
    )


def test_label_tex_relation_treats_distant_siblings_as_none_even_when_requested():
    payload = build_tex_ast(SAMPLE_TEX)
    nodes = tex_nodes_by_id(payload)
    items = nodes_by_type(nodes, "item")

    assert (
        label_tex_relation(items[0]["tex_id"], items[2]["tex_id"], nodes, adjacent_siblings_only=False)
        == TexRelationLabel.NONE
    )


def test_label_pdf_edges_uses_pdf_to_tex_mapping_and_none_for_missing_alignment():
    payload = build_tex_ast(SAMPLE_TEX)
    nodes = tex_nodes_by_id(payload)
    items = nodes_by_type(nodes, "item")
    mapping = {
        "P_1": items[0]["tex_id"],
        "P_2": items[0]["tex_id"],
        "P_3": items[1]["tex_id"],
    }

    labels = label_pdf_edges([("P_1", "P_2"), ("P_1", "P_3"), ("P_1", "P_missing")], mapping, payload)

    assert labels == [
        int(TexRelationLabel.MERGE),
        int(TexRelationLabel.NONE),
        int(TexRelationLabel.NONE),
    ]


def test_tex_ast_builder_masks_comments_without_collapsing_paragraph_breaks():
    payload = build_tex_ast("First paragraph.\n% ignored comment\n\nSecond paragraph.")
    texts = [node["text"] for node in payload["nodes"] if node["node_type"] == "paragraph_text"]

    assert texts == ["First paragraph.", "Second paragraph."]
