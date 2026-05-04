"""Generate supervised edge labels from PDF-to-TeX alignments."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from src.reasoning.latex_flattener import MATH_PLACEHOLDER, flatten_latex_file, mask_math_environments
from src.reasoning.tex_ast_builder import build_tex_ast_from_file, tex_nodes_by_id
from src.reasoning.tex_relation_labeler import TexRelationLabel, label_tex_relation


@dataclass(frozen=True)
class LabelGeneratorConfig:
    similarity_threshold: float = 0.55
    adjacent_siblings_only: bool = True
    directed_parent_child: bool = False
    orphan_label: int = int(TexRelationLabel.NONE)
    max_orphan_ratio: float = 0.30
    min_aligned_nodes: int = 1
    abort_on_bad_alignment: bool = True


class AlignmentQualityError(RuntimeError):
    """Raised when PDF-to-TeX alignment is too poor for supervised training."""


@dataclass(frozen=True)
class OrphanAlignment:
    node_index: int
    node_key: str
    reason: str
    score: float | None = None


@dataclass(frozen=True)
class LabelGenerationResult:
    data: Any
    label_counts: dict[int, int]
    orphan_alignments: list[OrphanAlignment]


@dataclass(frozen=True)
class TexAlignmentNode:
    tex_id: str
    node_type: str
    text: str
    clean: str
    path_ids: tuple[str, ...]


@dataclass(frozen=True)
class PdfAlignmentNode:
    node_index: int
    text: str
    clean: str
    item: dict[str, Any]


@dataclass(frozen=True)
class AlignmentMatch:
    pdf_node_index: int
    tex_id: str | None
    score: float


@dataclass(frozen=True)
class AlignmentLabelerConfig:
    similarity_threshold: float = 65.0
    min_clean_chars: int = 3
    max_orphan_ratio: float = 0.30
    abort_on_bad_alignment: bool = False
    output_mapping_json: Path | None = None


SECTION_LEVELS = {
    "part": 0,
    "chapter": 1,
    "section": 2,
    "subsection": 3,
    "subsubsection": 4,
    "paragraph": 5,
    "subparagraph": 6,
}
MATH_ENV_NAMES = {
    "$",
    "$$",
    "equation",
    "equation*",
    "align",
    "align*",
    "gather",
    "gather*",
    "multline",
    "multline*",
    "flalign",
    "flalign*",
    "math",
    "displaymath",
}
LIST_ENV_NAMES = {"itemize", "enumerate", "description"}
BLOCK_ENV_NAMES = {
    "abstract",
    "algorithm",
    "algorithmic",
    "algorithm2e",
    "caption",
    "figure",
    "figure*",
    "table",
    "table*",
}
CONTAINER_ENV_NAMES = {
    "document",
    "center",
    "flushleft",
    "flushright",
    "minipage",
    "small",
    "footnotesize",
    "scriptsize",
    "normalsize",
    "large",
    "Large",
}
SKIP_TEX_NODE_NAMES = {
    "documentclass",
    "usepackage",
    "label",
    "ref",
    "cite",
    "bibliographystyle",
    "bibliography",
}


class AlignmentLabeler:
    """Align MinerU PDF blocks to TexSoup nodes and inject edge labels into a graph."""

    def __init__(
        self,
        *,
        content_json_path: Path,
        tex_path: Path,
        graph_path: Path,
        config: AlignmentLabelerConfig | None = None,
    ) -> None:
        self.content_json_path = Path(content_json_path)
        self.tex_path = Path(tex_path)
        self.graph_path = Path(graph_path)
        self.config = config or AlignmentLabelerConfig()
        self.tex_nodes: dict[str, TexAlignmentNode] = {}
        self.pdf_nodes: list[PdfAlignmentNode] = []
        self.matches: list[AlignmentMatch] = []
        self.flattener_summary: dict[str, Any] | None = None

    def run(self, *, output_graph_path: Path | None = None, overwrite: bool = True) -> Any:
        graph = self.load_graph()
        self.pdf_nodes = self.parse_pdf_nodes()
        if len(self.pdf_nodes) != int(graph.num_nodes):
            raise ValueError(
                f"content node count ({len(self.pdf_nodes)}) does not match graph.num_nodes ({int(graph.num_nodes)})"
            )
        self.tex_nodes = {node.tex_id: node for node in self.parse_tex_nodes()}
        self.matches = self.align_pdf_to_tex()
        labels = self.build_edge_labels(graph)
        graph.y = labels
        graph.edge_label = labels
        graph.pdf_to_tex = [match.tex_id for match in self.matches]
        graph.pdf_to_tex_scores = [match.score for match in self.matches]
        graph.label_counts = label_counts(labels)
        graph.alignment_schema = {
            "strategy": "texsoup_rapidfuzz_partial_ratio_path_v1",
            "similarity_threshold": self.config.similarity_threshold,
            "content_json_path": str(self.content_json_path),
            "tex_path": str(self.tex_path),
            "flattener": self.flattener_summary,
        }
        self.assert_alignment_quality()
        if self.config.output_mapping_json is not None:
            self.write_mapping_json(self.config.output_mapping_json)
        destination = self.graph_path if output_graph_path is None and overwrite else output_graph_path
        if destination is not None:
            destination.parent.mkdir(parents=True, exist_ok=True)
            import torch

            torch.save(graph, destination)
        return graph

    def load_graph(self) -> Any:
        import torch

        return torch.load(self.graph_path, map_location="cpu", weights_only=False)

    def parse_pdf_nodes(self) -> list[PdfAlignmentNode]:
        content = json.loads(self.content_json_path.read_text(encoding="utf-8"))
        items = content.get("items", content if isinstance(content, list) else [])
        if not isinstance(items, list):
            raise ValueError(f"Expected {self.content_json_path} to contain an items list")
        nodes = []
        for index, item in enumerate(items):
            if not isinstance(item, dict):
                item = {"text_for_embedding": str(item)}
            text = pdf_item_text(item)
            nodes.append(PdfAlignmentNode(node_index=index, text=text, clean=clean_text(text), item=item))
        return nodes

    def parse_tex_nodes(self) -> list[TexAlignmentNode]:
        from TexSoup import TexSoup

        flattened = flatten_latex_file(self.tex_path)
        self.flattener_summary = flattened.summary()
        soup = TexSoup(flattened.content)
        builder = _TexSoupPathBuilder(self.config)
        builder.walk_soup(soup)
        return builder.nodes

    def align_pdf_to_tex(self) -> list[AlignmentMatch]:
        from rapidfuzz import fuzz

        tex_candidates = [node for node in self.tex_nodes.values() if len(node.clean) >= self.config.min_clean_chars]
        matches = []
        for pdf_node in self.pdf_nodes:
            if len(pdf_node.clean) < self.config.min_clean_chars:
                matches.append(AlignmentMatch(pdf_node_index=pdf_node.node_index, tex_id=None, score=0.0))
                continue
            best_node: TexAlignmentNode | None = None
            best_score = 0.0
            for tex_node in tex_candidates:
                score = float(fuzz.partial_ratio(pdf_node.clean, tex_node.clean))
                if score > best_score:
                    best_score = score
                    best_node = tex_node
            tex_id = best_node.tex_id if best_node is not None and best_score >= self.config.similarity_threshold else None
            matches.append(AlignmentMatch(pdf_node_index=pdf_node.node_index, tex_id=tex_id, score=best_score))
        return matches

    def build_edge_labels(self, graph: Any) -> Any:
        import torch

        labels = []
        edge_index = graph.edge_index.detach().cpu()
        for edge_pos in range(edge_index.shape[1]):
            source = int(edge_index[0, edge_pos].item())
            target = int(edge_index[1, edge_pos].item())
            labels.append(self.infer_relation(source, target))
        return torch.tensor(labels, dtype=torch.long)

    def infer_relation(self, source_index: int, target_index: int) -> int:
        source_match = self.matches[source_index] if 0 <= source_index < len(self.matches) else None
        target_match = self.matches[target_index] if 0 <= target_index < len(self.matches) else None
        if source_match is None or target_match is None or not source_match.tex_id or not target_match.tex_id:
            return int(TexRelationLabel.NONE)
        path_u = self.tex_nodes[source_match.tex_id].path_ids
        path_v = self.tex_nodes[target_match.tex_id].path_ids
        if path_u == path_v:
            return int(TexRelationLabel.MERGE)
        if path_v[:-1] == path_u:
            return int(TexRelationLabel.PARENT_CHILD)
        if len(path_u) == len(path_v) and path_u[:-1] == path_v[:-1]:
            return int(TexRelationLabel.SIBLING)
        return int(TexRelationLabel.NONE)

    def assert_alignment_quality(self) -> None:
        orphan_count = sum(1 for match in self.matches if not match.tex_id)
        orphan_ratio = orphan_count / max(1, len(self.matches))
        if orphan_ratio > self.config.max_orphan_ratio:
            message = (
                "bad fuzzy alignment quality: "
                f"orphan_count={orphan_count}, num_nodes={len(self.matches)}, "
                f"orphan_ratio={orphan_ratio:.2%}, max_orphan_ratio={self.config.max_orphan_ratio:.2%}"
            )
            if self.config.abort_on_bad_alignment:
                raise AlignmentQualityError(message)

    def write_mapping_json(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "alignment_mapping_v1",
            "content_json_path": str(self.content_json_path),
            "tex_path": str(self.tex_path),
            "graph_path": str(self.graph_path),
            "similarity_threshold": self.config.similarity_threshold,
            "flattener": self.flattener_summary,
            "matches": [asdict(match) for match in self.matches],
            "tex_nodes": [asdict(node) for node in self.tex_nodes.values()],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class _TexSoupPathBuilder:
    def __init__(self, config: AlignmentLabelerConfig) -> None:
        self.config = config
        self.nodes: list[TexAlignmentNode] = []
        self.next_id = 1
        self.section_by_level: dict[int, str] = {}
        self.path_by_id: dict[str, tuple[str, ...]] = {"ROOT": ("ROOT",)}

    def walk_soup(self, soup: Any) -> None:
        self.walk_children(getattr(soup, "contents", []) or [], parent_id="ROOT")

    def walk_children(self, children: list[Any], *, parent_id: str) -> None:
        paragraph_buffer: list[str] = []
        for child in children:
            name = tex_node_name(child)
            if name in SKIP_TEX_NODE_NAMES:
                continue
            if name in CONTAINER_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id))
                continue
            if name in SECTION_LEVELS:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                node_id = self.add_node(name, tex_node_text(child), self.section_parent(name))
                level = SECTION_LEVELS[name]
                self.section_by_level = {key: value for key, value in self.section_by_level.items() if key < level}
                self.section_by_level[level] = node_id
                continue
            if name in LIST_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                env_id = self.add_node(name, name, self.current_parent(parent_id))
                self.walk_children(getattr(child, "contents", []) or [], parent_id=env_id)
                continue
            if name == "item":
                self.flush_paragraphs(paragraph_buffer, parent_id)
                self.add_node(name, tex_node_text(child), parent_id)
                continue
            if name in MATH_ENV_NAMES:
                paragraph_buffer.append(" [MATH] ")
                continue
            if name in BLOCK_ENV_NAMES:
                self.flush_paragraphs(paragraph_buffer, parent_id)
                block_id = self.add_node(name, tex_node_text(child), self.current_parent(parent_id))
                if block_id is None:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id))
                continue
            if name == "text":
                paragraph_buffer.append(str(child))
                continue
            if hasattr(child, "contents"):
                text = tex_node_text(child)
                if text:
                    paragraph_buffer.append(f" {text} ")
                else:
                    self.walk_children(getattr(child, "contents", []) or [], parent_id=self.current_parent(parent_id))
        self.flush_paragraphs(paragraph_buffer, parent_id)

    def add_node(self, node_type: str, text: str, parent_id: str) -> str | None:
        clean = clean_text(text)
        if len(clean) < self.config.min_clean_chars and clean != "math":
            return None
        tex_id = f"T_{self.next_id:06d}"
        self.next_id += 1
        parent_path = self.path_by_id.get(parent_id, ("ROOT",))
        path = (*parent_path, tex_id)
        self.path_by_id[tex_id] = path
        self.nodes.append(TexAlignmentNode(tex_id=tex_id, node_type=node_type, text=text.strip(), clean=clean, path_ids=path))
        return tex_id

    def flush_paragraphs(self, paragraph_buffer: list[str], parent_id: str) -> None:
        if not paragraph_buffer:
            return
        raw_text = "".join(paragraph_buffer)
        paragraph_buffer.clear()
        for paragraph in re.split(r"(?:\r?\n\s*){2,}", raw_text):
            if paragraph.strip():
                self.add_node("text", paragraph, self.current_parent(parent_id))

    def section_parent(self, name: str) -> str:
        level = SECTION_LEVELS[name]
        lower = [key for key in self.section_by_level if key < level]
        if not lower:
            return "ROOT"
        return self.section_by_level[max(lower)]

    def current_parent(self, fallback_parent: str) -> str:
        if fallback_parent != "ROOT":
            return fallback_parent
        if not self.section_by_level:
            return "ROOT"
        return self.section_by_level[max(self.section_by_level)]


def clean_text(text: Any) -> str:
    """Aggressively normalize PDF/TeX text for fuzzy alignment."""

    value = mask_math_environments(str(text or ""))
    value = re.sub(r"\[math\]", f" {MATH_PLACEHOLDER} ", value, flags=re.IGNORECASE)
    value = value.lower()
    value = re.sub(r"\\[a-zA-Z]+\*?(?:\s*\[[^\]]*\])?", " ", value)
    value = re.sub(r"\\.", " ", value)
    value = re.sub(rf"[^0-9a-z\u4e00-\u9fff]+", "", value)
    return value


def pdf_item_text(item: dict[str, Any]) -> str:
    """Resolve the best text-bearing field from a MinerU content item."""

    for key in (
        "text_for_embedding",
        "text",
        "content",
        "caption",
        "latex",
        "html",
        "table_body",
        "reference_items",
        "spans",
    ):
        if key in item:
            text = stringify_text_payload(item[key])
            if text.strip():
                return text
    item_type = str(item.get("type") or item.get("raw_type") or "").lower()
    if "equation" in item_type or "formula" in item_type:
        return "[MATH]"
    if "figure" in item_type or "image" in item_type:
        return "[FIGURE]"
    if "table" in item_type:
        return "[TABLE]"
    return ""


def stringify_text_payload(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return " ".join(stringify_text_payload(item) for item in value)
    if isinstance(value, dict):
        parts: list[str] = []
        for key in ("text", "content", "title", "caption", "latex", "html", "raw"):
            if key in value:
                parts.append(stringify_text_payload(value[key]))
        if parts:
            return " ".join(parts)
        return " ".join(stringify_text_payload(item) for item in value.values())
    return str(value)


def tex_node_name(node: Any) -> str:
    name = getattr(node, "name", None)
    if name is None:
        return "text" if isinstance(node, str) else ""
    return str(name)


def tex_node_text(node: Any) -> str:
    name = tex_node_name(node)
    if name in MATH_ENV_NAMES:
        return "[MATH]"
    if name in SKIP_TEX_NODE_NAMES:
        return ""
    if isinstance(node, str):
        return node
    contents = getattr(node, "contents", None)
    if contents:
        return " ".join(tex_node_text(child) for child in contents)
    args = getattr(node, "args", None)
    if args:
        return str(args)
    return str(node or "")


def label_counts(labels: Any) -> dict[int, int]:
    values = labels.detach().cpu().tolist()
    return {label: values.count(label) for label in range(4)}


def load_pdf_to_tex_mapping(path: Path) -> dict[str, Any]:
    """Load a PDF-block to TeX-node alignment mapping from JSON or JSONL."""

    if not path.exists():
        raise FileNotFoundError(path)
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return {}
    if path.suffix.lower() == ".jsonl":
        mapping: dict[str, Any] = {}
        for line in text.splitlines():
            record = json.loads(line)
            pdf_id = record.get("pdf_id") or record.get("block_id") or record.get("source")
            tex_id = record.get("tex_id") or record.get("target")
            if pdf_id and tex_id:
                mapping[str(pdf_id)] = {"tex_id": str(tex_id), "score": record.get("score")}
        return mapping
    data = json.loads(text)
    if isinstance(data, dict) and isinstance(data.get("alignments"), list):
        mapping: dict[str, Any] = {}
        for record in data["alignments"]:
            pdf_id = record.get("pdf_id") or record.get("block_id") or record.get("source")
            tex_id = record.get("tex_id") or record.get("target")
            if pdf_id and tex_id:
                mapping[str(pdf_id)] = {"tex_id": str(tex_id), "score": record.get("score")}
        return mapping
    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain an object mapping or an alignments list")
    return data


def label_graph_edges_from_paths(
    data: Any,
    *,
    tex_path: Path,
    pdf_to_tex_path: Path,
    config: LabelGeneratorConfig | None = None,
    orphan_log_path: Path | None = None,
) -> LabelGenerationResult:
    tex_ast = build_tex_ast_from_file(tex_path)
    pdf_to_tex = load_pdf_to_tex_mapping(pdf_to_tex_path)
    return label_graph_edges(
        data,
        tex_ast=tex_ast,
        pdf_to_tex=pdf_to_tex,
        config=config,
        orphan_log_path=orphan_log_path,
    )


def label_graph_edges(
    data: Any,
    *,
    tex_ast: dict[str, Any] | list[dict[str, Any]] | dict[str, dict[str, Any]],
    pdf_to_tex: dict[str, Any],
    config: LabelGeneratorConfig | None = None,
    orphan_log_path: Path | None = None,
) -> LabelGenerationResult:
    """Attach `data.y` edge labels, falling back to None for orphan nodes."""

    import torch

    cfg = config or LabelGeneratorConfig()
    ast_nodes = tex_nodes_by_id(tex_ast)
    node_records = getattr(data, "node_records", None)
    if not isinstance(node_records, list):
        node_records = [{} for _ in range(int(data.num_nodes))]

    node_tex_ids: dict[int, str | None] = {}
    orphans: dict[int, OrphanAlignment] = {}
    for node_index in range(int(data.num_nodes)):
        tex_id, orphan = resolve_node_tex_id(node_index, node_records[node_index], pdf_to_tex, cfg)
        node_tex_ids[node_index] = tex_id
        if orphan is not None:
            orphans[node_index] = orphan
    orphan_list = list(orphans.values())
    if orphan_log_path is not None:
        write_orphan_log(orphan_log_path, orphan_list)
    assert_alignment_quality(num_nodes=int(data.num_nodes), orphan_count=len(orphan_list), config=cfg)

    labels: list[int] = []
    edge_index = data.edge_index.detach().cpu()
    for edge_pos in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        tex_source = node_tex_ids.get(source)
        tex_target = node_tex_ids.get(target)
        if not tex_source or not tex_target:
            labels.append(cfg.orphan_label)
            continue
        label = label_tex_relation(
            tex_source,
            tex_target,
            ast_nodes,
            adjacent_siblings_only=cfg.adjacent_siblings_only,
            directed_parent_child=cfg.directed_parent_child,
        )
        labels.append(int(label))

    y = torch.tensor(labels, dtype=torch.long)
    data.y = y
    data.edge_label = y
    data.label_schema = {
        "task": "edge_relation_classification",
        "labels": {
            int(TexRelationLabel.MERGE): "merge",
            int(TexRelationLabel.PARENT_CHILD): "parent_child",
            int(TexRelationLabel.SIBLING): "sibling",
            int(TexRelationLabel.NONE): "none",
        },
        "orphan_label": cfg.orphan_label,
        "similarity_threshold": cfg.similarity_threshold,
    }
    data.pdf_to_tex = [node_tex_ids.get(idx) for idx in range(int(data.num_nodes))]

    label_counts = {label: labels.count(label) for label in range(4)}
    return LabelGenerationResult(data=data, label_counts=label_counts, orphan_alignments=orphan_list)


def assert_alignment_quality(*, num_nodes: int, orphan_count: int, config: LabelGeneratorConfig) -> bool:
    aligned_nodes = max(0, num_nodes - orphan_count)
    orphan_ratio = orphan_count / max(1, num_nodes)
    if orphan_ratio > config.max_orphan_ratio or aligned_nodes < config.min_aligned_nodes:
        message = (
            "bad alignment quality: "
            f"orphan_count={orphan_count}, num_nodes={num_nodes}, "
            f"orphan_ratio={orphan_ratio:.2%}, max_orphan_ratio={config.max_orphan_ratio:.2%}, "
            f"aligned_nodes={aligned_nodes}, min_aligned_nodes={config.min_aligned_nodes}"
        )
        if config.abort_on_bad_alignment:
            raise AlignmentQualityError(message)
        return False
    return True


def resolve_node_tex_id(
    node_index: int,
    node_record: dict[str, Any],
    pdf_to_tex: dict[str, Any],
    config: LabelGeneratorConfig,
) -> tuple[str | None, OrphanAlignment | None]:
    node_key = resolve_node_key(node_index, node_record, pdf_to_tex)
    if node_key is None:
        return None, OrphanAlignment(node_index=node_index, node_key=str(node_index), reason="missing_alignment")

    raw_alignment = pdf_to_tex.get(node_key)
    tex_id, score = parse_alignment_value(raw_alignment)
    if not tex_id:
        return None, OrphanAlignment(node_index=node_index, node_key=node_key, reason="missing_tex_id", score=score)
    if score is not None and score < config.similarity_threshold:
        return None, OrphanAlignment(node_index=node_index, node_key=node_key, reason="low_similarity", score=score)
    return tex_id, None


def resolve_node_key(node_index: int, node_record: dict[str, Any], pdf_to_tex: dict[str, Any]) -> str | None:
    candidates: list[str] = []
    for key in ("block_id", "id", "pdf_id", "global_order", "visual_order"):
        value = node_record.get(key)
        if value is not None:
            candidates.append(str(value))
    candidates.extend([str(node_index), f"P_{node_index}", f"P_{node_index + 1}", f"B_{node_index}", f"B_{node_index + 1}"])
    for candidate in candidates:
        if candidate in pdf_to_tex:
            return candidate
    return None


def parse_alignment_value(value: Any) -> tuple[str | None, float | None]:
    if isinstance(value, str):
        return value, None
    if isinstance(value, dict):
        tex_id = value.get("tex_id") or value.get("target") or value.get("id")
        score = value.get("score")
        if score is None:
            score = value.get("similarity")
        if score is None:
            score = value.get("confidence")
        parsed_score = float(score) if isinstance(score, (int, float)) else None
        return str(tex_id) if tex_id else None, parsed_score
    return None, None


def write_orphan_log(path: Path, orphans: list[OrphanAlignment]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for orphan in orphans:
            file.write(json.dumps(asdict(orphan), ensure_ascii=False) + "\n")
