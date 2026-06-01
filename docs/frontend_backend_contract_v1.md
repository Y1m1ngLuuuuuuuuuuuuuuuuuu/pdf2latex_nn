# Frontend / Backend Contract v1

> Interface reference. For the current cross-paper interface registry, see
> `docs/INTERFACE_DESIGN_CURRENT_20260601.md`.

**Last updated**: 2026-05-24

This contract fixes the boundary between PDF extraction, optional TeX truth
generation, optional GNN training, decoding, and LaTeX rendering.  The current
default reconstruction path is v8/layout-first; the v7/GNN flow below is kept as
an optional relation-learning branch.

## Contract Summary

```text
PDF Frontend
  -> middle.json + content_list.json + optional style sidecar
  -> content_list_v8.json
  -> DocumentIR
  -> FrontMatterIR / heading style stack
  -> RenderTreeIR
  -> Generator

Optional Relation Branch
  -> content_v7_styles.json or future v8 graph-visible view
  -> GNNViewAdapter
  -> GraphInput.pt

TeX Label Backend
  -> graph_labeled.pt
  -> alignment mapping

GNN
  -> PredictedRelations / diagnostics

Generator
  -> .tex / .pdf
```

Each layer owns one job and must not silently rewrite another layer's facts.

## PDF Frontend Output

Current default internal JSON:

```text
*_content_list_v8.json
```

Current default IR adapter:

```text
src/adapters/mineru_v8_document_ir.py
```

Optional v7/GNN internal JSON:

```text
*_content_list_v7.json
*_content_list_v7_styles.json
```

Optional v7/GNN IR adapter:

```text
src/adapters/mineru_v7_document_ir.py
```

Frontend fields may include:

```text
id / index / page_idx
type / canonical_type / layout_role
text
bbox
style_spans
reading_order metadata
band / column / layout_layer metadata
toc/header/footer/noise flags
reference_items
float/table/figure metadata
footnote/margin-note candidates
```

Frontend must not write edge labels or model predictions.

## GNN View Adapter

The full normalized payload and `DocumentIR` are the complete fact layer for
rendering.  The graph model, when enabled for optional relation studies, uses a
narrower view produced by:

```text
src/perception/gnn_view_adapter.py
```

The adapter may exclude metadata, annotations, true page furniture, TOC entries,
or duplicate shadows from the graph view, but it must preserve a reversible map:

```text
gnn_to_v7_id
gnn_to_v7_ids
v7_id_to_gnn_idx
excluded_items_summary
```

This means "not sent to the GNN" never means "deleted from the document".
Generator code must render from full `DocumentIR` plus `RenderTreeIR` and, when
used, bridged predicted relations. It must not render from the filtered graph
view alone.

Float policy:

```text
figure/table/algorithm are not body text
figure/table/algorithm do enter the GNN view as float proxies
caption or placeholder text is used for embedding
raw table body / raw figure OCR is not used as paragraph semantics
MERGE is blocked for float proxies
message passing from float proxies into body text is masked
skip-over-float candidate edges preserve paragraph-continuation recall
```

## GraphInput

`graph.pt` is a PyTorch Geometric `Data` object:

```text
x           [N, node_dim] float32
edge_index  [2, E] long
edge_attr   [E, edge_dim] float32
```

Current known schema families:

```text
v7_registry_adapteraware_20260515_181724:
  node_dim = 832
  edge_dim = 22

v7_floatproxy_adapter_20260516_205926:
  node_dim = 832
  edge_dim = 26
```

Do not hard-code the tensor dimensions in new code.  Read the schema metadata
from the graph where possible, or bind the model config to the manifest family.

If labels are attached:

```text
y           [E] long, values in {0,1,2}
```

The graph schema is defined in:

```text
src/perception/schema.py
src/reasoning/graph_builder.py
```

## TeX Label Backend

Inputs:

```text
content_v7_styles.json
matching main.tex
unlabeled graph.pt
```

Outputs:

```text
labeled graph.pt
alignment_mapping.json
label report / quality errors
```

The label backend may reject a sample, but it must not repair PDF visual nodes in-place.

## PredictedRelations

Inference uses model probabilities over:

```text
MERGE
PARENT_CHILD
NONE
```

The raw GNN output is `edge_logits.pt`. Current inference also writes
`predicted_relations.json`, a `PredictedRelations` sidecar containing per-edge
probabilities, raw argmax labels, threshold config, and edge endpoints. This
sidecar is an audit record for the graph-visible candidate edges; it is not a
render source.

The decoder consumes those probabilities under deterministic constraints:

```text
raw logits/probabilities
  -> thresholded MERGE candidates
  -> merge contraction
  -> heading skeleton / active section scope
  -> constrained PARENT_CHILD selection
  -> ResolvedNode tree
  -> RenderTreeIR with v7 source ids
```

Calibration and deterministic guards belong to decoder configuration, not to
graph tensors. A high raw probability can still be rejected by merge barriers,
section-scope constraints, float/equation barriers, or graph-to-v7 bridge
checks.

## RenderTreeIR

Decoder output must preserve:

```text
ordered text nodes
merged node ids
parent-child tree
block types
float/caption associations
style references
source bbox references
source v7 node ids bridged from GNN indexes
```

Renderer output is allowed to be approximate, but should not drop content unless explicitly marked noise/no-render.

## Generator

Canonical renderer:

```text
src/generation/ir_renderer.py
src/generation/ir_renderers/
```

Low-level LaTeX helper module:

```text
src/generation/latex_helpers.py
```

Deprecated standalone tree renderer:

```text
src/generation/latex_renderer.py
```

`latex_helpers.py` keeps shared escaping, math, list, float, and algorithm
helpers used by the IR renderer. `latex_renderer.py` is not a production render
surface. New rendering behavior should go into the IR renderer, registry
renderers, or focused helper modules.

`OriginalLikeIRLatexRenderer` remains the production entrypoint, but role and
block rendering is now dispatched through a registry:

```text
OriginalLikeIRLatexRenderer
  -> IRRendererRegistry
    -> FrontMatterRenderer
    -> HeadingRenderer
    -> TextRenderer
    -> MathRenderer / AlgorithmCodeRenderer
    -> FigureRenderer
    -> TableRenderer
    -> ListRenderer
    -> ReferenceRenderer
    -> NoteRenderer
```

The registry layer is intentionally behavior-preserving: mature helper logic can
still live on `OriginalLikeIRLatexRenderer`, while new feature work should be
implemented in the specialized renderer for its `RenderRole` / `BlockType`.

## Invariants

```text
1. Production samples must have matching compiled PDF and TeX source.
2. v7 JSON is the only production frontend format.
3. Graph tensor dimensions are schema-family specific.  The locked registry
   baseline uses 832/22; the active float-proxy track uses 832/26.
4. Edge labels are 3-class only.
5. Renderer order must be derived from reading order / RenderTreeIR, not raw list order.
6. Table/figure crops are assets, not graph text features.
```
