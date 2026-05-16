# Frontend / Backend Contract v1

**Last updated**: 2026-05-14

This contract fixes the boundary between PDF extraction, TeX truth generation, GNN training, decoding, and LaTeX rendering.

## Contract Summary

```text
PDF Frontend
  -> content_v7_styles.json
  -> DocumentIR
  -> GNNViewAdapter
  -> GraphInput.pt

TeX Label Backend
  -> graph_labeled.pt
  -> alignment mapping

GNN
  -> PredictedRelations

Decoder
  -> RenderTreeIR

Generator
  -> .tex / .pdf
```

Each layer owns one job and must not silently rewrite another layer's facts.

## PDF Frontend Output

Canonical internal JSON:

```text
*_content_list_v7.json
*_content_list_v7_styles.json
```

Canonical IR adapter:

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

The full v7 JSON is the complete fact layer for rendering. The graph model uses
a narrower view produced by:

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
Generator code must render from full v7 / DocumentIR plus bridged predicted
relations, not from the filtered graph view alone.

## GraphInput

`graph.pt` is a PyTorch Geometric `Data` object:

```text
x           [N, 832] float32
edge_index  [2, E] long
edge_attr   [E, 22] float32
```

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

Calibration and deterministic guards can convert probabilities to decoded edges. These guards belong to decoder configuration, not to the graph tensors.

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
```

Renderer output is allowed to be approximate, but should not drop content unless explicitly marked noise/no-render.

## Generator

Canonical renderer:

```text
src/generation/ir_renderer.py
src/generation/ir_renderers/
```

Legacy renderer:

```text
src/generation/latex_renderer.py
```

The legacy renderer is compatibility-only. New behavior should go into the IR renderer and its helpers.

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
3. Graph tensors use the current 832/22 schema.
4. Edge labels are 3-class only.
5. Renderer order must be derived from reading order / RenderTreeIR, not raw list order.
6. Table/figure crops are assets, not graph text features.
```
