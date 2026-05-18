# Feature Schema v0

**Last updated**: 2026-05-18

This document fixes the current v7 graph tensor contract. The version name remains `feature_schema_v0` because downstream code imports that schema, but the active implementation is the v7 feature set.

## Versions

```text
pipeline_version: v7
graph_schema_version: graph_v7
node_feature_dim: 832
edge_attr_dim: 26
labels: MERGE=0, PARENT_CHILD=1, NONE=2
```

Historical locked reports/checkpoints from `v7_registry_adapteraware_20260515_181724`
used `edge_attr_dim=22`. The current checked-in schema adds four float
barrier/skip features and is being validated under the
`v7_floatproxy_adapter_20260516_205926` rebuild tag. Old checkpoints remain
valid only with their original graph/schema compatibility path.

Coordinates come from MinerU/PDF page space and are normalized or transformed by the graph builder. Raw bboxes stay available in node metadata.

## Node Feature Layout

Node features are built in `src/reasoning/graph_builder.py` and described in `src/perception/schema.py`.

Current high-level slices:

```text
semantic          SciBERT raw vector segment
type              MinerU/canonical block type one-hot
geometry          bbox anchors and local/global geometry
scroll            pseudo-y / document progress
derived           density, aspect, macro position
style             PyMuPDF font/style statistics
sequence          1D position encoding
column            local column assignment
title             heading/title probes
layout_layer      main/float/math/noise/front/back layer hints
flow_context      band/column/reading-flow context
```

The model may project and normalize semantic features internally. The `.pt` graph stores the full raw feature contract so ablations can mask feature groups at runtime without rewriting data.

Node tensors are built from the `GNNViewAdapter` output, not directly from the
full v7 item list. The graph stores mapping sidecars so predictions can be
bridged back to the complete v7 fact layer:

```text
gnn_to_v7_id
gnn_to_v7_ids
v7_id_to_gnn_idx
gnn_view_summary
```

Current adapter behavior:

```text
metadata/noise/annotation/toc/duplicate-shadow -> normally excluded
figure/table/algorithm -> included as float_proxy nodes
float_proxy text -> caption text when available, otherwise [FIGURE]/[TABLE]/[ALGORITHM]
raw table cells / figure OCR -> not embedded as ordinary text
```

## Edge Feature Layout

Edge features are directional. For edge `u -> v`, deltas are computed from source to target.

Current groups:

```text
semantic          cosine similarity
spatial           y gap, x delta, alignment, center distance
typography        font-size/style deltas
overlap_gutter    y-overlap and x-gutter barrier cues
index_bins        binned reading-order distance
punctuation       source terminal punctuation and hyphen probes
layout_flow       band/column/page transition cues
float_barrier     float_skip edge marker and intervening figure/table flags
```

`PARENT_CHILD` is directional. If `u -> v` is parent-child, the reverse `v -> u` is usually `NONE`.

## Candidate Edges

The graph builder uses a high-recall candidate strategy:

```text
sequential window
spatial neighbors
long-sight / local scope anchors
float skip window
forced reading-flow edges
```

The final four current raw edge features are:

```text
is_float_skip_edge
has_float_between
has_figure_between
has_table_between
```

They are hints for “paragraph continues across a visual float” without letting
the float body itself become a MERGE target.

Before training, candidate-edge recall is profiled against TeX-derived true positive edges. Production manifests should filter or reject samples below the configured recall threshold.

## Content Types

The graph keeps structural types separate:

```text
text
title
list
equation / inline_math
figure
table
caption
algorithm / code
reference
footnote / margin_note
toc
header_footer / noise
other
```

Tables and figures may carry captions and crop asset references. Their image/table body content should not be fed to SciBERT as ordinary paragraph text.

## Validation Rules

Graph tensors must satisfy:

```text
x.ndim == 2
edge_index.shape[0] == 2
edge_attr.shape[0] == edge_index.shape[1]
edge_attr.shape[1] == 26
y.shape[0] == edge_index.shape[1] when labeled
finite float32 node/edge features
non-empty node and edge sets
```

Auxiliary graph masks may also be present:

```text
message_edge_mask    [E] bool, optional, controls GAT propagation only
merge_candidate_mask [E] bool, optional, controls class-specific MERGE gating only
```

Validation and sanitization live in:

```text
src/datasets/document_dataset.py
src/pipeline/v7_contract.py
```
