# Ground Truth Labeling v0

**Last updated**: 2026-05-22

This document describes the current automatic label generator. It creates training labels from matching TeX source and v7 PDF graph candidates.

## Inputs

```text
content_v7_styles.json
matching main.tex
unlabeled graph.pt
```

Production samples must come from a compiled-source loop. Official PDFs paired only by id are not trusted for labels.

## TeX Processing

Implementation:

```text
src/reasoning/latex_flattener.py
src/reasoning/label_generator.py
```

The flattener:

```text
strips comments
expands input/include recursively
injects .bbl when available
expands simple zero-arg macros
masks math environments for robust parsing
silently drops known non-visual macros
rejects known layout-breaking drawing environments
```

The parser emits ordered TeX nodes:

```text
section
paragraph
equation_display
list_container
list_item
figure_caption
table_caption
algorithm
reference
front_matter / abstract where detectable
```

Unknown wrapper macros with text are unwrapped into paragraph-like nodes. Unknown environments are downgraded to containers unless they are known layout breakers.

## PDF / TeX Alignment

The alignment engine uses ordered sliding windows instead of global O(N^2) matching:

```text
1. flatten PDF visual nodes in v7 reading order
2. flatten TeX AST nodes in source order
3. clean both sides for fuzzy comparison
4. accumulate PDF nodes until a TeX node is matched
5. apply blind/weak alignment for formulas, floats, references, and captions
```

PDF nodes are the same graph-visible nodes produced by `GNNViewAdapter`.
The labeler does not align against the complete v7 fact layer directly, because
that would desynchronize graph indexes and labels. Excluded full-v7 nodes remain
available to the generator through the graph-to-v7 mapping.

For the current float-proxy experiment, figure/table/algorithm PDF-side nodes
are present in the graph-visible sequence as proxies. Their alignment text is
caption text or a stable placeholder, not raw table cells or noisy figure OCR.
This keeps TeX float/caption structure learnable while avoiding semantic
pollution of paragraph MERGE labels.

Expected non-main-flow nodes such as headers, footers, TOC entries, and some metadata are excluded or exempted from the effective orphan ratio.

## Labels

The output labels are:

```text
MERGE        = 0
PARENT_CHILD = 1
NONE         = 2
```

Rules:

```text
same TeX node + compatible visual types -> MERGE
TeX parent maps to first parent bbox and child maps to first child bbox -> PARENT_CHILD
anything else -> NONE
```

The historical rule above is still the compatibility baseline, but current
analysis treats it as insufficiently expressive for MERGE training. The active
direction is channel-aware relation supervision: TeX alignment is evidence, not
the whole label definition.

Additional audit-only fields now used for MERGE inspection:

```text
relation_family:
  BODY_TEXT_CONTINUATION
  LIST_CONTINUATION
  FORMULA_LEAD_IN
  FORMULA_CONTEXT
  FLOAT_SKIP_CONTINUATION
  WEAK_SAME_TEX
  LAYOUT_SCOPE_MISMATCH
  FLOAT_PROXY_ENDPOINT
  CAPTION_ENDPOINT
  REFERENCE_ENDPOINT
  HARD_NEGATIVE
  MASKED_UNKNOWN

label_strength:
  strong
  weak
  masked
  hard_negative
  exempt

proposed_loss_weight:
  strong = 1.0
  weak = 0.2
  masked/exempt = 0.0
  hard_negative = 1.0
```

Current MERGE policy direction:

```text
BODY_TEXT / LIST:
  strong MERGE only when same TeX node, source span is close, layout scope is
  compatible, and visual continuation passes.

REFERENCE:
  separate channel; do not share the low BODY_TEXT threshold.

FORMULA / FLOAT / CAPTION:
  weak or masked unless the relation is explicitly caption/float/reference
  specific.

LAYOUT_SCOPE_MISMATCH:
  hard negative.

missing candidate:
  only add extremely narrow forward body/list continuation candidates.
```

PARENT_CHILD is not being redefined in this pass. Production E2E section scope
is still governed primarily by the deterministic heading stack; parent labels
remain useful for edge-level learning and future controlled override
experiments.

Important guards:

```text
text and display equation do not MERGE just because they share a list item
table/figure/image bodies are weakly aligned, not ordinary text
float proxies do not MERGE with body text
references are preserved as reference items where possible
author biography / backmatter is excluded from unsafe MERGE candidates
footnote / margin-note layers are separated from body parent chains
candidate_edge_recall must meet the configured gate
```

Graph and label generation must be rebuilt together whenever the
`GNNViewAdapter` policy or raw edge schema changes. MinerU does not need to be
rerun unless OCR, bbox, or v7 fact extraction itself changes.

## Quality Gates

The labeler can reject samples for:

```text
too high effective orphan ratio
too high unmapped TeX ratio
too many isolated nodes
missing non-NONE edges
candidate-edge recall below threshold
layout-breaking TeX constructs
invalid graph tensor schema
```

Reports are written through the batch scripts as JSON/JSONL error logs and alignment mappings.

Current MERGE audit entrypoints:

```text
tools/audit/channel_aware_merge_label_audit.py
tools/audit/audit_missing_below_threshold_merge.py
tools/audit/family_specific_merge_calibration.py
tools/audit/probe_merge_visibility.py
```

## Current Entrypoints

Single graph:

```text
scripts/pipeline/step3_label_graph.py
```

Batch relabel:

```text
scripts/pipeline/relabel_manifest.py
scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

Full staged data production:

```text
scripts/pipeline/build_v7_dataset_staged.py
```
