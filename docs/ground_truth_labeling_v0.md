# Ground Truth Labeling v0

**Last updated**: 2026-05-14

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

Important guards:

```text
text and display equation do not MERGE just because they share a list item
table/figure/image bodies are weakly aligned, not ordinary text
references are preserved as reference items where possible
author biography / backmatter is excluded from unsafe MERGE candidates
footnote / margin-note layers are separated from body parent chains
candidate_edge_recall must meet the configured gate
```

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
