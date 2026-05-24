# Front Matter Entity Model Plan

**Last updated**: 2026-05-24

This note records the planned path from the current deterministic front-matter
preservation logic to precise author / affiliation / email parsing.

## Current Phase 0

The current v8 path implements deterministic front-matter preservation:

```text
DocumentIR
  -> FrontMatterLineBuilder
  -> RuleBasedFrontMatterSequenceTagger
  -> FrontMatterIR
  -> original-like front-matter renderer
```

Phase 0 identifies and preserves:

```text
document_title
author_block
affiliation-like lines
email / correspondence lines
front_note
abstract_title
abstract_body
body_start
```

Its purpose is conservative:

```text
do not drop visible front matter
do not duplicate front matter in body text
do not let author/affiliation/email become body headings
render a stable original-like title/author/abstract surface
```

Phase 0 does **not** solve precise author-affiliation-email linking.

## Target Phase 1

Phase 1 should introduce a lightweight front-matter entity recognizer.  This is
separate from the GNN relation model and separate from the body reconstruction
stack.

Inputs:

```text
front-page line/spans
text
bbox
font size
bold / italic
centeredness
line order
visual row / column grouping
superscript markers
local context before/after line
```

Predicted roles:

```text
TITLE
AUTHOR
AFFILIATION
EMAIL
ORCID
NOTE
ABSTRACT_TITLE
ABSTRACT_BODY
BODY
OTHER
```

Recommended model shape:

```text
small Transformer / BiLSTM-style line tagger
+ explicit layout features
+ weak labels from TeX author macros
```

Do not put this model into the main GNN graph.  It is a metadata/layout parser
that runs on the full document fact layer before body heading decoding.

## Target Phase 2

Phase 2 should build a link graph:

```text
AUTHOR -> AFFILIATION
AUTHOR -> EMAIL
```

Start with deterministic linking:

```text
superscript markers
author order
affiliation order
email prefixes
same visual column / row group
corresponding-author symbols
```

Only train a linker if the deterministic linker becomes the bottleneck.

## FrontMatterIR Extension

The future precise IR should move from text blocks to entities:

```json
{
  "authors": [
    {
      "name": "Alice Wang",
      "markers": ["1"],
      "source_line_ids": [],
      "source_v7_ids": [],
      "email_refs": ["alice@example.edu"],
      "affiliation_refs": ["aff_1"],
      "confidence": 0.92
    }
  ],
  "affiliations": [
    {
      "affiliation_id": "aff_1",
      "markers": ["1"],
      "text": "Department of Computer Science, Example University",
      "source_line_ids": [],
      "confidence": 0.88
    }
  ],
  "emails": [
    {
      "email": "alice@example.edu",
      "owner_hint": "Alice Wang",
      "source_line_ids": [],
      "confidence": 0.96
    }
  ]
}
```

## Renderer Policy

The renderer should support three modes:

```text
original_like:
  preserve visual front-matter layout with centered/minipage surfaces

semantic_latex:
  render clean \title / \author / abstract blocks

template_aware:
  render ACM / IEEE / AAAI style if a template contract is selected
```

The current default remains `original_like`.

## Evaluation

Precise front-matter parsing should be evaluated separately from body structure:

```text
title text F1
author name F1
affiliation text F1
email F1
author-affiliation link accuracy
author-email link accuracy
front matter text preservation
false body-heading promotion count
```

Do not mix these metrics into `section_attachment_body_no_float_f1`.

## Non-Goals

```text
Do not use an API/VLM as the production parser.
Do not mutate v7 JSON in Phase 1.
Do not add front-matter entities to the GNN graph.
Do not use author-affiliation linking to influence body heading scope.
```
