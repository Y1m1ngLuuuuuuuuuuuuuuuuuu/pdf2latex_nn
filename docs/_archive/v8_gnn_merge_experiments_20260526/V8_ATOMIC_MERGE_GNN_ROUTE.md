# V8 Atomic Merge GNN Route

**Last updated**: 2026-05-25

This document defines the optional middle-derived GNN learning route for
paragraph/list/reference continuation. It does **not** replace the v8 mainline
renderer in `docs/V8_MAINLINE_RECONSTRUCTION_PATH.md`.

## Purpose

The old v7/content-list graph operated on logical blocks that MinerU may have
already merged before reading order was corrected.  The v8 atomic route starts
earlier:

```text
MinerU middle.json
  -> v8 reading-order repair
  -> atomic line/span fragments
  -> continuation candidate edges
  -> optional label sidecar
  -> future MERGE model
  -> projection back to v8 logical paragraph/list/reference owners
```

The target task is local continuation only.  PARENT/heading hierarchy remains
owned by the v8 heading style stack.

## Canonical Converter

```bash
python3 tools/v8_atomic/build_v8_atomic_merge_json.py \
  --doc-id <doc_id> \
  --middle-json <path/to/*_middle.json> \
  --content-list-json <path/to/*_content_list.json> \
  --style-content-list-json <path/to/*_content_list_v7_styles.json> \
  --source-tex <optional/source.tex> \
  --output-dir data/09_eval_reports/v8_atomic_merge_<YYYYMMDD>/<doc_id>_<run_tag> \
  --candidate-window 4
```

The converter writes JSON only. It does not train, does not create PyG tensors,
does not run MinerU, and does not change v8 reconstruction outputs.

Current JSON contract:

```text
v8_atomic_merge_graph_view_v1_2
```

## Output Files

```text
<doc_id>_v8_atomic_merge_graph_view.json
<doc_id>_v8_atomic_nodes.json
<doc_id>_v8_atomic_candidate_edges.json
<doc_id>_v8_atomic_merge_labels.json
<doc_id>_v8_atomic_merge_record.json
```

Use:

```text
data/09_eval_reports/v8_atomic_merge_<YYYYMMDD>/<doc_id>_<run_tag>/
```

for smoke/debug outputs, and use a future `data/06_graph_features/v8_atomic_merge_<tag>/`
only after the JSON contract is stable.

## JSON Separation

The route intentionally separates frontend features from truth:

| JSON | Contains | Must Not Contain |
| --- | --- | --- |
| `*_v8_atomic_nodes.json` | atomic text, bbox, page, column, source middle ids, style sidecar ids | MERGE labels |
| `*_v8_atomic_candidate_edges.json` | forward candidate pairs, geometry/text features, skipped barrier summaries, SciBERT pair text | gold labels |
| `*_v8_atomic_merge_labels.json` | sidecar labels, label source, confidence, train mask, proposed weight | model input tensors |

This keeps the future GNN route honest: "should merge" never appears as an
input feature.

## Atomic Nodes

Each node is one middle-derived fragment, usually one middle line/span.  It
keeps:

```text
atomic_id
text
channel
raw_type
page_idx
bbox
page_size
reading_order
source_middle_block_id
source_line_id
column_id
is_full_width
source_content_list_index optional
style_content_list_index optional
```

Current channels:

```text
BODY_TEXT
LIST_ITEM
REFERENCE_ITEM
HEADING
CAPTION
DISPLAY_MATH
FLOAT_PROXY
FRONT_MATTER
PAGE_FURNITURE
UNKNOWN
```

## Candidate Edges

Candidate edges are forward-only and local.  They include text and geometry
features useful for continuation:

```text
src_tail
dst_head
pair_text = src_tail [SEP] dst_head
same_page
same_column
layout_scope
vertical_gap
x_overlap_ratio
skipped_channels
src_open_ended
src_hyphen_ended
dst_lowercase_start
dst_continuation_word_start
skipped_float_count
skipped_formula_count
skipped_barrier_count
```

`pair_text` is the intended SciBERT/BERT edge input.  It uses the previous
fragment tail and next fragment head instead of embedding each tiny line in
isolation.

## Label Sidecar

Labels can come from two sources:

1. Frontend deterministic evidence:
   - adjacent lines inside the same middle block;
   - v8 deterministic continuation merges.
2. Backend TeX weak supervision:
   - atomic fragments aligned to the same source paragraph/list/reference item;
   - fragments aligned to different source paragraphs as hard negatives.

Labels are stored as:

```text
label: MERGE | NONE | UNKNOWN
train_mask: true | false
label_strength: strong | weak | masked | hard_negative
proposed_loss_weight
label_source
confidence
```

TeX same-paragraph evidence is treated conservatively: it becomes strong only
when the edge also has local continuation evidence; otherwise it is weak.

## What This Route Does Not Do

- It does not change `content_list_v8.json`.
- It does not alter the default v8 renderer.
- It does not predict or label PARENT.
- It does not use figure/table/algorithm proxy nodes as body MERGE endpoints.
- It does not use source TeX at inference time.

## Learned Overlay Policy

The default reconstruction path remains deterministic v8 merge.  A learned
atomic MERGE model is only allowed to act as an overlay:

```text
v8 deterministic merge decisions
  + learned_merge_overlay candidates
  -> projected v8 owner-level merge sidecar
  -> normal v8 renderer
```

The overlay must be high precision.  It may only add local continuation edges
that satisfy all of these constraints:

```text
BODY_TEXT or LIST_ITEM endpoints
forward reading order
same or compatible layout scope
open sentence or hyphenated source tail
lowercase, parenthetical, or continuation-like destination head
no heading/math/front matter/reference/caption barrier between endpoints
no figure/table/algorithm proxy endpoint
```

PARENT/section scope is not part of this route.  It remains owned by the v8
style registry and heading stack skeleton.

### Edge Feature Direction

The graph feature family after 2026-05-25 is:

```text
v8_atomic_merge_graph_v1_4
```

It adds edge-level relative geometry and local continuation cues:

```text
relative bbox deltas
x/y overlap
vertical gap by line height
font-size ratio and delta
bold-state transition
indent continuity
tail/head text cues
hard terminal / soft continuation punctuation
common abbreviation tail guard
citation-closed tail cue
prev2 / next2 local channel context
between-node barrier counts
float/formula/table/code skip pattern
```

The v1.4 schema extends v1.3 with more non-owner cues:

```text
unclosed parenthesis/bracket/quote
tail after math symbol
tail/head stopword cues
punctuation or closing-bracket head
conjunction/preposition head
column transition class
near column bottom / near page top
same-column flow lane
wide/full-width skipped float counts
skipped max float width ratio
caption/math-between skip flags
adjacent paragraph rhythm gaps
candidate gap vs neighboring gap ratios
```

This feature schema is separate from the older selected200 checkpoints.  To use
these v1.4 features for training, rebuild the v8 atomic graph JSON/PyG family
and train a matching checkpoint.  Do not mix a v1.1 checkpoint with a v1.2 graph,
a v1.2 checkpoint with a v1.3 graph, or a v1.3 checkpoint with a v1.4 graph.

The punctuation features are deliberately soft features, not hard labels:

```text
hard terminal: . ? ! 。 ？ ！
soft continuation: , ; : ， ； ：
hyphenated: -
abbrev-like: et al. / Fig. / Sec. / Eq. / e.g. / i.e. / etc.
```

This lets the model learn that a true hard terminal is usually anti-MERGE, while
soft punctuation, hyphenation, and abbreviation tails can still indicate
continuation.

## Selected200 Projection Results

The first selected200 projection with a permissive overlay showed that learned
MERGE can reach the renderer, but it was not safe enough as a default path:

```text
output:
data/09_eval_reports/v8_atomic_merge_20260524/
  projection_eval_selected200_light8_body_list_focus_all_features/

doc_count: 200
generated_tex_changed_count: 169
model_added_owner_merge_total: 1398
mean_deterministic_missing_merge_rate: 0.108314
mean_learned_missing_merge_rate: 0.103626
mean_deterministic_wrong_merge_rate: 0.023316
mean_learned_wrong_merge_rate: 0.024874
```

Interpretation:

```text
learned overlay reduced missing merge slightly
but increased wrong merge
so it cannot replace deterministic v8 merge
```

The strict overlay smoke20 is more conservative:

```text
output:
data/09_eval_reports/v8_atomic_merge_20260524/
  projection_eval_selected200_strict_overlay_v1_smoke20/

doc_count: 20
generated_tex_changed_count: 14
model_added_owner_merge_total: 87
mean_deterministic_missing_merge_rate: 0.060026
mean_learned_missing_merge_rate: 0.060392
mean_deterministic_wrong_merge_rate: 0.019990
mean_learned_wrong_merge_rate: 0.021826
mean_deterministic_source_coverage_rate: 0.784359
mean_learned_source_coverage_rate: 0.778881
```

Interpretation:

```text
strict overlay is safer than the permissive branch
but still does not improve deterministic v8 on smoke20
```

Current decision:

```text
v8 deterministic merge remains the mainline.
GNN MERGE stays as learned_merge_overlay / research branch.
Do not enable it by default until a future v1.2 retrain improves both
missing merge and wrong merge against deterministic v8.
```

## 00050 Smoke

Current smoke output:

```text
data/09_eval_reports/v8_atomic_merge_20260524/2501.00050_json_contract_smoke/
```

Observed 00050 summary:

```text
atomic nodes: 634
candidate edges: 2346
deterministic v8 merges: 7
channels:
  BODY_TEXT 530
  DISPLAY_MATH 16
  FLOAT_PROXY 5
  FRONT_MATTER 1
  HEADING 23
  LIST_ITEM 21
  REFERENCE_ITEM 38
```

This is only a JSON-contract smoke. It is not a training result.
