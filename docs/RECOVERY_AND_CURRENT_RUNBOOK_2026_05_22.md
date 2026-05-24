# Recovery And Current Runbook 2026-05-22

**Last updated**: 2026-05-24

This note records the current recovery state, remote rebuild path, and latest
MERGE-label decisions so the project does not depend on chat history.

## Current Repository State

Local source root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

Remote runtime root:

```text
/root/autodl-tmp/pdf2latex_nn
```

GitHub remote:

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

The current pushed commit changes over time.  Check it with:

```bash
git log --oneline -5
```

The AutoDL project directory was reconstructed from a local Git bundle because
the private GitHub remote required authentication on the machine.  The remote
code should be treated as source-controlled code only; large runtime artifacts
must be regenerated or copied from explicit backups.

## Remote Assets That Remain

Verified remote assets:

```text
/root/miniconda3/envs/pdf2latex
/root/autodl-tmp/envs/mineru
/root/autodl-tmp/MinerU
/root/.cache/torch/hub/nougat-0.1.0-small
```

Important dependency checks:

```text
pdf2latex env:
  Python 3.12.3
  torch 2.8.0+cu128
  torch_geometric 2.6.1
  transformers 4.46.3
  fitz 1.27.2.3
  pytest 9.0.3

mineru env:
  Python 3.12.13
  MinerU import works
  torch 2.11.0+cu130
  transformers 4.57.6

Nougat:
  /root/.cache/torch/hub/nougat-0.1.0-small
  contains config/tokenizer/pytorch_model.bin
```

Deleted runtime artifacts are not assumed to be recoverable unless an external
AutoDL/cloud snapshot is restored.  Do not cite deleted remote reports unless
they also exist locally or in Git-tracked documentation.

## Current Data Rebuild Run

We are rebuilding a fresh TeX-source dataset from arXiv e-print sources.

Important distinction:

```text
We do not download arXiv-hosted original PDFs.
We download arXiv e-print TeX sources, compile them locally, and keep:
  data/03_tex_sources/<doc_id>/
  data/01_raw_pdfs/<doc_id>.pdf   # our compiled PDF, not the arXiv PDF
```

The Kaggle metadata path was abandoned for this run because downloading the
full `arxiv-metadata-oai-snapshot.json` was slow and unnecessary for immediate
rebuild.  The arXiv API path was also abandoned after HTTP 429 rate limiting.
The active run uses deterministic ID-scan candidates.

Active remote run:

```text
run_name: arxiv2025_compilable_tex8000_idscan_20260522
candidate_manifest: data/00_manifests/arxiv_2025_idscan_candidates_360000.jsonl
target_successes: 8000
candidate_limit: 360000
year: 2025
download_slots: 16
compile_slots: 4
download_backlog: 96
compile_backlog: 64
```

Runner and logs:

```text
logs/arxiv2025_compilable_tex8000_idscan_20260522_runner.sh
logs/arxiv2025_compilable_tex8000_idscan_20260522.pid
logs/arxiv2025_compilable_tex8000_idscan_20260522.log
data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/progress.json
data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/accepted.jsonl
data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/rejected.jsonl
data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/download_errors.jsonl
```

Monitoring:

```bash
cd /root/autodl-tmp/pdf2latex_nn

cat data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/progress.json

tail -f logs/arxiv2025_compilable_tex8000_idscan_20260522.log

ps -eo pid,etime,pcpu,pmem,cmd \
  | grep -E 'arxiv2025_compilable|step0_build_compilable' \
  | grep -v grep

wc -l data/09_eval_reports/arxiv2025_compilable_tex8000_idscan_20260522/accepted.jsonl

find data/03_tex_sources -mindepth 1 -maxdepth 1 -type d | wc -l
find data/01_raw_pdfs -type f -name '*.pdf' | wc -l
```

Progress fields:

```text
accepted_total      accepted compile-success samples
target_successes    target accepted samples
attempted_submitted candidates submitted into the staged pipeline
download_completed  candidates that finished download/unpack/main-tex detection
download_pending    active/pending download backlog
ready_to_compile    downloaded candidates waiting for compile queue
compile_submitted   candidates submitted to compile queue
compile_completed   candidates whose compile stage finished
compile_pending     active/pending compile tasks
status_counts       final state counts such as accepted, compile_failed,
                    source_not_found, error
```

## Current End-To-End Strategy

The project target remains:

```text
layout-aware, block-structure-preserving, compilable LaTeX reconstruction

As of 2026-05-24, the default reconstruction strategy has pivoted from
GNN-centered v7 E2E to v8 layout-aware reconstruction:

```text
MinerU middle/content-list outputs
  -> v8 reflow and reading-order repair
  -> FrontMatterIR
  -> heading style registry + stack skeleton
  -> RenderTreeIR
  -> OriginalLikeIRLatexRenderer
```

The GNN path remains available for optional relation-learning experiments and
historical ablations, but it should not be described as the default generator
dependency.
from rendered scientific PDFs
```

The production rendering path remains:

```text
full v7 fact layer
  -> GNN view for relation logits only
  -> constrained decoder / heading stack / float grouping
  -> RenderTreeIR
  -> full-v7-first IR renderer
```

Do not render directly from the GNN view.  Do not use TeX source at inference
time.  Do not weaken the heading stack just to make GNN parent edges appear more
important.

## Current GNN Interpretation

Current evidence says:

```text
PARENT_CHILD:
  useful as a hint/shadow signal;
  global section scope is dominated by deterministic heading stack.

MERGE:
  the most important place to inspect GNN value;
  accepted MERGE can enter RenderTreeIR and affect generated text;
  the main open question is which low-score / same-TeX candidates are safe.
```

Rules-only relation-source baselines must remain formal baselines.  If a
rules-only system performs close to the GNN system on easy documents, the paper
must report that honestly and evaluate GNN contribution on GNN-sensitive
hardsets.

## Channel-Aware MERGE Label Direction

The labeler should not be treated as a pure TeX-AST projector.  It should be a
PDF-first relation supervision generator:

```text
TeX alignment evidence
+ PDF visual/channel evidence
+ decoder usefulness
-> MERGE / PARENT_CHILD / NONE supervision
```

Current audit-only / branch fields:

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

Recommended policy direction:

```text
BODY_TEXT / LIST:
  allow lower MERGE threshold only under deterministic precision gates.

REFERENCE:
  do not share the same low threshold as body text;
  use a separate high threshold or reference-specific continuation.

WEAK / MASKED:
  continue masking or low-weighting.

LAYOUT_MISMATCH:
  keep as hard negative.

missing-candidate:
  only add extremely narrow body/list continuation candidates;
  do not make missing-candidate expansion the main path.
```

## Current Audit Tools

```text
tools/audit/channel_aware_merge_label_audit.py
tools/audit/audit_missing_below_threshold_merge.py
tools/audit/family_specific_merge_calibration.py
tools/audit/probe_merge_visibility.py
```

These tools are diagnostic/audit tools.  They should not silently mutate graph
labels or training data.

## Family-Aware Decoder Branch

`TreeDecoderConfig` has an experimental default-off family-aware MERGE branch:

```text
enable_family_aware_merge_policy
family_body_list_merge_threshold
family_reference_merge_threshold
enable_family_aware_missing_candidate_merge
family_missing_candidate_max_gap
family_missing_candidate_score_floor
```

This is a small experimental branch, not the main training path.  It exists to
test whether family-specific thresholds and narrow deterministic continuation
change RenderTreeIR / generated LaTeX before committing to full relabeling and
training.

Default production behavior should remain unchanged unless the flag is passed
explicitly.

## Next Safe Sequence

After the fresh 8000-source rebuild produces enough accepted samples:

```text
1. Verify accepted source/PDF counts and compile logs.
2. Run MinerU/v7 on a small batch first.
3. Build graph + labels with current code.
4. Run channel-aware MERGE audit on 100-200 docs.
5. Only then decide whether to relabel/retrain a MERGE-v2 branch.
6. Keep PARENT dominated by stack skeleton until a separate parent override
   ablation proves value.
```

Do not start full ablation immediately after the TeX-source rebuild.  First
validate that the new labels have sane channel/family breakdown and candidate
edge recall.
