# Project Source Of Truth

**Last updated**: 2026-05-07

This project should be managed as a source repository plus external runtime artifacts. The source repository is for code and reproducibility metadata. AutoDL is for data, feature caches, training, and generated outputs.

## Canonical Repository

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

The intended code flow is:

```text
local source edits -> GitHub -> AutoDL git pull
```

Avoid blind local-to-server recursive overwrites. If a direct sync is needed during recovery, use `upload_to_server.sh`, which only syncs lightweight source paths.

## Local Machine

Local root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

Responsibilities:

```text
code editing
documentation
small syntax checks
manifest inspection
GitHub commits and review
```

Do not run heavy training locally. The local Python/conda environment is only for code sanity checks.

Active source skeleton:

```text
src/
configs/
scripts/pipeline/
tests/
docs/
data/00_manifests/
data/01_raw_pdfs/
data/02_mineru_outputs/
data/03_tex_source_pool/
data/04_ground_truth_ir/
data/05_observed_ir/
data/06_graph_features_v7/
data/07_predicted_ir/
data/08_output_latex/
data/09_eval_reports/
```

Current maintained docs:

```text
docs/PROJECT_SOURCE_OF_TRUTH.md      repository / AutoDL boundary
docs/feature_schema_v0.md            PDF IR and tensor feature contract
docs/ground_truth_labeling_v0.md     TeX-to-PDF alignment and GNN label contract
docs/LOCAL_CONFIGURATION.md          local private configuration notes
```

Legacy reference material is stored outside the active skeleton:

```text
_legacy_reference/2026_04_29_pre_rebuild/
```

This directory is for lookup and migration only. It is ignored by Git and should not be committed.

## AutoDL Server

Remote project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Responsibilities:

```text
large downloads
PDF/source extraction
MinerU processing
feature generation
training
large QA batches
checkpoints and runtime artifacts
```

AutoDL has Miniconda and the working conda environment is:

```text
pdf2latex
```

## Secrets

Secrets stay outside Git.

Current private files:

```text
local:  /Users/lu/.kaggle/access_token
local:  .env.local
AutoDL: /root/.kaggle/access_token
AutoDL: /root/autodl-tmp/pdf2latex_nn/.env.autodl
```

The committed `.env.example` records variable names only.

## Commit To Git

Commit:

```text
README.md
目标.md
src/
scripts/
tools/
tests/
docs/
requirements.txt
requirements_server.txt
verify_environment.py
config*.yaml
.env.example
.gitignore
small manifests and sample id lists
```

Do not commit:

```text
data/
artifacts/
archive/
server_cache_backup/
paper_artifacts/
reports*/
demo_*/
logs*/
.model_cache/
.venv*/
_legacy_reference/
checkpoints
bulk PDFs
bulk arXiv sources
MinerU outputs
feature caches
private env files
Kaggle tokens
SSH passwords
```

## Data Rebuild Rule

Every dataset or experiment batch should have a manifest:

```text
dataset name
source
download command
sample ids
file counts
hashes where useful
generation timestamp
code commit hash
output root
```

The manifest can be committed. The bulk data should stay on AutoDL or external storage.

## Current Pipeline Contracts

The current PDF-side production JSON format is:

```text
*_content_list_v7.json
*_content_list_v7_styles.json
```

v7 keeps MinerU block granularity and raw bbox coordinates. It only adds list-marker metadata, column-reading-order repair, reference item preservation, and PyMuPDF style spans. Cross-paragraph and cross-page text merging should happen in decoder/generator logic, not in the v7 preprocessing JSON.

All graph-producing and graph-consuming scripts must use the v7 contract:

```text
graph_schema_version = graph_v7
pipeline_version = v7
source_path = *_content_list_v7_styles.json
```

Old v3/v4/v5 JSONs and old `data/06_graph_features/` graph caches are legacy diagnostics only. They must not be used for feature extraction, training, inference, or generator QA.

The current graph supervision target is a three-class edge label:

```text
MERGE = 0
PARENT_CHILD = 1
NONE = 2
```

`SIBLING` is intentionally removed. Sibling order is recovered from v7 reading order and renderer sorting, not from GNN labels.

Candidate edge construction is recall-first. The default graph builder uses:

```text
sequential_window = 15
spatial_k = 3
long_sight_window = 40
scope_anchor_window = 80
float_skip_window = 40
```

Before training on a rebuilt graph batch, run `tools/profile_candidate_edge_recall.py` on representative labeled samples. If any MERGE or PARENT_CHILD oracle edge is absent from `edge_index`, training on that sample is invalid because the GNN can never predict a label for an edge it was not given.

The current automatic truth-labeling contract is documented in:

```text
docs/ground_truth_labeling_v0.md
```

Batch labeling must use strict quality gates for training data. Samples with high PDF orphan ratio, high unmapped TeX ratio, missing core section structure, or excessive isolated nodes should be skipped rather than saved as dirty `.pt` files.

Orphan accounting is not raw node accounting. Expected visual-only artifacts such as page headers, footers, page numbers, and very short edge noise are exempt from orphan/isolated ratios. Successfully matched pre-section metadata such as title, authors, abstract, and keywords is treated as scoped under a virtual `DOCUMENT_ROOT` for quality checks and TreeDecoder rendering, while the stored graph node count remains unchanged.

## Current Cleanup Decision

The working tree currently contains many tracked deletions from previous cleanup/deletion attempts. Do not stage them automatically.

Before the first clean GitHub push, choose explicitly:

```text
keep old files as ignored legacy reference material, then remove them from the tracked source tree in a deliberate cleanup commit
```

or:

```text
restore selected old source files from Git into the new src/ skeleton after review
```

The safer default is to keep old material in `_legacy_reference/` and migrate only specific functions after reviewing them.
