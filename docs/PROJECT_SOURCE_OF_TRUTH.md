# Project Source Of Truth

**Last updated**: 2026-04-29

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
data/03_tex_sources/
data/04_ground_truth_ir/
data/05_observed_ir/
data/06_graph_features/
data/07_predicted_ir/
data/08_output_latex/
data/09_eval_reports/
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
source_code/
scripts/
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
