# Local Configuration And Secret Handling

**Last updated**: 2026-04-29

This repository should only store source code, lightweight configuration, documentation, and reproducibility manifests. Do not commit API keys, SSH passwords, Kaggle credentials, downloaded paper corpora, model checkpoints, or generated training artifacts.

## GitHub Remote

Primary source repository:

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

Use GitHub for code and docs synchronization only. Large runtime artifacts should stay outside Git.

## Kaggle Credentials

Current checked state:

```text
local:  ~/.kaggle/access_token configured, mode 600
AutoDL: /root/.kaggle/access_token configured, mode 600
```

Kaggle CLI 2.1+ supports token-only authentication. Prefer this project convention:

Local machine:

```bash
mkdir -p ~/.kaggle
chmod 700 ~/.kaggle
cp /path/to/access_token ~/.kaggle/access_token
chmod 600 ~/.kaggle/access_token
```

AutoDL server:

```bash
mkdir -p /root/.kaggle
chmod 700 /root/.kaggle
cp /path/to/access_token /root/.kaggle/access_token
chmod 600 /root/.kaggle/access_token
```

The CLI automatically checks `~/.kaggle/access_token`. Scripts can also point to the file explicitly:

```bash
export KAGGLE_API_TOKEN="$HOME/.kaggle/access_token"
```

Legacy Kaggle credentials may still use `~/.kaggle/kaggle.json` with `username` and `key`, but this project should prefer the token file unless there is a compatibility issue.

Never commit `access_token`, `kaggle.json`, or any `.env.*` file. The repo `.gitignore` excludes them.

Local CLI note:

```text
local Kaggle CLI: .venv_kaggle/bin/kaggle
AutoDL Kaggle CLI: conda env pdf2latex, kaggle 2.1.0
```

Authentication smoke test used:

```bash
kaggle datasets list -s arxiv -p 1
```

## AutoDL Connection

Do not store the AutoDL SSH password in committed files. If automation is needed, keep credentials in an ignored local file such as `.env.local` or in a password manager.

Safe fields to record in a private ignored file:

```bash
AUTODL_HOST=connect.bjb2.seetacloud.com
AUTODL_PORT=26034
AUTODL_USER=root
AUTODL_PROJECT=/root/autodl-tmp/pdf2latex_nn
KAGGLE_API_TOKEN=/root/.kaggle/access_token
KAGGLE_CONFIG_DIR=/root/.kaggle
```

Keep passwords, tokens, and keys out of Git.

## Data And Artifact Policy

Commit these:

```text
source_code/
scripts/
docs/
requirements*.txt
config*.yaml
README.md
small manifests and sample id lists
```

Do not commit these:

```text
data/
artifacts/
models/
outputs/
logs/
.model_cache/
.venv*/
bulk arXiv PDFs or sources
MinerU outputs
feature caches
checkpoints
generated QA reports
```

For large datasets, commit only manifests with enough information to reproduce them:

```text
dataset name
source
download command
sample ids
file counts
hashes where useful
generation timestamp
pipeline commit hash
```

## Recommended Sync Rule

Use this direction for code:

```text
local edits -> GitHub -> AutoDL git pull
```

Use this direction for large runtime data:

```text
download/generate on AutoDL -> write manifest -> commit manifest only
```

Do not use blind recursive overwrite between local and AutoDL project directories.
