# Local Configuration And Secret Handling

> Historical local setup note. The current path and repository configuration
> live in `docs/PATH_CONFIGURATION.md` and
> `docs/CANONICAL_PROJECT_PATHS_POST_SUBMISSION.md`.

**Last updated**: 2026-05-18

This document records local/private setup rules. Do not commit secrets.

## Repository

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

Local root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL root:

```text
/root/autodl-tmp/pdf2latex_nn
```

## Credentials

Never commit:

```text
AutoDL SSH password
Kaggle tokens
OpenAI/API keys
.env.local
.env.autodl
model download credentials
```

Private local files may contain:

```bash
AUTODL_HOST=connect.bjb2.seetacloud.com
AUTODL_PORT=26034
AUTODL_USER=root
AUTODL_PROJECT=/root/autodl-tmp/pdf2latex_nn
```

## Kaggle

Preferred token location:

```text
local:  ~/.kaggle/access_token
AutoDL: /root/.kaggle/access_token
```

Recommended permissions:

```bash
chmod 700 ~/.kaggle
chmod 600 ~/.kaggle/access_token
```

Smoke test:

```bash
kaggle datasets list -s arxiv -p 1
```

## AutoDL Notes

Use the network accelerator for downloads when available:

```bash
source /etc/network_turbo >/dev/null 2>&1 || true
```

Use the project conda environment for code:

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate pdf2latex
```

Large runtime data belongs under:

```text
/root/autodl-tmp/pdf2latex_nn/data
/root/autodl-tmp/pdf2latex_nn/logs
```

Do not clean checkpoints, evaluation reports, manifests, or generated PDFs
unless the user explicitly identifies the exact run as disposable.

## Current Remote Safety

Before starting new heavy jobs:

```bash
ps -eo pid,etime,pcpu,pmem,cmd | grep -E 'mineru|train_edge|relabel|rebuild|build_v7' | grep -v grep
tail -n 40 logs/<current_run>.log
df -h /root/autodl-tmp
```

Do not start ablation/training while a rebuild/relabel or MinerU batch is saturating the same resources unless intentionally scheduled.
