# WSL Migration Preparation Notes

This is not a migration package. It records path and environment conventions for
future migration.

- Prefer English-only paths under WSL.
- Avoid Chinese user/profile paths for source and runtime data.
- Keep source code separate from heavy runtime data.
- Do not copy raw PDFs, MinerU outputs, checkpoints, generated PDFs, compile
  logs, or selected2000 per-document outputs into Git.
- Prefer WSL ext4 or a data drive such as `/mnt/d` for heavy IO; avoid `/mnt/c`
  for large small-file workloads.

Suggested roots:

```text
/home/<user>/projects/pdf2latex_nn
/mnt/d/pdf2latex_nn_data
/mnt/d/pdf2latex_nn_outputs
```

