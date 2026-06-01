# WSL Migration Preparation Notes

This is not a migration package. It records conventions for a future migration.

## Recommended Shape

Source:

```text
/home/<user>/projects/pdf2latex-observable-reconstruction
```

Data and outputs:

```text
/mnt/d/pdf2latex_data
/mnt/d/pdf2latex_outputs
```

Keep source and heavy runtime separate. Do not copy raw PDFs, MinerU outputs,
checkpoints, generated PDFs, full compile logs, or selected2000 per-document
outputs into Git.

## Environment Variables

```bash
export PDF2LATEX_PROJECT_ROOT=/home/<user>/projects/pdf2latex-observable-reconstruction
export PDF2LATEX_DATA_ROOT=/mnt/d/pdf2latex_data
export PDF2LATEX_OUTPUT_ROOT=/mnt/d/pdf2latex_outputs
export PDF2LATEX_CONFIG=/home/<user>/projects/pdf2latex-observable-reconstruction/config/paths.local.yaml
```

## Windows Path Notes

- Prefer English-only paths.
- Avoid Chinese username/profile paths for source and heavy data.
- Prefer WSL ext4 or a data drive such as `/mnt/d` for large small-file
  workloads.
- Avoid putting high-volume IO under `/mnt/c` unless there is a specific reason.

## Paper Module Rule

When migrating, move the source repo first. Restore paper workspaces and runtime
backups separately. A future paper module should point to data through path
configuration rather than hard-coded local paths.
