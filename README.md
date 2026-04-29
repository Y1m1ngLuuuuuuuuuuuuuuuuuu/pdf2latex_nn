# PDF2LaTeX NN

**Last updated**: 2026-04-27  
**Current focus**: research-paper PDF structure recovery for LaTeX/IR reconstruction.

This project investigates a structure-aware PDF-to-LaTeX pipeline for born-digital research papers. The core target is not generic OCR. The target is to recover document structure: reading order, section hierarchy, formula/table/caption placement, and parent-child relations that can later drive LaTeX generation.

## Current Position

The main working direction is:

```text
arXiv source + compiled PDF
  -> SyncTeX / OCR / layout nodes
  -> structural labels and block-level features
  -> GNN parent/structure prediction
  -> predicted IR
  -> LaTeX reconstruction
```

The project has moved past the earliest smoke-test stage. The current bottleneck is **block-level layout consolidation quality**, not raw model capacity. Recent debugging found that many bad visualizations came from upstream block boxes before GNN inference:

- cross-column paragraph boxes,
- empty paragraph shadow boxes,
- section/header shadow fragments,
- formula-adjacent fragmentation,
- over-merged paragraph blocks.

The latest code filters empty paragraph nodes, infers per-page column splits, tightens paragraph merging, and makes section merging content-aware. This improves the worst cross-column and empty-box failures, but formula/heading fragmentation still needs more work before full regeneration and retraining.

## Code Size

Maintained project code, excluding `data/`, logs, reports, backups, archives, model caches, and generated artifacts:

| Scope | Files | Lines | Size |
|---|---:|---:|---:|
| `source_code/` + `scripts/` + root configs/scripts | 63 | 19,215 | 647.1 KiB |
| Python only | 44 | 17,298 | 593.7 KiB |
| Shell scripts | 11 | 1,190 | 39.6 KiB |
| YAML configs | 5 | 579 | 10.7 KiB |

The whole local workspace is about `1.8G`, mostly because it contains data, backups, logs, reports, and experiment artifacts.

## Active Documents

- [docs/PROJECT_STATUS_REPORT.md](docs/PROJECT_STATUS_REPORT.md): current local/remote state and known blockers.
- [docs/NEXT_STEPS_PLAN.md](docs/NEXT_STEPS_PLAN.md): recommended execution order from here.
- [docs/BLOCK_CONSOLIDATION.md](docs/BLOCK_CONSOLIDATION.md): current block consolidation diagnosis and fixes.
- [docs/PAPER_SCOPE_2026_04_26.md](docs/PAPER_SCOPE_2026_04_26.md): paper scope and research framing.
- [docs/IR_SCHEMA.md](docs/IR_SCHEMA.md): IR schema reference.
- [docs/DATASET_DOCUMENTATION.md](docs/DATASET_DOCUMENTATION.md): dataset and supervision format notes.
- [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md): repository layout.

Historical and server-cache material is left under `archive/` and `server_cache_backup/` for provenance, but it is no longer treated as current guidance.

## Important Paths

Local project root:

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

Remote AutoDL project root:

```text
/root/autodl-tmp/pdf2latex_nn
```

Main source directories:

```text
source_code/     core pipeline and model code
scripts/         operational and QA scripts
docs/            current maintained documentation
data/            generated/runtime data
logs*/           generated logs
reports*/        generated qualitative reports
```

## Current Recommendation

Do not retrain yet. First finish block-level QA:

1. Fix formula-adjacent fragmentation and header shadow-node issues.
2. Re-run a 50-sample block consolidation QA batch.
3. Inspect worst-case overlays, not only aggregate metrics.
4. Only after the overlays are acceptable, regenerate all 947 enhanced block samples.
5. Then retrain the block-level GNN on the regenerated features.
