# PRCV Evidence Registry

Last updated: 2026-06-01.

This registry belongs to the submitted PRCV 2026 paper module. It is not the
registry for the whole PDF2LaTeX project and should not be reused for future
papers except as a template.

## Canonical Evidence

Source-repo evidence starts at:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

Paper-facing evidence starts at:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh/paper_assets/00_FINAL_EVIDENCE_20260531/
```

AutoDL summary evidence starts at:

```text
/root/autodl-tmp/pdf2latex_nn/data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
```

## Evidence Hierarchy

1. **Primary large-scale direct-parser comparison**:
   selected2000, n=1972 fair intersection, Ours / ContentList Direct /
   MinerU Direct.
2. **Controlled external MMD/Markdown comparison**:
   selected200, same denominator, Ours / ContentList Direct / MinerU Direct /
   Nougat.
3. **Large-scale usability**:
   Ours selected2000 generated 2000/2000, compiled 1852/2000, converted
   1999/2000, and obtained structure metrics for 1980/2000.
4. **Diagnostics and explanation**:
   float-caption sidecar, attribution audit, compile-safety history, renderer
   sprint reports, and ablations are explanatory unless explicitly promoted in
   the canonical index.

## Paper Module Boundary

The PRCV module exercises these project interfaces:

- parser adapters for MinerU artifacts
- Observable Fact Layer
- DocumentIR and RenderTreeIR
- compile-safe role renderers
- ComparisonStructure evaluation
- paper-specific evidence registry

The PRCV module does not own future table-cell, front-matter entity-linking,
Nougat selected2000, or GNN relation-learning claims.

## What Not to Claim

- Nougat selected2000 completion.
- selected2000 four-method comparison.
- selected2000 Ours metrics 2000/2000.
- selected2000 Ours comparison conversion 2000/2000.
- compile or visual QA for parser-output or MMD baselines.
- source-level TeX AST recovery.
- solved semantic table-cell reconstruction.
- enabled broad Algorithm renderer.
- solved broad float-caption resolver.
- GNN/Y-network as the PRCV main contribution.

## Runtime Backups

The PRCV material backup is separate from GitHub source:

- clean PRCV export package/folder
- full8000 raw/MinerU package for future reuse
- extra Nougat runtime package for future Nougat work

Those packages are netdisk/runtime material, not paper claims by themselves.
