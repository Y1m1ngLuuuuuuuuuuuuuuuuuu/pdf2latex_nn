# Current Interface Design

Last updated: 2026-06-01.

This document names the interfaces that should remain stable as PDF2LaTeX grows
across multiple paper modules.

## Interface Map

| Interface | Primary files | Input | Output | Stability rule |
|---|---|---|---|---|
| PathConfig | `src/config/project_paths.py` | environment variables, local config, repo defaults | resolved project/data/output/report/runtime/paper roots | No hard-coded Mac/AutoDL/WSL paths in new active code. |
| DatasetManifest | `src/datasets/`, `data/00_manifests/` | doc ids, source paths, parser-output paths, gold target paths | reproducible document set | Paper modules must lock denominator and manifest source. |
| ParserAdapter | `src/adapters/` | MinerU/parser artifacts | normalized parser facts | Adapter preserves provenance and must not discard unused evidence. |
| ObservableFactLayer | `src/perception/` | parser facts and PDF geometry | observable document facts | Preserve geometry, reading order, style, formula, caption, reference, table, page-furniture, and provenance cues. |
| DocumentIR | `src/ir/`, `src/adapters/` | observable facts | structural document representation | Must be serializable and independent of a paper-specific table layout. |
| RenderTreeIR | `src/reasoning/`, `src/ir/` | DocumentIR plus reasoning modules | typed render tree | Should separate role assignment from LaTeX string emission. |
| RendererPlugin | `src/generation/ir_renderers/` | typed render nodes | LaTeX fragments | Must degrade safely and preserve compile behavior. |
| CompileEval | `src/evaluation/compile_eval.py`, `src/generation/compile_checker.py` | generated LaTeX | compile status and diagnostics | Diagnostic hardening may not redefine compile success. |
| ComparisonStructure | `src/evaluation/comparison_structure.py`, `tools/convert_*comparison*.py` | generated output or baseline output | neutral structural schema | Allows fair structural metrics across output contracts. |
| StructureMetrics | `src/evaluation/structure_metrics.py`, `tools/evaluate_comparison_structure.py` | predicted/gold ComparisonStructure | metric JSON/CSV | Metric definitions are shared; paper modules choose denominators. |
| BaselineConverter | `tools/baselines/`, `tools/api_baselines/`, `tools/convert_markdown_to_comparison.py` | parser/MMD/API output | ComparisonStructure | Baseline conversion is not LaTeX rendering unless explicitly stated. |
| EvidenceRegistry | `data/09_eval_reports/*`, `docs/*EVIDENCE*` | reports, tables, manifests | locked numbers and claim boundary | Each paper gets its own registry and do-not-claim list. |
| RuntimeBackup | AutoDL export folders, netdisk manifests | heavy runtime material | tar/checksum/restore package | Runtime packages do not enter GitHub source. |

## Paper Module Contract

Every paper module should include:

```text
paper_module/
  evidence registry
  locked denominator
  method list
  baseline list
  table sources
  compile/visual applicability
  claim boundary
  backup/restore notes
```

The module may add converters, renderers, metrics, or diagnostics, but those
changes should be expressed through the interfaces above.

## PRCV Interface Use

PRCV uses:

- `ParserAdapter` for MinerU parser artifacts.
- `ObservableFactLayer` for geometry, order, style, formula, caption, reference,
  table, front-matter, and provenance evidence.
- `DocumentIR` and `RenderTreeIR` for typed reconstruction.
- `RendererPlugin` for compile-safe formula, reference, float-caption, table,
  text, heading, list, and front-matter rendering.
- `ComparisonStructure` for neutral structural metrics.
- `EvidenceRegistry` for selected2000 and selected200 locked evidence.

PRCV does not use:

- GNN output as the paper-facing mainline renderer.
- source-level TeX AST recovery as the target.
- selected2000 Nougat as a completed comparison.

## New Interface Checklist

Before adding a new active interface:

1. Name the owner layer.
2. Define input and output artifacts.
3. State whether it is project-core, paper-module, or diagnostic-only.
4. Add a minimal test or conversion check.
5. Add documentation to this file or a dedicated contract doc.
6. Record whether runtime material belongs in Git, local paper workspace,
   AutoDL, or netdisk.

## Forbidden Couplings

Avoid:

- paper-specific denominator logic inside renderer code
- metric-definition changes hidden in paper integration scripts
- direct use of source TeX during inference
- hard-coded local/AutoDL paths in active source
- compile/visual metrics for non-LaTeX output contracts
- committing heavy runtime or paper-workspace material into the source repo
