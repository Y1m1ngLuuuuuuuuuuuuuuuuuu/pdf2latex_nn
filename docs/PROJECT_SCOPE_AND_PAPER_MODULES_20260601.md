# Project Scope and Paper Modules

Last updated: 2026-06-01.

PDF2LaTeX is the umbrella project. Individual papers are modules that use part
of the project and report a bounded evidence package. This distinction is
important because the project will continue after the PRCV 2026 submission.

## Whole Project

The project owns reusable interfaces and implementation layers:

```text
runtime/path config
dataset and manifest management
parser adapters
observable fact preservation
DocumentIR and RenderTreeIR
reasoning modules
compile-safe role renderers
ComparisonStructure evaluation
paper/evidence registry tooling
backup and runtime separation policy
```

The project should be able to support multiple future papers without rewriting
the core pipeline for each paper.

## Paper Module

A paper module is a bounded research claim package. It must define:

- paper name or target venue
- research question
- dataset denominator
- methods and baselines
- metrics and tables
- evidence registry
- locked numbers
- what not to claim
- backup package and restoration notes

Paper modules should depend on stable project interfaces. They should not hide
paper-specific assumptions inside parser adapters, renderers, or evaluators.

## Current Paper Module: PRCV 2026

Status: submitted.

Research target:

```text
observable-fact-guided PDF-to-LaTeX reconstruction for scientific papers
```

Evidence hierarchy:

- selected2000: primary large-scale direct-parser comparison, n=1972 fair
  intersection, Ours / ContentList Direct / MinerU Direct.
- selected200: controlled four-method comparison including Nougat.
- selected2000 usability: generated.tex 2000/2000, compile 1852/2000,
  comparison conversion 1999/2000, structure metrics 1980/2000.

PRCV evidence starts here:

```text
data/09_eval_reports/00_PRCV_FINAL_EVIDENCE_20260531/
docs/PRCV_EVIDENCE_REGISTRY_20260531.md
```

Paper workspace:

```text
/Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh
```

PRCV does not claim Nougat selected2000, source-level TeX AST recovery, solved
semantic table reconstruction, or compile/visual QA for parser-output baselines.

## Candidate Future Modules

These are project directions, not current PRCV claims:

| Candidate module | Possible paper question | Core interfaces it should use |
|---|---|---|
| Full 8000-scale material release | How reproducible and portable is the runtime corpus? | dataset manifests, backup manifests, parser outputs, path config |
| Nougat/API/MMD external baselines | How do MMD/Markdown or API systems compare under neutral structure metrics? | ComparisonStructure converters, baseline tools, manifest registry |
| Table structure reconstruction | Can table-cell semantics be recovered beyond safe fallback? | table evidence, DocumentIR table nodes, table renderer plugin, table metrics |
| Float-caption grounding | Can figures, subfigures, and captions be grounded robustly? | float-caption matcher, layout grouping, caption evidence metrics |
| Front-matter entity linking | Can authors, affiliations, emails, and footnotes be linked? | front-matter context groups, entity schema, renderer roles |
| Formula/context preservation | Can formula spans and context groups be reconstructed more faithfully? | formula context groups, math renderer, formula metrics |
| Relation learning / GNN | Can learned relation signals improve structure recovery? | optional GNN view adapter, graph tensors, decoder bridge, ablation registry |
| Algorithm-region handling | Can algorithm/pseudocode regions be rendered safely and structurally? | algorithm detector, renderer plugin, compile safety metrics |

Each future module should create a new evidence registry instead of mixing
numbers into the PRCV registry.

## Interface Design Rule

When adding a new paper module, first decide which interface it exercises:

- parser adapter
- observable fact layer
- typed IR
- renderer plugin
- evaluation converter
- paper/evidence registry

If the paper needs a new contract, document it before changing the source. A
paper-specific patch that cannot be explained as an interface extension should
remain in an experiment branch or diagnostic tool.

## Storage Rule

GitHub source repo stores code and small summary evidence. Netdisk stores
runtime material. The paper workspace stores manuscript assets. AutoDL stores
active heavy runtime until reset.

Do not collapse these roles.
