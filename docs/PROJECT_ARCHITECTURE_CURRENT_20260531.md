# Current Project Architecture

Last updated: 2026-06-01.

This is the current architecture source of truth for the reusable PDF2LaTeX
project core. PRCV 2026 is one paper module implemented on top of this core.

## Platform Mainline

```text
PDF / parser outputs
  -> Observable Fact Layer
  -> DocumentIR
  -> RenderTreeIR
  -> compile-safe role renderers
  -> generated LaTeX
  -> ComparisonStructure evaluation
```

The mainline is observable-fact-guided reconstruction. It is not a GNN-driven
renderer and it does not attempt deterministic source-level TeX AST recovery.

## Layer Ownership

| Layer | Code area | Stable responsibility |
|---|---|---|
| Path/runtime config | `src/config/` | Resolve local, AutoDL, WSL, output, report, and paper roots without hard-coded machine paths. |
| Dataset and manifests | `src/datasets/`, `data/00_manifests/` | Define document sets and input/output locations. |
| Parser adapters | `src/adapters/` | Convert MinerU/parser artifacts into project-visible facts. |
| Observable facts | `src/perception/` | Preserve geometry, reading order, text/style spans, formulas, captions, references, table cues, and provenance. |
| Typed IR | `src/ir/` | Define serializable document and render structures. |
| Reasoning modules | `src/reasoning/` | Build section hierarchy, front matter, formula/reference/page-furniture groups, float-caption associations, and optional graph views. |
| Role renderers | `src/generation/` | Emit compile-safe LaTeX from typed roles. |
| Evaluation | `src/evaluation/`, `tools/*comparison*` | Convert outputs to ComparisonStructure and compute neutral structural metrics. |
| Paper/evidence modules | `docs/`, `data/09_eval_reports/` | Record claim boundaries, denominators, locked numbers, and paper-specific evidence. |

## Observable Fact Layer

The fact layer preserves PDF-observable and parser-derived evidence before
structural decisions:

- page geometry and reading flow
- text and style spans
- formula line/span evidence
- figure/table/caption evidence
- reference subtype and bibliography evidence
- page furniture and front-matter cues
- provenance and artifact lineage

This layer should not delete information merely because a later model or paper
module does not use it.

## Typed IR and Rendering

`DocumentIR` is the structural fact interface. `RenderTreeIR` is the typed
rendering tree. Role renderers emit LaTeX from typed roles rather than directly
string-concatenating parser outputs.

Renderer contracts prioritize compile-safe degradation:

- formula safe fallback
- float-caption materialization
- reference-list fallback
- table safe fallback
- front-matter rendering
- list and paragraph rendering
- figure/table crop fallback when semantic reconstruction is incomplete

Algorithm-region handling and semantic table-cell reconstruction remain modular
extensions, not current paper claims.

## Evaluation

ComparisonStructure is the neutral structural evaluation layer. It lets output
families with different contracts be compared structurally without pretending
that every method emits compilable LaTeX.

Compile and visual QA are defined only for complete LaTeX outputs.
Parser-output and MMD/Markdown baselines should not receive compile/visual QA
numbers unless they are explicitly converted into a complete LaTeX-output
contract in a future paper module.

## Paper Module Boundary

A paper module may select a dataset denominator, baseline family, metric table,
and claim boundary, but it should not change the platform interfaces for a
paper-specific shortcut.

The current submitted PRCV module uses:

- selected2000 as the primary large-scale direct-parser comparison
- selected200 as the controlled four-method comparison including Nougat
- selected2000 usability counts for complete LaTeX behavior of Ours

Future modules may target table semantics, float-caption grounding, front
matter entity linking, relation learning, API/MMD baselines, or full 8000-scale
runtime release, but they should register those scopes separately.

## Boundaries

Do not claim:

- source-level TeX AST recovery as the project target
- Nougat selected2000 completion
- selected2000 four-method comparison completion
- selected2000 Ours metrics 2000/2000
- compile or visual QA for parser-output baselines
- solved semantic table-cell reconstruction
- enabled broad Algorithm renderer
- solved broad float-caption resolution
- GNN/Y-network as the current PRCV mainline

## Historical Modules

The repository still contains v7/v8/GNN documents, ablation records, and
training scripts. They are retained as research history and potential future
paper material. They are not the default project mainline unless a future paper
module explicitly promotes them with a new evidence registry and claim boundary.
