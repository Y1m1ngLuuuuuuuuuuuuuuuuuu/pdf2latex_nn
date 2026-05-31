# Current Project Architecture (Post-PRCV, 2026-05-31)

This is the current source of truth for the PRCV-facing pipeline.

## Mainline

```text
PDF / MinerU parser outputs
  -> Observable Fact Layer
  -> DocumentIR
  -> RenderTreeIR
  -> compile-safe role renderers
  -> generated LaTeX
  -> ComparisonStructure evaluation
```

The mainline is observable-fact-guided reconstruction. It is not a GNN-driven
pipeline and it does not attempt deterministic source-level TeX AST recovery.

## Observable Fact Layer

The fact layer preserves parser-derived evidence before structural decisions:
geometry, reading flow, text/style spans, formula evidence, caption/reference
evidence, table/front-matter cues, and provenance.

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

Algorithm-region handling and semantic table-cell reconstruction remain modular
extensions, not current paper claims.

## Evaluation

ComparisonStructure is the neutral structural evaluation layer. Compile and
visual QA are defined only for complete LaTeX outputs.

## Boundaries

Do not claim source-level TeX AST recovery, table-cell semantic reconstruction
solved, Algorithm renderer enabled, Nougat selected2000 completed, or broad
float-caption resolver solved.

