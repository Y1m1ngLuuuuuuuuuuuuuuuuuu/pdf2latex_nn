# Archived V8 Atomic MERGE / GNN Experiments

Archived on 2026-05-26 after the selected200 rerun closed the learned MERGE
branch as a production candidate.

The archived route explored middle-derived atomic blocks, PyG graph families,
GNN MERGE checkpoints, residual rankers, selector/veto policies, and ordered
coverage refreshes. The conclusion is that learned branches can change the
generated LaTeX, but they did not beat the deterministic v8 path under the
wrong-merge safety constraint.

Production now uses:

```text
middle.json -> v8 reflow/style/stack -> deterministic merge -> renderer
```

Do not treat this archive as the default reconstruction path.
