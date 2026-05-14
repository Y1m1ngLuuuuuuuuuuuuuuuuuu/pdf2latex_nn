# Interface Audit 2026-05-14

This note records the current v7-only interface check across frontend,
GNN training, labeling, inference, and generation.

## Current Contract

```text
full MinerU v7 styled JSON
  -> GNNViewAdapter
  -> graph.pt
  -> AlignmentLabeler labels on the same GNN view
  -> M05 / ablation training
  -> predicted GNN-view edges
  -> relation bridge back to full v7 node ids
  -> DocumentIR + RenderTreeIR
  -> OriginalLikeIRLatexRenderer
```

The full v7 JSON remains the fact layer. Nodes excluded from the GNN view
remain available to the generator.

## Checked Entrypoints

| Area | Entrypoint | Current status |
| --- | --- | --- |
| v7 rebuild/relabel | `scripts/pipeline/run_current_v7_rebuild_relabel.sh` | Adapter-aware run starts from existing v7 JSON; does not run MinerU. |
| graph build | `scripts/pipeline/rebuild_graphs_from_manifest.py` | Reads styled v7 JSON; calls `build_graph_from_content_v7`; preserves GNN-view mapping fields in graph tensors. |
| label generation | `scripts/pipeline/relabel_manifest.py` / `src/reasoning/label_generator.py` | Uses `GNNViewAdapter`, so graph nodes and labels share the same filtered node view. |
| training | `scripts/pipeline/train_edge_gnn_full.py` | Requires manifest/root explicitly; current clean manifest is `v7_adapteraware_20260514_2109_clean_trainable.json`. |
| ablation | `configs/ablation_matrix_v7_adapteraware_20260514_2109.json` | Current maintained matrix for adapter-aware labels. Older v3 is historical. |
| E2E QA | `scripts/pipeline/run_m05_e2e_comparison.py` | Defaults to current clean manifest + M05 checkpoint + `--renderer ir`. |
| batch visual QA | `scripts/pipeline/batch_visual_qa_inference.py` | Defaults to `--renderer ir`; `tree` is explicit regression/debug only. |
| single-doc generation | `scripts/pipeline/step5_generate_tex.py` | Defaults to `--renderer ir`; requires `--content-json` for IR rendering. |
| legacy step5 name | `scripts/pipeline/step5_run_inference.py` | Compatibility wrapper forwarding to `step5_generate_tex.py`; no independent legacy renderer path. |
| generator | `src/generation/render_surface.py` / `src/generation/ir_renderer.py` | Canonical surface is `OriginalLikeIRLatexRenderer`. |
| old renderer | `src/generation/latex_renderer.py` | Compatibility helpers and legacy tests only. |

## Removed Or Neutralized Old Paths

- Removed `configs/ablation_matrix_v1.json`.
- Removed `configs/ablation_matrix_v2.json`.
- Removed hard-coded old debug defaults from
  `scripts/debug/check_hierarchy_inversion.py`; callers must pass current
  `--content-json`, `--graph`, and either `--logits` or `--checkpoint`.
- `step5_run_inference.py` no longer renders directly through
  `TreeDecoder.render_document`.

## Intentional Compatibility

The following are allowed and are not production data inputs:

- `tests/test_v7_contract.py` contains a synthetic `content_v4` payload to
  verify that the v7 guard rejects old JSON.
- `TreeDecoder.render_document` remains for tests and explicit
  `--renderer tree` regression debugging.
- `DocumentDataset.normalize_edge_labels` folds old 4-class labels to the
  current 3-class schema only when an old graph is explicitly loaded; current
  manifests should already be 3-class.
- Historical `configs/ablation_matrix_v3.json` includes the
  `F03_raw_mineru_flow` control. This is a deliberate ablation, not a
  production manifest.

## Current Remote Runtime

The adapter-aware rebuild/relabel run started from existing v7 JSON only:

```text
TAG=v7_adapteraware_20260514_2109
INPUT_MANIFEST=data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json
OUTPUT_MANIFEST=data/00_manifests/v7_adapteraware_20260514_2109_clean_trainable.json
```

MinerU is not rerun in this path.

## Verification

Local:

```text
python3 -m py_compile scripts/pipeline/step5_generate_tex.py \
  scripts/pipeline/step5_run_inference.py \
  scripts/pipeline/batch_visual_qa_inference.py \
  scripts/debug/check_hierarchy_inversion.py
```

Remote:

```text
/root/miniconda3/envs/pdf2latex/bin/python -m py_compile ...
/root/miniconda3/envs/pdf2latex/bin/python -m pytest -q \
  tests/test_gnn_view_adapter.py \
  tests/test_mineru_v7_document_ir_adapter.py \
  tests/test_graph_builder_features.py::test_type_aware_message_mask_blocks_float_noise_and_reverse_title_pollution
```

Remote targeted tests passed: `13 passed`.
