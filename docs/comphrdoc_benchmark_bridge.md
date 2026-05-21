# CompHRDoc Benchmark Bridge

This project primarily reconstructs layout-aware, block-structure-preserving,
compilable LaTeX from rendered PDFs.  Microsoft CompHRDoc evaluates a related
but not identical task: hierarchical document structure analysis (HDSA), reading
order, page-object classification and page-object detection.

## Supported Path

The local bridge converts our full v7 / DocumentIR records to CompHRDoc
`hr_json` records:

```text
MinerU v7 JSON
  -> MinerUV7DocumentIRAdapter
  -> DocumentIR
  -> tools/comphrdoc/convert_to_comphrdoc.py
  -> CompHRDoc hr_json
  -> official CompHRDoc evaluation scripts
```

The converter writes fields expected by CompHRDoc:

```json
{
  "text": "...",
  "box": [x0, y0, x1, y1],
  "class": "section | fstline | paraline | figure | table | caption | ...",
  "page": 0,
  "is_meta": false,
  "line_id": 0,
  "parent_id": -1,
  "relation": "contain | connect | meta",
  "page_object_id": 0
}
```

The conversion manifest is deliberately written next to the `hr_json` folder
instead of inside it, because the official CompHRDoc scripts try to read every
entry in `gt_folder` / `pred_folder` as a document JSON.

## What We Can Fairly Compare

The current bridge is suitable for smoke-testing and for approximate comparison
on the overlapping structural tasks:

- reading-order tree/chain quality,
- coarse page-object class mapping,
- hierarchical document structure tree quality,
- float/caption grouping when both systems produce compatible units.

## What Is Best-Effort or Out of Scope

CompHRDoc includes fine-grained page object classes such as author, mail and
affiliation.  Our pipeline does not optimize those as core model targets.  The
bridge maps them with metadata/text heuristics where possible, but these fields
should not be treated as main success metrics for our paper.

CompHRDoc official annotations are line-level in many cases, while our v7 IR is
block-level.  Therefore raw official scores can be depressed by granularity
differences.  For paper claims, report this as a compatibility benchmark rather
than the primary evaluation of our reconstruction objective.

## Setup

The official repo is kept outside tracked source as a third-party artifact:

```bash
curl -L -o third_party/CompHRDoc.zip \
  https://github.com/microsoft/CompHRDoc/archive/refs/heads/main.zip
unzip -q third_party/CompHRDoc.zip -d third_party
mv third_party/CompHRDoc-main third_party/CompHRDoc

python3 -m venv .venv_comphrdoc
.venv_comphrdoc/bin/python -m pip install -U pip
.venv_comphrdoc/bin/python -m pip install tqdm apted scipy scikit-learn graphviz opencv-python-headless numpy
```

The official CompHRDoc dataset zip in the source archive is a Git LFS pointer.
Real benchmark annotations/images still need the proper dataset release.

## Convert Our Outputs

Single v7 file:

```bash
.venv_comphrdoc/bin/python tools/comphrdoc/convert_to_comphrdoc.py \
  --v7-json data/02_mineru_outputs/mineru_output/2501.00050/auto/2501.00050_content_list_v7_styles.json \
  --doc-id 2501.00050 \
  --out-dir local_outputs/comphrdoc_smoke/pred
```

Manifest:

```bash
.venv_comphrdoc/bin/python tools/comphrdoc/convert_to_comphrdoc.py \
  --manifest data/00_manifests/YOUR_MANIFEST.json \
  --out-dir local_outputs/comphrdoc/pred
```

## Run Official Evaluation

The official scripts require matching JSON filenames in `gt_folder` and
`pred_folder`.

```bash
.venv_comphrdoc/bin/python tools/comphrdoc/run_comphrdoc_eval.py \
  --gt-folder "$PWD/third_party/CompHRDoc/evaluation/examples/hr_json" \
  --pred-folder "$PWD/third_party/CompHRDoc/evaluation/examples/hr_json" \
  --output-dir local_outputs/comphrdoc_official_example
```

For our own predictions, first convert both the gold-compatible structure and
the predicted structure into the same `hr_json` shape, then run the wrapper.

## Current Validation

The local CompHRDoc venv successfully runs official example evaluation:

- `teds_eval.py`: macro/micro TEDS = 1.0 on official example self-comparison.
- `reading_order_eval.py`: macro/micro TEDS = 1.0 on official example self-comparison.
- `classify_eval.py`: detailed/macro/micro F1 = 1.0 on official example self-comparison.
