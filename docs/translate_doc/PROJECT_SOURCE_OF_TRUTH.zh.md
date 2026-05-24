# 项目事实源

**最后更新**：2026-05-24

本文件固定本地、GitHub、AutoDL 之间的边界，以及当前默认运行链路。

## 1. 源码同步原则

```text
local source edits -> GitHub -> AutoDL git pull / targeted source sync
```

不要把本地目录大范围递归覆盖到 AutoDL。运行时产物、数据集、checkpoint、
generated PDF 和长任务日志应保留在 AutoDL，不应通过源码同步覆盖。

GitHub：

```text
https://github.com/Y1m1ngLuuuuuuuuuuuuuuuuuu/pdf2latex_nn.git
```

本地：

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL：

```text
/root/autodl-tmp/pdf2latex_nn
```

## 2. 当前默认生产链路

当前默认 reconstruction path 是 v8 / layout-first：

```text
compiled PDF
  -> MinerU middle.json + content_list.json
  -> v8 middle reflow and reading-order repair
  -> DocumentIR
  -> FrontMatterIR
  -> heading style registry + stack skeleton
  -> RenderTreeIR
  -> StyleProfile / v8 style detector
  -> OriginalLikeIRLatexRenderer
```

默认 E2E reconstruction 不加载 GNN checkpoint，也不渲染 GNN view。

当前默认入口：

```bash
python scripts/pipeline/run_v8_layout_reconstruction.py ...
```

## 3. 保留的 v7 / GNN 关系学习分支

GNN 分支仍用于 relation-learning、ablation 和诊断：

```text
content_list_v7_styles.json
  -> GNNViewAdapter
  -> graph.pt
  -> TeX-derived relation labels
  -> GNN training / diagnostics / ablations
```

GNN view 是过滤/代理后的 graph-visible view，不是完整文档。生成器不得从
GNN view 直接渲染；任何 GNN 预测都必须通过 exact graph-to-v7 bridge
回到完整事实层。

历史/实验家族：

```text
v7_registry_adapteraware_20260515_181724  edge_attr_dim=22
v7_floatproxy_adapter_20260516_205926     edge_attr_dim=26
```

不要混用不同 schema 的 graph 和 checkpoint。

## 4. 数据与产物边界

应提交到 Git：

```text
source code
configs
docs
tests
small metadata manifests when useful
```

不要提交：

```text
PDF corpora
TeX corpora
MinerU outputs
graph .pt caches
model checkpoints
generated PDFs
API keys / .env / passwords / tokens
```

当前大型产物均应放在 `data/` 和 `logs/`，并按
`docs/PROJECT_FILE_LAYOUT.md` 的目录契约命名。

## 5. 当前活跃数据策略

arXiv 数据构建采用：

```text
download TeX source
compile locally to PDF
save source tree under data/03_tex_sources/{doc_id}/
save compiled PDF under data/01_raw_pdfs/{doc_id}.pdf
run MinerU over compiled PDFs
```

下载阶段不保存 arXiv 站点上的 PDF，但编译得到的 PDF 必须保留，因为它是
MinerU 输入和 layout evaluation 的对齐基准。

## 6. 当前维护文档

```text
README.md
docs/PROJECT_FILE_LAYOUT.md
docs/PROJECT_ARCHITECTURE_FULL.md
docs/PROJECT_SOURCE_OF_TRUTH.md
docs/PROJECT_OVERVIEW.md
docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md
docs/generator_logic_audit_2026_05_17.md
docs/layout_aware_reconstruction_target.md
docs/ground_truth_labeling_v0.md
docs/v7_training_and_monitoring.md
docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md
docs/ENVIRONMENT_SETUP.md
docs/MINERU_ADAPTER_CONTRACT.md
docs/TABLE_ENGINE_CONTRACT.md
docs/STYLE_TEMPLATE_CONTRACT.md
```
