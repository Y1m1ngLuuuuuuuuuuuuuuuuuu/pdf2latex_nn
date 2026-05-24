# PDF2LaTeX NN

**最后更新**：2026-05-24

PDF2LaTeX NN 是一个面向 born-digital 学术论文的、版式感知 PDF 到
LaTeX 重建流水线。它不是普通 OCR，也不是作者源码级 TeX AST 恢复。
当前默认路径从渲染后的 PDF 事实出发，通过 v8 middle reflow、
DocumentIR、RenderTreeIR 和 original-like renderer 生成可编译、保留
块级结构的 LaTeX。

## 当前部署路径

当前默认重建路径是 v8 / layout-first：

```text
compiled PDF
  -> MinerU middle.json + content_list.json
  -> v8 middle reflow / reading-order repair
  -> DocumentIR
  -> FrontMatterIR
  -> heading style registry + stack skeleton
  -> RenderTreeIR
  -> StyleProfile
  -> OriginalLikeIRLatexRenderer
  -> generated .tex / .pdf
```

GNN 关系模型现在是显式实验分支，不再是默认生成依赖：

```text
content_list_v7_styles.json
  -> GNNViewAdapter
  -> graph.pt
  -> TeX-derived MERGE/PARENT_CHILD/NONE labels
  -> GNN training / ablation / diagnostics
```

不要把 GNN view 当作 renderer source。最终生成必须消费完整
`DocumentIR` / `RenderTreeIR`。

## 当前默认能力

```text
middle-derived reading-order repair
source PDF page size preservation
document-local heading style registry
stack skeleton section hierarchy
FrontMatter Phase 0 title/author/affiliation/email/abstract preservation
single/two/mixed-column approximation
wide figure* / table* rendering
ordered enumerate / itemize recovery
citation and bibliography repair
```

## 可选关系学习任务

图模型预测三类有向边：

```text
MERGE        = 0  物理延续 / 段落缝合
PARENT_CHILD = 1  逻辑层级 / 挂载关系
NONE         = 2  无结构关系
```

`SIBLING` 不再作为学习类别。同级顺序由 v7 reading order 和 renderer 排序恢复。

## 活跃接口

```text
content_v7_styles.json  完整 PDF 事实层
GNNViewAdapter          过滤/代理后的图可见视图 + v7 映射
GraphInput.pt           节点/边张量
GraphLabels             基于 TeX 的 GNN 视图边标签
PredictedRelations      GNN 输出概率
RenderTreeIR            decoder 输出，并映射回完整 v7 id
StyleProfile            全局/局部版式 profile
CitationResolution      citation/reference 修复状态
```

详见 [docs/frontend_backend_contract_v1.md](../frontend_backend_contract_v1.md)。

## 关键脚本

```bash
# 运行当前 v8 layout reconstruction
python scripts/pipeline/run_v8_layout_reconstruction.py ...

# 可选：基于已有 v7 内容重建 graph tensor 并重新打标签
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh

# 可选：训练 GATv2/Y-Network 关系模型
python scripts/pipeline/train_edge_gnn_full.py ...

# 生成 ablation 命令
python scripts/pipeline/prepare_ablation_suite.py \
  --matrix configs/ablation_matrix_v7_adapteraware_20260514_2109.json \
  --output-sh data/08_runs/run_ablation_matrix_v7_adapteraware_20260514_2109.sh

# 使用当前 IR renderer 批量视觉 QA / E2E 推理
python scripts/pipeline/batch_visual_qa_inference.py --renderer ir ...
```

当前实验重建/重标注命令模板：

```bash
TAG=v7_floatproxy_adapter_$(date +%Y%m%d_%H%M%S) \
INPUT_MANIFEST=data/00_manifests/v7_layers_epigraph_20260514_0238_trainable_recall98.json \
WORKERS=4 \
PYTHON_BIN=/root/miniconda3/envs/pdf2latex/bin/python \
EMBEDDING_DEVICE=cpu \
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh
```

历史/可选 GNN 关系分支和 hard20 对比的完整评测套件：

```bash
# 运行历史/current GNN ablation、E2E generator QA、Nougat paired comparison 和最终汇总
python scripts/pipeline/run_current_full_eval_suite.py

# 只汇总已有输出，不训练、不生成
python scripts/pipeline/collect_current_eval_results.py
```

这些输出用于论文追溯和 GNN/Nougat 对比，不是当前 v8 默认生成路径本身：

```text
data/09_eval_reports/ablations_v7_floatproxy_adapter_20260516_205926_current/
data/09_eval_reports/current_e2e_comparison_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/nougat_current_paired_hard20_floatcaption_rerun_20260518_132615/
data/09_eval_reports/current_eval_rollup_hard20_floatcaption_rerun_20260518_132615_cleanmetrics/
```

## 当前文档

```text
docs/PROJECT_FILE_LAYOUT.md           本地/AutoDL 目录和产物地图
docs/PROJECT_ARCHITECTURE_FULL.md     完整架构、逻辑、指标和代码地图
docs/PROJECT_PAPER_DESCRIPTION_2026_05_18.md 面向论文写作的完整项目描述
docs/PROJECT_SOURCE_OF_TRUTH.md       本地 / GitHub / AutoDL 边界
docs/PROJECT_OVERVIEW.md              架构和实现摘要
docs/frontend_backend_contract_v1.md  解耦 IR 合约
docs/feature_schema_v0.md             图张量特征合约
docs/ground_truth_labeling_v0.md      TeX-to-PDF 真值标签生成
docs/ablation_plan_v2.md              当前 ablation 协议
docs/ablation_results_current.md      最新锁定 ablation 结果
docs/V8_MIDDLE_REFLOW_AND_STYLE_DETECTOR.md 当前 v8 路径和参数
docs/FRONT_MATTER_ENTITY_MODEL_PLAN.md 后续精确 author/affiliation/email 解析计划
docs/ENVIRONMENT_SETUP.md             conda/venv 环境安装和依赖 profile
docs/v7_training_and_monitoring.md    可选 GNN 关系学习 runbook
docs/interface_audit_2026_05_14.md    当前接口审计和旧路径检查
docs/LOCAL_CONFIGURATION.md           私有本地配置说明
```

新增中文评测说明：

```text
docs/translate_doc/TEST_RESULTS_AND_METRICS_2026_05_18.zh.md
```

## 重要路径

本地项目根目录：

```text
/Users/lu/Code/Project/pdf2latex_nn/test_4_19
```

AutoDL 项目根目录：

```text
/root/autodl-tmp/pdf2latex_nn
```

大型产物保存在 AutoDL：

```text
/root/autodl-tmp/pdf2latex_nn/data
```

当前目录契约：

```text
docs/PROJECT_FILE_LAYOUT.md
docs/translate_doc/PROJECT_FILE_LAYOUT.zh.md
```

不要提交数据集、checkpoint、生成 PDF、密钥或 AutoDL 凭据。
