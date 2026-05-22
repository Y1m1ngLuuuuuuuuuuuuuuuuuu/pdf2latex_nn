# PDF2LaTeX NN

**最后更新**：2026-05-18

PDF2LaTeX NN 是一个面向 born-digital 学术论文的、结构感知的 PDF 到 LaTeX 流水线。它不是把 PDF 转换简单看成 OCR，而是从 PDF 中提取视觉事实，从匹配的 TeX 源码中生成图关系真值标签，训练 GNN 预测文档关系，再通过解耦的 IR 渲染器重建可编译的 LaTeX。

## 当前部署路径

生产路径只使用 v7：

```text
compiled PDF + matching TeX
  -> MinerU content_v2
  -> v7 reading/layout cleanup
  -> PyMuPDF style spans
  -> GNNViewAdapter float-proxy graph view
  -> SciBERT + geometry/style/sequence graph features
  -> TeX AST alignment labels
  -> GATv2/Y-Network edge-relation model
  -> TreeDecoder / RenderTreeIR
  -> OriginalLikeIRLatexRenderer
  -> generated .tex / .pdf
```

旧的 v3/v4/v5 预处理版本不再作为生产输入，只作为历史实验保留。

当前代码明确区分两条数据/模型轨道：

```text
locked baseline/results:
  v7_registry_adapteraware_20260515_181724
  edge_attr_dim=22
  已有 M05/M07 checkpoint 和报告保持不动

current experimental rebuild:
  v7_floatproxy_adapter_20260516_205926
  edge_attr_dim=26
  figure/table/algorithm 节点以 caption/placeholder float proxy 的形式进入 GNN
```

在评估新的 float-proxy 路径时，不要删除锁定基线的 checkpoint 或报告。

## 主要关系任务

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
# 从 PDF + TeX 继续构建新的生产数据
python scripts/pipeline/build_v7_dataset_staged.py ...

# 基于已有 v7 内容重建 graph tensor 并重新打标签
bash scripts/pipeline/run_current_v7_rebuild_relabel.sh

# 训练当前 GATv2/Y-Network 关系模型
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

当前面向论文结果收集的完整评测套件：

```bash
# 运行当前 ablation、E2E generator QA、Nougat paired comparison 和最终汇总
python scripts/pipeline/run_current_full_eval_suite.py

# 只汇总已有输出，不训练、不生成
python scripts/pipeline/collect_current_eval_results.py
```

默认评测输出：

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
docs/v7_training_and_monitoring.md    生产数据/训练 runbook
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
