# 项目文件布局

> 历史翻译文档。当前目录布局请看
> `docs/MAIN_PATH_LAYOUT_AFTER_SUBMISSION.md` 和
> `docs/CANONICAL_PROJECT_PATHS_POST_SUBMISSION.md`。

**最后更新**：2026-05-24

本文档说明本地和 AutoDL 两边的目录语义。相同路径在两边应表示同一类
内容；本地可以只有小样本，AutoDL 可以有完整数据和运行产物。

## 1. 根目录

```text
local:  /Users/lu/Code/Project/pdf2latex_nn/test_4_19
AutoDL: /root/autodl-tmp/pdf2latex_nn
```

代码同步优先走 GitHub。不要 broad rsync 数据、checkpoint、generated PDF
或运行日志。

## 2. 顶层源码目录

| 路径 | 含义 | 是否进 Git |
| --- | --- | ---: |
| `src/` | 生产 Python 包 | 是 |
| `src/perception/` | PDF/v7 感知、阅读顺序、样式探针、GNNViewAdapter | 是 |
| `src/adapters/` | MinerU/v7 -> DocumentIR adapter | 是 |
| `src/reasoning/` | v8 reflow、heading skeleton、decoder、label/GNN 分支 | 是 |
| `src/ir/` | DocumentIR / RenderTreeIR schema、序列化、校验 | 是 |
| `src/generation/` | front matter、style、float/table/list/reference renderer | 是 |
| `src/evaluation/` | comparison structure 和指标 | 是 |
| `scripts/pipeline/` | 数据、训练、E2E、ablation 入口 | 是 |
| `tools/` | 转换、审计、API/外部 benchmark 工具 | 是 |
| `configs/` | ablation、prompt、外部评测配置 | 是 |
| `tests/` | 单测和回归测试 | 是 |
| `docs/` | 架构、接口、runbook、结果说明 | 是 |
| `environment.yml` | conda 基础环境入口 | 是 |
| `requirements.txt` | v8/GNN 基础依赖列表 | 是 |
| `requirements_server.txt` | AutoDL/server 额外依赖 | 是 |

## 3. `data/` 目录契约

| 路径 | 含义 |
| --- | --- |
| `data/00_manifests/` | 数据/运行 manifest |
| `data/01_raw_pdfs/` | MinerU 输入 PDF；当前重建中是由 TeX 本地编译出的 PDF |
| `data/02_mineru_outputs/` | 原始 MinerU 输出，包括 `middle.json`、`content_list.json` 等 |
| `data/03_tex_sources/` | 编译通过的 TeX 源码树 |
| `data/03_tex_source_pool/` | source staging/archive，只有 manifest 引用时才是活跃输入 |
| `data/04_ground_truth_ir/` | TeX-derived mapping、gold/comparison IR、label report |
| `data/05_observed_ir/` | 前端观测 IR、DocumentIR sidecar、诊断导出 |
| `data/06_graph_features/` | 当前 graph / labeled graph 家族 |
| `data/06_graph_features_v7/` | 历史 v7 graph 家族 |
| `data/07_predicted_ir/` | 模型预测 sidecar / predicted relation IR |
| `data/08_output_latex/` | 受控生成的 TeX/PDF |
| `data/08_runs/` | 批处理 shell 脚本和命令包 |
| `data/09_eval_reports/` | 报告、汇总、评测输出、审计输出、部分 checkpoint |
| `data/10_checkpoints/` | 可选 checkpoint 镜像 |
| `data/external/` | 外部 benchmark 资料和预测 |
| `data/_tmp_*` | 临时工作目录，确认无任务使用后才能清理 |

## 4. 当前 v8 输出家族

```text
data/09_eval_reports/v8_reflow_20260523/
```

该目录保存 v8 middle reflow / style detector / 00050 smoke 等当前默认路径
输出。新的 v8 小实验应继续使用明确日期和目的的 run tag，例如：

```text
data/09_eval_reports/v8_reflow_<YYYYMMDD>_<purpose>/
```

## 5. 当前 arXiv TeX/PDF 数据布局

下载 source，不下载站点 PDF；编译后的 PDF 需要保留：

```text
data/03_tex_sources/{doc_id}/       TeX source tree
data/01_raw_pdfs/{doc_id}.pdf       locally compiled PDF
data/02_mineru_outputs/<tag>/...    MinerU raw outputs
```

跑完后应冻结 manifest：

```text
data/00_manifests/arxiv*_compilable_tex_pdf_clean*.jsonl
```

每条至少包含：

```json
{
  "doc_id": "2501.12345",
  "source_dir": "data/03_tex_sources/2501.12345",
  "main_tex": "main.tex",
  "pdf_path": "data/01_raw_pdfs/2501.12345.pdf"
}
```

## 6. 归档和过时产物

过时但需要可追溯的内容放入：

```text
data/09_eval_reports/_archive/
```

当前已归档示例：

```text
data/09_eval_reports/_archive/20260509_legacy_e2e_outputs/
data/09_eval_reports/_archive/20260515_generator_iterations/
data/09_eval_reports/_archive/20260519_audit_bundle/
data/09_eval_reports/_archive/20260523_legacy_merge_gnn_generator_debug/
```

根目录 `e2e_outputs/` 已归档，不再作为新输出位置。

## 7. 什么东西放哪里

| 内容 | 正确位置 | 不要放在 |
| --- | --- | --- |
| 编译通过源码树 | `data/03_tex_sources/{doc_id}/` | `data/09_eval_reports/` |
| 编译得到的输入 PDF | `data/01_raw_pdfs/{doc_id}.pdf` | `data/09_eval_reports/` |
| 原始 MinerU 输出 | `data/02_mineru_outputs/<tag>/...` | `src/`, `docs/` |
| v8 content / sidecar | run tag 下，由 manifest 引用 | 随机根目录 |
| graph tensor | `data/06_graph_features/${TAG}_graphs/` | `data/09_eval_reports/` |
| checkpoint | `data/09_eval_reports/<run>/<model>/seed_*/best_model.pth` 或 `data/10_checkpoints/` | `src/` |
| 汇总报告 | `data/09_eval_reports/<run>/` | 随机顶层目录 |

## 8. 清理规则

删除前先分类：

```text
ACTIVE_KEEP
HISTORICAL_KEEP
ARCHIVE_CANDIDATE
DELETE_CANDIDATE_AFTER_CONFIRM
DO_NOT_TOUCH_RUNNING
```

不能删：

```text
data/03_tex_sources/
data/01_raw_pdfs/
活跃 manifest
活跃 graph/checkpoint/eval run family
活跃或刚完成长任务日志
```

可以候选清理，但必须先确认：

```text
data/_tmp_*
过期 smoke 输出
重复渲染 PDF
没有 manifest 引用的旧 cache/stage 目录
```
