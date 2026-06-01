# 文档更新报告 2026-06-01

本轮目标是把项目文档从“PRCV 单篇论文冲刺状态”整理为“PDF2LaTeX 长期项目 + 多篇论文模块”的结构。

## 已完成

- 将 README 改为项目入口，明确 PDF2LaTeX 是长期项目，不是 PRCV 单篇论文代码包。
- 重写 `docs/PROJECT_SOURCE_OF_TRUTH.md`，明确三层所有权：
  - GitHub source repo
  - AutoDL runtime
  - local paper workspace
- 新增 `docs/PROJECT_SCOPE_AND_PAPER_MODULES_20260601.md`，定义“整体项目”和“论文模块”的关系。
- 新增 `docs/INTERFACE_DESIGN_CURRENT_20260601.md`，列出当前可复用接口：
  - PathConfig
  - DatasetManifest
  - ParserAdapter
  - ObservableFactLayer
  - DocumentIR
  - RenderTreeIR
  - RendererPlugin
  - ComparisonStructure
  - EvidenceRegistry
  - RuntimeBackup
- 新增 `docs/DOCUMENTATION_INDEX_20260601.md`，区分当前 source-of-truth、PRCV 模块、接口合同和历史实现背景。
- 更新路径文档，正式使用新源码目录：
  `/Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction`
- 保留旧路径：
  `/Users/lu/Code/Project/pdf2latex_nn/test_4_19`
  作为兼容软链接。
- 更新 PRCV evidence registry，明确 PRCV 是一个已提交 paper module，不是整个项目。
- 给旧 v7/v8/GNN/ablation/translation 文档增加历史说明，避免误当作当前主线。
- 更新 `.env.example` 和 `config/paths.local.template.yaml` 的示例项目名。

## 当前项目关系

```text
PDF2LaTeX overall project
  -> reusable interfaces and implementation layers
  -> paper module: PRCV 2026 observable reconstruction
  -> future paper modules:
       table semantics
       float-caption grounding
       front-matter linking
       Nougat/API baselines
       relation learning / GNN
       full8000 runtime/material release
```

## 当前主路径

```text
source repo:
  /Users/lu/Code/Project/pdf2latex_nn/pdf2latex-observable-reconstruction

legacy compatibility symlink:
  /Users/lu/Code/Project/pdf2latex_nn/test_4_19

process history:
  /Users/lu/Code/Project/pdf2latex_nn/project_process_history

paper workspace:
  /Users/lu/University/Paper/pdf2latex/PRCV/LaTeX_Reconstruction_zh

AutoDL runtime:
  /root/autodl-tmp/pdf2latex_nn
```

## 验证

- `python3 scripts/setup/print_project_paths.py` 已通过。
- `python3 -m py_compile src/config/project_paths.py scripts/setup/print_project_paths.py` 已通过。
- 本轮没有运行实验。
- 本轮没有删除文件。
- 本轮没有修改论文正文。

## 建议下一步

1. 人工快速阅读 README、PROJECT_SOURCE_OF_TRUTH、PROJECT_SCOPE_AND_PAPER_MODULES、INTERFACE_DESIGN。
2. 如果满意，将文档更新作为一个单独 commit。
3. 后续新论文先创建 paper module registry，再接入核心接口。
