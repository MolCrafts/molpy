# 开发者指南

MolPy 是一个研究软件库，本文档面向有意贡献的开发者。根据你的目标快速定位：

- **修 Bug 或提第一个 PR** → 从[开发环境搭建](development-setup.md)开始，再看[贡献工作流](contributing.md)和[测试](testing.md)
- **加分析操作、文件格式或工具集成** → 用[扩展 MolPy](extending-compute.md) 里的插件配方，不动核心
- **改数据模型或力场内部** → 先读[架构概览](architecture-overview.md)，再到 GitHub 开 issue 讨论，然后看[扩展数据模型](extending-core.md)或[扩展力场](extending-forcefield.md)
- **了解 MolPy 怎么造出来的** → [架构概览](architecture-overview.md)和 [molrs 后端](molrs-backend.md)
- **发版本** → [发布流程](release-process.md)

教程附录[命名约定](../tutorials/naming-conventions.md)里列出了规范字段名和拓扑键。

## 贡献

日常开发流程：

- [开发环境搭建](development-setup.md) — 仓库克隆、可编辑安装、从源码构建 molrs、跑测试
- [贡献工作流](contributing.md) — PR 工作流、提交信息规范、pre-commit 钩子
- [编码风格](coding-style.md) — 标识符风格、格式化要求、可变性约定
- [测试](testing.md) — pytest 惯例、测试标记、覆盖率要求、本地测试和外部测试的区别
- [发布流程](release-process.md) — molpy/molrs 共用版本线、变更日志维护、CI 驱动的包发布
- [第三方归属](attribution.md) — 移植代码和参数数据使用的许可证说明

## 架构

扩展功能前需要了解的设计背景：

- [架构概览](architecture-overview.md) — 模块职责、图与表格两层、格式化器层次、可变性约定、构建循环性能模型
- [molrs 后端](molrs-backend.md) — Rust 列存储和计算内核在 Python 中的呈现：盒子、近邻列表、RDF 和其他分析功能

## 扩展 MolPy

按从外到内的顺序（从插件接口到核心改动）：

- [添加计算操作](extending-compute.md) — 通过 `Compute` 协议编写可复用的分析操作
- [添加 I/O 格式](extending-io.md) — 读写器基类和 `FieldFormatter` 规范化接口
- [添加包装器或适配器](extending-integration.md) — 子进程包装器约定和内存态适配器模式
- [扩展数据模型](extending-core.md) — 新增 `Entity` 和 `Link` 子类型、自定义 `Struct` 子类、身份哈希的不变性
- [扩展力场](extending-forcefield.md) — molrs 内核、命名的 `Style` 类、导出时的格式化器注册

## Issue 与讨论

- Bug 报告和功能请求：<https://github.com/MolCrafts/molpy/issues>
- 设计讨论：<https://github.com/MolCrafts/molpy/discussions>
- 智能体辅助开发：[MCP 套件](../user-guide/15_mcp.md) 向 LLM 智能体暴露 MolPy 的符号索引
