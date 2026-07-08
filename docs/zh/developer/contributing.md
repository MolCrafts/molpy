# 参与贡献

本文介绍为 MolPy 贡献代码、测试和文档的基本流程。

## 开始之前

先查查[已有议题](https://github.com/MolCrafts/molpy/issues)，看看有没有人在做同样的事。再去读一遍[行为准则](https://github.com/MolCrafts/molpy/blob/master/CODE_OF_CONDUCT.md)。

## 工作流程

Fork 仓库、克隆到本地、从 `master` 切出分支。分支名标明改动类型：`feature/morse-potential`、`fix/pdb-reader-crash`、`docs/typifier-guide`。

改代码的同时补上测试。推送前跑一遍本地检查：

```bash
ruff format --check src tests
ruff check src
pytest tests/ -v -m "not external"
```

提交 Pull Request，摘要写清楚。PR 描述要交代改了什么、为什么改、怎么验证。

## 拉取请求检查清单

- [ ] 范围集中 —— 一个 PR 只做一个逻辑变更
- [ ] 新功能有对应测试
- [ ] 公共 API 变更附带类型提示和 docstring
- [ ] 不兼容变更已在描述中注明
- [ ] 提交信息清晰（`feat:`、`fix:`、`refactor:`、`docs:`、`test:`）

## 什么样的 PR 描述才算好

好的 PR 描述能回答五个问题：

1. **解决了什么问题？** —— 关联议题，或说清痛点
2. **改了什么？** —— 概述代码变更，不是贴逐行 diff
3. **为什么选这个方案？** —— 解释设计取舍，尤其是有替代方案时
4. **有什么风险？** —— 标出破坏性变更、性能影响或边界情况
5. **怎么验证的？** —— 说明测试策略，可以附上输出或截图

## 文档期望

改了行为就更新 `docs/` 里的对应文档。三个层级，标准不同：

- **概念**（面向人类）—— 叙事性讲解，先讲背景再贴代码
- **指南**（面向人类）—— 完整工作流，附可运行代码
- **API 参考**（面向智能体）—— 快速查阅表 + mkdocstrings 自动生成

新增公共符号就加到对应 API 页面。改了面向用户的行为就更新相关指南或概念页面。
