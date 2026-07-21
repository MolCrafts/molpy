# 编码风格

MolPy 讲求显式而非巧妙。代码应当不依赖外部上下文，能直接读懂、独立测试、没有隐藏的副作用。


## 不可变性

核心规则：变换操作返回新对象，输入对象绝不就地修改。

```python
# 错误：修改了输入
def add_hydrogens(mol):
    mol.add_atoms(...)

# 正确：返回新对象
def add_hydrogens(mol):
    new_mol = mol.copy()
    # ... 填充 new_mol
    return new_mol
```

`typify()` 返回新的 `Atomistic`。`Struct.copy()` 深度拷贝实体与连接。新代码也应遵循这些模式。


## 函数与文件

函数不超过 50 行，只做一件事。文件不超过 800 行。模块超出此限制时，把内聚相关的部分抽取到新文件中。


## 代码标识符

函数和变量用 `snake_case`，类用 `PascalCase`，常量用 `UPPER_CASE`。规范字段名（`element`、`charge`、`mol_id`）在[命名约定](../tutorials/naming-conventions.md)中定义，开发者文档和扩展示例不应引入局部别名。MolPy 专有术语要使用规范含义：`topology` 指键图而非"连接"，`atom type` 指力场标识符而非"种类"，`struct` 指 `Struct` 基类而非泛指的"结构"。


## 类型注解

公开 API 都要加类型注解。签名不明确时，私有辅助函数也一样。用 `from __future__ import annotations` 处理前向引用。


## 导入

导入顺序：标准库、第三方包、`molpy`，各组用空行隔开。`molpy` 内部用绝对导入（`from molpy import Frame`，不用相对导入）。


## 错误处理

在公开函数的入口处验证输入。抛出具体异常（`ValueError`、`TypeError`、`FileNotFoundError`），消息中写明实际值和期望条件。绝不静默忽略异常。


## 文档字符串

公开函数和类用 Google 风格文档字符串，包含 `Args`、`Returns`、`Raises` 部分。物理量务必注明单位（Å、kcal/mol、弧度），数组参数说明期望形状。


## 格式化

格式化和 lint 统一用 Ruff。提交前运行 `ruff format src tests` 和 `ruff check src`，pre-commit 钩子会自动检查。


## 提交就绪检查清单

以下条件全部满足，变更才算就绪：

- [ ] 代码无需额外解释即可读懂
- [ ] 函数在 50 行以内
- [ ] 不修改输入对象
- [ ] 测试覆盖变更的行为
- [ ] 公开 API 包含类型注解和文档字符串
- [ ] `ruff format --check src tests` 通过
- [ ] `ruff check src` 通过
- [ ] `pre-commit run --all-files` 通过
