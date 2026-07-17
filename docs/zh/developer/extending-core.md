# 扩展数据模型

MolPy 的图数据模型直接由 molrs 支撑。`Atomistic` 与 `CoarseGrain` 是 native
world；`Atom`、`Bond`、`Bead` 等 Python 对象是稳定 handle 的实时视图。

!!! note "实现前先讨论"
    新增可存储的节点或关系种类会同时改变 molrs schema、Rust 图算法、Python
    绑定、Frame 投影和 I/O formatter。实现前请先提交
    [GitHub issue](https://github.com/MolCrafts/molpy/issues)。

## Python 可以扩展什么

Python 子类可以围绕已有 native 类型添加无状态便利层：构造器别名、选择辅助、
callback、展示方法和格式相关序列化。不得创建第二份属性存储、端点列表或 handle
注册表。

所有数据都通过 graph factory 创建：

```python
from molpy import Atomistic

mol = Atomistic(name="water")
oxygen = mol.def_atom(element="O", x=0.0, y=0.0, z=0.0)
hydrogen = mol.def_atom(element="H", x=0.96, y=0.0, z=0.0)
bond = mol.def_bond(oxygen, hydrogen)

assert mol.atoms[0] is oxygen
assert bond.atoms == (oxygen, hydrogen)
```

通过 ref 写入会立即更新 native world；`.atoms`、`.bonds`、`.impropers` 等集合
属性都是 molrs 实时视图。

## 新增可存储图类型

系统有意不提供 `TypeBucket.register_type` 钩子。真正的新类型需要依次完成：

1. 在 molrs 定义存储和关系 arity。
2. 让 native copy/merge/extract/topology 与 Frame 投影识别它。
3. 在 `molrs.views` 暴露 handle view 与 graph factory。
4. 从 `molpy.core` 重导出同一个对象；Python 侧只增加语法糖。
5. 更新相关 reader/writer，并补 Rust、绑定与 molpy 集成测试。

如果概念只是标注，优先给已有节点或关系增加 typed field。例如组装 site 使用
`fields.SITE`，不需要新的节点类。

## 检查清单

- [ ] 数据在 molrs 中只有一个 owner；没有 Python 镜像或 `_inner` facade。
- [ ] 实时视图中的 handle 稳定，写入会更新 world。
- [ ] native copy/merge/extract 与 Frame round trip 覆盖新类型。
- [ ] molpy 使用 API 前已更新 PyO3 export 与类型 stub。
- [ ] MolPy 只 re-export native 类型或添加真正的 native 子类。
- [ ] Rust、molrs-python 与 molpy 集成测试全部通过。
