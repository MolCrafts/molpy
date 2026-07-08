# 扩展数据模型

本文介绍如何向 MolPy 核心数据模型添加新的实体类型、连接类型和结构子类。

!!! note "实现前先讨论"
    此层改动会波及 `TypeBucket` 注册、复制语义和格式化器分发。请在实现前打开 [GitHub issue](https://github.com/MolCrafts/molpy/issues) 描述扩展内容；[架构概览](architecture-overview.md) 解释了相关的约束条件。

## 架构回顾

MolPy 数据模型分三层：

1. **Entity（节点）** — `UserDict` 的子类，使用身份标识哈希。例如 `Atom`、`Bead`。
2. **Link（边）** — 持有有序的 `Entity` 端点元组。例如 `Bond`、`Angle`、`Dihedral`。
3. **Struct（容器）** — 持有 `TypeBucket[Entity]` 和 `TypeBucket[Link]`，负责 CRUD 操作。例如 `Atomistic`、`CoarseGrain`。

`TypeBucket` 按具体类型存储条目。注册 `Atom` 后，`bucket[Atom]` 返回所有 `Atom` 实例，子类也会包含在父类查询结果中。

## 添加新的 Entity 类型

继承 `Entity` 即可。基类已提供字典风格存储和基于身份标识的哈希，无需额外方法。

```python
from molpy.core.entity import Entity

class VirtualSite(Entity):
    """一个无质量的相互作用位点（例如 TIP4P 氧孤对电子）。"""

    def __repr__(self) -> str:
        name = self.data.get("name", id(self))
        return f"<VirtualSite: {name}>"
```

## 添加新的 Link 类型

继承 `Link`，在 `__init__` 中约束端点数量和类型，添加命名的端点属性来提高可读性。

```python
from molpy.core.entity import Link
from molpy.core.atomistic import Atom

class Improper(Link):
    """Improper 二面角：一个中心原子与三个外围原子。"""

    def __init__(self, center: Atom, a: Atom, b: Atom, c: Atom, /, **attrs):
        super().__init__([center, a, b, c], **attrs)

    @property
    def center(self) -> Atom:
        return self.endpoints[0]

    @property
    def outer(self) -> tuple[Atom, Atom, Atom]:
        return self.endpoints[1], self.endpoints[2], self.endpoints[3]
```

## 在 Struct 中注册类型

新的实体和连接类型必须在结构体的 `__init__` 中注册，这样 `TypeBucket` 才会为其分配桶。未注册时，`bucket[MyType]` 返回空列表。

```python
from molpy.core.entity import Struct, MembershipMixin, ConnectivityMixin

class ExtendedAtomistic(Atomistic):
    """支持虚拟位点和 Improper 二面角的 Atomistic。"""

    def __init__(self, **props):
        super().__init__(**props)
        self.entities.register_type(VirtualSite)
        self.links.register_type(Improper)

    @property
    def virtual_sites(self):
        return self.entities[VirtualSite]

    @property
    def impropers(self):
        return self.links[Improper]

    def def_virtual_site(self, **attrs) -> VirtualSite:
        vs = VirtualSite(**attrs)
        self.entities.add(vs)
        return vs

    def def_improper(self, center, a, b, c, /, **attrs) -> Improper:
        imp = Improper(center, a, b, c, **attrs)
        self.links.add(imp)
        return imp
```

## TypeBucket 的工作原理

`TypeBucket` 用 `get_nearest_type(item)` 确定桶的键（即条目的具体类）。关键行为：

- `bucket.add(item)` — 将条目加入其具体类型对应的桶，已存在则跳过（基于身份标识判断）
- `bucket[SomeType]` — 返回 `SomeType` 及其所有子类的条目
- `bucket.register_type(SomeType)` — 确保桶存在，这样 `.atoms` 返回 `[]` 而不是抛异常
- `bucket.remove(item)` — 通过身份标识（`is`）移除，不用相等性判断

## Struct.copy() 与新类型

`Struct.copy()` 会深拷贝所有实体和连接，并重新映射端点引用。只要新的 `Link` 子类的 `__init__` 能接受 `(*endpoints, **attrs)` 或 `(endpoints, **attrs)` 两种签名之一，这一机制就能正常工作。复制逻辑先尝试位置参数，不行再试列表形式。

如果 Link 子类用了不同的构造器签名（比如命名参数），就需要在 Struct 子类中覆盖 `copy()`。

## 检查清单

- [ ] Entity 子类：`class MyEntity(Entity)` 并实现 `__repr__`
- [ ] Link 子类：`class MyLink(Link)` 并添加端点断言和属性
- [ ] 在 Struct 的 `__init__` 中注册：`self.entities.register_type(MyEntity)`
- [ ] 在 Struct 上添加 `def_*` 工厂方法
- [ ] 添加属性访问器（例如 `@property def my_links`）
- [ ] 验证 `Struct.copy()` 能正确处理新类型
- [ ] 在 `tests/test_core/` 中编写测试
