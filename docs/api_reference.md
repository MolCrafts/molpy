# MolPy核心模块API参考

本文档提供了MolPy框架中核心模块的完整API参考，包括详细的方法签名、参数说明和使用示例。

## molpy.core.struct

### Entity类

基础实体类，提供字典式的灵活数据存储。

```python
class Entity(UserDict):
```

#### 构造函数

```python
def __init__(self, **kwargs)
```

**参数：**
- `**kwargs`: 初始属性键值对

**示例：**
```python
entity = mp.Entity(name="example", value=42, type="test")
```

#### 方法

##### `clone(self, **modify) -> Entity`

创建实体的深拷贝，可选择性修改属性。

**参数：**
- `**modify`: 要在克隆中修改的键值对

**返回：**
- `Entity`: 新的实体实例

**示例：**
```python
original = mp.Entity(name="test", value=1)
modified = original.clone(value=2, new_attr="added")
```

##### `__call__(self, **modify) -> Entity`

`clone`方法的快捷调用方式。

**参数：**
- `**modify`: 要在克隆中修改的键值对

**返回：**
- `Entity`: 新的实体实例

##### `to_dict(self) -> dict`

将实体转换为标准Python字典。

**返回：**
- `dict`: 实体的字典表示

##### `keys(self)`

返回实体的所有键。

**返回：**
- 键的视图对象

---

### SpatialMixin类

为实体提供空间操作功能的抽象混入类。

```python
class SpatialMixin(ABC):
```

#### 抽象属性

##### `xyz: np.ndarray`

三维坐标属性（抽象属性，需要在子类中实现）。

#### 方法

##### `distance_to(self, other: SpatialMixin) -> float`

计算到另一个空间实体的欧几里得距离。

**参数：**
- `other`: 另一个空间实体

**返回：**
- `float`: 距离值

**示例：**
```python
atom1 = mp.Atom(name="C1", xyz=[0, 0, 0])
atom2 = mp.Atom(name="C2", xyz=[3, 4, 0])
distance = atom1.distance_to(atom2)  # 5.0
```

##### `move(self, vector: ArrayLike) -> SpatialMixin`

按给定向量平移实体。

**参数：**
- `vector`: 平移向量

**返回：**
- `SpatialMixin`: 自身，支持方法链式调用

**示例：**
```python
atom = mp.Atom(name="C1", xyz=[0, 0, 0])
atom.move([1, 2, 3])  # 移动到[1, 2, 3]
```

##### `rotate(self, theta: float, axis: ArrayLike) -> SpatialMixin`

绕指定轴旋转实体。

**参数：**
- `theta`: 旋转角度（弧度）
- `axis`: 旋转轴向量

**返回：**
- `SpatialMixin`: 自身，支持方法链式调用

**示例：**
```python
import numpy as np
atom = mp.Atom(name="C1", xyz=[1, 0, 0])
atom.rotate(np.pi/2, [0, 0, 1])  # 绕z轴旋转90度
```

##### `reflect(self, axis: ArrayLike) -> SpatialMixin`

沿指定轴反射实体。

**参数：**
- `axis`: 反射轴向量（会被自动归一化）

**返回：**
- `SpatialMixin`: 自身，支持方法链式调用

---

### Atom类

表示具有空间坐标的原子的类。

```python
class Atom(Entity, SpatialMixin):
```

#### 构造函数

```python
def __init__(self, name: str = "", xyz: Optional[ArrayLike] = None, **kwargs)
```

**参数：**
- `name`: 原子名称/符号
- `xyz`: 3D坐标
- `**kwargs`: 其他属性（如element, charge, mass等）

**示例：**
```python
atom = mp.Atom(
    name="CA", 
    element="C", 
    xyz=[1.0, 2.0, 3.0],
    charge=0.1,
    mass=12.01
)
```

#### 属性

##### `xyz: np.ndarray`

原子的三维坐标。

**类型：** `np.ndarray`
**形状：** `(3,)`

```python
atom = mp.Atom(name="C1", xyz=[1, 2, 3])
print(atom.xyz)  # [1. 2. 3.]
atom.xyz = [4, 5, 6]  # 设置新坐标
```

##### `name: str`

原子名称（只读属性）。

```python
atom = mp.Atom(name="CA")
print(atom.name)  # "CA"
```

---

### Bond类

表示两个原子之间化学键的类。

```python
class Bond(ManyBody):
```

#### 构造函数

```python
def __init__(self, atom1=None, atom2=None, itom=None, jtom=None, **kwargs)
```

**参数：**
- `atom1`: 键中的第一个原子
- `atom2`: 键中的第二个原子
- `itom`: 第一个原子的别名（向后兼容）
- `jtom`: 第二个原子的别名（向后兼容）
- `**kwargs`: 其他属性（如bond_type, length, order等）

**示例：**
```python
atom1 = mp.Atom(name="C1", xyz=[0, 0, 0])
atom2 = mp.Atom(name="C2", xyz=[1.5, 0, 0])
bond = mp.Bond(atom1, atom2, bond_type="single", order=1)
```

#### 属性

##### `atom1, atom2: Atom`

键中的原子（顺序可能与输入不同，会自动排序以保证一致性）。

##### `itom, jtom: Atom`

`atom1`和`atom2`的别名，用于向后兼容。

##### `length: float`

计算得出的键长（只读属性）。

```python
bond = mp.Bond(atom1, atom2)
print(f"键长: {bond.length:.2f} Å")
```

#### 方法

##### `__eq__(self, other) -> bool`

基于键中原子的相等性比较，与原子顺序无关。

---

### Angle类

表示三个原子之间角度的类。

```python
class Angle(ManyBody):
```

#### 构造函数

```python
def __init__(self, atom1: Atom, vertex: Atom, atom2: Atom, **kwargs)
```

**参数：**
- `atom1`: 角度中的第一个原子
- `vertex`: 顶点原子（角度的中心）
- `atom2`: 角度中的第三个原子
- `**kwargs`: 其他属性

**示例：**
```python
angle = mp.Angle(h1, carbon, h2, angle_type="tetrahedral")
```

#### 属性

##### `atom1, vertex, atom2: Atom`

角度中的原子，`vertex`是中心原子。

##### `value: float`

计算得出的角度值（弧度，只读属性）。

```python
angle = mp.Angle(atom1, vertex_atom, atom2)
angle_degrees = np.degrees(angle.value)
```

---

### Dihedral类

表示四个原子之间二面角的类。

```python
class Dihedral(ManyBody):
```

#### 构造函数

```python
def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom, **kwargs)
```

**参数：**
- `atom1, atom2, atom3, atom4`: 二面角中的四个原子
- `**kwargs`: 其他属性

#### 属性

##### `atom1, atom2, atom3, atom4: Atom`

二面角中的四个原子。

##### `value: float`

计算得出的二面角值（弧度，只读属性）。

---

### Entities类

用于存储和管理分子实体集合的容器类。

```python
class Entities:
```

#### 构造函数

```python
def __init__(self, entities=None)
```

**参数：**
- `entities`: 可选的初始实体列表

#### 方法

##### `add(self, entity)`

向集合中添加实体。

**参数：**
- `entity`: 要添加的实体

**返回：**
- 添加的实体

##### `remove(self, entity)`

从集合中移除实体。

**参数：**
- `entity`: 实体实例、索引或名称

##### `get_by(self, condition: Callable)`

根据条件获取实体。

**参数：**
- `condition`: 接受实体并返回布尔值的函数

**返回：**
- 满足条件的第一个实体，或None

##### `__getitem__(self, key)`

通过索引、名称或切片获取实体。

**参数：**
- `key`: 索引、切片、名称或索引/名称列表

**返回：**
- 实体或实体列表

---

### AtomicStructure类

包含原子、键、角度和二面角的完整分子结构类。

```python
class AtomicStructure(Struct, SpatialMixin, HierarchyMixin):
```

#### 构造函数

```python
def __init__(self, name: str = "", **props)
```

**参数：**
- `name`: 结构名称
- `**props`: 其他属性

#### 属性

##### `atoms: Entities`

结构中的原子集合。

##### `bonds: Entities`

结构中的键集合。

##### `angles: Entities`

结构中的角度集合。

##### `dihedrals: Entities`

结构中的二面角集合。

##### `xyz: np.ndarray`

所有原子的坐标数组。

**形状：** `(n_atoms, 3)`

```python
struct = mp.AtomicStructure()
# ... 添加原子 ...
coords = struct.xyz  # 获取所有原子坐标
struct.xyz = new_coords  # 设置所有原子坐标
```

#### 方法

##### `add_atom(self, atom) -> Atom`

向结构中添加原子。

**参数：**
- `atom`: 要添加的原子

**返回：**
- 添加的原子

##### `def_atom(self, **props) -> Atom`

创建并添加具有给定属性的原子。

**参数：**
- `**props`: 原子属性

**返回：**
- 创建的原子

**示例：**
```python
struct = mp.AtomicStructure()
atom = struct.def_atom(name="C1", element="C", xyz=[0, 0, 0])
```

##### `add_bond(self, bond) -> Bond`

向结构中添加键。

##### `def_bond(self, atom1, atom2, **kwargs) -> Bond`

创建并添加两个原子之间的键。

**参数：**
- `atom1, atom2`: 原子实例或索引
- `**kwargs`: 键属性

**返回：**
- 创建的键

##### `add_angle(self, angle) -> Angle`

向结构中添加角度。

##### `add_dihedral(self, dihedral) -> Dihedral`

向结构中添加二面角。

##### `remove_atom(self, atom)`

从结构中移除原子。

##### `remove_bond(self, bond)`

从结构中移除键。

##### `add_struct(self, struct) -> AtomicStructure`

将另一个结构添加到当前结构中（合并原子和拓扑）。

**参数：**
- `struct`: 要添加的结构

**返回：**
- 自身，支持方法链式调用

#### 类方法

##### `concat(cls, name: str, structs) -> AtomicStructure`

将多个结构连接成新结构。

**参数：**
- `name`: 新结构的名称
- `structs`: 要连接的结构序列

**返回：**
- 包含所有输入结构的新结构

---

### MolecularStructure类

完整的分子结构实现类。

```python
class MolecularStructure(AtomicStructure):
```

这是表示完整分子系统的主要类，包含原子、键、角度和二面角。继承了`AtomicStructure`的所有功能。

---

## molpy.core.frame

### Frame类

基于xarray的高效表格式数据存储和操作容器。

```python
class Frame(MutableMapping):
```

#### 构造函数

```python
def __init__(self, **data)
```

**参数：**
- `**data`: 初始数据字典

**示例：**
```python
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1'],
        'element': ['C', 'C', 'O'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0]]
    }
)
```

#### 方法

##### `__getitem__(self, key)`

获取数据数组。

**参数：**
- `key`: 数据键名

**返回：**
- `xr.DataArray`: 对应的数据数组

##### `__setitem__(self, key, value)`

设置数据数组。

**参数：**
- `key`: 数据键名
- `value`: 数据值（字典或DataArray）

##### `to_struct(self) -> AtomicStructure`

将Frame转换为AtomicStructure。

**返回：**
- `AtomicStructure`: 转换后的结构

---

## 示例用法

### 基础分子构建

```python
import molpy as mp
import numpy as np

# 创建分子结构
water = mp.AtomicStructure(name="water")

# 添加原子
o = water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
h1 = water.def_atom(name="H1", element="H", xyz=[0.757, 0.586, 0.0])
h2 = water.def_atom(name="H2", element="H", xyz=[-0.757, 0.586, 0.0])

# 添加键
water.def_bond(o, h1, bond_type="covalent")
water.def_bond(o, h2, bond_type="covalent")

# 添加角度
angle = mp.Angle(h1, o, h2, angle_type="bent")
water.add_angle(angle)
```

### 空间操作

```python
# 移动分子
water.xyz += [1, 0, 0]  # 沿x轴移动1单位

# 旋转单个原子
h1.rotate(np.pi/4, [0, 0, 1])  # 绕z轴旋转45度
```

### Frame操作

```python
# 创建Frame
frame = mp.Frame(atoms={
    'name': ['C1', 'C2', 'N1'],
    'element': ['C', 'C', 'N'],
    'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0]]
})

# 转换为结构
struct = frame.to_struct()
```

**参数：**
- `theta`: 旋转角度（弧度）
- `axis`: 旋转轴向量

**返回：**
- `SpatialMixin`: 自身，支持方法链式调用

##### `reflect(self, axis: ArrayLike) -> SpatialMixin`

沿指定轴反射实体。

**参数：**
- `axis`: 反射轴向量（将被归一化）

**返回：**
- `SpatialMixin`: 自身，支持方法链式调用

---

### Atom类

表示具有空间坐标的原子。

```python
class Atom(Entity, SpatialMixin):
```

#### 构造函数

```python
def __init__(self, name: str = "", xyz: Optional[ArrayLike] = None, **kwargs)
```

**参数：**
- `name`: 原子名称/符号
- `xyz`: 三维坐标
- `**kwargs`: 其他属性

**示例：**
```python
atom = mp.Atom(name="C1", element="C", xyz=[0.0, 0.0, 0.0], charge=0.0)
```

#### 属性

##### `xyz: np.ndarray`

获取或设置原子的三维坐标。

**类型：** `np.ndarray`，形状为(3,)

**异常：**
- `ValueError`: 如果设置的坐标不是3维向量

##### `name: str`

获取原子名称。

**类型：** `str`

---

### ManyBody类

涉及多个原子的实体的基类。

```python
class ManyBody(Entity):
```

#### 构造函数

```python
def __init__(self, *atoms: Atom, **kwargs)
```

**参数：**
- `*atoms`: 参与的原子
- `**kwargs`: 其他属性

**异常：**
- `TypeError`: 如果任何参数不是Atom实例

#### 属性

##### `atoms: tuple[Atom, ...]`

获取参与实体的原子元组。

**类型：** `tuple[Atom, ...]`

---

### Bond类

表示两个原子之间的化学键。

```python
class Bond(ManyBody):
```

#### 构造函数

```python
def __init__(self, atom1: Atom, atom2: Atom, **kwargs)
```

**参数：**
- `atom1`: 第一个原子
- `atom2`: 第二个原子
- `**kwargs`: 化学键属性（如bond_type, length等）

**异常：**
- `ValueError`: 如果两个原子是同一个实例

**示例：**
```python
bond = mp.Bond(atom1, atom2, bond_type="single", length=1.5)
```

#### 属性

##### `atom1: Atom`

第一个原子。

##### `atom2: Atom`

第二个原子。

##### `itom: Atom`

`atom1`的别名（向后兼容）。

##### `jtom: Atom`

`atom2`的别名（向后兼容）。

##### `length: float`

计算化学键长度。

**类型：** `float`

---

### Angle类

表示三个原子之间的角度。

```python
class Angle(ManyBody):
```

#### 构造函数

```python
def __init__(self, atom1: Atom, vertex: Atom, atom2: Atom, **kwargs)
```

**参数：**
- `atom1`: 第一个原子
- `vertex`: 顶点原子（角度中心）
- `atom2`: 第三个原子
- `**kwargs`: 角度属性

**异常：**
- `ValueError`: 如果三个原子不是唯一的

#### 属性

##### `atom1: Atom`

第一个原子。

##### `vertex: Atom`

顶点原子（角度中心）。

##### `atom2: Atom`

第三个原子。

##### `value: float`

计算角度值（弧度）。

**类型：** `float`

#### 方法

##### `to_dict(self) -> dict`

将角度转换为字典，包含原子ID信息。

---

### Dihedral类

表示四个原子之间的二面角。

```python
class Dihedral(ManyBody):
```

#### 构造函数

```python
def __init__(self, atom1: Atom, atom2: Atom, atom3: Atom, atom4: Atom, **kwargs)
```

**参数：**
- `atom1`: 第一个原子
- `atom2`: 第二个原子
- `atom3`: 第三个原子
- `atom4`: 第四个原子
- `**kwargs`: 二面角属性

**异常：**
- `ValueError`: 如果四个原子不是唯一的

#### 属性

##### `atom1: Atom`, `atom2: Atom`, `atom3: Atom`, `atom4: Atom`

二面角的四个原子。

##### `value: float`

计算二面角值（弧度）。

**类型：** `float`

---

### Entities类

用于存储和管理分子实体集合的容器。

```python
class Entities:
```

#### 构造函数

```python
def __init__(self, entities=None)
```

**参数：**
- `entities`: 可选的初始实体序列

#### 方法

##### `add(self, entity) -> entity`

向集合添加实体。

**参数：**
- `entity`: 要添加的实体

**返回：**
- 添加的实体

##### `remove(self, entity)`

从集合移除实体。

**参数：**
- `entity`: 实体实例、索引或名称

##### `get_by(self, condition: Callable[[Any], bool])`

根据条件获取实体。

**参数：**
- `condition`: 接受实体并返回布尔值的函数

**返回：**
- 满足条件的第一个实体，或None

##### `extend(self, entities)`

用多个实体扩展集合。

**参数：**
- `entities`: 要添加的实体序列

##### `__len__(self) -> int`

返回集合中实体的数量。

##### `__getitem__(self, key)`

通过索引、名称或多个键获取实体。

**参数：**
- `key`: 索引、切片、名称或索引/名称序列

**返回：**
- 实体或实体列表

---

### HierarchyMixin类

提供层次结构管理功能的混入类。

```python
class HierarchyMixin:
```

#### 属性

##### `parent`

获取父结构。

##### `children`

获取子结构列表的副本。

##### `root`

获取层次结构的根结构。

##### `is_root: bool`

检查是否为根结构。

##### `is_leaf: bool`

检查是否为叶子结构（无子结构）。

##### `depth: int`

获取在层次结构中的深度（根为0）。

#### 方法

##### `add_child(self, child) -> child`

添加子结构。

**参数：**
- `child`: 要添加的子结构

**返回：**
- 添加的子结构

##### `remove_child(self, child)`

移除子结构。

**参数：**
- `child`: 要移除的子结构

##### `get_descendants(self) -> list`

获取所有后代结构（子、孙等）。

##### `get_ancestors(self) -> list`

获取所有祖先结构（父、祖父等）。

##### `get_siblings(self) -> list`

获取所有兄弟结构。

##### `traverse_dfs(self, visit_func)`

使用深度优先搜索遍历层次结构。

**参数：**
- `visit_func`: 为每个节点调用的函数

##### `traverse_bfs(self, visit_func)`

使用广度优先搜索遍历层次结构。

**参数：**
- `visit_func`: 为每个节点调用的函数

##### `find_by_name(self, name: str)`

在层次结构中按名称查找结构。

**参数：**
- `name`: 要搜索的名称

**返回：**
- 具有给定名称的第一个结构，或None

##### `get_hierarchy_info(self) -> dict`

获取层次结构信息。

**返回：**
- 包含层次结构信息的字典

---

### AtomicStructure类

包含原子、化学键、角度和二面角的结构。

```python
class AtomicStructure(Struct, SpatialMixin, HierarchyMixin):
```

#### 构造函数

```python
def __init__(self, name: str = "", **props)
```

**参数：**
- `name`: 结构名称
- `**props`: 其他属性

#### 属性

##### `atoms: Entities`

结构中的原子集合。

##### `bonds: Entities`

结构中的化学键集合。

##### `angles: Entities`

结构中的角度集合。

##### `dihedrals: Entities`

结构中的二面角集合。

##### `xyz: np.ndarray`

所有原子的坐标数组。

**类型：** `np.ndarray`，形状为(n_atoms, 3)

#### 方法

##### `add_atom(self, atom: Atom) -> Atom`

向结构添加原子。

**参数：**
- `atom`: 要添加的原子

**返回：**
- 添加的原子

##### `def_atom(self, **props) -> Atom`

创建并添加具有给定属性的原子。

**参数：**
- `**props`: 原子属性

**返回：**
- 创建的原子

##### `add_atoms(self, atoms)`

向结构添加多个原子。

**参数：**
- `atoms`: 要添加的原子序列

##### `add_bond(self, bond: Bond) -> Bond`

向结构添加化学键。

**参数：**
- `bond`: 要添加的化学键

**返回：**
- 添加的化学键

##### `def_bond(self, atom1, atom2, **kwargs) -> Bond`

创建并添加两个原子之间的化学键。

**参数：**
- `atom1`: 第一个原子或其索引
- `atom2`: 第二个原子或其索引
- `**kwargs`: 化学键属性

**返回：**
- 创建的化学键

**异常：**
- `TypeError`: 如果参数不是Atom实例或有效索引

##### `add_struct(self, struct) -> self`

将另一个结构添加到当前结构。

**参数：**
- `struct`: 要添加的结构

**返回：**
- 自身，支持方法链式调用

##### `get_atom_by(self, condition: Callable[[Atom], bool])`

根据条件获取原子。

**参数：**
- `condition`: 接受原子并返回布尔值的函数

**返回：**
- 满足条件的第一个原子，或None

#### 类方法

##### `concat(cls, name: str, structs) -> AtomicStructure`

将多个结构连接成新结构。

**参数：**
- `name`: 新结构的名称
- `structs`: 要连接的结构序列

**返回：**
- 包含所有输入结构的新结构

---

### MolecularStructure类

完整的分子结构实现。

```python
class MolecularStructure(AtomicStructure):
```

这是表示完整分子系统的主要类，具有原子、化学键、角度和二面角。

---

## molpy.core.frame

### Frame类

高效的表格式数据存储和操作，基于xarray构建。

```python
class Frame(dict):
```

#### 构造函数

```python
def __init__(self, **kwargs)
```

**参数：**
- `**kwargs`: 数据映射，键为数据类型（如'atoms', 'bonds'），值为数据字典

**示例：**
```python
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2'],
        'element': ['C', 'C'],
        'xyz': [[0,0,0], [1.5,0,0]]
    },
    bonds={
        'i': [0],
        'j': [1],
        'bond_type': ['single']
    }
)
```

#### 方法

##### `to_struct(self, struct_class=None, **kwargs)`

将Frame转换为结构对象。

**参数：**
- `struct_class`: 目标结构类，默认为MolecularStructure
- `**kwargs`: 传递给结构构造函数的额外参数

**返回：**
- 结构对象

#### 类方法

##### `from_struct(cls, struct, include_bonds=True, **kwargs)`

从结构对象创建Frame。

**参数：**
- `struct`: 源结构对象
- `include_bonds`: 是否包含化学键信息
- `**kwargs`: 额外参数

**返回：**
- Frame对象

---

## 工具函数

### `_dict_to_dataarray(data: Dict[str, np.ndarray]) -> xr.DataArray`

将数组字典转换为xarray.DataArray。

**参数：**
- `data`: 将变量名映射到numpy数组或标量的字典

**返回：**
- `xr.DataArray`: 以数组/标量作为坐标的DataArray

**异常：**
- `ValueError`: 如果非标量数组具有不同的第一维长度

---

## 异常

### 常见异常类型

- `TypeError`: 传递了错误类型的参数
- `ValueError`: 传递了正确类型但无效值的参数
- `AttributeError`: 访问不存在的属性

---

## 使用模式

### 基本使用模式

```python
import molpy as mp
import numpy as np

# 创建原子
atom1 = mp.Atom(name="C1", element="C", xyz=[0, 0, 0])
atom2 = mp.Atom(name="C2", element="C", xyz=[1.5, 0, 0])

# 创建结构
struct = mp.AtomicStructure(name="ethane_fragment")
struct.add_atom(atom1)
struct.add_atom(atom2)

# 添加化学键
bond = mp.Bond(atom1, atom2, bond_type="single")
struct.add_bond(bond)

# 空间操作
struct.move([1, 0, 0])  # 平移整个结构
struct.rotate(np.pi/4, [0, 0, 1])  # 旋转结构
```

### Frame操作模式

```python
# 创建Frame
frame_data = {
    'atoms': {
        'name': ['C1', 'H1', 'H2'],
        'element': ['C', 'H', 'H'],
        'xyz': [[0,0,0], [1,0,0], [-1,0,0]],
        'mass': [12.01, 1.008, 1.008]
    }
}

frame = mp.Frame(**frame_data)

# 转换为结构
struct = mp.MolecularStructure.from_frame(frame)

# 转换回Frame
new_frame = struct.to_frame()
```

### 层次结构模式

```python
# 创建蛋白质层次结构
protein = mp.AtomicStructure(name="protein")
chain = mp.AtomicStructure(name="chain_A")
residue = mp.AtomicStructure(name="ALA_1")

protein.add_child(chain)
chain.add_child(residue)

# 添加原子到残基
ca_atom = residue.def_atom(name="CA", element="C", xyz=[0,0,0])

# 遍历层次结构
protein.traverse_dfs(lambda node: print(node.get('name')))
```

---

此API参考提供了MolPy核心模块的完整接口文档。所有类和方法都包含详细的参数说明、返回值类型和使用示例。
