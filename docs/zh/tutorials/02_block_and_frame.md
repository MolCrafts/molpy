# Block 与 Frame

化学结构一旦定型，重心就从图编辑转向数值计算。这时候需要紧凑对齐的表格：`Block` 是一张基于 NumPy 的列式表，`Frame` 是整个系统的快照。

## 为什么需要两种表示？

分子动力学引擎（LAMMPS、GROMACS、OpenMM）读的不是分子图，是坐标、原子类型、按索引组织的拓扑关系——全是扁平表格。反过来，用 SMILES 造分子或组装聚合物，靠的是图遍历，不是数组切片。

MolPy 明确区分这两套表示，不在背后自动做转换。`Atomistic` 是编辑用的图；`Block` 和 `Frame` 是导出用的表格。拿其他工具类比：`Frame` 做的事类似 LAMMPS 数据文件、GROMACS 的 `.gro` + `.top` 文件对，或者 MDAnalysis 的 `Universe` 快照——区别是它活在内存里，不绑定文件格式。

## 从图到表格

`Atomistic` 用来编辑化学结构：加原子、删键、查邻接。结构一旦确定，后续就是数值计算——算距离、算能量、导出文件。这类事情用紧凑对齐的数组，比用字典对象搭的图顺手得多。

MolPy 提供两个工具来解决。`Block` 是一张列式表：列名映射到 NumPy 数组，每列行数相同。`Frame` 是命名的 Block 集合，外加自由格式的元数据，表示一个完整的系统快照。

这样拆分是有意为之。Block 回答"原子有哪些？""键有哪些？"——每类数据一张表。Frame 回答"这个系统此刻的完整状态是什么？"——把相关的表捆在一起。


## Block：基于 NumPy 的列式表格

创建 Block 只需传一个字典，值用类数组对象即可。每个值自动转成 NumPy 数组。

```python
import molpy as mp
import numpy as np

atoms = mp.Block({
    "element": ["O", "H", "H"],
    "x": [0.000, 0.957, -0.239],
    "y": [0.000, 0.000, 0.927],
    "z": [0.000, 0.000, 0.000],
})

print(atoms.nrows)          # 3
print(list(atoms.keys()))   # ['element', 'x', 'y', 'z']
```

读取一列返回 `np.ndarray`，NumPy 的全部功能直接可用，不需要转换步骤或特殊取数方法。

```python
print(atoms["x"].dtype)         # float64
print(atoms["element"].dtype)   # <U1 (Unicode 字符串)
```

常见的做法是把数值列堆成二维数组，方便做向量化计算。

```python
xyz = atoms[["x", "y", "z"]]   # shape (3, 3)
r = np.linalg.norm(xyz, axis=1)
print(r)
```


## 行选择返回新的 Block

切片、布尔掩码、花式索引都返回新的 `Block`，原对象不受影响。

```python
hydrogens = atoms[atoms["element"] == "H"]
print(hydrogens.nrows)           # 2
print(hydrogens["x"])            # [0.957, -0.239]

first_two = atoms[0:2]
print(first_two["element"])      # ['O', 'H']
```

取单个标量值时，先取列，再取行。

```python
print(atoms["x"][0])   # 0.0
```


## 添加与删除列

赋值一个新键就是插入或覆盖一列。用 `del` 删键就是移除该列。操作方式和 Python 字典一致。

```python
atoms_with_r = atoms.copy()
atoms_with_r["r"] = np.linalg.norm(atoms_with_r[["x", "y", "z"]], axis=1)
print(list(atoms_with_r.keys()))   # ['element', 'x', 'y', 'z', 'r']

del atoms_with_r["r"]
print(list(atoms_with_r.keys()))   # ['element', 'x', 'y', 'z']
```


## 重命名列

`Block.rename()` 就地改列名。I/O 格式化系统内部用它做字段名转换——把格式特定的名字换成规范名。

```python
b = mp.Block({"q": [0.1, -0.2], "x": [1.0, 2.0]})
b.rename("q", "charge")
print(list(b.keys()))   # ['x', 'charge']
```


## 浅拷贝与深拷贝

`Block.copy()` 是浅拷贝：只复制映射关系，底层的 NumPy 数组还是共享的。对数组做原位修改会影响原对象和拷贝。

```python
shallow = atoms.copy()
shallow["x"][0] = 999.0
print(atoms["x"][0])    # 999.0 — 原始对象也被改变了！
```

要完全独立的话，需要显式复制数组。最安全的做法是，打算改哪列就复制哪列：

```python
# 为本页后续内容重建干净数据
atoms = mp.Block({
    "element": ["O", "H", "H"],
    "x": [0.000, 0.957, -0.239],
    "y": [0.000, 0.000, 0.927],
    "z": [0.000, 0.000, 0.000],
})

deep = atoms.copy()
deep["x"] = deep["x"].copy()
deep["x"][0] = 999.0
print(atoms["x"][0])    # 0.0 — 原始对象未改变
```

!!! tip "避免原位修改"
    MolPy 推荐的做法是避免对数组做原位修改。与其改一列，不如赋一个新值：`block["x"] = block["x"] + 1.0`。这样每次都是一份独立拷贝，也符合 MolPy 不可变数据的原则。


## Frame：Block 的命名集合

一个分子系统往往不止一张表。原子坐标是一张表，键索引是另一张表，快照本身还有时间步、描述、来源等元数据。`Frame` 把所有这些塞到一个对象里。

```python
frame = mp.Frame(
    blocks={
        "atoms": {
            "element": ["O", "H", "H"],
            "x": [0.000, 0.957, -0.239],
            "y": [0.000, 0.000, 0.927],
            "z": [0.000, 0.000, 0.000],
        },
        "bonds": {
            "atomi": [0, 0],
            "atomj": [1, 2],
        },
    },
    timestep=0,
    description="water",
)
```

`blocks` 之外的关键字参数都存到 `frame.metadata` 里，就是个普通字典。

```python
print(frame.metadata)   # {'timestep': 0, 'description': 'water'}
```

按名字拿到的块就是 `Block` 对象，之前学的所有列操作照用。

```python
atoms = frame["atoms"]
print(atoms["x"])   # [0.000, 0.957, -0.239]
```

可以随时添加、替换或删除块。

```python
frame["tags"] = {"label": ["oxygen", "hydrogen", "hydrogen"]}
print(type(frame["tags"]))   # <class 'molpy.core.frame.Block'>

del frame["tags"]
print("tags" in frame)       # False
```


## Box 是一级属性

周期性模拟盒直接挂在 `frame.box` 上，不塞进元数据里。这样 `Frame.copy()` 会带着盒子走，I/O 读写也不会丢。

```python
frame.box = mp.Box.cubic(20.0)
print(frame.box.lengths)   # [20. 20. 20.]

# copy() 保留盒子信息
frame2 = frame.copy()
print(frame2.box.lengths)   # [20. 20. 20.]
```

没设盒子时（比如孤立分子），`frame.box` 是 `None`。


## 序列化通过字典往返

`Block` 和 `Frame` 都有 `to_dict()` 和 `from_dict()`，用来做 JSON 友好的序列化。这是持久化或传输系统状态的稳定方式，不依赖具体文件格式。

```python
payload = frame.to_dict()
print(sorted(payload.keys()))   # ['blocks', 'metadata']

restored = mp.Frame.from_dict(payload)
print(sorted(restored.to_dict()["blocks"].keys()))   # ['atoms', 'bonds']
```


## 何时选择 Block 和 Frame

还在编辑分子图——加原子、定义键、查连接——就用 `Atomistic`。化学结构已定，后续是数组操作、导出、分析，就用 `Block` 和 `Frame`。

两种表示可以共存。很多工作流保留一个 `Atomistic` 做参考，同时生成做数值计算的 Frame。关键是在每个阶段弄清楚哪个对象承载你关心的语义。

系统一旦放到周期性盒子里，光有坐标就不够了——距离还要看盒子尺寸。这是下一页的内容。

另请参阅：[Atomistic 与拓扑](01_atomistic_and_topology.md)、[盒子与周期性](03_box_and_periodicity.md)。
