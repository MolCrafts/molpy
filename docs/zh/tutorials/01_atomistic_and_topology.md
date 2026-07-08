# Atomistic 与拓扑

MolPy 用图来表示分子：原子是节点，键是边，角、二面角和环则根据连接性按需推导。

## 为什么用图

分子模拟要处理两类问题。*几何*问题——"两个原子相距多远？"——有坐标就够了。*化学*问题——"哪些原子共享共价键？"、"这个碳在芳香环上吗？"、"打断这根键会怎样？"——需要知道原子之间的连接关系。

坐标列表回答不了化学问题，但分子图可以。MolPy 从图表示出发，原因就在这里：在把体系交给模拟引擎之前，图是构建、编辑和推理分子拓扑最自然的数据结构。

## 分子是一个可编辑的图

`Atomistic` 就是一个可编辑的分子图。原子是节点，键是边。跟坐标表不同，这种表示保留了身份信息——你知道*哪个*碳连了*哪个*氧，而不只是第三行原子碰巧靠近第五行。

这一点很实际。大多数工作流开始时，分子还没定型：要加原子、删离去基团、标记反应位点、检查连接性——全是图操作。`Atomistic` 就是干这个的。

构建分子从空容器开始。

```python
import molpy as mp

mol = mp.Atomistic(name="ethanol")
```

`def_atom` 创建一个原子并加入图。关键字参数可以传任意属性——`element`、坐标、`charge`，不限，没有固定模板。

```python
c1 = mol.def_atom(element="C", name="C1", x=0.0, y=0.0, z=0.0)
c2 = mol.def_atom(element="C", name="C2", x=1.54, y=0.0, z=0.0)
o  = mol.def_atom(element="O", name="O1", x=2.0, y=1.4, z=0.0)
h_o = mol.def_atom(element="H", name="HO", x=2.9, y=1.4, z=0.0)
```

`def_bond` 连接两个已存在的原子。跟原子一样，键也可以带关键字属性。

```python
mol.def_bond(c1, c2, order=1)
mol.def_bond(c2, o, order=1)
mol.def_bond(o, h_o, order=1)
print(f"{len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# 4 atoms, 3 bonds
```

这时 `mol` 里装的是乙醇的重原子骨架。原子和键在图中是活对象，不是数据的副本。

## 原子和键的行为类似字典

`Atom` 和 `Bond` 都是类字典对象。用方括号或 `.get()` 读写属性。

```python
print(c1["element"])      # "C"
print(c1.get("charge"))   # None — charge was never set

c1["charge"] = -0.18
print(c1["charge"])       # -0.18
```

键也一样。`Bond` 通过 `.itom` 和 `.jtom` 暴露两个端点。

```python
bond = mol.bonds[0]
print(bond.itom, bond.jtom)   # <Atom: C> <Atom: C>
print(bond.get("order"))      # 1
```

原子是引用，改了立即反映到图上，没有单独的"提交"步骤。

```python
for atom in mol.atoms:
    if atom["element"] == "C":
        atom["hybridization"] = "sp3"

print(c2["hybridization"])  # "sp3"
```

## 连接性由容器管理

原子不保存邻居信息。`Atomistic` 容器统一管理连接性，这样原子保持轻量，图操作也更清晰。

`get_neighbors` 返回与某原子直接成键的原子列表。

```python
neighbors = mol.get_neighbors(c2)
print([n["name"] for n in neighbors])  # ['C1', 'O1']
```

遍历 `bonds` 也可以检查原子周围的键。

```python
for bond in mol.bonds:
    if c2 in bond.endpoints:
        partner = bond.itom if bond.jtom is c2 else bond.jtom
        print(f"C2 —({bond.get('order')})— {partner['name']}")
```

邻居是普通 Python 对象，做高级查询很方便。下面这个函数收集从起始原子出发 *n* 跳内的所有原子。

```python
def n_hop_neighbors(mol, start, n):
    visited = {start}
    shell = {start}
    for _ in range(n):
        next_shell = set()
        for atom in shell:
            for nb in mol.get_neighbors(atom):
                if nb not in visited:
                    visited.add(nb)
                    next_shell.add(nb)
        shell = next_shell
    return visited - {start}

print({a["name"] for a in n_hop_neighbors(mol, c1, 2)})
# {'O1', 'C2'}
```

## 删除原子保持图一致性

用 `remove_entity` 删原子，关联的键自动清除，不会留下指向已删除原子的悬空键。

```python
print(f"Before: {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# Before: 4 atoms, 3 bonds

mol.remove_entity(h_o)

print(f"After:  {len(mol.atoms)} atoms, {len(mol.bonds)} bonds")
# After:  3 atoms, 2 bonds

print([n["name"] for n in mol.get_neighbors(o)])  # ['C2']
```

O–H 键随氢原子一并消失，剩下的图内部一致。

## 拷贝生成独立副本

`copy()` 把所有原子和键深拷贝到一个新的 `Atomistic` 对象。两个图完全独立，改一个不影响另一个。

```python
mol_copy = mol.copy()

c1_copy = [a for a in mol_copy.atoms if a["name"] == "C1"][0]
c1_copy["name"] = "C1_copy"

print(c1["name"])       # "C1" — original unchanged
print(c1_copy["name"])  # "C1_copy"
```

## 用 + 合并系统

两个 `Atomistic` 可以用 `+` 合并，结果包含两边的所有原子和键。

```python
water = mp.Atomistic(name="water")
ow = water.def_atom(element="O", x=0.0, y=0.0, z=0.0)
h1 = water.def_atom(element="H", x=0.957, y=0.0, z=0.0)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.0)
water.def_bond(ow, h1)
water.def_bond(ow, h2)

two_waters = water + water.copy().move([5.0, 0.0, 0.0])
print(f"{len(two_waters.atoms)} atoms, {len(two_waters.bonds)} bonds")
# 6 atoms, 4 bonds
```

多次拷贝用 `replicate` 更方便。接受数量和可选的变换函数。

```python
box = water.replicate(4, lambda mol, i: mol.move([i * 4.0, 0.0, 0.0]))
print(f"{len(box.atoms)} atoms")  # 12
```

## 拓扑从推导中产生

分子动力学不止需要键，还需要角（三原子序列）和二面角（四原子序列）。手动维护这些很容易出错——增删一次键，所有角、二面角列表都得更新。

MolPy 把拓扑当作*推导视图*。在 `Atomistic` 上调用 `get_topo`，它读取当前键图，生成一个带完整角和二面角的**新** `Atomistic`（原对象不变）。图变了就重新推导。推导本身（在键图上找 2 步和 3 步路径）由 molrs Rust 内核完成。

下面用一个完整丙烷分子演示。

```python
propane = mp.Atomistic(name="propane")
ca = propane.def_atom(element="C", name="C1", x=0.0, y=0.0, z=0.0)
cb = propane.def_atom(element="C", name="C2", x=1.54, y=0.0, z=0.0)
cc = propane.def_atom(element="C", name="C3", x=3.08, y=0.0, z=0.0)
propane.def_bond(ca, cb)
propane.def_bond(cb, cc)

print(f"Before: {len(propane.angles)} angles, {len(propane.dihedrals)} dihedrals")
# Before: 0 angles, 0 dihedrals

propane = propane.get_topo(gen_angle=True, gen_dihe=True)

print(f"After:  {len(propane.angles)} angles, {len(propane.dihedrals)} dihedrals")
# After:  1 angles, 0 dihedrals
```

`Angle` 和 `Dihedral` 跟 `Bond` 一样，通过 `.endpoints` 持有端点原子引用。

```python
for angle in propane.angles:
    names = [a["name"] for a in angle.endpoints]
    print(" — ".join(names))
# C1 — C2 — C3
```

## 在键图上做图查询

没有独立的拓扑对象——`get_topo` 返回的还是 `Atomistic`，图查询通过 molrs Rust 内核直接在结构上运行。`get_topo_neighbors` 按键数半径收集原子，`get_topo_distances` 返回从源原子到每个可达原子的键图（BFS）距离。

```python
print("within 1 bond of C2:", [a["name"] for a in propane.get_topo_neighbors(cb, radius=1)])
# within 1 bond of C2: ['C1', 'C2', 'C3']

dists = propane.get_topo_distances(ca)
print({a["name"]: d for a, d in dists.items()})
# {'C1': 0, 'C2': 1, 'C3': 2}
```

放大到更大的分子也一样：删除键后检查连通性、测量官能团之间的拓扑距离，都是在 `Atomistic` 上做 k 跳查询。

## 停留在 Atomistic，还是向前走

只要结构还在变动——加原子、定义键、检查连接性、运行反应——就用 `Atomistic`。这是*化学编辑*的层级。

一旦结构稳定下来，下一步是导出、分析或模拟，那么合适的表示就该换了。[Block 和 Frame](02_block_and_frame.md) 用对齐数组携带同样的体系并带有显式元数据——更适合数值计算和文件 I/O。

另请参阅：[Block 和 Frame](02_block_and_frame.md)、[API 参考：核心](../../api/index.md)。
