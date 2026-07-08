# 封装器与适配器 (Wrapper and Adapter)

调用外部程序和转换到另一个库的对象模型是两类不同的问题。MolPy 为它们分别命名，并设计了独立的抽象层。

## 两种不同的外部边界

分子模拟工作流很少局限在一个库里。总有那么些时候，你需要调用外部程序、转到另一个对象模型，或者两者都要。这些操作本质不同，MolPy 对它们各有专门的命名。

**封装器 (Wrapper) 跨越执行边界。适配器 (Adapter) 跨越表示边界。**

封装器运行外部可执行程序——比如 `antechamber`、`tleap`、`lmp_serial`——负责管理子进程、返回码、工作目录和磁盘文件。适配器在 MolPy 对象和另一个库的内存对象之间做转换——比如 RDKit 分子、OpenBabel 结构——处理字段映射和同步。

笼统地把两者都当作"调用另一个 API"，会掩盖工作流中最容易出问题的环节。MolPy 坚持把区分做清楚，出了问题你才知道该往哪个方向排查。

## 封装器：跨进程边界的受控执行

`Wrapper` 封装一个命令行工具，负责定位可执行文件、配置环境并执行命令。

```python
from molpy.wrapper import Wrapper

echo = Wrapper(name="echo_tool", exe="echo")
result = echo.run(args=["Hello", "from", "MolPy!"])

if result.returncode == 0:
    print(result.stdout.strip())   # Hello from MolPy!
else:
    print(result.stderr)
```

这里用 `echo` 做演示，因为它不需要额外安装。实际场景是 Antechamber、tleap 这类工具，但封装器的使用模式一样：指定可执行程序名、传参执行、检查结果。

工具如果装在隔离环境里，封装器会自动激活 Conda 或 virtualenv。

```python
# 示例（没有安装 AmberTools 则无法运行）：
# ac = Wrapper(
#     name="antechamber",
#     exe="antechamber",
#     env="AmberTools22",
#     env_manager="conda",
# )
# ac.run(args=["-i", "input.pdb", "-fi", "pdb", "-o", "out.mol2", "-fo", "mol2"])
```

## 适配器：在两个对象模型间同步状态

`Adapter` 持有 MolPy 内部对象和外部对象各一份，在两者之间同步状态。同步分两个方向：`sync_to_external()` 把内部转成外部，`sync_to_internal()` 把外部转回内部。

下面是一个最小适配器，在字典和分号分隔的字符串之间来回转换。

```python
from molpy.adapter import Adapter

class StringDictAdapter(Adapter[dict[str, str], str]):
    def _do_sync_to_external(self):
        self._external = ";".join(
            f"{k}={v}" for k, v in self._internal.items()
        )

    def _do_sync_to_internal(self):
        self._internal = dict(
            item.split("=") for item in self._external.split(";") if item
        )

adapter = StringDictAdapter(internal={"name": "MolPy", "role": "toolkit"})
adapter.sync_to_external()
print(adapter.get_external())   # name=MolPy;role=toolkit

adapter.set_external("name=MolPy;role=toolkit;version=0.2")
adapter.sync_to_internal()
print(adapter.get_internal())   # {'name': 'MolPy', 'role': 'toolkit', 'version': '0.2'}
```

这个示例刻意保持简单。重点不是数据格式，而是同步协议本身。没有外部进程，不涉及文件读写，核心就是一个信息在两种表示形式之间保持一致。

## 实际应用：用 RDKit 做几何优化

适配器的一个常见场景是借助外部库完成 MolPy 自身没有的算法。下面的 `RDKitAdapter` 在 `Atomistic` 分子和 RDKit 的 `Mol` 对象之间搭桥：在 RDKit 里做几何优化，然后把优化后的坐标写回 MolPy。

```python
from molpy import Atomistic
from molpy.adapter import RDKitAdapter
from rdkit.Chem import AllChem

mol = Atomistic()
c1 = mol.def_atom(element="C", x=0.0, y=0.0, z=0.0)
c2 = mol.def_atom(element="C", x=0.0, y=0.0, z=0.0)
o  = mol.def_atom(element="O", x=0.0, y=0.0, z=0.0)
mol.def_bond(c1, c2, order=1.0)
mol.def_bond(c2, o, order=2.0)

adapter = RDKitAdapter(internal=mol)
rd_mol = adapter.get_external()

AllChem.EmbedMolecule(rd_mol)
AllChem.MMFFOptimizeMolecule(rd_mol)

adapter.set_external(rd_mol)
adapter.sync_to_internal()

updated = adapter.get_internal()
atoms = list(updated.atoms)
print(f"C1: ({atoms[0]['x']:.2f}, {atoms[0]['y']:.2f}, {atoms[0]['z']:.2f})")
print(f"O:  ({atoms[2]['x']:.2f}, {atoms[2]['y']:.2f}, {atoms[2]['z']:.2f})")
```

!!! note
    此示例需要安装 RDKit。MolPy 把 RDKit 当作可选依赖，如果没装，适配器会给出清晰的错误信息。

## 怎么选

需要运行另一个程序：用封装器。关注点是执行——跑成功没有？生成了什么文件？

需要在 MolPy 对象和另一个库的对象之间转换：用适配器。关注点是保真度——两边还是同一个科学结构吗？

两个边界都要跨，那就两个都用——先跑 Antechamber（封装器），再把输出转成 MolPy 对象（适配器）。

参见：[Atomistic 与拓扑结构](01_atomistic_and_topology.md)、[力场](04_force_field.md)。
