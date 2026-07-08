# Moltemplate 命令行工具

MolPy 自带 moltemplate 引擎。`molpy moltemplate` 子命令解析 `.lt` 脚本，直接生成 **LAMMPS**、**OpenMM**、**GROMACS** 的输入文件——也可输出为 MolPy 规范的 XML 力场格式，或自包含的 **MolPy Python 脚本**。

## 快速上手

```bash
# 汇总脚本信息（原子类型、分子数目、样式）
molpy moltemplate info water.lt

# 生成 LAMMPS 输入文件（data + in.settings + in.init + starter in）
molpy moltemplate run water.lt --emit lammps --out-dir out/

# 一次性生成所有引擎的输入
molpy moltemplate run water.lt --emit all --out-dir out/

# 仅力场：将 .lt 力场转换为 MolPy XML
molpy moltemplate convert gaff2.lt gaff2.xml

# .lt → MolPy Python 脚本（moltemplate 的逆向转换——便于手动编辑）
molpy moltemplate convert water.lt water.py

# MolPy 系统 → .lt（ltemplify：将 Atomistic+ForceField 打包回 .lt 格式）
molpy moltemplate ltemplify water.lt water_regen.lt

# 导出解析后的中间表示（调试用）
molpy moltemplate parse water.lt --json ir.json
```

## 子命令

### `run` — 生成引擎输入

```
molpy moltemplate run SCRIPT [--emit ENGINE ...] [--out-dir DIR] [--prefix NAME]
```

| 引擎       | 生成的文件（`--prefix system` 时）                                       |
|-----------|------------------------------------------------------------------------------|
| `lammps`  | `system.data`, `system.in.settings`, `system.in.init`, `system.in`            |
| `openmm`  | `system.xml`, `system.pdb`, `system.py`                                       |
| `gromacs` | `system.gro`, `system.top`, `em.mdp`, `nvt.mdp`                              |
| `xml`     | `system.xml`, `system.pdb`（MolPy 规范格式）                                  |
| `all`     | 以上所有引擎                                                                 |

`--emit` 可重复指定：`--emit lammps --emit openmm`。

### `parse` — 调试中间表示

```
molpy moltemplate parse SCRIPT [--json OUT]
```

不传 `--json` 时，按语句类型逐行打印摘要。

### `info` — 单行摘要

```
molpy moltemplate info SCRIPT
```

展开所有 `new` 实例后，打印原子类型/原子/键/角/二面角计数。

### `convert` — `.lt` → XML 或 Python

```
molpy moltemplate convert SRC.lt DST.{xml,py}
```

输出格式由目标文件扩展名决定：

| 扩展名 | 输出内容 |
|--------|----------|
| `.xml` | MolPy 规范 XML 力场（仅力场）。 |
| `.py`  | 自包含的 MolPy Python 脚本，内含 `build_forcefield()`、`build_<ClassName>()`（每个 moltemplate 类对应一个），以及顶层 `build_system()`。生成的脚本不依赖原始 `.lt` 文件，可自由编辑。 |

### `ltemplify` — `.lt` / `.data` → `.lt`

```
molpy moltemplate ltemplify SRC.lt DST.lt [--class-name NAME]
molpy moltemplate ltemplify SRC.data DST.lt --ff SRC.in.settings
```

将 `Atomistic + ForceField` 序列化回 moltemplate 模板。第一种形式在解析后重新生成 `.lt`；第二种形式将 LAMMPS data 文件连同系数文件一起转换。

`read_moltemplate_system` → `ltemplify` 往返过程在原子/键/角/二面角/improper 数量上保持一致。但**不会**重建类层次结构——所有内容扁平化到单个类中。

## 支持的 moltemplate 功能

已在 [`moltemplate/examples`](https://github.com/jewettaij/moltemplate/tree/master/examples) 的真实示例上测试过。覆盖范围逐步扩展——下表列出哪些功能在上游测试用例中已验证通过，哪些会静默降级。

| 功能                                                | 状态 |
|------------------------------------------------------|------|
| `ClassName { ... }`，嵌套类                          | ✔    |
| `inherits Parent1, Parent2`                          | ✔    |
| `import "file.lt"`（递归导入）                       | ✔    |
| `write("...")`, `write_once("...")`                  | ✔    |
| `write('...')` 单引号区段名                          | ✔    |
| 包含 `(...)` 的区段名                                | ✔    |
| `Data Masses`、`Data Charges`、`In Charges`          | ✔    |
| `In Settings` 系数行（pair/bond/angle/…）            | ✔    |
| `Data Atoms`、`Data Bonds`、`Data Angles`            | ✔    |
| `Data Dihedrals`、`Data Impropers`                   | ✔    |
| `inst = new Cls`                                     | ✔    |
| `.move(x,y,z)`、`.rot(θ,ax,ay,az)`、`.scale(s)`     | ✔    |
| `new Cls [N].move(dx,dy,dz)` 一维数组                | ✔    |
| `new Cls [N].move(...) [M].move(...) [K].move(...)`（三维数组）| ✔    |
| `.rotvv(v1,v2)`                                      | 部分支持（已解析但未使用） |
| `new random([Cls1, Cls2], [w1, w2] [, seed])`        | ✔    |
| `$atom:submol/atom` 作用域引用                        | ✔    |
| `Data Bond List`（无 `@bond:T` 列）                  | ✔    |
| `Bonds/Angles/Dihedrals/Impropers By Type` 通配符     | ✔（规则匹配在自动拓扑之后应用） |
| `replace{ @atom:A @atom:B }`                          | ✔（在 Data Atoms 阶段应用装饰） |
| `create_var`、`delete_var`、`category`               | ✘（静默忽略） |
| `Impropers` 作为一等 `Improper` 链接                  | ✔    |
| oplsaa.lt（完整约 10000 行文件）                      | 可解析；力场加载正常；键/角/二面角类型通过 By-Type 通配符解析 |

### 诚实说明

`tip3p_2004_oplsaa.lt` + `oplsaa2024.lt`（真实 moltemplate OPLS 文件，约 10000 行）可端到端解析，生成包含数千个 `AtomType` 条目的 `ForceField`。`replace{ @atom:A @atom:B }` 装饰已生效，原子类型据此获知其键/角/二面角/improper 关联；`Bonds/Angles/Dihedrals/Impropers By Type` 通配符规则在*自动拓扑*完成后匹配，为每个键/角/二面角/improper 填入具体类型名称。通配符无法匹配时，回退到从 `In Settings` 系数生成的占位符名称，下游生成器不会因此崩溃。

### 验证自己的 `.lt` 文件

```bash
molpy moltemplate parse my_system.lt         # 中间表示摘要
molpy moltemplate info my_system.lt          # 展开后的原子/键计数
molpy moltemplate run my_system.lt --emit lammps --out-dir out/
```

## Python API

CLI 的全部功能均可通过 Python 代码调用。

```python
from molpy.io.forcefield.moltemplate import read_moltemplate_system
from molpy.io.emit import emit, emit_all
from molpy.parser.moltemplate import (
    emit_python,      # .lt → .py
    ltemplify,        # (atomistic, ff) → .lt 字符串
    parse_file,       # .lt → IR Document
    write_moltemplate,  # (atomistic, ff) → .lt 文件
)

atomistic, ff = read_moltemplate_system("water.lt")

# 单个引擎
emit("lammps", atomistic, ff, "out/", prefix="w")

# 所有引擎
emit_all(atomistic, ff, "out/", prefix="w")

# .lt → .py
emit_python(parse_file("water.lt"), "water.py")

# ltemplify：回到 .lt 模板
write_moltemplate(atomistic, ff, "water_regen.lt", class_name="Water")
```

**Python 钩子**：moltemplate 的 `include "foo.py"` 不直接对接 Python。改用 `molpy moltemplate convert foo.lt foo.py`，编辑生成的脚本即可。输出是纯粹的 MolPy，所有 MolPy Python API（builders、reacter、compute、wrapper）都可在原来挂接 Python 逻辑的地方继续使用。

`Atomistic` / `CoarseGrain` / `ForceField` 上的核心编辑原语（参见 `core.atomistic`、`core.cg`、`core.forcefield`）包括：

- `del_atom`、`del_bond`、`del_angle`、`del_dihedral`
- `rename_type(old, new, *, kind=Atom)`
- `set_property(selector, key, value, *, kind=Atom)`
- `select(predicate) -> Atomistic`
- `ForceField.rename_type / remove_type / remove_style`

这些是内核级操作，可在 moltemplate 管道之外使用。
