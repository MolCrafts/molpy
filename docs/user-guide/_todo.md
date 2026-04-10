# User-Guide Example 运行状态

最后更新: 2026-03-27

## 已通过

| Doc | 输出目录 | 输出文件 |
|---|---|---|
| 02_polymer_stepwise | `02_output/` | `peo5.data`, `peo5.ff` |
| 03_polymer_topology | `03_output/` | `linear.data`, `ring.data`, `branch.data` |
| 05_polydisperse_systems | `05_output/` | `lammps/lammps.data`, `lammps/lammps.ff` |
| 06_typifier | `06_output/` | `ethanol.data`, `ethanol.ff` |

## 阻塞

### 04_crosslinking

`TemplateResult.write()` 未实现。`TemplateReacter.run_with_template()` 返回的 `TemplateResult` 是纯 dataclass（含 `pre`, `post`, `init_atoms` 等字段），没有 `write` 方法。

需要：实现 `TemplateResult.write(base_path, typifier)` 或提供独立函数，输出 LAMMPS fix bond/react 需要的 pre/post molecule 文件和 map 文件。

### 07_ambertools_integration

两个问题：

1. **端基 validation 已修复** — `AmberPolymerBuilder` 现在允许端基（cap）只有一个端口（`<` 或 `>`），并只生成对应的 prepgen 变体（HEAD 或 TAIL）。

2. **atom `name` 属性缺失** — `parse_monomer()` + `Generate3D` 生成的 Atomistic 原子没有 PDB-style `name` 字段，但 `_prepare_monomers()` 中 `atom["name"]` 假设它存在。需要在 monomer 准备阶段自动生成原子名（如 C1, O2, H3...），或在 `AmberPolymerBuilder` 内部补全。
