---
slug: lammps-ff-p0-single-boundary
status: done
created: 2026-08-06
revised: 2026-08-06
repos: [molcrafts-molrs, molpy]
layer: io / ff boundary
grilled: true
---

# LAMMPS FF P0 — 单一边界 + Frame 主权 + map_type

## Summary

1. **单一算术边界**：data `*Coeffs` 的 form map（`K=k/2`、deg↔rad）与 units 只在 molrs；molpy 零 coeff 算术。
2. **Frame 主权**：写 data 时 type 信息只从 Frame 来——优先 `type`，其次 `type_id`，否则 **报错**。用户负责提供齐全 Frame；I/O **不**以用户另传 `type_labels` inventory 为主路径。
3. **映射属 ForceField**：`ff.map_type(frame)` 把 FF 类型体系落到 Frame；Writer 不持 `forcefield` 参数。
4. **组合由调用方**：结构写与 coeffs 写分离。

P0 **不加** class2 等新 style（P1 codec）。

## Domain basis

- LAMMPS data `*Coeffs` 与 `*_coeff` 数值约定相同，版式不同。
- molrs store：物理 real（Å, kcal/mol, rad, `½k`）；lj 透传。
- 纯数字 type：不写 Type Labels；id 恒等。非数字 label：有 `type` 则写 Labels（id 由 Frame 中出现的 label 稳定编号）。
- Grill 决议（2026-08-06）：Frame 齐全；`map_type` on FF；Writer 无 forcefield。

## Design

### Reuse decision

| 候选 | 决策 |
|------|------|
| `LammpsFfWriter` + `WriteUnits` | **reuse** — 增 `write_data_coeffs_str` |
| `LammpsUnitSystem` | **reuse** — 唯一单位面 |
| `read_data_coeffs` | **reuse** — 读路径优先；Python 暴露为推荐非硬 AC |
| molpy inject form 算术 | **删除** |
| Writer `forcefield=` | **删除** |
| Writer `type_labels=` 主路径 | **降级/删除主义务**；id 从 Frame 推 |
| `ForceField.map_type` | **新增**（P0 契约） |

### 调用形状

```python
ff.map_type(frame)   # 用户显式：类型语义 → Frame 列
mp.io.write_lammps_data(path, frame)   # 有什么写什么
mp.io.write_lammps_data_coeffs(path, frame, ff, units="real")  # 可选第二步
```

### Type 规则（写结构）

1. Block 有 `type`（str）→ 用 label；纯数字 label → 不写 Type Labels，id = int(label)；非数字 → Type Labels + dense id（出现序/稳定 sort）。
2. 无 `type` 有 `type_id` → 数字原样写出，不写 Type Labels。
3. 都没有 → `ValueError`。

### Type 规则（写 *Coeffs）

1. 从 Frame 各相关 block 收集已建立的 name→id（与结构写同一套）。
2. ForceField type 名映射到该 id；无法映射 → `ValueError`。
3. 参数文本由 molrs `write_data_coeffs` 生成（含 units/form）。

### `ForceField.map_type(frame)`

- **职责**：将本力场相关的类型信息写到 Frame（至少保证 I/O 所需的 `type` 和/或 `type_id` 列存在且一致）。
- **Mutation**：与 core 一致——in-place，返回 `frame`（或 self 若绑在包装上）。
- **P0 最小语义**（可测硬契约）：
  - 若 atoms 已有 `type` 且值为本 FF 已知 atom type 名：为 atoms（及可选 bonds/… 若已有 type 名）生成/对齐 `type_id`（1-based，按类型名稳定序或已有 id）。
  - 若仅有 `type_id`：保持，不臆造 label。
  - 若既无 `type` 也无 `type_id`：报错，提示先 typify 或赋值。
  - **不做** SMARTS 全量 typifier（那是 typifier 模块）；`map_type` 是「已有类型名/ id 与 FF 注册表对齐并落到列」。

### API（molrs）

```rust
impl LammpsFfWriter {
    pub fn write_data_coeffs_str(&self, ff: &ForceField) -> Result<String, String>;
}
// type ids: 调用方传入 name→id，或类型名可解析为整数时用整数
```

Python：

```python
molrs.ff.write_lammps_data_coeffs(ff, *, units="real", precision=6,
                                  atom_type_ids: dict[str,int] | None = None, ...) -> str
```

molpy：

```python
def write_lammps_data_coeffs(path, frame, forcefield, *, units="real") -> None:
    """Insert *Coeffs into existing data file; ids from frame; math in molrs."""
```

### 读路径

- molpy 可薄转换 section→文本或调 `read_data_coeffs`；**禁止** form/单位算术。
- data 往返默认 style 假设（lj/cut + harmonic\*）保持；不保证 fourier 经 data 往返。

## Files

| 路径 | 动作 |
|------|------|
| molrs `writers/lammps.rs` | `write_data_coeffs_str` + type-id 输入 |
| molrs ForceField | `map_type` 或 Python 层等价 |
| molrs-python bindings | 暴露写 data coeffs；map_type |
| molpy `io/data/lammps.py` | 去 forcefield；Frame type 规则；独立 coeffs 写 |
| molpy `io/writers.py` | API 拆分 |
| tests | 见 Testing |

## Tasks

1. **Add** molrs `write_data_coeffs_str`（数值与 command form 一致）。
2. **Expose** Python `write_lammps_data_coeffs`。
3. **Add** `ForceField.map_type(frame)`（P0 最小语义 + 测试）。
4. **Remove** `LammpsDataWriter.forcefield` / `write_lammps_data(..., forcefield=)`。
5. **Implement** `mp.io.write_lammps_data_coeffs(path, frame, ff, units=...)`。
6. **Enforce** Frame type/`type_id` 规则；删用户 inventory 主路径义务。
7. **Delete** molpy coeff form 算术（grep gate）。
8. **Migrate** tests 到组合 API + map_type 若需要。

## Testing

- molrs: data_coeffs vs `*.ff` 数值 parity（real/metal）。
- map_type: 有 type 名 → 写出 type_id；双缺 → Err。
- write data 缺 type 与 type_id → Err。
- peptide coeffs：`write data` + `write_lammps_data_coeffs` round-trip。
- molpy 无 `k/2`/`math.degrees` 于 coeffs 路径。

## Out of scope

- Style codec / class2 / morse / cvff / hybrid 完整。
- `store_units` 字段。
- 结构+FF 一次 read 全 sink。
- si/cgs。
- Thole/TT 进 molrs writer。
- SMARTS typifier 并入 `map_type`。
- 改变 Frame mutation 全局语义（map_type 仅约定 in-place）。
