---
slug: frame-block-inherit-molrs
status: code-complete
created: 2026-05-28
depends_on: molrs/frame-block-subclass
---

# molpy.Frame / Block 真子类化 molrs

## Summary

把 `molpy.Frame` 与 `molpy.Block` 从「组合 + `self._inner`」改成真正继承
`molrs.Frame` / `molrs.Block`。前置：molrs 端 spec `frame-block-subclass`
必须先 land（给两个 `#[pyclass]` 加 `subclass` 属性）。

改完之后：

- `molpy.Frame` 实例可以直接传给任何接受 `molrs.Frame` 的 API，不再
  需要 `.to_molrs()` / `._inner` 中转。
- `compute/rdf.py` 里 `mf = molrs.Frame()` 的桥接代码删除。
- `isinstance(molpy.Frame(...), molrs.Frame)` 为真，与 Box 现状对齐。
- Python-only 状态（`metadata` dict、object-dtype 列缓存）继续挂在
  Python `__dict__`，对 Rust 透明 —— 这是已经确认过的设计取舍。

不消除任何数据拷贝；这是 ergonomics + 一致性重构，不是性能工作。

## Domain basis

无新物理；只是把 Box 已经走过的继承路径推广到 Frame / Block。

## Design

### 顺序：Block 先于 Frame

Block 的改动是 Frame 的前提（`Frame.__setitem__` 内部通过 Block 取
`molrs.Block`）。顺序：Block → Frame → 调用点收尾。

### Block 重构

```python
# src/molpy/core/frame.py
class Block(molrs.Block, MutableMapping[str, np.ndarray]):
    """..."""
    # PyO3 原生基类与 __slots__ 冲突 → 去掉 __slots__，落在 __dict__

    def __init__(self, vars_=None):
        super().__init__()                # molrs.Block.__new__
        self._objects: dict[str, np.ndarray] = {}
        if vars_:
            for k, v in vars_.items():
                self[k] = v
```

- 删 `self._inner`；所有 `self._inner.insert(...)` 改成
  `super().insert(...)` 或 `molrs.Block.insert(self, ...)`。
- 保留 `self._objects`（object-dtype 列；Rust 装不下）。
- MRO 顺序 `(molrs.Block, MutableMapping)`，molrs.Block 在前确保 Rust
  slot 初始化优先。

### Frame 重构

```python
class Frame(molrs.Frame):
    def __init__(self, blocks=None, **props):
        super().__init__()                # molrs.Frame.__new__
        self.metadata: dict[str, Any] = dict(props)
        self._block_objects: dict[str, dict[str, np.ndarray]] = {}
        if blocks:
            ...
```

- 删 `self._inner`；所有 `self._inner.X` 改成 `super().X` 或
  `molrs.Frame.X(self, ...)`。
- `box` getter 仍把 `molrs.Box` 升格为 `molpy.Box`（已有逻辑，保留）。
- `to_molrs()` 直接删除 —— 不再需要桥接。

### Reader 升格

`molrs.read_xyz` / `read_pdb` / `read_lammps` / `LAMMPSTrajReader` 等
返回的是 `molrs.Frame`（非子类）。molpy 的 reader 层拿到 molrs.Frame
之后需要包一层把它升格为 molpy.Frame：

```python
def _upgrade(rs_frame: molrs.Frame) -> molpy.Frame:
    f = molpy.Frame.__new__(molpy.Frame)
    molrs.Frame.__init__(f)               # 子类的 Rust slot 初始化
    for k in rs_frame.keys():
        f[k] = rs_frame[k]                # zero-copy view 转移
    f.metadata = dict(rs_frame.meta) if rs_frame.meta else {}
    return f
```

如果 molrs 提供更便宜的 take-ownership 路径就走那条；否则用
`__setitem__` 循环（Block 是 zero-copy 视图，没有数据拷贝代价，只是
Python 侧多几个引用往返）。

具体哪些 reader 走这条路径，`/mol:impl` 启动时根据 `io/` 下实际
`import molrs.read_*` 的位置列清单。

### 删除 .to_molrs() 调用点

- `src/molpy/core/frame.py` 的 `to_molrs()` 方法删除。
- `src/molpy/compute/rdf.py:57` 的 `mf = molrs.Frame()` + 后续手工填充
  全部删掉；直接传 `frame`（它现在就是 `molrs.Frame`）。
- `grep -r '\._inner\|\.to_molrs' src/` 找其它调用点一并清理。

### Python-only 状态对 Rust 透明

`metadata: dict` 与 `_block_objects` / `_objects` 都挂在 Python
`__dict__`。`molpy.Frame` 传给 molrs kernel 时，kernel 只 downcast 到
`molrs.Frame` slot，看不到这些字段。这是已确认的设计：

- `metadata` 是 molpy 命名约定（`timestep`、`description` 等），Rust
  没有对应语义。
- object-dtype 列（`element` / `symbol` 字符串数组）在 `molrs.Block`
  当前 schema 之外，这些列对 Rust kernel 来说本来就不存在。

后续如果要把这些字段下沉到 Rust，单独立 spec。

### __slots__ 处理

PyO3 原生基类的 layout 不允许子类带 `__slots__`；`molpy.Frame` 和
`molpy.Block` 的 `__slots__` 都要去掉，回归 `__dict__`。这是 PyO3
子类化的标准代价；内存增量可以忽略（每实例几十字节）。

## Files to create or modify

- `src/molpy/core/frame.py` — (modify) Block 与 Frame 继承重构；删
  `_inner` / `to_molrs` / `__slots__`
- `src/molpy/compute/rdf.py` — (modify) 删 `molrs.Frame()` 桥接，直接
  传 `frame`
- `src/molpy/io/**/*.py` — (modify) reader 出口处包一层升格（清单在
  impl 时确定）
- `tests/test_core/test_frame_inheritance.py` — (new) isinstance + 直接
  传 molrs API + Python-only 状态隔离
- `tests/test_core/test_block_inheritance.py` — (new) 同上对 Block

## Tasks

> Phase 0 是 molrs 那边的 spec；本 spec 在 Phase 1 开始前 gate 检查
> molrs 端是否已 land。

**Phase 1 — Block 继承**

- [x] 写失败测试 `test_block_inheritance.py::test_molpy_block_is_a_molrs_block`
- [x] 重构 Block 为 `molrs.Block` 子类，删 `_inner`，去 `__slots__`
- [x] 写失败测试 `test_block_inheritance.py::test_object_columns_invisible_to_molrs`
- [x] 跑 `tests/test_core/` 全绿

**Phase 2 — Frame 继承**

- [x] 写失败测试 `test_frame_inheritance.py::test_molpy_frame_is_a_molrs_frame`
- [x] 重构 Frame 为 `molrs.Frame` 子类，删 `_inner`，去 `__slots__`
- [x] 写失败测试 `test_frame_inheritance.py::test_frame_directly_accepted_by_molrs_api`
  （例如 `molrs.NeighborQuery(frame.box, xyz, 2.0)` 或
  `molrs.write_xyz(path, frame)`）
- [x] 删 `to_molrs()` 方法
- [x] 跑 `tests/test_core/` 全绿

**Phase 3 — 调用点收尾**

- [x] 改 `compute/rdf.py`：删 `molrs.Frame()` 桥接
- [x] 给 `io/` 下所有 molrs reader 出口加 `_upgrade()`
- [x] `grep -r '\._inner\|to_molrs' src/molpy/` 必须无命中
- [x] 全套测试 `pytest tests/ -m "not external" -v` 通过

**Phase 4 — 文档**

- [x] 更新 `CLAUDE.md`「Frame is backed by molrs」段：从「wraps via
  `_inner`」改成「inherits」
- [x] 加 changelog 条目：`molpy.Frame` / `Block` 现在 `isinstance` of
  molrs equivalents

## Testing strategy

### Happy path
- `isinstance(molpy.Frame(...), molrs.Frame)` 与
  `isinstance(molpy.Block(), molrs.Block)` 都为真
- `frame.box` getter 仍返回 `molpy.Box`（不是 `molrs.Box`）
- `frame["atoms"]` 仍返回 `molpy.Block`
- `molrs.NeighborQuery` / `molrs.write_xyz` 等直接接受 `molpy.Frame`

### Python-only 状态隔离
- `frame.metadata["foo"] = 1` 后，传给 molrs API 不丢失（往回拿
  `frame.metadata["foo"] == 1`）
- molrs API 看不到 metadata —— 通过对比 `molrs.Frame()` 直接调用与
  `molpy.Frame()` 调用对 Rust 侧的副作用应一致

### Reader 升格
- 每个改过的 reader 出口跑一遍：返回值是 `molpy.Frame`，且 `metadata`
  字典存在（即使为空）

### 回归
- `tests/test_core/`、`tests/test_io/`、`tests/test_compute/` 全套通过
- `grep '\._inner\|to_molrs' src/molpy/` 无命中

## Out of scope

- 把 `metadata` / object-dtype 列下沉到 Rust 端 —— 独立 spec
- 任何性能优化 —— 本 spec 是一致性重构
- `molpy.Trajectory` 的继承化 —— 它不是 molrs 类型，无需改
