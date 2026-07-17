# 扩展 I/O 格式

为新文件格式和力场后端实现读写器，继承对应的 I/O 基类即可。

## 数据文件读写器

读写器分别继承 `DataReader` 和 `DataWriter`，两个基类都定义在 `molpy.io.data.base` 中。

### 读取器

```python
from pathlib import Path
from molrs import Frame, Block
from molpy.io.data.base import DataReader

from molpy.core.fields import FieldFormatter, CHARGE

class MyFieldFormatter(FieldFormatter):
    """为 .myformat 格式提供字段名翻译。"""
    _field_formatters = {
        "q": CHARGE,   # .myformat 使用 "q" 表示电荷
    }

class MyFormatReader(DataReader):
    """将 .myformat 文件读入 Frame。"""

    _formatter = MyFieldFormatter()

    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    def read(self, frame: Frame | None = None) -> Frame:
        if frame is None:
            frame = Frame()

        # 解析文件（self._path 由 FileBase 设置）
        with open(self._path) as f:
            lines = f.readlines()

        # 使用格式原生的字段名填充 blocks
        frame["atoms"] = Block({
            "element": [...],
            "x": [...],
            "y": [...],
            "z": [...],
        })

        # 将格式特定的字段名翻译为规范名称
        self._formatter.canonicalize_frame(frame)
        return frame
```

### 写入器

```python
from molpy.io.data.base import DataWriter

class MyFormatWriter(DataWriter):
    """将 Frame 写入 .myformat 文件。"""

    _formatter = MyFieldFormatter()

    def __init__(self, file: Path, **kwargs):
        super().__init__(path=file, **kwargs)

    def write(self, frame: Frame) -> None:
        # 将规范名称转换为格式特定名称（在副本上操作）
        frame = self._formatter.localize_frame(frame)

        atoms = frame["atoms"]
        with open(self._path, "w") as f:
            for i in range(atoms.nrows):
                f.write(f"{atoms['element'][i]} {atoms['x'][i]} ...\n")
```

### 注册工厂函数

把读写器加到 `molpy/io/readers.py` 和 `molpy/io/writers.py` 中，`mp.io.read_myformat()` 和 `mp.io.write_myformat()` 就能直接使用。


## 规范字段名

内部数据模型使用定义在 `molpy.core.fields` 的规范字段名。如果格式的列名与规范名不同，定义 `FieldFormatter` 子类，在 `_field_formatters` 中指明映射关系：

```python
from molpy.core.fields import FieldFormatter, FieldSpec, CHARGE, MOL_ID

class MyFieldFormatter(FieldFormatter):
    _field_formatters = {
        "q":   CHARGE,    # 格式 "q" → 规范名 "charge"
        "mol": MOL_ID,    # 格式 "mol" → 规范名 "mol_id"
    }
```

基础规范字段：`charge`（而非 `q`）、`mol_id`（而非 `mol`）、`id`、`type`、`mass`、`element`、`x`/`y`/`z`。

如果格式的字段名已经与规范名一致（比如 MOL2 用 `charge`），则无需 `FieldFormatter`。


## 力场写入器的格式化器结构

力场写入系统使用 `molpy.core.fields` 中的**双层格式化器继承结构**：

```
FieldFormatter                         — 数据字段映射：{format_key: FieldSpec}
    ↓
ForceFieldFormatter(FieldFormatter)    — 继承字段映射 + {StyleType: Callable}
```

每个格式的 `ForceFieldFormatter` 子类都继承 `FieldFormatter`，既包含数据字段映射，也注册了参数格式化器用于 Style/Type 序列化。

### 为自定义 Style 添加参数格式化器

```python
from molpy.io.forcefield.lammps import LammpsForceFieldFormatter

def _format_morse_bond(typ) -> list[float]:
    return [typ.params.kwargs["D"], typ.params.kwargs["alpha"], typ.params.kwargs["r0"]]

LammpsForceFieldFormatter.register_param_formatter(MorseBondStyle, _format_morse_bond)
```

注册**按子类隔离**——为某个写入器添加格式化器不会影响其他写入器。`__init_subclass__` 在子类创建时复制注册表，保证了这种隔离。


## 轨迹读写器

轨迹读取器用内存映射文件和持久化帧索引实现随机访问。继承 `BaseTrajectoryReader`，实现 `_scan_frames`（构建字节偏移索引）和 `_parse_frame_bytes`（解析单帧）即可：

```python
import mmap
from molpy.io.trajectory.base import BaseTrajectoryReader
from molpy.io.trajectory.index import FrameEntry
from molrs import Frame

class MyTrajectoryReader(BaseTrajectoryReader):
    _format_id = "myformat"

    def _scan_frames(self, file_idx: int, mm: mmap.mmap) -> list[FrameEntry]:
        entries = []
        # 扫描文件中的帧边界，记录字节偏移量
        return entries

    def _parse_frame_bytes(self, mm: mmap.mmap, entry: FrameEntry) -> Frame:
        # 从 mm[entry.offset:entry.offset+entry.length] 解析一帧
        return frame
```

持久索引（`.tridx`）在首次读取时自动构建并缓存，后续直接使用。写入端继承 `TrajectoryWriter`，实现 `write_frame()` 即可。


## 检查清单

- [ ] 继承 `DataReader`/`DataWriter` 或 `BaseTrajectoryReader`/`TrajectoryWriter`
- [ ] 字段名与规范名不同时，定义 `FieldFormatter` 子类
- [ ] 读取器返回前调用 `_formatter.canonicalize_frame(frame)`
- [ ] 写入器入口处调用 `_formatter.localize_frame(frame)`（操作副本）
- [ ] Box 存在 `frame.simbox`；精确类型元数据存在 `frame.meta`
- [ ] 在 `readers.py` / `writers.py` 中添加工厂函数
- [ ] 新增自定义 Style 时，在 `ForceFieldFormatter` 子类上注册参数格式化器
- [ ] 在 `tests/test_io/` 中编写往返测试（write → read → compare）
- [ ] 往返测试应验证字段已被规范化为 `charge` 和 `mol_id`（而非 `q` 和 `mol`）
