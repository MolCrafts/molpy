# 添加 Wrapper 或 Adapter

集成外部 CLI 工具（wrapper）或外部 Python 库（adapter），离不开两类组件：Wrapper 和 Adapter。

## 两者的区别

| | Wrapper | Adapter |
|--|---------|---------|
| **跨越的边界** | 执行边界（子进程） | 表示边界（内存中） |
| **关注点** | 可执行文件、环境、返回码、文件 | 字段映射、同步保真度 |
| **可以** | 运行子进程、生成文件 | 保持两个对象模型同步 |
| **不得** | 包含工作流逻辑或化学语义 | 执行子进程或产生副作用 |

## 添加 Wrapper

继承 `molpy.wrapper.base` 下的 `Wrapper` 即可。基类已经封装好了可执行文件解析、conda/virtualenv 环境激活、工作目录管理以及 stdout/stderr 捕获。

```python
from dataclasses import dataclass, field
from pathlib import Path
from molpy.wrapper.base import Wrapper

@dataclass
class GmxWrapper(Wrapper):
    """GROMACS gmx 命令的包装器。"""

    name: str = "gmx"
    exe: str = "gmx"

    def energy_minimize(self, tpr_file: Path) -> Path:
        """运行能量最小化。"""
        result = self.run(args=["mdrun", "-s", str(tpr_file), "-deffnm", "em"])
        if result.returncode != 0:
            raise RuntimeError(f"gmx mdrun 失败: {result.stderr}")
        return self.workdir / "em.gro"
```

像 `energy_minimize` 这类高层方法，本质上是 `self.run()` 的便捷封装，保持简短即可——工作流逻辑交给调用方，别塞进 wrapper。

### 要点

- `self.run(args=[...])` 执行命令，返回 `subprocess.CompletedProcess`
- `self.resolve_executable()` 在 PATH 或配置好的 conda 环境里找二进制文件
- `self.is_available()` 检查工具能否被找到（适合条件导入场景）
- `workdir` 会自动创建，所有执行都在这个目录下进行
- Wrapper 实例化不依赖工具是否已安装，没装也能安全创建

## 添加 Adapter

继承 `molpy.adapter.base` 中的 `Adapter[InternalT, ExternalT]`，实现 `_do_sync_to_internal()` 和 `_do_sync_to_external()` 两个方法。

```python
from molpy.adapter.base import Adapter
from molpy.core.atomistic import Atomistic

class AseAdapter(Adapter[Atomistic, "ase.Atoms"]):
    """在 MolPy Atomistic 和 ASE Atoms 之间同步。"""

    def _do_sync_to_external(self):
        """Atomistic → ASE Atoms。"""
        import ase
        symbols = [a.get("element") for a in self._internal.atoms]
        positions = [[a["x"], a["y"], a["z"]] for a in self._internal.atoms]
        self._external = ase.Atoms(symbols=symbols, positions=positions)

    def _do_sync_to_internal(self):
        """ASE Atoms → Atomistic。"""
        mol = Atomistic()
        for atom in self._external:
            mol.def_atom(
                element=atom.symbol,
                x=atom.position[0],
                y=atom.position[1],
                z=atom.position[2],
            )
        self._internal = mol
```

### 要点

- `get_external()` 在外部对象为 `None` 而内部对象已设置时会自动触发同步，反之亦然
- 外部库要写成可选导入：ASE 没装的时候，adapter 模块导入时可以优雅降级，而不是直接崩掉
- Adapter 内部**绝不能**启动子进程——那是 wrapper 的职责
- 测试往返保真度：`internal → external → internal` 后，原子数、连接性和坐标应当保持不变

## 处理可选依赖

可选导入按现有模式处理：

```python
# 在 adapter 模块中
try:
    import ase
    _HAS_ASE = True
except ImportError:
    _HAS_ASE = False
    ase = None

class AseAdapter(Adapter[Atomistic, "ase.Atoms"]):
    def __init__(self, **kwargs):
        if not _HAS_ASE:
            raise ImportError("需要 ASE 库: pip install ase")
        super().__init__(**kwargs)
```

这样模块在没有依赖时也能正常导入，只有真正用到的时候才会报错。

## 检查清单

- [ ] Wrapper：继承 `Wrapper`，`self.run()` 调用保持简洁
- [ ] Adapter：继承 `Adapter[I, E]`，实现 `_do_sync_to_internal/external`
- [ ] 可选依赖：守卫导入，在使用时而非导入时失败
- [ ] 测试：adapter 的往返保真度，wrapper 的返回码检查
- [ ] 测试文件放在 `tests/test_wrapper/` 或 `tests/test_adapter/` 下
- [ ] 需要外部工具的测试用 `@pytest.mark.external` 标记
