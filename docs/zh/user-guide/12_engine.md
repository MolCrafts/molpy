# Engine 模块桥接 Python 数据与 MD 程序

给力场类型分配完毕的分子体系，通过 `Engine` 就能转成 LAMMPS、CP2K 或 OpenMM 可直接读取的输入文件——至于把文件交给调度器还是直接启动计算，由你决定。

---

## 最后一公里问题

分子建好了，原子类型分配完了，坐标和力场文件也导出了。接下来是最后一公里：LAMMPS 要一个包含 `units`、`atom_style`、`run` 等命令的控制脚本；CP2K 要结构化的 `&GLOBAL` / `&FORCE_EVAL` 输入文件；OpenMM 要 PDB、XML 力场外加一个 Python 驱动脚本。每个程序调用语法不同，文件查找规则也不同。`engine` 模块负责处理这些差异，不让它们渗透到建模代码里。

**Engine 是 MolPy 与 MD 程序之间的适配器——它生成引擎可读的输入文件，也负责调用可执行文件。**

引擎不做什么也同样重要。它不构建分子，不分配原子类型，也不分析轨迹。LAMMPS data 文件或力场系数文件由 `mp.io` 写入，引擎不碰这些。引擎只管一件事：*控制脚本*——告诉 MD 程序跑什么物理过程、去哪儿找 `mp.io` 已经写好的坐标数据。

---

## 生成文件不花钱，跑才花钱

把 Engine 想象成实验室仪器的控制器，有两种工作模式。"打印协议"模式下，控制器写出完整的实验步骤——每项设置、每个操作——但不按下启动按钮。"运行仪器"模式执行同样的流程。两种情况下底层协议完全一样，区别只在于按钮按没按下。

这种分离是故意的。你可以在提交前检查生成的文件，手动修改，或者把它们复制到 HPC 集群用作业调度器提交，完全不用调 `run()`。仅生成模式不是降级模式——在没装 MD 可执行文件的机器上，它就是主要工作流。

对 LAMMPS 和 CP2K 来说，两步完全解耦：你构建 `Script` 对象，然后存盘或传给 `run()`。OpenMM 则更集成：`generate_inputs()` 从 MolPy 的 `Frame` 和 `ForceField` 直接生成全部三个必需文件——PDB、XML 力场和 Python 模拟脚本——这个方法甚至不需要安装 OpenMM。

---

## 第一幕——只生成文件，不运行

### LAMMPS：把控制脚本写入磁盘

`Script` 类持有输入文件的文本内容，知道怎么写入磁盘。`Script.from_text` 从字符串创建脚本。存盘前可以调用 `script.preview()` 预览内容——脚本由多个片段程序化拼装时，这尤其有用。

```python
import molpy as mp
from molpy.engine import LAMMPSEngine
from molpy.core.script import Script

lammps_input = """\
units           real
atom_style      full
read_data       system.data
include         system.ff

pair_style      lj/cut/coul/long 12.0
kspace_style    pppm 1.0e-4

thermo          1000
run             500000
"""

script = Script.from_text("input", lammps_input, language="other")
print(script.preview())          # 保存前检查

script.save("./submit/input.lmp")
# -> ./submit/input.lmp 已写入
```

这份控制脚本加上 `mp.io.write_lammps_system` 生成的 `system.data` 和 `system.ff`，凑齐一套完整的 LAMMPS 作业。没有预打包的 `.in` 文件——上面的控制脚本*就是*输入卡片组，由 `Script.save` 单独写入。把这三个文件放进 Slurm 提交脚本，集群上不需要装 MolPy 的任何东西。

`Script.from_path` 是逆向操作：加载已有文件，程序化修改，再存回去或传给 `run()`。

```python
script = Script.from_path("./submit/input.lmp")
```

### OpenMM：让引擎代劳

OpenMM 工作流的集成度更高，因为三个必需文件相互依赖——Python 驱动脚本里嵌入了 PDB 和 XML 力场的文件名。`OpenMMEngine.generate_inputs()` 不让你手动拼装，它接受 MolPy 的数据对象，一次性写入全部三个文件。

配置用 Pydantic 模型 `OpenMMSimulationConfig` 描述，每个字段都标注了单位。它支持与 JSON 互转，方便随生成文件一起保存，确保可重复性。

```python
from molpy.engine import OpenMMEngine, OpenMMSimulationConfig

config = OpenMMSimulationConfig(
    ensemble="NPT",
    temperature=300.0,       # K
    pressure=1.0,            # bar
    timestep_fs=2.0,         # fs
    n_steps=500_000,
    platform="CUDA",
)
config.to_json("./omm_run/config.json")

engine = OpenMMEngine(check_executable=False)
paths = engine.generate_inputs(frame, ff, config, "./omm_run")
# paths -> {"pdb": Path("./omm_run/system.pdb"),
#            "forcefield": Path("./omm_run/forcefield.xml"),
#            "script": Path("./omm_run/simulate.py")}
```

`check_executable=False` 让引擎构造时不验证 `python` 是否在 PATH 里。只生成文件时这样用就对了——真正跑模拟的 Python 解释器可能在另一台机器上。

`paths` 字典把字符串键映射为 `Path` 对象。后续可以把 `paths["script"]` 直接丢给 `engine.run()`，或者把三个文件一并交给装了 OpenMM 的集群作业。

---

## 第二幕——直接从 Python 跑

### 本地执行 LAMMPS

MD 可执行文件在本机能用时，`engine.run()` 把脚本写入工作目录然后启动子进程。返回值是标准的 `subprocess.CompletedProcess`，检查退出码、标准输出和错误输出都不需要 MolPy 额外处理。

```python
engine = LAMMPSEngine("lmp")

result = engine.run(
    script,
    workdir="./calc",
    capture_output=True,
    check=True,
)
print(result.returncode)          # 成功时返回 0
if result.stderr:
    print(result.stderr[:500])
```

`check=True` 让 `run()` 在退出码非零时抛出 `subprocess.CalledProcessError`——与 `subprocess.run` 语义一致。跑自动化参数扫描时设成 `check=False`，失败后继续执行，自己检查日志文件。

### MPI 和作业调度器启动器

MPI 并行在构造引擎时配置，不是运行时。传 `launcher` 会在子进程调用中把 MPI 命令加到 LAMMPS 可执行文件前面。

```python
# OpenMPI
engine = LAMMPSEngine("lmp", launcher=["mpirun", "-np", "16"])

# Slurm srun（HPC 集群上常用）
engine = LAMMPSEngine("lmp", launcher=["srun", "--ntasks=16"])

result = engine.run(script, workdir="./calc")
```

实际执行的命令是 `mpirun -np 16 lmp -in input.lmp -log log.lammps -screen none`。`-screen none` 自动加上，防止 LAMMPS 每步都往标准输出写数据，避免 `capture_output=True` 时管道缓冲区死锁。

### Conda 环境激活

有些 HPC 工作流把 LAMMPS 或 OpenMM 装在 Conda 环境里，但提交环境没激活。同时给出 `env` 和 `env_manager` 会用 `conda run` 包装子进程调用：

```python
engine = LAMMPSEngine(
    "lmp",
    env="lammps-env",
    env_manager="conda",
)
result = engine.run(script, workdir="./calc")
# 运行：conda run --no-capture-output -n lammps-env lmp -in input.lmp ...
```

`env` 和 `env_manager` 要么一起给，要么都不给——只给一个会触发 `ValueError`。

### 生成输入后运行 OpenMM

`generate_inputs()` 生成文件后，拿脚本路径调 `run()` 会在配置的解释器下启动生成的 Python 驱动脚本。

```python
engine = OpenMMEngine("python", env="openmm-env", env_manager="conda")
paths = engine.generate_inputs(frame, ff, config, "./omm_run")

result = engine.run(paths["script"], workdir="./omm_run", capture_output=True)
```

驱动脚本是自包含的：导入 OpenMM，从同一目录读 PDB 和 XML，然后运行。`generate_inputs` 和 `run` 之间可以直接手动编辑 `simulate.py`，不需要改 Python 代码。

---

## 场景选引擎

LAMMPS 适合复杂键合力场的经典 MD、用 `fix bond/react` 的反应体系、或任何需要 LAMMPS 特有 fix 命令的工作流。CP2K 适合 QM/MM、DFT 能量评估和从头算分子动力学。OpenMM 适合 GPU 加速的经典 MD 和炼金术 MD，或者你想用 Python 表达模拟逻辑的场景——`generate_inputs` 生成可编辑的人类可读脚本，不是二进制作业文件。

向集群调度器提交任务时用仅生成模式；运行前要检查或编辑文件时用仅生成模式；引擎可执行文件没装在本机上时也用它。本地原型开发、自动化参数扫描、CI 验证需要检查返回码和日志输出时，用 `run()`。

---

## 加新引擎只需三个方法

每个引擎继承 `Engine`，实现三个东西：返回人类可读标识符的 `name` 属性、返回主输入文件扩展名的 `_get_default_extension()`、以及构造子进程命令并调用 `subprocess.run` 的 `_execute()`。基类负责工作目录管理、脚本规范化、启动器前缀和 Conda 环境包装。

```python
from molpy.engine.base import Engine
import subprocess
from pathlib import Path

class GromacsEngine(Engine):
    @property
    def name(self) -> str:
        return "GROMACS"

    def _get_default_extension(self) -> str:
        return ".mdp"

    def _execute(self, run_dir: Path, capture_output=False,
                 check=True, timeout=None, **kwargs):
        cmd = self._build_full_command(
            ["grompp", "-f", self.input_script.path.name,
             "-o", "topol.tpr"]
        )
        return subprocess.run(cmd, cwd=run_dir,
                              capture_output=capture_output,
                              text=True, check=check, timeout=timeout)
```

`_build_full_command` 自动前置启动器和 Conda 包装。

---

## 参见

- [I/O 子系统](11_io.md)——写入 LAMMPS data 文件、力场系数文件、PDB 和 GRO 文件；引擎假设这些文件在运行前已就绪。
- [通过 AmberTools 处理 PEO-LiTFSI 电解质](13_ambertools_integration.md)——端到端工作流示例，写入 AMBER 输入文件并调用外部工具，展示同样的先生成后运行模式用于不同工具链。
- API 参考：`molpy.engine`、`molpy.core.script.Script`。
