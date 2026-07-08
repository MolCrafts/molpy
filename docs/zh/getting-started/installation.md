# 安装

MolPy 需要 Python 3.12+。用 pip 安装：

```bash
pip install molcrafts-molpy
```

## 可选依赖

MolPy 核心只装必要依赖，用到特定功能时再安装对应的附加组件：

| 分组 | 命令 | 使用场景 |
|-------|---------|-------------|
| **全部** | `pip install molcrafts-molpy[all]` | 装下面所有组件 |
| **开发** | `pip install molcrafts-molpy[dev]` | 测试、覆盖率、工具链 |
| **文档** | `pip install molcrafts-molpy[doc]` | 本地构建文档 |

## 每日构建

前沿快照发布在**独立**的 PyPI 项目 `molcrafts-molpy-nightly` 上，每次推送到 `nightly` 分支时自动更新。版本号格式为 `X.Y.Z.devN`（PEP 440 开发版），需要用 `--pre` 参数安装：

```bash
pip install --pre molcrafts-molpy-nightly
```

nightly 包导入时也用 `molpy`，所以它**不能**和稳定版同时安装（类似于 `tensorflow` 和 `tf-nightly` 的关系）。要么先卸载 `molcrafts-molpy`，要么建个专用虚拟环境来测试。

## 验证安装

跑个简单的导入检查，确认包已就绪，同时看看安装位置。

```python
import molpy as mp

print("MolPy:", mp.version)
print("Released on:", mp.release_date)
```

## 下一步

- 看[快速入门](quickstart.md)，走一遍构建、类型化、导出的完整流程。
- 再往后，[数据模型教程](../tutorials/index.md)讲解核心概念，[操作指南](../user-guide/index.md)按任务场景给出具体方案。
