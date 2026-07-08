# 开发环境配置

按本页指引操作，即可完成可编辑安装、pre-commit 钩子和测试验证，搭建一个正常的本地开发环境。

## 前置依赖

依赖只有 Python 3.12+、Git 和 pip，其余由设置脚本自动安装。


## 快速配置

克隆仓库，创建虚拟环境，以可编辑模式安装开发依赖，再跑一遍测试套件验证环境正常。

```bash
git clone https://github.com/MolCrafts/molpy.git
cd molpy
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e ".[dev]"
pre-commit install
pytest tests/ -v -m "not external"
```

全部测试通过即环境就绪。


## 从源码构建 molrs

上文的快速配置从 PyPI 拉取已发布的 molcrafts-molrs wheel，这就是 molrs——molpy 依赖的 Rust 计算核心。大部分 molpy 开发场景走这条路就行。

同时修改 Rust 核心和 molpy 时，改为从本地检出构建可编辑模式的 molrs。molrs 以 maturin 项目发布其 Python 绑定，这步需要先装好 Rust 工具链——通过 rustup 安装即可。molrs 在 rust-toolchain.toml 里固定了工具链频道和组件，进入检出目录后不用手动配置。

```bash
# 在 molpy 旁边的同级检出目录中
git clone https://github.com/MolCrafts/molrs.git
cd molrs
pip install maturin
maturin develop -m molrs-python/Cargo.toml --release   # 将 `molrs` 以可编辑模式安装到 venv

# 回到 molpy，可编辑安装现在会解析本地的 molrs
cd ../molpy
pip install -e ".[dev]"
python -c "import molrs, molpy as mp; print(mp.Box.cubic(10.0), isinstance(mp.Box.cubic(10.0), molrs.Box))"
```

改动 molrs Rust 源码后，重新运行 maturin develop 编译扩展。原生 crate 和 WASM 构建目标见
[molrs 从源码构建指南](https://molrs.molcrafts.org/getting-started/installation/)。


## 文档预览

文档站点用 [Zensical](https://zensical.org) 构建（Material for MkDocs 的继任者），配置写在仓库根目录的 zensical.toml 里。安装文档扩展后，从仓库根目录启动本地预览服务器。

```bash
pip install -e ".[doc]"
zensical serve
```

站点在 http://localhost:8000，修改 .md 文件后即时生效。

用户指南的 notebook 已预渲染为 Markdown（Zensical 构建时不会执行 notebook）。编辑 notebook 后，运行 python scripts/render_notebooks.py 重新生成页面。


## 外部工具

部分测试和工作流依赖外部可执行文件（非 Python 包）：LAMMPS、Packmol、AmberTools。核心开发用不上这些工具。依赖它们的测试均标了 @pytest.mark.external，默认运行时通过 -m "not external" 排除。

已安装其中某个工具、想运行对应测试的话：

```bash
pytest tests/ -v                          # 运行所有测试（包括外部工具测试）
pytest tests/ -v -k "lammps"              # 仅运行与 LAMMPS 相关的测试
```


## 常用命令

```bash
ruff format --check src tests             # 检查格式
ruff format src tests                     # 自动格式化
ruff check src                            # 检查源码树
pytest tests/ -v -m "not external"        # 本地测试套件
pytest --cov=src/molpy tests/ -v          # 带覆盖率
pre-commit run --all-files                # 运行所有 pre-commit 钩子
zensical build                            # 构建静态文档站点到 site/
```


## 故障排除

拉取新代码后导入失败，重新安装可编辑包：pip install -e "`.[dev]`"。要重新生成用户指南 notebook 页面，运行 python scripts/render_notebooks.py（需要 `[doc]` 扩展，包含 RDKit 和 Packmol）。格式化检查在 CI 上失败时，推送前先在本地执行 ruff format src tests。
