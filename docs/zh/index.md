---
title: MolPy
description: 用于构建、编辑和参数化复杂分子系统的 Python 工具包。
hide:
  - navigation
  - toc
hero:
  kicker: MolPy 手册
  title: MolPy
  description: 面向分子模拟工作流的可编程 Python 工具包。
  install:
    label: 安装
    command: pip install molcrafts-molpy
  badges:
    - img: https://img.shields.io/pypi/v/molcrafts-molpy?color=0284c7&label=PyPI
      href: https://pypi.org/project/molcrafts-molpy/
      alt: PyPI 版本
    - img: https://img.shields.io/pypi/pyversions/molcrafts-molpy?color=0f766e
      href: https://pypi.org/project/molcrafts-molpy/
      alt: Python 版本
    - img: https://img.shields.io/github/stars/MolCrafts/molpy?style=flat&color=c8841d
      href: https://github.com/MolCrafts/molpy
      alt: GitHub 星标
  actions:
    - label: 快速上手
      href: tutorials/
      style: primary
    - label: 浏览示例
      href: getting-started/examples/
    - label: API 参考
      href: api/
---

<h1 class="molcrafts-sr-only">MolPy</h1>

<div class="molcrafts-manual-home molpy-home" markdown>

<section class="molcrafts-manual-section molpy-system-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">管线</span>

## 输入分子描述，输出可运行系统

一个分子历经所有阶段，无需写入磁盘。化学信息、坐标和参数分层存放——可在任意边界暂停，检查已有内容，然后继续推进。

</div>

<div class="molpy-system-panel">
<div class="molpy-system-panel__header">
<span>同一表示 · 六个阶段</span>
<strong>同一个分子图，构建、类型化、填充坐标——全流程走完，直接导出或原地分析</strong>
</div>
<div class="molpy-system-flow">
<div>
<span>01 · 解析/构建</span>
<a href="user-guide/01_parsing_chemistry/"><strong>SMILES、BigSMILES 或文件 → 可编辑的图</strong></a>
</div>
<div>
<span>02 · 编辑</span>
<a href="user-guide/02_assembly/"><strong>在图上进行反应、交联和组装</strong></a>
</div>
<div>
<span>03 · 类型化</span>
<a href="user-guide/06_typifier/"><strong>分配 OPLS-AA / GAFF 类型和参数</strong></a>
</div>
<div>
<span>04 · 填充</span>
<a href="user-guide/09_packing/"><strong>无碰撞填充周期性盒子</strong></a>
</div>
<div>
<span>05 · 导出</span>
<a href="user-guide/11_io/"><strong>LAMMPS、GROMACS、PDB、HDF5 等多种格式</strong></a>
</div>
<div>
<span>06 · 分析</span>
<a href="compute/"><strong>RDF、MSD、序参数、谱——在同一个数据结构上直接计算</strong></a>
</div>
</div>
</div>

[快速入门](getting-started/quickstart/) 展示了一个完整流程；[示例画廊](getting-started/examples/) 收集了一批可直接复制运行的短工作流。

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">实践</span>

## 每个阶段只需几行 Python

下面六张卡片与管线的各阶段一一对应。以聚合物为 *演示领域*，是因为它涉及的耦合、交联和多分散性对编辑机制的每个环节都有挑战性——但这并非 MolPy 建模能力的边界。

</div>

<div class="molcrafts-workflow-list molpy-workflow-list" markdown>

<article markdown>

<div class="molcrafts-workflow-list__meta">01 · 解析/构建</div>

### [用文本描述化学结构](user-guide/01_parsing_chemistry/)

一行 SMILES 或 BigSMILES 就能定义可编辑的结构——无论单个分子还是整条聚合物链。

```python
from molpy.builder import polymer

mol = mp.parser.parse_molecule("CCO")   # 从 SMILES 得到一个分子
peo = polymer("{[<]CCOCC[>]}|10|")      # 或整条链，聚合度 DP = 10
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">02 · 编辑</div>

### [逐个原子地重新连接拓扑](user-guide/02_assembly/)

合并结构、创建和断开化学键、移除离去基团——然后在新连接处重新推导出角和二面角。

```python
chain = mon_a.merge(mon_b)                # 合并两个单体
chain.def_bond(anchor_C, port_O)          # 形成新的 C–O 键
chain.del_atom(o_leave, h1, h2)           # 移除离去的水分子
chain = chain.get_topo(gen_angle=True, gen_dihe=True)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">03 · 类型化</div>

### [在数据层面分配类型](user-guide/06_typifier/)

用 SMARTS 匹配把每个原子、键、角和二面角映射到力场参数——导出前就能检查和验证。

```python
ff    = mp.io.read_xml_forcefield("oplsaa.xml")      # 内置 OPLS-AA
typed = mp.typifier.OplsTypifier(ff).typify(chain)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">04 · 填充</div>

### [使用 Packmol 填满周期性盒子](user-guide/09_packing/)

在目标密度下无碰撞填充，由 Python 驱动 Packmol 来完成。想要纯 Rust 版本？我们的 [molpack](https://molcrafts.github.io/molpack/) 填充器尚在测试阶段，欢迎试用。

```python
from molpy.pack import Packmol, Target, InsideBoxConstraint

target = Target(typed.to_frame(), 500, InsideBoxConstraint(length=30.0))
packed = Packmol()([target], seed=42)     # 一个无碰撞的 Frame
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">05 · 导出</div>

### [写入你的引擎真正能运行的文件](user-guide/11_io/)

每个文件一条命令：同时写出 LAMMPS data 文件和力场系数。GROMACS、PDB 和 HDF5 的写入接口也遵循同样的模式。

```python
packed.box = mp.Box.cubic(30.0)
mp.io.write_lammps_data("system.data", packed, atom_style="full")
mp.io.write_lammps_forcefield("system.ff", ff)
```

</article>

<article markdown>

<div class="molcrafts-workflow-list__meta">06 · 分析</div>

### [将轨迹转换为可观测量](compute/)

同一份数据直接传给 Rust 后端计算内核——两次调用就能得到近邻搜索和 g(r) 结果。另有三十多种分析功能可用。

```python
from molpy.compute import NeighborList, RDF

neighbors = NeighborList(cutoff=8.0)(packed)
result    = RDF(n_bins=50, r_max=8.0)(packed, neighbors)   # 整个盒子的 g(r)
```

</article>

</div>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">设计理念</span>

## 为扩展而生

首先是库，不是黑盒：共享数据模型、Rust 计算内核，加上贯穿始终的显式接缝——你可以取用其中任意一块，替换另一块，不必分叉就能扩展任何部分。

</div>

<dl class="molcrafts-feature-matrix molpy-feature-matrix molpy-feature-cards">
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="18" cy="5" r="3"/><circle cx="6" cy="12" r="3"/><circle cx="18" cy="19" r="3"/><line x1="8.59" x2="15.42" y1="13.51" y2="17.49"/><line x1="15.41" x2="8.59" y1="6.51" y2="10.49"/></svg></span>
<dt><a href="tutorials/02_block_and_frame/">一个数据结构，整个生态</a></dt>
<dd>所有 MolCrafts 工具共享同一套 molrs 后端抽象数据结构——molpack、molvis、molmcp 都能直接读取。工具之间不需要转换器，不需要胶水代码。</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg></span>
<dt><a href="developer/molrs-backend/">全面依托 Rust 内核</a></dt>
<dd><code>Frame</code>、<code>Block</code> 以及每个计算算子的底层都是 molrs——一个 Rust 列存储，提供零拷贝 NumPy 视图和 O(N) 链接网格近邻搜索。</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 8V4H8"/><rect width="16" height="12" x="4" y="8" rx="2"/><path d="M2 14h2"/><path d="M20 14h2"/><path d="M15 13v2"/><path d="M9 13v2"/></svg></span>
<dt><a href="user-guide/15_mcp/">为 LLM 智能体打造</a></dt>
<dd>molmcp 套件通过 MCP 提供 MolPy 的符号、文档和实时结构——智能体能检查 Frame 并调用真实 API，依据源码而非猜测。</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><rect width="7" height="7" x="3" y="3" rx="1"/><rect width="7" height="7" x="14" y="3" rx="1"/><rect width="7" height="7" x="14" y="14" rx="1"/><rect width="7" height="7" x="3" y="14" rx="1"/></svg></span>
<dt><a href="developer/architecture-overview/">使用其中一块，或全部使用</a></dt>
<dd>解析器、构建器、类型化器、填充器以及 I/O 和计算模块之间只通过显式数据通信——没有隐藏的共享状态。只导入需要的那一层，其余可忽略。</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M19.439 7.85c-.049.322.059.648.289.878l1.568 1.568c.47.47.706 1.087.706 1.704s-.235 1.233-.706 1.704l-1.611 1.611a.98.98 0 0 1-.837.276c-.47-.07-.802-.48-.968-.925a2.501 2.501 0 1 0-3.214 3.214c.446.166.855.497.925.968a.979.979 0 0 1-.276.837l-1.61 1.61a2.404 2.404 0 0 1-1.705.707 2.402 2.402 0 0 1-1.704-.706l-1.568-1.568a1.026 1.026 0 0 0-.877-.29c-.493.074-.84.504-1.02.968a2.5 2.5 0 1 1-3.237-3.237c.464-.18.894-.527.967-1.02a1.026 1.026 0 0 0-.289-.877l-1.568-1.568A2.402 2.402 0 0 1 1.998 12c0-.617.236-1.234.706-1.704L4.23 8.77c.24-.24.581-.353.917-.303.515.077.877.528 1.073 1.01a2.5 2.5 0 1 0 3.259-3.259c-.482-.196-.933-.558-1.01-1.073-.05-.336.062-.676.303-.917l1.525-1.525A2.402 2.402 0 0 1 12 1.998c.617 0 1.234.236 1.704.706l1.568 1.568c.23.23.556.338.877.29.493-.074.84-.504 1.02-.968a2.5 2.5 0 1 1 3.237 3.237c-.464.18-.894.527-.967 1.02Z"/></svg></span>
<dt><a href="developer/extending-compute/">没有任何硬编码</a></dt>
<dd>新的计算算子、I/O 格式、力场风格或类型化器都可以从核心外部注册——内部的目录是开放注册表，而非硬编码列表。</dd>
</div>
<div>
<span class="molpy-feature-cards__icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3.85 8.62a4 4 0 0 1 4.78-4.77 4 4 0 0 1 6.74 0 4 4 0 0 1 4.78 4.78 4 4 0 0 1 0 6.74 4 4 0 0 1-4.77 4.78 4 4 0 0 1-6.75 0 4 4 0 0 1-4.78-4.77 4 4 0 0 1 0-6.76Z"/><path d="m9 12 2 2 4-4"/></svg></span>
<dt><a href="developer/coding-style/">端到端类型化</a></dt>
<dd>公共 API 全都带有完整的类型提示，CI 中用 Astral 的 <code>ty</code> 检查。编辑器会给出真实的签名补全，不会回退到 <code>Any</code>。</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section molpy-ecosystem-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">生态</span>

## 一个精心设计的抽象数据结构——无需胶水代码

你在 MolPy 中构建的抽象数据结构，同样能被其他 MolCrafts 工具直接读取。共享一个数据模型，工具之间不需要任何适配器。

</div>

<div class="molpy-stack-feature">
<div class="molpy-stack-feature__viewer">
<script type="module" src="https://cdn.jsdelivr.net/npm/@google/model-viewer@4.0.0/dist/model-viewer.min.js"></script>
<script src="assets/vendor/mv-init.js"></script>
<model-viewer
  class="molpy-aspirin"
  data-src="assets/models/aspirin.glb"
  alt="阿司匹林分子以球棍模型呈现"
  camera-controls
  disable-pan
  auto-rotate
  auto-rotate-delay="0"
  rotation-per-second="26deg"
  interaction-prompt="none"
  environment-image="neutral"
  shadow-intensity="0.28"
  shadow-softness="1"
  exposure="1.08"
  camera-orbit="0deg 76deg auto"
  loading="eager"></model-viewer>
</div>
<div class="molpy-stack-feature__body">
<h3>molvis — 交互式 3D 分子</h3>
<p>在浏览器中渲染核心抽象数据结构：一个独立 JavaScript 库、一个可嵌入 Jupyter 的笔记本查看器，以及一个编辑器，全都基于同一套 GPU 渲染引擎。拖动阿司匹林就能旋转它。</p>
<p><a class="molpy-stack-link" href="https://github.com/MolCrafts/molvis">探索 molvis →</a></p>
</div>
</div>

<dl class="molpy-integration-grid molpy-stack-grid">
<div>
<dt><a href="https://molcrafts.github.io/molpack/">molpack</a></dt>
<dd>分子填充引擎——同一个无碰撞填充器，以 CLI、Rust crate 和 Python 包三种形式提供。</dd>
</div>
<div>
<dt><a href="https://github.com/MolCrafts/molmcp">molmcp</a></dt>
<dd>面向 LLM 智能体的 MCP 服务器——基于图的代码发现以及实时生态提供者。</dd>
</div>
<div>
<dt><a href="https://github.com/MolCrafts/molrs">molrs</a></dt>
<dd>共享的 Rust 分子内核——核心抽象数据结构，带有 Python、WASM 和 C 绑定。</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section molpy-section--flip" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">集成</span>

## 与你已有的工具协同工作

外部工具通过显式适配器和封装器连接——每个集成都是可选的，每个边界都清晰可见。

</div>

<dl class="molpy-integration-grid molpy-integration-list">
<div>
<dt><a href="api/adapter/">RDKit</a></dt>
<dd>分子图和 RDKit Mol 之间的双向转换，可用于嵌入、构象生成和 SMILES 导出。</dd>
</div>
<div>
<dt><a href="user-guide/13_ambertools_integration/">AmberTools</a></dt>
<dd>用 Python 直接驱动 antechamber、parmchk2 和 tleap，生成 GAFF 电荷和拓扑。</dd>
</div>
<div>
<dt><a href="user-guide/09_packing/">Packmol</a></dt>
<dd>通过类型化约束接口，实现周期性盒子的无碰撞填充。</dd>
</div>
<div>
<dt><a href="user-guide/12_engine/">LAMMPS · CP2K · OpenMM</a></dt>
<dd>从 MolPy 数据对象生成完整可运行的输入脚本。</dd>
</div>
<div>
<dt><a href="developer/molrs-backend/">molrs · MCP</a></dt>
<dd>底层的 Rust 列存储和计算内核；MCP 套件向 LLM 智能体提供符号和文档。</dd>
</div>
</dl>

</section>

<section class="molcrafts-manual-section" markdown>

<div class="molcrafts-manual-section__header" markdown>

<span class="molcrafts-manual-eyebrow">找到你的页面</span>

## 手册

手册按阅读目的分为两类。**教程**侧重学习：从运行开始，逐章了解数据模型——适合从头到尾通读。**指南**侧重实践：完整的操作步骤——有具体任务时直接翻到对应篇章。

</div>

<div class="molcrafts-doc-map molpy-doc-map">
<section>
<h3><a href="tutorials/">教程</a></h3>
<p><strong>学习。</strong>安装，十五分钟内运行第一个系统，然后逐章了解数据模型。</p>
</section>
<section>
<h3><a href="user-guide/">指南</a></h3>
<p><strong>上手操作。</strong>端到端配方——解析、构建、类型化、填充、导出——假设已读过教程。</p>
</section>
<section>
<h3><a href="compute/">计算</a></h3>
<p>轨迹分析：分布、输运、序参数、光谱以及分析工作流。</p>
</section>
<section>
<h3><a href="api/">API 参考</a></h3>
<p>每个公共模块，从核心数据结构到引擎适配器。</p>
</section>
<section>
<h3><a href="developer/">开发者指南</a></h3>
<p>贡献流程、架构概览以及新功能的扩展点。</p>
</section>
</div>

</section>

</div>
