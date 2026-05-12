---
slug: compute-workflow
status: done
created: 2026-05-12
revised: 2026-05-12
---

# compute-workflow — molpy 轻量 Workflow + molrs-compute Graph 移除

## Summary

为 `molpy.compute` 增加一个零依赖（仅依赖标准库 `graphlib`）的轻量
`Workflow` 编排层，让用户通过命名节点 + 命名边将多个 `Compute` 串成 DAG，
自动进行拓扑排序、循环检测、外部输入校验和按拓扑顺序的串行执行（带
diamond 复用）。同步删除 `molrs-compute` 中已存在的 `Graph / Slot /
Store / Inputs / NodeId` 类型族，使 Rust 端只保留无状态的 `Compute`
trait 与各 analysis 模块；编排责任完全上移到 Python 调用方。两侧合并
后，molpy 用户用一个 Pythonic 的 `Workflow` 即可表达 `frame →
NeighborList → RDF` 这类多输入串接，而 molrs 维护者无需再同步两套
互不依赖的 DAG 实现。

## Domain basis

无物理基础；本 spec 为软件架构。

## Design

### Python 侧 (`molpy.compute.Workflow`)

新增一个文件 `src/molpy/compute/workflow.py`，约 80–120 行。核心数据
模型：

- `Workflow` 类持有三份只读视图：
  - `_nodes: dict[str, Compute]` —— 注册顺序保留（dict 插入序）。
  - `_edges: dict[str, dict[str, str]]` —— `_edges[node_name][param_name]
    = source_name`，`source_name` 既可能指向另一个节点名，也可能指向
    外部输入名。
  - `_graph: dict[str, set[str]]` —— 仅包含「节点 → 前驱节点集合」的
    内部 DAG；外部输入不入图（外部输入没有节点，无前驱可言）。
- `add(name, compute, inputs) -> str`：
  1. 重名 → `WorkflowDuplicateNodeError`。
  2. 把 `inputs` 拆成两组：source 是已注册节点的（写入 `_graph[name]`）
     和未注册的外部输入（仅记入 `_edges`，并在 `external_inputs`
     property 中聚合）。
  3. **加入新边后立即用 `graphlib.TopologicalSorter(_graph).prepare()`
     做一次 cycle check**，捕获 `graphlib.CycleError` 并重新抛
     `WorkflowCycleError`（同时回滚刚才对 `_nodes/_edges/_graph` 的修改，
     保证失败原子性）。
  4. 返回 `name`，允许 fluent 链式调用。
- `topological_order() -> list[str]`：每次重建
  `TopologicalSorter(_graph)`，调 `.static_order()` 后 `list()`。
  纯只读，可重复调用。
- `predecessors(name) -> set[str]`：返回 `_graph[name]`（仅节点，按
  spec 第 7 条排除外部输入）。
- `nodes` property → `list[str]`；`external_inputs` property →
  `set[str]`（所有 `_edges` 中 source 不在 `_nodes` 的 source_name
  去重集合）。
- `run(**externals) -> dict[str, Any]`：
  1. 调用 `topological_order()` 拿到执行顺序（同时保证调用时还能再
     检测一次循环）。
  2. 收集所有 `external_inputs`，与 `externals.keys()` 求差集；
     缺项即 `WorkflowMissingInputError(missing=...)`，**先验证再执行**
     （绝不部分跑）。
  3. `results: dict[str, Any] = {}`；按拓扑顺序，对每个节点 `name`
     构造 `resolved = {param: results[src] if src in results else
     externals[src] for param, src in _edges[name].items()}`，
     然后 `results[name] = self._nodes[name](**resolved)`。
  4. 返回 `results`（包含全部节点；调用方按需取子集）。

### 关键不变量

- **多输入通过 kwargs 透传，工作流不窥探节点签名**：
  `RDF.__call__(self, frames, neighbors)` 与
  `NeighborList.__call__(self, frame)` 在 workflow 看来都是
  `node(**resolved)`；Python 自身的参数绑定负责报错（缺参/多参）。
- **节点对象不被 workflow 改写**：`run()` 重复调用时仅写 `results`
  局部 dict，从不触碰 `compute._config` 或其他实例属性 —— 这条
  契约由 `test_rerun_does_not_mutate_nodes` 钉死。
- **Diamond 复用是缓存的副作用，不是单独的代码路径**：每个节点在
  `results` 里只写一次、读 N 次 —— 上游必然只跑一次。
- **`Compute.execute()` / `input_key` / `output_key`（molexp 适配层）
  保留不动**：workflow 不依赖也不替换该路径。

### Rust 侧 (`molrs-compute`)

纯删除，不引入新设计：

- 删除 `src/graph/`（`mod.rs / node.rs / inputs.rs / store.rs /
  topo.rs`）整目录。
- 从 `src/lib.rs` 移除 `pub mod graph;` 与对应
  `pub use graph::{Graph, Inputs, NodeId, Slot, Store};`，重写模块
  doc：删除 "# Graph" 段落、表格里 `Args` 列中描述 graph 时序的
  语言不变（每个 analysis 自身的 `Args` 仍有效），保留 "# Unified
  trait" / "# Results" / "# Available analyses"。
- 删除 `tests/graph_tests.rs` 和 `benches/graph.rs`，并从
  `benches/benchmarks.rs` 移除 `mod graph;` 与对应 `criterion_main!`
  入口。
- `src/traits.rs` 中 `Compute` trait 已经显式声明 "no hidden mutable
  state"（见行 21–22），保持不变；但 doc 注释里指向 `Graph::run` /
  `Store` 的两行必须重写（不再引用已删除的类型）。
- 改写 `README.md`：删除「Core ideas → Graph 图示」和 diamond-reuse
  代码段；新增一段 "Stateless `Compute` — orchestrate from the
  caller"，明确提到 Python 侧用 `molpy.compute.Workflow`。
- 各 analysis 模块（`rdf/`、`msd/`、`cluster/`、`kmeans/`、`pca/`、
  `center_of_mass.rs`、`cluster_centers.rs`、`gyration_tensor.rs`、
  `inertia_tensor.rs`、`radius_of_gyration.rs`）：仅做 audit，确认
  `fn compute(&self, ...)` 没有任何 `&mut self`、没有
  `RefCell/Cell/Mutex/AtomicXxx` 作为 `self` 的字段；`MSD` 中的
  `AtomicUsize` 作为 rayon reduction 的函数局部 `Arc` 是允许的。

### 兼容性

- 现有 `Compute.execute()` 与 `input_key/output_key` 是 molexp 适配
  层，本 spec 不动。`Workflow` 与之独立、互不调用。
- `src/molpy/compute/rdf.py` 直接调 `molrs.RDF.compute(...)`，从未
  引用 Rust 侧的 `Graph`，所以 molrs Graph 删除对 molpy 零下游影响。
- 验证 `molrs-python`（PyO3 bindings）有无再导出 `Graph`：审计任务
  会 grep 双仓库；若发现 hits 在该 binding crate 内，本 spec 在 Phase
  C 中追加最小编辑（不算 scope creep —— 同属 Graph 移除）。

## Files

### Python — `molpy`

| 文件 | 作用 |
|---|---|
| `src/molpy/compute/workflow.py` (new) | `Workflow` + 四个异常类 |
| `src/molpy/compute/__init__.py` | 导出 `Workflow` 与异常 |
| `tests/test_compute/test_workflow.py` (new) | 8 个单元/集成测试 |
| `docs/user/compute/workflow.md` (new) | textbook 风格教程页 |
| `mkdocs.yml` | 把 `workflow.md` 加入 `nav` |

### Rust — `molrs-compute`

| 文件 | 作用 |
|---|---|
| `molrs/molrs-compute/src/graph/mod.rs` | **delete** |
| `molrs/molrs-compute/src/graph/node.rs` | **delete** |
| `molrs/molrs-compute/src/graph/inputs.rs` | **delete** |
| `molrs/molrs-compute/src/graph/store.rs` | **delete** |
| `molrs/molrs-compute/src/graph/topo.rs` | **delete** |
| `molrs/molrs-compute/tests/graph_tests.rs` | **delete** |
| `molrs/molrs-compute/benches/graph.rs` | **delete** |
| `molrs/molrs-compute/src/lib.rs` | 移除 `pub mod graph;` + 重写模块 doc |
| `molrs/molrs-compute/src/traits.rs` | doc 注释去掉 `Graph::run` / `Store` 交叉引用 |
| `molrs/molrs-compute/benches/benchmarks.rs` | 移除 `mod graph;` 与 criterion 入口 |
| `molrs/molrs-compute/README.md` | 重写 Graph 段落，新增 "Stateless `Compute`" 段 |

## Tasks

- [x] RED: Write `tests/test_compute/test_workflow.py::test_topological_order_linear_chain` covering linear `a → b` ordering against `src/molpy/compute/workflow.py`.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_diamond_reuse_runs_upstream_once` with a counting `Compute` upstream and two downstreams.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_cycle_raises` asserting `WorkflowCycleError` is raised by `Workflow.add` when a back-edge closes a cycle.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_missing_external_input_raises` asserting `WorkflowMissingInputError` lists the missing external names.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_duplicate_node_name_raises` asserting `WorkflowDuplicateNodeError` on a second `add("a", …)`.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_multi_input_node_rdf` chaining `NeighborList → RDF` on a random `Frame`, asserting `results["rdf"]` is a `molrs.RDFResult` with positive `rdf` array.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_predecessors_excludes_externals` asserting `predecessors("rdf") == {"nlist"}` when one source is an external input name.
- [x] RED: Write `tests/test_compute/test_workflow.py::test_rerun_does_not_mutate_nodes` running the same workflow twice with identical externals and asserting deep-equal results.
- [x] GREEN: Implement `Workflow` + `WorkflowError / WorkflowCycleError / WorkflowMissingInputError / WorkflowDuplicateNodeError` in `src/molpy/compute/workflow.py`.
- [x] GREEN: Export `Workflow` and the four exceptions from `src/molpy/compute/__init__.py` (`__all__` updated).
- [x] Write textbook-style tutorial `docs/user/compute/workflow.md` and register it in `mkdocs.yml`.
- [x] Audit each `molrs/molrs-compute/src/{rdf,msd,cluster,kmeans,pca}/` and `*_tensor.rs / center_of_mass.rs / cluster_centers.rs / radius_of_gyration.rs` for on-`self` mutable state (`&mut self`, `RefCell`, `Cell`, `Mutex`, `AtomicXxx` as field); record findings and FAIL-fast if any genuine state appears.
- [x] Decouple `molrs/molrs-compute/src/graph/` directory — keep on disk, remove from crate compilation (`pub mod graph` dropped). Move `tests/graph_tests.rs` to `tests/disabled/`. Keep `benches/graph.rs` on disk, remove from benchmarks.rs.
- [x] Edit `molrs/molrs-compute/src/lib.rs` to remove `pub mod graph;` + re-exports and rewrite the module-level doc (drop `# Graph` section).
- [x] Edit `molrs/molrs-compute/src/traits.rs` doc comments to drop `Graph::run` / `Store` cross-references; verify trait body unchanged.
- [x] Edit `molrs/molrs-compute/benches/benchmarks.rs` to drop `mod graph;` and the graph criterion groups from `criterion_main!`.
- [x] Rewrite `molrs/molrs-compute/README.md`: remove Graph picture + diamond-reuse example, add "Stateless `Compute` — orchestrate from the caller" section pointing at `molpy.compute.Workflow`.
- [x] Verify with `rg -n 'molrs_compute::Graph|molrs\.compute\.Graph|pub use graph::|pub mod graph' molpy/ molrs/` returns no hits outside the deleted paths.
- [x] Run full check + test suite: `pytest tests/ -m "not external" -v`, `pre-commit run --all-files`, and (in molrs) `cargo test -p molcrafts-molrs-compute && cargo clippy -p molcrafts-molrs-compute -- -D warnings && cargo bench -p molcrafts-molrs-compute --no-run`.

## Testing strategy

- **Happy path (Python)**：linear chain、diamond fan-out + reuse、
  multi-input RDF chain —— 由 `test_topological_order_linear_chain`、
  `test_diamond_reuse_runs_upstream_once`、`test_multi_input_node_rdf`
  覆盖。
- **Error paths (Python)**：循环（`add` 时）、缺失外部输入（`run` 时）、
  节点名重复（`add` 时）—— 由 `test_cycle_raises`、
  `test_missing_external_input_raises`、
  `test_duplicate_node_name_raises` 覆盖。
- **只读 introspection (Python)**：`predecessors` 只返回节点名（排除
  外部输入）；`topological_order` / `nodes` / `external_inputs` 可在
  未 `run` 前调用 —— 由 `test_predecessors_excludes_externals` 覆盖。
- **非突变契约 (Python)**：相同外部输入下两次连续 `run` 产出深度相等
  的结果，且节点对象 `dump()` 在 run 前后字节一致 —— 由
  `test_rerun_does_not_mutate_nodes` 覆盖。
- **回归网 (Python)**：现有 `pytest tests/ -m "not external"` 必须
  全绿；新模块仅触碰 `compute/__init__.py` 的 export。
- **Rust audit**：grep 静态检查 ——
  `! rg -nq 'mut self' molrs/molrs-compute/src/{rdf,msd,cluster,kmeans,pca}/`
  外加对 `MSD::compute` 中任何字段级 `Atomic*` 的人工 review。
  任何阳性命中触发后续 refactor 子任务。
- **Rust build/test/clippy/bench**：删除后 `-p molcrafts-molrs-compute`
  上的 `cargo test`、`cargo clippy -- -D warnings`、
  `cargo bench --no-run` 都必须通过。
- **跨仓 grep**：`rg -n 'molrs_compute::Graph|molrs\.compute\.Graph'
  molpy/ molrs/` 返回 0 命中。
- **Docs build**：`mkdocs build` 在新 tutorial 注册到 nav 后成功。

## Out of scope

- 并行 / 流式执行（`TopologicalSorter.get_ready() / .done()` API）——
  v2 follow-up。
- Workflow 可视化（`to_dot()`、mermaid emission）—— v2。
- 把 workflow 定义持久化为 YAML / JSON —— v2（节点本身已有
  `Compute.dump()`）。
- 条件 / 动态节点（skip-on-condition、对集合做 fan-out）—— v2。
- 节点之间的类型检查（Python 动态类型；v1 契约是 kwarg-name 对齐）。
- 删除 `Compute.execute()` / `input_key` / `output_key` —— 那是
  molexp-compat 表面，本 spec 不动。
- 在 Rust 端重新引入任何 Graph / DAG / Store 抽象 —— 本 spec 明确
  禁止。
- 把 `mcd.py` / `pmsd.py` / `rdkit.py` 重写成 `Workflow` 示例 ——
  out of scope；tutorial 用 `NeighborList → RDF`。
