# 计算工作流

分析任务很少只涉及一个计算节点。算径向分布函数之前要先建近邻列表，聚类分析依赖每帧的特征描述符，均方位移算完还得接自相关函数。手工搭这条链——注册节点、串联输出输入、确保上游先于下游运行——既啰嗦又容易出错。

`molpy.compute.Workflow` 把这个过程自动化了：注册命名节点，声明参数来自哪个上游节点还是外部输入，工作流自动处理拓扑排序和结果缓存。除 Python 标准库外没有额外依赖。

## 线性链

最简单的形式是一条线性链。节点 *a* 先运行，节点 *b* 把 *a* 的输出当作自己的 `x` 参数。

```python
from molpy.compute import Workflow
from molpy.compute.base import Compute


class Square(Compute):
    """返回一个数的平方。"""

    def __init__(self):
        super().__init__()

    def _compute(self, x):
        return x * x


class AddOne(Compute):
    """加一。"""

    def __init__(self):
        super().__init__()

    def _compute(self, x):
        return x + 1


wf = Workflow()
wf.add("square", Square())
wf.add("add_one", AddOne(), inputs={"x": "square"})

results = wf.run(x=3)
print(results)  # {'square': 9, 'add_one': 10}
```

`wf.add()` 返回节点名称，支持链式调用：

```python
wf = Workflow()
(wf
 .add("square", Square())
 .add("add_one", AddOne(), inputs={"x": "square"}))
```

## 外部输入

`inputs` 里的参数名找不到对应节点时，工作流把它当作*外部输入*，调用方需要传给 `run()`。

```python
wf = Workflow()
wf.add("square", Square())

# 参数名称 "x" 不匹配任何节点 → 外部输入
results = wf.run(x=5)
print(results)  # {'square': 25}
```

漏了外部输入的话，`run()` 在*执行任何节点之前*直接抛出 `WorkflowMissingInputError`，不会出现半截执行。

```python
try:
    wf.run()
except WorkflowMissingInputError as exc:
    print(exc.missing)  # {'x'}
```

## 菱形复用

两个下游节点共享同一个上游节点时，上游只跑一次。这不是特殊优化——结果缓存自然保证了这一点。

```python
class Count(Compute):
    """统计自己被调用了多少次。"""

    def __init__(self):
        super().__init__()
        self.call_count = 0

    def _compute(self, x):
        self.call_count += 1
        return x


wf = Workflow()
upstream = Count()
wf.add("upstream", upstream)
wf.add("branch_a", AddOne(), inputs={"x": "upstream"})
wf.add("branch_b", AddOne(), inputs={"x": "upstream"})

results = wf.run(x=1)
assert upstream.call_count == 1  # 不是 2
```

## 真实示例：径向分布函数

径向分布函数 g(r) 同时需要帧（取盒子尺寸）和近邻列表（生成成对距离）。用两个节点组成工作流：

```python
import numpy as np
from molpy.compute import Workflow, NeighborList, RDF
import molpy as mp
import molpy

# 构建一个简单的测试帧——10 个原子在 10 Å 的立方体中
rng = np.random.default_rng(42)
xyz = rng.uniform(0.0, 10.0, size=(10, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = molpy.Box.cubic(10.0)

wf = Workflow()
wf.add("nlist", NeighborList(cutoff=5.0))
wf.add("rdf", RDF(n_bins=100, r_max=10.0),
       inputs={"frames": "frame", "neighbors": "nlist"})

results = wf.run(frame=frame)
rdf_array = np.asarray(results["rdf"].rdf)
print(f"g(r) 有 {len(rdf_array)} 个 bins，最大值为 {rdf_array.max():.3f}")
```

`NeighborList` 只依赖帧，参数全部来自外部输入。`RDF` 既需要原始帧（盒子尺寸）也需要近邻列表，所以它的 `inputs` 里既有 `"frame"`（外部）也有 `"nlist"`（上游节点）。

## 内省

运行之前可以检查工作流。

```python
wf.nodes            # ['nlist', 'rdf'] —— 插入顺序
wf.external_inputs  # {'frame'} —— 所有未注册的源名称
wf.topological_order()  # ['nlist', 'rdf'] —— 执行顺序
wf.predecessors("rdf")  # {'nlist'} —— 仅节点前驱（不包括外部输入）
```

`predecessors()` 特意排除外部输入——它描述的是工作流内部 DAG 拓扑，不是全部依赖关系。

## 循环检测

添加节点若导致循环，`add()` 直接抛出 `WorkflowCycleError` 并回滚，工作流状态保持不变。

```python
wf = Workflow()
wf.add("a", Square())

# b 依赖于 a → 没问题
wf.add("b", Square(), inputs={"x": "a"})

# 添加一个产生反向边 a → b 的节点会形成循环
try:
    wf.add("c", Square(), inputs={"x": "b"})  # 没问题，线性
except WorkflowCycleError:
    pass  # 不会到达这里——这是有效的
```

循环检测发生在注册时而非执行时，反馈即时。

## 不可变性约定

工作流不会修改已注册的计算节点。同样的外部输入调用两次 `run()`，两次结果一致；`node.dump()`（序列化节点配置）在运行前后返回相同字典。

```python
results1 = wf.run(x=5)
results2 = wf.run(x=5)
assert results1 == results2  # 始终为真
```

## 何时不使用 Workflow

`Workflow` 是串行同步的，目前不支持并行执行、条件节点或流式处理。这些场景可以直接用 `TopologicalSorter.get_ready()` / `.done()`，或者等后续版本。
