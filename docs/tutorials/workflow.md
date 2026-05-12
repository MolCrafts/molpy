# Composing Compute Nodes into a Workflow

Analysis tasks often need more than a single compute step. Radial distribution
functions require neighbour lists; cluster analysis feeds on per-frame
descriptors; mean-squared displacements chain into autocorrelation functions.
Doing this by hand — calling each node, threading outputs into inputs,
checking that upstream ran before downstream — is tedious and easy to get
wrong.

`Workflow` makes this mechanical. You register named compute nodes, declare
which parameters come from which upstream node (or from an external input),
and the workflow handles topological order and result caching. It adds zero
dependencies beyond Python's standard library.

## A linear chain

The simplest workflow is a straight pipeline.  Node *a* runs first; node *b*
takes *a*'s output as its `x` parameter.

```python
from molpy.compute import Workflow
from molpy.compute.base import Compute


class Square(Compute):
    """Return the square of a number."""

    def __init__(self):
        super().__init__()

    def _compute(self, x):
        return x * x


class AddOne(Compute):
    """Add one."""

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

`wf.add()` returns the node name, so you can chain calls:

```python
wf = Workflow()
(wf
 .add("square", Square())
 .add("add_one", AddOne(), inputs={"x": "square"}))
```

## External inputs

When a parameter name in `inputs` does not match any registered node, the
workflow treats it as an *external input*. You must supply it to `run()`.

```python
wf = Workflow()
wf.add("square", Square())

# The parameter name "x" does not match any node → external input
results = wf.run(x=5)
print(results)  # {'square': 25}
```

If you forget an external input, `run()` raises `WorkflowMissingInputError`
*before* executing any node — no partial execution.

```python
try:
    wf.run()
except WorkflowMissingInputError as exc:
    print(exc.missing)  # {'x'}
```

## Diamond reuse

When two downstream nodes share an upstream, the upstream runs exactly once.
This is not a special code path — it falls out of the result cache naturally.

```python
class Count(Compute):
    """Counts how many times it was called."""

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
assert upstream.call_count == 1  # not 2
```

## Real example: radial distribution function

A radial distribution function g(r) needs both a frame (for the box) and a
neighbour list (for the pair distances). We express this as a two-node
workflow.

```python
import numpy as np
from molpy.compute import Workflow, NeighborList, RDF
import molpy

# Build a simple test frame — 10 atoms in a 10 Å cube
rng = np.random.default_rng(42)
xyz = rng.uniform(0.0, 10.0, size=(10, 3))
frame = molpy.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = molpy.Box.cubic(10.0)

wf = Workflow()
wf.add("nlist", NeighborList(cutoff=5.0))
wf.add("rdf", RDF(n_bins=100, r_max=10.0),
       inputs={"frames": "frame", "neighbors": "nlist"})

results = wf.run(frame=frame)
rdf_array = np.asarray(results["rdf"].rdf)
print(f"g(r) has {len(rdf_array)} bins, max value {rdf_array.max():.3f}")
```

`NeighborList` needs only the frame, so it appears as a single external input.
`RDF` needs both the original frame (for box dimensions) and the neighbour
list — so its `inputs` map references both `"frame"` (external) and `"nlist"`
(upstream node).

## Introspection

You can inspect the workflow before running it.

```python
wf.nodes            # ['nlist', 'rdf'] — insertion order
wf.external_inputs  # {'frame'} — all unregistered source names
wf.topological_order()  # ['nlist', 'rdf'] — execution order
wf.predecessors("rdf")  # {'nlist'} — node predecessors only (no externals)
```

`predecessors()` deliberately excludes external inputs — it describes the
internal DAG topology, not every dependency.

## Cycle detection

If adding a node would create a cycle, `add()` raises `WorkflowCycleError`
and rolls back — the workflow state is unchanged.

```python
wf = Workflow()
wf.add("a", Square())

# b depends on a → OK
wf.add("b", Square(), inputs={"x": "a"})

# Adding a node that creates a back-edge a → b closes the cycle
try:
    wf.add("c", Square(), inputs={"x": "b"})  # OK, linear
except WorkflowCycleError:
    pass  # not reached — this is valid
```

Cycle detection happens at registration time, not at execution time. You get
immediate feedback.

## Immutability contract

The workflow never mutates registered compute nodes. Calling `run()` twice
with the same external inputs produces identical results, and `node.dump()`
(which serialises the node's configuration) returns the same dictionary before
and after.

```python
results1 = wf.run(x=5)
results2 = wf.run(x=5)
assert results1 == results2  # always true
```

## When not to use Workflow

`Workflow` is serial and synchronous. It does not (yet) support parallel
execution, conditional nodes, or streaming. For those patterns, use
`TopologicalSorter.get_ready()` / `.done()` directly, or wait for a future
version.
