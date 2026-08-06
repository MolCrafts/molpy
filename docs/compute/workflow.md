# Workflow

A real analysis is rarely one compute. You build a neighbour list, then feed it
to $g(r)$, to `LocalDensity`, and to `Steinhardt`; you cluster, then measure
shapes of the clusters. Written as a script that is a pile of intermediate
variables in the right order, and the ordering is implicit in where you happened
to put the lines.

`Workflow` makes the ordering explicit. You declare which node consumes which,
and it works out the execution order itself.

## What it is, and what it deliberately is not

A `Workflow` is a directed acyclic graph of callables. Each node has a name, a
callable, and a mapping from *its own parameter names* to the names of upstream
nodes or external inputs. On `run()` the graph is topologically sorted, each
node is called as `node(**resolved)`, and every result is returned in a dict
keyed by node name.

That is the whole model. There is no scheduler, no caching, no parallelism, no
progress bar, and no attempt to inspect your function signatures — binding is by
name, and if the names do not line up you get an error rather than a guess.

The value is not automation, it is that **the dependency structure becomes a
declaration you can read**, and that a missing input fails before anything runs
instead of halfway through a long job.

## Building one

```python
import numpy as np
import molpy as mp
from molpy.compute import NeighborList, RDF, Workflow

rng = np.random.default_rng(0)
xyz = rng.uniform(0.0, 20.0, size=(300, 3))
frame = mp.Frame()
frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
frame.box = mp.Box.cubic(20.0)
```

Each `add` call names the node, gives the callable, and maps parameter names to
sources:

```python
workflow = (
    Workflow()
    .add("nlist", lambda frame: [NeighborList(cutoff=8.0)(frame)],
         {"frame": "frame"})
    .add("rdf", lambda frames, neighbors: RDF(n_bins=80, r_max=8.0)(frames, neighbors),
         {"frames": "frames", "neighbors": "nlist"})
)

print(sorted(workflow.nodes))              # -> ['nlist', 'rdf']
print(sorted(workflow.external_inputs))    # -> ['frame', 'frames']
print(list(workflow.topological_order()))  # -> ['nlist', 'rdf']
```

Two details in that snippet are worth pausing on, because they are the whole
example. The `nlist` lambda wraps its result in a **list** because the consumer
is `RDF`, which takes parallel lists of frames and neighbour lists — the node
must hand on the shape the next node expects, and `Workflow` does no adapting.
And the same frame is supplied twice, as `frame=` for the neighbour search
(which takes one frame) and as `frames=[frame]` for `RDF` (which takes a list);
on a real trajectory those would be one frame and the whole list.

`add` returns `self`, so the calls chain. Read the second node's mapping as:
"my parameter `neighbors` comes from the node called `nlist`; my parameter
`frames` comes from outside." Anything not matching a registered node name
becomes an **external input**, which is how `external_inputs` gets computed —
and it is worth printing, because a typo in a source name shows up there as an
unexpected external rather than as a crash.

Mind the inconsistency while you are here: `nodes` and `external_inputs` are
plain attributes, while `topological_order()` and `predecessors()` are methods.

Then run it, supplying the externals as keyword arguments:

```python
results = workflow.run(frame=frame, frames=[frame])
print(sorted(results))                   # -> ['nlist', 'rdf']
print(results["rdf"].n_frames)           # -> 1
```

Every node's output is in the dict, not just the last one — intermediates are
usually worth keeping.

Forget an external and it says so up front:

```python
try:
    workflow.run(frame=frame)
except Exception as error:
    print(type(error).__name__)          # -> WorkflowMissingInputError
```

The other two failure modes are `WorkflowDuplicateNodeError` for a repeated node
name and `WorkflowCycleError` if an edge would close a loop — both raised at
`add` time, before any computation happens.

## When it earns its keep

Honestly: not for two nodes. The example above is longer than the four lines it
replaces, and for a linear pipeline you should just write the four lines.

It starts paying when

- **several consumers share one expensive input** — one neighbour list feeding
  RDF, density, and order parameters, where you want to be sure it is built once
  and not silently rebuilt;
- **the graph branches and rejoins**, so the correct execution order is no longer
  obvious from reading top to bottom;
- **the same analysis runs over many systems**, and you want the pipeline
  defined once and the inputs varied;
- **you want the dependency structure to be inspectable** — `topological_order()`
  and `external_inputs` are a description of your analysis that cannot drift out
  of date with the code, unlike a comment.

If none of those apply, a script is the better tool, and the
[compute overview](index.md) shows the direct style used throughout these pages.

## When it goes wrong

**`WorkflowMissingInputError` naming something you thought was a node.**
The source name in an `inputs` mapping did not match any registered node, so it
was treated as an external. Check spelling, and check ordering — a node can only
be referenced by name after it is added.

**A node gets the wrong argument.**
Binding is by parameter name, not position. The keys of the `inputs` dict must
match your callable's parameter names exactly.

**`WorkflowCycleError` on a graph you believe is acyclic.**
Two nodes reference each other, usually via an intermediate. Print
`workflow.predecessors(name)` for the node named in the error.

**It is no faster than the script it replaced.**
It will not be. Execution is sequential and nothing is cached; the win is
structure, not speed.

**A node runs twice.**
It does not — each node executes exactly once per `run()`. If work is being
repeated, it is inside one of your callables.

## Check yourself

- Print `topological_order()` and confirm it matches the order you would have
  written by hand. If it does not, one of your `inputs` mappings is wrong.
- Print `external_inputs` before running. Everything there should be something
  you intend to supply; anything unexpected is a typo'd node name.
- Add a deliberate cycle and confirm it is rejected at `add` time.

## See also

- [Compute overview](index.md) — the direct, un-orchestrated style
- [NeighborList](neighborlist.md) — the input most often shared between nodes
- [API reference](../api/compute.md)
