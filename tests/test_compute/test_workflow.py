"""Tests for molpy.compute.Workflow — DAG-based compute orchestration.

Covers:
- Topological ordering (linear chain)
- Diamond reuse (upstream called exactly once)
- Cycle detection with rollback
- Missing external input validation
- Duplicate node rejection
- Multi-input real computation (NeighborList → RDF)
- predecessors() excludes externals
- Rerun does not mutate nodes
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import molpy
import molrs
from molpy.compute import RDF, NeighborList
from molpy.compute.base import Compute
from molpy.compute.workflow import (
    Workflow,
    WorkflowCycleError,
    WorkflowDuplicateNodeError,
    WorkflowError,
    WorkflowMissingInputError,
)

# ===================================================================
# Mock Compute nodes
# ===================================================================


class _MockCompute(Compute):
    """A mock Compute that returns a configurable value and tracks calls.

    Parameters
    ----------
    name : str
        Arbitrary label for dump() output.
    transform : callable or None
        If set, ``__call__(**kwargs)`` returns ``transform(kwargs)``.
        Otherwise returns ``return_value``.
    return_value : object
        Default return value (used when *transform* is None).
    """

    def __init__(
        self,
        name: str = "mock",
        transform=None,
        return_value=None,
    ):
        super().__init__(name=name)
        self._name = name
        self._transform = transform
        self._return_value = return_value
        self.call_count = 0

    def _compute(self, input):
        raise NotImplementedError("use __call__ directly")

    def __call__(self, **kwargs):
        self.call_count += 1
        if self._transform is not None:
            return self._transform(kwargs)
        return self._return_value


class _CountingCompute(Compute):
    """A mock Compute that increments a counter on each call.

    This is used by the diamond reuse test to verify that an upstream node
    is invoked exactly once even when multiple downstreams consume its output.

    Parameters
    ----------
    return_value : object
        Value returned by __call__.
    """

    def __init__(self, return_value=None):
        super().__init__()
        self._return_value = return_value
        self.call_count = 0

    def _compute(self, input):
        raise NotImplementedError("use __call__ directly")

    def __call__(self, **kwargs):
        self.call_count += 1
        return self._return_value


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def simple_frame():
    """A minimal Frame with a few random atoms, cubic Box = 10 A."""
    rng = np.random.default_rng(42)
    xyz = rng.uniform(0.0, 10.0, size=(10, 3))
    frame = molpy.Frame()
    frame["atoms"] = {"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]}
    frame.box = molpy.Box.cubic(10.0)
    return frame


# ===================================================================
# Test 1: Topological order for a linear chain
# ===================================================================


def test_topological_order_linear_chain():
    """Linear a→b chain: topological_order() == ['a', 'b']."""
    a = _MockCompute(return_value=1)
    b = _MockCompute(return_value=2)

    wf = Workflow()
    wf.add("a", a)
    wf.add("b", b, inputs={"x": "a"})

    order = wf.topological_order()
    assert order == ["a", "b"], f"Expected ['a', 'b'], got {order}"


# ===================================================================
# Test 2: Diamond reuse — upstream runs exactly once
# ===================================================================


def test_diamond_reuse_runs_upstream_once():
    """Upstream 'a' is called exactly once when two downstreams consume it."""
    a = _CountingCompute(return_value=42)
    b = _MockCompute(return_value=1)
    c = _MockCompute(return_value=2)

    wf = Workflow()
    wf.add("a", a)
    wf.add("b", b, inputs={"x": "a"})
    wf.add("c", c, inputs={"x": "a"})

    results = wf.run()

    assert a.call_count == 1, f"Expected a called once, got {a.call_count}"
    assert results["a"] == 42
    assert "b" in results
    assert "c" in results


# ===================================================================
# Test 3: Cycle detection with rollback
# ===================================================================


def test_cycle_raises():
    """Adding a new node that closes a cycle via back-propagation raises
    WorkflowCycleError; workflow state is rolled back."""
    a = _MockCompute(return_value=1)
    b = _MockCompute(return_value=2)

    wf = Workflow()
    wf.add("a", a, inputs={"x": "b"})  # a depends on b (b not yet registered)

    # Adding b depending on a resolves the pending a→b edge AND
    # creates b→a, closing a cycle. The add must be atomic.
    with pytest.raises(WorkflowCycleError):
        wf.add("b", b, inputs={"y": "a"})

    # Rollback: only "a" remains, and its predecessor set is empty
    # (the back-propagation of a→b was part of the rolled-back add)
    assert set(wf.nodes) == {"a"}
    assert wf.predecessors("a") == set()


# ===================================================================
# Test 4: Missing external input
# ===================================================================


def test_missing_external_input_raises():
    """run() without a required external input raises WorkflowMissingInputError
    with the missing name exposed."""
    a = _MockCompute(return_value=1)

    wf = Workflow()
    wf.add("a", a, inputs={"x": "external_x"})

    with pytest.raises(WorkflowMissingInputError) as excinfo:
        wf.run()

    assert "external_x" in str(excinfo.value)
    # The exception should also carry a .missing attribute
    missing = getattr(excinfo.value, "missing", None)
    if missing is not None:
        assert "external_x" in missing


# ===================================================================
# Test 5: Duplicate node name
# ===================================================================


def test_duplicate_node_name_raises():
    """Adding a second node with the same name raises
    WorkflowDuplicateNodeError."""
    a = _MockCompute(return_value=1)
    b = _MockCompute(return_value=2)

    wf = Workflow()
    wf.add("a", a)

    with pytest.raises(WorkflowDuplicateNodeError):
        wf.add("a", b)


# ===================================================================
# Test 6: Multi-input real computation chain (NeighborList -> RDF)
# ===================================================================


def test_multi_input_node_rdf(simple_frame):
    """Chain NeighborList -> RDF; results['rdf'] is a molrs.RDFResult
    with non-negative entries."""
    nlist = NeighborList(cutoff=5.0)
    rdf_compute = RDF(n_bins=100, r_max=10.0)

    wf = Workflow()
    # NeighborList: __call__ parameter is "input" (from Compute base class)
    wf.add("nlist", nlist, inputs={"input": "frame"})
    # RDF: __call__ parameters are "frames" and "neighbors"
    wf.add("rdf", rdf_compute, inputs={"frames": "frame", "neighbors": "nlist"})

    results = wf.run(frame=simple_frame)

    assert "rdf" in results
    rdf_result = results["rdf"]
    assert isinstance(rdf_result, molrs.RDFResult), (
        f"Expected molrs.RDFResult, got {type(rdf_result)}"
    )
    rdf_array = np.asarray(rdf_result.rdf)
    assert len(rdf_array) > 0
    assert (rdf_array >= 0.0).all(), "RDF entries must be non-negative"


# ===================================================================
# Test 7: predecessors() excludes external inputs
# ===================================================================


def test_predecessors_excludes_externals():
    """Node 'b' depends on node 'a' and external 'y'; predecessors('b')
    returns {'a'} only."""
    a = _MockCompute(return_value=1)
    b = _MockCompute(return_value=2)

    wf = Workflow()
    wf.add("a", a, inputs={"x": "external_x"})
    wf.add("b", b, inputs={"src": "a", "y": "external_y"})

    preds_b = wf.predecessors("b")
    assert preds_b == {"a"}, f"Expected {{'a'}}, got {preds_b}"

    # 'a' depends only on external_x, so its predecessor set is empty
    preds_a = wf.predecessors("a")
    assert preds_a == set(), f"Expected empty set, got {preds_a}"


# ===================================================================
# Test 8: Rerun does not mutate nodes
# ===================================================================


def test_rerun_does_not_mutate_nodes():
    """Running the same workflow twice produces equal results;
    node dump() is unchanged before and after."""
    a = _MockCompute(return_value=42)
    b = _MockCompute(return_value=99)

    wf = Workflow()
    wf.add("a", a, inputs={"x": "external_x"})
    wf.add("b", b, inputs={"src": "a"})

    # Snapshot node dumps before any run
    dump_before = {"a": a.dump(), "b": b.dump()}

    # Two runs with identical externals
    results1 = wf.run(external_x=10)
    results2 = wf.run(external_x=10)

    # Results must be deep-equal
    assert results1 == results2, "Results differ between runs"

    # Node dumps must be unchanged after run (Workflow does not mutate nodes)
    dump_after = {"a": a.dump(), "b": b.dump()}
    assert dump_before == dump_after, "Node dump changed after run"
