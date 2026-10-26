"""molpy.compute.Compute is a runtime_checkable Protocol every batch compute
class satisfies; mp.compute.Stream the one every stream (Msd, Vacf, Rdf)
satisfies."""

from __future__ import annotations

import inspect

import molpy as mp
from molpy.compute import Compute, ComputeSpec, Stream


def _compute_classes() -> dict[str, type]:
    """Every exported class of ``mp.compute`` that has a ``compute``."""
    return {
        name: obj
        for name in mp.compute.__all__
        if inspect.isclass(obj := getattr(mp.compute, name))
        and obj is not Compute
        and callable(getattr(obj, "compute", None))
    }


def _stream_classes() -> dict[str, type]:
    """Every exported class of ``mp.compute`` that has an ``accumulator``."""
    return {
        name: obj
        for name in mp.compute.__all__
        if inspect.isclass(obj := getattr(mp.compute, name))
        and obj is not Stream
        and callable(getattr(obj, "accumulator", None))
    }


class OnlyCompute:
    def compute(self, **inputs):
        return None


class TestComputeProtocol:
    def test_structural_class_satisfies_the_protocol(self) -> None:
        assert isinstance(OnlyCompute(), Compute)

    def test_every_compute_class_satisfies_its_protocol(self) -> None:
        computes, streams = _compute_classes(), _stream_classes()
        assert set(streams) == {"Msd", "Vacf", "Rdf"}
        assert set(computes) == {spec.name for spec in ComputeSpec.all()} - set(streams)
        assert not [
            name for name, cls in computes.items() if not issubclass(cls, Compute)
        ]
        assert not [
            name for name, cls in streams.items() if not issubclass(cls, Stream)
        ]
        assert not [name for name, cls in streams.items() if issubclass(cls, Compute)]

    def test_an_instance_satisfies_the_protocol(self) -> None:
        assert isinstance(mp.compute.Correlator("product", 4), Compute)
        assert isinstance(mp.compute.Msd(4), Stream)
        assert not isinstance(mp.compute.Msd(4), Compute)

    def test_a_result_is_not_a_compute(self) -> None:
        assert not issubclass(mp.compute.TimeCorrelation, Compute)

    def test_contract_is_compute_only(self) -> None:
        members = {n for n in Compute.__dict__ if not n.startswith("_")}
        assert "compute" in members
        assert "dump" not in members
        assert "__call__" not in members

    def test_layout_doc_records_the_protocol(self) -> None:
        assert mp.compute.__doc__ is not None
        assert "protocol" in mp.compute.__doc__.lower()

    def test_presence_only_isinstance_ignores_signature(self) -> None:
        class WrongSig:
            def compute(self) -> int:
                return 1

        assert isinstance(WrongSig(), Compute)
