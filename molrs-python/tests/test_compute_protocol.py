"""molrs.compute.Compute is a runtime_checkable Protocol every compute class
satisfies."""

from __future__ import annotations

import inspect

import molrs
from molrs.compute import Compute, ComputeSpec


def _compute_classes() -> dict[str, type]:
    """Every exported class of ``molrs.compute`` that has a ``compute``."""
    return {
        name: obj
        for name in molrs.compute.__all__
        if inspect.isclass(obj := getattr(molrs.compute, name))
        and obj is not Compute
        and callable(getattr(obj, "compute", None))
    }


class OnlyCompute:
    def compute(self, **inputs):
        return None


class TestComputeProtocol:
    def test_structural_class_satisfies_the_protocol(self) -> None:
        assert isinstance(OnlyCompute(), Compute)

    def test_every_compute_class_satisfies_the_protocol(self) -> None:
        classes = _compute_classes()
        assert set(classes) == {spec.name for spec in ComputeSpec.all()}
        assert not [
            name for name, cls in classes.items() if not issubclass(cls, Compute)
        ]

    def test_an_instance_satisfies_the_protocol(self) -> None:
        assert isinstance(molrs.compute.Msd(4), Compute)

    def test_a_result_is_not_a_compute(self) -> None:
        assert not issubclass(molrs.compute.TimeCorrelation, Compute)

    def test_contract_is_compute_only(self) -> None:
        members = {n for n in Compute.__dict__ if not n.startswith("_")}
        assert "compute" in members
        assert "dump" not in members
        assert "__call__" not in members

    def test_layout_doc_records_the_protocol(self) -> None:
        assert molrs.compute.__doc__ is not None
        assert "protocol" in molrs.compute.__doc__.lower()

    def test_presence_only_isinstance_ignores_signature(self) -> None:
        class WrongSig:
            def compute(self) -> int:
                return 1

        assert isinstance(WrongSig(), Compute)
