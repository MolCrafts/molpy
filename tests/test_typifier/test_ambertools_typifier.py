"""AmberToolsTypifier: user-tunable radius + types-only snapshot (charges are
conserved by the reacter, never recomputed from a capped fragment — strategy B).

These exercise the pure-Python surface (radius knob + snapshot) with a stub
``result`` frame, so they need no AmberTools executables.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import molpy as mp
import pytest

from molpy.typifier.affected_region import AffectedRegion
from molpy.typifier.ambertools import AmberToolsTypifier
from molpy.typifier.scope import TypeScope


def _linear_carbon_region(interior_reach: int = 1) -> AffectedRegion:
    """Ball around the middle carbon of a short C5 chain."""
    s = mp.Atomistic()
    atoms = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(5)]
    for a, b in zip(atoms, atoms[1:], strict=False):
        s.def_bond(a, b)
    return AffectedRegion._from(
        s, [atoms[2]], extract_radius=interior_reach + 1, interior_reach=interior_reach
    )


def test_reach_is_required_and_drives_both_radii():
    amber = Mock()
    # antechamber is a black box: molpy cannot derive its reach, so there is no
    # default to fall back on.
    with pytest.raises(TypeError):
        AmberToolsTypifier(amber)  # type: ignore[call-arg]

    assert AmberToolsTypifier(amber, reach=2).scope == TypeScope(reach=2)
    # write-back ball(touched, 2); extraction ball(touched, 4)
    assert AmberToolsTypifier(amber, reach=2).scope.interior_reach == 2
    assert AmberToolsTypifier(amber, reach=2).scope.extract_radius == 4
    assert AmberToolsTypifier(amber, reach=6).scope.extract_radius == 12


def test_snapshot_captures_types_not_charges():
    """The interior snapshot records the retyped ``type`` only; any charge on the
    parameterised frame is deliberately ignored (conserved elsewhere)."""
    region = _linear_carbon_region(interior_reach=1)
    frame = region.to_frame()
    n = len(list(region.atoms))
    frame["atoms"]["type"] = ["c3"] * n
    frame["atoms"]["charge"] = [0.5] * n  # present, must be ignored
    result = types.SimpleNamespace(frame=frame)

    snap = AmberToolsTypifier._snapshot(region, result)

    assert snap.atoms, "interior atom(s) should be captured"
    for _pos, info in snap.atoms:
        assert info.type == "c3"
        assert info.params == ()  # no charge captured -> strategy B


def test_typify_region_uses_gas_charges():
    """typify_region drives the wrapper with ``gas`` charges — only types +
    bonded params (charge-method independent) are read back, so ``sqm`` is skipped."""
    region = _linear_carbon_region(interior_reach=1)
    frame = region.to_frame()
    n = len(list(region.atoms))
    frame["atoms"]["type"] = ["c3"] * n
    frame["atoms"]["charge"] = [0.0] * n

    amber = Mock()
    amber.parameterize.return_value = types.SimpleNamespace(
        frame=frame, ff="FF", forcefield="FF"
    )
    typ = AmberToolsTypifier(amber, reach=1)
    typ.typify_region(region)

    _, kwargs = amber.parameterize.call_args
    assert kwargs["charge_method"] == "gas"
