"""AmberToolsTypifier: user-tunable radius + types-only snapshot (charges are
conserved by the reacter, never recomputed from a capped fragment — strategy B).

These exercise the pure-Python surface (radius knob + snapshot) with a stub
``result`` frame, so they need no AmberTools executables.
"""

from __future__ import annotations

import types
from unittest.mock import Mock

import molpy as mp
from molpy.core.affected_region import AffectedRegion, region_radius
from molpy.typifier.ambertools import AmberToolsTypifier


def _linear_carbon_region(radius: int = 1) -> AffectedRegion:
    """Radius ball around the middle carbon of a short C5 chain."""
    s = mp.Atomistic()
    atoms = [s.def_atom(element="C", x=float(i), y=0.0, z=0.0) for i in range(5)]
    for a, b in zip(atoms, atoms[1:], strict=False):
        s.def_bond(a, b)
    return AffectedRegion._from(s, [atoms[2]], radius)


def test_context_radius_is_user_tunable_and_drives_extraction():
    amber = Mock()
    assert AmberToolsTypifier(amber).context_radius == 2  # default
    assert AmberToolsTypifier(amber, context_radius=5).context_radius == 5
    # region_radius trusts the declaration verbatim (no floor-up).
    assert region_radius(AmberToolsTypifier(amber, context_radius=2)) == 2
    assert region_radius(AmberToolsTypifier(amber, context_radius=6)) == 6


def test_snapshot_captures_types_not_charges():
    """The interior snapshot records the retyped ``type`` only; any charge on the
    parameterised frame is deliberately ignored (conserved elsewhere)."""
    region = _linear_carbon_region(radius=1)
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
    region = _linear_carbon_region(radius=1)
    frame = region.to_frame()
    n = len(list(region.atoms))
    frame["atoms"]["type"] = ["c3"] * n
    frame["atoms"]["charge"] = [0.0] * n

    amber = Mock()
    amber.parameterize.return_value = types.SimpleNamespace(
        frame=frame, ff="FF", forcefield="FF"
    )
    typ = AmberToolsTypifier(amber)
    typ.typify_region(region)

    _, kwargs = amber.parameterize.call_args
    assert kwargs["charge_method"] == "gas"
