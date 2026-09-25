"""Wiring: molpy exposes the native placer by identity.

The placement geometry itself belongs to the native core and is tested there;
the call the assembler makes into a placer is owned by
:class:`~molpy.builder.assembly.GraphAssembler` and tested in
``test_assembler.py`` / ``test_polymer.py``.
"""

import molrs

from molpy.builder.assembly import Placer, TracePlacer


class TestPlacer:
    def test_the_native_placer_is_the_one_molpy_exposes(self):
        assert Placer is molrs.Placer
        assert TracePlacer is molrs.TracePlacer
        assert issubclass(TracePlacer, Placer)


class TestPlacementThroughTheAssembler:
    def test_without_a_placer_the_templates_stay_stacked(self, builder_factory):
        stacked = builder_factory().build_linear("EO", 5)
        positions = [(a["x"], a["y"], a["z"]) for a in stacked.atoms]
        assert len(set(positions)) < len(positions)
