"""Unit tests for :mod:`molpy.builder.polymer.ambertools.amber_leap`."""

from pathlib import Path

from molpy.builder.polymer.ambertools.amber_leap import (
    generate_leap_script,
    generate_leap_script_with_ions,
)


class TestAmberLeapScripts:
    def test_basic_script_loads_templates_and_saves_outputs(self):
        script = generate_leap_script(
            "gaff2", [Path("M.prepi")], ["HM", "TM"], "polymer"
        )
        assert "source leaprc.gaff2" in script
        assert "mol = sequence { HM TM }" in script
        assert "saveamberparm mol polymer.prmtop polymer.inpcrd" in script

    def test_ion_script_adds_requested_ions_and_box(self):
        script = generate_leap_script_with_ions(
            "gaff2",
            [Path("M.prepi")],
            ["HM", "TM"],
            {"Na+": 2},
            "polymer",
            box_size=(20.0, 21.0, 22.0),
        )
        assert "addIonsRand mol Na+ 2" in script
        assert "set mol box { 20.0 21.0 22.0 }" in script
