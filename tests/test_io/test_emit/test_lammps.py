"""LammpsEmitter: data + settings from the force field, init and run scripts."""

import pytest

import molpy as mp
from molpy.io.emit import LammpsEmitter


@pytest.fixture
def water_ff() -> mp.ForceField:
    """The water fixture's types plus ``NA``, a pair type no water atom uses."""
    ff = mp.ForceField("tip3p", units="real")
    atoms = ff.def_style("atom", "full")
    ow = atoms.def_type("OW", mass=15.999)
    hw = atoms.def_type("HW", mass=1.008)
    na = atoms.def_type("NA", mass=22.99)
    ff.def_style("bond", "harmonic").def_type("OW-HW", ow, hw, k=900.0, r0=0.9572)
    pairs = ff.def_style("pair", "lj/cut", {"cutoff": 10.0})
    pairs.def_type("OW", ow, epsilon=0.1521, sigma=3.1507)
    pairs.def_type("HW", hw, epsilon=0.046, sigma=0.4)
    pairs.def_type("NA", na, epsilon=0.13, sigma=2.35)
    return ff


class TestLammpsEmitter:
    def test_writes_the_four_files_and_derives_styles(self, tmp_path, water, water_ff):
        paths = LammpsEmitter().emit(
            water, water_ff, tmp_path, prefix="w", units="real"
        )
        assert [p.name for p in paths] == [
            "w.data",
            "w.in.settings",
            "w.in.init",
            "w.in",
        ]
        assert all(p.exists() for p in paths)
        init = paths[2].read_text()
        assert "units real" in init
        assert "atom_style full" in init
        assert "bond_style harmonic" in init
        assert "pair_style lj/cut" in init
        run = paths[3].read_text()
        assert "read_data w.data" in run and "include w.in.settings" in run
        assert "3 atoms" in paths[0].read_text()

    def test_settings_keyed_by_emitted_frame_labels(self, tmp_path, water, water_ff):
        LammpsEmitter().emit(water, water_ff, tmp_path, prefix="w")
        pair_labels = {
            line.split()[1]
            for line in (tmp_path / "w.in.settings").read_text().splitlines()
            if line.startswith("pair_coeff")
        }
        assert pair_labels == {"OW", "HW"}

    def test_two_styles_in_one_category_raise(self, tmp_path, water, water_ff):
        ow = water_ff.get_style("atom", "full").get_type_by_name("OW")
        water_ff.def_style("bond", "morse").def_type(
            "OW-OW", ow, ow, D0=100.0, alpha=2.0, r0=1.0
        )
        with pytest.raises(ValueError, match="harmonic") as err:
            LammpsEmitter().emit(water, water_ff, tmp_path, prefix="w")
        assert "morse" in str(err.value)

    def test_settings_leave_units_to_init(self, tmp_path, water, water_ff):
        # The settings are included after read_data, where LAMMPS rejects `units`.
        LammpsEmitter().emit(water, water_ff, tmp_path, prefix="w", units="real")
        settings = (tmp_path / "w.in.settings").read_text().splitlines()
        assert not [line for line in settings if line.startswith("units")]
