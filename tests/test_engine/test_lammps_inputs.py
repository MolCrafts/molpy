"""``LammpsEngine.generate_inputs``: data + settings from the force field,
init and input script.

The ``*_style`` lines and coefficients are molrs's LAMMPS include, for every
category the system uses; molpy names no style itself.
"""

import os
import shutil
import subprocess

import pytest

import molpy as mp
from molpy.engine import LammpsEngine


@pytest.fixture
def water_ff() -> mp.ff.forcefield.ForceField:
    """The water fixture's types plus ``NA``, a pair type no water atom uses."""
    ff = mp.ff.forcefield.ForceField("tip3p", units="real")
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


def _deck(water, ff, out, **kw):
    """The water's deck, generated without a LAMMPS binary."""
    return LammpsEngine(check_executable=False).generate_inputs(
        water.to_frame(), ff, out, prefix="w", **kw
    )


class TestGenerateInputs:
    def test_writes_the_four_files_and_derives_styles(self, tmp_path, water, water_ff):
        paths = _deck(water, water_ff, tmp_path, units="real")
        assert {key: p.name for key, p in paths.items()} == {
            "data": "w.data",
            "settings": "w.in.settings",
            "init": "w.in.init",
            "input": "w.in",
        }
        assert all(p.exists() for p in paths.values())
        init = paths["init"].read_text(encoding="utf-8")
        assert "units real" in init
        assert "atom_style full" in init
        assert "_style " not in init.replace("atom_style ", "")
        settings = paths["settings"].read_text(encoding="utf-8")
        assert settings == mp.io.write_lammps_forcefield_str(
            water_ff, water.to_frame(), skip_units=True, units="real"
        )
        assert "bond_style harmonic" in settings
        assert "pair_style lj/cut" in settings
        run = paths["input"].read_text(encoding="utf-8")
        assert "read_data w.data" in run and "include w.in.settings" in run
        assert "3 atoms" in paths["data"].read_text(encoding="utf-8")

    def test_settings_keyed_by_emitted_frame_labels(self, tmp_path, water, water_ff):
        _deck(water, water_ff, tmp_path)
        pair_labels = {
            line.split()[1]
            for line in (tmp_path / "w.in.settings")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.startswith("pair_coeff")
        }
        assert pair_labels == {"OW", "HW"}

    def test_a_style_no_term_uses_is_not_written(self, tmp_path, water, water_ff):
        ow = water_ff.get_style("atom", "full").get_type_by_name("OW")
        water_ff.def_style("bond", "morse").def_type(
            "OW-OW", ow, ow, d0=100.0, alpha=2.0, r0=1.0
        )
        _deck(water, water_ff, tmp_path)
        settings = (tmp_path / "w.in.settings").read_text(encoding="utf-8")
        assert "bond_style harmonic" in settings
        assert "morse" not in settings

    def test_a_category_spanning_two_styles_is_hybrid(self, tmp_path, water, water_ff):
        hw = water_ff.get_style("atom", "full").get_type_by_name("HW")
        ow = water_ff.get_style("atom", "full").get_type_by_name("OW")
        water_ff.def_style("bond", "morse").def_type(
            "OW-HW2", ow, hw, d0=100.0, alpha=2.0, r0=1.0
        )
        water.links.exact_bucket(mp.Bond)[1]["type"] = "OW-HW2"
        _deck(water, water_ff, tmp_path)
        settings = (tmp_path / "w.in.settings").read_text(encoding="utf-8").splitlines()
        assert "bond_style hybrid harmonic morse" in settings

    def test_angle_charmm_carries_urey_bradley(self, tmp_path, water, water_ff):
        """``angle charmm`` (with its Urey-Bradley 1-3 term) is written as
        LAMMPS ``angle_style charmm``, ``K theta0 K_ub r_ub``."""
        _with_urey_bradley(water, water_ff)
        _deck(water, water_ff, tmp_path)
        settings = (tmp_path / "w.in.settings").read_text(encoding="utf-8").splitlines()
        assert "angle_style charmm" in settings
        (coeff,) = [line for line in settings if line.startswith("angle_coeff")]
        assert [float(v) for v in coeff.split()[2:]] == [55.0, 104.52, 20.0, 1.5139]

    def test_a_boxless_system_is_shrink_wrapped_around_its_atoms(
        self, tmp_path, water, water_ff
    ):
        """An Atomistic has no box: ``boundary s s s`` with an all-pairs
        neighbor search, and a data-file box enclosing every atom with a margin
        of 1 length unit, never a ``0 1`` placeholder."""
        assert water.to_frame().box is None
        _deck(water, water_ff, tmp_path)
        init = (tmp_path / "w.in.init").read_text(encoding="utf-8").splitlines()
        assert "boundary s s s" in init and "neighbor 2.0 nsq" in init
        data = (tmp_path / "w.data").read_text(encoding="utf-8").splitlines()
        atoms = water.to_frame()["atoms"]
        for axis in ("x", "y", "z"):
            (bounds,) = [line for line in data if line.endswith(f"{axis}lo {axis}hi")]
            lo, hi = (float(v) for v in bounds.split()[:2])
            assert lo == pytest.approx(min(atoms[axis]) - 1.0)
            assert hi == pytest.approx(max(atoms[axis]) + 1.0)

    def test_settings_leave_units_to_init(self, tmp_path, water, water_ff):
        # The settings are included after read_data, where LAMMPS rejects `units`.
        _deck(water, water_ff, tmp_path, units="real")
        settings = (tmp_path / "w.in.settings").read_text(encoding="utf-8").splitlines()
        assert not [line for line in settings if line.startswith("units")]


class TestAtomStyleAndUnits:
    """The data file, the init and the coefficients agree on ``atom_style``
    and ``units``: molrs's data writer is told both."""

    @staticmethod
    def _lj_fluid() -> tuple[mp.Frame, mp.ff.forcefield.ForceField]:
        ff = mp.ff.forcefield.ForceField("lj", units="real")
        ar = ff.def_style("atom", "full").def_type("AR", mass=39.948)
        ff.def_style("pair", "lj/cut", {"cutoff": 8.5}).def_type(
            "AR", ar, epsilon=0.238, sigma=3.405
        )
        frame = mp.Frame(
            blocks={
                "atoms": {
                    "type": ["AR", "AR"],
                    "x": [0.0, 4.0],
                    "y": [0.0, 0.0],
                    "z": [0.0, 0.0],
                }
            }
        )
        frame.box = mp.Box.cube(20.0)
        return frame, ff

    @staticmethod
    def _data_style(path) -> str:
        (header,) = [
            line
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.startswith("Atoms")
        ]
        return header.split("#")[1].strip()

    def test_atom_style_defaults_to_the_one_the_data_file_is_written_in(self, tmp_path):
        """An LJ fluid (no charge, no mol_id) is ``atomic`` in both files."""
        frame, ff = self._lj_fluid()
        paths = LammpsEngine(check_executable=False).generate_inputs(
            frame, ff, tmp_path
        )
        init = paths["init"].read_text(encoding="utf-8").splitlines()
        assert "atom_style atomic" in init
        assert self._data_style(paths["data"]) == "atomic"

    def test_a_named_atom_style_is_the_data_files(self, tmp_path, water, water_ff):
        paths = _deck(water, water_ff, tmp_path, atom_style="full")
        assert "atom_style full" in paths["init"].read_text(encoding="utf-8")
        assert self._data_style(paths["data"]) == "full"

    def test_units_reach_the_data_file_and_the_coefficients(
        self, tmp_path, water, water_ff
    ):
        paths = _deck(water, water_ff, tmp_path, units="metal")
        assert "units metal" in paths["init"].read_text(encoding="utf-8")
        title = paths["data"].read_text(encoding="utf-8").splitlines()[0]
        assert title.endswith("units = metal")
        assert paths["settings"].read_text(
            encoding="utf-8"
        ) == mp.io.write_lammps_forcefield_str(
            water_ff, water.to_frame(), skip_units=True, units="metal"
        )

    @pytest.mark.parametrize("units", ["si", "nano", "cgs", "electron", "micro"])
    def test_units_molrs_cannot_write_are_refused_before_any_file(
        self, tmp_path, water, water_ff, units
    ):
        with pytest.raises(ValueError, match="units must be one of"):
            _deck(water, water_ff, tmp_path / "deck", units=units)
        assert not (tmp_path / "deck").exists()


def _with_urey_bradley(water: mp.Atomistic, ff: mp.ff.forcefield.ForceField) -> None:
    """Give the water its H-O-H angle, typed ``angle charmm`` with a UB term."""
    atoms = ff.get_style("atom", "full")
    ow, hw = atoms.get_type_by_name("OW"), atoms.get_type_by_name("HW")
    ff.def_style("angle", "charmm").def_type(
        "HW-OW-HW", hw, ow, hw, k=55.0, theta0=104.52, k_ub=20.0, r_ub=1.5139
    )
    o, h1, h2 = list(water.atoms)
    water.def_angle(h1, o, h2, type="HW-OW-HW")


@pytest.mark.skipif(shutil.which("lmp") is None, reason="needs the lmp executable")
def test_lammps_prices_the_emitted_bonded_terms_as_molrs_does(
    tmp_path, water, water_ff
):
    """``run 0`` on the emitted deck: LAMMPS's bond and angle (UB included)
    energies are molrs's, so the styles read after ``read_data`` are in force."""
    _with_urey_bradley(water, water_ff)
    water_ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
    # The water carries no box: the deck as emitted, shrink-wrapped.
    _deck(water, water_ff, tmp_path, units="real")
    deck = (
        "include w.in.init\nread_data w.data\ninclude w.in.settings\n"
        "thermo_style custom step ebond eangle\n"
        "thermo_modify format float %.17g\nrun 0\n"
    )
    (tmp_path / "in.check").write_text(deck, encoding="utf-8")
    # A singleton run: inside a Slurm step, MPI must not join the step's PMI.
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith(("PMI", "PMIX", "SLURM", "OMPI"))
    }
    subprocess.run(
        ["lmp", "-in", "in.check", "-log", "log.check", "-screen", "none"],
        cwd=tmp_path,
        env=env,
        check=True,
        timeout=120,
    )
    lines = (tmp_path / "log.check").read_text(encoding="utf-8").splitlines()
    head = next(
        i for i, l in enumerate(lines) if l.split()[:3] == ["Step", "E_bond", "E_angle"]
    )
    _, ebond, eangle = (float(v) for v in lines[head + 1].split())

    # No `pairs` block: molrs prices the bonded terms alone.
    frame = water.to_frame()
    energy = mp.ff.compile.PotentialCompiler(water_ff).compile(frame).calc_energy(frame)
    assert ebond + eangle == pytest.approx(energy, rel=1e-5)
    assert eangle > 0.0
