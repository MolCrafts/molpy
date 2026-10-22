"""``GromacsEngine.generate_inputs``: gro + the whole topology, plus the two
mdp files, and ``grompp`` accepting them when GROMACS is installed."""

import shutil
import subprocess

import pytest

import molpy as mp

#: The GROMACS driver to check the written inputs with, if one is installed.
GMX = shutil.which("gmx") or shutil.which("gmx_d")


@pytest.fixture
def system(water, tip3p):
    """The water with its H-O-H angle typed: a whole topology for GROMACS.

    GROMACS excludes every pair within three bonds (nrexcl 3); the pair list
    the topology writer checks that against excludes 1-3 pairs by the frame's
    ``angles`` rows, so a typed system carries them.
    """
    o, h1, h2 = list(water.atoms)
    (ow,) = [t for t in tip3p.get_styles("atom")[0].get_types() if t.name == "OW"]
    (hw,) = [t for t in tip3p.get_styles("atom")[0].get_types() if t.name == "HW"]
    tip3p.def_style("angle", "harmonic").def_type(
        "HW-OW-HW", hw, ow, hw, k=55.0, theta0=104.52
    )
    water.add_angle(h1, o, h2, type="HW-OW-HW")
    frame = water.to_frame()
    frame.box = mp.Box(mp.core.Cell.lengths([30.0] * 3))
    return frame, tip3p


def test_writes_coordinates_topology_and_mdps(tmp_path, system):
    frame, ff = system
    engine = mp.engine.GromacsEngine(prefix="w", check_executable=False)
    paths = engine.generate_inputs(frame, ff, tmp_path, temperature=280.0)
    assert {key: p.name for key, p in paths.items()} == {
        "input": "em.mdp",
        "gro": "w.gro",
        "top": "w.top",
        "nvt": "nvt.mdp",
    }
    assert all(p.exists() for p in paths.values())
    assert paths["gro"].read_text(encoding="utf-8").splitlines()[1].strip() == "3"
    top = paths["top"].read_text(encoding="utf-8")
    # A whole topology: directives, the molecule, and the system's molecules.
    for directive in ("[ defaults ]", "[ moleculetype ]", "[ atoms ]", "[ angles ]"):
        assert directive in top
    assert "[ system ]" in top
    assert "[ molecules ]" in top
    assert "integrator      = steep" in paths["input"].read_text(encoding="utf-8")
    assert "ref_t           = 280.0" in paths["nvt"].read_text(encoding="utf-8")


def _mdp(path) -> dict[str, str]:
    """The ``key = value`` settings of an ``.mdp`` file."""
    return {
        key.strip(): value.strip()
        for key, _, value in (
            line.partition("=")
            for line in path.read_text(encoding="utf-8").splitlines()
            if "=" in line and not line.startswith(";")
        )
    }


def _nonbonded(mdp: dict[str, str]) -> dict[str, str]:
    return {k: mdp[k] for k in ("coulombtype", "rcoulomb", "rvdw", "rlist")}


def test_mdp_nonbonded_settings_are_the_force_fields(tmp_path, system):
    """A 10 Å lj/cut + 9 Å coul/cut force field runs cut-off Coulomb at
    0.9 nm and van der Waals at 1.0 nm, in both stages."""
    frame, ff = system
    ff.get_style("pair", "lj/cut")["cutoff"] = 10.0
    ff.def_style("pair", "coul/cut", {"cutoff": 9.0})
    paths = mp.engine.GromacsEngine(check_executable=False).generate_inputs(
        frame, ff, tmp_path
    )
    for stage in ("input", "nvt"):
        assert _nonbonded(_mdp(paths[stage])) == {
            "coulombtype": "Cut-off",
            "rcoulomb": "0.9",
            "rvdw": "1",
            "rlist": "1",
        }


def test_a_single_stated_cutoff_is_both(tmp_path, system):
    frame, ff = system
    ff.get_style("pair", "lj/cut")["cutoff"] = 11.0
    ff.def_style("pair", "coul/cut")
    paths = mp.engine.GromacsEngine(check_executable=False).generate_inputs(
        frame, ff, tmp_path
    )
    assert _nonbonded(_mdp(paths["input"])) == {
        "coulombtype": "Cut-off",
        "rcoulomb": "1.1",
        "rvdw": "1.1",
        "rlist": "1.1",
    }


def test_unstated_settings_fall_back_and_overrides_win(tmp_path, system):
    frame, ff = system  # lj/cut with no cutoff, no Coulomb style
    engine = mp.engine.GromacsEngine(check_executable=False)
    paths = engine.generate_inputs(frame, ff, tmp_path / "default")
    assert _nonbonded(_mdp(paths["input"])) == {
        "coulombtype": "PME",
        "rcoulomb": "1.2",
        "rvdw": "1.2",
        "rlist": "1.2",
    }
    paths = engine.generate_inputs(
        frame, ff, tmp_path / "override", coulombtype="Reaction-Field", rvdw=1.0
    )
    assert _nonbonded(_mdp(paths["nvt"])) == {
        "coulombtype": "Reaction-Field",
        "rcoulomb": "1.2",
        "rvdw": "1",
        "rlist": "1.2",
    }


def test_a_box_free_frame_is_refused_before_any_file(tmp_path, system):
    """GROMACS's Verlet scheme is periodic only: no zero box with pbc = xyz."""
    frame, ff = system
    frame.box = None
    with pytest.raises(ValueError, match="GROMACS.*needs a periodic box"):
        mp.engine.GromacsEngine(check_executable=False).generate_inputs(
            frame, ff, tmp_path / "out"
        )
    assert not (tmp_path / "out").exists()


def test_run_grompps_then_mdruns_the_input_mdp(tmp_path, system, monkeypatch):
    import subprocess

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    config = mp.config.load_config(
        {"engine": {"gromacs": {"launcher": ["mpirun", "-np", "4"]}}}
    )
    engine = mp.engine.GromacsEngine(config=config, check_executable=False)
    paths = engine.generate_inputs(*system, tmp_path)
    engine.run(mp.engine.Script.from_path(paths["input"]), workdir=tmp_path)
    assert calls == [
        [
            "gmx",
            "grompp",
            "-f",
            "em.mdp",
            "-c",
            "system.gro",
            "-p",
            "system.top",
            "-o",
            "em.tpr",
        ],
        ["mpirun", "-np", "4", "gmx", "mdrun", "-deffnm", "em"],
    ]


def test_a_later_stage_starts_from_the_previous_stages_output(
    tmp_path, system, monkeypatch
):
    """NVT after EM grompp's the minimised em.gro, not the initial gro."""

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = mp.engine.GromacsEngine(check_executable=False)
    paths = engine.generate_inputs(*system, tmp_path)
    engine.run(mp.engine.Script.from_path(paths["input"]), workdir=tmp_path)
    engine.run(
        mp.engine.Script.from_path(paths["nvt"]), workdir=tmp_path, structure="em.gro"
    )
    grompps = [cmd for cmd in calls if cmd[1] == "grompp"]
    assert [cmd[cmd.index("-c") + 1] for cmd in grompps] == ["system.gro", "em.gro"]
    assert [cmd[cmd.index("-o") + 1] for cmd in grompps] == ["em.tpr", "nvt.tpr"]
    assert calls[-1] == ["gmx", "mdrun", "-deffnm", "nvt"]


def test_grompp_accepts_the_generated_inputs(tmp_path, system):
    paths = mp.engine.GromacsEngine(check_executable=False).generate_inputs(
        *system, tmp_path
    )
    top = paths["top"].read_text(encoding="utf-8")
    assert "[ moleculetype ]" in top
    assert "[ molecules ]" in top
    if GMX is None:
        return
    done = subprocess.run(
        [
            GMX,
            "grompp",
            "-f",
            "em.mdp",
            "-c",
            "system.gro",
            "-p",
            "system.top",
            "-o",
            "em.tpr",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "em.tpr").is_file()
