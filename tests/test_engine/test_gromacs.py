"""``GromacsEngine.generate_inputs``: gro + the whole topology, plus the two
mdp files, and ``grompp`` accepting them when GROMACS is installed."""

import shutil
import subprocess

import pytest

import molpy as mp
from molpy.engine import GromacsEngine

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
    water.def_angle(h1, o, h2, type="HW-OW-HW")
    frame = water.to_frame()
    frame.box = mp.Box.cube(30.0)
    return frame, tip3p


def test_writes_coordinates_topology_and_mdps(tmp_path, system):
    frame, ff = system
    engine = GromacsEngine(prefix="w", check_executable=False)
    paths = engine.generate_inputs(frame, ff, tmp_path, temperature=280.0)
    assert {key: p.name for key, p in paths.items()} == {
        "gro": "w.gro",
        "top": "w.top",
        "em": "em.mdp",
        "nvt": "nvt.mdp",
    }
    assert all(p.exists() for p in paths.values())
    assert paths["gro"].read_text().splitlines()[1].strip() == "3"
    top = paths["top"].read_text()
    # A whole topology: directives, the molecule, and the system's molecules.
    for directive in ("[ defaults ]", "[ moleculetype ]", "[ atoms ]", "[ angles ]"):
        assert directive in top
    assert "[ system ]" in top
    assert "[ molecules ]" in top
    assert "integrator      = steep" in paths["em"].read_text()
    assert "ref_t           = 280.0" in paths["nvt"].read_text()


def test_run_grompps_then_mdruns_the_input_mdp(tmp_path, system, monkeypatch):
    import subprocess

    from molpy.engine import Script

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = GromacsEngine(
        "gmx", check_executable=False, launcher=["mpirun", "-np", "4"]
    )
    paths = engine.generate_inputs(*system, tmp_path)
    engine.run(Script.from_path(paths["em"]), workdir=tmp_path)
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


def test_grompp_accepts_the_generated_inputs(tmp_path, system):
    paths = GromacsEngine(check_executable=False).generate_inputs(*system, tmp_path)
    top = paths["top"].read_text()
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
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "em.tpr").is_file()
