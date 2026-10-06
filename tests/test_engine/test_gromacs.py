"""``GROMACSEngine.generate_inputs``: gro + top from the force field, plus the
two mdp files."""

from molpy.engine import GROMACSEngine


def test_writes_coordinates_topology_and_mdps(tmp_path, water, tip3p):
    engine = GROMACSEngine(prefix="w", check_executable=False)
    paths = engine.generate_inputs(water.to_frame(), tip3p, tmp_path, temperature=280.0)
    assert {key: p.name for key, p in paths.items()} == {
        "gro": "w.gro",
        "top": "w.top",
        "em": "em.mdp",
        "nvt": "nvt.mdp",
    }
    assert all(p.exists() for p in paths.values())
    assert paths["gro"].read_text().splitlines()[1].strip() == "3"
    assert "[ defaults ]" in paths["top"].read_text()
    assert "integrator      = steep" in paths["em"].read_text()
    assert "ref_t           = 280.0" in paths["nvt"].read_text()


def test_run_grompps_then_mdruns_the_input_mdp(tmp_path, water, tip3p, monkeypatch):
    import subprocess

    from molpy.engine import Script

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    engine = GROMACSEngine(
        "gmx", check_executable=False, launcher=["mpirun", "-np", "4"]
    )
    paths = engine.generate_inputs(water.to_frame(), tip3p, tmp_path)
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
