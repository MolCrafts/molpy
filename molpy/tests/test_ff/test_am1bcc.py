"""Am1BccModel with antechamber faked.

``subprocess.run`` is patched, as in ``test_ambertools.py``: antechamber
copies ``ac/ethanol_am1.ac`` into place. That file is what AmberTools 26's
``antechamber -c mul -at bcc -fo ac`` wrote for the ethanol built below
(SMILES ``CCO``, conformer seed 42).
"""

from __future__ import annotations


import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import molpy as mp

# The ac's charges: sqm's Mulliken charges, three printed decimals.
AM1 = [-0.254, -0.024, -0.327, 0.084, 0.088, 0.073, 0.104, 0.061, 0.195]


def _ethanol() -> mp.Atomistic:
    mol = mp.io.smiles.read_smiles("CCO").to_atomistic()
    return mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(mol)[0]


class _FakeAntechamber:
    """Writes ``ac`` at antechamber's ``-o``, or nothing when ``silent``."""

    def __init__(self, ac: Path, *, silent: bool = False) -> None:
        self.ac = ac
        self.silent = silent
        self.calls: list[list[str]] = []
        self.cwds: list[Path] = []

    def __call__(self, argv: list[str], *, cwd: str, **_: object):
        self.calls.append(argv)
        self.cwds.append(Path(cwd))
        if not self.silent:
            shutil.copy(self.ac, argv[argv.index("-o") + 1])
        return subprocess.CompletedProcess(argv, 0, "", "")


def _faked(fake: _FakeAntechamber):
    return (
        patch.object(
            mp.tool.Tool, "resolve_executable", return_value="/amber/bin/tool"
        ),
        patch("molpy.tool.process.subprocess.run", side_effect=fake),
    )


@pytest.fixture
def ac(TEST_DATA_DIR: Path) -> Path:
    return TEST_DATA_DIR / "ac" / "ethanol_am1.ac"


def test_the_qm_step_is_antechamber_mulliken(ac, tmp_path):
    fake = _FakeAntechamber(ac)
    first, second = _faked(fake)
    with first, second:
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).assign(
            _ethanol()
        )
    ((tool, *args),) = fake.calls
    assert tool == "antechamber"
    assert args[args.index("-c") + 1] == "mul"
    assert args[args.index("-fo") + 1] == "ac"
    assert args[args.index("-nc") + 1] == "0"


def test_am1_charges_are_the_ac_s_in_graph_order(ac, tmp_path):
    first, second = _faked(_FakeAntechamber(ac))
    with first, second:
        charges = mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(
            _ethanol()
        )
    assert charges.tolist() == pytest.approx(AM1, abs=1e-12)


@pytest.mark.parametrize("parameter_set", ["bcc", "abcg2"])
def test_the_qm_charges_are_corrected_natively(ac, tmp_path, parameter_set):
    mol = _ethanol()
    first, second = _faked(_FakeAntechamber(ac))
    with first, second:
        charges = mp.ff.charge.Am1BccModel(
            parameter_set=parameter_set, work_dir=tmp_path
        ).assign(mol)
    want = mp.ff.charge.BccModel(parameter_set=parameter_set).assign(
        mol, qm=np.array(AM1)
    )
    assert charges.tolist() == pytest.approx(want.tolist(), abs=1e-12)
    # Equivalenced: the three methyl hydrogens share one charge.
    assert charges[3] == pytest.approx(charges[4]) == pytest.approx(charges[5])
    assert all(atom.get("charge") is None for atom in mol.atoms)


def test_net_charge_is_the_sum_of_formal_charges(ac, tmp_path):
    mol = _ethanol()
    list(mol.atoms)[2]["formal_charge"] = -1
    fake = _FakeAntechamber(ac)
    first, second = _faked(fake)
    with first, second:
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(mol)
    args = fake.calls[0]
    assert args[args.index("-nc") + 1] == "-1"


def test_an_ac_in_another_atom_order_raises(TEST_DATA_DIR, tmp_path):
    mol = mp.Atomistic()
    for index, element in enumerate(["O", "C", "C", *["H"] * 6]):
        mol.add_atom(
            atomic_number=mp.Element.by_symbol(element).z, x=float(index), y=0.0, z=0.0
        )
    first, second = _faked(_FakeAntechamber(TEST_DATA_DIR / "ac" / "ethanol_am1.ac"))
    with first, second, pytest.raises(ValueError, match="reordered or dropped"):
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(mol)


def test_a_rerun_never_reads_the_old_ac(ac, tmp_path):
    first, second = _faked(_FakeAntechamber(ac))
    with first, second:
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(_ethanol())
    first, second = _faked(_FakeAntechamber(ac, silent=True))
    with (
        first,
        second,
        pytest.raises(mp.tool.ToolError, match="left sqm.ac unwritten"),
    ):
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(_ethanol())


def test_the_net_charge_keys_the_directory(ac, tmp_path):
    neutral = _FakeAntechamber(ac)
    first, second = _faked(neutral)
    with first, second:
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(_ethanol())
    anion = _ethanol()
    list(anion.atoms)[2]["formal_charge"] = -1
    charged = _FakeAntechamber(ac)
    first, second = _faked(charged)
    with first, second:
        mp.ff.charge.Am1BccModel(parameter_set="bcc", work_dir=tmp_path).am1(anion)
    assert set(neutral.cwds).isdisjoint(charged.cwds)


def test_parameter_set_is_required_and_checked():
    with pytest.raises(TypeError):
        mp.ff.charge.Am1BccModel()  # type: ignore[call-arg]
    with pytest.raises(ValueError):
        mp.ff.charge.Am1BccModel(parameter_set="gaff2")  # type: ignore[arg-type]
    assert mp.ff.charge.Am1BccModel(parameter_set="abcg2").parameter_set == "abcg2"
