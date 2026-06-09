"""The high-level wrapper methods forward ``check=`` down to ``subprocess.run``.

This lets callers opt into "raise on non-zero exit" without hand-rolling a
``returncode`` check, by leaning on molpy's existing ``Wrapper.run(check=...)``.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from molpy.wrapper import (
    AntechamberWrapper,
    Parmchk2Wrapper,
    PrepgenWrapper,
    TLeapWrapper,
)


def _call(wrapper_factory, invoke, tmp_path: Path, *, check):
    """Invoke a wrapper method under a patched subprocess and return run kwargs."""
    wrapper = wrapper_factory(tmp_path / "work")
    with patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        mock_run.return_value.stdout = ""
        mock_run.return_value.stderr = ""
        invoke(wrapper, check)
        mock_run.assert_called_once()
        return mock_run.call_args.kwargs


@pytest.mark.parametrize("check", [True, False])
def test_antechamber_forwards_check(tmp_path: Path, check: bool):
    kwargs = _call(
        lambda wd: AntechamberWrapper(name="ante", workdir=wd),
        lambda w, c: w.atomtype_assign("in.pdb", "out.mol2", check=c),
        tmp_path,
        check=check,
    )
    assert kwargs["check"] is check


@pytest.mark.parametrize("check", [True, False])
def test_parmchk2_forwards_check(tmp_path: Path, check: bool):
    kwargs = _call(
        lambda wd: Parmchk2Wrapper(name="parmchk2", workdir=wd),
        lambda w, c: w.generate_parameters("in.ac", "out.frcmod", check=c),
        tmp_path,
        check=check,
    )
    assert kwargs["check"] is check


@pytest.mark.parametrize("check", [True, False])
def test_prepgen_forwards_check(tmp_path: Path, check: bool):
    kwargs = _call(
        lambda wd: PrepgenWrapper(name="prepgen", workdir=wd),
        lambda w, c: w.generate_residue(
            "in.ac", "out.prepi", "ctrl.chain", "MOL", check=c
        ),
        tmp_path,
        check=check,
    )
    assert kwargs["check"] is check


@pytest.mark.parametrize("check", [True, False])
def test_tleap_forwards_check(tmp_path: Path, check: bool):
    kwargs = _call(
        lambda wd: TLeapWrapper(name="tleap", workdir=wd),
        lambda w, c: w.run_from_script("quit", check=c),
        tmp_path,
        check=check,
    )
    assert kwargs["check"] is check
