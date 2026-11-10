"""``read_bigsmiles`` across the seam: a string becomes the polymer model of
``molpy.builder``, and refusals cross as ``SmilesError`` stamped
``"bigsmiles"``.

The grammar and the lowering are pinned by the Rust fragment tests in
``crates/molrs/src/io/bigsmiles/``. Fixtures are inline strings.
"""

from __future__ import annotations

import molpy as mp
import pytest

PE = "{[][<]CC[>];[<][H],[>][H][]}|poisson(300)|.|3000|"


def test_a_system_string_samples_to_its_amount() -> None:
    ir = mp.io.bigsmiles.read_bigsmiles(PE)
    assert ir.n_molecules == 1
    system = ir.to_system()
    assert isinstance(system, mp.builder.System)
    assert system.measure == "heavy_atom_mass"
    sample = system.sample(7)
    assert sum(m.heavy_atom_mass for m in sample.molecules) >= 3000.0


def test_units_are_named_by_position() -> None:
    (mol,) = mp.io.bigsmiles.read_bigsmiles(
        "[H]{[>][<]CC[>];[<][H][]}|gauss(100, 0)|"
    ).to_molecules()
    assert set(mol.library()) == {"0.0", "0.1.r0", "0.1.e0"}


def test_a_parse_refusal_is_a_bigsmiles_error() -> None:
    with pytest.raises(mp.io.smiles.SmilesError) as excinfo:
        mp.io.bigsmiles.read_bigsmiles("{[][<]CC[>][]}|zipf(2)|")
    assert excinfo.value.kind == "BigUnknownDistribution"
    assert excinfo.value.notation == "bigsmiles"


def test_a_missing_amount_is_a_build_error() -> None:
    ir = mp.io.bigsmiles.read_bigsmiles("{[][<]CC[>];[<]C,[>]C[]}|poisson(100)|")
    with pytest.raises(mp.io.smiles.SmilesError) as excinfo:
        ir.to_system()
    assert excinfo.value.kind == "BigBuild"
