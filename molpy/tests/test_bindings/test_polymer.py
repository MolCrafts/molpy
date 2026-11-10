"""The polymer model of ``molpy.builder`` across the seam.

Growth, sizes and system amounts are pinned by the Rust fragment tests in
``crates/molrs/src/builder/polymer/``; these tests prove that Python
constructs the model, samples it, gets its records and site graph back, and
sees refusals as ``ValueError``. Templates are read from inline fragment
bodies.
"""

from __future__ import annotations

import molpy as mp
import pytest

B = mp.builder


def fragment(name: str, body: str) -> mp.builder.Fragment:
    return B.Fragment(name, mp.io.smiles.read_smiles_fragment(body).to_template())


def chain(n: float) -> mp.builder.Molecule:
    """CH3-(CH2CH2)n-CH3 with exactly ``n`` repeat units."""
    obj = B.StochasticObject(
        [fragment("E", "[<]CC[>]")],
        [fragment("<Me", "[<]C")],
        left=B.Terminal(">"),
        size=B.ChainSize(B.Distribution.Gauss(mu=n, sigma=0.0), "repeat_units"),
    )
    return B.Molecule([fragment("Me>", "C[>]"), obj])


def test_distribution_variants_keep_their_parameters() -> None:
    d = B.Distribution.SchulzZimm(mw=1500.0, mn=1000.0)
    assert isinstance(d, B.Distribution)
    assert (d.mw, d.mn) == (1500.0, 1000.0)
    size = B.ChainSize(d, "heavy_atom_mass")
    assert size.distribution == d
    assert size.measure == "heavy_atom_mass"


def test_out_of_range_parameters_are_a_value_error() -> None:
    with pytest.raises(B.PolymerError, match="parameter") as excinfo:
        B.ChainSize(B.Distribution.FlorySchulz(a=1.5), "repeat_units")
    assert excinfo.value.kind == "InvalidParameter"
    # A mean no sampler represents is a refusal, not a panic.
    with pytest.raises(B.PolymerError):
        B.ChainSize(B.Distribution.Poisson(n=1e30), "repeat_units")
    with pytest.raises(ValueError, match="measure"):
        B.ChainSize(B.Distribution.Poisson(n=3.0), "monomers")


def test_unit_reports_its_masses_and_ports() -> None:
    e = fragment("E", "[<]CC[>]")
    assert e.name == "E"
    assert e.n_ports == 2
    assert e.heavy_atom_mass == pytest.approx(2 * 12.011, abs=1e-2)
    assert e.mass > e.heavy_atom_mass
    assert e.template.n_ports == 2


def test_unit_weights_cross_as_floats_and_lists() -> None:
    template = mp.io.smiles.read_smiles_fragment("[<]CC[>]").to_template()
    B.Fragment("E", template, [0.5, [1.0, 0.0]])
    with pytest.raises(ValueError, match="2 ports"):
        B.Fragment("E", template, [1.0])


def test_molecule_samples_a_site_graph_named_by_unit() -> None:
    mol = chain(4)
    sites = mol.sample(1)
    assert isinstance(sites, mp.core.CoarseGrain)
    assert sites.n_beads == 6
    assert sites.n_bonds == 5
    assert set(mol.library()) == {"Me>", "E", "<Me"}
    assert all(isinstance(t, mp.core.Atomistic) for t in mol.library().values())
    parts = mol.parts
    assert isinstance(parts[0], B.Fragment)
    assert isinstance(parts[1], B.StochasticObject)


def test_system_sample_records_each_molecule() -> None:
    system = B.System(
        [(chain(4), B.Amount.Total(value=1000.0))], measure="heavy_atom_mass"
    )
    sample = system.sample(3)
    assert sum(m.heavy_atom_mass for m in sample.molecules) >= 1000.0
    assert sum(m.n_fragments for m in sample.molecules) == sample.sites.n_beads
    assert all(m.n_repeat_units == 4 for m in sample.molecules)
    assert system.targets() == [1000.0]


def test_percent_amounts_are_shares_of_the_total() -> None:
    mol = chain(4)
    system = B.System(
        [(mol, B.Amount.Total(value=800.0)), (mol, B.Amount.Percent(percent=20.0))],
        measure="heavy_atom_mass",
    )
    assert system.targets() == pytest.approx([800.0, 200.0])


def test_a_stuck_growth_is_a_polymer_error_of_kind_stuck() -> None:
    obj = B.StochasticObject(
        [fragment("E", "[<]CC[>]")],
        [fragment("Me>", "C[>]")],
        size=B.ChainSize(B.Distribution.Gauss(mu=2.0, sigma=0.0), "repeat_units"),
    )
    with pytest.raises(B.PolymerError, match="stuck") as excinfo:
        B.Molecule([obj]).sample(0)
    assert excinfo.value.kind == "Stuck"
    assert isinstance(excinfo.value, ValueError)


def test_an_invalid_model_is_refused_when_built() -> None:
    # Nothing before the object and no end group to start from.
    obj = B.StochasticObject([fragment("E", "[$]CC[$]")])
    with pytest.raises(B.PolymerError) as excinfo:
        B.Molecule([obj])
    assert excinfo.value.kind == "InvalidMolecule"


def test_terminal_inputs_are_checked() -> None:
    assert B.Terminal(">", order=2).order == 2
    for bad in ({"kind": ">", "order": 7}, {"kind": "!"}, {"kind": ">", "weights": []}):
        with pytest.raises(B.PolymerError) as excinfo:
            B.Terminal(**bad)
        assert excinfo.value.kind == "InvalidObject"
