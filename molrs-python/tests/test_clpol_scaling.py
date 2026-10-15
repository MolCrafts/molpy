import molrs
import pytest


def _forcefield():
    ff = molrs.ff.forcefield.ForceField()
    atom_style = ff.def_style("atom", "full")
    cr = atom_style.def_type("CR", type_=1.0)
    b = atom_style.def_type("B", type_=2.0)
    pair_style = ff.def_style("pair", "lj/cut")
    pair_style.def_type("CR-B", cr, b, epsilon=1.0, sigma=3.5)
    pair_style.def_type("CR", cr, epsilon=2.0, sigma=3.6)
    return ff


def _epsilon(ff, name):
    return ff.get_style("pair", "lj/cut").get_type_by_name(name)["epsilon"]


def test_native_scale_lj_clones_and_scales_cross_pair():
    ff = _forcefield()
    fragments = {
        "c2c1im": (["CR"], [(0.0, 0.0, 0.0)], [12.0]),
        "bf4": (["B"], [(4.0, 0.0, 0.0)], [11.0]),
    }
    output = molrs.ff.clpol_scaling.scale_lj(ff, fragments)
    expected = molrs.ff.clpol_scaling.compute_k_ij(
        molrs.ff.clpol_scaling.fragment_table()["c2c1im"],
        molrs.ff.clpol_scaling.fragment_table()["bf4"],
        4.0,
    )
    assert isinstance(output, molrs.ff.forcefield.ForceField)
    assert _epsilon(output, "CR-B") == pytest.approx(expected)
    assert _epsilon(output, "CR") == 2.0
    assert _epsilon(ff, "CR-B") == 1.0


def test_native_scale_lj_missing_data_is_key_error():
    fragments = {"missing": (["CR"], [(0.0, 0.0, 0.0)], [12.0])}
    with pytest.raises(KeyError, match="no scaling data"):
        molrs.ff.clpol_scaling.scale_lj(_forcefield(), fragments)


def test_clpol_polarizability_ships_alpha_ff():
    table = molrs.ff.params.clpol_polarizability()
    assert table["NBT"] == {
        "m_D": 0.4,
        "q_D_sign": -1.0,
        "k_D": 4184.0,
        "alpha": 1.698,
        "a_thole": 2.6,
    }
    assert table["HC"]["k_D"] == 0.0
    assert len(table) == 78


def test_read_clpol_alpha_reads_a_file(tmp_path):
    text = "# mine\nXX 0.4 -1.0 4184.0 2.0 2.6\nXX 0.4 -1.0 4184.0 3.0 2.6\n"
    path = tmp_path / "alpha.ff"
    path.write_text(text, encoding="utf-8")
    rows = [
        {"type_name": "XX", "m_D": 0.4, "q_D_sign": -1.0, "k_D": 4184.0, "alpha": a, "a_thole": 2.6}
        for a in (2.0, 3.0)
    ]
    assert molrs.io.read_clpol_alpha(path) == rows
    assert molrs.io.read_clpol_alpha(molrs.io.Text(text)) == rows
