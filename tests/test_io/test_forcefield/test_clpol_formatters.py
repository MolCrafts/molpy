"""LAMMPS parameter formatters for the CL&Pol damping pair potentials.

Each potential serializes through the formatter registry on
``LammpsForceFieldFormatter`` (dispatched by the style's ``(category, name)``),
never a bespoke writer.
"""

import molpy as mp
from molpy.core.forcefield import (
    PairCoulTTStyle,
    PairStyle,
    PairTholeStyle,
    PairType,
)
from molpy.io.forcefield.lammps import LammpsForceFieldFormatter


def _format(ff):
    fmt = LammpsForceFieldFormatter()
    typ = next(iter(ff.get_types(PairType)))
    style = ff.get_styles(PairStyle)[0]
    return fmt.format_params(typ, style)


def test_thole_formatter_emits_alpha_and_a_thole():
    ff = mp.ForceField("thole")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=1.0, alpha=1.5, a_thole=2.6)
    # LAMMPS pair_style thole coefficient is the polarizability + Thole width.
    assert _format(ff) == [1.5, 2.6]


def test_thole_formatter_defaults_a_thole():
    ff = mp.ForceField("thole2")
    pstyle = ff.def_style(PairTholeStyle())
    pstyle.def_type("A", charge=1.0, alpha=0.625)
    assert _format(ff) == [0.625, 2.6]


def test_coul_tt_formatter_emits_b_n_c_defaults():
    ff = mp.ForceField("tt")
    pstyle = ff.def_style(PairCoulTTStyle())
    pstyle.def_type("A", charge=1.0)
    # Tang-Toennies damping defaults (b=4.5 1/A, n=4, c=1.0).
    assert _format(ff) == [4.5, 4, 1.0]


def test_formatters_registered_on_class():
    formatters = LammpsForceFieldFormatter._param_formatters
    assert PairTholeStyle in formatters
    assert PairCoulTTStyle in formatters
