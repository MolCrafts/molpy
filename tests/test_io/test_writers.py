"""``mp.io.write_lammps_bond_react_system``: the bond/react system writer's force-field coverage."""

from __future__ import annotations

from pathlib import Path

import molpy as mp
from molpy.core import Atomistic
from molpy.io.lammps_bond_react import BondReactTemplate


def _forcefield() -> mp.ff.forcefield.ForceField:
    """Two atom types and two bond types; ``c3-oh`` is used only by the template."""
    ff = mp.ff.forcefield.ForceField("hand")
    atoms = ff.def_style("atom", "full")
    c3 = atoms.def_type("c3", mass=12.011)
    oh = atoms.def_type("oh", mass=15.999)
    pairs = ff.def_style("pair", "lj/cut", {"cutoff": 9.0})
    pairs.def_type("c3", c3, epsilon=0.1, sigma=3.4)
    pairs.def_type("oh", oh, epsilon=0.2, sigma=3.0)
    bonds = ff.def_style("bond", "harmonic")
    bonds.def_type("c3-c3", c3, c3, k=600.0, r0=1.53)
    bonds.def_type("c3-oh", c3, oh, k=640.0, r0=1.41)
    return ff


def _system_frame() -> mp.Frame:
    """A c3-c3 dimer: the system itself never uses ``oh`` or ``c3-oh``."""
    mol = Atomistic()
    a = mol.def_atom(element="C", type="c3", x=0.0, y=0.0, z=0.0, charge=0.0, mol_id=1)
    b = mol.def_atom(element="C", type="c3", x=1.53, y=0.0, z=0.0, charge=0.0, mol_id=1)
    mol.def_bond(a, b, type="c3-c3")
    frame = mol.to_frame()
    frame.box = mp.Box.cube(20.0)
    return frame


def _template() -> BondReactTemplate:
    """c3 + oh → c3-oh: the new bond type exists only in the post template."""
    pre = Atomistic()
    c_pre = pre.def_atom(
        element="C", type="c3", x=0.0, y=0.0, z=0.0, charge=0.0, react_id=1
    )
    o_pre = pre.def_atom(
        element="O", type="oh", x=3.0, y=0.0, z=0.0, charge=0.0, react_id=2
    )
    post = Atomistic()
    c_post = post.def_atom(
        element="C", type="c3", x=0.0, y=0.0, z=0.0, charge=0.0, react_id=1
    )
    o_post = post.def_atom(
        element="O", type="oh", x=1.41, y=0.0, z=0.0, charge=0.0, react_id=2
    )
    post.def_bond(c_post, o_post, type="c3-oh")
    return BondReactTemplate(
        pre=pre,
        post=post,
        initiator_atoms=[c_pre, o_pre],
        edge_atoms=[],
        deleted_atoms=[],
    )


class TestWriteLammpsBondReactSystem:
    def test_ff_covers_template_only_labels(self, tmp_path: Path) -> None:
        workdir = tmp_path / "rxn"
        mp.io.write_lammps_bond_react_system(
            workdir, _system_frame(), _forcefield(), {"rxn1": _template()}
        )
        coeff_lines = [
            line.split()[:3]
            for line in (workdir / "rxn.ff").read_text().splitlines()
            if line.startswith(("pair_coeff", "bond_coeff"))
        ]
        assert ["bond_coeff", "c3-oh"] in [line[:2] for line in coeff_lines]
        assert ["pair_coeff", "oh", "oh"] in coeff_lines
