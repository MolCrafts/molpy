"""Generate LAMMPS fix bond/react templates from a reaction.

Minimal end-to-end path mirroring docs/user-guide/04_crosslinking.ipynb:
run a :class:`BondReactReacter` C-C coupling on two small fragments and
export the complete reactive system (data + force field + pre/post
templates + map) with :func:`molpy.io.write_lammps_bond_react_system`.

Runs without RDKit (coordinates are set explicitly).
"""

from pathlib import Path

import molpy as mp
from molpy.core.atomistic import Atomistic
from molpy.reacter import (
    BondReactReacter,
    find_port,
    form_single_bond,
    select_one_hydrogen,
    select_port,
)


def ethane_fragment(x0: float, port: str) -> Atomistic:
    """Two carbons with explicit hydrogens; the port sits on one carbon."""
    struct = Atomistic()
    c1 = struct.def_atom(element="C", type="c3", x=x0, y=0.0, z=0.0, charge=0.0)
    c2 = struct.def_atom(element="C", type="c3", x=x0 + 1.54, y=0.0, z=0.0, charge=0.0)
    hydrogens = [
        (c1, x0 - 0.5, 0.9, 0.0),
        (c1, x0 - 0.5, -0.45, 0.78),
        (c1, x0 - 0.5, -0.45, -0.78),
        (c2, x0 + 2.04, 0.9, 0.0),
        (c2, x0 + 2.04, -0.45, 0.78),
        (c2, x0 + 2.04, -0.45, -0.78),
    ]
    for carbon, x, y, z in hydrogens:
        hydrogen = struct.def_atom(element="H", type="hc", x=x, y=y, z=z, charge=0.0)
        struct.def_bond(carbon, hydrogen)
    struct.def_bond(c1, c2)
    c2["port"] = port
    return struct


def main() -> None:
    left = ethane_fragment(0.0, ">")
    right = ethane_fragment(4.0, "<")

    reacter = BondReactReacter(
        name="cc_coupling",
        anchor_selector_left=select_port,
        anchor_selector_right=select_port,
        leaving_selector_left=select_one_hydrogen,
        leaving_selector_right=select_one_hydrogen,
        bond_former=form_single_bond,
        radius=2,
    )
    result = reacter.run(
        left,
        right,
        port_atom_L=find_port(left, ">"),
        port_atom_R=find_port(right, "<"),
        compute_topology=True,
    )
    template = result.template
    assert template is not None

    # Type the topology so the writers can build unified type tables.
    for struct in (template.pre, template.post, result.product):
        for atom in struct.atoms:
            atom.data.pop("port", None)
        for link_view in (struct.bonds, struct.angles, struct.dihedrals):
            for link in link_view:
                names = tuple(str(ep["type"]) for ep in link.endpoints)
                link["type"] = "-".join(min(names, names[::-1]))

    for index, atom in enumerate(result.product.atoms, start=1):
        atom["id"] = index
        atom["mol_id"] = 1
    frame = result.product.to_frame()
    frame.box = mp.Box.cubic(30.0)

    forcefield = mp.ForceField("cc_coupling_demo", units="real")
    atom_style = forcefield.def_style(mp.AtomStyle("full"))
    c3 = atom_style.def_type("c3", mass=12.011)
    hc = atom_style.def_type("hc", mass=1.008)
    types = {"c3": c3, "hc": hc}

    def collect(kind: str) -> set[str]:
        names: set[str] = set()
        for struct in (template.pre, template.post, result.product):
            for link in getattr(struct, kind):
                names.add(str(link["type"]))
        return names

    bond_style = forcefield.def_style(mp.BondStyle("harmonic"))
    for name in sorted(collect("bonds")):
        i, j = name.split("-")
        bond_style.def_type(types[i], types[j], name=name, k=300.0, r0=1.53)
    angle_style = forcefield.def_style(mp.AngleStyle("harmonic"))
    for name in sorted(collect("angles")):
        i, j, k = name.split("-")
        angle_style.def_type(
            types[i], types[j], types[k], name=name, k=50.0, theta0=110.0
        )
    dihedral_style = forcefield.def_style(mp.DihedralStyle("opls"))
    for name in sorted(collect("dihedrals")):
        endpoint_types = [types[part] for part in name.split("-")]
        dihedral_style.def_type(
            *endpoint_types, name=name, k1=0.5, k2=1.0, k3=0.0, k4=0.0
        )

    workdir = Path("bond_react_output")
    mp.io.write_lammps_bond_react_system(
        workdir, frame, forcefield, templates={"rxn1": template}
    )
    produced = sorted(p.name for p in workdir.iterdir())
    print(f"wrote {len(produced)} files to {workdir}/: {', '.join(produced)}")


if __name__ == "__main__":
    main()
