"""LAMMPS actually runs what molpy writes for ``fix bond/react``.

The unit tests in ``test_lammps_bond_react.py`` check that the ``.map`` and
molecule templates *look* right. This one hands them to the real consumer: a
LAMMPS binary with the REACTION package. It is the only test in the suite that
can tell us the files are correct rather than merely well-formed, and it found
two bugs the first time it ran:

* the ``Molecules`` section of a molecule template was written as ``1.000000``;
  LAMMPS parses molecule IDs as integers.
* a post-reaction template must not carry angles/dihedrals through the atoms the
  reaction deletes, and it must extend far enough that every term the edit
  creates lies strictly inside it ("Atom affected by reaction is too close to
  template edge") — the same two-radii argument
  :meth:`~molpy.typifier.affected_region.AffectedRegion.around` makes.

Reference: https://docs.lammps.org/fix_bond_react.html and the LAMMPS
``examples/PACKAGES/reaction`` inputs.
"""

from __future__ import annotations

import math
import shutil
import subprocess

import pytest

import molpy as mp
from molpy.core.atomistic import Atomistic
from molpy.io.data.lammps_bond_react import BondReactTemplate
from molpy.typifier.affected_region import AffectedRegion

pytestmark = pytest.mark.external

LMP = shutil.which("lmp_serial") or shutil.which("lmp")

#: bonds must be at their equilibrium length or the run explodes before reacting
R0 = {"c3-c3": 1.53, "c3-hc": 1.09}
#: how far the template reaches around the initiators. A dihedral created by the
#: new bond spans 2 bonds from it, so the template edge must be further out.
TEMPLATE_RADIUS = 3


def _pentane(x0: float, port_label: str, port_on_last_carbon: bool) -> Atomistic:
    """C5H12 laid along x, with one terminal carbon marked as the reactive port."""
    s = mp.Atomistic()
    carbons = [
        s.def_atom(
            element="C", type="c3", x=x0 + 1.54 * i, y=0.0, z=0.0, charge=0.0, mol_id=1
        )
        for i in range(5)
    ]
    for a, b in zip(carbons, carbons[1:], strict=False):
        s.def_bond(a, b)
    for i, carbon in enumerate(carbons):
        n_h = 3 if i in (0, 4) else 2
        for j in range(n_h):
            angle = 2.0 * math.pi * j / n_h
            s.def_bond(
                carbon,
                s.def_atom(
                    element="H",
                    type="hc",
                    x=float(carbon["x"]),
                    y=1.09 * math.cos(angle),
                    z=1.09 * math.sin(angle),
                    charge=0.0,
                    mol_id=1,
                ),
            )
    (carbons[4] if port_on_last_carbon else carbons[0])["port"] = port_label
    return s


def _port_atom(struct: Atomistic, label: str):
    return next(a for a in struct.atoms if a.get("port") == label)


def _one_hydrogen(struct: Atomistic, anchor):
    hydrogens = [n for n in struct.get_neighbors(anchor) if n.get("element") == "H"]
    return min(hydrogens, key=lambda a: a.handle)


def _link_type(link) -> str:
    names = tuple(str(ep["type"]) for ep in link.endpoints)
    return "-".join(min(names, names[::-1]))


def _assign_link_types(struct: Atomistic) -> None:
    for views in (struct.bonds, struct.angles, struct.dihedrals):
        for link in views:
            link["type"] = _link_type(link)


def _strip_ports(struct: Atomistic) -> None:
    for atom in struct.atoms:
        atom.data.pop("port", None)


def _build() -> tuple[Atomistic, BondReactTemplate]:
    """The unreacted two-pentane system, and the C-C coupling template for it."""
    world = _pentane(0.0, ">", port_on_last_carbon=True)
    world.merge(_pentane(9.08, "<", port_on_last_carbon=False))
    for react_id, atom in enumerate(world.atoms, start=1):
        atom["react_id"] = react_id

    anchor_l, anchor_r = _port_atom(world, ">"), _port_atom(world, "<")
    leaving = [_one_hydrogen(world, anchor_l), _one_hydrogen(world, anchor_r)]

    pre = AffectedRegion._from(
        world,
        [anchor_l, anchor_r],
        extract_radius=TEMPLATE_RADIUS,
        interior_reach=TEMPLATE_RADIUS,
    )
    by_rid = {a["react_id"]: a for a in pre.atoms}

    post = pre.copy()
    post_by_rid = {a["react_id"]: a for a in post.atoms}
    for hydrogen in leaving:
        target = post_by_rid[hydrogen["react_id"]]
        for bond in list(post.bonds):
            if target in bond.endpoints:
                post.del_bond(bond)
    post.def_bond(post_by_rid[anchor_l["react_id"]], post_by_rid[anchor_r["react_id"]])
    # a deleted atom must survive in the post template, bonded to nothing
    post.generate_topology(gen_angle=True, gen_dihedral=True, clear_existing=True)

    template = BondReactTemplate(
        pre=pre,
        post=post,
        initiator_atoms=[by_rid[anchor_l["react_id"]], by_rid[anchor_r["react_id"]]],
        edge_atoms=list(pre.boundary),
        deleted_atoms=[by_rid[h["react_id"]] for h in leaving],
        pre_react_id_to_atom=by_rid,
        post_react_id_to_atom=post_by_rid,
    )

    world.generate_topology(gen_angle=True, gen_dihedral=True)
    for struct in (world, template.pre, template.post):
        _strip_ports(struct)
        _assign_link_types(struct)
    return world, template


def _forcefield(*structs: Atomistic) -> mp.ForceField:
    bonds, angles, dihedrals = set(), set(), set()
    for struct in structs:
        bonds.update(str(b["type"]) for b in struct.bonds)
        angles.update(str(a["type"]) for a in struct.angles)
        dihedrals.update(str(d["type"]) for d in struct.dihedrals)

    ff = mp.ForceField("bond_react_run", units="real")
    atom_style = ff.def_style(mp.AtomStyle(name="full"))
    types = {
        "c3": atom_style.def_type("c3", mass=12.011),
        "hc": atom_style.def_type("hc", mass=1.008),
    }
    bond_style = ff.def_style(mp.BondStyle(name="harmonic"))
    for name in sorted(bonds):
        i, j = name.split("-")
        bond_style.def_type(types[i], types[j], name=name, k=340.0, r0=R0[name])
    angle_style = ff.def_style(mp.AngleStyle(name="harmonic"))
    for name in sorted(angles):
        i, j, k = name.split("-")
        angle_style.def_type(
            types[i], types[j], types[k], name=name, k=50.0, theta0=110.0
        )
    dihedral_style = ff.def_style(mp.DihedralStyle(name="opls"))
    for name in sorted(dihedrals):
        endpoints = [types[p] for p in name.split("-")]
        dihedral_style.def_type(*endpoints, name=name, k1=0.0, k2=0.0, k3=0.3, k4=0.0)
    return ff


INPUT = """units real
boundary p p p
atom_style full
pair_style lj/cut 8.5
bond_style harmonic
angle_style harmonic
dihedral_style opls
special_bonds lj/coul 0 0 0.5
read_data rxn.data extra/bond/per/atom 5 extra/angle/per/atom 20 &
  extra/dihedral/per/atom 30 extra/special/per/atom 30
include rxn.ff
pair_coeff * * 0.066 3.5
velocity all create 300.0 4928459 dist gaussian
molecule mol1 rxn1_pre.mol
molecule mol2 rxn1_post.mol
fix myrxns all bond/react stabilization no &
  react rxn1 all 1 0.0 5.0 mol1 mol2 rxn1.map
fix 1 all nve/limit 0.02
timestep 0.1
thermo 100
thermo_style custom step temp f_myrxns[1] atoms bonds
run 400
print "RESULT reactions=$(f_myrxns[1]) atoms=$(atoms) bonds=$(bonds)"
"""


@pytest.mark.skipif(LMP is None, reason="no LAMMPS binary (lmp_serial) on PATH")
def test_lammps_reacts_a_molpy_written_bond_react_system(tmp_path):
    world, template = _build()

    for i, atom in enumerate(world.atoms, start=1):
        atom["id"] = i
    frame = world.to_frame()
    frame.simbox = mp.Box.cubic(40.0)

    n_atoms = frame["atoms"].nrows
    n_bonds = frame["bonds"].nrows
    assert template.edge_atoms, "template must exercise the EdgeIDs branch"

    out = tmp_path / "rxn"
    mp.io.write_lammps_bond_react_system(
        out,
        frame,
        _forcefield(world, template.pre, template.post),
        templates={"rxn1": template},
    )
    (out / "in.rxn").write_text(INPUT)

    proc = subprocess.run(
        [LMP, "-in", "in.rxn", "-log", "none"],
        cwd=out,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-1000:]

    # LAMMPS refuses to build topology across a template edge that is too close
    # to the reacting atoms; it warns rather than failing, so assert on the text.
    assert "too close to template edge" not in proc.stdout

    result = next(l for l in proc.stdout.splitlines() if l.startswith("RESULT"))
    parsed = dict(kv.split("=") for kv in result.split()[1:])

    assert int(parsed["reactions"]) == 1
    # the condensation drops two hydrogens and nets one bond (-2 C-H, +1 C-C)
    assert int(parsed["atoms"]) == n_atoms - 2
    assert int(parsed["bonds"]) == n_bonds - 1
