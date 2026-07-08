"""Non-linear polymer topologies: build() takes any CGSmiles graph.

``build_sequence`` (02) makes linear chains. ``PolymerBuilder.build()`` takes an
arbitrary CGSmiles graph, so one monomer library + connector builds linear,
ring, and branched architectures — the topology lives entirely in the string:

    "{[#EO2]|4[#PS]}"                     linear
    "{[#EO2]1[#PS][#EO2][#PS][#EO2]1}"    ring (…1…1 closes it)
    "{[#PS][#EO3]([#PS])[#PS]}"           branch (EO3 is a 3-arm junction)

Guide: docs/user-guide/03_polymer_topology.md
Run:   python 03_polymer_topology.py
"""

import molpy as mp
from molpy.builder.polymer import (
    Connector,
    CovalentSeparator,
    LinearOrienter,
    Placer,
    PolymerBuilder,
)
from molpy.conformer import Conformer
from molpy.reacter import (
    Reacter,
    find_neighbors,
    form_single_bond,
    select_neighbor,
    select_self,
)

# Monomers (BigSMILES). EO3 carries three $ ports — a branch junction.
BIGSMILES = {
    "EO2": "{[][$]OCCO[$][]}",
    "EO3": "{[][$]OCC(CO[$])(CO[$])[]}",
    "PS": "{[][$]OCC(c1ccccc1)CO[$][]}",
}

TOPOLOGIES = {
    "linear": "{[#EO2]|4[#PS]}",
    "ring": "{[#EO2]1[#PS][#EO2][#PS][#EO2]1}",
    "branch": "{[#PS][#EO3]([#PS])[#PS]}",
}


def _hydroxyl(struct, site):
    """Leaving group on one side: the -OH (O + its H) on the anchor carbon."""
    for neighbor in find_neighbors(struct, site):
        if neighbor.get("element") == "O":
            hydrogens = find_neighbors(struct, neighbor, element="H")
            if hydrogens:
                return [neighbor, hydrogens[0]]
    raise ValueError("no hydroxyl group at the reaction site")


def _one_hydrogen(struct, site):
    """Leaving group on the other side: a single H (ether-forming dehydration)."""
    hydrogens = find_neighbors(struct, site, element="H")
    if not hydrogens:
        raise ValueError("no hydrogen at the reaction site")
    return [hydrogens[0]]


def main() -> None:
    # Prepare each monomer: parse -> 3D (native Conformer) -> angles/dihedrals.
    library = {}
    for label, bigsmiles in BIGSMILES.items():
        monomer, _ = Conformer(add_hydrogens=True, seed=42).generate(
            mp.parser.parse_monomer(bigsmiles)
        )
        library[label] = monomer.get_topo(gen_angle=True, gen_dihe=True)

    # One reaction (ether-forming dehydration) + connector rules for every pair.
    reaction = Reacter(
        name="dehydration",
        anchor_selector_left=select_neighbor("C"),
        anchor_selector_right=select_self,
        leaving_selector_left=_hydroxyl,
        leaving_selector_right=_one_hydrogen,
        bond_former=form_single_bond,
    )
    rules = {(left, right): ("$", "$") for left in library for right in library}
    builder = PolymerBuilder(
        library,
        connector=Connector(port_map=rules, reacter=reaction),
        placer=Placer(CovalentSeparator(buffer=-0.1), LinearOrienter()),
    )

    # Same builder, three architectures — only the CGSmiles graph changes.
    for name, expr in TOPOLOGIES.items():
        result = builder.build(expr)
        print(
            f"{name:7s} {expr:38s} -> {len(list(result.polymer.atoms))} atoms, "
            f"{result.total_steps} bonds formed"
        )


if __name__ == "__main__":
    main()
