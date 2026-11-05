"""Build a fixed PBIX-h-2.47 anion-exchange network and its 42 hydroxide counterions.

Run from molpy/: uv run --no-sync python examples/pbix_deterministic.py OUTPUT

This is a finite, unrelaxed example topology, not a sampled membrane. The
ladder backbone and its H termination follow the source structure drawing;
the h linker is N+(CH3)2-(CH2)6-N+(CH3)2, with the unconnected end of a
single-ended fragment methylated. The drawing gives 40 repeat units, 19
bridges and two single-ended fragments; the bridge table below is one
illustrative pairing that realises those counts.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import molpy as mp
import numpy as np


def main() -> None:
    """Compose native fragment, conformer, assembly and file operations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path, help="Directory for the example's artifacts"
    )
    args = parser.parse_args()

    # The aryl-propyl side group belongs to the linker template. The fifth
    # backbone port is on the carbon bearing CF3, at its bond to that aryl.
    bodies = {
        "PBIX": "CC1(C)CC2(c3cc([<a])c([<b])cc31)CC(C)(C)c1cc(O[>a])c(C([>b])(C(F)(F)F)[>s])cc12",
        "Hbridge": "[<s]c1ccc(CCC[N+](C)(C)CCCCCC[N+](C)(C)CCCc2ccc([<s])cc2)cc1",
        "Hsingle": "[<s]c1ccc(CCC[N+](C)(C)CCCCCC[N+](C)(C)C)cc1",
    }
    expected_ports = {
        "PBIX": [("<", "a"), ("<", "b"), (">", "a"), (">", "b"), (">", "s")],
        "Hbridge": [("<", "s"), ("<", "s")],
        "Hsingle": [("<", "s")],
    }
    conformer = mp.conformer.Conformer(seed=42)
    library = {}
    port_anchors = {}
    for name, body in bodies.items():
        template = mp.io.smiles.read_smiles_fragment(body).to_template()
        library[name] = conformer.generate(template)[0]
        assert [
            (p["port_kind"], p["port_label"]) for p in library[name].ports
        ] == expected_ports[name]
        for row, atom in enumerate(library[name].atoms):
            atom["template_atom"] = row
        port_anchors[name] = [
            int(port.anchor["template_atom"]) for port in library[name].ports
        ]

    sites = mp.CoarseGrain()
    chains = [[sites.add_bead(bead_type="PBIX") for _ in range(10)] for _ in range(4)]
    sources = [
        {"frag_id": 10 * chain + site, "template": "PBIX", "chain": chain, "site": site}
        for chain in range(4)
        for site in range(10)
    ]
    connections = []
    for chain, beads in enumerate(chains):
        for site, (left, right) in enumerate(zip(beads, beads[1:])):
            for src_port, dst_port in [(2, 0), (3, 1)]:
                sites.add_bond(left, right, src_port=src_port, dst_port=dst_port)
                connections.append(
                    {
                        "src": 10 * chain + site,
                        "src_port": src_port,
                        "dst": 10 * chain + site + 1,
                        "dst_port": dst_port,
                        "role": "backbone",
                    }
                )

    bridge_pairs = [((0, k), (1, k)) for k in range(9)]
    bridge_pairs += [((2, k), (3, k)) for k in range(9)]
    bridge_pairs += [((0, 9), (2, 9))]
    singles = [(1, 9), (3, 9)]
    used_sites = Counter()
    for bridge, (left, right) in enumerate(bridge_pairs):
        frag_id = 40 + bridge
        bead = sites.add_bead(bead_type="Hbridge")
        sources.append(
            {"frag_id": frag_id, "template": "Hbridge", "ends": [left, right]}
        )
        for port, (chain, site) in enumerate((left, right)):
            sites.add_bond(chains[chain][site], bead, src_port=4, dst_port=port)
            connections.append(
                {
                    "src": 10 * chain + site,
                    "src_port": 4,
                    "dst": frag_id,
                    "dst_port": port,
                    "role": "bridge",
                }
            )
            used_sites[chain, site] += 1
    for single, (chain, site) in enumerate(singles):
        frag_id = 59 + single
        bead = sites.add_bead(bead_type="Hsingle")
        sites.add_bond(chains[chain][site], bead, src_port=4, dst_port=0)
        connections.append(
            {
                "src": 10 * chain + site,
                "src_port": 4,
                "dst": frag_id,
                "dst_port": 0,
                "role": "single",
            }
        )
        sources.append(
            {"frag_id": frag_id, "template": "Hsingle", "end": [chain, site]}
        )
        used_sites[chain, site] += 1

    assert used_sites == Counter(
        {(chain, site): 1 for chain in range(4) for site in range(10)}
    )
    assert sites.n_beads == 61 and sites.n_bonds == 112
    polymer = mp.builder.Assembler(library, mp.builder.GrowthPlacer()).assemble(sites)
    polymer_frame = polymer.to_frame()
    polymer_atoms = polymer_frame["atoms"]
    polymer_topology = mp.Topology.from_frame(polymer_frame)
    assert polymer_topology.n_components == 1
    assert int(polymer_atoms["formal_charge"].sum()) == 42
    assert polymer.n_ports == 16  # Four unconsumed backbone ports per chain.
    assert all(port.handle_atom["atomic_number"] == 1 for port in polymer.ports)
    nitrogens = [atom for atom in polymer.atoms if atom["atomic_number"] == 7]
    assert len(nitrogens) == 42
    assert all(
        atom["formal_charge"] == 1 and len(polymer.neighbors(atom)) == 4
        for atom in nitrogens
    )
    assert not np.any(polymer_atoms["atomic_number"] == 35)  # No residual precursor Br.

    # Check the actual product against the requested fragment pairs. Two
    # bonds between successive PBIX units are deliberately counted twice.
    frag_ids = polymer_atoms["frag_id"]
    endpoints = polymer_topology.bonds()
    actual_pairs = Counter(
        tuple(sorted((int(frag_ids[a]), int(frag_ids[b]))))
        for a, b in endpoints
        if frag_ids[a] != frag_ids[b]
    )
    expected_pairs = Counter(tuple(sorted((c["src"], c["dst"]))) for c in connections)
    assert actual_pairs == expected_pairs
    template_atoms = polymer_atoms["template_atom"]
    actual_anchors = Counter(
        tuple(
            sorted(
                (
                    (int(frag_ids[a]), int(template_atoms[a])),
                    (int(frag_ids[b]), int(template_atoms[b])),
                )
            )
        )
        for a, b in endpoints
        if frag_ids[a] != frag_ids[b]
    )
    expected_anchors = Counter(
        tuple(
            sorted(
                (
                    (
                        c["src"],
                        port_anchors[sources[c["src"]]["template"]][c["src_port"]],
                    ),
                    (
                        c["dst"],
                        port_anchors[sources[c["dst"]]["template"]][c["dst_port"]],
                    ),
                )
            )
        )
        for c in connections
    )
    assert actual_anchors == expected_anchors

    system = copy.copy(polymer)
    hydroxide = conformer.generate(mp.io.smiles.read_smiles("[OH-]").to_atomistic())[0]
    assert hydroxide.n_atoms == 2
    assert sum(int(atom.get("formal_charge", 0)) for atom in hydroxide.atoms) == -1
    hydroxide.translate(-hydroxide.center(hydroxide.atoms))
    positions = np.column_stack([polymer_atoms[key] for key in ("x", "y", "z")])
    center = positions.mean(axis=0)
    radius = float(np.linalg.norm(positions - center, axis=1).max()) + 10.0
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for ion in range(42):
        z = 1.0 - 2.0 * (ion + 0.5) / 42
        r = np.sqrt(1.0 - z * z)
        theta = golden_angle * ion
        delta = center + radius * np.array([r * np.cos(theta), r * np.sin(theta), z])
        fragment = copy.copy(hydroxide)
        fragment.translate(delta)
        for row, atom in enumerate(fragment.atoms):
            atom["frag_id"] = 61 + ion
            atom["mol_id"] = 2 + ion
            atom["template_atom"] = row
        system.merge(fragment)
        sources.append({"frag_id": 61 + ion, "template": "OH", "ion": ion})

    system_frame = system.to_frame()
    system_topology = mp.Topology.from_frame(system_frame)
    assert system_topology.n_components == 43
    assert int(system_frame["atoms"]["formal_charge"].sum()) == 0
    assert np.isfinite(positions).all()
    assert system.n_atoms == polymer.n_atoms + 84

    args.output.mkdir(parents=True, exist_ok=True)
    for name, frame in [("polymer", polymer_frame), ("system", system_frame)]:
        # Native wire JSON retains chemical columns, ports and fragment ids.
        (args.output / f"{name}.json").write_bytes(
            mp.io.write_wire_frame(frame, "json")
        )
        mp.io.write_mol2(args.output / f"{name}.mol2", frame)
        mp.io.write_pdb(args.output / f"{name}.pdb", frame)
    report = {
        "topology": "fixed illustrative PBIX-h-2.47, not an experimental microscopic network",
        "seed": 42,
        "optimized": False,
        "polymer_atoms": polymer.n_atoms,
        "system_atoms": system.n_atoms,
        "polymer_fragments": 61,
        "connections": 112,
        "side_sites_consumed": 40,
        "bridges": 19,
        "single_ended_fragments": 2,
        "hydroxides": 42,
        "polymer_components": 1,
        "system_components": 43,
        "polymer_formal_charge": 42,
        "system_formal_charge": 0,
        "ion_shell_radius_angstrom": radius,
    }
    artifacts = {
        "fragments": bodies,
        "connections": connections,
        "sources": sources,
        "report": report,
        "recipe": {
            "chemistry_sources": [
                "Chemical structure for simulation, pages 1-2: h-type charged fragments and model counts",
                "Ladder backbone in a common orientation, H termination, no specified stereochemistry, (CH2CH2)3 = six methylenes",
                "Illustrative bridge table realising 19 bridges and 2 single ends; 42 hydroxide counterions",
            ],
            "bridge_pairs": bridge_pairs,
            "single_sites": singles,
            "port_order": expected_ports,
            "port_anchor_template_atoms": port_anchors,
            "length_unit": "angstrom",
        },
        "atom_sources": [
            {
                "atom_row": row,
                "frag_id": int(frag_id),
                "template_atom": int(system_frame["atoms"]["template_atom"][row]),
            }
            for row, frag_id in enumerate(system_frame["atoms"]["frag_id"])
        ],
    }
    for name, value in artifacts.items():
        (args.output / f"{name}.json").write_text(
            json.dumps(value, indent=2) + "\n", encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
