from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Generator, Any

import molpy as mp
from molpy.io import write_pdb
from molpy.typifier.graph import SMARTSGraph, _find_chordless_cycles
from molpy.typifier.parser import SmartsParser
from molpy.core import ForceField
import molq


class BaseTypifier: ...


class ForceFieldTypifier(BaseTypifier):

    def __init__(self, forcefield: ForceField):
        self.forcefield = forcefield

    def typify_atoms(self, struct):
        return struct

    def typify_bonds(self, struct):
        bonds = struct.bonds
        bondtypes = self.forcefield.get_bondtypes()
        for bond in bonds:
            for bondtype in bondtypes:
                if bondtype.match(bond):
                    bond["type"] = str(bondtype)
                    break
        return struct

    def typify_angles(self, struct):
        angles = struct.angles
        angletype = self.forcefield.get_angletypes()
        for angle in angles:
            for angletype in angletype:
                if angletype.match(angle):
                    angle["type"] = str(angletype)
                    break
            if "type" not in angle:
                raise ValueError(
                    f"Angle {angle} type not found in forcefield {self.forcefield}"
                )
        return struct

    def typify(self, struct):
        """
        Typify the structure using the forcefield.
        """
        # Typify atoms
        struct = self.typify_atoms(struct)

        # Typify bonds
        struct = self.typify_bonds(struct)

        # Typify angles
        struct = self.typify_angles(struct)

        return struct


class SmartsTypifier(ForceFieldTypifier):

    def __init__(self, forcefield):

        super().__init__(forcefield)
        self.parser = SmartsParser()
        self.smarts_graphs = self.read_smarts(forcefield)

    def read_smarts(self, forcefield):

        smarts_graphs = {}
        smarts_overrides = {}

        probe_atomtype = forcefield.get_atomtypes()[0]

        if "def" in probe_atomtype:
            flag = "def"
        elif "smirks" in probe_atomtype:
            flag = "smirks"
        else:
            raise ValueError("No SMARTS or SMIRKS found in atomtype")

        for atomtype in forcefield.get_atomtypes():
            label = atomtype.label
            smarts = atomtype[flag]
            graph = SMARTSGraph(smarts, self.parser, label, overrides=None)
            smarts_graphs[label] = graph
            overrides = atomtype.get("overrides", None)
            if overrides is not None:
                smarts_overrides[label] = overrides.split(",")

        for label, override in smarts_overrides.items():
            print(f"Overriding {label} with {override}")
            graph = smarts_graphs[label]
            graph.override([smarts_graphs[atom] for atom in override])

        print(sorted(smarts_graphs.items(), key=lambda x: x[1].priority))
        smarts_graphs = dict(sorted(smarts_graphs.items(), key=lambda x: x[1].priority))

        return smarts_graphs

    def typify(self, struct, use_residue_map=False, max_iter=10):

        graph = struct.get_topology(attrs=["name", "number", "type"])
        self.prepare_graph(graph)
        for typename, rule in self.smarts_graphs.items():
            result = rule.find_matches(graph)
            if result:
                for i, j in enumerate(result):
                    struct["atoms"][j]["type"] = typename
                    print("atom", struct["atoms"][j], "type", typename)
                    print()
                # if all([atom['type'] for atom in struct['atoms']]):
                #     break

        return struct

    def prepare_graph(self, graph):

        all_cycles = _find_chordless_cycles(graph, max_cycle_size=8)
        for i, cycles in zip(graph.vs, all_cycles):
            for cycle in cycles:
                graph.vs[i.index]["cycles"].add(tuple(cycle))

