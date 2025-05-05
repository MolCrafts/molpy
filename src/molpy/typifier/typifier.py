import subprocess
from tempfile import TemporaryDirectory, mkdtemp
from molpy.typifier.parser import SmartsParser
from molpy.typifier.graph import SMARTSGraph, _find_chordless_cycles
from molpy.io import write_pdb
from pathlib import Path


class Typifier:

    def __init__(self, forcefield):
        self.forcefield = forcefield

    def typify_bonds(self, structure):
        bonds = structure.get_bonds()
        bondtypes = self.forcefield.get_bondtypes()
        # if bond.itom.type and bond.jtom.type equal to bondtype
        # match the bondtype to the bond
        for bond in bonds:
            for bondtype in bondtypes:
                if bondtype.match(bond):
                    bondtype.apply(bond)
                    break
        return structure


class SmartsTypifier(Typifier):

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

    def typify(self, structure, use_residue_map=False, max_iter=10):

        graph = structure.get_topology(attrs=["name", "number", "type"])
        self.prepare_graph(graph)
        for typename, rule in self.smarts_graphs.items():
            result = rule.find_matches(graph)
            if result:
                for i, j in enumerate(result):
                    structure["atoms"][j]["type"] = typename
                    print("atom", structure["atoms"][j], "type", typename)
                    print()
                # if all([atom['type'] for atom in structure['atoms']]):
                #     break

        return structure

    def prepare_graph(self, graph):

        all_cycles = _find_chordless_cycles(graph, max_cycle_size=8)
        for i, cycles in zip(graph.vs, all_cycles):
            for cycle in cycles:
                graph.vs[i.index]["cycles"].add(tuple(cycle))


class AmberToolsTypifier:

    def __init__(self, forcefield: str, charge_type: str = "bcc", conda_env: str = "AmberTools25"):
        self.forcefield = forcefield
        self.charge_type = charge_type
        self.conda_env = conda_env
        self.check_antechamber()

    def check_antechamber(self):
        """
        Check if antechamber is available in the target conda env.
        """
        cmd = f'''
        source $(conda info --base)/etc/profile.d/conda.sh && \
        conda activate {self.conda_env} && \
        antechamber -h
        '''
        result = subprocess.run(["bash", "-c", cmd], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.returncode != 0:
            raise RuntimeError("Antechamber not found in conda env " + self.conda_env)

    def typify(self, struct, workdir=None):
        """
        Typify the struct using AmberTools.
        """
        from molpy.io import write_pdb  # 假设你已有此函数

        net_charge = struct.get("net_charge", 0.0)

        with TemporaryDirectory() if workdir is None else Path(workdir) as tmpdir:
            workdir = Path(tmpdir)
            input_pdb = workdir / "struct.pdb"
            output_ac = workdir / "struct.ac"
            write_pdb(input_pdb, struct.to_frame())

            bash_cmd = f'''
            source $(conda info --base)/etc/profile.d/conda.sh && \
            conda activate {self.conda_env} && \
            antechamber -i {input_pdb} -fi pdb -o {output_ac} -fo ac -an y -at {self.forcefield} -c {self.charge_type} -nc {net_charge}
            '''
            result = subprocess.run(["bash", "-c", bash_cmd], cwd=workdir)
            if result.returncode != 0:
                raise RuntimeError("Antechamber failed.")

            print(f"AC file written to: {output_ac}")