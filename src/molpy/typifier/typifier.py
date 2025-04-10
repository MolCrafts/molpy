from collections import OrderedDict
from molpy.typifier.parser import SmartsParser
from molpy.typifier.graph import SMARTSGraph, _find_chordless_cycles

class SmartsTypifier:

    def __init__(self, forcefield):

        self.parser = SmartsParser()
        self.forcefield = forcefield
        self.smarts_graphs = self.read_smarts(forcefield)

    def read_smarts(self, forcefield):

        smarts_graphs = {}
        smarts_override = {}

        probe_atomtype = forcefield.get_atomtypes()[0]

        if 'def' in probe_atomtype:
            flag = 'def'
        elif 'smirks' in probe_atomtype:
            flag = 'smirks'
        else:
            raise ValueError('No SMARTS or SMIRKS found in atomtype')

        for atomtype in forcefield.get_atomtypes():
            name = atomtype.name
            smarts = atomtype[flag]
            graph = SMARTSGraph(smarts, self.parser, name, overrides=None)
            smarts_graphs[name] = graph
            override = atomtype.get('override', None)
            if override is not None:
                smarts_override[name] = override

        for name, override in smarts_override.items():
            graph = smarts_graphs[name]
            graph.override([smarts_graphs[atom] for atom in override])
            
        smarts_graphs = OrderedDict(
            sorted(smarts_graphs.items(), key=lambda x: x[1].priority, reverse=True)
        )

        return smarts_graphs
    

    def typify(self, structure, use_residue_map=False, max_iter=10):

        graph = structure.get_topology()
        self.prepare_graph(graph)
        for typename, rule in self.smarts_graphs.items():
            result = rule.find_matches(graph)
            if result:
                for i, j in enumerate(result):
                    structure['atoms'][i]['type'] = typename
                if all([atom['type'] for atom in structure['atoms']]):
                    break

        return structure

    def prepare_graph(self, graph):

        all_cycles = _find_chordless_cycles(graph, max_cycle_size=8)
        for i, cycles in zip(graph.vs, all_cycles):
            for cycle in cycles:
                graph.vs[i.index]["cycles"].add(tuple(cycle))