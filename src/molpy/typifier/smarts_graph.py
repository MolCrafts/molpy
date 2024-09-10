"""Module for SMARTSGraph and SMARTS matching logic."""

from molpy import Element
from .smarts import SMARTS
from igraph import Graph
from collections import defaultdict

class SMARTSGraph:

    def __init__(self, name:str, smarts_string: str):
        self.smarts_string = smarts_string
        self.name = name

        self.ast = SMARTS().parse(smarts_string)
        self.graph = Graph()

    def concrete(self):
        self._atom_indices = dict()
        self._add_vertices()
        self._add_edges()
        self._add_label_edges()
        
    def _add_vertices(self):
        """ Add all atoms in the SMARTS string as vertices in the graph.
        """
        atoms = [node for node in self.ast.iter_subtrees_topdown() if node.data == 'atom']
        self.graph.add_vertices(len(atoms), attributes={'atom': atoms})

    def _add_edges(self, trunk=None):
        """Add all bonds in the SMARTS string as edges in the graph."""
        for ast_child in self.ast.children:
            if ast_child.data == "atom":
                atom_idx = self.graph.vs.select(atom=ast_child)
                if trunk is not None:
                    trunk_idx = self.graph.vs.select(atom=trunk)
                    self.graph.add_edge(atom_idx, trunk_idx)
                trunk = ast_child
            elif ast_child.data == "branch":
                self._add_edges(ast_child, trunk)

    def _add_label_edges(self):
        """Add edges between all atoms with the same atom_label in rings."""
        # We need each individual label and atoms with multiple ring labels
        # would yield e.g. the string '12' so split those up.
        label_digits = defaultdict(list)
        for vertex in self.graph.vs:
            atom = vertex['atom']
            for label in atom.find_data('atom_label'):
                digits = list(label.children[0])
                for digit in digits:
                    label_digits[digit].append(atom)

        for label, (atom1, atom2) in label_digits.items():
            atom1_idx = atom1.index
            atom2_idx = atom2.index
            self.graph.add_edge(atom1_idx, atom2_idx)

    def _node_match(self, host, pattern):
        """Determine if two graph nodes are equal."""
        atom_expr = pattern["atom"].children[0]
        atom = host["atom_data"]
        bond_partners = host["bond_partners"]
        return self._atom_expr_matches(atom_expr, atom, bond_partners)
    
    def _atom_expr_matches(self, atom_expr, atom, bond_partners):
        """Evaluate SMARTS string expressions."""
        if atom_expr.data == "not_expression":
            return not self._atom_expr_matches(
                atom_expr.children[0], atom, bond_partners
            )
        elif atom_expr.data in ("and_expression", "weak_and_expression"):
            return self._atom_expr_matches(
                atom_expr.children[0], atom, bond_partners
            ) and self._atom_expr_matches(atom_expr.children[1], atom, bond_partners)
        elif atom_expr.data == "or_expression":
            return self._atom_expr_matches(
                atom_expr.children[0], atom, bond_partners
            ) or self._atom_expr_matches(atom_expr.children[1], atom, bond_partners)
        elif atom_expr.data == "atom_id":
            return self._atom_id_matches(
                atom_expr.children[0], atom, bond_partners, self.typemap
            )
        elif atom_expr.data == "atom_symbol":
            return self._atom_id_matches(atom_expr, atom, bond_partners, self.typemap)
        else:
            raise TypeError(
                "Expected atom_id, atom_symbol, and_expression, "
                "or_expression, or not_expression. "
                "Got {}".format(atom_expr.data)
            )
        