# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-15
# version: 0.0.1

from ast import literal_eval
from functools import cached_property

from typing import Dict, List, Optional, Union
from itertools import permutations
from molpy.utils.typing import ArrayLike, Number, N

value = Number  # generic type: interpreted as either a number or str
digitalize = literal_eval

class SelectError(BaseException):
    pass

class Node:
    def __init__(self, tag, **attrs):

        self.tag = tag
        #TODO: self.parent = parent
        self.attrs = attrs
        self.children = []

    def add_child(self, child: 'Node'):

        self.children.append(child)

    def add_children(self, children: List['Node']):
        self.children.extend(children)

    def get_children(self, key):
        return [c for c in self.children if c.tag == key]

    def get_child(self, key):
        for child in self.children:
            if child.tag == key:
                return child

    def __getitem__(self, key):

        return self.children[key]

    def __getattr__(self, __name: str):
        return self.attrs[__name]

    def __repr__(self):
        return f'<{self.tag}: {self.attrs}, with {len(self.children)} subnodes>'

    def __iter__(self):
        return iter(self.children)

    def __contains__(self, key):
        if self.get_child(key):
            return True
        else:
            return False

    def get_nodes(self, path:str)->List['Node']:
        """
        get all nodes of a certain path

        Args:
            path: str. a path-like str to locate nodes

        Examples:
            >>> fftree.get_nodes('HarmonicBondForce/Bond')
            >>> [<Bond 1>, <Bond 2>, ...]

        Returns:
            List[Node]: a list of Node
        """
        steps = path.split("/")
        val = self
        for nstep, step in enumerate(steps):
            name, index = step, -1
            if "[" in step:
                name, index = step.split("[")
                index = int(index[:-1])
            val = [c for c in val.children if c.tag == name]
            if index >= 0:
                val = val[index]
            elif nstep < len(steps) - 1:
                val = val[0]
        return val

    def get_attribs(self, path:str, attrname:Union[str, List[str]])->ArrayLike[N, value]:
        """
        get all values of attributes of nodes which nodes matching certain path

        Examples:
            >>> fftree.get_attribs('HarmonicBondForce/Bond', 'k')
            >>> [[2.0], [2.0], [2.0], [2.0], [2.0], ...]
            >>> fftree.get_attribs('HarmonicBondForce/Bond', ['k', 'r0'])
            >>> [[2.0, 1.53], [2.0, 1.53], ...]
        Args:
            parser str: a path to locate nodes
            attrname str: attribute name or a list of attribute names of a node
        Returns:
            List[Union[float, str]]
                a list of values of attributes
        """
        sel = self.get_nodes(path)

        if isinstance(attrname, str):
            attrname = [attrname]

        ret = []
        for item in sel:
            vals = [digitalize(item.attrs[an]) if an in item.attrs else None for an in attrname]
            ret.append(vals)
        return ret

    def set_node(self, parser:str, values:List[Dict[str, value]])->None:
        """
        set attributes of nodes which nodes matching certain path
        Parameters
        ----------
        parser : str
            path to locate nodes
        values : List[Dict[str, value]]
            a list of Dict[str, value], where value is any type can be convert to str of a number.
        Examples
        --------
        >>> fftree.set_node('HarmonicBondForce/Bond', 
                            [{'k': 2.0, 'r0': 1.53}, 
                             {'k': 2.0, 'r0': 1.53}])
        """
        nodes = self.get_nodes(parser)
        for nit in range(len(values)):
            for key in values[nit]:
                nodes[nit].attrs[key] = f"{values[nit][key]}"

    def set_attrib(self, path:str, attrname:str, values:Union[value, List[value]]):
        """
        set ONE Attribute of nodes which nodes matching certain path
        Parameters
        ----------
        path : str
            path to locate nodes
        attrname : str
            attribute name
        values : Union[float, str, List[float, str]]
            attribute value or a list of attribute values of a node
        Examples
        --------
        >>> fftree.set_attrib('HarmonicBondForce/Bond', 'k', 2.0)
        >>> fftree.set_attrib('HarmonicBondForce/Bond', 'k', [2.0, 2.0, 2.0, 2.0, 2.0])
        """
        if len(values) == 0:
            valdicts = [{attrname: values}]
        else:
            valdicts = [{attrname: i} for i in values]
        self.set_node(path, valdicts)

class TypeMatcher:
    def __init__(self, fftree: Node, parser):
        """
        Freeze type matching list.
        """
        atypes = fftree.get_attribs("AtomTypes/Type", "name")
        aclasses = fftree.get_attribs("AtomTypes/Type", "class")
        self.class2type = {}
        for nline in range(len(atypes)):
            if aclasses[nline] not in self.class2type:
                self.class2type[aclasses[nline]] = []
            self.class2type[aclasses[nline]].append(atypes[nline])
        self.class2type[""] = atypes
        funcs = fftree.get_nodes(parser)
        self.functions = []
        for node in funcs:
            tmp = []
            for key in node.attrs:
                if len(key) > 4 and "type" == key[:4]:
                    nit = int(key[4:])
                    if len(node.attrs[key]) == 0:
                        tmp.append((nit, atypes))
                    else:
                        tmp.append((nit, [node.attrs[key]]))
                elif key == "type":
                    tmp.append((1, [node.attrs[key]]))
                elif len(key) > 5 and "class" == key[:5]:
                    nit = int(key[5:])
                    tmp.append((nit, self.class2type[node.attrs[key]]))
                elif key == "class":
                    tmp.append((1, self.class2type[node.attrs[key]]))
            tmp = sorted(tmp, key=lambda x: x[0])
            self.functions.append([i[1] for i in tmp])

    def matchGeneral(self, types):
        matches = []
        for nterm, term in enumerate(self.functions):
            ifMatch, ifForward = self._match(types, term)
            if ifMatch:
                matches.append((ifMatch, ifForward, nterm))
        if len(matches) == 0:
            return False, False, -1
        return matches[-1]

    def _match(self, types, term):
        if len(types) != len(term):
            raise ValueError(
                "The length of matching types is not equivalent to the forcefield term."
            )
        # Forward
        ifMatchForward = True
        for ntypes in range(len(types)):
            if len(types[ntypes]) == 0:
                continue
            if types[ntypes] not in term[ntypes]:
                ifMatchForward = False
                break
        ifMatchReverse = True
        for ntypes in range(len(types)):
            if len(types[len(types) - ntypes - 1]) == 0:
                continue
            if types[len(types) - ntypes - 1] not in term[ntypes]:
                ifMatchReverse = False
                break
        return ifMatchForward or ifMatchReverse, ifMatchForward

    def matchImproper(self, torsion, data, ordering="amber"):
        type1, type2, type3, type4 = [data.atomType[data.atoms[torsion[i]]] for i in range(4)]
        match = None
        for nterm, term in enumerate(self.functions):
            types1 = term[0]
            types2 = term[1]
            types3 = term[2]
            types4 = term[3]
            hasWildcard = (len(self.class2type[""])
                           in (len(types1), len(types2), len(types3),
                               len(types4)))
            if type1 in types1:
                for (t2, t3, t4) in permutations(((type2, 1), (type3, 2), (type4, 3))):
                    if t2[0] in types2 and t3[0] in types3 and t4[0] in types4:
                        if ordering == 'default':
                            # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                            # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                            # to pick the order.
                            a1 = torsion[t2[1]]
                            a2 = torsion[t3[1]]
                            e1 = data.atoms[a1].element
                            e2 = data.atoms[a2].element
                            if e1 == e2 and a1 > a2:
                                (a1, a2) = (a2, a1)
                            elif e1.symbol != "C" and (e2.symbol == "C" or e1.mass < e2.mass):
                                (a1, a2) = (a2, a1)
                            match = (a1, a2, torsion[0], torsion[t4[1]], nterm)
                            break
                        elif ordering == 'charmm':
                            if hasWildcard:
                                # Workaround to be more consistent with AMBER.  It uses wildcards to define most of its
                                # impropers, which leaves the ordering ambiguous.  It then follows some bizarre rules
                                # to pick the order.
                                a1 = torsion[t2[1]]
                                a2 = torsion[t3[1]]
                                e1 = data.atoms[a1].element
                                e2 = data.atoms[a2].element
                                if e1 == e2 and a1 > a2:
                                    (a1, a2) = (a2, a1)
                                elif e1.symbol != "C" and (e2.symbol == "C" or e1.mass < e2.mass):
                                    (a1, a2) = (a2, a1)
                                match = (a1, a2, torsion[0], torsion[t4[1]], nterm)
                            else:
                                # There are no wildcards, so the order is unambiguous.
                                match = (torsion[0], torsion[t2[1]], torsion[t3[1]], torsion[t4[1]], nterm)
                            break
                        elif ordering == 'amber':
                            # topology atom indexes
                            a2 = torsion[t2[1]]
                            a3 = torsion[t3[1]]
                            a4 = torsion[t4[1]]
                            # residue indexes
                            r2 = data.atoms[a2].residue.index
                            r3 = data.atoms[a3].residue.index
                            r4 = data.atoms[a4].residue.index
                            # template atom indexes
                            ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                            ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                            ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                            # elements
                            e2 = data.atoms[a2].element
                            e3 = data.atoms[a3].element
                            e4 = data.atoms[a4].element
                            if not hasWildcard:
                                if t2[0] == t4[0] and (r2 > r4 or (r2 == r4 and ta2 > ta4)):
                                    (a2, a4) = (a4, a2)
                                    r2 = data.atoms[a2].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if t3[0] == t4[0] and (r3 > r4 or (r3 == r4 and ta3 > ta4)):
                                    (a3, a4) = (a4, a3)
                                    r3 = data.atoms[a3].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if t2[0] == t3[0] and (r2 > r3 or (r2 == r3 and ta2 > ta3)):
                                    (a2, a3) = (a3, a2)
                            else:
                                if e2 == e4 and (r2 > r4 or (r2 == r4 and ta2 > ta4)):
                                    (a2, a4) = (a4, a2)
                                    r2 = data.atoms[a2].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta2 = data.atomTemplateIndexes[data.atoms[a2]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if e3 == e4 and (r3 > r4 or (r3 == r4 and ta3 > ta4)):
                                    (a3, a4) = (a4, a3)
                                    r3 = data.atoms[a3].residue.index
                                    r4 = data.atoms[a4].residue.index
                                    ta3 = data.atomTemplateIndexes[data.atoms[a3]]
                                    ta4 = data.atomTemplateIndexes[data.atoms[a4]]
                                if r2 > r3 or (r2 == r3 and ta2 > ta3):
                                    (a2, a3) = (a3, a2)
                            match = (a2, a3, torsion[0], a4, nterm)
                            break
                        elif ordering == 'smirnoff':
                            # topology atom indexes
                            a1 = torsion[0]
                            a2 = torsion[t2[1]]
                            a3 = torsion[t3[1]]
                            a4 = torsion[t4[1]]
                            # enforce exact match
                            match = (a1, a2, a3, a4, nterm)
                            break
        return match

class ForceField:

    def __init__(self, ):

        self.root = Node('ForceField')
        self._atomTypes = Node('AtomTypes')
        self._residues = Node('Residues')
        self.root.add_children([
            self._atomTypes,
            self._residues,
        ])

    @property
    def atomTypes(self):
        return self._atomTypes

    @property
    def residues(self):
        return self._residues

    def def_atom(self, typeName:str, typeClass:Optional[str]=None, **attributes):

        self.atomTypes.add_child(Node('Type', typeName, typeClass, **attributes))

    def get_atom_by_name(self, typeName:str):

        for atom in self.atomTypes.children:
            if atom.name == typeName:
                return atom

    def def_bond(self, name, type1, type2, **params):
        pass

    def load_xml(self, fpath):
        from molpy.io.xml import XMLParser
        self.xmlparser = XMLParser(self.root)
        self.xmlparser.parse(fpath)