# author: Roy Kid
# contact: lijichen365@126.com
# date: 2024-02-27
# version: 0.0.1
from igraph import Graph

class ItemType:

    def __init__(self, name, **params):
        self._name = name
        self._params = params

    @property
    def params(self):
        return self._params

class AtomType(ItemType):

    def __init__(self, name, **params):
        super().__init__(**params)
        self._name = name

    @classmethod
    def from_vertex(cls, v):
        params = dict(v.attributes())
        return cls(params.pop('name'), **params.attributes())

    @property
    def name(self):
        return self._name

class BondType(ItemType):

    def __init__(self, itom, jtom, **params):
        super().__init__(**params)
        self._itom = itom
        self._jtom = jtom
    
    @classmethod
    def from_edge(cls, e):
        params = dict(e.attributes())
        return cls(e.source, e.target, **params)

class AngleType(ItemType):
    ...

class Forcefield:

    def __init__(self, name:str, n_atomtypes:int):
        self._name = name
        self._graph = Graph(n_atomtypes)
        self._bondtypes = {}
        self._angletypes = {}

    def def_atomtype(self, name, **params):
        v = self._graph.add_vertex(name, **params)
        return AtomType.from_vertex(v)

    def def_bond(self, itom, jtom, **params):
        e = self._graph.add_edge(itom, jtom, **params)
        return BondType.from_edge(e)
    
    def def_atomtypes(self, names, **params):
        n = len(names)
        params = params.update({'name': names})
        self._graph.add_vertices(n, **params)
    
    def def_bondtypes(self, conect: list[tuple[int, int]], **params):
        self._graph.add_edges(conect, **params)

    def def_angletype(self, itom, jtom, ktom, **params):
        
        if itom > ktom:
            itom, ktom = ktom, itom
        if not self._graph.are_connected(itom, ktom):
            self._graph.add_edge(itom, jtom)
        if not self._graph.are_connected(ktom, jtom):
            self._graph.add_edge(ktom, jtom)
        self._angletypes[(itom, jtom, ktom)] = AngleType(**params)

    def get_atomtype(self, name):
        return AtomType.from_vertex(self._graph.vs.find(name=name))
    
    def get_bondtype(self, )