# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-26
# version: 0.0.1
# source: pckroon/pysmiles

"""
Exposes functionality needed for parsing SMILES strings.
"""

from collections import defaultdict
import enum
from molpy.group import Group
from molpy.atom import Atom
import warnings
import molpy
from molpy.element import Element

@enum.unique
class TokenType(enum.Enum):
    """Possible SMILES token types"""
    ATOM = 1
    BOND_TYPE = 2
    BRANCH_START = 3
    BRANCH_END = 4
    RING_NUM = 5
    EZSTEREO = 6


def _tokenize(smiles):
    """
    Iterates over a SMILES string, yielding tokens.
    Parameters
    ----------
    smiles : iterable
        The SMILES string to iterate over
    Yields
    ------
    tuple(TokenType, str)
        A tuple describing the type of token and the associated data
    """
    organic_subset = 'B C N O P S F Cl Br I * b c n o s p'.split()
    smiles = iter(smiles)
    token = ''
    peek = None
    while True:
        char = peek if peek else next(smiles, '')
        peek = None
        if not char:
            break
        if char == '[':
            token = char
            for char in smiles:
                token += char
                if char == ']':
                    break
            yield TokenType.ATOM, token
        elif char in organic_subset:
            peek = next(smiles, '')
            if char + peek in organic_subset:
                yield TokenType.ATOM, char + peek
                peek = None
            else:
                yield TokenType.ATOM, char
        elif char in '-=#$:.':
            yield TokenType.BOND_TYPE, char
        elif char == '(':
            yield TokenType.BRANCH_START, '('
        elif char == ')':
            yield TokenType.BRANCH_END, ')'
        elif char == '%':
            # If smiles is too short this will raise a ValueError, which is
            # (slightly) prettier than a StopIteration.
            yield TokenType.RING_NUM, int(next(smiles, '') + next(smiles, ''))
        elif char in '/\\':
            yield TokenType.EZSTEREO, char
        elif char.isdigit():
            yield TokenType.RING_NUM, int(char)


def read_smiles(smiles, explicit_hydrogen=False, zero_order_bonds=True, 
                reinterpret_aromatic=True):
    """
    Parses a SMILES string.
    Parameters
    ----------
    smiles : iterable
        The SMILES string to parse. Should conform to the OpenSMILES
        specification.
    explicit_hydrogen : bool
        Whether hydrogens should be explicit nodes in the outout graph, or be
        implicit in 'hcount' attributes.
    reinterprit_aromatic : bool
        Whether aromaticity should be determined from the created molecule,
        instead of taken from the SMILES string.
    Returns
    -------
    nx.Graph
        A graph describing a molecule. Nodes will have an 'element', 'aromatic'
        and a 'charge', and if `explicit_hydrogen` is False a 'hcount'.
        Depending on the input, they will also have 'isotope' and 'class'
        information.
        Edges will have an 'order'.
    """
    bond_to_order = {'-': 1, '=': 2, '#': 3, '$': 4, ':': 1.5, '.': 0}
    mol = Group(smiles)
    anchor = None
    idx = 0
    default_bond = 1
    next_bond = None
    branches = []
    ring_nums = {}
    for tokentype, token in _tokenize(smiles):
        if tokentype == TokenType.ATOM:
            mol.add(Atom('-'.join([token, str(idx)]), **parse_atom(token)))
            if anchor is not None:
                if next_bond is None:
                    next_bond = default_bond
                if next_bond or zero_order_bonds:
                    mol.addBondByIndex(anchor, idx, bondType=next_bond)
                next_bond = None
            anchor = idx
            idx += 1
        elif tokentype == TokenType.BRANCH_START:
            branches.append(anchor)
        elif tokentype == TokenType.BRANCH_END:
            anchor = branches.pop()
        elif tokentype == TokenType.BOND_TYPE:
            if next_bond is not None:
                raise ValueError('Previous bond (order {}) not used. '
                                 'Overwritten by "{}"'.format(next_bond, token))
            next_bond = bond_to_order[token]
        elif tokentype == TokenType.RING_NUM:
            if token in ring_nums:
                jdx, order = ring_nums[token]
                if next_bond is None and order is None:
                    next_bond = default_bond
                elif order is None:  # Note that the check is needed,
                    next_bond = next_bond  # But this could be pass.
                elif next_bond is None:
                    next_bond = order
                elif next_bond != order:  # Both are not None
                    raise ValueError('Conflicting bond orders for ring '
                                     'between indices {}'.format(token))
                # idx is the index of the *next* atom we're adding. So: -1.
                if mol.getBondByIndex(idx-1, jdx):
                    raise ValueError('Edge specified by marker {} already '
                                     'exists'.format(token))
                if idx-1 == jdx:
                    raise ValueError('Marker {} specifies a bond between an '
                                     'atom and itself'.format(token))
                if next_bond or zero_order_bonds:
                    mol.addBondByIndex(idx - 1, jdx, bondType=next_bond)
                next_bond = None
                del ring_nums[token]
            else:
                if idx == 0:
                    raise ValueError("Can't have a marker ({}) before an atom"
                                     "".format(token))
                # idx is the index of the *next* atom we're adding. So: -1.
                ring_nums[token] = (idx - 1, next_bond)
                next_bond = None
        elif tokentype == TokenType.EZSTEREO:
            warnings.warn(f'E/Z stereochemical information, which is specified by {token}, will be discarded')
    if ring_nums:
        raise KeyError('Unmatched ring indices {}'.format(list(ring_nums.keys())))

    # Time to deal with aromaticity. This is a mess, because it's not super
    # clear what aromaticity information has been provided, and what should be
    # inferred. In addition, to what extend do we want to provide a "sane"
    # molecule, even if this overrides what the SMILES string specifies?
    cycles = mol.getBasisCycles()
    ring_atoms = set()
    for cycle in cycles:
        ring_atoms.update(cycle)
    non_ring_atoms = set(mol.atoms) - ring_atoms
    for n_atoms in non_ring_atoms:
        if n_atoms.get('aromatic', False):
        # if getattr(mol[n_idx], False):
            raise ValueError("You specified an aromatic atom outside of a"
                             " ring. This is impossible")
    
    mark_aromatic_edges(mol)
    fill_valence(mol)
    if reinterpret_aromatic:
        mark_aromatic_atoms(mol)
        mark_aromatic_edges(mol)
        for bond in mol.getBonds():
            if ((not bond.atom1.get('aromatic', False) or
                    not bond.atom2.get('aromatic', False))
                    and bond.get('order', 1) == 1.5):
                bond.order = 1

    if explicit_hydrogen:
        add_explicit_hydrogens(mol)
    else:
        remove_explicit_hydrogens(mol)
    return mol


import re

ISOTOPE_PATTERN = r'(?P<isotope>[\d]+)?'
ELEMENT_PATTERN = r'(?P<element>b|c|n|o|s|p|\*|[A-Z][a-z]{0,2})'
STEREO_PATTERN = r'(?P<stereo>@|@@|@TH[1-2]|@AL[1-2]|@SP[1-3]|@OH[\d]{1,2}|'\
                  r'@TB[\d]{1,2})?'
HCOUNT_PATTERN = r'(?P<hcount>H[\d]?)?'
CHARGE_PATTERN = r'(?P<charge>(-|\+)(\++|-+|[\d]{1,2})?)?'
CLASS_PATTERN = r'(?::(?P<class>[\d]+))?'
ATOM_PATTERN = re.compile(r'^\[' + ISOTOPE_PATTERN + ELEMENT_PATTERN +
                          STEREO_PATTERN + HCOUNT_PATTERN + CHARGE_PATTERN +
                          CLASS_PATTERN + r'\]$')

VALENCES = {"B": (3,), "C": (4,), "N": (3, 5), "O": (2,), "P": (3, 5),
            "S": (2, 4, 6), "F": (1,), "Cl": (1,), "Br": (1,), "I": (1,)}

AROMATIC_ATOMS = "B C N O P S Se As *".split()


def parse_atom(atom) -> dict:
    """
    Parses a SMILES atom token, and returns a dict with the information.
    Note
    ----
    Can not deal with stereochemical information yet. This gets discarded.
    Parameters
    ----------
    atom : str
        The atom string to interpret. Looks something like one of the
        following: "C", "c", "[13CH3-1:2]"
    Returns
    -------
    dict
        A dictionary containing at least 'element', 'aromatic', and 'charge'. If
        present, will also contain 'hcount', 'isotope', and 'class'.
    """
    defaults = {'charge': 0, 'hcount': 0, 'aromatic': False}
    if not atom.startswith('[') and not atom.endswith(']'):
        if atom != '*':
            # Don't specify hcount to signal we don't actually know anything
            # about it
            return {'element': atom.capitalize(), 'charge': 0,
                    'aromatic': atom.islower()}
        else:
            return defaults.copy()
    match = ATOM_PATTERN.match(atom)
    if match is None:
        raise ValueError('The atom {} is malformatted'.format(atom))
    out = defaults.copy()
    out.update({k: v for k, v in match.groupdict().items() if v is not None})

    if out.get('element', 'X').islower():
        out['aromatic'] = True

    parse_helpers = {
        'isotope': int,
        'element': str.capitalize,
        'stereo': lambda x: x,
        'hcount': parse_hcount,
        'charge': parse_charge,
        'class': int,
        'aromatic': lambda x: x,
    }

    for attr, val_str in out.items():
        out[attr] = parse_helpers[attr](val_str)

    if out['element'] == '*':
        del out['element']

    if out.get('element') == 'H' and out.get('hcount', 0):
        raise ValueError("A hydrogen atom can't have hydrogens")

    if 'stereo' in out:
        warnings.warn(message=f'Atom {atom} contains stereochemical information that will be discarded.')

    return out


def format_atom(molecule, node_key, default_element=Element.getBySymbol('*')):
    """
    Formats a node following SMILES conventions. Uses the attributes `element`,
    `charge`, `hcount`, `stereo`, `isotope` and `class`.
    Parameters
    ----------
    molecule : nx.Graph
        The molecule containing the atom.
    node_key : hashable
        The node key of the atom in `molecule`.
    default_element : str
        The element to use if the attribute is not present in the node.
    Returns
    -------
    str
        The atom as SMILES string.
    """
    node = node_key
    name = node.get('element', default_element).symbol
    charge = node.get('charge', 0)
    hcount = node.get('hcount', 0)
    stereo = node.get('stereo', None)
    isotope = node.get('isotope', '')
    class_ = node.get('class', '')
    aromatic = node.get('aromatic', False)
    default_h = has_default_h_count(molecule, node_key)

    if stereo is not None:
        raise NotImplementedError

    if aromatic:
        name = name.lower()

    if (stereo is None and isotope == '' and charge == 0 and default_h and
            class_ == '' and name.lower() in 'b c n o p s se as *'.split()):
        return name

    if hcount:
        hcountstr = 'H'
        if hcount > 1:
            hcountstr += str(hcount)
    else:
        hcountstr = ''

    if charge > 0:
        chargestr = '+'
        if charge > 1:
            chargestr += str(charge)
    elif charge < 0:
        chargestr = '-'
        if charge < -1:
            chargestr += str(-charge)
    else:
        chargestr = ''

    if class_ != '':
        class_ = ':{}'.format(class_)
    fmt = '[{isotope}{name}{stereo}{hcount}{charge}{class_}]'
    return fmt.format(isotope=isotope, name=name, stereo='', hcount=hcountstr,
                      charge=chargestr, class_=class_)


def parse_hcount(hcount_str):
    """
    Parses a SMILES hydrogen count specifications.
    Parameters
    ----------
    hcount_str : str
        The hydrogen count specification to parse.
    Returns
    -------
    int
        The number of hydrogens specified.
    """
    if not hcount_str:
        return 0
    if hcount_str == 'H':
        return 1
    return int(hcount_str[1:])


def parse_charge(charge_str):
    """
    Parses a SMILES charge specification.
    Parameters
    ----------
    charge_str : str
        The charge specification to parse.
    Returns
    -------
    int
        The charge.
    """
    if not charge_str:
        return 0
    signs = {'-': -1, '+': 1}
    sign = signs[charge_str[0]]
    if len(charge_str) > 1 and charge_str[1].isdigit():
        charge = sign * int(charge_str[1:])
    else:
        charge = sign * charge_str.count(charge_str[0])
    return charge


def add_explicit_hydrogens(mol):
    """
    Adds explicit hydrogen nodes to `mol`, the amount is determined by the node
    attribute 'hcount'. Will remove the 'hcount' attribute.
    Parameters
    ----------
    mol : nx.Graph
        The molecule to which explicit hydrogens should be added. Is modified
        in-place.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    h_atom = parse_atom('[H]')
    if 'hcount' in h_atom:
        del h_atom['hcount']
    for i, atom in enumerate(mol.atoms):
        hcount = atom.get('hcount', 0)
        idxs = range(mol.natoms + 1, mol.natoms + hcount + 1)
        # Get the defaults from parse_atom.
        for idx in idxs:
            atom = Atom(f'H-{idx}', **h_atom)
            mol.addAtom(atom)
            
        for jdx in idxs:
            mol.addBondByIndex(i, jdx, order=1)
        if hasattr(atom, 'hcount'):
            delattr(atom, 'hcount')

def remove_explicit_hydrogens(mol):
    """
    Removes all explicit, simple hydrogens from `mol`. Simple means it is
    identical to the SMILES string "[H]", and has exactly one bond. Increments
    'hcount' where appropriate.
    Parameters
    ----------
    mol : nx.Graph
        The molecule whose explicit hydrogens should be removed. Is modified
        in-place.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    to_remove = set()
#    defaults = parse_atom('[H]')
    for atom in mol.atoms:
        neighbors = atom.bondedAtoms
        # TODO: get these defaults from parsing [H]. But do something smart
        #       with the hcount attribute.
        if (atom.get('charge', 0) == 0 and atom.get('element', '') == 'H' and
                'isotope' not in atom and atom.get('class', 0) == 0 and
                len(neighbors) == 1):
            neighbor = neighbors[0]
            if (neighbor.get('element', '') == 'H' or
                    atom.bondto(neighbor).get('order', 1) != 1):
                # The molecule is H2, or the bond order is not 1.
                continue
            to_remove.add(atom)
            neighbor.hcount = neighbor.get('hcount', 0) + 1
    mol.deleteAtoms(to_remove)
    for atom in mol.atoms:
        if 'hcount' not in atom:
            atom.hcount = 0


def fill_valence(mol, respect_hcount=True, respect_bond_order=True,
                 max_bond_order=3):
    """
    Sets the attribute 'hcount' on all nodes in `mol` that don't have it yet.
    The value to which it is set is based on the node's 'element', and the
    number of bonds it has. Default valences are as specified by the global
    variable VALENCES.
    Parameters
    ----------
    mol : nx.Graph
        The molecule whose nodes should get a 'hcount'. Is modified in-place.
    respect_hcount : bool
        If True, don't change the hcount on nodes that already have it set.
    respect_bond_order : bool
        If False, first try to fill the valence by increasing bond orders, and
        add hydrogens after.
    max_bond_order : number
        Only meaningful if respect_bond_order is False. This is the highest
        bond order that will be set.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    if not respect_bond_order:
        increment_bond_orders(mol, max_bond_order=max_bond_order)
    for n_idx in mol:
        node = mol.nodes[n_idx]
        if 'hcount' in node and respect_hcount:
            continue
        missing = max(bonds_missing(mol, n_idx), 0)
        node['hcount'] = node.get('hcount', 0) + missing


def bonds_missing(mol, node_idx, use_order=True):
    """
    Returns how much the specified node is under valence. If use_order is
    False, treat all bonds as if they are order 1.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    node_idx : hashable
        The node to look at. Should be in mol.
    use_order : bool
        If False, treat all bonds as single.
    Returns
    -------
    int
        The number of missing bonds.
    """
    bonds = _bonds(mol, node_idx, use_order)
    bonds += mol.nodes[node_idx].get('hcount', 0)
    valence = _valence(mol, node_idx, bonds)
    return int(valence - bonds)


def _valence(mol, node_idx, minimum=0):
    """
    Returns the valence of the specified node. Since some elements can have
    multiple valences, give the smallest one that is more than `minimum`.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    node_idx : hashable
        The node to look at. Should be in mol.
    minimum : int
        The minimum value of valence.
    Returns
    -------
    int
        The smallest valence of node more than `minimum`.
    """
    element = node_idx.get('element', Element.getBySymbol('*')).symbol
    if element not in VALENCES:
        return 0
    val = VALENCES.get(element)
    try:
        val = min(filter(lambda a: a >= minimum, val))
    except ValueError:  # More bonds than possible
        val = max(val)
    return val


def _bonds(mol, node_idx, use_order=True):
    """
    Returns how many explicit bonds the specified node has. If use_order is
    False, treat all bonds as if they are order 1.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    node_idx : hashable
        The node to look at. Should be in mol.
    use_order : bool
        If False, treat all bonds as single.
    Returns
    -------
    int
        The number of bonds.
    """
    if use_order:
        # bond_orders = map(operator.itemgetter(2),
        #                   mol.edges(nbunch=node_idx, data='order', default=1))
        bond_orders = map(lambda atom: getattr(atom, 'order', 1), node_idx.bondedAtoms)
        bonds = sum(bond_orders)
    else:
        bonds = len(mol[node_idx])
    return bonds


def has_default_h_count(mol, node_idx, use_order=True):
    """
    Returns whether the hydrogen count for this atom is non-standard.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    node_idx : hashable
        The node to look at. Should be in mol.
    use_order : bool
        If False, treat all bonds as single.
    Returns
    -------
    bool
    """
    bonds = _bonds(mol, node_idx, use_order)
    valence = _valence(mol, node_idx, bonds)
    hcount = node_idx.get('hcount', 0)
    return valence - bonds == hcount


def _hydrogen_neighbours(mol, atom):
    bondedAtoms = atom.bondedAtoms
    h_bondedAtoms = 0
    for btom in bondedAtoms:
        if (btom.get('element', '*') == 'H' and
                atom.bondto(btom).get('order', 1) == 1):
            h_bondedAtoms += 1
    return h_bondedAtoms


def mark_aromatic_atoms(mol, atoms=None):
    """
    Sets the 'aromatic' attribute for all nodes in `mol`. Requires that
    the 'hcount' on atoms is correct.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    atoms: collections.abc.Iterable
        The atoms to act on. Will still analyse the full molecule.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    if atoms is None:
        atoms = set(mol.atoms)
    aromatic = set()
    # Only cycles can be aromatic
    for cycle in mol.getBasisCycles():
        # All atoms should be sp2, so each contributes an electron. We make
        # sure they are later.
        electrons = len(cycle)
        maybe_aromatic = True

        for atom in cycle:
            element = atom.get('element', '*')  #TODO: 
            hcount = atom.get('hcount', 0)
            degree = len(atom.bonds) + hcount
            hcount += _hydrogen_neighbours(mol, atom)
            # Make sure they are possibly aromatic, and are sp2 hybridized
            if element not in AROMATIC_ATOMS or degree not in (2, 3):
                maybe_aromatic = False
                break
            # Some of the special cases per group. N and O type atoms can
            # donate an additional electron from a lone pair.
            # missing cases:
            #   extracyclic sp2 heteroatom (e.g. =O)
            #   some charged cases
            if element in 'N P As'.split() and hcount == 1:
                electrons += 1
            elif element in 'O S Se'.split():
                electrons += 1
            if atom.get('charge', 0) == +1 and not (element == 'C' and hcount == 0):
                electrons -= 1
        if maybe_aromatic and int(electrons) % 2 == 0:
            # definitely (anti) aromatic
            aromatic.update(cycle)
    for atom in atoms:
        if atom not in aromatic:
            atom.aromatic = False
        else:
            atom.aromatic = True


def mark_aromatic_edges(mol):
    """
    Set all bonds between aromatic atoms (attribute 'aromatic' is `True`) to
    1.5. Gives all other bonds that don't have an order yet an order of 1.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    for cycle in mol.getBasisCycles():
        for bond in mol.getSubGroup('', cycle).bonds:
            if bond.atom1 not in cycle or bond.atom2 not in cycle:
                continue
            if (bond.atom1.get('aromatic', False)
                    and bond.atom2.get('aromatic', False)):
                bond.order = 1.5
    for bond in mol.getBonds():
        if hasattr(bond, 'order'):
            bond.order = 1


def correct_aromatic_rings(mol):
    """
    Sets hcount for all atoms, marks aromaticity for all atoms, and the order of
    all aromatic bonds to 1.5.
    Parameters
    ----------
    mol : nx.Graph
        The molecule.
    Returns
    -------
    None
        `mol` is modified in-place.
    """
    fill_valence(mol)
    mark_aromatic_atoms(mol)
    mark_aromatic_edges(mol)


def increment_bond_orders(molecule, max_bond_order=3):
    """
    Increments bond orders up to what the atom's valence allows.
    Parameters
    ----------
    molecule : nx.Graph
        The molecule to process.
    max_bond_order : number
        The highest bond order allowed to make.
    Returns
    -------
    None
        molecule is modified in-place.
    """
    # Gather the number of open spots for all atoms beforehand, since some
    # might have multiple oxidation states (e.g. S). We don't want to change
    # oxidation state halfway through for some funny reason. It shouldn't be
    # nescessary, but it can't hurt.
    missing_bonds = {}
    for idx in molecule:
        missing_bonds[idx] = max(bonds_missing(molecule, idx), 0)

    for idx, jdx in molecule.edges:
        missing_idx = missing_bonds[idx]
        missing_jdx = missing_bonds[jdx]
        edge_missing = min(missing_idx, missing_jdx)
        current_order = molecule.edges[idx, jdx].get("order", 1)
        if current_order == 1.5:
            continue
        new_order = edge_missing + current_order
        new_order = min(new_order, max_bond_order)
        molecule.edges[idx, jdx]['order'] = new_order
        missing_bonds[idx] -= edge_missing
        missing_bonds[jdx] -= edge_missing



def _get_ring_marker(used_markers):
    """
    Returns the lowest number larger than 0 that is not in `used_markers`.
    Parameters
    ----------
    used_markers : Container
        The numbers that can't be used.
    Returns
    -------
    int
        The lowest number larger than 0 that's not in `used_markers`.
    """
    new_marker = 1
    while new_marker in used_markers:
        new_marker += 1
    return new_marker


def _write_edge_symbol(bond):
    """
    Determines whether a symbol should be written for the edge between `n_idx`
    and `n_jdx` in `mol`. It should not be written if it's a bond of order
    1 or an aromatic bond between two aromatic atoms; unless it's a single bond
    between two aromatic atoms.
    Parameters
    ----------
    mol : nx.Graph
        The mol.
    n_idx : Hashable
        The first node key describing the edge.
    n_jdx : Hashable
        The second node key describing the edge.
    Returns
    -------
    bool
        Whether an explicit symbol is needed for this edge.
    """
    order = bond.get('order', 1)
    aromatic_atoms = bond.atom1.get('element', Element.getBySymbol('*')).symbol.islower() and\
                     bond.atom2.get('element', Element.getBySymbol('*')).symbol.islower()
    aromatic_bond = aromatic_atoms and order == 1.5
    cross_aromatic = aromatic_atoms and order == 1
    single_bond = order == 1
    return cross_aromatic or not (aromatic_bond or single_bond)


def write_smiles(mol, default_element=Element.getBySymbol('*'), start=None):
    """
    Creates a SMILES string describing `molecule` according to the OpenSMILES
    standard.
    Parameters
    ----------
    molecule : nx.Graph
        The molecule for which a SMILES string should be generated.
    default_element : str
        The element to write if the attribute is missing for a node.
    start : Hashable
        The atom at which the depth first traversal of the molecule should
        start. A sensible one is chosen: preferably a terminal heteroatom.
    Returns
    -------
    str
        The SMILES string describing `molecule`.
    """
    remove_explicit_hydrogens(mol)

    if start is None:
        # Start at a terminal atom, and if possible, a heteroatom.
        def keyfunc(atom):
            """Key function for finding the node at which to start."""
            return (mol.getDegreeOfAtom(atom),
                    # True > False
                    atom.get('element', default_element) == 'C',
                    atom)
        start = min(mol.atoms, key=keyfunc)


    order_to_symbol = {0: '.', 1: '-', 1.5: ':', 2: '=', 3: '#', 4: '$'}

    dfs_successors = molpy.dfs_successors(mol, source=start)

    predecessors = defaultdict(list)
    for node_key, successors in dfs_successors.items():
        for successor in successors:
            predecessors[successor].append(node_key)
    predecessors = dict(predecessors)
    # We need to figure out which edges we won't cross when doing the dfs.
    # These are the edges we'll need to add to the smiles using ring markers.
    edges = set()
    for n_idx, n_jdxs in dfs_successors.items():
        for n_jdx in n_jdxs:
            edges.add(frozenset((n_idx, n_jdx)))
    total_edges = frozenset(mol.bonds)
    ring_edges = total_edges - edges

    atom_to_ring_idx = defaultdict(list)
    ring_idx_to_bond = {}
    ring_idx_to_marker = {}
    for ring_idx, bond in enumerate(ring_edges, 1):
        atom_to_ring_idx[bond.atom1].append(ring_idx)
        atom_to_ring_idx[bond.atom2].append(ring_idx)
        ring_idx_to_bond[ring_idx] = bond

    branch_depth = 0
    branches = set()
    to_visit = [start]
    smiles = ''

    while to_visit:
        current = to_visit.pop()
        if current in branches:
            branch_depth += 1
            smiles += '('
            branches.remove(current)

        if current in predecessors:
            # It's not the first atom we're visiting, so we want to see if the
            # edge we last crossed to get here is interesting.
            previous = predecessors[current]
            assert len(previous) == 1
            previous = previous[0]
            bond = previous.bondto(current)
            if _write_edge_symbol(bond):
                order = bond.get('order', 1)
                smiles += order_to_symbol[order]
        smiles += format_atom(mol, current, default_element)
        if current in atom_to_ring_idx:
            # We're going to need to write a ring number
            ring_idxs = atom_to_ring_idx[current]
            for ring_idx in ring_idxs:
                ring_bond = ring_idx_to_bond[ring_idx]  # bond
                if ring_idx not in ring_idx_to_marker:
                    marker = _get_ring_marker(ring_idx_to_marker.values())
                    ring_idx_to_marker[ring_idx] = marker
                    new_marker = True
                else:
                    marker = ring_idx_to_marker.pop(ring_idx)
                    new_marker = False

                if _write_edge_symbol(ring_bond) and new_marker:
                    order = ring_bond.get('order', 1)
                    smiles += order_to_symbol[order]
                smiles += str(marker) if marker < 10 else '%{}'.format(marker)

        if current in dfs_successors:
            # Proceed to the next node in this branch
            next_nodes = dfs_successors[current]
            # ... and if needed, remember to return here later
            branches.update(next_nodes[1:])
            to_visit.extend(next_nodes)
        elif branch_depth:
            # We're finished with this branch.
            smiles += ')'
            branch_depth -= 1

    smiles += ')' * branch_depth
    return smiles