from molpy.element import Element
from molpy.typifier.smarts import SmartsParser
from molpy.typifier.smarts_graph import SMARTSGraph
from warnings import warn


def _check_independent_residues(structure):
    return False


class SmartsTypifier:

    def __init__(self, forcefield):

        self.forcefield = forcefield
        self.non_element_types = list()
        self.parser = SmartsParser(self.non_element_types)

    def typify(self, structure, use_residue_map=False, max_iter=10):

        typemap = self._typify_atom(
            structure, use_residue_map=use_residue_map, max_iter=max_iter
        )

        # self._apply_typemap(structure, typemap)
        for i, atom in enumerate(structure["atoms"]):
            atom["type"] = typemap[i]["atomtype"]

        if not all([atom["type"] for a in structure["atoms"]]):
            raise ValueError("Not all atoms in topology have atom types")

        return structure

    def _typify_atom(self, structure, use_residue_map=True, max_iter=10):
        if use_residue_map:
            independent_residues = _check_independent_residues(structure)

            if independent_residues:
                residue_map = dict()

                # Need to call this only once and store results for later id() comparisons
                # for res_id, res in enumerate(structure.residues):
                #     if structure.residues[res_id].name not in residue_map.keys():
                #         tmp_res = _structure_from_residue(res, structure)
                #         typemap = find_atomtypes(tmp_res, forcefield=self)
                #         residue_map[res.name] = typemap

                # typemap = _unwrap_typemap(structure, residue_map)
                raise NotImplementedError("Residue mapping is not implemented yet.")

            else:
                typemap = self._find_atomtypes(structure, max_iter)

        else:
            typemap = self._find_atomtypes(structure, max_iter)

        return typemap

    def _find_atomtypes(self, structure, max_iter=10):
        """Determine atomtypes for all atoms.

        Parameters
        ----------
        structure : parmed.Structure, or gmso.Topology, or TopologyGraph
            The topology that we are trying to atomtype. If a parmed.Structure or
            gmso.Topology is provided, it will be converted to a TopologyGraph before
            atomtyping.
        forcefield : AtomTypingRulesProvider, foyer.ForceField
            The atomtyping rules provider object/foyer forcefield.
        max_iter : int, optional, default=10
            The maximum number of iterations.
        """

        # TODO: create temporary one or inplace
        # atomtype_rules = self.forcefield  # TODO: extract new object?

        typemap = {
            atom_index: {"whitelist": set(), "blacklist": set(), "atomtype": None}
            for atom_index, _ in enumerate(structure["atoms"])
        }

        # _load_rules
        # For every SMARTS string in the force field,
        # create a SMARTSGraph object
        rules = dict()
        for atomtype in self.forcefield.get_atomtypes():
            name = atomtype.name
            overrides = atomtype.get("overrides")
            smarts = atomtype.get("def", None)

            if not smarts:  # We want to skip over empty smarts definitions
                continue
            if overrides is not None:
                overrides = set(overrides)
            else:
                overrides = set()
            rules[name] = SMARTSGraph(
                smarts_string=smarts,
                parser=self.parser,
                name=name,
                overrides=overrides,
                typemap=typemap,
            )

        # Only consider rules for elements found in topology
        subrules = dict()

        system_elements = set()
        for atom in structure["atoms"]:
            # First add non-element types, which are strings, then elements
            name = atom["name"]
            if name.startswith("_"):
                system_elements.add(name)
            else:
                atomic_number = atom.get("number", None)
                atomic_symbol = atom.get("element", None)
                elem = Element(atomic_number or atomic_symbol)

                assert (
                    atomic_number is not None or atomic_symbol is not None
                ), f"Atom {atom} does not have an atomic number or symbol"
                system_elements.add(elem.symbol)

        for name, smartgraph in rules.items():
            atom = smartgraph.vs[0]["atom"]
            if len(list(atom.find_data("atom_symbol"))) == 1 and not list(
                atom.find_data("not_expression")
            ):
                try:
                    element = next(atom.find_data("atom_symbol")).children[0]
                except IndexError:
                    try:
                        atomic_num = next(atom.find_data("atomic_num")).children[0]
                        element = Element(atomic_num).symbol
                    except IndexError:
                        element = None
            else:
                element = None
            if element is None or element in system_elements:
                subrules[name] = smartgraph
        rules = subrules

        # _iterate_rules(rules, structure, typemap, max_iter=max_iter)
        for _ in range(max_iter):
            max_iter -= 1
            found_something = False
            for rule in rules.values():
                print(rule.name)
                for match_index in rule.find_matches(structure, typemap):
                    atom = typemap[match_index]
                    # This conditional is not strictly necessary, but it prevents
                    # redundant set addition on later iterations
                    if rule.name not in atom["whitelist"]:
                        atom["whitelist"].add(rule.name)
                        atom["blacklist"] |= rule.overrides
                        print('!!! changed !!!')
                        found_something = True
            if not found_something:
                break
        else:
            warn("Reached maximum iterations. Something probably went wrong.")

        # _resolve_atomtypes(structure, typemap)
        """Determine the final atomtypes from the white- and blacklists."""
        atoms = {atom_idx: data for atom_idx, data in enumerate(structure["atoms"])}
        for atom_id, atom in typemap.items():
            atomtype = [
                rule_name for rule_name in atom["whitelist"] - atom["blacklist"]
            ]
            if len(atomtype) == 1:
                atom["atomtype"] = atomtype[0]
            elif len(atomtype) > 1:
                raise ValueError(
                    "Found multiple types for atom {} ({}): {}.".format(
                        atom_id, atoms[atom_id].atomic_number, atomtype
                    )
                )
            else:
                raise ValueError(
                    "Found no types for atom numbered {} which is atomic number {}. Forcefield file is missing this atomtype, so try to add SMARTS definitions to account for this atom.".format(
                        atom_id, atoms[atom_id]["number"]
                    )
                )
        return typemap
