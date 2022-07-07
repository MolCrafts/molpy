# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-06-25
# version: 0.0.1

from typing import Dict, List
import numpy as np
from molpy.atoms import Atoms
from molpy.io.fileHandler import FileHandler
from molpy.io.base import TrajReader, DataReader
from math import pi
import sys
import warnings
from collections import OrderedDict
PI = pi

def _parse_atom_index(index):
    """Parse the string containing an atom index, which might be either decimal or hex."""
    try:
        return int(index)
    except:
        return int(index, 16) - 0xA0000 + 100000

class PdbStructure:

    """
    PdbStructure object holds a parsed Protein Data Bank format file.
    Examples:
    Load a pdb structure from a file:
    > pdb = PdbStructure(open("1ARJ.pdb"))
    Fetch the first atom of the structure:
    > print pdb.iter_atoms().next()
    ATOM      1  O5'   G N  17      13.768  -8.431  11.865  1.00  0.00           O
    Loop over all of the atoms of the structure
    > for atom in pdb.iter_atoms():
    >     print atom
    ATOM      1  O5'   G N  17      13.768  -8.431  11.865  1.00  0.00           O
    ...
    Get a list of all atoms in the structure:
    > atoms = list(pdb.iter_atoms())
    also:
    residues = list(pdb.iter_residues())
    positions = list(pdb.iter_positions())
    chains = list(pdb.iter_chains())
    models = list(pdb.iter_models())
    Fetch atomic coordinates of first atom:
    > print pdb.iter_positions().next()
    [13.768, -8.431, 11.865] A
     or
    > print pdb.iter_atoms().next().position
    [13.768, -8.431, 11.865] A
    Strip the length units from an atomic position:
    > import openmm.unit
    > pos = pdb.iter_positions().next()
    > print pos
    [13.768, -8.431, 11.865] A
    > print pos / openmm.unit.angstroms
    [13.768, -8.431, 11.865]
    > print pos / openmm.unit.nanometers
    [1.3768, -0.8431, 1.1865]
    The hierarchical structure of the parsed PDB structure is as follows:
    PdbStructure
      Model
        Chain
          Residue
            Atom
              Location
    Model - A PDB structure consists of one or more Models.  Each model corresponds to one version of
    an NMR structure, or to one frame of a molecular dynamics trajectory.
    Chain - A Model contains one or more Chains.  Each chain corresponds to one molecule, although multiple
    water molecules are frequently included in the same chain.
    Residue - A Chain contains one or more Residues.  One Residue corresponds to one of the repeating
    unit that constitutes a polymer such as protein or DNA.  For non-polymeric molecules, one Residue
    represents one molecule.
    Atom - A Residue contains one or more Atoms.  Atoms are chemical atoms.
    Location - An atom can sometimes have more that one position, due to static disorder in X-ray
    crystal structures.  To see all of the atom positions, use the atom.iter_positions() method,
    or pass the parameter "include_alt_loc=True" to one of the other iter_positions() methods.
    > for pos in pdb.iter_positions(include_alt_loc=True):
    >   ...
    Will loop over all atom positions, including multiple alternate locations for atoms that have
    multiple positions.  The default value of include_alt_loc is False for the iter_positions()
    methods.
    """


    def __init__(self, input_stream, load_all_models=False, extraParticleIdentifier='EP'):
        """Create a PDB model from a PDB file stream.
        Parameters
        ----------
        self : PdbStructure
            The new object that is created.
        input_stream : stream
            An input file stream, probably created with open().
        load_all_models : bool
            Whether to load every model of an NMR structure or trajectory, or
            just load the first model, to save memory.
        extraParticleIdentifier : string='EP'
            if this value appears in the element column for an ATOM record, the Atom's element will be set to 'EP' to mark it as an extra particle
        """
        # initialize models
        self.load_all_models = load_all_models
        self.extraParticleIdentifier = extraParticleIdentifier
        self.models = []
        self._current_model = None
        self.default_model = None
        self.models_by_number = {}
        self._periodic_box_vectors = None
        self.sequences = []
        self.modified_residues = []
        # read file
        self._load(input_stream)

    def _load(self, input_stream):
        self._reset_atom_numbers()
        self._reset_residue_numbers()
        # Read one line at a time
        for pdb_line in input_stream:
            if not isinstance(pdb_line, str):
                pdb_line = pdb_line.decode('utf-8')
            command = pdb_line[:6]
            # Look for atoms
            if command == "ATOM  " or command == "HETATM":
                self._add_atom(Atom(pdb_line, self, self.extraParticleIdentifier))
            elif command == "CONECT":
                atoms = [_parse_atom_index(pdb_line[6:11])]
                for pos in (11,16,21,26):
                    try:
                        atoms.append(_parse_atom_index(pdb_line[pos:pos+5]))
                    except:
                        pass
                self._current_model.connects.append(atoms)
            # Notice MODEL punctuation, for the next level of detail
            # in the structure->model->chain->residue->atom->position hierarchy
            elif pdb_line[:5] == "MODEL":
                model_number = int(pdb_line[10:14])
                self._add_model(Model(model_number))
                self._reset_atom_numbers()
                self._reset_residue_numbers()
            elif command == "ENDMDL":
                self._current_model._finalize()
                if not self.load_all_models:
                    break
            elif pdb_line[:3] == "END":
                self._current_model._finalize()
                if not self.load_all_models:
                    break
            elif pdb_line[:3] == "TER" and pdb_line.split()[0] == "TER":
                self._current_model._current_chain._add_ter_record()
                self._reset_residue_numbers()
            elif command == "CRYST1":
                a_length = float(pdb_line[6:15])*0.1
                b_length = float(pdb_line[15:24])*0.1
                c_length = float(pdb_line[24:33])*0.1
                alpha = float(pdb_line[33:40])*PI/180.0
                beta = float(pdb_line[40:47])*PI/180.0
                gamma = float(pdb_line[47:54])*PI/180.0
                if 0 not in (a_length, b_length, c_length):
                    self._periodic_box_vectors = computePeriodicBoxVectors(a_length, b_length, c_length, alpha, beta, gamma)
            elif command == "SEQRES":
                chain_id = pdb_line[11]
                if len(self.sequences) == 0 or chain_id != self.sequences[-1].chain_id:
                    self.sequences.append(Sequence(chain_id))
                self.sequences[-1].residues.extend(pdb_line[19:].split())
            elif command == "MODRES":
                self.modified_residues.append(ModifiedResidue(pdb_line[16], int(pdb_line[18:22]), pdb_line[12:15].strip(), pdb_line[24:27].strip()))
        self._finalize()

    def _reset_atom_numbers(self):
        self._next_atom_number = 1

    def _reset_residue_numbers(self):
        self._next_residue_number = 1

    def write(self, output_stream=sys.stdout):
        """Write out structure in PDB format"""
        for model in self.models:
            if len(model.chains) == 0:
                continue
            if len(self.models) > 1:
                print("MODEL     %4d" % (model.number), file=output_stream)
            model.write(output_stream)
            if len(self.models) > 1:
                print("ENDMDL", file=output_stream)
        print("END", file=output_stream)

    def _add_model(self, model):
        if self.default_model is None:
            self.default_model = model
        self.models.append(model)
        self._current_model = model
        if model.number not in self.models_by_number:
            self.models_by_number[model.number] = model

    def get_model(self, model_number):
        return self.models_by_number[model_number]

    def model_numbers(self):
        return self.models_by_number.keys()

    def __contains__(self, model_number):
        return self.models_by_number.__contains__(model_number)

    def __getitem__(self, model_number):
        return self.models_by_number[model_number]

    def __iter__(self):
        for model in self.models:
                yield model

    def iter_models(self, use_all_models=False):
        if use_all_models:
            for model in self:
                yield model
        elif len(self.models) > 0:
            yield self.models[0]

    def iter_chains(self, use_all_models=False):
        for model in self.iter_models(use_all_models):
            for chain in model.iter_chains():
                yield chain

    def iter_residues(self, use_all_models=False):
        for model in self.iter_models(use_all_models):
            for res in model.iter_residues():
                yield res

    def iter_atoms(self, use_all_models=False):
        for model in self.iter_models(use_all_models):
            for atom in model.iter_atoms():
                yield atom

    def iter_positions(self, use_all_models=False, include_alt_loc=False):
        """
        Iterate over atomic positions.
        Parameters
        ----------
        use_all_models : bool=False
            Get positions from all models or just the first one.
        include_alt_loc : bool=False
            Get all positions for each atom, or just the first one.
        """
        for model in self.iter_models(use_all_models):
            for loc in model.iter_positions(include_alt_loc):
                yield loc

    def __len__(self):
        return len(self.models)

    def _add_atom(self, atom):
        """
        """
        if self._current_model is None:
            self._add_model(Model(0))
        atom.model_number = self._current_model.number
        # Atom might be alternate position for existing atom
        self._current_model._add_atom(atom)

    def _finalize(self):
        """Establish first and last residues, atoms, etc."""
        for model in self.models:
            model._finalize()

    def get_periodic_box_vectors(self):
        """Get the vectors defining the crystallographic unit cell (may be None)."""
        return self._periodic_box_vectors


class Sequence(object):
    """Sequence holds the sequence of a chain, as specified by SEQRES records."""
    def __init__(self, chain_id):
        self.chain_id = chain_id
        self.residues = []

class ModifiedResidue(object):
    """ModifiedResidue holds information about a modified residue, as specified by a MODRES record."""
    def __init__(self, chain_id, number, residue_name, standard_name):
        self.chain_id = chain_id
        self.number = number
        self.residue_name = residue_name
        self.standard_name = standard_name


class Model(object):
    """Model holds one model of a PDB structure.
    NMR structures usually have multiple models.  This represents one
    of them.
    """
    def __init__(self, model_number=1):
        self.number = model_number
        self.chains = []
        self._current_chain = None
        self.chains_by_id = {}
        self.connects = []

    def _add_atom(self, atom):
        """
        """
        if len(self.chains) == 0:
            self._add_chain(Chain(atom.chain_id))
        # Create a new chain if the chain id has changed
        if self._current_chain.chain_id != atom.chain_id:
            self._add_chain(Chain(atom.chain_id))
        # Create a new chain after TER record, even if ID is the same
        elif self._current_chain.has_ter_record:
            self._add_chain(Chain(atom.chain_id))
        self._current_chain._add_atom(atom)

    def _add_chain(self, chain):
        self.chains.append(chain)
        self._current_chain = chain
        if not chain.chain_id in self.chains_by_id:
            self.chains_by_id[chain.chain_id] = chain

    def get_chain(self, chain_id):
        return self.chains_by_id[chain_id]

    def chain_ids(self):
        return self.chains_by_id.keys()

    def __contains__(self, chain_id):
        return self.chains_by_id.__contains__(chain_id)

    def __getitem__(self, chain_id):
        return self.chains_by_id[chain_id]

    def __iter__(self):
        return iter(self.chains)

    def iter_chains(self):
        for chain in self:
            yield chain

    def iter_residues(self):
        for chain in self:
            for res in chain.iter_residues():
                yield res

    def iter_atoms(self):
        for chain in self:
            for atom in chain.iter_atoms():
                yield atom

    def iter_positions(self, include_alt_loc=False):
        for chain in self:
            for loc in chain.iter_positions(include_alt_loc):
                yield loc

    def __len__(self):
        return len(self.chains)

    def write(self, output_stream=sys.stdout):
        # Start atom serial numbers at 1
        sn = Model.AtomSerialNumber(1)
        for chain in self.chains:
            chain.write(sn, output_stream)

    def _finalize(self):
        for chain in self.chains:
            chain._finalize()


    class AtomSerialNumber(object):
        """pdb.Model inner class for pass-by-reference incrementable serial number"""
        def __init__(self, val):
            self.val = val

        def increment(self):
            self.val += 1


class Chain(object):
    def __init__(self, chain_id=' '):
        self.chain_id = chain_id
        self.residues = []
        self.has_ter_record = False
        self._current_residue = None
        self.residues_by_num_icode = {}
        self.residues_by_number = {}

    def _add_atom(self, atom):
        """
        """
        # Create a residue if none have been created
        if len(self.residues) == 0:
            self._add_residue(Residue(atom.residue_name_with_spaces, atom.residue_number, atom.insertion_code, atom.alternate_location_indicator))
        # Create a residue if the residue information has changed
        elif self._current_residue.number != atom.residue_number:
            self._add_residue(Residue(atom.residue_name_with_spaces, atom.residue_number, atom.insertion_code, atom.alternate_location_indicator))
        elif self._current_residue.insertion_code != atom.insertion_code:
            self._add_residue(Residue(atom.residue_name_with_spaces, atom.residue_number, atom.insertion_code, atom.alternate_location_indicator))
        elif self._current_residue.name_with_spaces == atom.residue_name_with_spaces:
            # This is a normal case: number, name, and iCode have not changed
            pass
        elif atom.alternate_location_indicator != ' ':
            # OK - this is a point mutation, Residue._add_atom will know what to do
            pass
        else: # Residue name does not match
            # Only residue name does not match
            warnings.warn("WARNING: two consecutive residues with same number (%s, %s)" % (atom, self._current_residue.atoms[-1]))
            self._add_residue(Residue(atom.residue_name_with_spaces, atom.residue_number, atom.insertion_code, atom.alternate_location_indicator))
        self._current_residue._add_atom(atom)

    def _add_residue(self, residue):
        if len(self.residues) == 0:
            residue.is_first_in_chain = True
        self.residues.append(residue)
        self._current_residue = residue
        key = str(residue.number) + residue.insertion_code
        # only store the first residue with a particular key
        if key not in self.residues_by_num_icode:
            self.residues_by_num_icode[key] = residue
        if residue.number not in self.residues_by_number:
            self.residues_by_number[residue.number] = residue

    def write(self, next_serial_number, output_stream=sys.stdout):
        for residue in self.residues:
            residue.write(next_serial_number, output_stream)
        if self.has_ter_record:
            r = self.residues[-1]
            print("TER   %5d      %3s %1s%4d%1s" % (next_serial_number.val, r.name_with_spaces, self.chain_id, r.number, r.insertion_code), file=output_stream)
            next_serial_number.increment()

    def _add_ter_record(self):
        self.has_ter_record = True
        self._finalize()

    def get_residue(self, residue_number, insertion_code=' '):
        return self.residues_by_num_icode[str(residue_number) + insertion_code]

    def __contains__(self, residue_number):
        return self.residues_by_number.__contains__(residue_number)

    def __getitem__(self, residue_number):
        """Returns the FIRST residue in this chain with a particular residue number"""
        return self.residues_by_number[residue_number]

    def __iter__(self):
        for res in self.residues:
            yield res

    def iter_residues(self):
        for res in self:
            yield res

    def iter_atoms(self):
        for res in self:
            for atom in res:
                yield atom;

    def iter_positions(self, include_alt_loc=False):
        for res in self:
            for loc in res.iter_positions(include_alt_loc):
                yield loc

    def __len__(self):
        return len(self.residues)

    def _finalize(self):
        self.residues[0].is_first_in_chain = True
        self.residues[-1].is_final_in_chain = True
        for residue in self.residues:
            residue._finalize()


class Residue(object):
    def __init__(self, name, number, insertion_code=' ', primary_alternate_location_indicator=' '):
        alt_loc = primary_alternate_location_indicator
        self.primary_location_id = alt_loc
        self.locations = {}
        self.locations[alt_loc] = Residue.Location(alt_loc, name)
        self.name_with_spaces = name
        self.number = number
        self.insertion_code = insertion_code
        self.atoms = []
        self.atoms_by_name = OrderedDict()
        self.is_first_in_chain = False
        self.is_final_in_chain = False
        self._current_atom = None

    def _add_atom(self, atom):
        """
        """
        alt_loc = atom.alternate_location_indicator
        if alt_loc not in self.locations:
            self.locations[alt_loc] = Residue.Location(alt_loc, atom.residue_name_with_spaces)
        assert atom.residue_number == self.number
        assert atom.insertion_code == self.insertion_code
        # Check whether this is an existing atom with another position
        if (atom.name_with_spaces in self.atoms_by_name):
            old_atom = self.atoms_by_name[atom.name_with_spaces]
            # Unless this is a duplicated atom (warn about file error)
            if atom.alternate_location_indicator in old_atom.locations:
                warnings.warn("WARNING: duplicate atom (%s, %s)" % (atom, old_atom._pdb_string(old_atom.serial_number, atom.alternate_location_indicator)))
            else:
                for alt_loc, position in atom.locations.items():
                    old_atom.locations[alt_loc] = position
                return # no new atom added
        # actually use new atom
        self.atoms_by_name[atom.name] = atom
        self.atoms_by_name[atom.name_with_spaces] = atom
        self.atoms.append(atom)
        self._current_atom = atom

    def write(self, next_serial_number, output_stream=sys.stdout, alt_loc = "*"):
        for atom in self.atoms:
            atom.write(next_serial_number, output_stream, alt_loc)

    def _finalize(self):
        if len(self.atoms) > 0:
            self.atoms[0].is_first_atom_in_chain = self.is_first_in_chain
            self.atoms[-1].is_final_atom_in_chain = self.is_final_in_chain
            for atom in self.atoms:
                atom.is_first_residue_in_chain = self.is_first_in_chain
                atom.is_final_residue_in_chain = self.is_final_in_chain

    def set_name_with_spaces(self, name, alt_loc=None):
        # Gromacs ffamber PDB files can have 4-character residue names
        # assert len(name) == 3
        if alt_loc is None:
            alt_loc = self.primary_location_id
        loc = self.locations[alt_loc]
        loc.name_with_spaces = name
        loc.name = name.strip()
    def get_name_with_spaces(self, alt_loc=None):
        if alt_loc is None:
            alt_loc = self.primary_location_id
        loc = self.locations[alt_loc]
        return loc.name_with_spaces
    name_with_spaces = property(get_name_with_spaces, set_name_with_spaces, doc='four-character residue name including spaces')

    def get_name(self, alt_loc=None):
        if alt_loc is None:
            alt_loc = self.primary_location_id
        loc = self.locations[alt_loc]
        return loc.name
    name = property(get_name, doc='residue name')

    def get_atom(self, atom_name):
        return self.atoms_by_name[atom_name]

    def __contains__(self, atom_name):
        return self.atoms_by_name.__contains__(atom_name)

    def __getitem__(self, atom_name):
        """Returns the FIRST atom in this residue with a particular atom name"""
        return self.atoms_by_name[atom_name]

    def __iter__(self):
        """
        >>> pdb_lines = [ \
                "ATOM    188  N   CYS A  42      40.714  -5.292  12.123  1.00 11.29           N",\
                "ATOM    189  CA  CYS A  42      39.736  -5.883  12.911  1.00 10.01           C",\
                "ATOM    190  C   CYS A  42      40.339  -6.654  14.087  1.00 22.28           C",\
                "ATOM    191  O   CYS A  42      41.181  -7.530  13.859  1.00 13.70           O",\
                "ATOM    192  CB  CYS A  42      38.949  -6.825  12.002  1.00  9.67           C",\
                "ATOM    193  SG  CYS A  42      37.557  -7.514  12.922  1.00 20.12           S"]
        >>> res = Residue("CYS", 42)
        >>> for l in pdb_lines:
        ...     res._add_atom(Atom(l))
        ...
        >>> for atom in res:
        ...     print atom
        ATOM    188  N   CYS A  42      40.714  -5.292  12.123  1.00 11.29           N
        ATOM    189  CA  CYS A  42      39.736  -5.883  12.911  1.00 10.01           C
        ATOM    190  C   CYS A  42      40.339  -6.654  14.087  1.00 22.28           C
        ATOM    191  O   CYS A  42      41.181  -7.530  13.859  1.00 13.70           O
        ATOM    192  CB  CYS A  42      38.949  -6.825  12.002  1.00  9.67           C
        ATOM    193  SG  CYS A  42      37.557  -7.514  12.922  1.00 20.12           S
        """
        for atom in self.iter_atoms():
            yield atom

    # Three possibilities: primary alt_loc, certain alt_loc, or all alt_locs
    def iter_atoms(self, alt_loc=None):
        if alt_loc is None:
            locs = [self.primary_location_id]
        elif alt_loc == "":
            locs = [self.primary_location_id]
        elif alt_loc == "*":
            locs = None
        else:
            locs = list(alt_loc)
        # If an atom has any location in alt_loc, emit the atom
        for atom in self.atoms:
            use_atom = False # start pessimistic
            for loc2 in atom.locations.keys():
                # print "#%s#%s" % (loc2,locs)
                if locs is None: # means all locations
                    use_atom = True
                elif loc2 in locs:
                    use_atom = True
            if use_atom:
                yield atom

    def iter_positions(self, include_alt_loc=False):
        """
        Returns one position per atom, even if an individual atom has multiple positions.
        >>> pdb_lines = [ \
                         "ATOM    188  N   CYS A  42      40.714  -5.292  12.123  1.00 11.29           N",\
                         "ATOM    189  CA  CYS A  42      39.736  -5.883  12.911  1.00 10.01           C",\
                         "ATOM    190  C   CYS A  42      40.339  -6.654  14.087  1.00 22.28           C",\
                         "ATOM    191  O   CYS A  42      41.181  -7.530  13.859  1.00 13.70           O",\
                         "ATOM    192  CB  CYS A  42      38.949  -6.825  12.002  1.00  9.67           C",\
                         "ATOM    193  SG  CYS A  42      37.557  -7.514  12.922  1.00 20.12           S"]
        >>> res = Residue("CYS", 42)
        >>> for l in pdb_lines: res._add_atom(Atom(l))
        >>> for c in res.iter_positions:
        ...     print c
        Traceback (most recent call last):
          File "<stdin>", line 1, in <module>
        TypeError: 'instancemethod' object is not iterable
        >>> for c in res.iter_positions():
        ...     print c
        [40.714, -5.292, 12.123] A
        [39.736, -5.883, 12.911] A
        [40.339, -6.654, 14.087] A
        [41.181, -7.53, 13.859] A
        [38.949, -6.825, 12.002] A
        [37.557, -7.514, 12.922] A
        """
        for atom in self:
            if include_alt_loc:
                for loc in atom.iter_positions():
                    yield loc
            else:
                yield atom.position

    def __len__(self):
        return len(self.atoms)

    # Residues can have multiple locations, based on alt_loc indicator
    class Location:
        """
        Inner class of residue to allow different residue names for different alternate_locations.
        """
        def __init__(self, alternate_location_indicator, residue_name_with_spaces):
            self.alternate_location_indicator = alternate_location_indicator
            self.residue_name_with_spaces = residue_name_with_spaces


class Atom(object):
    """Atom represents one atom in a PDB structure.
    """
    def __init__(self, pdb_line, pdbstructure=None, extraParticleIdentifier='EP'):
        """Create a new pdb.Atom from an ATOM or HETATM line.
        Example line:
        ATOM   2209  CB  TYR A 299       6.167  22.607  20.046  1.00  8.12           C
        00000000011111111112222222222333333333344444444445555555555666666666677777777778
        12345678901234567890123456789012345678901234567890123456789012345678901234567890
        ATOM line format description from
          http://deposit.rcsb.org/adit/docs/pdb_atom_format.html:
        COLUMNS        DATA TYPE       CONTENTS
        --------------------------------------------------------------------------------
         1 -  6        Record name     "ATOM  "
         7 - 11        Integer         Atom serial number.
        13 - 16        Atom            Atom name.
        17             Character       Alternate location indicator.
        18 - 20        Residue name    Residue name.
        22             Character       Chain identifier.
        23 - 26        Integer         Residue sequence number.
        27             AChar           Code for insertion of residues.
        31 - 38        Real(8.3)       Orthogonal coordinates for X in Angstroms.
        39 - 46        Real(8.3)       Orthogonal coordinates for Y in Angstroms.
        47 - 54        Real(8.3)       Orthogonal coordinates for Z in Angstroms.
        55 - 60        Real(6.2)       Occupancy (Default = 1.0).
        61 - 66        Real(6.2)       Temperature factor (Default = 0.0).
        73 - 76        LString(4)      Segment identifier, left-justified.
        77 - 78        LString(2)      Element symbol, right-justified.
        79 - 80        LString(2)      Charge on the atom.
        """
        # We might modify first/final status during _finalize() methods
        self.is_first_atom_in_chain = False
        self.is_final_atom_in_chain = False
        self.is_first_residue_in_chain = False
        self.is_final_residue_in_chain = False
        # Start parsing fields from pdb line
        self.record_name = pdb_line[0:6].strip()
        try:
            self.serial_number = _parse_atom_index(pdb_line[6:11])
        except:
            # Just give it the next number in sequence.
            self.serial_number = pdbstructure._next_atom_number
        self.name_with_spaces = pdb_line[12:16]
        alternate_location_indicator = pdb_line[16]

        self.residue_name_with_spaces = pdb_line[17:20]
        # In some MD codes, notably ffamber in gromacs, residue name has a fourth character in
        # column 21
        possible_fourth_character = pdb_line[20:21]
        if possible_fourth_character != " ":
            # Fourth character should only be there if official 3 are already full
            if len(self.residue_name_with_spaces.strip()) != 3:
                raise ValueError('Misaligned residue name: %s' % pdb_line)
            self.residue_name_with_spaces += possible_fourth_character
        self.residue_name = self.residue_name_with_spaces.strip()

        self.chain_id = pdb_line[21]
        try:
            self.residue_number = int(pdb_line[22:26])
        except:
            try:
                self.residue_number = int(pdb_line[22:26], 16) - 0xA000 + 10000
            except:
                # When VMD runs out of hex values it starts filling the residue ID field with ****.
                # Look at the most recent atoms to figure out whether this is a new residue or not.
                if pdbstructure._current_model is None or pdbstructure._current_model._current_chain is None or pdbstructure._current_model._current_chain._current_residue is None:
                    # This is the first residue in the model.
                    self.residue_number = pdbstructure._next_residue_number
                else:
                    currentRes = pdbstructure._current_model._current_chain._current_residue
                    if currentRes.name_with_spaces != self.residue_name_with_spaces:
                        # The residue name has changed.
                        self.residue_number = pdbstructure._next_residue_number
                    elif self.name_with_spaces in currentRes.atoms_by_name:
                        # There is already an atom with this name.
                        self.residue_number = pdbstructure._next_residue_number
                    else:
                        self.residue_number = currentRes.number
        self.insertion_code = pdb_line[26]
        # coordinates, occupancy, and temperature factor belong in Atom.Location object
        x = float(pdb_line[30:38])
        y = float(pdb_line[38:46])
        z = float(pdb_line[46:54])
        try:
            occupancy = float(pdb_line[54:60])
        except:
            occupancy = 1.0
        try:
            temperature_factor = unit.Quantity(float(pdb_line[60:66]), unit.angstroms**2)
        except:
            temperature_factor = unit.Quantity(0.0, unit.angstroms**2)
        self.locations = {}
        loc = Atom.Location(alternate_location_indicator, unit.Quantity(Vec3(x,y,z), unit.angstroms), occupancy, temperature_factor, self.residue_name_with_spaces)
        self.locations[alternate_location_indicator] = loc
        self.default_location_id = alternate_location_indicator
        # segment id, element_symbol, and formal_charge are not always present
        self.segment_id = pdb_line[72:76].strip()
        self.element_symbol = pdb_line[76:78].strip()
        try: self.formal_charge = int(pdb_line[78:80])
        except ValueError: self.formal_charge = None
        # figure out atom element
        if self.element_symbol == extraParticleIdentifier:
            self.element = 'EP'
        else:
            try:
                # Try to find a sensible element symbol from columns 76-77
                self.element = element.get_by_symbol(self.element_symbol)
            except KeyError:
                self.element = None
        if pdbstructure is not None:
            pdbstructure._next_atom_number = self.serial_number+1
            pdbstructure._next_residue_number = self.residue_number+1

    def iter_locations(self):
        """
        Iterate over Atom.Location objects for this atom, including primary location.
        >>> atom = Atom("ATOM   2209  CB  TYR A 299       6.167  22.607  20.046  1.00  8.12           C")
        >>> for c in atom.iter_locations():
        ...     print c
        ...
        [6.167, 22.607, 20.046] A
        """
        for alt_loc in self.locations:
            yield self.locations[alt_loc]

    def iter_positions(self):
        """
        Iterate over atomic positions.  Returns Quantity(Vec3(), unit) objects, unlike
        iter_locations, which returns Atom.Location objects.
        """
        for loc in self.iter_locations():
            yield loc.position

    def iter_coordinates(self):
        """
        Iterate over x, y, z values of primary atom position.
        >>> atom = Atom("ATOM   2209  CB  TYR A 299       6.167  22.607  20.046  1.00  8.12           C")
        >>> for c in atom.iter_coordinates():
        ...     print c
        ...
        6.167 A
        22.607 A
        20.046 A
        """
        for coord in self.position:
            yield coord

    # Hide existence of multiple alternate locations to avoid scaring casual users
    def get_location(self, location_id=None):
        id = location_id
        if id is None:
            id = self.default_location_id
        return self.locations[id]
    def set_location(self, new_location, location_id=None):
        id = location_id
        if id is None:
            id = self.default_location_id
        self.locations[id] = new_location
    location = property(get_location, set_location, doc='default Atom.Location object')

    def get_position(self):
        return self.location.position
    def set_position(self, coords):
        self.location.position = coords
    position = property(get_position, set_position, doc='orthogonal coordinates')

    def get_alternate_location_indicator(self):
        return self.location.alternate_location_indicator
    alternate_location_indicator = property(get_alternate_location_indicator)

    def get_occupancy(self):
        return self.location.occupancy
    occupancy = property(get_occupancy)

    def get_temperature_factor(self):
        return self.location.temperature_factor
    temperature_factor = property(get_temperature_factor)

    def get_x(self): return self.position[0]
    x = property(get_x)

    def get_y(self): return self.position[1]
    y = property(get_y)

    def get_z(self): return self.position[2]
    z = property(get_z)

    def _pdb_string(self, serial_number=None, alternate_location_indicator=None):
        """
        Produce a PDB line for this atom using a particular serial number and alternate location
        """
        if serial_number is None:
            serial_number = self.serial_number
        if alternate_location_indicator is None:
            alternate_location_indicator = self.alternate_location_indicator
        # produce PDB line in three parts: names, numbers, and end
        # Accomodate 4-character residue names that use column 21
        long_res_name = self.residue_name_with_spaces
        if len(long_res_name) == 3:
            long_res_name += " "
        assert len(long_res_name) == 4
        names = "%-6s%5d %4s%1s%4s%1s%4d%1s   " % (
            self.record_name, serial_number, \
            self.name_with_spaces, alternate_location_indicator, \
            long_res_name, self.chain_id, \
            self.residue_number, self.insertion_code)
        numbers = "%8.3f%8.3f%8.3f%6.2f%6.2f      " % (
            self.x.value_in_unit(unit.angstroms), \
            self.y.value_in_unit(unit.angstroms), \
            self.z.value_in_unit(unit.angstroms), \
            self.occupancy, \
            self.temperature_factor.value_in_unit(unit.angstroms * unit.angstroms))
        end =  "%-4s%2s" % (\
            self.segment_id, self.element_symbol)
        formal_charge = "  "
        if (self.formal_charge != None): formal_charge = "%+2d" % self.formal_charge
        return names+numbers+end+formal_charge

    def __str__(self):
        return self._pdb_string(self.serial_number, self.alternate_location_indicator)

    def write(self, next_serial_number, output_stream=sys.stdout, alt_loc = "*"):
        """
        alt_loc = "*" means write all alternate locations
        alt_loc = None means write just the primary location
        alt_loc = "AB" means write locations "A" and "B"
        """
        if alt_loc is None:
            locs = [self.default_location_id]
        elif alt_loc == "":
            locs = [self.default_location_id]
        elif alt_loc == "*":
            locs = self.locations.keys()
            locs.sort()
        else:
            locs = list(alt_loc)
        for loc_id in locs:
            print(self._pdb_string(next_serial_number.val, loc_id), file=output_stream)
            next_serial_number.increment()

    def set_name_with_spaces(self, name):
        assert len(name) == 4
        self._name_with_spaces = name
        self._name = name.strip()
    def get_name_with_spaces(self):
        return self._name_with_spaces
    name_with_spaces = property(get_name_with_spaces, set_name_with_spaces, doc='four-character residue name including spaces')

    def get_name(self):
        return self._name
    name = property(get_name, doc='residue name')

    class Location(object):
        """
        Inner class of Atom for holding alternate locations
        """
        def __init__(self, alt_loc, position, occupancy, temperature_factor, residue_name):
            self.alternate_location_indicator = alt_loc
            self.position = position
            self.occupancy = occupancy
            self.temperature_factor = temperature_factor
            self.residue_name = residue_name

        def __iter__(self):
            """
            Examples
            >>> from openmm.vec3 import Vec3
            >>> import openmm.unit as unit
            >>> l = Atom.Location(' ', Vec3(1,2,3)*unit.angstroms, 1.0, 20.0*unit.angstroms**2, "XXX")
            >>> for c in l:
            ...     print c
            ...
            1 A
            2 A
            3 A
            """
            for coord in self.position:
                yield coord

        def __str__(self):
            return str(self.position)    

class PDBFile:

    """PDBFile parses a Protein Data Bank (PDB) file and constructs a Topology and a set of atom positions from it.
    This class also provides methods for creating PDB files.  To write a file containing a single model, call
    writeFile().  You also can create files that contain multiple models.  To do this, first call writeHeader(),
    then writeModel() once for each model in the file, and finally writeFooter() to complete the file."""

    _residueNameReplacements = {}
    _atomNameReplacements = {}
    _standardResidues = ['ALA', 'ASN', 'CYS', 'GLU', 'HIS', 'LEU', 'MET', 'PRO', 'THR', 'TYR',
                         'ARG', 'ASP', 'GLN', 'GLY', 'ILE', 'LYS', 'PHE', 'SER', 'TRP', 'VAL',
                         'A', 'G', 'C', 'U', 'I', 'DA', 'DG', 'DC', 'DT', 'DI', 'HOH']

    def __init__(self, file, extraParticleIdentifier='EP'):
        """Load a PDB file.
        The atom positions and Topology can be retrieved by calling getPositions() and getTopology().
        Args:
            file : string or file. The name of the file to load.  Alternatively you can pass an open file object.
            extraParticleIdentifier : string='EP'. If this value appears in the element column for an ATOM record, the Atom's element will be set to None to mark it as an extra particle
        """
        metalElements = ['Al','As','Ba','Ca','Cd','Ce','Co','Cs','Cu','Dy','Fe','Gd','Hg','Ho','In','Ir','K','Li','Mg',
        'Mn','Mo','Na','Ni','Pb','Pd','Pt','Rb','Rh','Sm','Sr','Te','Tl','V','W','Yb','Zn']

        # TODO: init topo

        # load the PDB file

        if isinstance(file, PdbStructure):
            pdb = file
        else:
            inputfile = file
            own_handle = False
            if isinstance(file, str):
                inputfile = open(file)
                own_handle = True
            pdb = PdbStructure(inputfile, load_all_models=True, extraParticleIdentifier=extraParticleIdentifier)
            if own_handle:
                inputfile.close()
        PDBFile._loadNameReplacementTables()


        # Build the topology

        atomByNumber = {}
        for chain in pdb.iter_chains():
            c = top.addChain(chain.chain_id)
            for residue in chain.iter_residues():
                resName = residue.get_name()
                if resName in PDBFile._residueNameReplacements:
                    resName = PDBFile._residueNameReplacements[resName]
                r = top.addResidue(resName, c, str(residue.number), residue.insertion_code)
                if resName in PDBFile._atomNameReplacements:
                    atomReplacements = PDBFile._atomNameReplacements[resName]
                else:
                    atomReplacements = {}
                processedAtomNames = set()
                for atom in residue.atoms_by_name.values():
                    atomName = atom.get_name()
                    if atomName in processedAtomNames or atom.residue_name != residue.get_name():
                        continue
                    processedAtomNames.add(atomName)
                    if atomName in atomReplacements:
                        atomName = atomReplacements[atomName]
                    atomName = atomName.strip()
                    element = atom.element
                    if element == 'EP':
                        element = None
                    elif element is None:
                        # Try to guess the element.

                        upper = atomName.upper()
                        while len(upper) > 1 and upper[0].isdigit():
                            upper = upper[1:]
                        if upper.startswith('CL'):
                            element = elem.chlorine
                        elif upper.startswith('NA'):
                            element = elem.sodium
                        elif upper.startswith('MG'):
                            element = elem.magnesium
                        elif upper.startswith('BE'):
                            element = elem.beryllium
                        elif upper.startswith('LI'):
                            element = elem.lithium
                        elif upper.startswith('K'):
                            element = elem.potassium
                        elif upper.startswith('ZN'):
                            element = elem.zinc
                        elif len(residue) == 1 and upper.startswith('CA'):
                            element = elem.calcium
                        elif upper.startswith('D') and any(a.name == atomName[1:] for a in residue.iter_atoms()):
                            pass # A Drude particle
                        else:
                            try:
                                element = elem.get_by_symbol(upper[0])
                            except KeyError:
                                pass
                    newAtom = top.addAtom(atomName, element, r, str(atom.serial_number))
                    atomByNumber[atom.serial_number] = newAtom
        self._positions = []
        for model in pdb.iter_models(True):
            coords = []
            for chain in model.iter_chains():
                for residue in chain.iter_residues():
                    processedAtomNames = set()
                    for atom in residue.atoms_by_name.values():
                        if atom.get_name() in processedAtomNames or atom.residue_name != residue.get_name():
                            continue
                        processedAtomNames.add(atom.get_name())
                        pos = atom.get_position().value_in_unit(nanometers)
                        coords.append(Vec3(pos[0], pos[1], pos[2]))
            self._positions.append(coords*nanometers)
        ## The atom positions read from the PDB file.  If the file contains multiple frames, these are the positions in the first frame.
        self.positions = self._positions[0]
        self.topology.setPeriodicBoxVectors(pdb.get_periodic_box_vectors())
        self.topology.createStandardBonds()
        self.topology.createDisulfideBonds(self.positions)
        self._numpyPositions = None

        # Add bonds based on CONECT records. Bonds between metals of elements specified in metalElements and residues in standardResidues are not added.

        connectBonds = []
        for connect in pdb.models[-1].connects:
            i = connect[0]
            for j in connect[1:]:
                if i in atomByNumber and j in atomByNumber:    
                    if atomByNumber[i].element is not None and atomByNumber[j].element is not None:
                        if atomByNumber[i].element.symbol not in metalElements and atomByNumber[j].element.symbol not in metalElements:
                            connectBonds.append((atomByNumber[i], atomByNumber[j])) 
                        elif atomByNumber[i].element.symbol in metalElements and atomByNumber[j].residue.name not in PDBFile._standardResidues:
                            connectBonds.append((atomByNumber[i], atomByNumber[j])) 
                        elif atomByNumber[j].element.symbol in metalElements and atomByNumber[i].residue.name not in PDBFile._standardResidues:
                            connectBonds.append((atomByNumber[i], atomByNumber[j]))     
                    else:
                        connectBonds.append((atomByNumber[i], atomByNumber[j]))         
        if len(connectBonds) > 0:
            # Only add bonds that don't already exist.
            existingBonds = set(top.bonds())
            for bond in connectBonds:
                if bond not in existingBonds and (bond[1], bond[0]) not in existingBonds:
                    top.addBond(bond[0], bond[1])
                    existingBonds.add(bond)

    def getTopology(self):
        """Get the Topology of the model."""
        return self.topology

    def getNumFrames(self):
        """Get the number of frames stored in the file."""
        return len(self._positions)

    def getPositions(self, asNumpy=False, frame=0):
        """Get the atomic positions.
        Parameters
        ----------
        asNumpy : boolean=False
            if true, the values are returned as a numpy array instead of a list
            of Vec3s
        frame : int=0
            the index of the frame for which to get positions
        """
        if asNumpy:
            if self._numpyPositions is None:
                self._numpyPositions = [None]*len(self._positions)
            if self._numpyPositions[frame] is None:
                self._numpyPositions[frame] = Quantity(numpy.array(self._positions[frame].value_in_unit(nanometers)), nanometers)
            return self._numpyPositions[frame]
        return self._positions[frame]

    @staticmethod
    def _loadNameReplacementTables():
        """Load the list of atom and residue name replacements."""
        if len(PDBFile._residueNameReplacements) == 0:
            tree = etree.parse(os.path.join(os.path.dirname(__file__), 'data', 'pdbNames.xml'))
            allResidues = {}
            proteinResidues = {}
            nucleicAcidResidues = {}
            for residue in tree.getroot().findall('Residue'):
                name = residue.attrib['name']
                if name == 'All':
                    PDBFile._parseResidueAtoms(residue, allResidues)
                elif name == 'Protein':
                    PDBFile._parseResidueAtoms(residue, proteinResidues)
                elif name == 'Nucleic':
                    PDBFile._parseResidueAtoms(residue, nucleicAcidResidues)
            for atom in allResidues:
                proteinResidues[atom] = allResidues[atom]
                nucleicAcidResidues[atom] = allResidues[atom]
            for residue in tree.getroot().findall('Residue'):
                name = residue.attrib['name']
                for id in residue.attrib:
                    if id == 'name' or id.startswith('alt'):
                        PDBFile._residueNameReplacements[residue.attrib[id]] = name
                if 'type' not in residue.attrib:
                    atoms = copy(allResidues)
                elif residue.attrib['type'] == 'Protein':
                    atoms = copy(proteinResidues)
                elif residue.attrib['type'] == 'Nucleic':
                    atoms = copy(nucleicAcidResidues)
                else:
                    atoms = copy(allResidues)
                PDBFile._parseResidueAtoms(residue, atoms)
                PDBFile._atomNameReplacements[name] = atoms

    @staticmethod
    def _parseResidueAtoms(residue, map):
        for atom in residue.findall('Atom'):
            name = atom.attrib['name']
            for id in atom.attrib:
                map[atom.attrib[id]] = name

    @staticmethod
    def writeFile(topology, positions, file=sys.stdout, keepIds=False, extraParticleIdentifier='EP'):
        """Write a PDB file containing a single model.
        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write to
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDB format.  Otherwise, the output file will be invalid.
        extraParticleIdentifier : string='EP'
            String to write in the element column of the ATOM records for atoms whose element is None (extra particles)
        """
        PDBFile.writeHeader(topology, file)
        PDBFile.writeModel(topology, positions, file, keepIds=keepIds, extraParticleIdentifier=extraParticleIdentifier)
        PDBFile.writeFooter(topology, file)

    @staticmethod
    def writeHeader(topology, file=sys.stdout):
        """Write out the header for a PDB file.
        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        """
        print("REMARK   1 CREATED WITH OPENMM %s, %s" % (Platform.getOpenMMVersion(), str(date.today())), file=file)
        vectors = topology.getPeriodicBoxVectors()
        if vectors is not None:
            a, b, c, alpha, beta, gamma = computeLengthsAndAngles(vectors)
            RAD_TO_DEG = 180/math.pi
            print("CRYST1%9.3f%9.3f%9.3f%7.2f%7.2f%7.2f P 1           1 " % (
                    a*10, b*10, c*10, alpha*RAD_TO_DEG, beta*RAD_TO_DEG, gamma*RAD_TO_DEG), file=file)

    @staticmethod
    def writeModel(topology, positions, file=sys.stdout, modelIndex=None, keepIds=False, extraParticleIdentifier='EP'):
        """Write out a model to a PDB file.
        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write the model to
        modelIndex : int=None
            If not None, the model will be surrounded by MODEL/ENDMDL records
            with this index
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDB format.  No guarantees are made about what will happen if they
            are not, and the output file could be invalid.
        extraParticleIdentifier : string='EP'
            String to write in the element column of the ATOM records for atoms whose element is None (extra particles)
        """

        if len(list(topology.atoms())) != len(positions):
            raise ValueError('The number of positions must match the number of atoms')
        if is_quantity(positions):
            positions = positions.value_in_unit(angstroms)
        if any(math.isnan(norm(pos)) for pos in positions):
            raise ValueError('Particle position is NaN.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan')
        if any(math.isinf(norm(pos)) for pos in positions):
            raise ValueError('Particle position is infinite.  For more information, see https://github.com/openmm/openmm/wiki/Frequently-Asked-Questions#nan')
        nonHeterogens = PDBFile._standardResidues[:]
        nonHeterogens.remove('HOH')
        atomIndex = 1
        posIndex = 0
        if modelIndex is not None:
            print("MODEL     %4d" % modelIndex, file=file)
        for (chainIndex, chain) in enumerate(topology.chains()):
            if keepIds and len(chain.id) == 1:
                chainName = chain.id
            else:
                chainName = chr(ord('A')+chainIndex%26)
            residues = list(chain.residues())
            for (resIndex, res) in enumerate(residues):
                if len(res.name) > 3:
                    resName = res.name[:3]
                else:
                    resName = res.name
                if keepIds and len(res.id) < 5:
                    resId = res.id
                else:
                    resId = _formatIndex(resIndex+1, 4)
                if len(res.insertionCode) == 1:
                    resIC = res.insertionCode
                else:
                    resIC = " "
                if res.name in nonHeterogens:
                    recordName = "ATOM  "
                else:
                    recordName = "HETATM"
                for atom in res.atoms():
                    if atom.element is not None:
                        symbol = atom.element.symbol
                    else:
                        symbol = extraParticleIdentifier
                    if len(atom.name) < 4 and atom.name[:1].isalpha() and len(symbol) < 2:
                        atomName = ' '+atom.name
                    elif len(atom.name) > 4:
                        atomName = atom.name[:4]
                    else:
                        atomName = atom.name
                    coords = positions[posIndex]
                    line = "%s%5s %-4s %3s %s%4s%1s   %s%s%s  1.00  0.00          %2s  " % (
                        recordName, _formatIndex(atomIndex, 5), atomName, resName, chainName, resId, resIC, _format_83(coords[0]),
                        _format_83(coords[1]), _format_83(coords[2]), symbol)
                    if len(line) != 80:
                        raise ValueError('Fixed width overflow detected')
                    print(line, file=file)
                    posIndex += 1
                    atomIndex += 1
                if resIndex == len(residues)-1:
                    print("TER   %5s      %3s %s%4s" % (_formatIndex(atomIndex, 5), resName, chainName, resId), file=file)
                    atomIndex += 1
        if modelIndex is not None:
            print("ENDMDL", file=file)

    @staticmethod
    def writeFooter(topology, file=sys.stdout):
        """Write out the footer for a PDB file.
        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        """
        # Identify bonds that should be listed as CONECT records.

        conectBonds = []
        for atom1, atom2 in topology.bonds():
            if atom1.residue.name not in PDBFile._standardResidues or atom2.residue.name not in PDBFile._standardResidues:
                conectBonds.append((atom1, atom2))
            elif atom1.name == 'SG' and atom2.name == 'SG' and atom1.residue.name == 'CYS' and atom2.residue.name == 'CYS':
                conectBonds.append((atom1, atom2))
        if len(conectBonds) > 0:

            # Work out the index used in the PDB file for each atom.

            atomIndex = {}
            nextAtomIndex = 0
            prevChain = None
            for chain in topology.chains():
                for atom in chain.atoms():
                    if atom.residue.chain != prevChain:
                        nextAtomIndex += 1
                        prevChain = atom.residue.chain
                    atomIndex[atom] = nextAtomIndex
                    nextAtomIndex += 1

            # Record which other atoms each atom is bonded to.

            atomBonds = {}
            for atom1, atom2 in conectBonds:
                index1 = atomIndex[atom1]
                index2 = atomIndex[atom2]
                if index1 not in atomBonds:
                    atomBonds[index1] = []
                if index2 not in atomBonds:
                    atomBonds[index2] = []
                atomBonds[index1].append(index2)
                atomBonds[index2].append(index1)

            # Write the CONECT records.

            for index1 in sorted(atomBonds):
                bonded = atomBonds[index1]
                while len(bonded) > 4:
                    print("CONECT%5s%5s%5s%5s" % (_formatIndex(index1, 5), _formatIndex(bonded[0], 5), _formatIndex(bonded[1], 5), _formatIndex(bonded[2], 5)), file=file)
                    del bonded[:4]
                line = "CONECT%5s" % _formatIndex(index1, 5)
                for index2 in bonded:
                    line = "%s%5s" % (line, _formatIndex(index2, 5))
                print(line, file=file)
        print("END", file=file)


def _format_83(f):
    """Format a single float into a string of width 8, with ideally 3 decimal
    places of precision. If the number is a little too large, we can
    gracefully degrade the precision by lopping off some of the decimal
    places. If it's much too large, we throw a ValueError"""
    if -999.999 < f < 9999.999:
        return '%8.3f' % f
    if -9999999 < f < 99999999:
        return ('%8.3f' % f)[:8]
    raise ValueError('coordinate "%s" could not be represented '
                     'in a width-8 field' % f)

def _formatIndex(index, places):
    """Create a string representation of an atom or residue index.  If the value is larger than can fit
    in the available space, switch to hex.
    """
    if index < 10**places:
        format = f'%{places}d'
        return format % index
    format = f'%{places}X'
    shiftedIndex = (index - 10**places + 10*16**(places-1)) % (16**places)
    return format % shiftedIndex