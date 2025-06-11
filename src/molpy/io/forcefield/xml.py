import logging
import xml.etree.ElementTree as ET
from pathlib import Path

logger = logging.getLogger("molpy")

class XMLForceFieldReader:

    def __init__(self, file: Path, style=""):

        self._file = file

    def read(self, system):
        """Load an XML file and add the definitions from it to this ForceField.

        Parameters
        ----------
        files : string or file or tuple
            An XML file or tuple of XML files containing force field definitions.
            Each entry may be either an absolute file path, a path relative to the current working
            directory, a path relative to this module's data subdirectory (for
            built in force fields), or an open file-like object with a read()
            method from which the forcefield XML data can be loaded.
        """
        ff = system.forcefield

        trees = ET.parse(self._file)
        main = trees.getroot()

        atomstyle = self._read_atomtypes(main, ff)

        nonbonded_flag = ['vdW', 'NonbondedForce']
        for flag in nonbonded_flag:
            for section in main:
                if flag in section.tag:
                    pairstyle = ff.def_pairstyle(flag, section.attrib)
                    self._read_nonbonded(section, atomstyle, pairstyle)

        for section in main:
            if "Bond" in section.tag:
                pot_name = section.tag
                bondstyle = ff.def_bondstyle(pot_name, section.attrib)
                self._read_bonds(section, atomstyle, bondstyle)

        return system


    def _read_atomtypes(self, main, ff):
        atomstyle = ff.def_atomstyle(self._file.stem)
        atomtypes = main.find("AtomTypes")
        if atomtypes is not None:
            for atomtype in atomtypes:
                attrib = atomtype.attrib
                name = attrib.pop("name")
                atomstyle.def_type(name, **atomtype.attrib)
        
        logger.info(f"Read {len(ff.atomtypes)} atom types")
        return atomstyle

    def _read_nonbonded(self, nonbonded, atomstyle, pairstyle):

        atoms = nonbonded.findall("Atom")
        probe_atom = atoms[0]

        name_flag = self._guess_item_name(probe_atom)
        
        for i, atom in enumerate(atoms):
            attrib = atom.attrib
            name = attrib.pop(name_flag, str(i))
            at = atomstyle.get(name)
            if at is None:
                at = atomstyle.def_type(name)
            if "def" in attrib:
                attrib["smirks"] = attrib.pop("def")
            if "smirks" in attrib:
                smirks = attrib.pop("smirks")
                at["smirks"] = smirks
            pairstyle.def_type(name, at, at, **attrib)

        logger.info(f"Read {atomstyle.n_types} pair types")

    def _read_bonds(self, bonds, atomstyle, bondstyle):

       
        bonds = bonds.findall("Bond")
        name_flag = self._guess_item_name(bonds[0])

        for bond in bonds:
            attrib = bond.attrib
            name = attrib.pop(name_flag, "")
            type1 = attrib.pop("type1", None)
            type2 = attrib.pop("type2", None)
            if type1 is not None and type2 is not None:
                at1 = atomstyle.get_type(type1)
                at2 = atomstyle.get_type(type2)
                bondstyle.def_type(name, at1, at2, **attrib
            )
            else:
                class1 = attrib.pop("class1", None)
                class2 = attrib.pop("class2", None)
                if class1 is not None and class2 is not None:
                    for c1 in atomstyle.get_class(class1):
                        for c2 in atomstyle.get_class(class2):
                            bondstyle.def_type(name, c1, c2, **attrib)

        logger.info(f"Read {len(bondstyle.types)} bond types")

    def _guess_item_name(self, item):

        if 'type' in item.attrib:
            name_flag = 'type'
        elif 'id' in item.attrib:
            name_flag = 'id'
        else:
            name_flag = 'name'

        return name_flag