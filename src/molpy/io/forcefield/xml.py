from pathlib import Path
import xml.etree.ElementTree as etree
import os


class XMLForceFieldReader:

    def __init__(self, filepaths: Path | list[Path]):

        self.filepaths = []
        if isinstance(filepaths, Path):
            self.filepaths.append(filepaths)
        elif isinstance(filepaths, list):
            self.filepaths.extend(filepaths)

        self._atomTypes = {}
        self._templates = {}
        self._patches = {}
        self._templatePatches = {}
        self._templateSignatures = {None: []}
        self._atomClasses = {"": set()}
        self._forces = []
        self._scripts = []
        self._templateMatchers = []
        self._templateGenerators = []

    def read(self, system, resname_prefix: str = ""):
        """Load an XML file and add the definitions from it to this ForceField.

        Parameters
        ----------
        files : string or file or tuple
            An XML file or tuple of XML files containing force field definitions.
            Each entry may be either an absolute file path, a path relative to the current working
            directory, a path relative to this module's data subdirectory (for
            built in force fields), or an open file-like object with a read()
            method from which the forcefield XML data can be loaded.
        prefix : string
            An optional string to be prepended to each residue name found in the
            loaded files.
        """
        ff = system.forcefield

        atomstyle = ff.def_atomstyle("full")  # dont know what's the name is


        files = self.filepaths

        trees = []

        i = 0
        while i < len(files):
            file = files[i]
            tree = etree.parse(file)
            if tree is None:
                raise ValueError('Could not locate file "%s"' % file)
            trees.append(tree)
            i += 1

            # Process includes in this file.

            if isinstance(file, str):
                parentDir = os.path.dirname(file)
            else:
                parentDir = ""
            for included in tree.getroot().findall("Include"):
                includeFile = included.attrib["file"]
                joined = os.path.join(parentDir, includeFile)
                if os.path.isfile(joined):
                    includeFile = joined
                if includeFile not in files:
                    files.append(includeFile)

        # Load the atom types.

        for tree in trees:
            if tree.getroot().find("AtomTypes") is not None:
                for type in tree.getroot().find("AtomTypes").findall("Type"):
                    class_ = type.attrib.pop("class")
                    if "overrides" in type.attrib:
                        type.attrib["overrides"] = type.attrib.pop("overrides").split(',')
                    atomstyle.def_type(name=type.attrib.pop("name"), **type.attrib, class_=class_)


        # Load force definitions

        for tree in trees:
            for child in tree.getroot():
                if child.tag.endswith("BondForce"):
                    bondstyle = ff.get_bondstyle(child.tag)
                    if ff.get_bondstyle(child.tag) is None:
                        bondstyle = ff.def_bondstyle(child.tag)
                    for c in child:
                        force = c.attrib
                        class1 = force.pop("class1")
                        class2 = force.pop("class2")
                        atomclass1 = atomstyle.get_class(class1)
                        atomclass2 = atomstyle.get_class(class2)
                        for atom1 in atomclass1:
                            for atom2 in atomclass2:
                                bondstyle.def_type(atom1, atom2, kw_params=force)

                elif child.tag.endswith("AngleForce"):
                    anglestyle = ff.get_anglestyle(child.tag)
                    if ff.get_anglestyle(child.tag) is None:
                        anglestyle = ff.def_anglestyle(child.tag)
                    for c in child:
                        force = c.attrib
                        class1 = force.pop("class1")
                        class2 = force.pop("class2")
                        class3 = force.pop("class3")
                        atomclass1 = atomstyle.get_class(class1)
                        atomclass2 = atomstyle.get_class(class2)
                        atomclass3 = atomstyle.get_class(class3)
                        for atom1 in atomclass1:
                            for atom2 in atomclass2:
                                for atom3 in atomclass3:
                                    anglestyle.def_type(atom1, atom2, atom3, kw_params=force)

        return system

        # Load the residue templates.

        # for tree in trees:
        #     if tree.getroot().find('Residues') is not None:
        #         for residue in tree.getroot().find('Residues').findall('Residue'):
        #             resName = resname_prefix+residue.attrib['name']
        #             template = ForceField._TemplateData(resName)
        #             if 'override' in residue.attrib:
        #                 template.overrideLevel = int(residue.attrib['override'])
        #             if 'rigidWater' in residue.attrib:
        #                 template.rigidWater = (residue.attrib['rigidWater'].lower() == 'true')
        #             for key in residue.attrib:
        #                 template.attributes[key] = residue.attrib[key]
        #             atomIndices = template.atomIndices
        #             for ia, atom in enumerate(residue.findall('Atom')):
        #                 params = {}
        #                 for key in atom.attrib:
        #                     if key not in ('name', 'type'):
        #                         params[key] = _convertParameterToNumber(atom.attrib[key])
        #                 atomName = atom.attrib['name']
        #                 if atomName in atomIndices:
        #                     raise ValueError('Residue '+resName+' contains multiple atoms named '+atomName)
        #                 typeName = atom.attrib['type']
        #                 atomIndices[atomName] = ia
        #                 template.atoms.append(ForceField._TemplateAtomData(atomName, typeName, self._atomTypes[typeName].element, params))
        #             for site in residue.findall('VirtualSite'):
        #                 template.virtualSites.append(ForceField._VirtualSiteData(site, atomIndices))
        #             for bond in residue.findall('Bond'):
        #                 if 'atomName1' in bond.attrib:
        #                     template.addBondByName(bond.attrib['atomName1'], bond.attrib['atomName2'])
        #                 else:
        #                     template.addBond(int(bond.attrib['from']), int(bond.attrib['to']))
        #             for bond in residue.findall('ExternalBond'):
        #                 if 'atomName' in bond.attrib:
        #                     template.addExternalBondByName(bond.attrib['atomName'])
        #                 else:
        #                     template.addExternalBond(int(bond.attrib['from']))
        #             for patch in residue.findall('AllowPatch'):
        #                 patchName = patch.attrib['name']
        #                 if ':' in patchName:
        #                     colonIndex = patchName.find(':')
        #                     self.registerTemplatePatch(resName, patchName[:colonIndex], int(patchName[colonIndex+1:])-1)
        #                 else:
        #                     self.registerTemplatePatch(resName, patchName, 0)
        #             self.registerResidueTemplate(template)

        # Load the patch definitions.

        # for tree in trees:
        #     if tree.getroot().find('Patches') is not None:
        #         for patch in tree.getroot().find('Patches').findall('Patch'):
        #             patchName = patch.attrib['name']
        #             if 'residues' in patch.attrib:
        #                 numResidues = int(patch.attrib['residues'])
        #             else:
        #                 numResidues = 1
        #             patchData = ForceField._PatchData(patchName, numResidues)
        #             for key in patch.attrib:
        #                 patchData.attributes[key] = patch.attrib[key]
        #             for atom in patch.findall('AddAtom'):
        #                 params = {}
        #                 for key in atom.attrib:
        #                     if key not in ('name', 'type'):
        #                         params[key] = _convertParameterToNumber(atom.attrib[key])
        #                 atomName = atom.attrib['name']
        #                 if atomName in patchData.allAtomNames:
        #                     raise ValueError('Patch '+patchName+' contains multiple atoms named '+atomName)
        #                 patchData.allAtomNames.add(atomName)
        #                 atomDescription = ForceField._PatchAtomData(atomName)
        #                 typeName = atom.attrib['type']
        #                 patchData.addedAtoms[atomDescription.residue].append(ForceField._TemplateAtomData(atomDescription.name, typeName, self._atomTypes[typeName].element, params))
        #             for atom in patch.findall('ChangeAtom'):
        #                 params = {}
        #                 for key in atom.attrib:
        #                     if key not in ('name', 'type'):
        #                         params[key] = _convertParameterToNumber(atom.attrib[key])
        #                 atomName = atom.attrib['name']
        #                 if atomName in patchData.allAtomNames:
        #                     raise ValueError('Patch '+patchName+' contains multiple atoms named '+atomName)
        #                 patchData.allAtomNames.add(atomName)
        #                 atomDescription = ForceField._PatchAtomData(atomName)
        #                 typeName = atom.attrib['type']
        #                 patchData.changedAtoms[atomDescription.residue].append(ForceField._TemplateAtomData(atomDescription.name, typeName, self._atomTypes[typeName].element, params))
        #             for atom in patch.findall('RemoveAtom'):
        #                 atomName = atom.attrib['name']
        #                 if atomName in patchData.allAtomNames:
        #                     raise ValueError('Patch '+patchName+' contains multiple atoms named '+atomName)
        #                 patchData.allAtomNames.add(atomName)
        #                 atomDescription = ForceField._PatchAtomData(atomName)
        #                 patchData.deletedAtoms.append(atomDescription)
        #             for bond in patch.findall('AddBond'):
        #                 atom1 = ForceField._PatchAtomData(bond.attrib['atomName1'])
        #                 atom2 = ForceField._PatchAtomData(bond.attrib['atomName2'])
        #                 patchData.addedBonds.append((atom1, atom2))
        #             for bond in patch.findall('RemoveBond'):
        #                 atom1 = ForceField._PatchAtomData(bond.attrib['atomName1'])
        #                 atom2 = ForceField._PatchAtomData(bond.attrib['atomName2'])
        #                 patchData.deletedBonds.append((atom1, atom2))
        #             for bond in patch.findall('AddExternalBond'):
        #                 atom = ForceField._PatchAtomData(bond.attrib['atomName'])
        #                 patchData.addedExternalBonds.append(atom)
        #             for bond in patch.findall('RemoveExternalBond'):
        #                 atom = ForceField._PatchAtomData(bond.attrib['atomName'])
        #                 patchData.deletedExternalBonds.append(atom)
        #             # The following three lines are only correct for single residue patches.  Multi-residue patches with
        #             # virtual sites currently don't work correctly.  See issue #2848.
        #             atomIndices = dict((atom.name, i) for i, atom in enumerate(patchData.addedAtoms[0]+patchData.changedAtoms[0]))
        #             for site in patch.findall('VirtualSite'):
        #                 patchData.virtualSites[0].append(ForceField._VirtualSiteData(site, atomIndices))
        #             for residue in patch.findall('ApplyToResidue'):
        #                 name = residue.attrib['name']
        #                 if ':' in name:
        #                     colonIndex = name.find(':')
        #                     self.registerTemplatePatch(name[colonIndex+1:], patchName, int(name[:colonIndex])-1)
        #                 else:
        #                     self.registerTemplatePatch(name, patchName, 0)
        #             self.registerPatch(patchData)

        # Load scripts

        # for tree in trees:
        #     for node in tree.getroot().findall('Script'):
        #         self.registerScript(node.text)

        # Execute initialization scripts.

        # for tree in trees:
        #     for node in tree.getroot().findall('InitializationScript'):
        #         exec(node.text, locals())
