# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-24
# version: 0.0.1

from molpy.atom import Atom
from molpy.group import Group
from xml.dom import minidom


def read_xml(fileobj):
    
    fileobj = minidom.parse(fileobj)
    
    groupTemplates = {}
    atomTemplates = {}
    
    # define atom
    for a in fileobj.getElementByTagName('AtomTypes'):
        name_ = a.getAttribute('name')
        class_ = a.getAttribute('class')
        element = a.getAttribute('element')
        mass = a.getAttribute('mass')
        atom = Atom(name_, atomClass=class_, element=element, mass=mass)
        atomTemplates[name_] = atom
          
    for r in fileobj.getElementByTagName('Residue'):
        
        resName = r.getAttribute('name')
        residue = Group(resName)
        groupTemplates[resName] = residue
        
        # read atom
        for a in r.getElementsByTagName('Atom'):
            atomName = a.getAttribute('name')
            atomType = a.getAttribute('type')
            atom = Atom(atomName, type=atomType)
            
            residue.addAtom(atom)
            
        # read bond
        for b in r.getElementsByTagName('Bond'):
            from_ = b.getAttribute('from')
            to_   = b.getAttribute('to')
            residue.addBondByIndex(from_, to_)
    
        
    for multipole in fileobj.getElementsByTagName("Multipole"):
        
        multiDict = {
            "c0": float(multipole.getAttribute("c0")),
            "dX": float(multipole.getAttribute("dX")),
            "dY": float(multipole.getAttribute("dY")),
            "dZ": float(multipole.getAttribute("dZ")),
            "qXX": float(multipole.getAttribute("qXX")),
            "qXY": float(multipole.getAttribute("qXY")),
            "qYY": float(multipole.getAttribute("qYY")),
            "qXZ": float(multipole.getAttribute("qXZ")),
            "qYZ": float(multipole.getAttribute("qYZ")),
            "qZZ": float(multipole.getAttribute("qZZ")),
            "oXXX": float(multipole.getAttribute("oXXX")),
            "oXXY": float(multipole.getAttribute("oXXY")),
            "oXYY": float(multipole.getAttribute("oXYY")),
            "oYYY": float(multipole.getAttribute("oYYY")),
            "oXXZ": float(multipole.getAttribute("oXXZ")),
            "oXYZ": float(multipole.getAttribute("oXYZ")),
            "oYYZ": float(multipole.getAttribute("oYYZ")),
            "oXZZ": float(multipole.getAttribute("oXZZ")),
            "oYZZ": float(multipole.getAttribute("oYZZ")),
            "oZZZ": float(multipole.getAttribute("oZZZ")),
            "kx": multipole.getAttribute("kx"),
            "kz": multipole.getAttribute("kz"),
            "ky": multipole.getAttribute("ky")
        }

        # update residue
        for template in atomTemplates:
            if template['type'] == multipole.getAttribute('type'):
                template.update(**multiDict)
                
    # init residues
    
    