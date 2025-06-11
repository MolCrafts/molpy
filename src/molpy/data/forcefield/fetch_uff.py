from collections import namedtuple
import requests
import molpy as mp
import numpy as np

class FetchUFF:
    #
    # Open Babel file: UFF.prm
    #
    # Force field parameters for UFF, the Universal Force Field
    # Used by OBForceField and OBUFFForceField
    #
    # J. Am. Chem. Soc. (1992) 114(25) p. 10024-10035.
    # The parameters in this file are taken from the UFF implementation in RDKit
    # http://openbabel.org/
    #

    def __init__(self, url):
        self.url = url
        
    def fetch(self, url):
        """
        Fetch the UFF force field parameters from the given URL.
        """
        response = requests.get(url)
        if response.status_code == 200:
            content = response.text
        else:
            raise Exception(f"Failed to fetch data from {url}. Status code: {response.status_code}")
        
        with open("UFF.prm", "w") as file:
            file.write(content)

    def parse(self):

        """
        Parse the UFF force field parameters from the fetched file.
        """
        with open("UFF.prm", "r") as file:
            lines = file.readlines()
        
        raw_atom = namedtuple("atom", ["smarts", "type", "descr"])
        raw_param = namedtuple("param", "Atom          r1	theta0	x1	D1	zeta	Z1	Vi	Uj	Xi	Hard	Radius".split())
        atoms = []
        params = []
        for line in lines:
            if line.startswith("#"):
                continue
            elif line.startswith("atom"):
                atom = line.split()
                if len(atom) < 3:
                    atom.append("")
                atoms.append(raw_atom(*params[1:]))
            elif line.startswith("param"):
                param = line.split()
                params.append(raw_param(*param[1:]))

        ff = mp.ForceField()
        atomstyle = ff.def_atomstyle("full")
        for atom in atoms:
            atomstyle.def_type(
                label=atom.type,
                smarts=atom.smarts,
                descr=atom.descr,
            )
        bondstyle = ff.def_bondstyle("harmonic")
        anglestyle = ff.def_anglestyle("harmonic")
        dihestyle = ff.def_dihedralstyle("harmonic")
        pairstyle = ff.def_pairstyle("lj")
        for param in params:
            bondstyle.def_type(
                
            )