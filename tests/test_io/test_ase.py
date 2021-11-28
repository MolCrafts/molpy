# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-11-17
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np
from molpy.ioapi import fromASE, fromASE_S, fromCIF, fromCIF_S, toASE
from pathlib import Path
from os.path import join

try:
    import ase.build.molecule                     
except ImportError:
    pass;

SAMPLE_DIR = join(Path(__file__).resolve().parent.parent, "samples")
class TestASE:
    def test_from_ase(self):
        mol = ase.build.molecule("bicyclobutane")
        g   = fromASE(mol)
        symbols = mol.get_chemical_symbols()
        g_symbols = g.getSymbols()
        assert len(symbols) == len(g_symbols)
        assert all([a == b for a, b in zip(symbols, g_symbols)])    
        dpos = g.positions - mol.get_positions()
        assert not np.any(dpos)

    def test_from_ase(self):
        CC_bond = 1.42
        GRA = ase.build.graphene_nanoribbon(10, 10, vacuum = CC_bond/2)
        GRA.cell[1] = [0, 20, 0]
        GRA.positions[:,1] = 10
        sys = fromASE_S(GRA)
        assert sys.name == GRA.get_chemical_formula()
        assert sys.cell.volume == pytest.approx(GRA.cell.volume, 1e-8)
        assert sys.atoms[0].name == "C"

    def test_from_CIF(self):
        fpath = join(SAMPLE_DIR,"UiO-66_vesta.cif")
        MOF = fromCIF(fpath)
        assert MOF[0].name == "Zr1"
        assert MOF[100].name == "O3"

        MOF = fromCIF(join(SAMPLE_DIR,"UiO-66_all.cif"))
        assert MOF[10].name == "Zr11"
        assert MOF[100].name == "O77"

        MOF = fromCIF(fpath, fromLabel = False)
        assert MOF[10].name == "Zr"
        assert MOF[100].name == "O"

    def test_from_CIF_S(self):
        fpath = join(SAMPLE_DIR,"UiO-66_vesta.cif")
        MOF = fromCIF_S(fpath)
        ASE_atoms = ase.io.read(fpath)
        MOF_atom = MOF.atoms
        assert MOF_atom[0].name == "Zr1"
        assert MOF_atom[100].name == "O3"
        assert MOF.name == "VESTA_phase_1" # "C192H128O128Zr24"
        assert MOF.cell.volume == pytest.approx(ASE_atoms.cell.volume, 1e-8)
        # g = from

        ASE_atoms = toASE(MOF)
        assert ASE_atoms[0].symbol == "Zr"
        assert ASE_atoms[100].symbol == "O"
        assert MOF.cell.volume == pytest.approx(ASE_atoms.cell.volume, 1e-8)
        ASE_atoms = toASE(MOF.groups[0])
        assert ASE_atoms.cell.volume == 0.0

        xyz = MOF.getPositions()
        set_value =  [4.4, 5.5, 6.6]
        xyz[3]    =  set_value
        MOF.setPositions(xyz)
        with pytest.raises(AssertionError):
            MOF.setPositions(xyz[:-2])
        assert not np.any(MOF.atoms[3].position - set_value)
        symbols = MOF.getSymbols()
        assert symbols[0] == "Zr"
        assert symbols[100] == "O"
    

