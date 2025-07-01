import pytest
import numpy as np
import molpy as mp

from molpy.core.forcefield import KernelMeta, ForceField
from molpy.potential import Potential

class TestRegisterPotential:

    def test_metaclass_register(self):

        class MetaclassRegisterPotential(metaclass=KernelMeta):
            name = "PotentialA"

        assert ForceField._kernel_registry["PotentialA"] is MetaclassRegisterPotential
    
    def test_base_potential_register(self):

        class BasePotential(Potential):
            name = "BasePotential"

        assert ForceField._kernel_registry["BasePotential"] is BasePotential

    def test_type_potential_register(self):

        class TypedPotential(Potential):
            name = "harmonic"
            type = "angle"

        assert ForceField._kernel_registry["angle"]["harmonic"] is TypedPotential