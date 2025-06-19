#!/usr/bin/env python3
"""
Test script for __call__ method functionality.
"""

import molpy as mp
import numpy as np

class SPCE(mp.AtomicStructure):
    def __init__(self, name="spce", molid=1):
        super().__init__(name=name)
        o = self.def_atom(
            name="o", molid=molid, type="O", q=-0.8476, xyz=[0.00000, 0.00000, 0.00000]
        )
        h1 = self.def_atom(
            name="h1",
            molid=molid,
            type="H",
            q=0.4238,
            xyz=[0.8164904, 0.5773590, 0.00000],
        )
        h2 = self.def_atom(
            name="h2",
            molid=molid,
            type="H",
            q=0.4238,
            xyz=[-0.8164904, 0.5773590, 0.00000],
        )
        self.def_bond(o, h1)
        self.def_bond(o, h2)

def test_direct_call():
    """Test AtomicStructure.__call__ directly."""
    print("=== Testing AtomicStructure.__call__ directly ===")
    
    # Create original molecule
    mol1 = SPCE(name="mol1", molid=1)
    print(f"Original molecule: {len(mol1.atoms)} atoms, {len(mol1.bonds)} bonds")
    
    # Print original atom properties
    print("Original atoms:")
    for i, atom in enumerate(mol1.atoms):
        print(f"  Atom {i}: name={atom.get('name')}, molid={atom.get('molid')}, q={atom.get('q')}, xyz={atom.get('xyz')}")
    
    # Create copy with different molid
    mol2 = mol1(molid=2)
    print(f"\nCopied molecule: {len(mol2.atoms)} atoms, {len(mol2.bonds)} bonds")
    
    # Print copied atom properties
    print("Copied atoms:")
    for i, atom in enumerate(mol2.atoms):
        print(f"  Atom {i}: name={atom.get('name')}, molid={atom.get('molid')}, q={atom.get('q')}, xyz={atom.get('xyz')}")
    
    # Check if atoms are different objects
    print(f"\nAre atoms different objects? {mol1.atoms._data[0] is not mol2.atoms._data[0]}")
    
    # Move one molecule and check if they're independent
    mol2_wrapper = mp.SpatialWrapper(mol2)
    mol2_wrapper.move([5.0, 5.0, 5.0])
    
    print(f"\nAfter moving mol2:")
    print(f"mol1 first atom xyz: {list(mol1.atoms)[0].get('xyz')}")
    print(f"mol2 first atom xyz: {list(mol2.atoms)[0].get('xyz')}")

def test_wrapper_call():
    """Test SpatialWrapper.__call__."""
    print("\n=== Testing SpatialWrapper.__call__ ===")
    
    # Create wrapped molecule
    mol1 = SPCE(name="mol1", molid=1)
    wrapper1 = mp.SpatialWrapper(mol1)
    
    print(f"Original wrapped molecule: {len(wrapper1._wrapped.atoms)} atoms")
    
    # Create copy using wrapper call
    wrapper2 = wrapper1(molid=2)
    
    print(f"Copied wrapped molecule: {len(wrapper2._wrapped.atoms)} atoms")
    
    # Check if it's properly wrapped
    print(f"Is wrapper2 a SpatialWrapper? {isinstance(wrapper2, mp.SpatialWrapper)}")
    
    # Print atom properties
    print("Original wrapped atoms:")
    for i, atom in enumerate(wrapper1._wrapped.atoms):
        print(f"  Atom {i}: molid={atom.get('molid')}, q={atom.get('q')}")
    
    print("Copied wrapped atoms:")
    for i, atom in enumerate(wrapper2._wrapped.atoms):
        print(f"  Atom {i}: molid={atom.get('molid')}, q={atom.get('q')}")

if __name__ == "__main__":
    test_direct_call()
    test_wrapper_call()
