#!/usr/bin/env python3
"""
Test System.to_dict functionality and Frame integration
"""

import molpy as mp
import numpy as np
import json
from pathlib import Path

def test_system_to_dict():
    """Test System.to_dict functionality"""
    print("=== Testing System.to_dict ===")
    
    # Create a simple system
    system = mp.System()
    
    # Set forcefield
    ff = mp.ForceField(name="test_ff", unit="real")
    system.set_forcefield(ff)
    
    # Set box
    box_matrix = np.diag([10.0, 10.0, 10.0])
    system.def_box(box_matrix)
    
    # Create a simple molecule
    mol = mp.AtomicStructure(name="test_mol")
    mol.def_atom(name="C1", element="C", xyz=[0, 0, 0], molid=1)
    mol.def_atom(name="H1", element="H", xyz=[1, 0, 0], molid=1)
    mol.def_atom(name="H2", element="H", xyz=[0, 1, 0], molid=1)
    
    system.add_struct(mol)
    
    # Test to_dict
    system_dict = system.to_dict()
    
    print(f"System dict keys: {list(system_dict.keys())}")
    print(f"Number of structures: {system_dict['n_structures']}")
    print(f"Forcefield name: {system_dict['forcefield'].get('name', 'N/A')}")
    print(f"Box keys: {list(system_dict['box'].keys())}")
    
    # Test Frame conversion
    frame = system.to_frame()
    print(f"Frame has {len(frame['atoms'])} atoms")
    
    # Test Frame to_dict
    frame_dict = frame.to_dict()
    print(f"Frame dict keys: {list(frame_dict.keys())}")
    
    # Save to JSON for inspection
    output_dir = Path("data/test_output")
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "system_dict.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        json_dict = convert_for_json(system_dict)
        json.dump(json_dict, f, indent=2)
    
    print(f"✅ System dict saved to {output_dir / 'system_dict.json'}")
    
    return system, frame

def convert_for_json(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, dict):
        return {k: convert_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_for_json(item) for item in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif hasattr(obj, '__dict__'):
        # Handle custom objects by converting to dict
        return f"<{obj.__class__.__name__} object>"
    else:
        return str(obj)

def test_spce_molecule():
    """Test SPCE molecule creation with proper molid"""
    print("\n=== Testing SPCE Molecule ===")
    
    # Create SPCE molecule with specific molid
    mol = mp.AtomicStructure(name="spce_mol")
    molid = 42  # Test with specific molid
    
    o = mol.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0], molid=molid, q=-0.8476)
    h1 = mol.def_atom(name="H1", element="H", xyz=[0.8164904, 0.5773590, 0.0], molid=molid, q=0.4238)
    h2 = mol.def_atom(name="H2", element="H", xyz=[-0.8164904, 0.5773590, 0.0], molid=molid, q=0.4238)
    
    mol.def_bond(o, h1)
    mol.def_bond(o, h2)
    
    print(f"SPCE molecule created with molid={molid}")
    print(f"Atoms: {len(mol.atoms)}")
    print(f"Bonds: {len(mol.bonds)}")
    
    # Check atom molids
    for atom in mol.atoms:
        print(f"  Atom {atom['name']}: molid={atom.get('molid', 'N/A')}")
    
    # Create system with this molecule
    system = mp.System()
    system.add_struct(mol)
    
    # Convert to frame
    frame = system.to_frame()
    atoms_data = frame['atoms']
    
    print(f"Frame atoms: {len(atoms_data)}")
    if 'molid' in atoms_data:
        print(f"Molids in frame: {atoms_data['molid'].values}")
    else:
        print("No molid field in frame")
    
    return mol, system, frame

if __name__ == "__main__":
    try:
        system, frame = test_system_to_dict()
        mol, system2, frame2 = test_spce_molecule()
        print("\n🎉 All tests completed successfully!")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
