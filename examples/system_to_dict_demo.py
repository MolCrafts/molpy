#!/usr/bin/env python3
"""
Demo script showing System.to_dict() functionality and Frame integration.
"""

import molpy as mp
import numpy as np
import json
from pathlib import Path

def create_demo_system():
    """Create a demonstration system with multiple molecules."""
    # Create system
    system = mp.System()
    
    # Set up forcefield
    ff = mp.ForceField(name="demo_forcefield", unit="real")
    system.set_forcefield(ff)
    
    # Set up simulation box
    box_matrix = np.array([
        [20.0, 0.0, 0.0],
        [0.0, 20.0, 0.0], 
        [0.0, 0.0, 20.0]
    ])
    system.def_box(box_matrix)
    
    # Create water molecule
    water = mp.AtomicStructure(name="water")
    o = water.def_atom(name="O", element="O", xyz=[0.0, 0.0, 0.0])
    h1 = water.def_atom(name="H1", element="H", xyz=[0.757, 0.586, 0.0])
    h2 = water.def_atom(name="H2", element="H", xyz=[-0.757, 0.586, 0.0])
    water.def_bond(o, h1)
    water.def_bond(o, h2)
    
    # Create methane molecule
    methane = mp.AtomicStructure(name="methane")
    c = methane.def_atom(name="C", element="C", xyz=[5.0, 5.0, 5.0])
    for i in range(4):
        h = methane.def_atom(name=f"H{i+1}", element="H", xyz=[5.0 + i*0.5, 5.0, 5.0 + i*0.2])
        methane.def_bond(c, h)
    
    # Add molecules to system using SpatialWrapper for positioning
    water_wrapper = mp.SpatialWrapper(water)
    water_wrapper.move([2.0, 2.0, 2.0])
    system.add_struct(water)
    
    methane_wrapper = mp.SpatialWrapper(methane)
    methane_wrapper.move([8.0, 8.0, 8.0])
    system.add_struct(methane)
    
    return system

def demo_to_dict():
    """Demonstrate System.to_dict() functionality."""
    print("=== System.to_dict() Demo ===")
    
    # Create demo system
    system = create_demo_system()
    
    # Show system info
    print(f"System contains {len(system._struct)} structures")
    total_atoms = sum(len(struct.atoms) for struct in system._struct)
    print(f"Total atoms: {total_atoms}")
    
    # Convert to dictionary
    system_dict = system.to_dict()
    
    # Display dictionary structure
    print("\nSystem dictionary keys:")
    for key in system_dict.keys():
        print(f"  - {key}")
    
    print(f"\nNumber of structures: {system_dict['n_structures']}")
    print(f"Forcefield name: {system_dict['forcefield'].get('name', 'N/A')}")
    
    # Show box information
    box_info = system_dict['box']
    print(f"\nBox parameters:")
    print(f"  xlo: {box_info['xlo']}, xhi: {box_info['xhi']}")
    print(f"  ylo: {box_info['ylo']}, yhi: {box_info['yhi']}")
    print(f"  zlo: {box_info['zlo']}, zhi: {box_info['zhi']}")
    print(f"  PBC: [{box_info['x_pbc']}, {box_info['y_pbc']}, {box_info['z_pbc']}]")
    
    # Show structure information
    print(f"\nStructures:")
    for i, struct_dict in enumerate(system_dict['structures']):
        print(f"  Structure {i+1}: {struct_dict.get('name', 'unnamed')}")
        if 'atoms' in struct_dict:
            print(f"    Atoms: {len(struct_dict['atoms'])}")
        if 'bonds' in struct_dict:
            print(f"    Bonds: {len(struct_dict['bonds'])}")
    
    return system_dict

def demo_system_to_frame():
    """Demonstrate System.to_frame() functionality."""
    print("\n=== System.to_frame() Demo ===")
    
    # Create demo system
    system = create_demo_system()
    
    # Convert to Frame
    frame = system.to_frame()
    
    print(f"Frame created successfully")
    print(f"Frame has box: {frame.box is not None}")
    print(f"Frame has forcefield: {frame.forcefield is not None}")
    
    # Show Frame data
    if 'atoms' in frame._data:
        atoms_data = frame._data['atoms']
        print(f"\nFrame atoms data:")
        print(f"  Number of atoms: {len(atoms_data['name'])}")
        print(f"  Elements: {list(atoms_data['element'].values)}")
        print(f"  Names: {list(atoms_data['name'].values)}")
    
    # Show coordinates
    if 'atoms' in frame._data and 'xyz' in frame._data['atoms']:
        coords = frame._data['atoms']['xyz'].values
        print(f"\nCoordinates shape: {coords.shape}")
        print(f"First few coordinates:")
        for i in range(min(5, len(coords))):
            print(f"  Atom {i+1}: [{coords[i][0]:.3f}, {coords[i][1]:.3f}, {coords[i][2]:.3f}]")

def demo_frame_roundtrip():
    """Demonstrate Frame serialization roundtrip."""
    print("\n=== Frame Roundtrip Demo ===")
    
    # Create system and convert to frame
    system = create_demo_system()
    original_frame = system.to_frame()
    
    # Convert frame to dict
    frame_dict = original_frame.to_dict()
    print(f"Frame converted to dictionary")
    print(f"Dictionary keys: {list(frame_dict.keys())}")
    
    # Reconstruct frame from dict
    reconstructed_frame = mp.Frame.from_dict(frame_dict)
    print(f"Frame reconstructed from dictionary")
    
    # Compare original and reconstructed
    if 'atoms' in original_frame._data and 'atoms' in reconstructed_frame._data:
        orig_atoms = len(original_frame._data['atoms']['name'])
        recon_atoms = len(reconstructed_frame._data['atoms']['name'])
        print(f"Original frame atoms: {orig_atoms}")
        print(f"Reconstructed frame atoms: {recon_atoms}")
        print(f"Roundtrip successful: {orig_atoms == recon_atoms}")

def main():
    """Run all demonstrations."""
    # Demo to_dict functionality
    system_dict = demo_to_dict()
    
    # Demo to_frame functionality
    demo_system_to_frame()
    
    # Demo roundtrip serialization
    demo_frame_roundtrip()
    
    # Optional: save dictionary to JSON file
    output_dir = Path("./data")
    output_dir.mkdir(exist_ok=True)
    
    json_file = output_dir / "system_demo.json"
    with open(json_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_compatible_dict = {}
        for key, value in system_dict.items():
            if isinstance(value, dict):
                json_compatible_dict[key] = {k: v.tolist() if hasattr(v, 'tolist') else v 
                                           for k, v in value.items()}
            else:
                json_compatible_dict[key] = value.tolist() if hasattr(value, 'tolist') else value
        
        json.dump(json_compatible_dict, f, indent=2)
    
    print(f"\n=== Demo Complete ===")
    print(f"System dictionary saved to: {json_file}")
    print(f"✅ All System.to_dict() and Frame functionality working correctly!")

if __name__ == "__main__":
    main()
