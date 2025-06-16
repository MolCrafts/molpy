# Complete Frame Module Tutorial

## Overview

The Frame module provides efficient tabular data storage and manipulation functionality based on xarray, specifically designed for handling molecular data batch operations and analysis. Frame is similar to pandas DataFrame but optimized for molecular data, supporting multi-dimensional data (like coordinates, tensors) and bidirectional conversion with Struct objects.

## 1. Frame Fundamentals

### 1.1 What is Frame

Frame is a container in MolPy for storing and manipulating structured molecular data, built on xarray.DataArray:

```python
import molpy as mp
import numpy as np

# Create basic Frame
atom_data = {
    'name': ['C1', 'C2', 'O1', 'H1'],
    'element': ['C', 'C', 'O', 'H'],
    'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2.5,0,0]],
    'charge': [0.0, 0.0, -0.4, 0.1]
}

frame = mp.Frame(atoms=atom_data)
print("Frame created successfully")
print(f"Frame keys: {list(frame.keys())}")
```

**Core Features of Frame:**
- Efficient storage based on xarray.DataArray
- Support for multi-dimensional data (coordinates, tensors, etc.)
- Automatic handling of scalar and vector data
- Bidirectional conversion with Struct objects
- Efficient array operations and broadcasting

### 1.2 Frame Structure

Frame is essentially a dictionary that can contain various types of data:

```python
# Create complex Frame
complex_frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'N1'],
        'element': ['C', 'C', 'N'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0]],
        'mass': [12.01, 12.01, 14.01],
        'velocity': [[0.1,0,0], [0,0.1,0], [0,0,0.1]],
        'charge': [-0.1, 0.1, -0.3]
    },
    bonds={
        'atom1_idx': [0, 1],
        'atom2_idx': [1, 2], 
        'bond_type': ['single', 'single'],
        'length': [1.5, 1.4]
    }
)

print(f"Atoms shape: {complex_frame['atoms'].shape}")
print(f"Bonds shape: {complex_frame['bonds'].shape}")
```

**Frame Components:**
- **atoms**: Atom-related data (coordinates, properties, etc.)
- **bonds**: Bond connectivity and properties
- **angles**: Angular geometric information
- **dihedrals**: Dihedral angle data
- **Custom fields**: User-defined data arrays

## 2. Data Access and Manipulation

### 2.1 Basic Data Access

```python
# Create test Frame
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1'],
        'element': ['C', 'C', 'O'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0]],
        'mass': [12.01, 12.01, 15.99]
    }
)

# Access atoms data
atoms_data = frame['atoms']
print(f"Atoms DataArray: {type(atoms_data)}")

# Access specific properties
names = frame['atoms'].coords['name']
coordinates = frame['atoms'].coords['xyz']
print(f"Names: {names.values}")
print(f"Coordinates shape: {coordinates.shape}")

# Index-based access
first_atom_coord = coordinates[0].values
print(f"First atom coordinates: {first_atom_coord}")
```

### 2.2 Data Filtering and Selection

```python
# Create larger dataset
large_frame = mp.Frame(
    atoms={
        'name': [f'A{i}' for i in range(10)],
        'element': ['C']*5 + ['N']*3 + ['O']*2,
        'xyz': [[i, 0, 0] for i in range(10)],
        'charge': np.random.randn(10) * 0.5
    }
)

# Filter by element
carbons = large_frame['atoms'].where(
    large_frame['atoms'].coords['element'] == 'C', 
    drop=True
)
print(f"Carbon atoms: {len(carbons.atom)}")

# Combined filters
carbon_positive = large_frame['atoms'].where(
    (large_frame['atoms'].coords['element'] == 'C') & 
    (large_frame['atoms'].coords['charge'] > 0),
    drop=True
)
print(f"Positive carbon atoms: {len(carbon_positive.atom)}")
```

### 2.3 Data Transformation

```python
# Mathematical operations
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'C3'],
        'xyz': [[0,0,0], [1,0,0], [2,0,0]],
        'mass': [12.01, 12.01, 12.01]
    }
)

# Coordinate transformations
coords = frame['atoms'].coords['xyz']

# Center coordinates
center = coords.mean(dim='atom')
centered_coords = coords - center
print(f"Original center: {center.values}")
print(f"New center: {centered_coords.mean(dim='atom').values}")

# Add new calculated properties
distances = np.linalg.norm(coords.values, axis=1)
frame['atoms'] = frame['atoms'].assign_coords(
    distance_from_origin=('atom', distances)
)
```

## 3. Frame-Struct Integration

### 3.1 Converting Struct to Frame

```python
# Create Struct object
water = mp.AtomicStructure(name="water")
o = water.def_atom(name="O", element="O", xyz=[0, 0, 0], charge=-0.834)
h1 = water.def_atom(name="H1", element="H", xyz=[0.757, 0.586, 0], charge=0.417)
h2 = water.def_atom(name="H2", element="H", xyz=[-0.757, 0.586, 0], charge=0.417)

# Add bonds
bond1 = water.def_bond(o, h1, bond_type="covalent", length=0.96)
bond2 = water.def_bond(o, h2, bond_type="covalent", length=0.96)

# Convert to Frame
frame = water.to_frame()
print(f"Frame keys: {list(frame.keys())}")
print(f"Atoms: {len(frame['atoms'].atom)}")
print(f"Bonds: {len(frame['bonds'].bond)}")
```

### 3.2 Converting Frame to Struct

```python
# Create Frame data
frame_data = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'C3', 'C4'],
        'element': ['C', 'C', 'C', 'C'],
        'xyz': [[0,0,0], [1.5,0,0], [1.5,1.5,0], [0,1.5,0]]
    },
    bonds={
        'atom1_idx': [0, 1, 2, 3],
        'atom2_idx': [1, 2, 3, 0],
        'bond_type': ['single', 'single', 'single', 'single']
    }
)

# Convert to Struct
cyclobutane = frame_data.to_structure(name="cyclobutane")
print(f"Structure name: {cyclobutane['name']}")
print(f"Atoms: {len(cyclobutane.atoms)}")
print(f"Bonds: {len(cyclobutane.bonds)}")
```

## 4. Advanced Frame Operations

### 4.1 Time Series and Trajectories

```python
# Simulate trajectory data
n_frames = 100
n_atoms = 3

# Generate trajectory
trajectory = np.random.randn(n_frames, n_atoms, 3) * 0.1

# Create trajectory Frame
traj_frame = mp.Frame(
    atoms={
        'name': ['O', 'H1', 'H2'],
        'element': ['O', 'H', 'H'],
        'mass': [15.999, 1.008, 1.008]
    },
    trajectory={
        'xyz': trajectory,  # Shape: (frame, atom, spatial)
        'time': np.arange(n_frames) * 0.1  # ps
    }
)

print(f"Trajectory shape: {traj_frame['trajectory'].coords['xyz'].shape}")

# Analyze trajectory
positions = traj_frame['trajectory'].coords['xyz']
ref_pos = positions[0]  # First frame as reference
rmsd_values = []

for frame_idx in range(n_frames):
    current_pos = positions[frame_idx]
    rmsd = np.sqrt(np.mean((current_pos - ref_pos)**2))
    rmsd_values.append(rmsd)

print(f"RMSD range: {min(rmsd_values):.3f} - {max(rmsd_values):.3f}")
```

### 4.2 Statistical Analysis

```python
# Create ensemble data
n_structures = 50
ensemble_frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1'] * n_structures,
        'element': ['C', 'C', 'O'] * n_structures,
        'structure_id': np.repeat(range(n_structures), 3),
        'xyz': np.random.randn(n_structures * 3, 3),
        'energy': np.random.randn(n_structures * 3) * 10
    }
)

# Group by structure
grouped = ensemble_frame['atoms'].groupby('structure_id')

# Calculate statistics per structure
structure_energies = grouped.sum('energy')
print(f"Energy statistics:")
print(f"  Mean: {structure_energies.mean().values:.2f}")
print(f"  Std: {structure_energies.std().values:.2f}")
```

### 4.3 Property Calculations

```python
# Calculate molecular properties
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'C3', 'H1', 'H2', 'H3'],
        'element': ['C', 'C', 'C', 'H', 'H', 'H'],
        'xyz': [[0,0,0], [1.5,0,0], [3,0,0], [0,1,0], [1.5,1,0], [3,1,0]],
        'mass': [12.01, 12.01, 12.01, 1.008, 1.008, 1.008]
    }
)

# Calculate center of mass
coords = frame['atoms'].coords['xyz'].values
masses = frame['atoms'].coords['mass'].values
total_mass = np.sum(masses)
com = np.sum(coords * masses[:, np.newaxis], axis=0) / total_mass

print(f"Center of mass: {com}")

# Calculate radius of gyration
com_distances = coords - com
rg_squared = np.sum(masses[:, np.newaxis] * com_distances**2) / total_mass
rg = np.sqrt(np.sum(rg_squared))

print(f"Radius of gyration: {rg:.3f} Å")
```

## 5. Performance Optimization

### 5.1 Efficient Data Operations

```python
# Large system performance
n_atoms = 10000
large_frame = mp.Frame(
    atoms={
        'name': [f'A{i}' for i in range(n_atoms)],
        'element': np.random.choice(['C', 'N', 'O', 'H'], n_atoms),
        'xyz': np.random.randn(n_atoms, 3) * 10,
        'charge': np.random.randn(n_atoms) * 0.5
    }
)

# Efficient operations using xarray
import time

# Vectorized distance calculation
start_time = time.time()
coords = large_frame['atoms'].coords['xyz']
distances = np.linalg.norm(coords.values, axis=1)
vectorized_time = time.time() - start_time

print(f"Vectorized calculation time: {vectorized_time:.4f} seconds")

# Efficient filtering
heavy_atoms = large_frame['atoms'].where(
    large_frame['atoms'].coords['element'] != 'H',
    drop=True
)
print(f"Heavy atoms: {len(heavy_atoms.atom)} / {n_atoms}")
```

### 5.2 Memory Management

```python
# Memory-efficient operations for large datasets
def process_large_trajectory(n_frames, n_atoms):
    """Process trajectory in chunks to manage memory."""
    chunk_size = 100
    
    for start_idx in range(0, n_frames, chunk_size):
        end_idx = min(start_idx + chunk_size, n_frames)
        
        # Load chunk
        chunk_coords = np.random.randn(end_idx - start_idx, n_atoms, 3)
        
        # Process chunk
        chunk_com = np.mean(chunk_coords, axis=1)
        
        # Store results
        print(f"Processed frames {start_idx}-{end_idx}")
        
        # Cleanup
        del chunk_coords, chunk_com

# Demonstrate
process_large_trajectory(1000, 5000)
```

## 6. Integration with Analysis Tools

### 6.1 NumPy Integration

```python
# Seamless NumPy integration
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'C3'],
        'xyz': [[0,0,0], [1,0,0], [2,0,0]],
        'mass': [12.01, 12.01, 12.01]
    }
)

# Extract as NumPy arrays
coords_array = frame['atoms'].coords['xyz'].values
masses_array = frame['atoms'].coords['mass'].values

print(f"Coordinates type: {type(coords_array)}")
print(f"Shape: {coords_array.shape}")

# NumPy operations
distances = np.linalg.norm(coords_array, axis=1)
com = np.average(coords_array, weights=masses_array, axis=0)

# Add results back to Frame
frame['atoms'] = frame['atoms'].assign_coords(
    distance=('atom', distances),
    com_offset=('atom', coords_array - com)
)
```

### 6.2 Pandas Integration

```python
# Convert to pandas for advanced data analysis
frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'O1', 'H1', 'H2'],
        'element': ['C', 'C', 'O', 'H', 'H'],
        'xyz': [[0,0,0], [1.5,0,0], [0,1.5,0], [2,0,0], [0,2,0]],
        'residue': ['RES1', 'RES1', 'RES1', 'RES2', 'RES2'],
        'charge': [-0.1, 0.1, -0.4, 0.2, 0.2]
    }
)

# Convert to pandas DataFrame
df = frame['atoms'].to_dataframe()
print(f"DataFrame shape: {df.shape}")

# Pandas operations
residue_stats = df.groupby('residue').agg({
    'charge': ['mean', 'sum'],
    'element': 'count'
})
print("Residue statistics:")
print(residue_stats)
```

## 7. Best Practices

### 7.1 Data Organization

```python
# Good: Consistent data organization
good_frame = mp.Frame(
    atoms={
        'name': ['C1', 'C2', 'C3'],
        'element': ['C', 'C', 'C'],
        'xyz': [[0,0,0], [1,0,0], [2,0,0]],
        'atom_type': ['C.3', 'C.3', 'C.3']
    },
    bonds={
        'atom1_idx': [0, 1],
        'atom2_idx': [1, 2],
        'bond_type': ['single', 'single']
    }
)
```

### 7.2 Error Handling

```python
try:
    frame = mp.Frame(
        atoms={
            'name': ['C1', 'C2'],
            'xyz': [[0,0,0], [1,0,0]]
        }
    )
    
    # Safe property access
    if 'charge' in frame['atoms'].coords:
        charges = frame['atoms'].coords['charge']
    else:
        charges = np.zeros(len(frame['atoms'].atom))
        
except ValueError as e:
    print(f"Frame creation error: {e}")
```

### 7.3 Performance Tips

```python
# 1. Use vectorized operations
coords = frame['atoms'].coords['xyz'].values
distances = np.linalg.norm(coords, axis=1)  # Vectorized

# 2. Pre-allocate arrays for large datasets
n_atoms = 10000
results = np.empty(n_atoms)

# 3. Use appropriate data types
frame['atoms'] = frame['atoms'].assign_coords(
    int_property=('atom', np.array([1, 2, 3], dtype=np.int32)),
    float_property=('atom', np.array([1.0, 2.0, 3.0], dtype=np.float32))
)
```

## 8. Advanced Use Cases

### 8.1 Molecular Dynamics Analysis

```python
# MD trajectory analysis
def analyze_md_trajectory(trajectory_frame):
    """Analyze MD trajectory data."""
    coords = trajectory_frame['trajectory'].coords['xyz']
    n_frames, n_atoms, _ = coords.shape
    
    # Calculate RMSD over time
    ref_coords = coords[0]
    rmsd_values = []
    
    for frame in range(n_frames):
        diff = coords[frame] - ref_coords
        rmsd = np.sqrt(np.mean(diff**2))
        rmsd_values.append(rmsd)
    
    return np.array(rmsd_values)

# Example usage
n_frames = 1000
n_atoms = 100
traj = mp.Frame(
    atoms={
        'name': [f'A{i}' for i in range(n_atoms)],
        'mass': np.random.uniform(1, 16, n_atoms)
    },
    trajectory={
        'xyz': np.random.randn(n_frames, n_atoms, 3),
        'time': np.arange(n_frames) * 0.001
    }
)

rmsd_results = analyze_md_trajectory(traj)
print(f"RMSD range: {rmsd_results.min():.3f} - {rmsd_results.max():.3f}")
```

### 8.2 Quantum Chemistry Data

```python
# QC calculation results
qc_frame = mp.Frame(
    atoms={
        'name': ['C', 'O', 'H1', 'H2'],
        'element': ['C', 'O', 'H', 'H'],
        'xyz': [[0,0,0], [1.2,0,0], [-1,0.5,0], [-1,-0.5,0]],
        'mulliken_charge': [0.1, -0.3, 0.1, 0.1]
    },
    orbitals={
        'orbital_id': ['HOMO-1', 'HOMO', 'LUMO', 'LUMO+1'],
        'energy': [-0.45, -0.35, -0.05, 0.05],
        'occupation': [2.0, 2.0, 0.0, 0.0]
    }
)

# Analyze electronic properties
homo_energy = qc_frame['orbitals'].where(
    qc_frame['orbitals'].coords['orbital_id'] == 'HOMO'
).coords['energy'].values[0]

lumo_energy = qc_frame['orbitals'].where(
    qc_frame['orbitals'].coords['orbital_id'] == 'LUMO'
).coords['energy'].values[0]

homo_lumo_gap = lumo_energy - homo_energy
print(f"HOMO-LUMO gap: {homo_lumo_gap:.3f} Hartree")
```

## Summary

The Frame module provides powerful capabilities for molecular data management:

- **Efficient Storage**: xarray-based multi-dimensional data handling
- **Flexible Structure**: Support for various molecular data types
- **Analysis Ready**: Integration with NumPy, pandas, and analysis workflows
- **Scalable**: Optimized for both small molecules and large systems
- **Interoperable**: Seamless conversion with Struct objects

Frame excels in scenarios requiring bulk data operations, statistical analysis, and integration with computational chemistry workflows. Combined with Struct for detailed molecular building, it provides a complete solution for molecular data management.
