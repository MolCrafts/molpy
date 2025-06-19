import pytest
import numpy as np
import molpy as mp


class TestAtomicStructure:
    """Test AtomicStructure class functionality."""

    def test_basic_creation(self):
        """Test basic AtomicStructure creation."""
        struct = mp.AtomicStructure("test_structure")
        assert struct.get("name") == "test_structure"
        assert len(struct.atoms) == 0
        assert len(struct.bonds) == 0
        assert len(struct.angles) == 0

    def test_add_atoms(self):
        """Test adding atoms to structure."""
        struct = mp.AtomicStructure("test")
        
        # Add atoms using def_atom
        o = struct.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = struct.def_atom(name="H1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0])
        h2 = struct.def_atom(name="H2", type="H", q=0.4238, xyz=[0.0, 1.0, 0.0])
        
        assert len(struct.atoms) == 3
        assert o.get("name") == "O"
        assert h1.get("type") == "H"
        assert np.allclose(h2.xyz, [0.0, 1.0, 0.0])

    def test_add_bonds(self):
        """Test adding bonds to structure."""
        struct = mp.AtomicStructure("test")
        
        o = struct.def_atom(name="O", type="O")
        h1 = struct.def_atom(name="H1", type="H")
        h2 = struct.def_atom(name="H2", type="H")
        
        # Add bond
        bond1 = struct.def_bond(o, h1)
        bond2 = struct.def_bond(o, h2)
        
        assert len(struct.bonds) == 2
        assert bond1.atom1 == o or bond1.atom2 == o
        assert bond2.atom1 == o or bond2.atom2 == o

    def test_add_angles(self):
        """Test adding angles to structure."""
        struct = mp.AtomicStructure("test")
        
        o = struct.def_atom(name="O", type="O")
        h1 = struct.def_atom(name="H1", type="H")
        h2 = struct.def_atom(name="H2", type="H")
        
        # Add angle
        angle = mp.Angle(h1, o, h2, theta0=109.47, k=1000.0)
        struct.add_angle(angle)
        
        assert len(struct.angles) == 1
        assert angle.jtom == o  # Central atom

    def test_xyz_property(self):
        """Test xyz property for getting/setting all coordinates."""
        struct = mp.AtomicStructure("test")
        
        struct.def_atom(name="A", xyz=[0.0, 0.0, 0.0])
        struct.def_atom(name="B", xyz=[1.0, 1.0, 1.0])
        struct.def_atom(name="C", xyz=[2.0, 2.0, 2.0])
        
        # Test getter
        coords = struct.xyz
        expected = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
        assert np.allclose(coords, expected)
        
        # Test setter
        new_coords = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        struct.xyz = new_coords
        
        assert np.allclose(struct.atoms[0].xyz, [3.0, 3.0, 3.0])
        assert np.allclose(struct.atoms[1].xyz, [4.0, 4.0, 4.0])
        assert np.allclose(struct.atoms[2].xyz, [5.0, 5.0, 5.0])

    def test_to_frame(self):
        """Test converting structure to frame."""
        struct = mp.AtomicStructure("test")
        
        o = struct.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = struct.def_atom(name="H1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0])
        struct.def_bond(o, h1)
        
        frame = struct.to_frame()
        
        assert "atoms" in frame
        assert "bonds" in frame
        
        # Check atoms data
        atoms = frame["atoms"]
        assert "q" in atoms.data_vars
        assert "xyz" in atoms.data_vars or ("x" in atoms.data_vars and "y" in atoms.data_vars and "z" in atoms.data_vars)
        
        # Check bonds data
        bonds = frame["bonds"]
        assert "i" in bonds.data_vars
        assert "j" in bonds.data_vars

    def test_call_method_deep_copy(self):
        """Test that __call__() method creates proper deep copies."""
        # Create template structure
        template = mp.AtomicStructure("template", molid=1)
        o = template.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0], molid=1)
        h1 = template.def_atom(name="H1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0], molid=1)
        h2 = template.def_atom(name="H2", type="H", q=0.4238, xyz=[0.0, 1.0, 0.0], molid=1)
        template.def_bond(o, h1)
        template.def_bond(o, h2)
        angle = mp.Angle(h1, o, h2, theta0=109.47)
        template.add_angle(angle)
        
        # Create copy using __call__()
        copy1 = template(molid=2)
        
        # Verify it's a deep copy
        assert copy1 is not template
        assert len(copy1.atoms) == 3
        assert len(copy1.bonds) == 2
        assert len(copy1.angles) == 1
        
        # Verify atoms are different objects
        for orig_atom, copy_atom in zip(template.atoms, copy1.atoms):
            assert orig_atom is not copy_atom
            assert orig_atom.get("name") == copy_atom.get("name")
            assert orig_atom.get("type") == copy_atom.get("type")
            assert orig_atom.get("q") == copy_atom.get("q")
            # molid should be updated
            assert copy_atom.get("molid") == 2
        
        # Verify bonds are different objects but reference the new atoms
        assert template.bonds[0] is not copy1.bonds[0]
        # The bond in copy1 should reference atoms from copy1, not template
        copy_bond = copy1.bonds[0]
        assert copy_bond.atom1 in copy1.atoms
        assert copy_bond.atom2 in copy1.atoms

    def test_coordinate_independence_after_copy(self):
        """Test that coordinates are independent after copying."""
        template = mp.AtomicStructure("template")
        template.def_atom(name="A", xyz=[0.0, 0.0, 0.0])
        template.def_atom(name="B", xyz=[1.0, 1.0, 1.0])
        
        # Create copy
        copy1 = template()
        
        # Modify template coordinates
        template.atoms[0].xyz = [5.0, 5.0, 5.0]
        
        # Copy coordinates should be unchanged
        assert np.allclose(copy1.atoms[0].xyz, [0.0, 0.0, 0.0])
        assert np.allclose(copy1.atoms[1].xyz, [1.0, 1.0, 1.0])
        
        # Modify copy coordinates
        copy1.atoms[1].xyz = [10.0, 10.0, 10.0]
        
        # Template coordinates should be unchanged (except the one we modified earlier)
        assert np.allclose(template.atoms[0].xyz, [5.0, 5.0, 5.0])
        assert np.allclose(template.atoms[1].xyz, [1.0, 1.0, 1.0])

    def test_spatial_wrapper_with_deep_copy(self):
        """Test that SpatialWrapper works correctly with deep copied structures."""
        template = mp.AtomicStructure("template")
        template.def_atom(name="A", xyz=[0.0, 0.0, 0.0])
        template.def_atom(name="B", xyz=[1.0, 0.0, 0.0])
        
        # Create copy and wrap with SpatialWrapper
        copy1 = template()
        spatial_copy = mp.SpatialWrapper(copy1)
        
        # Move the copy
        spatial_copy.move([3.0, 0.0, 0.0])
        
        # Template should be unchanged
        assert np.allclose(template.atoms[0].xyz, [0.0, 0.0, 0.0])
        assert np.allclose(template.atoms[1].xyz, [1.0, 0.0, 0.0])
        
        # Copy should be moved
        assert np.allclose(copy1.atoms[0].xyz, [3.0, 0.0, 0.0])
        assert np.allclose(copy1.atoms[1].xyz, [4.0, 0.0, 0.0])

    def test_multiple_independent_copies(self):
        """Test creating multiple independent copies."""
        template = mp.AtomicStructure("template")
        o = template.def_atom(name="O", xyz=[0.0, 0.0, 0.0])
        h = template.def_atom(name="H", xyz=[1.0, 0.0, 0.0])
        template.def_bond(o, h)
        
        # Create multiple copies
        copies = []
        for i in range(5):
            copy = template(molid=i+1)
            copies.append(copy)
        
        # Verify all are independent
        for i, copy in enumerate(copies):
            assert len(copy.atoms) == 2
            assert len(copy.bonds) == 1
            
            # Modify this copy's coordinates to unique values
            unique_coord = [i+10, i+10, i+10]  # Use unique values to avoid conflicts
            copy.atoms[0].xyz = unique_coord
            
            # Verify other copies are unchanged
            for j, other_copy in enumerate(copies):
                if i != j:
                    assert not np.allclose(other_copy.atoms[0].xyz, unique_coord)

    def test_bond_angle_references_after_copy(self):
        """Test that bonds and angles reference the correct atoms after copying."""
        template = mp.AtomicStructure("template")
        
        o = template.def_atom(name="O", xyz=[0.0, 0.0, 0.0])
        h1 = template.def_atom(name="H1", xyz=[1.0, 0.0, 0.0])
        h2 = template.def_atom(name="H2", xyz=[0.0, 1.0, 0.0])
        
        bond1 = template.def_bond(o, h1)
        bond2 = template.def_bond(o, h2)
        angle = mp.Angle(h1, o, h2)
        template.add_angle(angle)
        
        # Create copy
        copy = template()
        
        # Verify bonds in copy reference atoms from copy, not template
        copy_bond1 = copy.bonds[0]
        copy_bond2 = copy.bonds[1]
        
        assert copy_bond1.atom1 in copy.atoms
        assert copy_bond1.atom2 in copy.atoms
        assert copy_bond2.atom1 in copy.atoms
        assert copy_bond2.atom2 in copy.atoms
        
        # Verify atoms are not from template
        assert copy_bond1.atom1 not in template.atoms
        assert copy_bond1.atom2 not in template.atoms
        
        # Verify angles in copy reference atoms from copy
        copy_angle = copy.angles[0]
        assert copy_angle.itom in copy.atoms
        assert copy_angle.jtom in copy.atoms
        assert copy_angle.ktom in copy.atoms

    def test_waterbox_integration_workflow(self):
        """Test the waterbox-like scenario that was originally failing."""
        # Create template water molecule
        template = mp.AtomicStructure("water_template", molid=1)
        o = template.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = template.def_atom(name="H1", type="H", q=0.4238, xyz=[0.816, 0.577, 0.0])
        h2 = template.def_atom(name="H2", type="H", q=0.4238, xyz=[-0.816, 0.577, 0.0])
        template.def_bond(o, h1)
        template.def_bond(o, h2)
        angle = mp.Angle(h1, o, h2, theta0=109.47)
        template.add_angle(angle)
        
        # Create system
        system = mp.System()
        system.def_box(np.diag([10.0, 10.0, 10.0]))
        
        # Add multiple water molecules
        n_molecules = 0
        for i in range(2):
            for j in range(2):
                molid = n_molecules + 1
                
                # Create new molecule
                water = template(molid=molid)
                
                # Update molid for all atoms
                for atom in water.atoms:
                    atom['molid'] = molid
                
                # Move molecule
                position = [i * 3.0, j * 3.0, 0.0]
                spatial_water = mp.SpatialWrapper(water)
                spatial_water.move(position)
                
                # Add to system
                system.add_struct(water)
                n_molecules += 1
        
        # Verify system
        assert n_molecules == 4
        total_atoms = sum(len(struct.atoms) for struct in system._struct)
        total_bonds = sum(len(struct.bonds) for struct in system._struct)
        total_angles = sum(len(struct.angles) for struct in system._struct)
        
        assert total_atoms == 12  # 3 atoms per molecule * 4 molecules
        assert total_bonds == 8   # 2 bonds per molecule * 4 molecules
        assert total_angles == 4  # 1 angle per molecule * 4 molecules
        
        # Convert to frame and verify
        frame = system.to_frame()
        
        # Check atom count
        n_atoms_frame = 0
        if "atoms" in frame:
            atoms_data = frame["atoms"]
            if hasattr(atoms_data, 'sizes'):
                for dim_name in ['index', 'dim_id_0', 'dim_q_0', 'dim_xyz_0']:
                    if dim_name in atoms_data.sizes:
                        n_atoms_frame = atoms_data.sizes[dim_name]
                        break
        
        assert n_atoms_frame == 12
        
        # Check that coordinates are different for different molecules
        if "xyz" in frame["atoms"]:
            coords = frame["atoms"]["xyz"].values
        else:
            coords = np.column_stack([
                frame["atoms"]["x"].values,
                frame["atoms"]["y"].values,
                frame["atoms"]["z"].values
            ])
        
        # Should have 4 distinct molecule positions
        # Check that not all atoms are at the same position
        unique_positions = np.unique(coords, axis=0)
        assert len(unique_positions) > 4  # Should have more than 4 unique positions
