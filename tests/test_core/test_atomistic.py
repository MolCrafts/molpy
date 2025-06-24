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
        assert np.allclose(h2["xyz"], [0.0, 1.0, 0.0])

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
        assert bond1.itom == o or bond1.jtom == o
        assert bond2.itom == o or bond2.jtom == o

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
        coords = struct["atoms", "xyz"]
        expected = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        assert coords == expected
        
        # Test setter using consistent syntax
        new_coords = np.array([[3.0, 3.0, 3.0], [4.0, 4.0, 4.0], [5.0, 5.0, 5.0]])
        struct["atoms", "xyz"] = new_coords
        
        assert np.allclose(struct.atoms[0]["xyz"], [3.0, 3.0, 3.0])
        assert np.allclose(struct.atoms[1]["xyz"], [4.0, 4.0, 4.0])
        assert np.allclose(struct.atoms[2]["xyz"], [5.0, 5.0, 5.0])

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

    def test_from_frame(self):
        """Test reconstructing structure from frame."""
        # Create original structure
        original = mp.AtomicStructure("test_molecule")
        o = original.def_atom(name="O", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0], mass=15.999)
        h1 = original.def_atom(name="H1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0], mass=1.008)
        h2 = original.def_atom(name="H2", type="H", q=0.4238, xyz=[0.0, 1.0, 0.0], mass=1.008)
        original.def_bond(o, h1)
        original.def_bond(o, h2)
        
        # Convert to frame
        frame = original.to_frame()
        
        # Reconstruct from frame using class method
        reconstructed = mp.AtomicStructure.from_frame(frame)
        
        # Verify atoms
        assert len(reconstructed.atoms) == len(original.atoms)
        for i, (orig_atom, recon_atom) in enumerate(zip(original.atoms, reconstructed.atoms)):
            assert orig_atom.get('name') == recon_atom.get('name')
            assert np.isclose(orig_atom.get('q', 0), recon_atom.get('q', 0))
            assert np.allclose(orig_atom.get('xyz', [0,0,0]), recon_atom.get('xyz', [0,0,0]))
        
        # Verify bonds
        assert len(reconstructed.bonds) == len(original.bonds)
        
        # Verify structure name restoration
        assert reconstructed.get('name') == original.get('name')

    def test_from_frame_empty(self):
        """Test from_frame with empty structure."""
        empty_original = mp.AtomicStructure("empty")
        frame = empty_original.to_frame()
        
        # Use class method to reconstruct
        empty_reconstructed = mp.AtomicStructure.from_frame(frame)
        
        assert len(empty_reconstructed.atoms) == 0
        assert len(empty_reconstructed.bonds) == 0

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
        assert copy_bond.itom in copy1.atoms
        assert copy_bond.jtom in copy1.atoms

    def test_coordinate_independence_after_copy(self):
        """Test that coordinates are independent after copying."""
        template = mp.AtomicStructure("template")
        template.def_atom(name="A", xyz=[0.0, 0.0, 0.0])
        template.def_atom(name="B", xyz=[1.0, 0.0, 0.0])
        
        # Create copy
        copy1 = template()
        
        # Modify template coordinates
        template.atoms[0]["xyz"] = [5.0, 5.0, 5.0]
        
        # Copy coordinates should be unchanged
        assert np.allclose(copy1.atoms[0]["xyz"], [0.0, 0.0, 0.0])
        assert np.allclose(copy1.atoms[1]["xyz"], [1.0, 0.0, 0.0])
        
        # Modify copy coordinates
        copy1.atoms[1]["xyz"] = [10.0, 10.0, 10.0]
        
        # Template coordinates should be unchanged (except the one we modified earlier)
        assert np.allclose(template.atoms[0]["xyz"], [5.0, 5.0, 5.0])
        assert np.allclose(template.atoms[1]["xyz"], [1.0, 0.0, 0.0])

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
        assert np.allclose(template.atoms[0]["xyz"], [0.0, 0.0, 0.0])
        assert np.allclose(template.atoms[1]["xyz"], [1.0, 0.0, 0.0])
        
        # Copy should be moved
        assert np.allclose(copy1.atoms[0]["xyz"], [3.0, 0.0, 0.0])
        assert np.allclose(copy1.atoms[1]["xyz"], [4.0, 0.0, 0.0])

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
            copy.atoms[0]["xyz"] = unique_coord
            
            # Verify other copies are unchanged
            for j, other_copy in enumerate(copies):
                if i != j:
                    assert not np.allclose(other_copy.atoms[0]["xyz"], unique_coord)

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
        
        assert copy_bond1.itom in copy.atoms
        assert copy_bond1.jtom in copy.atoms
        assert copy_bond2.itom in copy.atoms
        assert copy_bond2.jtom in copy.atoms
        
        # Verify atoms are not from template
        assert copy_bond1.itom not in template.atoms
        assert copy_bond1.jtom not in template.atoms
        
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
        print(coords)
        assert len(unique_positions) > 4  # Should have more than 4 unique positions


class TestAtomicStructureDeepCopy:
    """Test AtomicStructure deep copy functionality via __call__ method."""

    def test_deep_copy_atom_ids_different(self):
        """Test that deep copying creates atoms with different object IDs."""
        # Create original structure
        original = mp.AtomicStructure("original")
        
        # Add atoms with different properties
        o1 = original.def_atom(name="O1", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = original.def_atom(name="H1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0])
        h2 = original.def_atom(name="H2", type="H", q=0.4238, xyz=[0.0, 1.0, 0.0])
        
        # Add bonds
        bond1 = original.def_bond(o1, h1, type="OH")
        bond2 = original.def_bond(o1, h2, type="OH")
        
        # Add angles
        angle1 = mp.Angle(h1, o1, h2, type="HOH")
        original.add_angle(angle1)
        
        # Create deep copy
        copied = original()
        
        # Test that the structures are different objects
        assert id(original) != id(copied)
        assert original is not copied
        
        # Test that atoms are different objects but have same data
        assert len(copied.atoms) == len(original.atoms)
        for orig_atom, copy_atom in zip(original.atoms, copied.atoms):
            # Different object IDs
            assert id(orig_atom) != id(copy_atom)
            assert orig_atom is not copy_atom
            
            # Same data content
            assert orig_atom.get("name") == copy_atom.get("name")
            assert orig_atom.get("type") == copy_atom.get("type")
            assert orig_atom.get("q") == copy_atom.get("q")
            if "xyz" in orig_atom:
                assert np.allclose(orig_atom["xyz"], copy_atom["xyz"])
        
        # Test that bonds are different objects but reference correct atoms
        assert len(copied.bonds) == len(original.bonds)
        for orig_bond, copy_bond in zip(original.bonds, copied.bonds):
            # Different object IDs
            assert id(orig_bond) != id(copy_bond)
            assert orig_bond is not copy_bond
            
            # Atom references should point to copied atoms, not original ones
            assert orig_bond.itom is not copy_bond.itom
            assert orig_bond.jtom is not copy_bond.jtom
            
            # But atom names should match (note: Bond constructor may reorder atoms)
            orig_atom_names = {orig_bond.itom.get("name"), orig_bond.jtom.get("name")}
            copy_atom_names = {copy_bond.itom.get("name"), copy_bond.jtom.get("name")}
            assert orig_atom_names == copy_atom_names
        
        # Test that angles are different objects but reference correct atoms
        assert len(copied.angles) == len(original.angles)
        for orig_angle, copy_angle in zip(original.angles, copied.angles):
            # Different object IDs
            assert id(orig_angle) != id(copy_angle)
            assert orig_angle is not copy_angle
            
            # Atom references should point to copied atoms, not original ones
            assert orig_angle.itom is not copy_angle.itom
            assert orig_angle.jtom is not copy_angle.jtom
            assert orig_angle.ktom is not copy_angle.ktom
            
            # But atom names should match (note: Angle constructor may reorder atoms)
            orig_atom_names = {orig_angle.itom.get("name"), orig_angle.jtom.get("name"), orig_angle.ktom.get("name")}
            copy_atom_names = {copy_angle.itom.get("name"), copy_angle.jtom.get("name"), copy_angle.ktom.get("name")}
            assert orig_atom_names == copy_atom_names
            
            # Center atom should be the same
            assert orig_angle.jtom.get("name") == copy_angle.jtom.get("name")

    def test_deep_copy_with_modifications(self):
        """Test deep copying with property modifications."""
        # Create original structure
        original = mp.AtomicStructure("original")
        
        o1 = original.def_atom(name="O1", type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0])
        h1 = original.def_atom(name="H1", type="H", q=0.4238, xyz=[1.0, 0.0, 0.0])
        
        # Create copy with modifications to atom properties
        copied = original(q=0.0)  # Set all atom charges to 0
        
        # Original atoms should be unchanged
        assert original.atoms[0].get("q") == -0.8476
        assert original.atoms[1].get("q") == 0.4238
        
        # Copied atoms should have modified charges
        assert copied.atoms[0].get("q") == 0.0
        assert copied.atoms[1].get("q") == 0.0
        
        # Other properties should be unchanged
        assert copied.atoms[0].get("name") == "O1"
        assert copied.atoms[1].get("name") == "H1"

    def test_deep_copy_with_dihedrals(self):
        """Test deep copying structures with dihedrals."""
        # Create structure with 4 atoms
        original = mp.AtomicStructure("with_dihedrals")
        
        c1 = original.def_atom(name="C1", type="C", xyz=[0.0, 0.0, 0.0])
        c2 = original.def_atom(name="C2", type="C", xyz=[1.0, 0.0, 0.0])
        c3 = original.def_atom(name="C3", type="C", xyz=[2.0, 0.0, 0.0])
        c4 = original.def_atom(name="C4", type="C", xyz=[3.0, 0.0, 0.0])
        
        # Add dihedral
        dihedral1 = mp.Dihedral(c1, c2, c3, c4, type="CCCC")
        original.add_dihedral(dihedral1)
        
        # Create deep copy
        copied = original()
        
        # Test dihedral copying
        assert len(copied.dihedrals) == len(original.dihedrals)
        orig_dihedral = original.dihedrals[0]
        copy_dihedral = copied.dihedrals[0]
        
        # Different object IDs
        assert id(orig_dihedral) != id(copy_dihedral)
        
        # Atom references should point to copied atoms
        assert orig_dihedral.itom is not copy_dihedral.itom
        assert orig_dihedral.jtom is not copy_dihedral.jtom
        assert orig_dihedral.ktom is not copy_dihedral.ktom
        assert orig_dihedral.ltom is not copy_dihedral.ltom
        
        # But atom names should match (note: Dihedral constructor may reorder atoms)
        orig_atom_names = {orig_dihedral.itom.get("name"), orig_dihedral.jtom.get("name"), 
                          orig_dihedral.ktom.get("name"), orig_dihedral.ltom.get("name")}
        copy_atom_names = {copy_dihedral.itom.get("name"), copy_dihedral.jtom.get("name"),
                          copy_dihedral.ktom.get("name"), copy_dihedral.ltom.get("name")}
        assert orig_atom_names == copy_atom_names

    def test_deep_copy_with_impropers(self):
        """Test deep copying structures with impropers."""
        # Create structure with 4 atoms for improper
        original = mp.AtomicStructure("with_impropers")
        
        c1 = original.def_atom(name="C1", type="C", xyz=[0.0, 0.0, 0.0])
        h1 = original.def_atom(name="H1", type="H", xyz=[1.0, 0.0, 0.0])
        h2 = original.def_atom(name="H2", type="H", xyz=[0.0, 1.0, 0.0])
        h3 = original.def_atom(name="H3", type="H", xyz=[0.0, 0.0, 1.0])
        
        # Initialize impropers collection and add improper
        if 'impropers' not in original:
            original['impropers'] = mp.Entities()
        
        improper1 = mp.Improper(c1, h1, h2, h3, type="CHHH")
        original['impropers'].add(improper1)
        
        # Create deep copy
        copied = original()
        
        # Test improper copying
        assert len(copied['impropers']) == len(original['impropers'])
        orig_improper = original['impropers'][0]
        copy_improper = copied['impropers'][0]
        
        # Different object IDs
        assert id(orig_improper) != id(copy_improper)
        
        # Atom references should point to copied atoms
        assert orig_improper.itom is not copy_improper.itom
        assert orig_improper.jtom is not copy_improper.jtom
        assert orig_improper.ktom is not copy_improper.ktom
        assert orig_improper.ltom is not copy_improper.ltom
        
        # But atom names should match (note: Improper constructor may reorder atoms)
        orig_atom_names = {orig_improper.itom.get("name"), orig_improper.jtom.get("name"),
                          orig_improper.ktom.get("name"), orig_improper.ltom.get("name")}
        copy_atom_names = {copy_improper.itom.get("name"), copy_improper.jtom.get("name"),
                          copy_improper.ktom.get("name"), copy_improper.ltom.get("name")}
        assert orig_atom_names == copy_atom_names

    def test_deep_copy_independence(self):
        """Test that modifications to copied structure don't affect original."""
        # Create original structure
        original = mp.AtomicStructure("original")
        
        o1 = original.def_atom(name="O1", type="O", q=-0.8, xyz=[0.0, 0.0, 0.0])
        h1 = original.def_atom(name="H1", type="H", q=0.4, xyz=[1.0, 0.0, 0.0])
        bond1 = original.def_bond(o1, h1)
        
        # Create deep copy
        copied = original()
        
        # Modify copied structure
        copied.atoms[0]["q"] = -1.0
        copied.atoms[0]["xyz"] = [10.0, 10.0, 10.0]
        copied.def_atom(name="N1", type="N", q=-0.5)
        
        # Original should be unchanged
        assert original.atoms[0].get("q") == -0.8
        assert np.allclose(original.atoms[0]["xyz"], [0.0, 0.0, 0.0])
        assert len(original.atoms) == 2
        
        # Copied should have modifications
        assert copied.atoms[0].get("q") == -1.0
        assert np.allclose(copied.atoms[0]["xyz"], [10.0, 10.0, 10.0])
        assert len(copied.atoms) == 3

    def test_deep_copy_empty_structure(self):
        """Test deep copying of empty structure."""
        original = mp.AtomicStructure("empty")
        copied = original()
        
        assert id(original) != id(copied)
        assert len(copied.atoms) == 0
        assert len(copied.bonds) == 0
        assert len(copied.angles) == 0
        assert len(copied.dihedrals) == 0

    def test_deep_copy_preserves_custom_properties(self):
        """Test that custom properties are properly deep copied."""
        original = mp.AtomicStructure("test")
        
        # Add custom properties
        original["custom_list"] = [1, 2, 3]
        original["custom_dict"] = {"a": 1, "b": 2}
        original["custom_value"] = 42
        
        copied = original()
        
        # Properties should be copied but be independent objects
        assert copied["custom_list"] == [1, 2, 3]
        assert copied["custom_dict"] == {"a": 1, "b": 2}
        assert copied["custom_value"] == 42
        
        # Modify copied properties
        copied["custom_list"].append(4)
        copied["custom_dict"]["c"] = 3
        copied["custom_value"] = 100
        
        # Original should be unchanged
        assert original["custom_list"] == [1, 2, 3]
        assert original["custom_dict"] == {"a": 1, "b": 2}
        assert original["custom_value"] == 42
