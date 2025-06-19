import pytest
import numpy as np
import tempfile
from pathlib import Path
import molpy as mp


class TestWaterboxIntegration:
    """Integration tests for the complete waterbox workflow."""

    def test_waterbox_generation_and_export(self):
        """Test the complete waterbox generation and LAMMPS export workflow."""
        
        # Create SPCE water molecule template
        template = mp.AtomicStructure("spce_template", molid=1)
        o = template.def_atom(
            name="o", molid=1, type="O", q=-0.8476, xyz=[0.0, 0.0, 0.0]
        )
        h1 = template.def_atom(
            name="h1", molid=1, type="H", q=0.4238, xyz=[0.8164904, 0.5773590, 0.0]
        )
        h2 = template.def_atom(
            name="h2", molid=1, type="H", q=0.4238, xyz=[-0.8164904, 0.5773590, 0.0]
        )
        template.def_bond(o, h1)
        template.def_bond(o, h2)
        angle = mp.Angle(h1, o, h2, theta0=109.47, k=1000.0)
        template.add_angle(angle)
        
        # Create forcefield
        ff = mp.ForceField(name="spce", unit="real")
        atomstyle = ff.def_atomstyle("full")
        o_type = atomstyle.def_type("O", mass=15.999)
        h_type = atomstyle.def_type("H", mass=1.008)
        
        bondstyle = ff.def_bondstyle("harmonic")
        bondstyle.def_type(o_type, h_type, k=1000.0, r0=1.0)
        
        anglestyle = ff.def_anglestyle("harmonic")
        anglestyle.def_type(h_type, o_type, h_type, k=1000.0, theta0=109.47)
        
        pairstyle = ff.def_pairstyle("lj/charmm/coul/long", inner=9.0, outer=10.0, cutoff=10.0)
        pairstyle.def_type(o_type, o_type, epsilon=0.1554, sigma=3.1656)
        pairstyle.def_type(h_type, h_type, epsilon=0.0, sigma=0.0)
        
        # Create system
        system = mp.System()
        system.set_forcefield(ff)
        box_size = 10.0
        system.def_box(np.diag([box_size, box_size, box_size]))
        
        # Create 2x2x2 = 8 water molecules
        n_molecules = 0
        spacing = 3.0
        
        spce_template = mp.SpatialWrapper(template)
        
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    molid = n_molecules + 1
                    
                    # Create new water molecule
                    water_mol = spce_template(molid=molid)
                    
                    # Update atom molids
                    for atom in water_mol.atoms:
                        atom['molid'] = molid
                    
                    # Move to position
                    position = [i * spacing, j * spacing, k * spacing]
                    spatial_water = mp.SpatialWrapper(water_mol)
                    spatial_water.move(position)
                    
                    # Add to system
                    system.add_struct(water_mol)
                    n_molecules += 1
        
        # Verify system statistics
        assert n_molecules == 8
        total_atoms = sum(len(struct.atoms) for struct in system._struct)
        total_bonds = sum(len(struct.bonds) for struct in system._struct)
        total_angles = sum(len(struct.angles) for struct in system._struct)
        
        assert total_atoms == 24  # 3 atoms * 8 molecules
        assert total_bonds == 16  # 2 bonds * 8 molecules
        assert total_angles == 8  # 1 angle * 8 molecules
        
        # Convert to frame
        frame = system.to_frame()
        
        # Verify frame structure
        assert "atoms" in frame
        assert "bonds" in frame
        assert "angles" in frame
        
        # Check atom count in frame
        atoms_data = frame["atoms"]
        n_atoms_frame = 0
        if hasattr(atoms_data, 'sizes'):
            for dim_name in ['index', 'dim_id_0', 'dim_q_0', 'dim_xyz_0']:
                if dim_name in atoms_data.sizes:
                    n_atoms_frame = atoms_data.sizes[dim_name]
                    break
        assert n_atoms_frame == 24
        
        # Verify coordinates are distributed correctly
        if "xyz" in atoms_data:
            coords = atoms_data["xyz"].values
        else:
            coords = np.column_stack([
                atoms_data["x"].values,
                atoms_data["y"].values,
                atoms_data["z"].values
            ])
        
        # Check coordinate distribution
        assert coords.shape == (24, 3)
        # Note: Some hydrogen coordinates may be negative due to molecular geometry
        assert np.all(coords >= -1.0)  # Allow some negative coordinates
        assert np.all(coords <= box_size + 1.0)  # All coordinates should be roughly within box
        
        # Verify charges are present and correct
        assert "q" in atoms_data
        charges = atoms_data["q"].values
        assert len(charges) == 24
        
        # Should have 8 oxygen atoms with charge -0.8476
        o_charges = charges[charges < 0]
        assert len(o_charges) == 8
        assert np.allclose(o_charges, -0.8476)
        
        # Should have 16 hydrogen atoms with charge 0.4238
        h_charges = charges[charges > 0]
        assert len(h_charges) == 16
        assert np.allclose(h_charges, 0.4238)
        
        # Export to LAMMPS data file
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "waterbox.data"
            ff_file = Path(tmpdir) / "waterbox.ff"
            
            # Write LAMMPS files
            mp.io.write_lammps_data(data_file, frame)
            mp.io.write_lammps_forcefield(ff_file, ff)
            
            # Verify files were created
            assert data_file.exists()
            assert ff_file.exists()
            assert data_file.stat().st_size > 0
            assert ff_file.stat().st_size > 0
            
            # Read and verify LAMMPS data file content
            with open(data_file, 'r') as f:
                content = f.read()
            
            # Check header
            assert "24 atoms" in content
            assert "16 bonds" in content
            assert "8 angles" in content
            assert "2 atom types" in content
            assert "1 bond types" in content
            assert "1 angle types" in content
            
            # Check box dimensions
            assert f"{box_size:.6f}" in content
            
            # Check atom type labels
            assert "Atom Type Labels" in content
            assert "H" in content
            assert "O" in content
            
            # Check charges are written correctly
            assert "-0.847600" in content
            assert "0.423800" in content
            
            # Check sections exist
            assert "Atoms" in content
            assert "Bonds" in content
            assert "Angles" in content
            
            # Verify bond indices are 1-based and within range
            lines = content.split('\n')
            bonds_section = False
            bond_count = 0
            for line in lines:
                if line.strip() == "Bonds":
                    bonds_section = True
                    continue
                elif line.strip() == "Angles":
                    bonds_section = False
                    continue
                
                if bonds_section and line.strip() and not line.startswith('#'):
                    parts = line.split()
                    if len(parts) >= 4:
                        bond_id = int(parts[0])
                        atom_i = int(parts[2])
                        atom_j = int(parts[3])
                        
                        # LAMMPS uses 1-based indexing
                        assert 1 <= atom_i <= 24
                        assert 1 <= atom_j <= 24
                        assert atom_i != atom_j
                        bond_count += 1
            
            assert bond_count == 16

    def test_trajectory_export_import(self):
        """Test exporting and importing LAMMPS trajectory."""
        # Create simple system
        struct = mp.AtomicStructure("test")
        struct.def_atom(name="A", type=1, xyz=[0.0, 0.0, 0.0])  # Use numeric type
        struct.def_atom(name="B", type=2, xyz=[1.0, 0.0, 0.0])  # Use numeric type
        
        system = mp.System()
        system.add_struct(struct)
        system.def_box(np.diag([5.0, 5.0, 5.0]))
        
        # Create trajectory frames
        frames = []
        for i in range(3):
            frame = system.to_frame()
            frame["timestep"] = i * 100
            
            # Modify coordinates for each frame
            if "xyz" in frame["atoms"]:
                coords = frame["atoms"]["xyz"].values.copy()
                coords[:, 0] += i * 0.1  # Move atoms slightly each frame
                frame["atoms"]["xyz"] = (frame["atoms"]["xyz"].dims, coords)
            
            frames.append(frame)
        
        # Export trajectory
        with tempfile.TemporaryDirectory() as tmpdir:
            traj_file = Path(tmpdir) / "test.dump"
            
            writer = mp.io.trajectory.LammpsTrajectoryWriter(traj_file)
            for frame in frames:
                # Pass timestep explicitly
                timestep = frame.get("timestep", 0)
                writer.write_frame(frame, timestep)
            writer.close()
            
            # Verify file was created
            assert traj_file.exists()
            assert traj_file.stat().st_size > 0
            
            # Read back trajectory
            reader = mp.io.trajectory.LammpsTrajectoryReader(traj_file)
            
            read_frames = []
            for i, frame in enumerate(reader):
                read_frames.append(frame)
                if i >= 2:  # Read 3 frames
                    break
            
            assert len(read_frames) == 3
            
            # Verify timesteps
            for i, frame in enumerate(read_frames):
                assert frame["timestep"] == i * 100
                assert "atoms" in frame
                assert frame.box is not None

    def test_error_handling(self):
        """Test error handling in the waterbox workflow."""
        
        # Test structure with minimal data
        simple_struct = mp.AtomicStructure("simple")
        simple_struct.def_atom(name="A", type="A", xyz=[0.0, 0.0, 0.0])
        frame = simple_struct.to_frame()
        
        # Should handle simple structure gracefully
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "simple.data"
            
            # Should not crash on simple structure
            writer = mp.io.data.LammpsDataWriter(data_file)
            writer.write(frame)
            
            assert data_file.exists()
            with open(data_file, 'r') as f:
                content = f.read()
            assert "1 atoms" in content

    def test_large_system_performance(self):
        """Test performance with larger systems (but not too large for CI)."""
        # Create template
        template = mp.AtomicStructure("template")
        template.def_atom(name="A", type="A", xyz=[0.0, 0.0, 0.0])
        template.def_atom(name="B", type="B", xyz=[1.0, 0.0, 0.0])
        template.def_bond(template.atoms[0], template.atoms[1])
        
        # Create system with 27 molecules (3x3x3)
        system = mp.System()
        system.def_box(np.diag([10.0, 10.0, 10.0]))
        
        template_wrapper = mp.SpatialWrapper(template)
        
        n_molecules = 0
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    mol_wrapper = template_wrapper(molid=n_molecules+1)
                    
                    # Move molecule
                    mol_wrapper.move([i*2.0, j*2.0, k*2.0])
                    
                    # Extract the underlying structure for add_struct
                    mol = mol_wrapper._wrapped
                    system.add_struct(mol)
                    n_molecules += 1
        
        assert n_molecules == 27
        
        # Convert to frame (should be fast)
        frame = system.to_frame()
        
        # Verify system
        atoms_data = frame["atoms"]
        n_atoms = 0
        if hasattr(atoms_data, 'sizes'):
            for dim_name in ['index', 'dim_id_0', 'dim_xyz_0']:
                if dim_name in atoms_data.sizes:
                    n_atoms = atoms_data.sizes[dim_name]
                    break
        
        assert n_atoms == 54  # 2 atoms * 27 molecules
        
        # Export (should also be reasonably fast)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_file = Path(tmpdir) / "large.data"
            
            writer = mp.io.data.LammpsDataWriter(data_file)
            writer.write(frame)
            
            assert data_file.exists()
            
            # Verify content
            with open(data_file, 'r') as f:
                content = f.read()
            assert "54 atoms" in content
            assert "27 bonds" in content


class TestForceFieldIntegration:
    """Test forcefield integration with the new Frame API."""
    
    def test_forcefield_export(self):
        """Test that forcefield information is correctly exported."""
        # Create forcefield
        ff = mp.ForceField(name="test_ff", unit="real")
        
        atomstyle = ff.def_atomstyle("full")
        a_type = atomstyle.def_type("A", mass=1.0)
        b_type = atomstyle.def_type("B", mass=2.0)
        
        bondstyle = ff.def_bondstyle("harmonic")
        bondstyle.def_type(a_type, b_type, k=100.0, r0=1.5)
        
        # Create system with forcefield
        system = mp.System()
        system.set_forcefield(ff)
        
        struct = mp.AtomicStructure("test")
        struct.def_atom(name="A", type="A")
        struct.def_atom(name="B", type="B")
        struct.def_bond(struct.atoms[0], struct.atoms[1])
        
        system.add_struct(struct)
        system.def_box(np.diag([5.0, 5.0, 5.0]))
        
        frame = system.to_frame()
        
        # Export forcefield
        with tempfile.TemporaryDirectory() as tmpdir:
            ff_file = Path(tmpdir) / "test.ff"
            
            mp.io.write_lammps_forcefield(ff_file, ff)
            
            assert ff_file.exists()
            
            with open(ff_file, 'r') as f:
                content = f.read()
            
            # Should contain forcefield information
            assert "atom_style full" in content or "atomstyle full" in content or "full" in content
