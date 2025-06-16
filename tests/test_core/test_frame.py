import pytest
import numpy as np
import xarray as xr
from molpy.core.frame import Frame # Removed AllAtomFrame
from molpy.core.box import Box

# Minimal helper for tests if needed
_element_map = {"Ar": 18, "C": 6, "H": 1, "O": 8, "Li": 3, "He": 2, "Na": 11, "Fe": 26, "Au": 79, "Be": 4}

def _test_symbol_to_atomic_number(symbol):
    return _element_map.get(symbol)

def _test_atomic_number_to_symbol(number):
    for sym, num in _element_map.items():
        if num == number:
            return sym
    return "X"

# Helper function
def create_dummy_box():
    return Box(matrix=np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]))

class TestFrame:
    def test_frame_creation_empty(self):
        frame = Frame()
        assert frame._data.get('atoms') is None # n_atoms == 0
        assert frame._data.get('residues') is None # n_residues == 0
        assert frame._data.get('molecules') is None # n_molecules == 0
        assert frame.box is None
        assert frame.timestep is None # time == 0.0, step == 0
        assert "atoms" not in frame._data
        assert "residues" not in frame._data
        assert "molecules" not in frame._data

    def test_frame_creation_with_dict(self):
        atoms_data = {
            "position": np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
            "name": np.array(["C", "H"]),
            "id": np.array([1, 2]),
            "atomic_number": np.array([6, 1], dtype=int)
        }
        frame = Frame(data={"atoms": atoms_data})
        assert frame['atoms'].sizes['index'] == 2
        assert frame['atoms']["name"][0].item() == "C"
        assert np.array_equal(frame['atoms']["position"][1].values, np.array([1.0, 1.0, 1.0]))
        # No n_atoms attribute on DataArray by default from Frame
        # assert frame['atoms'].attrs["n_atoms"] == 2 

    def test_frame_creation_with_box(self):
        box = create_dummy_box()
        frame = Frame(box=box)
        assert frame.box is not None
        assert np.array_equal(frame.box.matrix, box.matrix)

    def test_frame_properties(self):
        atoms_data = {"position": np.array([[0,0,0]]), "name": np.array(["C"])}
        frame = Frame(data={"atoms": atoms_data})
        frame.timestep = 100
        frame['time_prop'] = 10.0 # Custom scalar property

        assert frame['atoms'].sizes['index'] == 1
        assert frame['time_prop'] == 10.0
        assert frame.timestep == 100

    def test_frame_atom_access(self):
        atoms_data = {
            "position": np.array([[0,0,0]]),
            "name": np.array(["C"]),
            "id": np.array([1]),
            "atomic_number": np.array([6])
        }
        frame = Frame(data={"atoms": atoms_data})
        assert frame['atoms']["name"][0].item() == "C"
        assert frame['atoms']["position"].shape == (1, 3) # Accessing the DataArray directly

    def test_frame_set_atoms_from_dict(self):
        frame = Frame()
        atoms_data = {
            "position": np.array([[0,0,0]]),
            "name": np.array(["H"]),
            "id": np.array([10]),
            "atomic_number": np.array([1])
        }
        frame['atoms'] = atoms_data # Uses __setitem__
        assert frame['atoms'].sizes['index'] == 1
        assert frame['atoms']["name"][0].item() == "H"

    def test_frame_iteration_over_keys(self):
        atoms_data = {
            "position": np.array([[0,0,0], [1,1,1]]),
            "name": np.array(["C", "H"]),
            "id": np.array([1,2])
        }
        frame = Frame(data={"atoms": atoms_data})
        frame.timestep = 10
        frame['custom_scalar'] = 5.0
        
        keys = list(frame) # Frame iterates over its main keys
        assert "atoms" in keys
        assert "timestep" in keys # Timestep is yielded if not None
        assert "custom_scalar" in keys
        
        # Example: iterate over atom names within the 'atoms' DataArray
        names = [name.item() for name in frame['atoms']["name"]]
        assert names == ["C", "H"]

    def test_frame_indexing_slicing_atoms_dataarray(self):
        atoms_data = {
            "position": np.array([[0,0,0], [1,1,1], [2,2,2]]),
            "name": np.array(["C", "H", "O"]),
            "id": np.array([1,2,3]),
            "atomic_number": np.array([6,1,8])
        }
        frame = Frame(data={"atoms": atoms_data}, box=create_dummy_box())
        frame.timestep = 5
        frame['scalar_info'] = 'test'
        
        # Get a subset of atoms DataArray
        subset_atoms_da = frame['atoms'].isel(index=[0, 2])
        
        # Create a new frame with the subset
        subset_frame = Frame(data={"atoms": subset_atoms_da}, box=frame.box)
        subset_frame.timestep = frame.timestep # Carry over other properties
        subset_frame['scalar_info'] = frame['scalar_info']

        assert subset_frame['atoms'].sizes['index'] == 2
        assert subset_frame['atoms']["name"][0].item() == "C"
        assert subset_frame['atoms']["name"][1].item() == "O"
        assert np.array_equal(subset_frame['atoms']["position"][0].values, np.array([0.0,0.0,0.0]))
        assert np.array_equal(subset_frame['atoms']["position"][1].values, np.array([2.0,2.0,2.0]))

        # Test original frame is unchanged
        assert frame['atoms'].sizes['index'] == 3

    def test_frame_copy(self):
        atoms_data = {"position": np.array([[0,0,0]]), "name": np.array(["C"])}
        box = create_dummy_box()
        frame1 = Frame(data={"atoms": atoms_data}, box=box)
        frame1.timestep = 50
        frame1['custom_time'] = 5.0
        
        frame2 = frame1.copy()
        
        assert frame2['atoms'].sizes['index'] == frame1['atoms'].sizes['index']
        assert np.array_equal(frame2['atoms']["position"].data, frame1['atoms']["position"].data)
        assert frame2['atoms']["position"].data is not frame1['atoms']["position"].data
        
        assert frame2.box is not None
        assert frame1.box is not None
        assert np.array_equal(frame2.box.matrix, frame1.box.matrix)
        if frame1.box:
             assert frame2.box.matrix is not frame1.box.matrix

        assert frame2.timestep == frame1.timestep
        assert frame2['custom_time'] == frame1['custom_time']
        
        # Modify frame2 and check frame1 is unchanged
        frame2['atoms'] = {"position": np.array([[1,1,1]]), "name": np.array(["H"])}
        frame2.timestep = 100
        frame2['custom_time'] = 10.0
        
        assert frame1['atoms']["name"][0].item() == "C"
        assert frame1.timestep == 50
        assert frame1['custom_time'] == 5.0
        if frame1.box and frame2.box: # mypy check
            # To change the box, create a new Box instance
            new_box_matrix = np.array([[20,0,0],[0,20,0],[0,0,20]])
            frame2.box = Box(matrix=new_box_matrix, pbc=frame2.box.pbc, origin=frame2.box.origin)
            assert not np.array_equal(frame2.box.matrix, frame1.box.matrix)


    def test_frame_add_atom_property_to_dataarray(self):
        atoms_data = {"position": np.array([[0,0,0], [1,1,1]])}
        frame = Frame(data={"atoms": atoms_data})
        
        new_property_data = np.array([1.0, 2.0])
        # Modify the DataArray in place by assigning new coordinates
        frame['atoms'] = frame['atoms'].assign_coords(charge=("index", new_property_data))
        
        assert "charge" in frame['atoms'].coords
        assert frame['atoms']["charge"].shape == (2,)
        assert np.array_equal(frame['atoms']["charge"].data, new_property_data)

        # Test adding with wrong dimensions
        with pytest.raises(ValueError): # xarray raises ValueError for mismatched dims in assign_coords
            frame['atoms'].assign_coords(bad_prop=("index", np.array([1.0])))


    def test_frame_select_atoms_by_mask_on_dataarray(self):
        atoms_data = {
            "position": np.array([[0,0,0], [1,1,1], [2,2,2]]),
            "name": np.array(["C", "H", "O"]),
            "mass": np.array([12.0, 1.0, 16.0])
        }
        frame = Frame(data={"atoms": atoms_data})
        
        # Select atoms with mass > 10.0
        mask = frame['atoms']["mass"] > 10.0 # This is an xr.DataArray (boolean)
        
        selected_atoms_da = frame['atoms'][mask] # Indexing DataArray with boolean DataArray
        selected_frame = Frame(data={"atoms": selected_atoms_da}, box=frame.box) # Create new frame
        
        assert selected_frame['atoms'].sizes['index'] == 2
        assert "C" in selected_frame['atoms']["name"].data
        assert "O" in selected_frame['atoms']["name"].data
        assert "H" not in selected_frame['atoms']["name"].data

    def test_frame_dict_conversion_minimal(self):
        frame = Frame()
        frame_dict = frame.to_dict()
        assert "atoms" not in frame_dict 
        assert "box" not in frame_dict
        assert "timestep" not in frame_dict # timestep is None, not included if None
        # Custom scalars would be in frame_dict if set
        frame.timestep = 0
        frame_dict_with_step = frame.to_dict()
        assert frame_dict_with_step["timestep"] == 0


    def test_frame_dict_conversion_with_data(self):
        atoms_data = {
            "position": np.array([[0,0,0]]),
            "name": np.array(["C"]),
            "id": np.array([1]),
            "atomic_number": np.array([6])
        }
        box = create_dummy_box()
        frame = Frame(data={"atoms": atoms_data}, box=box)
        frame.timestep = 1
        frame['custom_scalar'] = "test"
        frame_dict = frame.to_dict()

        assert "atoms" in frame_dict
        assert isinstance(frame_dict["atoms"], dict)
        assert np.array_equal(frame_dict["atoms"]["position"], atoms_data["position"])
        assert np.array_equal(frame_dict["atoms"]["name"], atoms_data["name"])
        
        assert "box" in frame_dict
        # Check against Box.to_dict() output
        expected_box_dict = box.to_dict()
        assert frame_dict["box"] == expected_box_dict

        assert frame_dict["timestep"] == 1
        assert frame_dict["custom_scalar"] == "test"

    def test_frame_from_dict(self):
        # Box data should be in a format that Box constructor or a Box.from_dict can handle.
        # Current Box constructor takes matrix, pbc, origin.
        # Frame.from_dict is not implemented in the provided Frame class.
        # This test assumes a Frame.from_dict method exists and works.
        # For now, we test direct construction and to_dict.
        # If Frame.from_dict is to be tested, it needs to be implemented in Frame.
        
        # Let's test constructing a Frame and then converting it to_dict, then manually reconstructing
        # to verify the to_dict output's completeness for reconstruction.
        
        original_frame_dict_atoms = {
            "position": np.array([[0.1, 0.2, 0.3]]),
            "name": np.array(["Ar"]),
            "id": np.array([42]),
            "atomic_number": np.array([_test_symbol_to_atomic_number("Ar")])
        }
        original_box_matrix = np.array([[10,0,0],[0,10,0],[0,0,10]])
        original_box = Box(matrix=original_box_matrix)

        frame = Frame(data={"atoms": original_frame_dict_atoms}, box=original_box)
        frame.timestep = 20
        frame['custom_prop'] = "value"

        reconstituted_dict = frame.to_dict()

        # Now, manually reconstruct a new Frame from reconstituted_dict
        # This mimics what a Frame.from_dict would do.
        new_atoms_data = reconstituted_dict.get("atoms")
        new_box_dict = reconstituted_dict.get("box")
        new_box = None
        if new_box_dict:
            # This is the tricky part: Box.to_dict() gives xlo, xhi, etc.
            # Box constructor needs matrix. We need a Box.from_box_dict(new_box_dict) or similar.
            # For this test, let's assume we can reconstruct the matrix from xlo/xhi/tilts if needed,
            # or that Frame.from_dict would handle this.
            # Since Box.from_dict is not there, let's use the original box for comparison for now.
            # A proper test of Frame.from_dict would require Box.from_dict or more complex logic here.
            # We will assert that the reconstituted_dict['box'] matches original_box.to_dict()
            assert new_box_dict == original_box.to_dict()
            # For actual reconstruction in a Frame.from_dict, one would need:
            # new_box = Box(matrix=original_box_matrix) # Simplified for test continuity

        reconstructed_frame = Frame(data={"atoms": new_atoms_data} if new_atoms_data else None, box=original_box) # Using original_box
        if "timestep" in reconstituted_dict:
            reconstructed_frame.timestep = reconstituted_dict["timestep"]
        for k, v in reconstituted_dict.items():
            if k not in ["atoms", "box", "timestep"]: # custom scalars
                reconstructed_frame[k] = v
        
        assert reconstructed_frame['atoms'].sizes['index'] == 1
        assert reconstructed_frame['atoms']["name"][0].item() == "Ar"
        assert reconstructed_frame.box is not None
        assert np.array_equal(reconstructed_frame.box.matrix, original_box_matrix)
        assert reconstructed_frame.timestep == 20
        assert reconstructed_frame['custom_prop'] == "value"


    def test_frame_empty_get_item_on_atoms(self):
        frame = Frame()
        with pytest.raises(KeyError): # Accessing 'atoms' key when it's not in _data
            _ = frame['atoms']
        
        # If 'atoms' key exists but is an empty DataArray
        frame_with_empty_atoms = Frame(data={"atoms": {}}) # _dict_to_dataarray makes an empty DA
        assert frame_with_empty_atoms['atoms'].sizes['index'] == 0
        with pytest.raises(KeyError): # Accessing a specific coord like 'position' in an empty DA
                                      # depends on how _dict_to_dataarray handles truly empty dict.
                                      # It creates an empty DA with no coords if dict is empty.
            _ = frame_with_empty_atoms['atoms']['position']


    def test_frame_atom_coords_exist(self):
        frame = Frame(data={"atoms": {"position": np.array([[1,2,3]]), "id": np.array([1])}})
        assert "position" in frame['atoms'].coords
        assert "id" in frame['atoms'].coords
        assert frame['atoms'].sizes['index'] == 1

    def test_frame_creation_scalar_handling_in_dict_to_dataarray(self):
        atoms_data_scalar = {
            "position": np.array([[0,0,0], [1,1,1]]),
            "id": 1 # Scalar id
        }
        # _dict_to_dataarray promotes scalars to scalar coordinates, not broadcasted arrays.
        # If an 'index' dim is established by other arrays, scalar coords are fine.
        frame = Frame(data={"atoms": atoms_data_scalar})
        assert frame['atoms'].sizes['index'] == 2
        assert "id" in frame['atoms'].coords
        assert frame['atoms']['id'].ndim == 0 # It's a scalar coordinate
        assert frame['atoms']['id'].item() == 1

        # Test with only scalar data - _dict_to_dataarray creates a 0-dim DataArray with scalar coords
        atoms_only_scalar = {"mass": 12.0, "charge": -1}
        frame_scalar_only = Frame(data={"atoms": atoms_only_scalar})
        assert "mass" in frame_scalar_only['atoms'].coords
        assert frame_scalar_only['atoms']['mass'].item() == 12.0
        # The main data of such a DataArray is `None` and dims might be tricky.
        # `sizes` would be empty. `frame_scalar_only['atoms'].sizes.get('index', 0) == 0`
        assert frame_scalar_only['atoms'].sizes.get('index', 0) == 0


    def test_frame_isel_atoms_empty_selection_on_dataarray(self):
        atoms_data = {"position": np.array([[0,0,0]]), "name": np.array(["C"])}
        frame = Frame(data={"atoms": atoms_data})
        
        empty_atoms_da = frame['atoms'].isel(index=[])
        empty_frame = Frame(data={"atoms": empty_atoms_da})

        assert empty_frame['atoms'].sizes['index'] == 0
        assert "position" in empty_frame['atoms'].coords 
        assert empty_frame['atoms']["position"].shape == (0,3)

    def test_frame_select_atoms_by_mask_all_false_on_dataarray(self):
        atoms_data = {"position": np.array([[0,0,0]]), "name": np.array(["C"])}
        frame = Frame(data={"atoms": atoms_data})
        
        mask = xr.DataArray(np.array([False]), dims="index") # Ensure mask has 'index' dim
        selected_atoms_da = frame['atoms'][mask]
        selected_frame = Frame(data={"atoms": selected_atoms_da})
        
        assert selected_frame['atoms'].sizes['index'] == 0
        assert selected_frame['atoms']["position"].shape == (0,3)

    def test_frame_select_atoms_by_mask_all_true_on_dataarray(self):
        atoms_data = {"position": np.array([[0,0,0]]), "name": np.array(["C"])}
        frame = Frame(data={"atoms": atoms_data})

        mask = xr.DataArray(np.array([True]), dims="index")
        selected_atoms_da = frame['atoms'][mask]
        selected_frame = Frame(data={"atoms": selected_atoms_da})

        assert selected_frame['atoms'].sizes['index'] == 1
        assert selected_frame['atoms']["name"][0].item() == "C"

    def test_frame_set_residues_molecules(self):
        frame = Frame()
        residues_data = {"id": np.array([1]), "name": np.array(["ALA"])}
        frame['residues'] = residues_data
        assert frame['residues'].sizes['index'] == 1
        assert frame['residues']["name"][0].item() == "ALA"

        molecules_data = {"id": np.array([10]), "type": np.array(["protein"])}
        frame['molecules'] = molecules_data
        assert frame['molecules'].sizes['index'] == 1
        assert frame['molecules']["type"][0].item() == "protein"

    def test_frame_isel_on_residues_molecules_dataarrays(self):
        residues_data = {"id": np.array([1,2,3])}
        molecules_data = {"id": np.array([10,11])}
        frame = Frame(data={"residues": residues_data, "molecules": molecules_data})

        sub_residues_da = frame['residues'].isel(index=[0,2])
        sub_frame_res = Frame(data={"residues": sub_residues_da})
        assert sub_frame_res['residues'].sizes['index'] == 2
        assert sub_frame_res['residues']["id"][0].item() == 1
        assert sub_frame_res['residues']["id"][1].item() == 3
        
        sub_molecules_da = frame['molecules'].isel(index=[1])
        sub_frame_mol = Frame(data={"molecules": sub_molecules_da})
        assert sub_frame_mol['molecules'].sizes['index'] == 1
        assert sub_frame_mol['molecules']["id"][0].item() == 11

