import pytest
import numpy as np
import xarray as xr
from molpy.core.frame import Frame
from molpy.core.box import Box

# Helper function
def create_dummy_box():
    return Box(matrix=np.array([[10.0, 0, 0], [0, 10.0, 0], [0, 0, 10.0]]))

class TestFrame:
    def test_frame_creation_empty(self):
        frame = Frame()
        assert frame._data.get('atoms') is None
        assert frame._data.get('residues') is None
        assert frame._data.get('molecules') is None
        assert frame.box is None
        assert frame.timestep is None
        assert "atoms" not in frame._data
        assert "residues" not in frame._data
        assert "molecules" not in frame._data

    def test_frame_creation_with_dict(self):
        atoms_data = {
            "position": np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
            "name": np.array(["C", "H"], dtype="U1"),
            "id": np.array([1, 2], dtype=int),
            "atomic_number": np.array([6, 1], dtype=int)
        }
        frame = Frame(data={"atoms": atoms_data})
        
        # 检查数据变量是否正确存储
        assert "position" in frame['atoms'].data_vars
        assert "name" in frame['atoms'].data_vars
        assert "id" in frame['atoms'].data_vars
        assert "atomic_number" in frame['atoms'].data_vars
        
        # 检查数据值
        assert frame['atoms']["name"][0].item() == "C"
        assert np.array_equal(frame['atoms']["position"][1].values, np.array([1.0, 1.0, 1.0]))

    def test_frame_creation_with_box(self):
        box = create_dummy_box()
        frame = Frame(box=box)
        assert frame.box is not None
        assert np.array_equal(frame.box.matrix, box.matrix)

    def test_frame_properties(self):
        atoms_data = {"position": np.array([[0,0,0]], dtype=float), "name": np.array(["C"], dtype="U1")}
        frame = Frame(data={"atoms": atoms_data})
        frame.timestep = 100
        frame['time_prop'] = 10.0 # Custom scalar property
        
        # 检查数据是否正确存储
        assert "position" in frame['atoms'].data_vars
        assert "name" in frame['atoms'].data_vars
        assert frame['time_prop'] == 10.0
        assert frame.timestep == 100

    def test_frame_atom_access(self):
        atoms_data = {
            "position": np.array([[0,0,0]], dtype=float),
            "name": np.array(["C"], dtype="U1"),
            "id": np.array([1], dtype=int),
            "atomic_number": np.array([6], dtype=int)
        }
        frame = Frame(data={"atoms": atoms_data})
        assert frame['atoms']["name"][0].item() == "C"
        assert frame['atoms']["position"].shape == (1, 3)

    def test_frame_set_atoms_from_dict(self):
        frame = Frame()
        atoms_data = {
            "position": np.array([[0,0,0]], dtype=float),
            "name": np.array(["H"], dtype="U1"),
            "id": np.array([10], dtype=int),
            "atomic_number": np.array([1], dtype=int)
        }
        frame['atoms'] = atoms_data # Uses __setitem__
        
        # 检查数据变量是否正确存储
        assert "position" in frame['atoms'].data_vars
        assert "name" in frame['atoms'].data_vars
        assert frame['atoms']["name"][0].item() == "H"

    def test_frame_iteration_over_keys(self):
        atoms_data = {
            "position": np.array([[0,0,0], [1,1,1]], dtype=float),
            "name": np.array(["C", "H"], dtype="U1"),
            "id": np.array([1,2], dtype=int)
        }
        frame = Frame(data={"atoms": atoms_data})
        frame.timestep = 10
        frame['custom_scalar'] = 5.0
        
        keys = list(frame)
        assert "atoms" in keys
        assert "timestep" in keys
        assert "custom_scalar" in keys
        
        # 遍历原子名
        names = [name.item() for name in frame['atoms']["name"]]
        assert names == ["C", "H"]

    def test_frame_copy(self):
        atoms_data = {"position": np.array([[0,0,0]], dtype=float), "name": np.array(["C"], dtype="U1")}
        box = create_dummy_box()
        frame1 = Frame(data={"atoms": atoms_data}, box=box)
        frame1.timestep = 50
        frame1['custom_time'] = 5.0
        
        frame2 = frame1.copy()
        
        # 检查数据是否正确复制
        assert "position" in frame2['atoms'].data_vars
        assert "name" in frame2['atoms'].data_vars
        assert np.array_equal(frame2['atoms']["position"].data, frame1['atoms']["position"].data)
        assert frame2['atoms']["position"].data is not frame1['atoms']["position"].data
        
        assert frame2.box is not None
        assert frame1.box is not None
        assert np.array_equal(frame2.box.matrix, frame1.box.matrix)
        if frame1.box:
             assert frame2.box.matrix is not frame1.box.matrix

        assert frame2.timestep == frame1.timestep
        assert frame2['custom_time'] == frame1['custom_time']

    def test_frame_to_dict(self):
        atoms_data = {
            "xyz": np.array([[0,0,0]], dtype=float),
            "name": np.array(["C"], dtype="U1"),
            "id": np.array([1], dtype=int),
            "atomic_number": np.array([6], dtype=int)
        }
        box = create_dummy_box()
        frame = Frame(data={"atoms": atoms_data}, box=box)
        frame.timestep = 1
        frame['custom_scalar'] = "test"
        frame_dict = frame.to_dict()

        data = frame_dict["data"]
        print(data)
        assert "atoms" in data
        assert isinstance(data["atoms"], dict)
        
        # 检查xarray的to_dict输出格式
        atoms_dict = data["atoms"]
        assert "data_vars" in atoms_dict
        assert "coords" in atoms_dict
        assert "dims" in atoms_dict
        assert "attrs" in atoms_dict
        
        # 检查数据变量
        data_vars = atoms_dict["data_vars"]
        assert "xyz" in data_vars
        assert "name" in data_vars
        assert "id" in data_vars
        assert "atomic_number" in data_vars

        metadata = frame_dict["metadata"]
        assert metadata["timestep"] == 1
        assert metadata["custom_scalar"] == "test"
        
        # Box is stored at top level, not in metadata
        assert "box" in frame_dict
        expected_box_dict = {
            'matrix': box.matrix.tolist(),
            'pbc': box.pbc.tolist(),
            'origin': box.origin.tolist()
        }
        assert frame_dict["box"] == expected_box_dict

    def test_frame_concat_dataset(self):
        atoms1 = {"position": np.array([[0,0,0]], dtype=float), "name": np.array(["C"], dtype="U1")}
        atoms2 = {"position": np.array([[1,1,1]], dtype=float), "name": np.array(["H"], dtype="U1")}
        f1 = Frame({"atoms": atoms1})
        f2 = Frame({"atoms": atoms2})
        
        f_concat = Frame.concat([f1, f2])
        
        # 检查数据变量是否存在
        assert "position" in f_concat["atoms"].data_vars
        assert "name" in f_concat["atoms"].data_vars
        
        # 检查数据值（新的维度结构下）
        assert f_concat["atoms"]["name"][0].item() == "C"
        assert f_concat["atoms"]["name"][1].item() == "H"
        assert np.array_equal(f_concat["atoms"]["position"][0].values, [0,0,0])
        assert np.array_equal(f_concat["atoms"]["position"][1].values, [1,1,1])

    def test_frame_mixed_datatypes(self):
        """测试Dataset能正确处理不同数据类型"""
        atoms_data = {
            "position": np.array([[0.0, 0.0, 0.0], [1.5, 2.3, -0.5]], dtype=float),
            "name": np.array(["C", "O"], dtype="U2"),  # 字符串
            "id": np.array([1, 2], dtype=int),  # 整数
            "charge": np.array([0.0, -0.5], dtype=float),  # 浮点数
            "is_backbone": np.array([True, False], dtype=bool)  # 布尔值
        }
        frame = Frame({"atoms": atoms_data})
        
        # 验证每种数据类型都正确存储
        assert frame["atoms"]["position"].dtype == np.float64
        assert frame["atoms"]["name"].dtype.kind == "U"  # Unicode string
        assert frame["atoms"]["id"].dtype == np.int64
        assert frame["atoms"]["charge"].dtype == np.float64
        assert frame["atoms"]["is_backbone"].dtype == np.bool_
        
        # 验证形状
        assert frame["atoms"]["position"].shape == (2, 3)
        assert frame["atoms"]["name"].shape == (2,)
        assert frame["atoms"]["charge"].shape == (2,)
        
        # 验证值
        assert frame["atoms"]["name"][0].item() == "C"
        assert frame["atoms"]["charge"][1].item() == -0.5
        assert frame["atoms"]["is_backbone"][0].item() == True
        assert frame["atoms"]["is_backbone"][1].item() == False

    def test_frame_dataset_coordinates(self):
        """测试Dataset的维度系统（新的_dict_to_dataset实现）"""
        atoms_data = {
            "position": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
            "velocity": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float),
            "name": np.array(["C", "H"], dtype="U1")
        }
        frame = Frame({"atoms": atoms_data})
        
        # 检查数据变量是否正确存储
        assert "position" in frame["atoms"].data_vars
        assert "velocity" in frame["atoms"].data_vars  
        assert "name" in frame["atoms"].data_vars
        
        # 检查维度名称（新的系统化命名）
        position_dims = frame["atoms"]["position"].dims
        velocity_dims = frame["atoms"]["velocity"].dims
        name_dims = frame["atoms"]["name"].dims
        
        # 验证形状
        assert frame["atoms"]["position"].shape == (2, 3)
        assert frame["atoms"]["velocity"].shape == (2, 3)
        assert frame["atoms"]["name"].shape == (2,)
        
        # 验证数据值
        assert frame["atoms"]["name"][0].item() == "C"
        assert frame["atoms"]["name"][1].item() == "H"
        assert np.array_equal(frame["atoms"]["position"][0].values, [0.0, 0.0, 0.0])
        assert np.array_equal(frame["atoms"]["position"][1].values, [1.0, 1.0, 1.0])

    def test_frame_empty_dataset(self):
        """测试空Dataset的处理"""
        frame = Frame({"atoms": {}})
        
        # 检查空dataset的基本属性
        assert isinstance(frame["atoms"], xr.Dataset)
        assert len(frame["atoms"].data_vars) == 0
        assert len(frame["atoms"].coords) == 0

    def test_frame_lammps_data_compatibility(self):
        """Test Frame compatibility with LAMMPS data format."""
        # Test with 'q' field instead of 'charge'
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 1],
            'type': ['O', 'H', 'H'],
            'q': [-0.8476, 0.4238, 0.4238],  # Using 'q' not 'charge'
            'xyz': [[0.0, 0.0, 0.0], [0.816, 0.577, 0.0], [-0.816, 0.577, 0.0]]
        }
        frame = Frame()
        frame["atoms"] = atoms_data
        frame.box = Box(np.eye(3) * 10.0)
        
        # Verify frame structure
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert "q" in atoms.data_vars
        assert "xyz" in atoms.data_vars
        
        # Check dimensions and values
        assert atoms.sizes[list(atoms.sizes.keys())[0]] == 3  # 3 atoms
        charges = atoms["q"].values if hasattr(atoms["q"], 'values') else atoms["q"]
        assert np.isclose(charges[0], -0.8476)
        assert np.isclose(charges[1], 0.4238)

    def test_frame_bond_angle_data(self):
        """Test Frame with bonds and angles data."""
        frame = Frame()
        
        # Water molecule with bonds and angles
        atoms_data = {
            'id': [0, 1, 2],
            'molid': [1, 1, 1],
            'type': ['O', 'H', 'H'],
            'q': [-0.8476, 0.4238, 0.4238],
            'xyz': [[0.0, 0.0, 0.0], [0.816, 0.577, 0.0], [-0.816, 0.577, 0.0]]
        }
        
        bonds_data = {
            'id': [0, 1],
            'i': [0, 0],  # O-H bonds
            'j': [1, 2]
        }
        
        angles_data = {
            'id': [0],
            'i': [1],  # H-O-H angle
            'j': [0],
            'k': [2]
        }
        
        frame["atoms"] = atoms_data
        frame["bonds"] = bonds_data
        frame["angles"] = angles_data
        frame.box = Box(np.eye(3) * 10.0)
        
        # Verify all sections exist
        assert "atoms" in frame
        assert "bonds" in frame
        assert "angles" in frame
        
        # Check data structure
        bonds = frame["bonds"]
        assert "i" in bonds.data_vars
        assert "j" in bonds.data_vars
        assert bonds.sizes[list(bonds.sizes.keys())[0]] == 2  # 2 bonds
        
        angles = frame["angles"]
        assert "i" in angles.data_vars
        assert "j" in angles.data_vars
        assert "k" in angles.data_vars
        assert angles.sizes[list(angles.sizes.keys())[0]] == 1  # 1 angle

    def test_frame_coordinate_formats(self):
        """Test Frame handling of different coordinate formats."""
        frame1 = Frame()
        
        # Test with xyz array format
        atoms_data_xyz = {
            'id': [0, 1, 2],
            'type': ['A', 'A', 'A'],
            'xyz': [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
        }
        frame1["atoms"] = atoms_data_xyz
        
        # Test with separate x,y,z fields
        frame2 = Frame()
        atoms_data_separate = {
            'id': [0, 1, 2],
            'type': ['A', 'A', 'A'],
            'x': [0.0, 1.0, 2.0],
            'y': [0.0, 1.0, 2.0],
            'z': [0.0, 1.0, 2.0]
        }
        frame2["atoms"] = atoms_data_separate
        
        # Both should work
        assert "xyz" in frame1["atoms"].data_vars
        assert "x" in frame2["atoms"].data_vars
        assert "y" in frame2["atoms"].data_vars
        assert "z" in frame2["atoms"].data_vars
        
        # Check values
        xyz_coords = frame1["atoms"]["xyz"].values if hasattr(frame1["atoms"]["xyz"], 'values') else frame1["atoms"]["xyz"]
        assert np.allclose(xyz_coords[0], [0.0, 0.0, 0.0])
        assert np.allclose(xyz_coords[2], [2.0, 2.0, 2.0])

    def test_frame_timestep_handling(self):
        """Test Frame timestep handling for trajectory data."""
        frame = Frame()
        
        atoms_data = {
            'id': [0, 1],
            'type': [1, 2],
            'x': [0.0, 1.0],
            'y': [0.0, 0.0],
            'z': [0.0, 0.0]
        }
        frame["atoms"] = atoms_data
        frame["timestep"] = 1000
        frame.box = Box(np.eye(3) * 5.0)
        
        # Check timestep access
        assert frame.get("timestep") == 1000
        assert frame["timestep"] == 1000
        
        # Test timestep modification
        frame["timestep"] = 2000
        assert frame["timestep"] == 2000

    def test_frame_empty_handling(self):
        """Test Frame graceful handling of empty data."""
        frame = Frame()
        
        # Empty atoms data
        empty_atoms = {}
        frame["atoms"] = empty_atoms
        
        # Should create empty dataset
        assert "atoms" in frame
        atoms = frame["atoms"]
        assert isinstance(atoms, xr.Dataset)
        assert len(atoms.data_vars) == 0
        
        # Test with minimal data
        minimal_atoms = {'id': [0]}
        frame["atoms"] = minimal_atoms
        assert "id" in frame["atoms"].data_vars