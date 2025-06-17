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
        assert frame['atoms'].sizes['index'] == 2
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
        assert frame['atoms'].sizes['index'] == 1
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
        assert frame['atoms'].sizes['index'] == 1
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

    def test_frame_to_dict(self):
        atoms_data = {
            "position": np.array([[0,0,0]], dtype=float),
            "name": np.array(["C"], dtype="U1"),
            "id": np.array([1], dtype=int),
            "atomic_number": np.array([6], dtype=int)
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
        expected_box_dict = box.to_dict()
        assert frame_dict["box"] == expected_box_dict
        assert frame_dict["timestep"] == 1
        assert frame_dict["custom_scalar"] == "test"

    def test_frame_concat_dataset(self):
        atoms1 = {"position": np.array([[0,0,0]], dtype=float), "name": np.array(["C"], dtype="U1")}
        atoms2 = {"position": np.array([[1,1,1]], dtype=float), "name": np.array(["H"], dtype="U1")}
        f1 = Frame({"atoms": atoms1})
        f2 = Frame({"atoms": atoms2})
        
        f_concat = Frame.concat([f1, f2])
        assert f_concat["atoms"].sizes["index"] == 2
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
        """测试Dataset的坐标系统"""
        atoms_data = {
            "position": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float),
            "velocity": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=float),
            "name": np.array(["C", "H"], dtype="U1")
        }
        frame = Frame({"atoms": atoms_data})
        
        # 验证坐标
        assert "index" in frame["atoms"].coords
        assert "spatial" in frame["atoms"].coords
        assert list(frame["atoms"].coords["spatial"].values) == ["x", "y", "z"]
        assert len(frame["atoms"].coords["index"]) == 2
        
        # 验证可以通过坐标选择数据
        x_positions = frame["atoms"]["position"].sel(spatial="x")
        assert x_positions.shape == (2,)
        assert x_positions[0].item() == 0.0
        assert x_positions[1].item() == 1.0

    def test_frame_empty_dataset(self):
        """测试空Dataset的处理"""
        frame = Frame({"atoms": {}})
        assert frame["atoms"].sizes["index"] == 0
        assert len(frame["atoms"].data_vars) == 0