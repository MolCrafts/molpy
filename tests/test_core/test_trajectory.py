import pytest
import numpy as np
import xarray as xr
from molpy.core.atomistic import AtomicStruct, Atom
from molpy.core.trajectory import Frame, Trajectory

def make_frame_dataset(x=0, time=0.0, label=None):
    """构造一个合规的 Dataset 数据，模拟原子坐标"""
    atoms_data = {
        "position": np.array([[x, 0, 0], [x+1, 0, 0]], dtype=float),
        "name": np.array(["C", "H"], dtype="U1")
    }
    meta = {"time": time}
    if label:
        meta["label"] = label
    return Frame({"atoms": atoms_data}, meta=meta)

def test_frame_basic():
    f = make_frame_dataset(0, time=0.0, label="start")
    assert f.get_meta("time") == 0.0
    assert f.get_meta("label") == "start"
    f2 = f.copy()
    assert f2 is not f
    # Check that both frames have atoms data
    assert "atoms" in f2
    assert "atoms" in f
    # Compare the structure
    assert f2["atoms"]["position"].values.shape == f["atoms"]["position"].values.shape
    assert (f2["atoms"]["position"].values == f["atoms"]["position"].values).all()

def test_trajectory_append_and_slice():
    traj = Trajectory()
    for i in range(5):
        traj.append(make_frame_dataset(i, time=i))
    assert len(traj) == 5
    assert isinstance(traj[0], Frame)
    subtraj = traj[1:4]
    assert isinstance(subtraj, Trajectory)
    assert len(subtraj) == 3
    assert subtraj[0].get_meta("time") == 1

def test_trajectory_iter_and_extend():
    traj = Trajectory()
    frames = [make_frame_dataset(i, time=i) for i in range(3)]
    traj.extend(frames)
    times = [f.get_meta("time") for f in traj]
    assert times == [0,1,2]

def test_trajectory_copy():
    traj = Trajectory([make_frame_dataset(i, time=i) for i in range(2)], foo="bar")
    traj2 = traj.copy()
    assert traj2 is not traj
    assert len(traj2) == 2
    assert traj2.meta["foo"] == "bar"
    for f1, f2 in zip(traj, traj2):
        assert f1 is not f2
        assert (f1["atoms"]["position"].values == f2["atoms"]["position"].values).all()

def test_trajectory_concat():
    traj1 = Trajectory([make_frame_dataset(1, time=1, label="a")], foo=1)
    traj2 = Trajectory([make_frame_dataset(2, time=2, label="b")], foo=2)
    concat_traj = Trajectory.concat([traj1, traj2])
    assert len(concat_traj) == 2
    assert isinstance(concat_traj[0], Frame)
    # 检查合并后的原子数
    frame0 = concat_traj[0]
    atoms0 = frame0["atoms"]  # This returns xr.Dataset
    assert atoms0["position"].values.shape[0] == 2  # 2 atoms
    assert concat_traj[1].get_meta("label") == "b"
    assert concat_traj.get_meta("foo") == 1

def test_trajectory_to_dict():
    traj = Trajectory([make_frame_dataset(0, time=0, label="x")], foo="meta")
    d = traj.to_dict()
    assert "frames" in d and isinstance(d["frames"], list)
    assert d["meta"]["foo"] == "meta"
    # 检查frames中的metadata
    frame_dict = d["frames"][0]
    assert "metadata" in frame_dict
    assert frame_dict["metadata"]["label"] == "x"

def test_trajectory_meta_methods():
    traj = Trajectory()
    traj.set_meta("baz", 123)
    assert traj.get_meta("baz") == 123
    assert traj.meta["baz"] == 123

def test_frame_concat_atoms_dtype_mismatch():
    """测试Frame.concat时atoms的同名字段dtype不同应报错"""
    import pytest
    atoms1 = {"position": np.array([[0,0,0]], dtype=float), "name": np.array(["C"], dtype="U1")}
    atoms2 = {"position": np.array([[1,1,1]], dtype=float), "name": np.array([b"H"], dtype="S1")}
    f1 = Frame({"atoms": atoms1})
    f2 = Frame({"atoms": atoms2})
    # 合并时name字段dtype不同，应该报错或警告
    with pytest.raises(Exception):
        Frame.concat([f1, f2])
