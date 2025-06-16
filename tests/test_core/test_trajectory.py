import pytest
import numpy as np
from molpy.core.struct import AtomicStructure, Atom
from molpy.core.trajectory import Frame, Trajectory

def make_frame(x=0, time=0.0, label=None):
    # 构造一个合规的 dict 数据，模拟原子坐标
    data = {
        "position": {"xyz": [[x, 0, 0], [x+1, 0, 0]]},
        "element": {"name": ["C", "H"]}
    }
    meta = {"time": time}
    if label:
        meta["label"] = label
    return Frame(data, meta=meta)

def test_frame_basic():
    f = make_frame(0, time=0.0, label="start")
    assert f.get_meta("time") == 0.0
    assert f.get_meta("label") == "start"
    f2 = f.copy()
    assert f2 is not f
    assert f2["position"].shape == f["position"].shape
    assert (f2["position"].values == f["position"].values).all()

def test_trajectory_append_and_slice():
    traj = Trajectory()
    for i in range(5):
        traj.append(make_frame(i, time=i))
    assert len(traj) == 5
    assert isinstance(traj[0], Frame)
    subtraj = traj[1:4]
    assert isinstance(subtraj, Trajectory)
    assert len(subtraj) == 3
    assert subtraj[0].get_meta("time") == 1

def test_trajectory_iter_and_extend():
    traj = Trajectory()
    frames = [make_frame(i, time=i) for i in range(3)]
    traj.extend(frames)
    times = [f.get_meta("time") for f in traj]
    assert times == [0,1,2]

def test_trajectory_copy():
    traj = Trajectory([make_frame(i, time=i) for i in range(2)], foo="bar")
    traj2 = traj.copy()
    assert traj2 is not traj
    assert len(traj2) == 2
    assert traj2.meta["foo"] == "bar"
    for f1, f2 in zip(traj, traj2):
        assert f1 is not f2
        assert (f1["position"].values == f2["position"].values).all()

def test_trajectory_concat():
    traj1 = Trajectory([make_frame(1, time=1, label="a")], foo=1)
    traj2 = Trajectory([make_frame(2, time=2, label="b")], foo=2)
    concat_traj = Trajectory.concat([traj1, traj2])
    assert len(concat_traj) == 2
    assert isinstance(concat_traj[0], Frame)
    # 检查 Frame 的 position 数据 shape
    assert concat_traj[0]._data["position"].shape[0] == 2
    assert concat_traj[1].get_meta("label") == "b"
    # Meta comes from the first
    assert concat_traj.get_meta("foo") == 1

def test_trajectory_to_dict():
    traj = Trajectory([make_frame(0, time=0, label="x")], foo="meta")
    d = traj.to_dict()
    assert "frames" in d and isinstance(d["frames"], list)
    assert d["meta"]["foo"] == "meta"
    assert d["frames"][0]["label"] == "x"

def test_trajectory_meta_methods():
    traj = Trajectory()
    traj.set_meta("baz", 123)
    assert traj.get_meta("baz") == 123
    assert traj.meta["baz"] == 123
