import pytest
import numpy.testing as npt
import numpy as np
import xarray as xr
import h5py
import io
import molpy as mp

class TestFrame:
    @pytest.fixture()
    def frame(self):
        atoms = xr.Dataset({
            'id': ('index', [1,2,3,4]),
            'type': ('index', [1,2,3,4]),
            'x': ('index', [0,1,2,3]),
            'y': ('index', [0,1,2,3]),
            'z': ('index', [0,1,2,3]),
        })
        return mp.Frame({'atoms': atoms}, style="atomic")

    def test_slice(self, frame):
        assert list(frame['atoms'].data_vars) == ['id','type','x','y','z']
        assert frame['atoms']['id'].to_numpy().tolist() == [1,2,3,4]

    def test_concat(self, frame):
        atoms2 = xr.Dataset({
            'id': ('index', [5,6]),
            'type': ('index', [5,6]),
            'x': ('index', [4,5]),
            'y': ('index', [4,5]),
            'z': ('index', [4,5]),
        })
        frame2 = mp.Frame({'atoms': atoms2})
        concatenated = mp.Frame.from_frames([frame, frame2])
        npt.assert_equal(concatenated['atoms']['id'].to_numpy(), [1,2,3,4,5,6])

    def test_split(self, frame):
        split_frames = frame.split([1,1,2,2])
        assert len(split_frames) == 2
        npt.assert_equal(split_frames[0]['atoms']['id'].to_numpy(), [1,2])
        npt.assert_equal(split_frames[1]['atoms']['id'].to_numpy(), [3,4])

    def test_box_property(self):
        box = mp.Box()
        frame = mp.Frame()
        frame.box = box
        assert frame.box == box
        frame.box = None
        assert frame.box is None

    def test_to_struct(self, frame):
        frame['bonds'] = xr.Dataset({'i': ('index', [1]), 'j': ('index', [2])})
        struct = frame.to_struct()
        assert 'atoms' in struct
        assert 'bonds' in struct
        assert len(struct['atoms']) == 4
        assert len(struct['bonds']) == 1

    def test_copy(self, frame):
        frame_copy = frame.copy()
        assert frame_copy is not frame
        assert frame_copy['atoms'].equals(frame['atoms'])

    def test_add_operator(self, frame):
        atoms2 = xr.Dataset({
            'id': ('index', [5,6]),
            'type': ('index', [5,6]),
            'x': ('index', [4,5]),
            'y': ('index', [4,5]),
            'z': ('index', [4,5]),
        })
        frame2 = mp.Frame({'atoms': atoms2})
        combined = frame + frame2
        assert combined['atoms']['id'].to_numpy().tolist() == [1,2,3,4,5,6]

    def test_mul_operator(self, frame):
        multiplied = frame * 2
        assert multiplied['atoms']['id'].size == 8

    def test_init_all_atom_frame(self):
        frame = mp.Frame(style='atomic')
        assert isinstance(frame, mp.AllAtomFrame)

    def test_to_h5df_bytes(self, frame):
        frame.box = mp.Box.cubic(1.0)
        data = frame.to_h5df()
        assert isinstance(data, (bytes, bytearray))
        with h5py.File(io.BytesIO(data), "r") as h5:
            assert "atoms" in h5
            assert "box" in h5
            assert np.array_equal(h5["atoms"]["id"][:], [1, 2, 3, 4])

    def test_to_h5df_path(self, tmp_path, frame):
        frame.box = mp.Box.cubic(1.0)
        path = tmp_path / "frame.h5"
        ret = frame.to_h5df(path)
        assert path.exists()
        assert ret == b""
        with h5py.File(path, "r") as h5:
            assert "atoms" in h5
            assert "box" in h5

