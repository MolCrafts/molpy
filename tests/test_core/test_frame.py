import pytest
import molpy as mp
from nesteddict import ArrayDict

class TestFrame:

    @pytest.fixture()
    def frame(self):
        return mp.Frame({
            'atoms': ArrayDict({
                'id': [1, 2, 3, 4],
                'type': [1, 2, 3, 4],
                'x': [0, 1, 2, 3],
                'y': [0, 1, 2, 3],
                'z': [0, 1, 2, 3],
            })
        }, style="atomic")
    
    def test_slice(self, frame):
        assert isinstance(frame['atoms'], ArrayDict)
        assert len(frame['atoms']) == 5
        assert frame['atoms']['id'].tolist() == [1, 2, 3, 4]
        
    def test_init_with_dataframe(self):
        data = {
            'atoms': ArrayDict({
                'id': [1, 2],
                'type': [1, 2],
                'x': [0.0, 1.0],
                'y': [0.0, 1.0],
                'z': [0.0, 1.0],
            })
        }
        frame = mp.Frame(data)
        assert 'atoms' in frame
        assert isinstance(frame['atoms'], ArrayDict)

    def test_init_with_dict(self):
        data = {
            'atoms': ArrayDict({
                'id': [1, 2],
                'type': [1, 2],
                'x': [0.0, 1.0],
                'y': [0.0, 1.0],
                'z': [0.0, 1.0],
            })
        }
        frame = mp.Frame(data)
        assert 'atoms' in frame
        assert isinstance(frame['atoms'], ArrayDict)

    def test_concat(self, frame):
        frame2 = mp.Frame({
            'atoms': ArrayDict({
                'id': [5, 6],
                'type': [5, 6],
                'x': [4, 5],
                'y': [4, 5],
                'z': [4, 5],
            })
        })
        concatenated = mp.Frame.from_frames([frame, frame2])
        assert concatenated['atoms']['id'].tolist() == [1, 2, 3, 4, 5, 6]

    def test_split(self, frame):
        split_frames = frame.split([1, 1, 2, 2])
        assert len(split_frames) == 2
        assert split_frames[0]['atoms']['id'].tolist() == [1, 2]
        assert split_frames[1]['atoms']['id'].tolist() == [3, 4]

    def test_box_property(self):
        box = mp.Box()
        frame = mp.Frame()
        frame.box = box
        assert frame.box == box
        frame.box = None
        assert frame.box is None

    def test_to_struct(self, frame):
        frame['bonds'] = ArrayDict({'i': [1], 'j': [2]})
        struct = frame.to_struct()
        assert 'atoms' in struct
        assert 'bonds' in struct
        assert len(struct['atoms']) == 4
        assert len(struct['bonds']) == 1

    def test_copy(self, frame):
        frame_copy = frame.copy()
        assert frame_copy is not frame
        assert frame_copy['atoms'] == frame['atoms']

    def test_add_operator(self, frame):
        frame2 = mp.Frame({
            'atoms': ArrayDict({
                'id': [5, 6],
                'type': [5, 6],
                'x': [4, 5],
                'y': [4, 5],
                'z': [4, 5],
            })
        })
        combined = frame + frame2
        assert combined['atoms']['id'].tolist() == [1, 2, 3, 4, 5, 6]

    def test_mul_operator(self, frame):
        multiplied = frame * 2
        assert len(multiplied) == 5
