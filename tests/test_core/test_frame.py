import pytest
import molpy as mp
import pandas as pd

class TestFrame:

    @pytest.fixture()
    def frame(self):
        return mp.Frame({
            'atoms': {
                'id': [1, 2, 3, 4],
                'type': [1, 2, 3, 4],
                'x': [0, 1, 2, 3],
                'y': [0, 1, 2, 3],
                'z': [0, 1, 2, 3],
            }
        })
    
    def test_slice(self, frame):
        assert isinstance(frame['atoms'], pd.DataFrame)
        assert frame['atoms'].shape == (4, 5)
        assert frame['atoms']['id'].tolist() == [1, 2, 3, 4]