# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-08
# version: 0.0.1

import pytest
import molpy as mp
import numpy as np

class TestStructData:

    @pytest.fixture(name='data')
    def struct_data(self):
        raw = {
            'int': np.ones(10, dtype=np.int32),
            'float': np.ones(10, dtype=np.float32),
            '2d': np.ones((10, 3), dtype=np.float32),
            'string': np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']),
            'bool': np.ones(10, dtype=bool),
            'pyObject': np.array([None]* 10)
        }
        return mp.core.struct.ArrayDict(raw)

    def test_length(self):

        data = mp.ArrayDict()
        assert data.length == 0
        
        

    def test_get_field(self, data):

        assert data.get('int').dtype == np.int32
        assert data.get('float').dtype == np.float32
        assert data.get('2d').dtype == np.float32
        assert data.get('string').dtype == np.dtype('U1')
        assert data.get('bool').dtype == bool
        assert data.get('pyObject').dtype == np.object_

        assert data[['int', 'float']].dtype.names == ('int', 'float')
        assert data[['int', 'float', '2d']].dtype.names == ('int', 'float', '2d')

    def test_slice(self, data):

        assert len(data[0:5]) == 5
        assert data[0:5][['int', 'float']].dtype.names == ('int', 'float')
        assert data[0:5][['int', 'float', '2d']].dtype.names == ('int', 'float', '2d')

    def test_set_field(self, data):

        with pytest.raises(ValueError):
            data.set('int', np.ones(5, dtype=np.int32))