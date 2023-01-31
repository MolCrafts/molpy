# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from pathlib import Path
import pytest
import sys
sys.path.append('/home/roy/work/molpy/build/lib.linux-x86_64-cpython-310')

@pytest.fixture(name='test_data_path')
def find_test_data():

    data_path = Path(__file__).parent / 'tests-data'

    return data_path
