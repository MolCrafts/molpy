# author: Roy Kid
# contact: lijichen365@126.com
# date: 2023-01-29
# version: 0.0.1

from pathlib import Path
import pytest
import subprocess

@pytest.fixture(name='test_data_path', scope='session')
def find_test_data():

    data_path = Path(__file__).parent / 'tests-data'
    if not data_path.exists():
        print('Downloading test data...')
        p = subprocess.Popen(f'git clone https://github.com/chemfiles/tests-data.git {data_path.parent}/tests-data', shell=True)
        p.wait()
        if p.returncode == 0:
            print('Download test data successfully.')
        else:
            raise RuntimeError('Download test data failed.')

    return data_path
