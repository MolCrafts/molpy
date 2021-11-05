# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-25
# version: 0.0.1

from molpy.factory import fromPDB
from pathlib import Path

def test_read_D_lactic():
    
    lactic = fromPDB(Path(__file__).parent.parent/'samples/D-lactic.pdb')
    assert lactic[1].natoms == 12
    assert lactic[1].nbonds == 11