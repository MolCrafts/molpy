# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-31
# version: 0.0.1

from molpy.io.xml import read_xml_forcefield
from pathlib import Path

class TestXMLForceField:
    
    def test_mpid(self):
        with open(Path(__file__).parent.parent/'samples/mpidwater.xml') as f:
            read_xml_forcefield(f)