from molpy.core.forcefield import Forcefield
from pathlib import Path

data_path = Path(__file__).parent / Path('data/forcefields')

def tip3p():

    ff = Forcefield.from_xml(data_path / 'tip3p.xml')
    return ff