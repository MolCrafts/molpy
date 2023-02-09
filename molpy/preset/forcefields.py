from molpy.core.forcefield import Forcefield
from pathlib import Path
import urllib

data_path = Path(__file__).parent / Path('data/forcefields')

def tip3p():

    ff = Forcefield.from_xml(data_path / 'tip3p.xml')
    return ff

def gaff():
    # if not data_path / Path(f'gaff-{str(version)}.xml').exists():
    #     urllib.request.urlopen(f'https://raw.githubusercontent.com/openmm/openmmforcefields/35496e5095101c9db5ee8a10c7237d1c52dc7f6e/openmmforcefields/ffxml/amber/gaff/ffxml/gaff-2.11.xml')
    ff = Forcefield.from_xml(data_path / 'gaff-2.11.xml')
    return ff