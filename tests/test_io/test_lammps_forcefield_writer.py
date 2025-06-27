import pytest
from io import StringIO
import numpy as np
from unittest.mock import MagicMock, PropertyMock
from molpy.io.forcefield.lammps import LAMMPSForceFieldWriter
import molpy as mp

@pytest.fixture
def mock_ff_dict():
    """Fixture for a mock force field dictionary."""
    return {
        'pair': {
            'style': 'lj/cut',
            'coeffs': {
                1: {'epsilon': 1.0, 'sigma': 1.0},
                2: {'epsilon': 1.5, 'sigma': 1.2},
            }
        },
        'bond': {
            'style': 'harmonic',
            'coeffs': {
                1: {'k': 100.0, 'r0': 1.5},
                2: {'k': 120.0, 'r0': 1.2},
            }
        },
        'angle': {
            'style': 'harmonic',
            'coeffs': {
                1: {'k': 50.0, 'theta0': 109.5},
            }
        },
        'dihedral': {
            'style': 'hybrid',
            'styles': ['opls', 'harmonic'],
            'coeffs': {
                1: {'style': 'opls', 'coeffs': [1.0, -0.5, 0.25, 0.0]},
                2: {'style': 'harmonic', 'coeffs': [25.0, -1, 3]},
            }
        },
        'improper': {
            'style': 'cvff',
            'coeffs': {
                1: {'k': 30.0, 'd': -1, 'n': 2},
            }
        }
    }

def dict_to_ff_mock(d: dict):
    """Converts a dictionary to a mocked ForceField object for testing."""
    ff = MagicMock(spec=mp.ForceField)

    # Helper to create a style mock
    def create_style_mock(info, style_type):
        if not info:
            return []
        style = MagicMock()
        style.name = info['style']
        type(style).parms = PropertyMock(return_value=info.get('parms', []))
        type(style).__contains__ = lambda _, item: item in info
        
        types = []
        if 'coeffs' in info:
            for name, coeffs in info['coeffs'].items():
                typ = MagicMock()
                type(typ).name = PropertyMock(return_value=str(name))
                
                if style_type == 'pair':
                    at = MagicMock()
                    type(at).name = PropertyMock(return_value=str(name))
                    type(typ).atomtypes = PropertyMock(return_value=[at, at])
                    parms = [coeffs['epsilon'], coeffs['sigma']]
                elif style_type == 'bond':
                    parms = [coeffs['k'], coeffs['r0']]
                elif style_type == 'angle':
                    parms = [coeffs['k'], coeffs['theta0']]
                elif style_type == 'improper':
                    parms = [coeffs['k'], coeffs['d'], coeffs['n']]
                else:
                    parms = []
                
                type(typ).parms = PropertyMock(return_value=parms)
                types.append(typ)
        type(style).types = PropertyMock(return_value=types)
        return [style]

    # Helper for hybrid styles
    def create_hybrid_style_mock(info):
        if not info:
            return []
        styles = []
        style_map = {}
        for s_name in info['styles']:
            style = MagicMock()
            style.name = s_name
            type(style).parms = PropertyMock(return_value=[])
            type(style).types = PropertyMock(return_value=[])
            styles.append(style)
            style_map[s_name] = style

        type_map = {s_name: [] for s_name in info['styles']}
        if 'coeffs' in info:
            for name, data in info['coeffs'].items():
                typ = MagicMock()
                type(typ).name = PropertyMock(return_value=str(name))
                type(typ).parms = PropertyMock(return_value=data['coeffs'])
                type_map[data['style']].append(typ)
        
        for s_name, types in type_map.items():
            type(style_map[s_name]).types = PropertyMock(return_value=types)
            
        return styles

    type(ff).pairstyles = PropertyMock(return_value=create_style_mock(d.get('pair'), 'pair'))
    type(ff).bondstyles = PropertyMock(return_value=create_style_mock(d.get('bond'), 'bond'))
    type(ff).anglestyles = PropertyMock(return_value=create_style_mock(d.get('angle'), 'angle'))
    type(ff).improperstyles = PropertyMock(return_value=create_style_mock(d.get('improper'), 'improper'))
    
    dihedral_info = d.get('dihedral')
    if dihedral_info and dihedral_info.get('style') == 'hybrid':
        type(ff).dihedralstyles = PropertyMock(return_value=create_hybrid_style_mock(dihedral_info))
    elif dihedral_info:
        type(ff).dihedralstyles = PropertyMock(return_value=create_style_mock(dihedral_info, 'dihedral'))
    else:
        type(ff).dihedralstyles = PropertyMock(return_value=[])

    return ff

def test_write_single_style(mock_ff_dict):
    """Test writing a section with a single style."""
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter()
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()

    assert "Pair Coeffs" in result
    assert "pair_coeff 1 1.000000 1.000000 # lj/cut" in result
    assert "pair_coeff 2 1.500000 1.200000 # lj/cut" in result
    assert "Bond Coeffs" in result
    assert "bond_coeff 1 100.000000 1.500000 # harmonic" in result
    assert "Angle Coeffs" in result
    assert "angle_coeff 1 50.000000 109.500000 # harmonic" in result
    assert "Improper Coeffs" in result
    assert "improper_coeff 1 30.000000 -1 2 # cvff" in result

def test_write_hybrid_style(mock_ff_dict):
    """Test writing a section with a hybrid style."""
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter()
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()

    assert "Dihedral Coeffs" in result
    assert "dihedral_coeff 1 opls 1.000000 -0.500000 0.250000 0.000000" in result
    assert "dihedral_coeff 2 harmonic 25.000000 -1 3" in result

def test_precision_control(mock_ff_dict):
    """Test the float_precision parameter."""
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter(precision=3)
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()

    assert "Pair Coeffs" in result
    assert "pair_coeff 1 1.000 1.000 # lj/cut" in result
    assert "Bond Coeffs" in result
    assert "bond_coeff 1 100.000 1.500 # harmonic" in result

def test_style_definitions(mock_ff_dict):
    """Test that style definitions are written correctly."""
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter()
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()

    assert "pair_style lj/cut" in result
    assert "bond_style harmonic" in result
    assert "angle_style harmonic" in result
    assert "dihedral_style hybrid opls harmonic" in result
    assert "improper_style cvff" in result

def test_missing_section(mock_ff_dict):
    """Test that missing sections are handled gracefully."""
    del mock_ff_dict['angle']
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter()
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()

    assert "Angle Coeffs" not in result
    assert "angle_style" not in result

def test_invalid_coeff_type(mock_ff_dict):
    """Test error handling for invalid coefficient types."""
    mock_ff_dict['pair']['coeffs'][1] = "invalid"
    with pytest.raises(TypeError):
        dict_to_ff_mock(mock_ff_dict)

def test_empty_coeffs(mock_ff_dict):
    """Test handling of empty coefficient dictionaries."""
    mock_ff_dict['bond']['coeffs'] = {}
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter()
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()
    assert "Bond Coeffs" not in result
    assert "bond_style harmonic" in result

def test_numpy_values(mock_ff_dict):
    """Test that numpy float and int types are handled correctly."""
    mock_ff_dict['pair']['coeffs'][1]['epsilon'] = np.float64(1.0)
    mock_ff_dict['improper']['coeffs'][1]['n'] = np.int32(2)
    ff = dict_to_ff_mock(mock_ff_dict)
    writer = LAMMPSForceFieldWriter(precision=2)
    output = StringIO()
    writer.write(ff, output)
    result = output.getvalue()
    assert "pair_coeff 1 1.00 1.00 # lj/cut" in result
    assert "improper_coeff 1 30.00 -1 2 # cvff" in result
