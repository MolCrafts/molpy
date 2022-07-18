# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-07
# version: 0.0.1

import pint

unit = pint.UnitRegistry()

def is_quantity(x):
    return isinstance(x, unit.Quantity)

# --= common unit =--
lines = [
    '@system nano using international',
        'nanoseconds', # time
        'nanometers', # length
        'attograms', # mass
]

unit.System.from_lines(lines, unit.get_base_units)
