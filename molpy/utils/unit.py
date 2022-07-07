# author: Roy Kid
# contact: lijichen365@126.com
# date: 2022-07-07
# version: 0.0.1

import pint

unit = pint.UnitRegistry()

def is_quantity(x):
    return isinstance(x, unit.Quantity)

# --= common unit =--
