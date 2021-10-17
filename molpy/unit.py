# author: Roy Kid
# contact: lijichen365@126.com
# date: 2021-10-10
# version: 0.0.1

from pint import UnitRegistry
from pint.util import to_units_container
from typing import Any

class Unit:
    
    def __init__(self, system='SI') -> None:
        self.ureg = UnitRegistry()
        self.system = system
        
    def __getattr__(self, name: str) -> None:
        return getattr(self.ureg, name)
    
    def mass(self, m):
        if self.system == 'SI':
            return m * self.ureg.kg
    
    def isMass(self, o):
        return o == to_units_container('mass')
    
    def isLength(self, o):
        return o == to_units_container('length')