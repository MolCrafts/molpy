from dataclasses import dataclass
from pint import UnitRegistry
from ._utils import Singleton

class _Constants(metaclass=Singleton):

    def __init__(self, system, ureg) -> None:
        self._constants = {
            "boltzmann": "1.380649e-23 J/K",
            "avogadro": "6.02214076e23 mol^-1",
            "gas": "8.314462618 J/(mol K)"
        }
        self.system = system
        self.ureg = ureg

    def __getitem__(self, key: str) -> float:
        return self.ureg(self._constants[key]).magnitude
    
    def __getattr__(self, key: str) -> float:
        return self.ureg(self._constants[key]).magnitude
    

class _Units:

    @dataclass
    class Units:
        mass: str
        distance: str
        time: str
        energy: str
        charge: str
        velocity: str
        force: str
        torque: str
        temperature: str
        pressure: str
        dynamic_viscosity: str
        charge: str
        electric_field: str
        density: str

    _systems: dict[str, Units] = {

        "real": Units(
            mass="g/mol",
            distance="angstrom",
            time="fs",
            energy="kcal/mol",
            charge="e",
            velocity="angstrom/fs",
            force="kcal/(mol*angstrom)",
            torque="kcal/mol",
            temperature="K",
            pressure="atm",
            dynamic_viscosity="poise",
            electric_field="V/angstrom",
            density="g/cm^3"
        )

    }

    ureg = UnitRegistry()

    def __init__(self, system: str = "real") -> None:

        self.system = self._systems[system]
        self._constants = _Constants(self.system, self.ureg)

    def __call__(self, system: str):
        self.system = self._systems[system]

    def set_fundamental(self, mass: str, distance: str, energy: str):
        self.fmass = self.ureg(mass)
        self.fdistance = self.ureg(distance)
        self.fenergy = self.ureg(energy)
        self.ftime = (self.fenergy / (self.fmass * self.fdistance**2))**0.5

    def convert(self, value: float, unit: str, to: str) -> float:
        return value * self.ureg(unit).to(to).magnitude
    
    def reduce(self, value: float, unit: str) -> float:

        origin = value * self.ureg(unit)
        base_unit = origin.dimensionality
        if "[length]" in base_unit:
            origin /= self.fdistance
        if "[mass]" in base_unit:
            origin /= self.fmass
        if "[energy]" in base_unit:
            origin /= self.fenergy
        if "[time]" in base_unit:
            origin *= self.ftime
        return origin.to("").magnitude
    
    @property
    def constants(self):
        return self._constants
    
Unit = _Units("real")