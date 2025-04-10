import pint


class Unit(pint.UnitRegistry):
    
    UNIT_STYLES = {
        "real": {
            "length": "angstrom",
            "energy": "kcal/N_A",
            "mass": "gram/mole",
            "time": "femtosecond"
        },
        "lj": {  # define when initializing
            "length": "2.5 angstrom",
            "energy": "1.0 kcal/N_A",
            "mass": "12.011 gram/mole",
            "time": "1 nanosecond"
        }
    }

    def __init__(self, style="real", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.style = style

    def _after_init(self):
        super()._after_init()

        style = self.UNIT_STYLES[self.style]
        for dim, unit in style.items():
            self.define(f"{dim} = {unit}")
