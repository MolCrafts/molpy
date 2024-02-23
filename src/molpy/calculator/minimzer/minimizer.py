from ..calculator import BaseCalculator

class Minimizer(BaseCalculator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def minimize(self, *args, **kwargs):
        pass