class Potential:

    def __init__(self):
        pass

    def __call__(self,input):
        return self.forward(input)

    def forward(self):
        pass

    def energy(self):
        pass

    def forces(self):
        pass

class Potentials(Potential):

    def __init__(self, *potentials):
        super().__init__()
        self.potentials = potentials

    def __call__(self):
        pass

    def forward(self):
        pass