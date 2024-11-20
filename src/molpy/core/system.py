from .space import Box
from .frame import Frame
from .forcefield import ForceField

class System:

    def __init__(
        self,
        frame: Frame | None = None,
        box: Box  | None = None,
        forcefield: ForceField | None = None,
    ):
        self.frame = frame or Frame()
        self.box = box or Box()
        self.forcefield = forcefield or ForceField()

    @property
    def n_atomtypes(self) -> int:
        return self.forcefield.n_atomtypes or len(self.frame["atoms"]["type"].unique())
    
    @property
    def n_bondtypes(self) -> int:
        return self.forcefield.n_bondtypes or len(self.frame["bonds"]["type"].unique())
    
    @property
    def n_angletypes(self) -> int:
        return self.forcefield.n_angletypes or len(self.frame["angles"]["type"].unique())
    
    @property
    def n_dihedraltypes(self) -> int:
        return self.forcefield.n_dihedraltypes or len(self.frame["dihedrals"]["type"].unique())
    
    def merge(self, other: "System") -> "System":
        system = System(
            frame=self.frame.merge(other.frame),
            box=self.box.merge(other.box),
            forcefield=self.forcefield.merge(other.forcefield),
        )
        return system

    @property
    def struct(self):
        return self.frame.to_struct()