from abc import abstractmethod

from molpy.core.atomistic import Angle, Bond
from molpy.core.entity import StructLike
from molpy.core.forcefield import ForceField


class TypifierError(Exception): ...


class Typifier[T: StructLike]:
    def __init__(self, forcefield: ForceField):
        self.ff = forcefield

    @abstractmethod
    def typify(self, struct: T) -> T: ...


class AtomTypifierMixin[T](Typifier):
    def typify(self, struct: T) -> T:
        return struct


class BondTypifierMixin[T](Typifier):
    def typify(self, struct: T) -> T:
        return struct


class AngleTypifierMixin[T](Typifier):
    def typify(self, struct: T) -> T:
        return struct


class BaseTypifier:
    """Base class for typifiers."""

    pass


class AtomisticTypifier[T: StructLike](
    AtomTypifierMixin[T], BondTypifierMixin[T], AngleTypifierMixin[T]
): ...


class BondTypifier(BondTypifierMixin):
    def typify(self, bond: Bond) -> Bond: ...


class AngleTypifier(AngleTypifierMixin):
    def typifier(self, angle: Angle) -> Angle: ...


class OplsAtomisticTypifier(AtomisticTypifier):
    pass
