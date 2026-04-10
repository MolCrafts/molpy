"""Periodic improper dihedral force field styles (used by GAFF/AMBER)."""

from molpy.core.forcefield import AtomType, ImproperStyle, ImproperType


class ImproperPeriodicType(ImproperType):
    """Periodic improper dihedral type.

    V(phi) = k * (1 + cos(n * phi - phi0))
    """

    def __init__(
        self,
        name: str,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        **kwargs,
    ):
        super().__init__(name, itom, jtom, ktom, ltom, **kwargs)


class ImproperPeriodicStyle(ImproperStyle):
    """Periodic improper dihedral style (AMBER/GAFF).

    V(phi) = k * (1 + cos(n * phi - phi0))
    """

    def __init__(self):
        super().__init__("periodic")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        name: str = "",
        **kwargs,
    ) -> ImproperPeriodicType:
        """Define periodic improper type.

        Args:
            itom: First atom type
            jtom: Second atom type (usually central atom in AMBER)
            ktom: Third atom type
            ltom: Fourth atom type
            name: Optional name
            **kwargs: Periodic parameters (periodicity1, k1, phase1, ...)

        Returns:
            ImproperPeriodicType instance
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        it = ImproperPeriodicType(name, itom, jtom, ktom, ltom, **kwargs)
        self.types.add(it)
        return it
