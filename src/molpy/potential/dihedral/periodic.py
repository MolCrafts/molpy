"""Periodic torsion dihedral force field styles (used by GAFF/AMBER)."""

from molpy.core.forcefield import AtomType, DihedralStyle, DihedralType


class DihedralPeriodicType(DihedralType):
    """Periodic torsion dihedral type with multiple terms.

    Each term has: periodicity (n), force constant (k), phase angle (phi0).
    V(phi) = sum_i k_i * (1 + cos(n_i * phi - phi0_i))
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


class DihedralPeriodicStyle(DihedralStyle):
    """Periodic torsion dihedral style (AMBER/GAFF).

    Uses periodic cosine terms: V(phi) = sum_i k_i * (1 + cos(n_i * phi - phi0_i))
    """

    def __init__(self):
        super().__init__("periodic")


class DihedralFourierStyle(DihedralStyle):
    """LAMMPS *fourier* dihedral style for AMBER/GAFF force fields.

    Parameters stored per type: k1=K, k2=n (int), k3=phase (degrees), k4=weight.
    Written as: dihedral_coeff ID m K n phase  (m=1 for single-term AMBER dihedrals).
    """

    def __init__(self):
        super().__init__("fourier")

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        name: str = "",
        **kwargs,
    ) -> DihedralPeriodicType:
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        dt = DihedralPeriodicType(name, itom, jtom, ktom, ltom, **kwargs)
        self.types.add(dt)
        return dt

    def def_type(
        self,
        itom: AtomType,
        jtom: AtomType,
        ktom: AtomType,
        ltom: AtomType,
        name: str = "",
        **kwargs,
    ) -> DihedralPeriodicType:
        """Define periodic dihedral type.

        Args:
            itom: First atom type
            jtom: Second atom type
            ktom: Third atom type
            ltom: Fourth atom type
            name: Optional name (defaults to itom-jtom-ktom-ltom)
            **kwargs (Any): Periodic parameters (periodicity1, k1, phase1, ...).

        Returns:
            DihedralPeriodicType instance
        """
        if not name:
            name = f"{itom.name}-{jtom.name}-{ktom.name}-{ltom.name}"
        dt = DihedralPeriodicType(name, itom, jtom, ktom, ltom, **kwargs)
        self.types.add(dt)
        return dt
