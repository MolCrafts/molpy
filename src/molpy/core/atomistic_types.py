"""Atom / Bond / Angle / Dihedral / Improper / VirtualSite view types.

Handle views over a molrs ``Atomistic`` world. Kept separate from the
:class:`~molpy.core.atomistic.Atomistic` leaf so the leaf file stays a thin
def_* / algorithm-wrapper shell.
"""

from __future__ import annotations

from typing import Any

from .entity import Entity, Link


class Atom(Entity):
    """Atom view: one node in a molrs ``Atomistic`` world (or pending)."""

    __slots__ = ()

    def __repr__(self) -> str:
        ident = self.get("element") or self.get("symbol") or self.get("type")
        return f"<Atom: {ident if ident is not None else id(self)}>"

    @property
    def is_virtual(self) -> bool:
        """True if this node carries a virtual-site marker (``vsite`` field).

        The marker is a stored data field, not the Python class: molrs
        re-materialises nodes as plain :class:`Atom`, so identity must be read
        from the persisted ``vsite`` field rather than ``isinstance``.
        """
        return self.get("vsite") is not None


class VirtualSite(Atom):
    """A massless / rule-placed auxiliary particle. Data only — no energy.

    Carries a persistent ``vsite`` marker field (set on construction) plus the
    usual atom data. Subclasses set the marker value via ``_vsite_kind``.
    Identity after a molrs round-trip is read from the ``vsite`` field
    (:attr:`Atom.is_virtual`), since the Python subclass is not preserved.
    """

    __slots__ = ()
    _vsite_kind = "virtual"

    def __init__(self, mapping: Any = None, /, **attrs: Any) -> None:
        attrs.setdefault("vsite", self._vsite_kind)
        super().__init__(mapping, **attrs)


class DrudeParticle(VirtualSite):
    """Polarizable Drude shell: co-located with its core, spring-bound."""

    __slots__ = ()
    _vsite_kind = "drude"


class MasslessSite(VirtualSite):
    """Rigid geometric site (e.g. TIP4P M-site, lone pair); no spring."""

    __slots__ = ()
    _vsite_kind = "massless"


class Bond(Link[Atom]):
    """Covalent bond between two atoms (molrs relation kind ``bonds``)."""

    __slots__ = ()
    _kind = "bonds"

    def __init__(self, a: Atom, b: Atom, /, **attrs: Any) -> None:
        assert isinstance(a, Atom), f"atom a must be an Atom instance, got {type(a)}"
        assert isinstance(b, Atom), f"atom b must be an Atom instance, got {type(b)}"
        super().__init__([a, b], **attrs)

    def __repr__(self) -> str:
        return f"<Bond: {self.itom} - {self.jtom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]


class Angle(Link[Atom]):
    """Valence angle over three atoms i--j--k (kind ``angles``)."""

    __slots__ = ()
    _kind = "angles"

    def __init__(self, a: Atom, b: Atom, c: Atom, /, **attrs: Any) -> None:
        for x in (a, b, c):
            assert isinstance(x, Atom), f"endpoint must be an Atom, got {type(x)}"
        super().__init__([a, b, c], **attrs)

    def __repr__(self) -> str:
        return f"<Angle: {self.itom} - {self.jtom} - {self.ktom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]


class Dihedral(Link[Atom]):
    """Proper dihedral (torsion) over four atoms (kind ``dihedrals``)."""

    __slots__ = ()
    _kind = "dihedrals"

    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any) -> None:
        for x in (a, b, c, d):
            assert isinstance(x, Atom), f"endpoint must be an Atom, got {type(x)}"
        super().__init__([a, b, c, d], **attrs)

    def __repr__(self) -> str:
        return f"<Dihedral: {self.itom} - {self.jtom} - {self.ktom} - {self.ltom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        return self.endpoints[3]


class Improper(Link[Atom]):
    """Improper torsion over four atoms, ``i`` central (kind ``impropers``)."""

    __slots__ = ()
    _kind = "impropers"

    def __init__(self, a: Atom, b: Atom, c: Atom, d: Atom, /, **attrs: Any) -> None:
        for x in (a, b, c, d):
            assert isinstance(x, Atom), f"endpoint must be an Atom, got {type(x)}"
        super().__init__([a, b, c, d], **attrs)

    def __repr__(self) -> str:
        return f"<Improper: {self.itom} - {self.jtom} - {self.ktom} - {self.ltom}>"

    @property
    def itom(self) -> Atom:
        return self.endpoints[0]

    @property
    def jtom(self) -> Atom:
        return self.endpoints[1]

    @property
    def ktom(self) -> Atom:
        return self.endpoints[2]

    @property
    def ltom(self) -> Atom:
        return self.endpoints[3]
