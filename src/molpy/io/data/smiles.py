"""Read a SMILES / BigSMILES-monomer string into a 3D structure.

Unlike the file readers in this package the *source* is the notation string
itself, so :class:`SmilesReader` does not extend :class:`DataReader`; it keeps
the same ``.read()`` idiom and — like every other data reader — defaults to a
:class:`~molrs.Frame`.

**3D embedding is molrs-only** (:class:`molrs.conformer.Conformer` via
:class:`molpy.conformer.Conformer`). Plain SMILES parsing is also molrs
(:class:`molrs.io.SmilesIR`). Do **not** route this path through RDKit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, TypeVar, overload

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic
    from molrs import Frame

T = TypeVar("T")
_AsKind = type | Literal["frame", "atomistic"] | str


class SmilesReader:
    """Turn a SMILES (or BigSMILES ``{...}`` monomer) string into a structure.

    - Plain SMILES → :class:`molrs.io.SmilesIR` → graph
    - Leading ``{`` → rejected (use assembly topology helpers)
    - 3D → :class:`molpy.conformer.Conformer` (molrs-backed)

    :meth:`read` returns a tabular :class:`~molrs.Frame` (same default as
    :class:`~molpy.io.data.base.DataReader`). For the rich molecular graph
    use :meth:`read_as` with :class:`~molpy.core.atomistic.Atomistic`.

    Optionally derives angle/dihedral topology and assigns a unique ``name``
    to every atom (required by PDB export and the AmberTools wrappers).

    Example:
        >>> frame = SmilesReader("CC(=O)Oc1ccccc1C(=O)O").read()
        >>> mol = SmilesReader("CCO").read_as(Atomistic)
    """

    def __init__(
        self,
        smiles: str,
        *,
        add_hydrogens: bool = True,
        optimize: bool = True,
        gen_topo: bool = False,
        name_atoms: bool = True,
        seed: int | None = 0,
    ) -> None:
        self.smiles = smiles
        self.add_hydrogens = add_hydrogens
        self.optimize = optimize
        self.gen_topo = gen_topo
        self.name_atoms = name_atoms
        self.seed = seed

    def read(self) -> "Frame":
        """Parse, embed 3D, return a tabular :class:`~molrs.Frame`.

        Matches the :class:`~molpy.io.data.base.DataReader` contract used by
        XYZ / PDB / … readers. For the graph form see :meth:`read_as`.
        """
        return self.read_as("frame")

    @overload
    def read_as(self, kind: type["Frame"] | Literal["frame"]) -> "Frame": ...

    @overload
    def read_as(
        self, kind: type["Atomistic"] | Literal["atomistic"]
    ) -> "Atomistic": ...

    def read_as(self, kind: _AsKind = "frame") -> Any:
        """Read as a chosen result type.

        Parameters
        ----------
        kind
            ``Frame`` / ``"frame"`` (default) or ``Atomistic`` / ``"atomistic"``.
        """
        from molpy.core.atomistic import Atomistic
        from molrs import Frame

        mol = self._read_atomistic()
        target = self._resolve_kind(kind, Frame=Frame, Atomistic=Atomistic)
        if target is Atomistic:
            return mol
        if target is Frame:
            return mol.to_frame()
        raise TypeError(
            f"SmilesReader.read_as expects Frame or Atomistic; got {kind!r}"
        )

    @staticmethod
    def _resolve_kind(
        kind: _AsKind,
        *,
        Frame: type,
        Atomistic: type,
    ) -> type:
        if kind is Frame or kind is Atomistic:
            return kind  # type: ignore[return-value]
        if isinstance(kind, str):
            key = kind.strip().lower()
            if key in {"frame", "frames"}:
                return Frame
            if key in {"atomistic", "mol", "molecule", "graph"}:
                return Atomistic
        # Allow subclasses / aliases registered as type objects with matching name.
        name = getattr(kind, "__name__", "")
        if name == "Frame":
            return Frame
        if name == "Atomistic":
            return Atomistic
        raise TypeError(
            f"SmilesReader.read_as expects Frame or Atomistic; got {kind!r}"
        )

    def _read_atomistic(self) -> "Atomistic":
        """Parse with molrs, embed 3D with molrs Conformer, optionally name atoms."""
        import molrs

        from molpy.conformer import Conformer
        from molpy.core.atomistic import Atomistic

        mol = self._parse_graph(Atomistic, molrs)
        speed = "medium" if self.optimize else "fast"
        out, _report = Conformer(
            speed=speed,
            add_hydrogens=self.add_hydrogens,
            seed=self.seed,
        ).generate(mol)
        if self.gen_topo:
            out = out.get_topo(gen_angle=True, gen_dihe=True)
        if self.name_atoms:
            for idx, atom in enumerate(out.atoms, start=1):
                if atom.get("name") is None:
                    atom["name"] = f"{atom.get('element', 'X')}{idx}"
        return out

    def _parse_graph(self, Atomistic: type, molrs: object) -> "Atomistic":
        """Build a 2D graph Atomistic via molrs; never touches RDKit or Lark."""
        smiles = self.smiles
        if smiles.lstrip().startswith("{"):
            raise ValueError(
                "molpy does not parse BigSMILES / CGSmiles brace notation. Use "
                "plain SMILES with molrs.io.SmilesIR, or build polymer topology "
                "via molpy.builder.assembly (linear_topology / PolymerBuilder)."
            )

        ir = molrs.io.SmilesIR(smiles)
        n_comp = getattr(ir, "n_components", 1)
        if n_comp != 1:
            raise ValueError(
                "SmilesReader expects a single-component SMILES string; "
                f"got {n_comp} components ('.'-separated). "
                "Parse each component separately."
            )
        return Atomistic.adopt(ir.to_atomistic())


class SmilesWriter:
    """Write an :class:`~molpy.core.atomistic.Atomistic` graph to a SMILES string.

    Delegates to molrs::

        SmilesIR.from_atomistic(mol, **flags).write_smiles()

    All science/representation choices are **keyword-only flags** forwarded to
    molrs — this class does not invent a second printer.

    This is an **io** surface. It is deliberately *not* a method on
    :class:`~molpy.core.atomistic.Atomistic` (that would invert
    ``io → core`` into ``core → io``).

    Parameters
    ----------
    mol:
        Atomistic (or molrs Atomistic-compatible) graph.
    **flags:
        Forwarded to :meth:`molrs.io.SmilesIR.from_atomistic`: ``canonical``,
        ``root``, ``aromatic``, ``hydrogens``, ``include_stereo``,
        ``multi_component``, ``organic_subset``. See molrs docs for defaults.
    """

    def __init__(self, mol: "Atomistic", /, **flags: Any) -> None:
        self.mol = mol
        self.flags = flags

    def write(self) -> str:
        """Return the SMILES string for ``mol`` under the configured flags."""
        import molrs

        ir = molrs.io.SmilesIR.from_atomistic(self.mol, **self.flags)
        return ir.write_smiles()


def write_smarts(
    mol: "Atomistic",
    center: Any,
    /,
    **flags: Any,
) -> str:
    """Encode local topology around ``center`` as a SMARTS string.

    Prefer ``molrs.io.write_local_smarts`` when available; fall back to
    ``molrs.io.write_smarts``. ``center`` may be an
    :class:`~molpy.core.atomistic.Atom` view or an integer handle.

    Parameters
    ----------
    mol:
        Parent graph.
    center:
        Atom view or handle.
    **flags:
        Forwarded to molrs local-SMARTS options (``reach``, ``atomic_number``,
        ``include_degree``, …).
    """
    import molrs

    handle = int(getattr(center, "handle", center))
    write = getattr(molrs.io, "write_local_smarts", None) or molrs.io.write_smarts
    return write(mol, handle, **flags)


def write_local_smarts(
    mol: "Atomistic",
    center: Any,
    /,
    **flags: Any,
) -> str:
    """Alias of :func:`write_smarts`."""
    return write_smarts(mol, center, **flags)
