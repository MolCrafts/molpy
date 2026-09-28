"""Read a SMILES string into a molecular graph.

Unlike the file readers in this package the *source* is the notation string
itself, so :class:`SmilesReader` does not extend :class:`DataReader`; it keeps
the same ``.read()`` idiom and — like every other data reader — defaults to a
:class:`~molpy.Frame`.

Parsing is native (``SmilesIR``) and open valences are filled
by :meth:`mp.Perceive.find_hydrogens <molpy.Perceive.find_hydrogens>`. A SMILES string
carries no coordinates, so none are invented here: 3D embedding is a
separate conformer step (:class:`molpy.conformer.Conformer`) the caller
composes. Do **not** route this path through RDKit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeVar, overload

if TYPE_CHECKING:
    from molpy.core.atomistic import Atomistic
    from molrs import Frame

T = TypeVar("T")
_AsKind = type | Literal["frame", "atomistic"] | str


class SmilesReader:
    """Turn a plain SMILES string into a structure.

    - Plain SMILES → ``SmilesIR`` → graph
    - Leading ``{`` → rejected; CGsmiles is parsed with
      ``mp.CGSmilesIR(text).to_coarsegrain()`` (bead graph) or
      ``mp.CGSmilesIR(text).to_atomistic()`` (all-atom graph)
    - ``add_hydrogens`` → :meth:`mp.Perceive.find_hydrogens <molpy.Perceive.find_hydrogens>`

    :meth:`read` returns a tabular :class:`~molpy.Frame` (same default as
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
        gen_topo: bool = False,
        name_atoms: bool = True,
    ) -> None:
        self.smiles = smiles
        self.add_hydrogens = add_hydrogens
        self.gen_topo = gen_topo
        self.name_atoms = name_atoms

    def read(self) -> "Frame":
        """Parse and return a tabular :class:`~molpy.Frame`.

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

    def read_as(self, kind: _AsKind = "frame") -> "Frame | Atomistic":
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
        raise TypeError(
            f"SmilesReader.read_as expects Frame or Atomistic; got {kind!r}"
        )

    def _read_atomistic(self) -> "Atomistic":
        """Parse natively, fill open valences, optionally name atoms."""
        from molrs.perceive import Perceive

        out = self._parse_graph()
        if self.add_hydrogens:
            out = Perceive().find_hydrogens(out)
        if self.gen_topo:
            out.generate_topology(gen_angle=True, gen_dihedral=True)
        if self.name_atoms:
            for idx, atom in enumerate(out.atoms, start=1):
                if atom.get("name") is None:
                    atom["name"] = f"{atom['element']}{idx}"
        return out

    def _parse_graph(self) -> "Atomistic":
        """Build a 2D graph Atomistic natively; never touches RDKit or Lark."""
        import molrs

        smiles = self.smiles
        if smiles.lstrip().startswith("{"):
            raise ValueError(
                "molpy's SMILES reader does not parse BigSMILES / CGsmiles "
                "brace notation. Parse CGsmiles with "
                "mp.CGSmilesIR(text).to_coarsegrain() (bead graph) or "
                "mp.CGSmilesIR(text).to_atomistic() (all-atom graph)."
            )

        ir = molrs.io.SmilesIR(smiles)
        n_comp = ir.n_components
        if n_comp != 1:
            raise ValueError(
                "SmilesReader expects a single-component SMILES string; "
                f"got {n_comp} components ('.'-separated). "
                "Parse each component separately."
            )
        return ir.to_atomistic()
