"""A dp=3 oligomer written as three SMILES, and the prepgen cuts of it.

The middle piece is the chain residue. tleap repeats that residue; this
module only builds the oligomer and says which atoms each residue keeps.
"""

from __future__ import annotations

from dataclasses import dataclass

from molrs.conformer import Conformer
from molrs.io.smiles import SmilesIr
from molrs.core import Atomistic

from ._cut import AmberCut


@dataclass(frozen=True)
class AmberPieces:
    """Head, repeat and tail, each a SMILES in backbone order.

    Concatenation is the dp=3 oligomer. The first atom of ``repeat`` bonds
    to the open atom at the end of ``head``, and ``tail`` bonds on in the
    same way. A branch such as ``(=O)`` stays inside its own piece.

    Example:
        >>> pieces = AmberPieces("COCC", "OCC", "OCCOC")
        >>> oligomer, cuts = pieces.oligomer()
    """

    head: str
    repeat: str
    tail: str

    def oligomer(self, *, seed: int = 42) -> tuple[Atomistic, dict[str, AmberCut]]:
        """Embed the trimer and return it with the head, chain and tail cuts."""
        smiles, head, chain, tail = _spans(self)
        mol = Conformer(seed=seed).generate(SmilesIr(smiles).to_atomistic())[0]
        atoms = list(mol.atoms)
        seen: dict[str, int] = {}
        for atom in atoms:
            element = str(atom.get("element"))
            seen[element] = seen.get(element, 0) + 1
            # antechamber's ac writer has a 3-character name column.
            name = f"{element}{seen[element]}"
            if len(name) > 3:
                raise RuntimeError(f"atom name {name} does not fit an ac file")
            atom["name"] = name
        neighbors = _neighbors(mol)
        head_tail, chain_head = _junction(neighbors, head, chain)
        chain_tail, tail_head = _junction(neighbors, chain, tail)
        groups = {
            "head": _with_hydrogens(atoms, neighbors, tuple(head)),
            "chain": _with_hydrogens(atoms, neighbors, tuple(chain)),
            "tail": _with_hydrogens(atoms, neighbors, tuple(tail)),
        }
        covered = groups["head"] | groups["chain"] | groups["tail"]
        if covered != set(range(len(atoms))) or sum(map(len, groups.values())) != len(
            atoms
        ):
            raise RuntimeError("trimer is not split into three monomers")

        def atom_name(index: int) -> str:
            return str(atoms[index]["name"])

        def omit(keep: set[int]) -> tuple[str, ...]:
            return tuple(atom_name(i) for i in range(len(atoms)) if i not in keep)

        cuts = {
            "head": AmberCut(
                tail=atom_name(head_tail),
                post_tail=atom_name(chain_head),
                omit=omit(groups["head"]),
            ),
            "chain": AmberCut(
                head=atom_name(chain_head),
                tail=atom_name(chain_tail),
                pre_head=atom_name(head_tail),
                post_tail=atom_name(tail_head),
                omit=omit(groups["chain"]),
            ),
            "tail": AmberCut(
                head=atom_name(tail_head),
                pre_head=atom_name(chain_tail),
                omit=omit(groups["tail"]),
            ),
        }
        return mol, cuts


def _heavy_count(smiles: str) -> int:
    mol = SmilesIr(smiles).to_atomistic()
    return sum(1 for atom in mol.atoms if atom.get("element") != "H")


def _spans(spec: AmberPieces) -> tuple[str, range, range, range]:
    n_head = _heavy_count(spec.head)
    n_repeat = _heavy_count(spec.repeat)
    n_tail = _heavy_count(spec.tail)
    head = range(0, n_head)
    chain = range(n_head, n_head + n_repeat)
    tail = range(n_head + n_repeat, n_head + n_repeat + n_tail)
    return spec.head + spec.repeat + spec.tail, head, chain, tail


def _neighbors(mol: Atomistic) -> list[list[int]]:
    atoms = list(mol.atoms)
    index = {atom.handle: i for i, atom in enumerate(atoms)}
    found: list[list[int]] = [[] for _ in atoms]
    for bond in mol.bonds:
        left, right = bond.endpoints
        i, j = index[left.handle], index[right.handle]
        found[i].append(j)
        found[j].append(i)
    return found


def _with_hydrogens(
    atoms: list, neighbors: list[list[int]], heavy: tuple[int, ...]
) -> set[int]:
    keep = set(heavy)
    for i, atom in enumerate(atoms):
        if atom.get("element") == "H" and any(n in keep for n in neighbors[i]):
            keep.add(i)
    return keep


def _junction(neighbors: list[list[int]], left: range, right: range) -> tuple[int, int]:
    """The one bond that crosses two pieces: (atom in left, atom in right)."""
    right_ids = set(right)
    hits = [(i, j) for i in left for j in neighbors[i] if j in right_ids]
    if len(hits) != 1:
        raise RuntimeError(f"expected one bond between pieces, found {len(hits)}")
    return hits[0]
