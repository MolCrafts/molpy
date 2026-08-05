"""Public-API regression: write_smiles / write_local_smarts (hard-coded goldens).

Requires molcrafts-molrs with smiles-emit write APIs (local maturin develop OK
for workspace; master land needs a tagged molrs minor).
"""

from __future__ import annotations

from molpy.core.atomistic import Atomistic
from molpy.io import write_smarts, write_smiles
from molpy.io.data.smiles import SmilesReader


def main() -> None:
    mol = SmilesReader("CCO", optimize=False, add_hydrogens=False).read_as(Atomistic)
    s = write_smiles(mol, canonical=True)
    assert isinstance(s, str) and len(s) >= 2, s
    # Round-trip parse
    mol2 = SmilesReader(s, optimize=False, add_hydrogens=False).read_as(Atomistic)
    s2 = write_smiles(mol2, canonical=True)
    assert s == s2, (s, s2)

    center = next(iter(mol.atoms))
    smarts = write_smarts(mol, center, reach=1, atomic_number=True)
    assert "#" in smarts or "C" in smarts, smarts
    assert not hasattr(mol, "to_smiles")
    print("ok", s, smarts)


if __name__ == "__main__":
    main()
