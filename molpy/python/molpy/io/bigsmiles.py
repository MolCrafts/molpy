"""BigSMILES and generative BigSMILES — ``molrs::io::bigsmiles``.

BigSMILES writes a polymer as SMILES in which a *stochastic object*
``{[] … ; … []}`` stands for a chain of repeat units of unspecified length;
generative BigSMILES adds what a generator needs — weights inside bonding
descriptors (``[<|0.5|]``), a size distribution after an object
(``|schulz_zimm(1500, 1400)|``) and a system size after a molecule
(``.|5e5|``, or ``.|20%|`` for a share of a mixture).

:func:`read_bigsmiles` reads a string into a :class:`BigSmilesIr`, whose
``to_molecules()`` and ``to_system()`` give the polymer model of
:mod:`molpy.builder` (``Molecule``, ``System``). Sizes and amounts count
heavy-atom mass in g/mol, as the notation writes them. A refusal is a
:class:`molpy.io.smiles.SmilesError` with ``notation == "bigsmiles"``.
"""

from molpy._core import BigSmilesIr, read_bigsmiles

__all__ = ["BigSmilesIr", "read_bigsmiles"]
