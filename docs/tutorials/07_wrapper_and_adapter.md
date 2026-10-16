# Wrapper and Adapter

You need `antechamber` for charges and RDKit for a fingerprint. Are those the
same kind of “integration”?

No. One starts a **process**; the other maps **in-memory objects**. Conflating
them hides the failures that cost the most time.

**Wrappers cross execution boundaries** (subprocess, working directory, return
codes). **Adapters cross representation boundaries** (field mapping between
MolPy and RDKit, OpenBabel, …).

What they are **not**: each other — and neither replaces I/O readers that only
parse files into `Frame`s.

## Wrapper: controlled execution across a process boundary

A `Wrapper` locates an executable, sets the environment, and runs the command.
The examples below share this setup:

```python
import molpy as mp

mol = mp.io.smiles.SmilesIr("CCO").to_atomistic()
```

```python
from molpy.wrapper import AntechamberWrapper

ante = AntechamberWrapper("ante_work")
print(ante.executable)  # antechamber
print(ante.environment)  # EnvironmentSpec(system)
print(ante.is_available())  # True only where AmberTools is installed
```

A wrapper takes its working directory and nothing else: which executable it
runs, in which environment, with which variables and time limit, are its
settings in molpy's configuration (`[wrapper.antechamber]`, falling back to
`[wrapper]`). For tools installed in an isolated environment, name it there
and the wrapper runs every command through Conda or the virtualenv:

```python
# docs: skip — needs AmberTools in a conda env named AmberTools25
from molpy.config import load_config

config = load_config({"wrapper": {"env": "AmberTools25", "env_manager": "conda"}})
ante = AntechamberWrapper("ante_work", config=config)
ante.run(["-i", "input.pdb", "-fi", "pdb", "-o", "out.mol2", "-fo", "mol2"])
# runs: conda run --no-capture-output -n AmberTools25 antechamber -i input.pdb ...
```

A run that fails raises `molpy.wrapper.ToolError`, carrying the tool
(`"wrapper.antechamber"`), its exit code and the tail of its output; pass
`check=False` to get the failed `CompletedProcess` back instead.

Each run is logged to `molpy.wrapper.antechamber` as structured
[mollog](https://docs.molcrafts.org/mollog/) records; see
[Configure and Log a Run](../user-guide/14_configure_and_log.md).


## Adapter: conversion across two object models

An adapter is a pair of functions, `to_<lib>` and `from_<lib>`. Each builds a
new object from the other side's object and leaves its input untouched: there
is no cached pair to fall out of sync, and nothing is written onto your
structure. MolPy keeps one: `to_rdkit` / `from_rdkit`, between an `Atomistic`
and an RDKit `Mol`. RDKit is an optional extra — molpy never requires it, and
`molpy.adapter.to_rdkit` raises a clear `ImportError` when it is missing.

An adapter earns its keep when you need an algorithm MolPy does not implement:
convert, let RDKit work on its own object, convert back.

```python
# docs: skip — RDKit optional adapter example; not unit-tested
import molpy as mp
from molpy.adapter import from_rdkit, to_rdkit
from rdkit import Chem
from rdkit.Chem import AllChem

mol = mp.io.smiles.SmilesIr("CCO").to_atomistic()

rd_mol = Chem.AddHs(to_rdkit(mol))
AllChem.EmbedMolecule(rd_mol)
AllChem.MMFFOptimizeMolecule(rd_mol)

updated = from_rdkit(rd_mol)  # a new Atomistic; `mol` is unchanged
```

The concern is fidelity, so know what crosses. RDKit owns element, formal
charge, bond type and coordinates, and its values win on the way back. Every
other atom and bond component — names, residues, partial charges, force-field
types — rides along as an RDKit property and returns on the atoms and bonds
RDKit kept. What does not survive: atoms RDKit removes (they are gone from the
result), angles/dihedrals/impropers (only bonds cross; regenerate them), and
stereo, which is not translated between molpy's `stereo` component and RDKit's
chiral tags. The full list is in the [adapter API](../api/adapter.md).

**But not for this one.** 3D embedding is native, and the native path is the
supported one — no third-party install, and it returns a report of what each
stage of the pipeline did:

```python
mol_3d, report = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(mol)
```

That is the line to remember about adapters: use one to reach *their*
algorithms, not to repeat *ours*. Wrappers are for genuine external programs
(AmberTools / antechamber). Packing is native
[molpack](https://docs.molcrafts.org/molpack/), not a subprocess wrapper.

## Choosing the right boundary

Use a **wrapper** when your workflow must run another program. The concern is execution: did it succeed? What files did it produce?

Use an **adapter** when your workflow must translate between MolPy's objects and another library's objects. The concern is fidelity: do both sides still represent the same scientific structure?

Use both when the workflow genuinely spans both boundaries — for example, running Antechamber (wrapper) and then converting its output into MolPy objects (adapter).

See also: [Atomistic and Topology](01_atomistic_and_topology.md), [Force Field](04_force_field.md).
