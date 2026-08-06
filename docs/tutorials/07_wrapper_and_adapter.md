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

mol = mp.io.read_smiles("CCO")
```

```python
from molpy.wrapper import Wrapper

echo = Wrapper(name="echo_tool", exe="echo")
result = echo.run(args=["Hello", "from", "MolPy!"])

if result.returncode == 0:
    print(result.stdout.strip())  # Hello from MolPy!
else:
    print(result.stderr)
```

The example uses `echo` because it requires no installation, but the real use cases are tools like Antechamber and tleap. The wrapper pattern is the same: create the wrapper with the executable name, run it with arguments, check the result.

For tools installed in isolated environments, wrappers handle Conda or virtualenv activation automatically.

```python
# Example (not runnable without AmberTools installed):
# ac = Wrapper(
#     name="antechamber",
#     exe="antechamber",
#     env="AmberTools22",
#     env_manager="conda",
# )
# ac.run(args=["-i", "input.pdb", "-fi", "pdb", "-o", "out.mol2", "-fo", "mol2"])
```


## Adapter: synchronized state across two object models

An `Adapter` holds an internal MolPy object and an external object, and keeps them synchronized. The protocol has two directions: `sync_to_external()` translates MolPy → external, and `sync_to_internal()` translates external → MolPy.

Here is a minimal adapter that converts a dictionary to a semicolon-separated string and back.

```python
from molpy.adapter import Adapter


class StringDictAdapter(Adapter[dict[str, str], str]):
    def _do_sync_to_external(self):
        self._external = ";".join(f"{k}={v}" for k, v in self._internal.items())

    def _do_sync_to_internal(self):
        self._internal = dict(
            item.split("=") for item in self._external.split(";") if item
        )


adapter = StringDictAdapter(internal={"name": "MolPy", "role": "toolkit"})
adapter.sync_to_external()
print(adapter.get_external())  # name=MolPy;role=toolkit

adapter.set_external("name=MolPy;role=toolkit;version=0.2")
adapter.sync_to_internal()
print(adapter.get_internal())  # {'name': 'MolPy', 'role': 'toolkit', 'version': '0.2'}
```

The example is deliberately simple. The important point is not the data format — it is the synchronization protocol. No external process ran. No file was written. The concern is purely about keeping two representations of the same information consistent.


## Real-world adapter: reaching RDKit's own algorithms

An adapter earns its keep when you need an algorithm MolPy does not implement.
`RDKitAdapter` bridges an `Atomistic` to an RDKit `Mol`, lets RDKit work on its
own object, and brings the result back. RDKit is an optional extra — molpy
never requires it.

```python
# docs: skip — RDKit optional adapter example; not unit-tested
import molpy as mp
from molpy.adapter import RDKitAdapter
from rdkit.Chem import AllChem

mol = mp.io.read_smiles("CCO")

adapter = RDKitAdapter(internal=mol)
rd_mol = adapter.get_external()

AllChem.EmbedMolecule(rd_mol)
AllChem.MMFFOptimizeMolecule(rd_mol)

adapter.set_external(rd_mol)
adapter.sync_to_internal()
updated = adapter.get_internal()
```

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
