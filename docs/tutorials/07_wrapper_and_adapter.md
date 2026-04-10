[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/molcrafts/molpy/blob/master/docs/tutorials/07_wrapper_and_adapter.ipynb)

# Wrapper and Adapter

After reading this page you will understand the difference between execution boundaries and representation boundaries, and be able to use both in your workflows.

## Two kinds of external boundary

No molecular workflow stays inside one library forever. At some point you call an external program, convert to another object model, or both. These are fundamentally different operations, and MolPy names them differently.

**Wrappers cross execution boundaries. Adapters cross representation boundaries.**

A wrapper runs an external executable — `antechamber`, `tleap`, `lmp_serial` — and deals with subprocesses, return codes, working directories, and files on disk. An adapter translates between MolPy objects and another library's in-memory objects — RDKit molecules, OpenBabel structures — and deals with field mapping and synchronization.

Treating both as "calling another API" hides the part of the workflow that most often fails. MolPy keeps the distinction explicit so you know what kind of failure to expect and which layer to inspect.


## Wrapper: controlled execution across a process boundary

A `Wrapper` encapsulates a command-line tool. It handles locating the executable, setting up the environment, and running the command.

```python
from molpy.wrapper import Wrapper

echo = Wrapper(name="echo_tool", exe="echo")
result = echo.run(args=["Hello", "from", "MolPy!"])

if result.returncode == 0:
    print(result.stdout.strip())   # Hello from MolPy!
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
        self._external = ";".join(
            f"{k}={v}" for k, v in self._internal.items()
        )

    def _do_sync_to_internal(self):
        self._internal = dict(
            item.split("=") for item in self._external.split(";") if item
        )

adapter = StringDictAdapter(internal={"name": "MolPy", "role": "toolkit"})
adapter.sync_to_external()
print(adapter.get_external())   # name=MolPy;role=toolkit

adapter.set_external("name=MolPy;role=toolkit;version=0.2")
adapter.sync_to_internal()
print(adapter.get_internal())   # {'name': 'MolPy', 'role': 'toolkit', 'version': '0.2'}
```

The example is deliberately simple. The important point is not the data format — it is the synchronization protocol. No external process ran. No file was written. The concern is purely about keeping two representations of the same information consistent.


## Real-world adapter: geometry optimization with RDKit

A practical use case for adapters is leveraging external libraries for algorithms MolPy does not implement. Here, the `RDKitAdapter` bridges an `Atomistic` molecule to an RDKit `Mol` object, runs geometry optimization in RDKit, and brings the optimized coordinates back.

```python
from molpy import Atomistic
from molpy.adapter import RDKitAdapter
from rdkit.Chem import AllChem

mol = Atomistic()
c1 = mol.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
c2 = mol.def_atom(symbol="C", x=0.0, y=0.0, z=0.0)
o  = mol.def_atom(symbol="O", x=0.0, y=0.0, z=0.0)
mol.def_bond(c1, c2, order=1.0)
mol.def_bond(c2, o, order=2.0)

adapter = RDKitAdapter(internal=mol)
rd_mol = adapter.get_external()

AllChem.EmbedMolecule(rd_mol)
AllChem.MMFFOptimizeMolecule(rd_mol)

adapter.set_external(rd_mol)
adapter.sync_to_internal()

updated = adapter.get_internal()
atoms = list(updated.atoms)
print(f"C1: ({atoms[0]['x']:.2f}, {atoms[0]['y']:.2f}, {atoms[0]['z']:.2f})")
print(f"O:  ({atoms[2]['x']:.2f}, {atoms[2]['y']:.2f}, {atoms[2]['z']:.2f})")
```

!!! note
    This example requires RDKit to be installed. MolPy treats RDKit as an optional dependency — the adapter gracefully fails if RDKit is not available.


## Choosing the right boundary

Use a **wrapper** when your workflow must run another program. The concern is execution: did it succeed? What files did it produce?

Use an **adapter** when your workflow must translate between MolPy's objects and another library's objects. The concern is fidelity: do both sides still represent the same scientific structure?

Use both when the workflow genuinely spans both boundaries — for example, running Antechamber (wrapper) and then converting its output into MolPy objects (adapter).

See also: [Atomistic and Topology](01_atomistic_and_topology.md), [Force Field](04_force_field.md).
