# molcrafts-molpy

The Python binding of [molpy](https://github.com/MolCrafts/molrs): a toolkit
for the full molecular-system workflow — parsing, building, editing, typing,
analyzing, and reading/writing simulation formats — on a Rust core. The wheel
carries the native core as `molpy._core`.

```bash
pip install molcrafts-molpy
```

```python
import molpy as mp

mol = mp.io.smiles.read_smiles("CCO").to_atomistic()
```

Documentation: <https://docs.molcrafts.org/molpy/>
