# Data

Locators for the data files bundled with MolPy — built-in force fields
(`tip3p.xml`, an OpenMM XML; `clp.xml`, the CL&P typing force field read by
`mp.ff.typifier.OPLSAATypifier(path)`) and other packaged assets. The CL&Pol
Drude table ships with molrs (`mp.ff.params.clpol_polarizability()`). These helpers
return filesystem paths you can hand to a reader; they do not parse anything
themselves. Available via `import molpy as mp`
(`mp.data.get_path("forcefield/tip3p.xml")`).

OPLS-AA is not a bundled file: it ships with molrs and comes from
`mp.ff.typifier.OPLSAATypifier()` — `.library()` returns every OPLS-AA type,
`.forcefield()` only the types assigned by the last `.typify(...)`.

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `get_path(name)` | Path to a bundled data file (`"forcefield/tip3p.xml"`) | Loading a built-in force field or any packaged asset |
| `list_files(subdir)` | Names of the bundled data files (`list_files("forcefield")`) | Discovering what ships with MolPy |
| `exists(name)` | Whether a bundled file is present | Guarding optional assets |

```python
import molpy as mp

ff = mp.io.read_forcefield_xml(mp.data.get_path("forcefield/tip3p.xml"))

opls = mp.ff.typifier.OPLSAATypifier().library() # all OPLS-AA types, from molrs
```

---

## Full API

::: molpy.data
