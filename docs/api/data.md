# Data

Locators for the data files bundled with MolPy — built-in force fields
(`oplsaa.xml`, `clp.xml`, `tip3p.xml`, `alpha.ff`, …) and other packaged
assets. These helpers return filesystem paths you can hand to a reader; they do
not parse anything themselves. Available via `import molpy as mp`
(`mp.data.get_forcefield_path`).

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `get_forcefield_path(name)` | Path to a bundled force-field file | Loading a built-in force field |
| `list_forcefields()` | Names of the bundled force fields | Discovering what ships with MolPy |
| `get_path(name)` | Path to any bundled data file | Accessing a packaged asset |
| `list_files()` | Names of all bundled data files | Enumerating packaged assets |
| `exists(name)` | Whether a bundled file is present | Guarding optional assets |

```python
import molpy as mp

ff = mp.io.read_xml_forcefield(mp.data.get_forcefield_path("oplsaa.xml"))
```

---

## Full API

::: molpy.data
