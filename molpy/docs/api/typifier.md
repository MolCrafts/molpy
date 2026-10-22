# Typifier

Graph typification and force-field parameter assignment.

## The contract

A typifier is `Atomistic -> Atomistic`: it takes a molecular graph and returns a
typed copy whose atoms and links carry force-field types and parameters. Every
typifier runs the same flow — copy, assign, write the annotations back and define
the types — so `typify()` is written once, on the base class, and `assign()` is
the single step a typifier implements.

```text
class Typifier:
    def typify(self, mol) -> Atomistic                # final: copy, assign, write back
    def assign(self, graph) -> TypeAssignment         # the only thing that differs
    def source_forcefield(self) -> ForceField         # optional: what the output declares
    def forcefield(self) -> ForceField                # the union of every type typify assigned
```

`typify` is final — a subclass that defines it raises `TypeError` at class
creation — and it is the only writer of `forcefield()`.

**Typifiers are named after the force field or the tool that decides the types.**

## Quick reference

| Symbol | Summary | Preferred for |
|--------|---------|---------------|
| `Typifier` | The contract: implement `assign` (and optionally `source_forcefield`) | Writing your own |
| `TypeAssignment` | What `assign` returns: node and link annotations, styles, pair rows | Writing your own |
| `OplsAaTypifier` | Full OPLS-AA typing pipeline (native) | OPLS-AA all-atom force fields |
| `Mmff94Typifier` / `Mmff94sTypifier` | Full MMFF94 / MMFF94s typing pipeline (native) | MMFF all-atom force fields |
| `ElementTypifier` | `type` labels from element symbols; defines no force field | Writers that need labels on an untyped molecule |
| `AtdTypifier` | `AtdTypifier(parameter_set="gaff2")`: antechamber's atom-type tables (`gaff`, `gaff2`, `amber`, `bcc`, …) evaluated natively — atom types only, no charges or parameters | GAFF / GAFF2 or BCC atom types without running antechamber |
| `GaffTypifier` | `GaffTypifier(parameter_set="gaff2")`: GAFF / GAFF2 bonded terms and parameters, natively, for atoms `AtdTypifier` already typed (exact `gaff.dat` rows, wildcards, or `parmchk2`-style estimates marked `estimated`) | GAFF / GAFF2 parameters without AmberTools |
| `AntechamberTypifier` | `AntechamberTypifier(parameter_set="gaff2")`: antechamber → parmchk2 → tleap for one complete molecule; needs AmberTools | An antechamber charge method other than AM1-BCC / ABCG2; exporting an estimated GAFF term while molrs's LAMMPS / GROMACS writers refuse it |
| `TleapTypifier` | `TleapTypifier(parameter_set="gaff2")`: tleap alone over a finished graph that already carries AMBER types and charges (a graph with ports is refused); needs AmberTools | Extra frcmod parameters on top of the leaprc |

Every name is on `mp.ff.typifier`, which mirrors `molrs.ff.typifier` (the
native typifiers are the molrs objects) and adds the two AmberTools
typifiers. GAFF / GAFF2 is typed natively by `AtdTypifier` then
`GaffTypifier`; that is the default route, and it assigns the types and
parameter values tleap does for the same set. Its AM1-BCC charges come from
`mp.ff.charge.Am1BccModel`: sqm's AM1 charges (through antechamber `-c mul`),
corrected by the native `BccModel` exactly as antechamber `-c bcc` corrects
them. A native GAFF force field merges with one read from tleap's prmtop, an
`AmberPolymerBuilder` chain's included, so a small molecule mixed with a
chain is typed natively too. The AmberTools typifiers read tleap's prmtop
back; they are for what only that pipeline does: parameters of your own on
top of the leaprc (`TleapTypifier`) and antechamber's other charge methods
(`AntechamberTypifier`). molrs 0.18.0's LAMMPS and GROMACS writers refuse the
`estimated` parameter `GaffTypifier` puts on a term it estimated, so such a
molecule is exported through `AntechamberTypifier` for now. Every GAFF entry
point takes the same required `parameter_set` (`"gaff"` or `"gaff2"`, no
default). The AmberTools typifiers shell out through [`mp.tool`](tool.md); see
[Optional external tools](../getting-started/external-tools.md#ambertools-gaff-parameters).
A GAFF polymer is not typed by joining typed monomers (each join folds the
leaving group's charge onto its anchor): build it with
`mp.builder.AmberPolymerBuilder`, which cuts one antechamber-typed
oligomer with prepgen and joins the residues with tleap — see
[Builder](builder.md) and the
[AmberTools guide](../user-guide/13_ambertools_integration.md).

UFF typing is in the native core (Rust) only; it has no Python binding.

## Canonical example

```python
import molpy as mp
from molpy.ff.typifier import OplsAaTypifier

mol, _ = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.smiles.read_smiles("CCO").to_atomistic()
)

typifier = OplsAaTypifier(strict=True)
typed_mol = typifier.typify(mol)  # returns a new Atomistic
ff = typifier.forcefield()  # the parameters of the types just assigned
frame = typed_mol.to_frame()
```

## Key behavior

- `typify()` returns a **new** graph — the original is not modified
- `forcefield()` is the union of what every `typify` call assigned, as an
  independent copy; a definition that contradicts one already held raises
- A term the force field does not parameterise is left **undecided**, never
  stamped with `None`
- SMARTS matching is native; MolPy carries no matcher of its own

## Writing a typifier

Implement `assign` and stop. It returns a `TypeAssignment`: one annotation mapping per
node, positional against `graph.atoms`, and per link class one mapping per row,
positional against `graph.links.exact_bucket(cls)`. A type annotation is
`(style, name, endpoints, params)`: it stamps `name` and every param, and
defines the type `name` on `endpoints` (atom-type names, empty for an atom
type). The name is never parsed — the endpoints are required. `styles` declares
each style used; `pairs` adds `(style, name, endpoints, params)` pair rows.

```python
class Tip3pTypifier(mp.ff.typifier.Typifier):
    """TIP3P water: OW / HW atoms and one OW-HW bond type."""

    SITES = {"O": ("OW", 15.999, -0.834), "H": ("HW", 1.008, 0.417)}

    def source_forcefield(self):
        return mp.ff.forcefield.ForceField("tip3p", units="real")  # the output declares real units

    def assign(self, graph):
        nodes = []
        for atom in graph.atoms:
            name, mass, charge = self.SITES[atom["element"]]
            nodes.append({"type": ("full", name, (), {"mass": mass}), "charge": charge})
        bonds = [
            {"type": ("harmonic", "OW-HW", ["OW", "HW"], {"k": 450.0, "r0": 0.9572})}
            for _ in graph.links.exact_bucket(mp.Bond)
        ]
        return mp.ff.typifier.TypeAssignment(
            nodes,
            {mp.Bond: bonds},
            styles=[("atom", "full", {}), ("bond", "harmonic", {}), ("pair", "lj/cut", {})],
            pairs=[
                ("lj/cut", "OW", ["OW"], {"epsilon": 0.1521, "sigma": 3.1507}),
                ("lj/cut", "HW", ["HW"], {"epsilon": 0.0, "sigma": 0.0}),
            ],
        )


tip3p = Tip3pTypifier()
water = tip3p.typify(mp.io.smiles.read_smiles("[H]O[H]").to_atomistic())
assert [bond["type"] for bond in water.bonds] == ["OW-HW", "OW-HW"]
assert tip3p.forcefield().get_style("bond", "harmonic").get_type_by_name("OW-HW")["k"] == 450.0
```

## Related

- [Guide: Force Field Typification](../user-guide/06_typifier.md)
- [Guide: AmberTools Integration](../user-guide/13_ambertools_integration.md)
- [Concepts: Force Field](../tutorials/04_force_field.md)
- [Extending Typifiers](../developer/extending-typifiers.md)

---

## Full API

### Typifiers

::: molpy.ff.typifier
