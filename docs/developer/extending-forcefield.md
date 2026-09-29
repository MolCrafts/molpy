# Extending the Force Field

This page shows how to add custom interaction styles and their export support.

!!! note "Discuss before you build"
    New interaction styles are a molrs change (the Rust kernel plus the writer arms); molpy re-exports the result. Open a [GitHub issue](https://github.com/MolCrafts/molpy/issues) describing the functional form before implementation; the [Architecture Overview](architecture-overview.md) explains where the pieces live.

## Where the math lives

The force-field model — `ForceField`, the `Style` tree, the `Type` tree, and all
energy/force kernels — is owned by the **molrs** Rust extension. molpy does not
maintain a parallel Python potential layer. There is no `style.to_potential()`,
no `ff.to_potentials()`, and no Python kernel class; evaluation always goes
through `PotentialCompiler`:

```python
import molpy as mp

mol, _ = mp.Conformer(seed=42).generate(mp.io.read_smiles("CCO"))
typifier = mp.typifier.OPLSAATypifier()
frame = typifier.typify(mol).to_frame()
ff = typifier.forcefield()  # the OPLS-AA types just assigned

mp.PotentialCompiler(ff).compile(frame).calc_energy(frame)  # and .calc_forces(frame)
```

This changes what "extending the force field" means:

1. **Kernel** — the numerical form (energy + forces) is implemented in molrs
   (Rust) and registered there under its style name, so `PotentialCompiler`
   can dispatch on it.
2. **Style name** — on the Python side there is nothing to subclass: the style
   name *is* the handle, `ff.def_style("bond", "morse")`.
3. **Writer arms** — each export backend (LAMMPS, GROMACS, XML) is a molrs
   writer; serializing a new style's parameters is a match arm in that writer,
   not a Python formatter.

If molrs already ships the kernel you need, there is nothing to build: define
the style by name and use it. Adding a brand-new functional form is a molrs
change.


## Step 1: add the kernel in molrs

A new functional form is implemented in molrs under `molrs/src/ff/potential/`
(the Morse bond, `bond/morse.rs`, is a complete example): write the energy and
force expressions, then register the constructor under its style name — a
built-in goes into `KernelRegistry::builtin()` in `potential/registry.rs`
(`r.register("bond", "morse", bond::morse::bond_morse_ctor)`), and
`register_kernel` adds or overrides an entry at runtime. Rebuild the molrs
wheel (`maturin develop` / `maturin build`) and reinstall it; molpy picks up
the new kernel automatically because it re-exports the molrs hierarchy.


## Step 2: use the style by name

Once registered, the style name is usable directly with the generic builder.
A type is given its name — the label a typed `Frame` uses — and then its
endpoints, the atom types it connects:

```python
ff = mp.ForceField(name="custom", units="real")
atom_style = ff.def_style("atom", "full")
c = atom_style.def_type("C", mass=12.011)
o = atom_style.def_type("O", mass=15.999)
ff.def_style("bond", "morse").def_type(  # dispatches to the molrs kernel
    "C-O", c, o, D=100.0, alpha=1.8, r0=1.43
)
```

Types and parameters flow through molrs natively — there is no named `Style`
subclass and no `def_type()` override to write.


## Step 3: add the writer arms in molrs

The force-field writers are molrs functions that molpy re-exports
(`mp.io.write_lammps_forcefield`, `mp.io.write_gromacs_forcefield`,
`mp.io.write_xml_forcefield`). Each maps a `(category, style)` pair to the file's
columns in one place: `coeff_fields` in `ff/forcefield/writers/lammps.rs` and
`bonded_columns` in `ff/forcefield/writers/gromacs.rs`. Add an arm for the new
style to each backend that should emit it; a style without an arm is refused
with a `ValueError` naming it, never written half-formed. The Morse bond, for
example, has a GROMACS arm (`bondtypes` function 3) and no LAMMPS arm yet.


## Using the custom interaction

Build the model, then evaluate it against a typed `Frame`:

```python
# Two atoms exactly at r0 → Morse energy is 0.
frame = mp.Frame(
    blocks={
        "atoms": {"x": [0.0, 1.43], "y": [0.0, 0.0], "z": [0.0, 0.0], "type": ["C", "O"]},
        "bonds": {"atomi": [0], "atomj": [1], "type": ["C-O"]},
    }
)

pots = mp.PotentialCompiler(ff).compile(frame)
print(pots.calc_energy(frame))  # 0.0 at r0
```


## Checklist

- [ ] Kernel implemented and registered in molrs (`ff/potential/registry.rs`), wheel rebuilt
- [ ] Validate the kernel: energy at equilibrium = 0, monotonic increase away from it
- [ ] Writer arms in molrs for each backend that should emit the style (LAMMPS, GROMACS, XML)
- [ ] Tests in molrs: kernel energy/force values, writer output and refusal
- [ ] molpy tests: `PotentialCompiler(ff).compile(frame).calc_energy(frame)` through the re-exported API
