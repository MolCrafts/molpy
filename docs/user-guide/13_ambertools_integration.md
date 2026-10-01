# PEO-LiTFSI with AmberTools

Type TFSI and the PEO monomers with antechamber, assemble PEO chains from the typed monomers and finish them with tleap, and pack a PEO-LiTFSI electrolyte at target density — a complete AmberTools workflow driven from MolPy.

!!! warning "External dependencies"
    This guide requires **AmberTools** (via conda) and
    **molcrafts-molpack**. Only the inputs (TFSI, Li⁺ and the PEO monomers)
    run without them; every block marked `# docs: skip` needs AmberTools or
    molpack.

??? note "Setting up AmberTools"
    Install AmberTools in a dedicated conda environment:

    ```bash
    conda create -n AmberTools25 -c conda-forge ambertools=25
    conda activate AmberTools25
    # Verify installation
    which antechamber # should print a path
    which tleap # should print a path
    ```

    MolPy's AmberTools typifiers activate the conda environment automatically when running commands, so you do not need to keep it active in your shell. The `env="AmberTools25"` parameter in the code below tells them which environment to activate.

    If you use a different environment name, replace `"AmberTools25"` throughout this guide.

## Workflow overview

Two typifiers in `mp.typifier` drive AmberTools. `AntechamberTypifier` types a complete molecule from scratch: antechamber (GAFF2 types, BCC charges) → parmchk2 (missing parameters) → tleap. `TLeapTypifier` runs tleap alone on a graph whose atoms already carry AMBER types and charges. TFSI and each PEO monomer go through antechamber once; the chain is assembled from the typed monomers, so its types and charges travel with the templates, and tleap adds the junction terms. tleap never re-assigns types or charges, so each monomer is a complete molecule whose leaving groups mimic the neighbours its atoms have in the chain. Li⁺ gets a force field you define. The three force fields are merged, molpack places the molecules at the target density, and the system is exported to LAMMPS.

## Antechamber assigns GAFF2 types and BCC charges to TFSI

The net charge antechamber is given comes from the atoms' formal charges — here the `[N-]` of the SMILES. Antechamber needs 3D coordinates, so the anion is embedded first.

```python
from pathlib import Path
import molpy as mp

output_dir = Path("13_output")
output_dir.mkdir(exist_ok=True)

tfsi = mp.io.read_smiles("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F")
tfsi = mp.Conformer(add_hydrogens=False, seed=42).generate(tfsi)[0]
```

```python
# docs: skip — needs AmberTools
tfsi_ante = mp.typifier.AntechamberTypifier(
    atom_type="gaff2",
    charge_method="bcc",
    work_dir=output_dir / "tfsi",
    env="AmberTools25",
    env_manager="conda",
)
tfsi = tfsi_ante.typify(tfsi)  # a typed copy: GAFF2 types, BCC charges, bonded terms
tfsi_ff = tfsi_ante.forcefield()  # the parameters of the types just assigned
```

## Li⁺ needs no charge calculation — you define its force field

Li⁺ has no bonded terms and no partial charges to compute, so antechamber is not needed. Build the atom with its type and charge, and define the force field it uses.

**Li⁺ nonbond parameters** — Åqvist (1990), J. Phys. Chem. 94, 8021–8024, DOI: 10.1021/j100384a009.
These were fitted to hydration free energies and are the standard choice for polymer electrolyte simulations with GAFF.

| Parameter | Value |
|-----------|-------|
| Rmin/2 | 1.137 Å |
| ε | 0.0183 kcal/mol |

```python
li = mp.Atomistic()
li.def_atom(element="Li", type="Li+", charge=1.0, mass=6.94, x=0.0, y=0.0, z=0.0)

li_ff = mp.ForceField("li", units="real")
li_type = li_ff.def_style("atom", "full").def_type("Li+", mass=6.94, charge=1.0)
# sigma = 2 * Rmin/2 / 2^(1/6)
li_ff.def_style("pair", "lj/cut").def_type("Li+", li_type, epsilon=0.0183, sigma=2.0259)
```

## Each PEO monomer is typed once

AmberTools has no notion of ports or residue joins, so the chain is built by
MolPy; the topology `{[#CAPA][#EO]|10[#CAPB]}` is a methyl cap, ten
ethylene-oxide units and a methoxy cap. See
[Polymer Topologies](topology/index.md) for other architectures.

tleap joins units but never re-assigns atom types or charges: every atom of the
chain keeps the GAFF2 type and charge antechamber gave it in its monomer. So
each monomer must already put every atom that stays in the chain into its
in-chain environment — the leaving group of each port (the atoms removed on
linking) mimics the neighbour its anchor will have in the chain. An H-capped EO
unit, `[<]OCC[>]`, is ethanol to antechamber: its O is typed hydroxyl `oh` with
alcohol charges. The EO unit is instead CH3–O–CH2–CH2–O–CH3: the O-side port's
leaving group is the terminal CH3 and the C-side port's is the terminal O–CH3,
so the kept O is typed ether `os` and its charges are ether-like. The caps are
dimethyl ether on the same rule. This is how AMBER polymer residues are cut: the
junction atoms keep their types.

A CGsmiles bonding descriptor always becomes a capping hydrogen, so these units
are embedded from SMILES and their ports set with `def_port(anchor, leaving,
kind)`: the leaving handle may be any atom bonded to the anchor, and its whole
branch leaves. The conformer keeps the SMILES atom order and appends the
hydrogens, so the indices below are SMILES positions.

```python
conformer = mp.Conformer(seed=42)
# name: (complete molecule, ports as (anchor, leaving handle, kind) SMILES indices)
recipes = {
    "CAPA": ("COC", [(0, 1, ">")]),  # keeps CH3; O–CH3 leaves
    "EO": ("COCCOC", [(1, 0, "<"), (3, 4, ">")]),  # keeps O–CH2–CH2; CH3 and O–CH3 leave
    "CAPB": ("COC", [(1, 2, "<")]),  # keeps O–CH3; CH3 leaves
}
units = {}
for name, (smiles, ports) in recipes.items():
    unit = conformer.generate(mp.io.read_smiles(smiles))[0]
    atoms = list(unit.atoms)
    for anchor, leaving, kind in ports:
        unit.def_port(atoms[anchor], atoms[leaving], kind)
    units[name] = unit
print({name: unit.n_atoms for name, unit in units.items()})  # leaving groups included

sites = mp.CGSmilesIR("{[#CAPA][#EO]|10[#CAPB]}").to_coarsegrain()
draft = mp.Assembler(units, mp.GrowthPlacer()).assemble(sites, mp.Atomistic)
print(draft.n_atoms)  # 79: CH3-(OCH2CH2)10-OCH3, no leaving group left in the chain
```

Every unit is a complete molecule — each port's leaving group is made of real
atoms — so antechamber types it like any small molecule. The typed copy keeps
its ports.

```python
# docs: skip — needs AmberTools
ante = mp.typifier.AntechamberTypifier(
    atom_type="gaff2",
    charge_method="bcc",
    work_dir=output_dir / "units",
    env="AmberTools25",
    env_manager="conda",
)
lib = {name: ante.typify(unit) for name, unit in units.items()}
```

## tleap finishes the assembled chain

`mp.Assembler` joins one port of each neighbour per bond. Types and charges
travel with the templates; each join removes the two leaving groups and folds
their charge onto the anchors, so the chain's net charge is the sum of its
monomers'. The growth placer can leave overlaps, so the chain is re-embedded
before packing; the types and charges are kept.

`TLeapTypifier` writes the monomers' parameters as a frcmod, runs tleap only,
and takes the junction terms from `leaprc.gaff2`. It never runs antechamber or
parmchk2, and leaves the types and charges unchanged — which is why the
monomers' leaving groups had to mimic the chain.

```python
# docs: skip — needs AmberTools
chain = mp.Assembler(lib, mp.GrowthPlacer()).assemble(sites, mp.Atomistic)
chain = conformer.generate(chain)[0]  # re-embed: the growth placer can leave overlaps

leap = mp.typifier.TLeapTypifier(
    leaprc="gaff2",
    forcefield=ante.forcefield(),
    work_dir=output_dir / "polymer",
    env="AmberTools25",
    env_manager="conda",
)
peo = leap.typify(chain)
peo_ff = leap.forcefield()  # units real, AMBER 1-4 scaling declared
print(f"PEO 10-mer: {peo.n_atoms} atoms")  # CH3-(OCH2CH2)10-OCH3, 79 atoms
```

## Merging three force fields before packing prevents type conflicts

Merging is done before packing rather than after because packing operates on
coordinates only — it has no awareness of force field types. If two components
share a type name with different parameters, `merge` raises, so a collision is
an error before coordinates are generated.

```python
# docs: skip — needs AmberTools and molpack
from molpack import GenCanPack, Target

ff = peo_ff.merge(tfsi_ff).merge(li_ff)

box_size = 60.0
box = mp.Cuboid([0.0, 0.0, 0.0], [box_size] * 3)
targets = [
    Target(peo.to_frame(), count=3).with_restraint(box),
    Target(li.to_frame(), count=10).with_restraint(box),
    Target(tfsi.to_frame(), count=10).with_restraint(box),
]
system = GenCanPack().with_seed(12345).run(targets, max_loops=200).frame
system.box = mp.Box.cube(box_size)
```

## Exporting skips pair_style because long-range electrostatics need it in the script

```python
# docs: skip — needs AmberTools and molpack
lammps_dir = output_dir / "lammps"
lammps_dir.mkdir(exist_ok=True)

# full atom style needs mol_id: one per connected molecule
system["atoms"]["mol_id"] = mp.Topology.from_frame(system).connected_components() + 1
mp.io.write_lammps_data(lammps_dir / "system.data", system)
mp.io.write_lammps_forcefield(lammps_dir / "system.ff", ff, system, skip_pair_style=True)
```

`skip_pair_style=True` omits the `pair_style` and `special_bonds` lines from the force-field file. This is required when using kspace (long-range electrostatics), because the `pair_style` — and its cutoff — must be set by the simulation input script rather than the force-field file.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Antechamber fails | The molecule has 3D coordinates (embed it first) and every port's leaving group is present |
| Chain O typed `oh`, or alcohol-like charges | A port leaves a hydrogen where the chain has a heavy neighbour; make each leaving group mimic that neighbour (CH3 for a backbone C, O–CH3 for a backbone O) |
| TFSI charge wrong | The formal charges sum to the net charge (`[N-]` gives −1); use `charge_method="bcc"` |
| `ValueError: formal charges sum to …` | A charged atom lacks its `formal_charge`, or the sum is not an integer |
| tleap fails on the chain | A junction term missing from `leaprc.gaff2`; the error carries tleap's stderr |
| `ValueError: tleap changed atom …` | The chain's types or charges differ from what tleap read back; retype the monomers |
| Polymer assembly fails | Check each unit's ports (`<` joins `>`, each leaving handle bonded to its anchor) and that every topology name is in the library |
| Force field merge conflict | Inspect type names for collisions between PEO, TFSI and Li⁺ |
| Packing fails | Increase box size or reduce molecule count |

The raw subprocess wrappers (`AntechamberWrapper`, `Parmchk2Wrapper`,
`TLeapWrapper`, …) stay in `molpy.wrapper` for scripts that drive the
executables directly.

See also: [Force Field Typification](06_typifier.md), [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md).
