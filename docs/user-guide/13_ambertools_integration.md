# PEO-LiTFSI with AmberTools

Type TFSI with antechamber, build GAFF PEO chains from one antechamber-typed oligomer cut by prepgen and sequenced by tleap, and pack a PEO-LiTFSI electrolyte at target density — a complete AmberTools workflow driven from MolPy.

!!! warning "External dependencies"
    This guide requires **AmberTools** (via conda) and
    **molcrafts-molpack**. Only the inputs (TFSI, Li⁺, the PEO oligomer and
    its cuts) run without them; every block marked `# docs: skip` needs
    AmberTools or molpack.

??? note "Setting up AmberTools"
    Install AmberTools in a dedicated conda environment:

    ```bash
    conda create -n AmberTools25 -c conda-forge ambertools=25
    conda activate AmberTools25
    # Verify installation
    which antechamber # should print a path
    which tleap # should print a path
    ```

    MolPy's AmberTools typifiers and `AmberPolymerBuilder` activate the conda environment automatically when running commands, so you do not need to keep it active in your shell. The `env="AmberTools25"` parameter in the code below tells them which environment to activate.

    If you use a different environment name, replace `"AmberTools25"` throughout this guide.

## Workflow overview

`AntechamberTypifier` types a complete molecule: antechamber (GAFF types, BCC charges) → parmchk2 → tleap. A polymer is not that molecule. You build one oligomer in which the head, chain and tail monomers are already bonded, and you say how prepgen cuts it (`AmberCut`: the atoms each residue omits, and the atom across each junction). `AmberPolymerBuilder` then runs antechamber and parmchk2 on that oligomer, prepgen for each cut, and tleap `sequence`. `TLeapTypifier` is only for a finished molecule that already carries types and charges; a graph that still has ports is refused. Li⁺ gets a force field you define. The three force fields are merged, molpack places the molecules at the target density, and the system is exported to LAMMPS.

## Antechamber assigns GAFF2 types and BCC charges to TFSI

The net charge antechamber is given comes from the atoms' formal charges — here the `[N-]` of the SMILES. Antechamber needs 3D coordinates, so the anion is embedded first.

```python
from pathlib import Path
import molpy as mp

output_dir = Path("13_output")
output_dir.mkdir(exist_ok=True)

tfsi = mp.io.SmilesIR("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F").to_atomistic()
tfsi = mp.Conformer(add_hydrogens=False, seed=42).generate(tfsi)[0]
```

```python
# docs: skip — needs AmberTools
tfsi_ante = mp.ff.typifier.AntechamberTypifier(
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

li_ff = mp.ff.forcefield.ForceField("li", units="real")
li_type = li_ff.def_style("atom", "full").def_type("Li+", mass=6.94, charge=1.0)
# sigma = 2 * Rmin/2 / 2^(1/6)
li_ff.def_style("pair", "lj/cut").def_type("Li+", li_type, epsilon=0.0183, sigma=2.0259)
```

## The chain is a tleap sequence, cut by prepgen

`Assembler` folds the whole leaving-group charge onto the anchor, so an
ether oxygen of about −0.42 e becomes about −0.20 e. A GAFF chain is built
the way AMBER residues are made instead: antechamber types one oligomer in
which the head, chain and tail monomers are already bonded, so every atom
it types sees its chain neighbours. prepgen cuts that oligomer into a head,
a chain and a tail residue (`AmberCut`: the atoms each residue omits, and
the atom across each junction), spreading the omitted charge over the atoms
it keeps; the builder does not edit the prepi afterwards. tleap `sequence`
joins the residues.

`AmberPieces` writes the oligomer from three SMILES — head, repeat and tail,
each in backbone order — and the three cuts with it: each residue keeps its
own monomer and omits the other two.

```python
pieces = mp.builder.AmberPieces(head="COCC", repeat="OCC", tail="OCCOC")
oligomer, peo_cuts = pieces.oligomer(seed=42)  # CH3O(CH2CH2O)3CH3, embedded
print(oligomer.n_atoms)  # 30
print(peo_cuts["chain"].head, peo_cuts["chain"].tail)  # O2 C5
print(len(peo_cuts["chain"].omit))  # 23: the head and tail monomers
```

`{[#PEO]|10}` is ten residues: the head cut, eight chain cuts and the tail
cut, CH3–(OCH2CH2)10–OCH3. See [Polymer Topologies](topology/index.md) for
other architectures. The site graph is only that sequence; it must be one
linear path.

```python
sites = mp.io.CGSmilesIR("{[#PEO]|10}").to_coarsegrain()
```

```python
# docs: skip — needs AmberTools
built = mp.builder.AmberPolymerBuilder(
    {"PEO": oligomer},
    {"PEO": peo_cuts},
    force_field="gaff2",
    charge_method="bcc",
    work_dir=output_dir / "peo",
    env="AmberTools25",
    env_manager="conda",
).assemble(sites)
peo = built.chain  # typed graph: tleap's coordinates, types, charges and terms
peo_ff = built.forcefield  # units real, AMBER 1-4 scaling declared
print(f"PEO 10-mer: {peo.n_atoms} atoms")  # 79
```

antechamber and parmchk2 run on the oligomer, once, under
`work_dir/monomers/<label>/`. They are not run on the assembled chain.
Junction terms come from `leaprc.gaff2`. PEO and TFSI use the same
GAFF generation, so their shared types (`c3`, `os`, …) agree when the force
fields are merged. A later `assemble` reuses what is
in `work_dir`: antechamber reruns only when the oligomer or the charge
settings change, prepgen when a cut changes, and tleap when the sequence or
a file it loads changes.

### Cutting an oligomer you built yourself

For an oligomer you prepared elsewhere, write the cuts by atom name; they
are the prepgen control files. GroPoB's ethyl PEO oligomer `PEO.ac` keeps
every monomer between two dummy methyls: the chain residue omits both, the
head residue only the tail methyl (`C7`, `H15`, `H16`, `H17`), the tail
residue only the head methyl (`C3`, `H6`, `H7`, `H8`). `C1` and `C6` are the
connection atoms. `pre_head` / `post_tail` name the atom across a junction;
its GAFF type, read from antechamber's ac file, becomes `PRE_HEAD_TYPE` /
`POST_TAIL_TYPE`. `pre_head_type` / `post_tail_type` write a type directly.

```python
AmberCut = mp.builder.AmberCut
head_methyl = ("C3", "H6", "H7", "H8")
tail_methyl = ("C7", "H15", "H16", "H17")
gropob_cuts = {
    "head": AmberCut(tail="C6", post_tail="C7", omit=tail_methyl),
    "chain": AmberCut(
        head="C1",
        tail="C6",
        pre_head="C3",
        post_tail="C7",
        omit=head_methyl + tail_methyl,
    ),
    "tail": AmberCut(head="C1", pre_head="C3", omit=head_methyl),
}
```

An ac, mol2 and frcmod you put in `work_dir/monomers/<label>/` yourself (as
`<label>.ac`, `<label>.mol2` and `<label>.frcmod`) are used as they are:
only prepgen and tleap run.

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
mp.ff.forcefield.write_lammps_forcefield(
    lammps_dir / "system.ff", ff, system, skip_pair_style=True, skip_units=True
)
```

`skip_pair_style=True` omits the `pair_style` line from the force-field file. This is required when using kspace (long-range electrostatics), because the `pair_style` — and its cutoff — must be set by the simulation input script rather than the force-field file. The file keeps the force field's `special_bonds` (AMBER's 1-4 weights) and `pair_modify mix`, which need a pair style, so the input reads it after its `pair_style`. `skip_units=True` leaves out the `units` line: the input states `units` before `read_data`, and LAMMPS refuses a second one once the box exists.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Antechamber fails | The oligomer has 3D coordinates (embed it first) |
| `ValueError: … cut names [...]` | An atom name in a cut is not on the oligomer; `AmberPieces` names atoms element + count (`C1`, `O2`, …) |
| Chain O near −0.20 e, or typed `oh` | The chain was joined with `Assembler` (the leaving-group charge was folded onto the anchor). Build the oligomer with the monomers already bonded and cut it with prepgen |
| TFSI charge wrong | The formal charges sum to the net charge (`[N-]` gives −1); use `charge_method="bcc"` |
| tleap fails on the chain | A junction term missing from the leaprc (`leaprc.gaff2` here); the error carries tleap's output. The connection atoms are the ones in the oligomer you cut |
| `ValueError: … still has N ports` | `TLeapTypifier` was given a template. A polymer goes through `AmberPolymerBuilder` |
| Polymer assembly fails | The site graph is one path of at least two sites, and every bead type has an oligomer and a cut for the residue that site uses |
| `ValueError: … both make tleap residue …` | Residue names are the first characters of the bead type (`HPE`, `PEO`, `TPE` for `PEO`); give the bead types distinct prefixes |
| Force field merge conflict | Inspect type names for collisions between PEO, TFSI and Li⁺ |
| Packing fails | Increase box size or reduce molecule count |

The raw subprocess wrappers (`AntechamberWrapper`, `Parmchk2Wrapper`,
`TLeapWrapper`, …) stay in `molpy.wrapper` for scripts that drive the
executables directly.

See also: [Force Field Typification](06_typifier.md), [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md).
