# PEO-LiTFSI with AmberTools

Type TFSI natively with AM1-BCC charges, build GAFF PEO chains from one antechamber-typed oligomer cut by prepgen and sequenced by tleap, merge the two force fields, and pack a PEO-LiTFSI electrolyte at target density for LAMMPS.

!!! warning "External dependencies"
    This guide requires **AmberTools** (via conda) and
    **molcrafts-molpack**: sqm for TFSI's AM1-BCC charges, and antechamber,
    parmchk2, prepgen and tleap for the chain. Only the inputs (TFSI's atom
    types, Li⁺, the PEO oligomer and its cuts) run without them; every block
    marked `# docs: skip` needs AmberTools or molpack.

??? note "Setting up AmberTools"
    Install AmberTools in a dedicated conda environment:

    ```bash
    conda create -n AmberTools25 -c conda-forge ambertools=25
    conda activate AmberTools25
    # Verify installation
    which antechamber # should print a path
    which tleap # should print a path
    ```

    `Am1BccModel`, `AmberPolymerBuilder` and MolPy's AmberTools typifiers activate the conda environment automatically when running commands, so you do not need to keep it active in your shell. Which environment they activate is a setting of molpy's configuration, shared by every AmberTools program (`[ambertools]`; see [Configure and Log a Run](14_configure_and_log.md)). Put it in `molpy.toml` next to your scripts:

    ```toml
    [ambertools]
    env = "AmberTools25"
    env_manager = "conda"
    ```

    If you use a different environment name, change it there; nothing in the code below names it.

## Workflow overview

A small molecule takes the native route, in a builder system as on its own:
`AtdTypifier` stamps antechamber's GAFF2 atom types, `mp.ff.charge.Am1BccModel`
assigns AM1-BCC charges with sqm as its one program, and `GaffTypifier`
assigns the bonded terms from `gaff2.dat` (see
[Force Field Typification](06_typifier.md)). Against AmberTools these are the
types and parameter values tleap gives the same molecule, and antechamber
`-c bcc`'s charges to within sqm's rounding (0.001 e). The PEO chain needs
the AmberTools pipeline: it is cut by prepgen and joined by tleap, and its
force field is read back from tleap's prmtop.

The two force fields merge. molrs's `ForceField.merge` compares what each
style prices (defaults filled in, within a relative 1e-6), and native GAFF
names its terms and rounds its estimates as tleap's prmtop does, so a type or
term the chain and TFSI share (`c3`) is one definition.
Every GAFF entry point takes `parameter_set` (`"gaff"` or `"gaff2"`, as
molrs names it); it has no default, so name the same set for every
component.

A polymer is not typed as one molecule. You build one oligomer in which the head, chain and tail monomers are already bonded, and you say how prepgen cuts it (`AmberCut`: the atoms each residue omits, and the atom across each junction). `AmberPolymerBuilder` then runs antechamber and parmchk2 on that oligomer, prepgen for each cut, and tleap `sequence`. Li⁺ gets a force field you define. The three force fields are merged, molpack places the molecules at the target density, and `LammpsEngine` writes the LAMMPS deck.

## TFSI is typed natively with AM1-BCC charges

Atom types need no program. sqm, which computes the AM1 charges, needs 3D
coordinates, so the anion is embedded first.

```python
from pathlib import Path
import molpy as mp
from molpy.ff.typifier import AtdTypifier, GaffTypifier

output_dir = Path("13_output")
output_dir.mkdir(exist_ok=True)

tfsi = mp.io.smiles.read_smiles("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F").to_atomistic()
tfsi = mp.conformer.Conformer(add_hydrogens=False, seed=42).generate(tfsi)[0]
tfsi = AtdTypifier(parameter_set="gaff2").typify(tfsi)  # GAFF2 atom types
print(sorted({atom["type"] for atom in tfsi.atoms}))  # ['c3', 'f', 'ne', 'o', 's6', 'sy']
```

`Am1BccModel` reads the AmberTools environment from `molpy.toml`
(`load_config()` resolves it; passing nothing does the same). The net charge
sqm is given comes from the atoms' formal charges — here the `[N-]` of the
SMILES:

```python
# docs: skip — needs AmberTools (sqm)
from molpy.config import load_config
from molpy.ff.charge import Am1BccModel

config = load_config()  # defaults, then ~/.molcrafts/molpy/config/config.toml, then ./molpy.toml
am1bcc = Am1BccModel(parameter_set="bcc", work_dir=output_dir / "tfsi", config=config)
charges = am1bcc.assign(tfsi)  # sqm's AM1 charges, then the native BCC correction
for atom, charge in zip(tfsi.atoms, charges):
    atom["charge"] = float(charge)
print(round(float(charges.sum()), 3))  # -0.998

tfsi_gaff = GaffTypifier(parameter_set="gaff2")
tfsi = tfsi_gaff.typify(tfsi)  # a typed copy: bonded terms from gaff2.dat
tfsi_ff = tfsi_gaff.forcefield()  # the parameters of the types just assigned
```

The anion carries −0.998 e, not −1: sqm prints its Mulliken charges to three
decimals, and neither the BCC correction nor antechamber renormalises
(antechamber `-c bcc` gives the same charges). Four of TFSI's terms
(`s6-ne-sy`, `f-c3-sy`, `o-s6-ne-sy`, `c3-s6-ne-sy`) are not in `gaff2.dat`;
`GaffTypifier` estimates them as parmchk2 does and marks them `estimated`.

## Li⁺ needs no charge calculation — you define its force field

Li⁺ has no bonded terms and no partial charges to compute, and GAFF has no lithium type (`GaffTypifier` refuses it: `gaff2.dat has no exact parameter for 1 term(s): MASS Li`). Build the atom with its type and charge, and define the force field it uses.

**Li⁺ nonbond parameters** — Åqvist (1990), J. Phys. Chem. 94, 8021–8024, DOI: 10.1021/j100384a009.
These were fitted to hydration free energies and are the standard choice for polymer electrolyte simulations with GAFF.

| Parameter | Value |
|-----------|-------|
| Rmin/2 | 1.137 Å |
| ε | 0.0183 kcal/mol |

```python
li = mp.Atomistic()
li.add_atom(atomic_number=3, type="Li+", charge=1.0, mass=6.94, x=0.0, y=0.0, z=0.0)

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
sites = mp.io.cgsmiles.read_cgsmiles("{[#PEO]|10}").to_coarsegrain()
```

```python
# docs: skip — needs AmberTools
built = mp.builder.AmberPolymerBuilder(
    {"PEO": oligomer},
    {"PEO": peo_cuts},
    parameter_set="gaff2",
    charge_method="bcc",
    work_dir=output_dir / "peo",
    config=config,
).assemble(sites)
peo = built.chain  # typed graph: tleap's coordinates, types, charges and terms
peo_ff = built.forcefield  # units real, AMBER 1-4 scaling declared
print(f"PEO 10-mer: {peo.n_atoms} atoms")  # 79
```

antechamber runs on the oligomer once (`-fo ac`: types and charges from one
sqm run), and parmchk2 and prepgen read its ac file, under
`work_dir/monomers/<label>/`. They are not run on the assembled chain.
Junction terms come from `leaprc.gaff2`. PEO and TFSI use the same
GAFF generation, so their shared type `c3` agrees when the force fields are
merged. A later `assemble` reuses what is
in `work_dir`: antechamber reruns only when the oligomer or the charge
settings change, prepgen when a cut changes, and tleap when the sequence or
a file it loads changes. A step that runs deletes its old outputs first, and
the typifiers rerun every step in a directory keyed by the molecule and every
setting, so a rerun never reads an earlier run's files.

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

An ac and frcmod you put in `work_dir/monomers/<label>/` yourself (as
`<label>.ac` and `<label>.frcmod`) are used as they are: only prepgen and
tleap run.

## Merging three force fields before packing prevents type conflicts

Merging is done before packing rather than after because packing operates on
coordinates only — it has no awareness of force field types. If two components
share a type name with different parameters, `merge` raises, so a collision is
an error before coordinates are generated. The chain's force field is tleap's
prmtop and TFSI's is native GAFF; they merge like any two force fields.
`merge` adds the other force field's styles and types to `peo_ff` in place and
returns it; where both define a style, `peo_ff`'s is kept.

```python
# docs: skip — needs AmberTools and molpack
from molpack import GencanPack, Target

ff = peo_ff.merge(tfsi_ff).merge(li_ff)  # peo_ff, now holding all three

box_size = 60.0
box = mp.core.Cuboid([0.0, 0.0, 0.0], [box_size] * 3)
targets = [
    Target(peo.to_frame(), count=3).with_restraint(box),
    Target(li.to_frame(), count=10).with_restraint(box),
    Target(tfsi.to_frame(), count=10).with_restraint(box),
]
system = GencanPack().with_seed(12345).run(targets, max_loops=200).frame
system.box = mp.Box(mp.core.Cell.lengths([box_size] * 3))
```

## The LAMMPS deck sets the cutoff and the long-range solver

Neither the prmtop nor GAFF states a cutoff: it is a run setting, so the deck
names it. `generate_inputs` writes the data file, the force-field settings
(coefficients, AMBER's 1-4 `special_bonds`, `pair_modify mix`), the init
(`units`, `atom_style`, `boundary`, `pair_style`) and the input script. The
`pair_style` given here replaces the force field's cut-off Coulomb with
`coul/long`, whose PPPM solver the body sets:

```python
# docs: skip — needs AmberTools and molpack
from molpy.engine import LammpsEngine

# full atom style needs mol_id: one per connected molecule
system["atoms"]["mol_id"] = mp.Topology.from_frame(system).connected_components() + 1
deck = LammpsEngine().generate_inputs(
    system,
    ff,
    output_dir / "lammps",
    pair_style="lj/cut/coul/long 12.0",
    body="kspace_style pppm 1.0e-4\nthermo 100\nminimize 1.0e-4 1.0e-6 1000 10000\n",
)
print(sorted(deck))  # ['data', 'init', 'input', 'settings']
```

`LammpsEngine().run(deck["input"], workdir=output_dir / "lammps")` runs it;
it is the starting point for the equilibration you add to `body`. The four
terms `GaffTypifier` estimated for TFSI (it marks them `estimated`, with the
penalty, method and analog) export like any other: every molrs writer skips
the provenance and writes the parameters.

## When the AmberTools typifiers are still the route

The native route covers GAFF / GAFF2 typing with AM1-BCC (or ABCG2) or
Gasteiger charges, merged with a chain or not. The AmberTools pipeline,
read back from tleap's prmtop, is kept for what it alone does:

- **A GAFF polymer chain**: `AmberPolymerBuilder` (prepgen cuts, tleap
  `sequence`), above.
- **Parameters of your own on top of the leaprc**: `TleapTypifier(forcefield=...)`
  writes them as a frcmod that tleap loads after `leaprc.<parameter_set>`,
  over a graph whose atoms already carry AMBER types and charges.
  `GaffTypifier` takes no extra parameters, and `merge` refuses a type it
  would redefine.
- **An antechamber charge method other than AM1-BCC / ABCG2**:
  `AntechamberTypifier(charge_method=...)` passes it to antechamber `-c`.

## Troubleshooting

| Symptom | Check |
|---------|-------|
| Antechamber fails | The oligomer has 3D coordinates (embed it first) |
| `ValueError: … cut names [...]` | An atom name in a cut is not on the oligomer; `AmberPieces` names atoms element + count (`C1`, `O2`, …) |
| Chain O near −0.20 e, or typed `oh` | The chain was joined with `Assembler` (the leaving-group charge was folded onto the anchor). Build the oligomer with the monomers already bonded and cut it with prepgen |
| TFSI charge wrong | The formal charges sum to the net charge (`[N-]` gives −1). −0.998 e is sqm's three-decimal rounding; antechamber `-c bcc` gives the same |
| tleap fails on the chain | A junction term missing from the leaprc (`leaprc.gaff2` here); the error carries tleap's output. The connection atoms are the ones in the oligomer you cut |
| `ValueError: … still has N ports` | `TleapTypifier` was given a template. A polymer goes through `AmberPolymerBuilder` |
| Polymer assembly fails | The site graph is one path of at least two sites, and every bead type has an oligomer and a cut for the residue that site uses |
| `ValueError: … both make tleap residue …` | Residue names are the first characters of the bead type (`HPE`, `PEO`, `TPE` for `PEO`); give the bead types distinct prefixes |
| Force field merge conflict | Every component names the same `parameter_set`; a type you define yourself (Li⁺) does not reuse a GAFF type name |
| Packing fails | Increase box size or reduce molecule count |

See also: [Force Field Typification](06_typifier.md), [Tool](../api/tool.md).
