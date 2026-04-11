# PEO-LiTFSI with AmberTools

This guide presents an AmberTools-based workflow for parameterizing ions, building PEO chains, and assembling a PEO-LiTFSI electrolyte system within MolPy.

!!! warning "External dependencies"
    This guide requires **AmberTools** (via conda), **RDKit**, and **Packmol**. All three must be installed and accessible. Without AmberTools, no code on this page will run.

??? note "Setting up AmberTools"
    Install AmberTools in a dedicated conda environment:

    ```bash
    conda create -n AmberTools25 -c conda-forge ambertools=25
    conda activate AmberTools25
    # Verify installation
    which antechamber   # should print a path
    which tleap         # should print a path
    ```

    MolPy's wrapper classes activate the conda environment automatically when running commands, so you do not need to keep it active in your shell. The `env="AmberTools25"` parameter in the code below tells the wrapper which environment to activate.

    If you use a different environment name, replace `"AmberTools25"` throughout this guide.

## Workflow overview

The workflow begins with parameterization of TFSI, the anion, using the standard Amber small-molecule sequence of antechamber, parmchk2, and tleap. Li⁺ is treated separately: its nonbonded parameters are taken from Åqvist (1990) and written to an frcmod file. With both ions parameterized, PEO chains are built using `AmberPolymerBuilder`, which wraps prepgen and tleap internally. The resulting component force fields are then merged, Packmol places the molecules at the target density, and the final system is exported to LAMMPS.


## Antechamber assigns GAFF types and BCC charges to TFSI

The Amber workflow for small molecules is: antechamber (assign types + charges) → parmchk2 (missing parameters) → tleap (topology + coordinates).

```python
from pathlib import Path
import molpy as mp
from molpy.adapter import RDKitAdapter
from molpy.tool import Generate3D
from molpy.io.writers import write_pdb
from molpy.wrapper import AntechamberWrapper, Parmchk2Wrapper, TLeapWrapper

output_dir = Path("07_output")
ions_dir = output_dir / "ions"
ions_dir.mkdir(parents=True, exist_ok=True)

# Create TFSI from SMILES and generate 3D coordinates
tfsi = mp.parser.parse_molecule("O=S(=O)(C(F)(F)F)[N-]S(=O)(=O)C(F)(F)F")
adapter = RDKitAdapter(internal=tfsi)
adapter = Generate3D(
    add_hydrogens=False,
    embed=True,
    optimize=True,
    update_internal=True,
)(adapter)
tfsi = adapter.get_internal()

# Write PDB for antechamber input
write_pdb(ions_dir / "tfsi.pdb", tfsi.to_frame())
```

```python
conda_env = "AmberTools25"

# Step 1: antechamber — assign GAFF types and BCC charges
ac = AntechamberWrapper(
    name="antechamber", workdir=ions_dir, env=conda_env, env_manager="conda"
)
ac.atomtype_assign(
    input_file=(ions_dir / "tfsi.pdb").absolute(),
    output_file=(ions_dir / "tfsi.mol2").absolute(),
    input_format="pdb",
    output_format="mol2",
    charge_method="bcc",
    atom_type="gaff2",
    net_charge=-1,
)

# Step 2: parmchk2 — generate missing parameters
parmchk2 = Parmchk2Wrapper(
    name="parmchk2", workdir=ions_dir, env=conda_env, env_manager="conda"
)
parmchk2.run(args=["-i", "tfsi.mol2", "-o", "tfsi.frcmod", "-f", "mol2", "-s", "gaff2"])

# Step 3: tleap — generate prmtop and inpcrd
leap_script = """source leaprc.gaff2
TFSI = loadmol2 tfsi.mol2
loadamberparams tfsi.frcmod
saveamberparm TFSI tfsi.prmtop tfsi.inpcrd
quit
"""
(ions_dir / "tfsi_leap.in").write_text(leap_script)

tleap = TLeapWrapper(name="tleap", workdir=ions_dir, env=conda_env, env_manager="conda")
tleap.run(args=["-f", "tfsi_leap.in"])
```


## Li⁺ needs no charge calculation — literature parameters go directly into an frcmod file

Li⁺ has no bonded terms and no partial charges to compute, so antechamber is not needed. Instead, write the nonbond parameters from Åqvist (1990) directly into an frcmod file and create the prmtop with tleap.

**Li⁺ nonbond parameters** — Åqvist (1990), J. Phys. Chem. 94, 8021–8024, DOI: 10.1021/j100384a009.
These were fitted to hydration free energies and are the standard choice for polymer electrolyte simulations with GAFF.

| Parameter | Value |
|-----------|-------|
| Rmin/2    | 1.137 Å |
| ε         | 0.0183 kcal/mol |

```python
from molpy.io import read_amber

li_dir = output_dir / "li"
li_dir.mkdir(parents=True, exist_ok=True)

# Write Åqvist (1990) frcmod — NONBON uses Rmin/2 and epsilon
li_frcmod = """Li+ Aqvist 1990 parameters
MASS
LI    6.941               0.0000000

BOND

ANGLE

DIHE

IMPROPER

NONBON
  LI        1.137        0.0183

"""
(li_dir / "li.frcmod").write_text(li_frcmod)

# Minimal mol2 for a single Li+ atom (net charge = +1)
li_mol2 = """@<TRIPOS>MOLECULE
LIT
 1 0 0 0 0
SMALL
USER_CHARGES

@<TRIPOS>ATOM
      1 LI          0.0000    0.0000    0.0000 LI    1  LIT      1.000000
@<TRIPOS>BOND
"""
(li_dir / "li.mol2").write_text(li_mol2)

# tleap: generate prmtop for Li+
li_leap = """source leaprc.gaff2
loadamberparams li.frcmod
LIT = loadmol2 li.mol2
saveamberparm LIT li.prmtop li.inpcrd
quit
"""
(li_dir / "li_leap.in").write_text(li_leap)

tleap_li = TLeapWrapper(
    name="tleap", workdir=li_dir, env=conda_env, env_manager="conda"
)
tleap_li.run(args=["-f", "li_leap.in"])

li_frame, li_ff = read_amber(li_dir / "li.prmtop", li_dir / "li.inpcrd")
print(
    f"Li+: {li_frame['atoms'].nrows} atom, charge={li_frame['atoms']['charge'][0]:.1f}"
)
```


## Three monomer variants define the chain start, interior, and end

Port markers in BigSMILES (`[>]` and `[<]`) define connection points. A monomer with both ports is an interior repeat unit; one with a single port is an end cap. The builder uses the port annotations to decide which prepgen variant (HEAD / CHAIN / TAIL) to generate.

```python
def parse_monomer_3d(bigsmiles):
    mol = mp.parser.parse_monomer(bigsmiles)
    adapter = RDKitAdapter(internal=mol)
    adapter = Generate3D(
        add_hydrogens=True,
        embed=True,
        optimize=True,
        update_internal=True,
    )(adapter)
    return adapter.get_internal()


# Head cap:  only < port → start of chain
me_head = parse_monomer_3d("{[][<]C[]}")

# Chain unit: both < and > → interior repeat
eo_chain = parse_monomer_3d("{[][<]COC[>][]}")

# Tail cap:  only > port → end of chain
me_tail = parse_monomer_3d("{[]C[>][]}")

library = {"MeH": me_head, "EO": eo_chain, "MeT": me_tail}
```


## AmberPolymerBuilder runs the full Amber pipeline internally

`AmberPolymerBuilder` wraps the monomer library, connector rules, and Amber tool chain (prepgen + tleap) into one builder that produces fully parameterized chains. Each unique chain length writes its Amber intermediate files into its own subdirectory under `work_dir` to prevent file conflicts.

```python
from molpy.builder.polymer.ambertools import AmberPolymerBuilder

polymer_dir = output_dir / "polymer"
polymer_dir.mkdir(exist_ok=True)

builder = AmberPolymerBuilder(
    library=library,
    force_field="gaff2",
    charge_method="bcc",
    env="AmberTools25",
    env_manager="conda",
    work_dir=polymer_dir,
)

result = builder.build("{[#MeH][#EO]|10[#MeT]}")
```

`AmberPolymerBuilder.build()` internally runs antechamber, parmchk2, prepgen, and tleap. The result carries the polymer Frame, ForceField, and paths to the intermediate Amber files.

```python
peo_frame = result.frame
peo_ff = result.forcefield
print(f"PEO 10-mer: {peo_frame['atoms'].nrows} atoms")
```


## Merging three force fields before packing prevents type conflicts

Merging is done before packing rather than after because Packmol operates on coordinates only — it has no awareness of force field types. If two components share an atom type name with different parameters, a post-packing merge would silently overwrite one of them. Merging first makes any type name collision an error before coordinates are generated.

```python
import numpy as np
from molpy.io import read_amber
from molpy.pack import Molpack, InsideBoxConstraint

# Read TFSI from Amber files generated in Stage 1
tfsi_frame, tfsi_ff = read_amber(
    ions_dir / "tfsi.prmtop",
    ions_dir / "tfsi.inpcrd",
)

# Merge all three force fields: PEO + TFSI + Li+
combined_ff = peo_ff.merge(tfsi_ff).merge(li_ff)

# Pack system
box_size = 60.0
packer = Molpack(workdir=output_dir / "packmol")
constraint = InsideBoxConstraint(length=[box_size] * 3, origin=[0.0] * 3)
packer.add_target(peo_frame, number=3, constraint=constraint)
packer.add_target(li_frame, number=10, constraint=constraint)
packer.add_target(tfsi_frame, number=10, constraint=constraint)

system = packer.optimize(max_steps=20000, seed=12345)
system.box = mp.Box.cubic(box_size)
```


## Exporting skips pair_style because long-range electrostatics need it in the script

```python
from molpy.io.writers import write_lammps_data, write_lammps_forcefield

lammps_dir = output_dir / "lammps"
lammps_dir.mkdir(exist_ok=True)
write_lammps_data(lammps_dir / "system.data", system, atom_style="full")
write_lammps_forcefield(lammps_dir / "system.ff", combined_ff, skip_pair_style=True)
```

`skip_pair_style=True` omits the `pair_style` line from the force-field file. This is required when using kspace (long-range electrostatics), because the `pair_style` must be set by the simulation input script rather than the force-field file.


## Troubleshooting

| Symptom | Check |
|---------|-------|
| Antechamber fails | Verify PDB has correct atom names and no duplicate IDs |
| TFSI charge wrong | Use `charge_method="bcc"` and verify `-nc -1` |
| tleap fails for Li⁺ | Confirm the mol2 atom type (`LI`) matches the frcmod NONBON entry |
| Polymer build fails | Check port markers in monomer SMILES |
| Force field merge conflict | Inspect atom type names for collisions between PEO and TFSI |
| Packing fails | Increase box size or reduce molecule count |

See also: [Force Field Typification](06_typifier.ipynb), [Wrapper and Adapter](../tutorials/07_wrapper_and_adapter.md).
