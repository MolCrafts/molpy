# Getting Started

This section establishes a working MolPy environment and introduces the explicit `Atomistic → Frame → export` workflow that underlies most system-preparation tasks. The pages below are intended to be read in sequence by new users; readers already familiar with MolPy's data model may proceed directly to any subsection.

1. **[Installation](installation.md)** — dependency requirements, package installation, and environment verification
2. **[Quickstart](quickstart.md)** — construction of a solvated system, force field assignment, and LAMMPS input generation
3. **[Example Gallery](examples.md)** — copy-paste-ready workflows, from a single small molecule to packed boxes and polymer networks
4. **[FAQ](faq.md)** — troubleshooting, comparison with related software, and answers to frequently asked questions

Looking for reference material that used to live here? Canonical field names and the glossary are now appendices of [Concepts](../tutorials/index.md); the MCP suite guide lives under [Guides → Tools & Ecosystem](../user-guide/15_mcp.md).


## Preliminary Verification

Before proceeding to the full quickstart, the following minimal example confirms that MolPy is correctly installed. No optional dependencies — including RDKit — are required.

```python
import molpy as mp

water = mp.Atomistic(name="water")
o  = water.def_atom(element="O", x=0.000, y=0.000, z=0.000)
h1 = water.def_atom(element="H", x=0.957, y=0.000, z=0.000)
h2 = water.def_atom(element="H", x=-0.239, y=0.927, z=0.000)
water.def_bond(o, h1)
water.def_bond(o, h2)

frame = water.to_frame()
mp.io.write_pdb("water.pdb", frame)
print(f"Wrote {frame['atoms'].nrows} atoms to water.pdb")
```

Successful execution prints `Wrote 3 atoms to water.pdb`.


## Bundled Data Files

MolPy distributes commonly required force field files as package data. File paths such as `"oplsaa.xml"` are resolved relative to MolPy's internal data directory; no external download is necessary.

```python
ff = mp.io.read_xml_forcefield("oplsaa.xml")  # resolves from package data
```

The complete inventory of bundled files may be enumerated programmatically:

```python
from molpy.data import list_files
print(list(list_files("forcefield")))   # e.g. ['oplsaa.xml', 'tip3p.xml']
```


## The Standard Workflow

The majority of MolPy workflows follow a common five-stage pipeline:

```text
SMILES / file               Atomistic                   Frame
  input       ──parser──>   (editable graph)  ────────> (columnar arrays)
                                  │                           │
                            typifier + ff               io.write_*
                                  │                           │
                            Typed Atomistic           LAMMPS / GROMACS
                                                      simulation files
```

1. **Parse or construct** — produce an `Atomistic` structure from a SMILES string, an existing file, or explicit atom and bond definitions
2. **Edit** — modify connectivity, run reactions, or assemble larger structures from building blocks
3. **Typify** — assign force field atom types through SMARTS-based pattern matching
4. **Convert** — invoke `atomistic.to_frame()` to produce columnar NumPy arrays suitable for numerical operations
5. **Export** — write to LAMMPS, GROMACS, PDB, or other simulation formats


Upon completing these pages, a reader should be able to answer the following questions:

- Under what circumstances should a molecular system be edited as an `Atomistic` graph rather than as a `Frame`?
- Why is topology (angles, dihedrals) derived from bond connectivity rather than stored independently?
- At which stage does force field typification occur, and what does it produce?

Readers for whom these questions remain unclear should consult [Concepts](../tutorials/index.md) for a rigorous treatment of the data model. Those who prefer to proceed directly to a concrete modeling task should refer to [Guides](../user-guide/index.md).
