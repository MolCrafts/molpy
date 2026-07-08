# Third-Party Attributions & Licenses

MolPy is distributed under the [BSD 3-Clause License](https://github.com/MolCrafts/molpy/blob/master/LICENSE)
(Copyright © 2024–2025, Roy Kid). It ports or adapts code and reference data from
the projects listed here. Numerical kernels executed through the **molrs** Rust
backend are attributed in
[molrs `docs/attribution.md`](https://github.com/MolCrafts/molrs/blob/master/docs/attribution.md);
this page covers what MolPy's own Python layer copies.

> External tools that MolPy merely invokes as a subprocess or imports as a
> library (Packmol, AmberTools, LAMMPS, CP2K, RDKit, OpenBabel, …) are runtime
> dependencies, not copied code, and are not listed here.

## Ported / adapted — permissive licenses

| Project | SPDX | Copyright | Used in | Upstream |
|---|---|---|---|---|
| **foyer** | `MIT` | © 2015 Vanderbilt University | historical OPLS-AA SMARTS typing reference; `data/forcefield/oplsaa.xml`, `tip3p.xml` | [mosdef-hub/foyer](https://github.com/mosdef-hub/foyer) |
| **OpenMM** | `MIT` (`openmm/app/element.py`) | Stanford University and the Authors | `core.element` — element name/symbol/mass table | [openmm/openmm](https://github.com/openmm/openmm) |
| **moltemplate** | `MIT` | © 2013 Andrew Jewett, UC Santa Barbara | `parser.moltemplate.*`, `cli.moltemplate` — `.lt` format reader/writer & `ltemplify` | [jewettaij/moltemplate](https://github.com/jewettaij/moltemplate) |
| **tame** | `BSD-3-Clause` | © Yunqi Shao | `compute.mcd`, `compute.pmsd`, `compute.time_series`, `compute.jacf`, `compute.onsager`, `compute.persist` | [yqshao-archive/tame](https://github.com/yqshao-archive/tame) (archived) |

> `tame` declares `license = "BSD-3-Clause"` in its `pyproject.toml` (it ships no
> standalone `LICENSE` file). It is an archived third-party project by Yunqi Shao.

## Bundled parameter data

| Data file | Origin | License |
|---|---|---|
| `data/forcefield/oplsaa.xml`, `tip3p.xml` | OPLS-AA / TIP3P force-field XML from **foyer** | `MIT` (foyer) |
| `data/forcefield/clp.xml` | CL&P ionic-liquid force field, subset from **paduagroup/clandp** (`il.ff`) | Academic; DOI 10.1021/jp0362133 |
| `data/forcefield/alpha.ff`, `clpol_fragments.ff` | **paduagroup/clandpol** CL&Pol polarizable data (A. Padua, K. Goloviznina) | Academic; DOI 10.1021/acs.jctc.9b00689 |

## Kernels via the molrs backend

Analyses such as RDF, Steinhardt/hexatic/nematic order, radical Voronoi,
structure factor, and PMFT run through molrs, whose backend ports **freud**
(`BSD-3-Clause`), **voro++** (`BSD-3-Clause-LBNL`), and **RDKit**
(`BSD-3-Clause`, MMFF). See the
[molrs attribution file](https://github.com/MolCrafts/molrs/blob/master/docs/attribution.md).

## Specifications implemented (cite, not license)

Grammars under `parser/grammar/` and `parser/smiles/grammars/` implement public
**notation specifications**, not another project's code:

| Specification | Cited as | Implemented in |
|---|---|---|
| **OpenSMILES** | opensmiles.org community specification | `smiles.lark`, `base.lark` |
| **Daylight SMILES/SMARTS** | Daylight Theory Manual (© Daylight C.I.S. — proprietary documentation, cited only) | `smarts.lark` |
| **BigSMILES** | Lin et al., *ACS Cent. Sci.* **5**, 1523 (2019), DOI 10.1021/acscentsci.9b00476 | `bigsmiles.lark` |
| **G-BigSMILES** | generative-BigSMILES extension | `gbigsmiles.lark`, `gbigsmiles_new.lark` |
| **CGSmiles** | Grünewald et al. (2025) | `cgsmiles.lark` |
