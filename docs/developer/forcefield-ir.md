# Force-field IR

The Rust core holds every force field in one intermediate representation, the
**force-field IR**: one set of styles, each with a fixed energy expression,
parameter meanings and parameter units. The IR adopts LAMMPS's definitions as
its standard — for every style the core registers, the expression, the factors
(no hidden ½), the parameter meanings and the units are those of the LAMMPS
style of the same name, or of the LAMMPS style it corresponds to. Reading a
LAMMPS force field and writing it back is therefore the identity on
coefficients, and every other engine — GROMACS, OpenMM, AMBER, the GAFF /
OPLS-AA / MMFF / UFF tables — converts to and from the IR at its reader,
writer or typifier, never in a kernel.

The IR is data — `mp.ff.forcefield.ForceField` (Rust
`molrs::ff::forcefield::ForceField`) — and reads no file. Every engine's
reader and writer is `mp.io`'s, as every other file reader and writer is:
one class per format and direction, its settings the constructor's keywords
(Rust: one constructor and `with_*` setters), `reader.read(source) ->
ForceField` and `writer.write(sink, forcefield)`, plus what else the format
reads or writes as methods of the same class —
`mp.io.lammps.LammpsForcefieldReader` / `LammpsForcefieldWriter`
(`read_data_coeffs`, `read_cmap`; `write_data_coeffs`, `write_cmap`),
`mp.io.gromacs.GromacsTopForcefieldReader` / `GromacsTopForcefieldWriter`
(`read_system`, `write_system`), `mp.io.amber.AmberPrmtopForcefieldReader`
(`read_system`) / `AmberFrcmodWriter`,
`mp.io.openmm_xml.OpenmmXmlReader` / `OpenmmXmlWriter`, and
`mp.io.xml.XmlForcefieldReader` / `XmlForcefieldWriter`. The same classes in
Rust (`molrs::io::lammps`, `molrs::io::gromacs`, `molrs::io::amber`,
`molrs::io::openmm_xml`, `molrs::io::XmlForcefieldReader`) read through
`ForceFieldReader::read(src)` and write through `ForceFieldWriter::write(sink, &ff)`,
a source a path or a whole file's bytes and a sink a path or a `Vec<u8>`.

This page is the reference for the IR: what each style computes, what its
parameters mean, which engine form maps onto it and how, and how Urey–Bradley,
explicit 1-4 pairs and CMAP are represented. The IR is also a protocol: a
new style or category, from Rust or Python, registers into it with
nothing rebuilt — see [Extending the force-field IR](extending-forcefield-ir.md).

## Units

A `ForceField` declares a LAMMPS unit preset (`ForceField.units`: `real`,
`metal`, `lj`; `None` when undeclared, and whatever reads its numbers — a unit
conversion, a record — refuses undeclared units rather than assume one). Every parameter is a number in that
preset. Two rules are LAMMPS's own and hold in every preset:

!!! important "Angles are degrees"
    Every **angle-valued parameter** — an equilibrium angle (`theta0`,
    `chi0`), a dihedral or improper phase (`phase`, `phase<m>`,
    `phi1..phi3`) — is stored in **degrees**, exactly as a LAMMPS
    `*_coeff` line writes it. A **force constant** is per **radian**ⁿ
    (`angle harmonic` `k` is energy/rad²), as in LAMMPS. The kernels convert
    the angle to radians once, when they are built.

    This makes "the force-field IR stores exactly what LAMMPS writes" literally true for the
    stored values: `angle_coeff c3-c3-oh 76.79 109.66` is the type
    `{k: 76.79, theta0: 109.66}`, with no conversion in either direction.

The `forcefield` section of a record states `"angle": "degree"` beside its
preset. A section built in memory with `"angle": "radian"` beside a preset is refused
(its preset and its stated angle unit disagree) rather than read in the wrong
units.

The LAMMPS reader keeps the file's `units` (a `metal` file is a `metal` force
field, `coul/cut` taking LAMMPS's `metal` Coulomb constant); the LAMMPS writer
converts energies and lengths only when the target `units` differs from the
force field's. Every other reader produces `real`.

## Style reference

`k`, `r0`, `theta0`, … are the parameter names. They are LAMMPS's symbols,
lower-cased, with one exception: LAMMPS spells two different things `d` — a
phase (`dihedral charmm`, `fourier`) and a sign (`dihedral harmonic`,
`improper cvff`) — and its multiplicity `n`, so the IR names those slots
`phase`, `sign` and `periodicity`. The values and their order are LAMMPS's.

A parameter with a default may be left out: the compile fills the spec's
default (`StyleSpec::gather`, the one place a default is applied) before any
kernel, built-in or expression, sees the row, so leaving it out and stating
the default price alike. A parameter a kernel needs that has no default is
`MissingParam`; a value of the wrong kind or out of its domain (text for a
number, a `mixing` outside its choices) is `BadValue`.

### Bonds

| Style | Energy | Parameters (units) | LAMMPS |
|---|---|---|---|
| `harmonic` | k (r − r0)² | `k` (E/L²), `r0` (L) | `bond_style harmonic` `K r0` |
| `morse` | d0 [1 − e^(−alpha (r − r0))]² | `d0` (E), `alpha` (1/L), `r0` (L) | `bond_style morse` `D0 alpha r0` |
| `class2` | k2 Δ² + k3 Δ³ + k4 Δ⁴, Δ = r − r0 | `r0`, `k2`, `k3`, `k4` | `bond_style class2` `r0 K2 K3 K4` |
| `mmff_bond` | MMFF94 cubic-quartic stretch | per-instance `kb`, `r0` | none |
| `uff_bond` | ½ kb (r − r0)² (RDKit) | per-instance `kb`, `r0` | none |

### Angles

| Style | Energy | Parameters (units) | LAMMPS |
|---|---|---|---|
| `harmonic` | k (θ − theta0)² | `k` (E/rad²), `theta0` (deg) | `angle_style harmonic` `K theta0` |
| `charmm` | k (θ − theta0)² + k_ub (r₁₃ − r_ub)² | `k` (E/rad²), `theta0` (deg), `k_ub` (E/L²), `r_ub` (L) | `angle_style charmm` `K theta0 K_ub r_ub` |
| `class2` | k2 Δ² + k3 Δ³ + k4 Δ⁴, Δ = θ − theta0 | `theta0` (deg), `k2`, `k3`, `k4` (E/radⁿ) | `angle_style class2` `theta0 K2 K3 K4`, its `bb` / `ba` lines written at zero (a non-zero cross term is refused: the IR has none) |
| `mmff_angle`, `mmff_stbn` | MMFF94 bend / stretch-bend | per-instance `ka`, `theta0` (deg), `kba_*` | none |
| `uff_angle` | UFF Fourier / order-n bend (RDKit) | per-instance `ka`, `order`, `c0..c2` (`theta0` kept as metadata, deg) | none |

### Dihedrals

| Style | Energy | Parameters (units) | LAMMPS |
|---|---|---|---|
| `periodic` | Σₘ kₘ [1 + cos(nₘ φ − γₘ)] | `k<m>` (E), `periodicity<m>`, `phase<m>` (deg); or one term `k`, `periodicity`, `phase` | `dihedral_style fourier` `m K1 n1 d1 …` |
| `charmm` | k [1 + cos(n φ − d)], plus `w`·(its end atoms' 1-4 pair) | `k` (E), `periodicity`, `phase` (deg), `w` | `dihedral_style charmm` `K n d w` |
| `opls` | ½[k1(1 + cos φ) + k2(1 − cos 2φ) + k3(1 + cos 3φ) + k4(1 − cos 4φ)] | `k1..k4` (E) | `dihedral_style opls` |
| `multi/harmonic` | Σₙ₌₁⁵ aₙ cosⁿ⁻¹ φ | `a1..a5` (E) | `dihedral_style multi/harmonic` |
| `nharmonic` | Σᵢ₌₁ᴺ aᵢ cosⁱ⁻¹ φ | `a1..aN` (E), contiguous, N ≥ 1 | `dihedral_style nharmonic` `N A1 … AN` |
| `harmonic` | k [1 + sign cos(n φ)] | `k` (E), `sign` (±1), `periodicity` | `dihedral_style harmonic` `K d n` |
| `class2` | Σₙ₌₁³ kₙ [1 − cos(n φ − phiₙ)] | `k1, phi1, k2, phi2, k3, phi3` (E, deg) | `dihedral_style class2` `K1 phi1 K2 phi2 K3 phi3`, its `mbt`/`ebt`/`at`/`aat`/`bb13` lines written at zero (a non-zero one is refused) |
| `mmff_torsion` | ½[V1(1 + cos φ) + V2(1 − cos 2φ) + V3(1 + cos 3φ)] | per-instance `v1..v3` | none |
| `uff_torsion` | V/2 [1 − cosTerm cos(n φ)] (RDKit) | per-instance `V`, `order`, `cosTerm` | none |

### Impropers

| Style | Energy | Parameters (units) | Atom order | LAMMPS |
|---|---|---|---|---|
| `harmonic` | k (χ − chi0)², χ = \|φ(I,J,K,L)\| | `k` (E/rad²), `chi0` (deg) | I is the centre | `improper_style harmonic` `K chi0` |
| `cvff` | k [1 + sign cos(n φ(I,J,K,L))] | `k` (E), `sign` (±1), `periodicity` | I is the centre | `improper_style cvff` `K d n` |
| `periodic` | k [1 + cos(n φ(I,J,K,L) − γ)] | `k` (E), `periodicity`, `phase` (deg) | AMBER's: K is the centre | `improper_style cvff` (one term, γ ∈ {0°, 180°}: `d = cos γ`) |
| `mmff_oop` | ½·143.9325 koop χ², χ the Wilson angle of bond I→L to plane (I,J,K) | per-instance `koop` | I is the centre | none (geometry of `improper_style umbrella` / `fourier`) |
| `uff_inversion` | K [c0 + c1 cos ω + c2 cos 2ω], ω of bond I→L to plane (I,J,K) | per-instance `K`, `c0..c2` | I is the centre | `improper_style fourier` |

### Pair styles

| Style | Energy | Parameters (units) | LAMMPS |
|---|---|---|---|
| `lj/cut` | C ε [(σ/r)ⁿ − (σ/r)ᵐ], C = n/(n−m)·(n/m)^(m/(n−m)); 4ε[(σ/r)¹² − (σ/r)⁶] at n = 12, m = 6 | `epsilon` (E), `sigma` (L); style `cutoff`, `mixing`, `n`, `m`, `shift` | `pair_style lj/cut` (`shift`: `pair_modify shift yes`; `mixing`: `pair_modify mix`; n ≠ 12 or m ≠ 6, LAMMPS's `mie/cut`, is neither read nor written) |
| `lj/class2` | ε [2(σ/r)⁹ − 3(σ/r)⁶] | `epsilon`, `sigma`; style `cutoff`, `mixing` (default `sixthpower`, LAMMPS's) | `pair_style lj/class2` (LAMMPS mixes it `sixthpower` whatever `pair_modify` says: another `mixing` is refused, and it reads as `sixthpower`) |
| `buck` | a e^(−r/rho) − c/r⁶ | `a` (E), `rho` (L), `c` (E·L⁶) | `pair_style buck` `A rho C` |
| `morse` | d0 [(1 − e^(−alpha (r − r0)))² − 1] | `d0` (E), `alpha` (1/L), `r0` (L) | `pair_style morse` `D0 alpha r0` |
| `coul/cut` | coulomb qᵢqⱼ / (dielectric (r + delta)) | style `coulomb` (E·L/e²), `dielectric` (default 1), `delta` (L, default 0), `cutoff` (default ∞) | `pair_style coul/cut` with `delta = 0` (the buffer is the core's, for MMFF; the LAMMPS writer refuses `delta ≠ 0` and `dielectric ≠ 1`). LAMMPS fixes the constant (`qqr2e`) per `units` |
| `lj/charmm` | 4ε[(σ/r)¹² − (σ/r)⁶]·S(r), S CHARMM's switch from `inner` to `cutoff` | `epsilon`, `sigma`, `epsilon14`, `sigma14` (absent → `epsilon`, `sigma`); style `inner`, `cutoff`, `mixing` (default `arithmetic`), `one_four` (`"regular"`, the default, or `"epsilon14"`: what a `special_bonds` 1-4 pair is priced at, see [1-4](#1-4-interactions)) | `pair_style lj/charmm/coul/charmm`, van-der-Waals half; `pair_coeff i j ε σ ε₁₄ σ₁₄` (`one_four = "epsilon14"` has no LAMMPS form) |
| `coul/charmm` | coulomb qᵢqⱼ/(dielectric r)·S(r); force (C qᵢqⱼ/r²)·S(r), LAMMPS's switched force, not the gradient | style `coulomb`, `dielectric` (default 1), `inner`, `cutoff` | `pair_style lj/charmm/coul/charmm`, Coulomb half (`inner2 outer2` when its cutoffs differ) |
| `coul/long/pme` | Ewald-summed coulomb qᵢqⱼ/r over the frame's periodic box | style `coulomb`, `cutoff`, `alpha`, `order`, `grid_*`; the cell is the frame's box (`frame.box`), as LAMMPS's kspace reads its simulation box | `pair_style lj/cut/coul/long` (the real-space half; `kspace_style` states an accuracy, not `alpha`, so the Ewald parameters are neither read nor written, and a LAMMPS-read style is refused (`MissingParam`) until they are stated) |
| `thole` | T(r) qᵢqⱼ/r, T = 1 − (1 + s r/2) e^(−s r), s = ½(aᵢ + aⱼ)/(αᵢαⱼ)^(1/6) | per type `charge`, `alpha` (L³), `damp` | `pair_style thole` `alpha damp` (LAMMPS damps the Drude charges of the atoms; the core's per-type `charge` is its own) |
| `coul/tt` | fₙ(r) qᵢqⱼ/r (Tang–Toennies) | per type `charge`; style `b`, `c`, `order` | `pair_style coul/tt` (`n` = `order`) |
| `uff_lj`, `mmff_vdw` | UFF x/D LJ; MMFF buffered 14-7 | per-instance / per-type | none |

`special_bonds` (the force field's `[1-2, 1-3, 1-4]` weights for van der
Waals and Coulomb) is LAMMPS's `special_bonds lj … coul …`.

**Cutoffs.** Every pair style prices a pair only at `r < cutoff`, its own
`cutoff`, as LAMMPS's pair styles do (`rsq < cutsq`) — and the CHARMM
styles switch between `inner` and `cutoff` — at **both** compile doors:
`PotentialCompiler.compile` prices the rows of the frame's `pairs` list that
are inside the cutoff, exactly as `compile_typed` prices the pairs a
neighbour search finds. A `special_bonds` 1-4 pair is a pair-style pair and
is truncated too (LAMMPS applies `special_lj` / `special_coul` inside the
`rsq < cutsq` branch). This holds for built-in, run-time (Tier 1 and 2,
Python callables included) and expression styles alike. A style that
states no `cutoff` has cutoff ∞: the declared default of `lj/cut`,
`lj/class2`, `buck`, `morse` and `coul/cut`, and every style whose spec
declares none (`thole`, `coul/tt`, `uff_lj`, `mmff_vdw`), so a field read
from an engine that priced it untruncated (OpenMM `NoCutoff`, a prmtop)
prices every listed pair. A neighbour-driven evaluation (MD) still needs a
finite `cutoff` and refuses ∞ (`BadValue`). The 1-4 exceptions kernel — a
`dihedral charmm` `w`, per-pair overrides ([1-4](#1-4-interactions)) — is
no pair style and is never truncated, as `dihedral_charmm.cpp` prices its
1-4 pair at any distance.

**The periodic box is frame data.** `coul/long/pme` reads its cell from the
frame being compiled (`frame.box`), never from a style parameter. A frame
without a box, with a box not periodic in x, y and z, or with a cell
outside LAMMPS's restricted triclinic form (`a` along x, `b` in the xy
plane) is refused by name (Rust `CompileError::NoBox`; Python
`ValueError`). Its spec declares exactly the parameters its kernel reads.
A `pairs` block prices the real-space erfc as the pair form (a 1-4 row at
`coulomb14scale`, an exclusion row omitted); with no `pairs` block the
compiled door keeps the minimum-image loop inside the kernel. The
neighbour-driven door always prices that erfc over the neighbour table.
The reciprocal sum still reads the box and takes no pair weight, so a 1-4
scale changes only the real-space term.

## Improper atom order

Every LAMMPS improper style that is a dihedral (`harmonic`, `cvff`) prices
the dihedral **I-J-K-L of the atoms in the order the data file lists them**,
and so does every kernel of such a style. LAMMPS names the first atom
the centre ("atom of symmetry") of `harmonic` and `cvff`; CHARMM writes its
impropers that way. The out-of-plane styles (`fourier`, `umbrella`; the core's
`uff_inversion`, `mmff_oop`) take the centre first too.

AMBER prices its improper over the dihedral with the centre **third**. No
order with the centre first has that dihedral — an AMBER improper's axis runs
through its centre — so an AMBER improper cannot be re-ordered centre-first
and keep its energy. The IR therefore stores `improper periodic` (the AMBER /
GAFF / OpenMM improper) in **AMBER's order**: the order whose dihedral is the
improper angle, which is also the order a LAMMPS data file lists it in for
`cvff` to reproduce AMBER (verified with LAMMPS: the AMBER improper of the
hand molecule below gives `0.00338479168892827` kcal/mol as `cvff` in AMBER
order; the core gives `0.003384791688934619`).

| Source | File order | Stored order |
|---|---|---|
| LAMMPS data / `improper_coeff` | I, J, K, L | as written |
| GROMACS funct 4 (periodic), funct 2 (harmonic) | i, j, k, l (GROMACS prices φ(i,j,k,l); pdb2gmx lists a CHARMM improper centre first, an AMBER one centre third) | as written |
| AMBER prmtop, frcmod, GAFF typifier | i, j, K, l (centre third) | as written |
| chamber prmtop `CHARMM_IMPROPERS` (`improper harmonic`) | I (centre), J, K, L | as written |
| OpenMM `<Improper class1 … class4>`, `ordering` default / `amber` | c1 = centre; OpenMM prices φ(c2, c3, c1, c4) | (c2, c3, c1, c4); the writer writes the inverse |
| OpenMM, `ordering="charmm"`, no wildcard | OpenMM prices φ(c1, c2, c3, c4) | as written |
| OpenMM, `ordering="smirnoff"` | three permutations averaged | refused |
| OpenMM `<CustomTorsionForce>` harmonic improper (default ordering `charmm`) | OpenMM prices φ(c1, c2, c3, c4) without a wildcard, φ(c2, c3, c1, c4) with one | as OpenMM prices; the writer refuses a wildcard |
| OpenMM `<RBTorsionForce><Improper>` | — | refused (no improper style is a cosine polynomial) |
| topology perception (`generate_topology`, `Topology.impropers`) | centre first | a force field that wants AMBER's order re-orders (GAFF does) |
| UFF, MMFF typifiers | — | centre first |

A frame's `impropers` row lists its atoms in the order of its type's
endpoints. Where OpenMM picks the order of two peripherals of equal type by
element and index, a caller that builds frame rows from an OpenMM-read field
decides the same way.

The OpenMM reader re-orders each improper row so the kernel prices
φ(c2, c3, c1, c4), the dihedral OpenMM prices. The OpenMM writer refuses
`cvff` (OpenMM cannot price a dihedral that starts at the centre).

## 1-4 interactions

LAMMPS has three mechanisms, and the IR represents each with LAMMPS's
parameters:

1. **Global weights** — `special_bonds lj w12 w13 w14 coul w12 w13 w14`:
   `ForceField.special_bonds`. AMBER is `lj 0 0 ½ coul 0 0 5/6`, OPLS-AA
   `0 0 ½`, CHARMM `0 0 0` (its 1-4 pairs are priced by the dihedral, below).
   The pair styles price a 1-4 pair at their own parameters (and switch),
   times the weight.
2. **Per-type 1-4 Lennard-Jones** — `pair_style lj/charmm/coul/charmm inner
   outer [inner2 outer2]`, `pair_coeff I J epsilon sigma [epsilon14
   sigma14]`. In the IR, a pair style `lj/charmm` (van der Waals half, as
   `lj/cut` is of `lj/cut/coul/cut`) with columns `epsilon`, `sigma`,
   `epsilon14`, `sigma14` and style params `inner`, `cutoff`, `mixing`
   (LAMMPS's default for these styles is `arithmetic`; `epsilon14` /
   `sigma14` mix like `epsilon` / `sigma`, as LAMMPS's `init_one` mixes them);
   self rows per type, explicit cross rows per type pair. Its Coulomb half is
   `coul/charmm` (`coulomb`, `dielectric`, `inner`, `cutoff`). Both switch
   with CHARMM's
   S(r) = (r_c² − r²)²(r_c² + 2r² − 3r_in²)/(r_c² − r_in²)³ between `inner`
   and `cutoff`, at both compile doors; the Lennard-Jones force is the
   gradient, the Coulomb force is LAMMPS's switched force C qᵢqⱼ S/r², which
   is not (`pair_lj_charmm_coul_charmm.cpp`). `lj/charmm/coul/long` is not
   read. GROMACS `[ pairtypes ]` land in `epsilon14`/`sigma14` (self rows
   and cross rows, see [GROMACS topologies](#gromacs-topologies)).

   LAMMPS prices a `special_bonds` 1-4 pair under this style at the
   **regular** `epsilon` / `sigma`; `epsilon14` / `sigma14` reach only the
   `w` pairs of mechanism 3. Engines that price every bond-graph 1-4 pair
   once with its own 1-4 parameters (OpenMM's `<LennardJonesForce>`
   `sigma14`/`epsilon14`, GROMACS `[ pairtypes ]`, a chamber prmtop's 1-4
   table) mean it at `epsilon14` / `sigma14`. The style param `one_four`
   says which:

   | `one_four` | A `special_bonds` 1-4 pair (no `w`, no override) is priced at |
   |---|---|
   | absent or `"regular"` | the regular `epsilon` / `sigma` — LAMMPS |
   | `"epsilon14"` | `epsilon14` / `sigma14` (the cross row's for an explicit pair, else the two types' mixed by `mixing`) |

   Any other value is refused (by the compile doors, `ForceFieldSection.from_forcefield` /
   `to_forcefield` and every reader of a record). Under `"epsilon14"` both
   compile doors price each 1-4 special pair (a `pairs` row of `n_bonds` 3)
   that no `w > 0` dihedral covers in the exceptions kernel (below), at its
   1-4 parameters and the `special_bonds` 1-4 weights, as LAMMPS's
   `dihedral charmm` prices a `w` pair; the regular pair styles weigh it 0.
   LAMMPS's pair style has no `"epsilon14"` form, so the LAMMPS writer
   refuses the param (its deck would price the regular parameters).
3. **Per-dihedral weight** — `dihedral_style charmm` `w`: each dihedral
   prices the pair of its own end atoms,

   E₁₄ = w·[4ε₁₄((σ₁₄/r)¹² − (σ₁₄/r)⁶) + C qᵢqⱼ/r],

   with the `lj/charmm` 1-4 parameters of the two types, the Coulomb style's
   C = `coulomb`/`dielectric`, no cutoff and no switch
   (`dihedral_charmm.cpp`; LAMMPS tallies it into `evdwl` and `ecoul`). `w`
   is 1, ½ in six-membered rings, 0 in four- and five-membered rings; a pair
   at the ends of several dihedrals takes the sum of their `w`. As in LAMMPS,
   a field with any `w > 0` is refused unless its `special_bonds` 1-4 weights
   are both 0 (else the pair would be priced twice) and it has a `lj/charmm`
   style and a Coulomb style; `w` outside [0, 1] is refused. `w = 0` (AMBER's
   use of the style) prices LAMMPS's `K[1 + cos(nφ − d)]` alone.

**Special pairs.** The pair styles price every pair of atoms. A frame's
`pairs` block names the pairs they price otherwise — its *special pairs* —
one row per unordered pair: `atomi`, `atomj` and `n_bonds`, the pair's class,
its shortest bond distance (1, 2, 3; a pair a format excludes beyond three
bonds carries its own distance, 0 when no bonds connect it). A row of class
1-3 is weighed by `special_bonds` of that class; any other row by 1; a pair
no row names is at full strength. The rows are topology, generated with the
angles and dihedrals (`Atomistic::generate_topology`, which every typifier
runs; `Topology::pairs_block`), and a frame whose bonds name a pair no row
does is refused by both compile doors and by the GROMACS writer. Exceptions
LAMMPS cannot express — a GROMACS `[ pairs ]` row with explicit parameters,
an OpenMM `NonbondedForce` exception, an AMBER dihedral whose `SCEE`/`SCNB`
differ from the field's, a pair a format excludes — are nullable float
columns on the same rows:

| Column | Meaning |
|---|---|
| `epsilon`, `sigma` | the LJ parameters of this pair, in place of the pair style's row or mixing (or the dihedral's `epsilon14` / `sigma14`) |
| `lj_scale` | the van-der-Waals weight of this pair, in place of `special_bonds.lj` for its class (or of `w`) |
| `charge_product` | qᵢqⱼ (e²) of this pair, in place of the atoms' product |
| `coul_scale` | the Coulomb weight of this pair, in place of `special_bonds.coul` (or of `w`) |

A null cell (the column's validity mask) takes the value the pair has
without the row. OpenMM's exception `(chargeProd, sigma, epsilon)` is
`charge_product`, `sigma`, `epsilon` with both scales 1; a GROMACS funct-1
`[ pairs ]` row with parameters is `sigma`, `epsilon` (`lj_scale` 1,
`coul_scale` fudgeQQ), a funct-2 row adds `charge_product` and its own
fudgeQQ; AMBER's
per-dihedral divisors are `lj_scale = 1/SCNB`, `coul_scale = 1/SCEE`. A pair
style other than a 12-6 `lj/cut` or `lj/charmm` and a plain Coulomb
(`coul/cut` with `delta = 0`, `coul/charmm`) has no exception form, and a
field with one beside an exception is refused. LAMMPS can express none of
these per pair, so the LAMMPS writers refuse a frame that carries them,
naming the columns. (A `pair14` table in a record is an unknown category,
kept as such.)

**Precedence**, per pair and per quantity: **per-pair override > dihedral
charmm `w` > `special_bonds`**. Every pair that carries an override cell or
sits at the ends of a `w > 0` dihedral is priced once, by one exceptions
kernel at both compile doors,

E = lj_w·4ε[(σ/r)¹² − (σ/r)⁶] + coul_w·C qᵢqⱼ/r   (no cutoff, no switch),

where each of ε, σ, qᵢqⱼ, lj_w, coul_w is the override cell if there is one,
else the dihedral's (ε₁₄, σ₁₄, qᵢqⱼ, Σw, Σw) when `w > 0`, else the pair
style's (ε, σ, qᵢqⱼ, `special_bonds` weight of the pair's bond-distance
class). The regular pair kernels price such a pair at weight 0, at both
compile doors (`SpecialWeights`). A `w` pair needs no such step — its `special_bonds` 1-4
weight is 0. A cell is priced only by the style it belongs to: `epsilon`,
`sigma`, `lj_scale` under a Lennard-Jones style, `charge_product`,
`coul_scale` under a Coulomb style. A field without that style ignores them,
so a bonded-only (or Coulomb-only) field prices no Lennard-Jones whatever the
rows state. Converting an exception table to a global
weight is not exact (r₁₄ depends on φ and on the other coordinates), so no
reader or writer does it.

## Urey–Bradley

LAMMPS carries Urey–Bradley in one angle style, and so does the IR:
**`angle charmm`**, E = k(θ − theta0)² + k_ub(r₁₃ − r_ub)², parameters `k`
(E/rad²), `theta0` (deg), `k_ub` (E/L²), `r_ub` (L) — `angle_coeff t K theta0
K_ub r_ub`, r₁₃ the distance between the angle's end atoms. It adds no
exclusion (the 1-3 pair is excluded by `special_bonds`). There is no separate
Urey–Bradley category. A field that mixes it with other angle styles is
LAMMPS's `angle_style hybrid harmonic charmm` (`angle_coeff t charmm K theta0
K_ub r_ub`), which the LAMMPS reader and writer read and write; the core prices
each angle row under the style that defines its type. The LAMMPS, OpenMM
and GROMACS readers and writers and the prmtop reader (a chamber file) hold
it; the maps below are how each source writes it and maps onto the IR
(there is no reader of CHARMM's own `.prm`):

| Source | `angle charmm` |
|---|---|
| CHARMM `.prm` `ANGLES` `Ktheta Theta0 Kub S0` | as written (CHARMM has no ½) |
| GROMACS `[ angletypes ]` funct 5 `θ₀ k_θ r13 k_UB` (½k forms, nm, kJ/mol) | `k = k_θ/(2·4.184)`, `theta0 = θ₀`, `k_ub = k_UB/(2·418.4)`, `r_ub = 10·r13` |
| OpenMM `<AmoebaUreyBradleyForce><UreyBradley … k d>` (`AmoebaUreyBradleyForceBuilder.addUreyBradleys` adds `HarmonicBondForce.addBond(a1, a3, d, 2*k)`, energy ½·2k(r − d)², so `k` is un-halved) | `k_ub = k/418.4`, `r_ub = 10·d`, joined with the `HarmonicAngleForce` row of the same three labels in either direction; without one, `k = 0`. Checked against OpenMM's energy ([OpenMM XML](#openmm-xml)) |
| chamber prmtop `CHARMM_UREY_BRADLEY` (i, k, type), `…_FORCE_CONSTANT` `K_ub`, `…_EQUIL_VALUE` | every angle of the file `angle charmm`: `k = TK`, `theta0`, and `k_ub = K_ub`, `r_ub` of the term on its end atoms (0, 0 without one) |

## CMAP

The IR follows LAMMPS `fix cmap` (CHARMM's correction map), and the
kernel is `cmap charmm`, a step-for-step port of LAMMPS's
`src/MOLECULE/fix_cmap.cpp`:

- a crossterm names five atoms `(atomi, atomj, atomk, atoml, atomm)`;
  φ = dihedral(atomi, atomj, atomk, atoml), ψ = dihedral(atomj, atomk, atoml,
  atomm) — LAMMPS's `atan2` dihedrals, in degrees, the IUPAC sign (the core's
  `compute_dihedral`), in [−180°, 180°) with 180° read as −180°;
- a map is an N×N grid (CHARMM and LAMMPS: N = 24, 15° spacing) of energies
  in the force field's energy unit, stored **φ-major**: element `[i][j]` (flat
  index `i·N + j`) is the energy at φ = −180° + i·Δ, ψ = −180° + j·Δ,
  Δ = 360°/N — the order of a CHARMM / LAMMPS `.cmap` file (each `# phi`
  block is one φ row of N ψ values);
- the energy between grid points is LAMMPS's bicubic interpolation, with the
  derivatives LAMMPS precomputes from cubic splines.

A `cmap` style's row holds one map as its `grid` array parameter (`f64`
N×N, the layout above; the `forcefield` section's `grid` column is
`f64[T, N, N]`), and a frame's `cmaps` block lists the crossterms
(`atomi` … `atomm`, `type`).

### The interpolation, exactly

**Node derivatives** (`set_map_derivatives`). With x_m = ⌊N/2⌋, the map is
extended periodically to 2N × 2N: `T[p][q] = E[(p − x_m) mod N][(q − x_m) mod N]`,
so index `p` is the angle −180° + (p − x_m)·Δ and the extension covers
[−360°, 360°). `spline(y)` is LAMMPS's *natural* cubic spline on spacing Δ:
second derivatives `y''` with `y''₀ = y''_{n−1} = 0` and, for i = 1 … n−2,

```text
p = 1/(y''_{i−1} + 4),  y''_i = −p,
u_i = ((6y_{i+1} − 12y_i + 6y_{i−1})/Δ² − u_{i−1})·p,   u₀ = 0
y''_j = y''_j·y''_{j+1} + u_j   for j = n−2 … 0.
```

Every row `T[k]` is splined along ψ. For a node (φ index i, ψ index j), the
row splines give, down all 2N rows k, the value `Y_k = T[k][j']` and the slope
`D_k = (T[k][j'+1] − T[k][j'])/Δ − (Δ/3)·T''[k][j'] − (Δ/6)·T''[k][j'+1]`
(j' = j + x_m); `Y` and `D` are splined along φ, and with i' = i + x_m

```text
∂E/∂φ   = (Y_{i'+1} − Y_{i'})/Δ − (Δ/3)·Y''_{i'} − (Δ/6)·Y''_{i'+1}
∂E/∂ψ   = D_{i'}
∂²E/∂φ∂ψ = (D_{i'+1} − D_{i'})/Δ − (Δ/3)·D''_{i'} − (Δ/6)·D''_{i'+1}
```

per degree (per degree² for the cross term). The doubled map keeps the
natural end conditions half a period from every node they are read at, so
the slopes are those of a periodic spline to within its decay.

**Patch** (`bc_coeff`, `bc_interpol`). The cell is
`c_φ = ⌊(φ + 180°)/Δ⌋`, `c_ψ = ⌊(ψ + 180°)/Δ⌋`; its corners, counter-clockwise
from (c_φ, c_ψ), wrap mod N. The 16 coefficients `c_ab` are Numerical Recipes'
`bcucof` weight matrix applied to the corner values, Δ·∂E/∂φ, Δ·∂E/∂ψ and
Δ²·∂²E/∂φ∂ψ, and with t = (φ − φ_{c_φ})/Δ, u = (ψ − ψ_{c_ψ})/Δ

```text
E = Σ_{a,b=0..3} c_ab tᵃ uᵇ,
∂E/∂φ = (180/π)/Δ · Σ a·c_ab tᵃ⁻¹ uᵇ,   ∂E/∂ψ = (180/π)/Δ · Σ b·c_ab tᵃ uᵇ⁻¹
```

(per radian). At a node E is the grid value; E and its slopes are continuous
across every cell edge and across ±180°.

**Forces** (`post_force`). F = −(∂E/∂φ)·∇φ − (∂E/∂ψ)·∇ψ with LAMMPS's own
∇φ, ∇ψ expressions, so the crossterm is distributed onto the five atoms
exactly as LAMMPS distributes it: atomi feels φ only, atomm ψ only, and the
three shared atoms both; the forces sum to zero and exert no torque.

Two LAMMPS behaviours are kept because they change energies: a crossterm
with a degenerate dihedral plane (any of the four cross products
|r_ij × r_jk|² below 10⁻⁴ Å⁴) contributes **nothing**; and the map is a true
2-D function — no sum of 1-D torsions reproduces it. One is generalised: LAMMPS
fixes N = 24 and at most six maps, the kernel takes any N ≥ 2 and any number
of maps (LAMMPS stores the derivative grids rotated by N/2 and finds their cell
from φ wrapped to [0°, 360°); the core stores them unrotated and reads them at the
value cell — the same numbers but for float ties on a cell edge).

### Crossterms and LAMMPS files

- `ff.assign_cmaps(mol)` (Rust `ForceField::assign_cmaps`) replaces a
  typed molecule's `cmaps` terms (written by `to_frame` as the `cmaps` block):
  five atoms whose dihedrals `(a, b, c, d)` and `(b, c, d, e)` are both
  dihedrals of the molecule (either stored direction) and whose atom types
  equal a cmap row's `itom … mtom` **forward** — never reversed, since
  reading the five atoms backwards swaps φ and ψ.
- A LAMMPS `fix cmap` file reads (`LammpsForcefieldReader.read_cmap`,
  Rust `LammpsForcefieldReader::read_cmap`) into rows named `"1"` … `"K"` — map `t`
  is crossterm type `t` — and writes (`LammpsForcefieldWriter.write_cmap`,
  Rust `LammpsForcefieldWriter::write_cmap`) the `cmaps` labels' grids in label id
  order, in CHARMM's layout: CHARMM's own file comes back line for line.
- The data file's `N crossterms` header line and `CMAP` section
  (`index type a1 … a5`) are the frame's `cmaps` block (`type_id` = map
  index), both ways.
- The include writer emits `fix cmap all cmap <cmap_file>` and
  `fix_modify cmap energy yes`; that fix must reach LAMMPS before
  `read_data <data> fix cmap crossterm CMAP`. The include reader reads a
  `fix cmap` line's file relative to the include.

OpenMM's `CMAPTorsionForce` stores `energy[i + N·j]` at φ = 2πi/N, ψ = 2πj/N
(origin 0, φ fastest — `CMAPTorsionForce::addMap`): its reader maps element
`(i, j)` to the IR's `[(i + N/2) mod N][(j + N/2) mod N]` (each index shifted by
N/2, the axes swapped into φ-major), and refuses an odd N, which puts no
OpenMM node on −180°; the writer is the inverse. OpenMM takes its node slopes
from periodic splines where LAMMPS splines the doubled map with natural ends,
so energies off the grid points differ at the spline's end effect: 3·10⁻¹²
relative on CHARMM36's alanine map at an ACE-ALA-NME conformer
([OpenMM XML](#openmm-xml)). OpenMM's generator matches a crossterm's five
types forward or backward; `assign_cmaps` matches forward only. GROMACS `[ cmaptypes ]` lists CHARMM's grid in
the IR's layout (φ-major from −180°, GROMACS's `cmap_setup_grid_index`), so
its reader keeps element `i·N + j` at `[i][j]`; GROMACS's own interpolation
is CHARMM's too, and the CHARMM dipeptide's two crossterms price as GROMACS
2025.3 prices them to 6 × 10⁻¹⁶ (see [How this is checked](#how-this-is-checked)).

A prmtop stores a map as ParmEd's `CmapType.grid` — CHARMM's parameter-file
order, φ-major from −180° — in `CHARMM_CMAP_PARAMETER_nn` (a chamber file)
or `CMAP_PARAMETER_nn` (ff19SB), and the prmtop reader takes it as written.
sander's CMAP energy on a CHARMM36 alanine dipeptide and on an ff19SB one
off the grid points equals the core's and LAMMPS's to 1e-14 relative (see
[AMBER prmtop](#amber-prmtop)): sander interpolates as LAMMPS does.

## Torsion forms and their exact conversions

Every Class-I torsion and dihedral-angle improper style above is a finite
Fourier series in φ — the signed dihedral of the atoms as stored, the angle
every kernel of the core and LAMMPS compute:

    E(φ) = Σₙ₌₀ aₙ cos nφ + bₙ sin nφ

That series is the intermediate of every conversion between forms
(`molrs::ff::ir::torsion`). Each form **embeds** exactly (the series
is the same function of φ, constant included) and **projects** back exactly
— every coefficient, a₀ included — or refuses, naming the condition that
prevents it. Rows of several styles on one quadruple are one torsion: their
series add.

| Form | aₙ, bₙ of the form (n ≥ 1) | constant a₀ | image condition |
|---|---|---|---|
| `dihedral periodic` (LAMMPS `fourier`), Σₘ kₘ[1 + cos(nₘφ − γₘ)] | aₙ += kₘ cos γₘ, bₙ += kₘ sin γₘ | Σₘ kₘ; a periodicity-0 term k₀[1 + cos 0] = 2k₀ holds any other | none — every series; back: kₙ = √(aₙ² + bₙ²), γₙ = atan2(bₙ, aₙ), k₀ = (a₀ − Σkₙ)/2 |
| `dihedral charmm`, k[1 + cos(nφ − d)] | as one `periodic` term; `w` is no torsion parameter: a row with w ≠ 0 does not embed, a projection gives w = 0 | k | one order, √(aₙ² + bₙ²) = \|a₀\|; back: k = a₀ |
| `improper periodic`, k[1 + cos(nφ − γ)] | as one `periodic` term | k | as `charmm` |
| `dihedral harmonic`, `improper cvff`, k[1 + d cos nφ] | aₙ = k d | k | one order, bₙ = 0, \|aₙ\| = \|a₀\|; back: k = a₀, d = aₙ/a₀ |
| `dihedral opls`, ½Σ kₙ[1 ± cos nφ] | a₁ = k₁/2, a₂ = −k₂/2, a₃ = k₃/2, a₄ = −k₄/2 | ½Σkₙ | bₙ = 0, n ≤ 4, a₀ = a₁ − a₂ + a₃ − a₄ (E(180°) = 0) |
| `dihedral class2` (torsion part), Σₙ₌₁³ kₙ[1 − cos(nφ − φₙ)] | aₙ = −kₙ cos φₙ, bₙ = −kₙ sin φₙ | Σkₙ | n ≤ 3, a₀ = Σ ±√(aₙ² + bₙ²) for some signs |
| `dihedral multi/harmonic`, Σₙ₌₁⁵ Aₙ cosⁿ⁻¹φ | cosᵏφ = 2⁻ᵏ Σⱼ C(k, j) cos((k − 2j)φ) | free | bₙ = 0, n ≤ 4; back: cos nφ = Tₙ(cos φ) |
| `dihedral nharmonic`, Σᵢ₌₁ᴺ Aᵢ cosⁱ⁻¹φ | as `multi/harmonic` | free | bₙ = 0 |
| `dihedral rb` (GROMACS funct 3, OpenMM `RBTorsionForce`), Σₙ₌₀⁵ Cₙ cosⁿ(φ − 180°) | `nharmonic` with Aₙ₊₁ = (−1)ⁿ Cₙ (cos(φ − 180°) = −cos φ) | free | bₙ = 0, n ≤ 5 |

Every periodicity must be an integer, as LAMMPS requires of every style
here. A "sine term" (bₙ ≠ 0) is a phase other than 0° or 180°; phases on
multiples of 90° are evaluated exactly, so a 180° phase is bₙ = 0, not
1.2·10⁻¹⁶. The n ≥ 1 coefficients are compared exactly; the constant, a sum
of products, to 10⁻¹² of the series' scale Σ(|aₙ| + |bₙ|).

**The constant term.** a₀ is part of the energy, so an exact conversion
keeps it: the polynomial forms (`multi/harmonic`, `nharmonic`, RB) and
`periodic` carry any constant; every other form fixes its constant by its
other parameters, which is part of its image condition (a single-term form
takes k = a₀, so a negative k survives the round trip). A constant shifts no
force — two torsions are the same physics iff their series agree for n ≥ 1,
which `FourierSeries::canonical` compares — and a conversion *up to* a
constant is a fit with a free offset ([Converting between
forms](#converting-between-forms)), which reports the offset. In particular
ΣCₙ = 0 is the condition of RB → OPLS beyond C₅ = 0. The GROMACS reader
(`[ dihedraltypes ]` funct 3) and the OpenMM XML reader (`<RBTorsionForce>`)
read RB as the polynomial it is (`multi/harmonic`, or `nharmonic` when
C₅ ≠ 0), which holds every RB row exactly, constant included; neither
refuses ΣCₙ ≠ 0 or C₅ ≠ 0.

The familiar chains are instances:

- **RB ↔ `multi/harmonic`**: A₁ = C₀, A₂ = −C₁, A₃ = C₂, A₄ = −C₃, A₅ = C₄
  (C₅ = 0); with C₅ ≠ 0 the exact target is `nharmonic` (N = 6).
- **RB ↔ OPLS** (GROMACS manual Eqs. 200–201): k₁ = −2C₁ − 3C₃/2,
  k₂ = −C₂ − C₄, k₃ = −C₃/2, k₄ = −C₄/4; back C₀ = k₂ + (k₁ + k₃)/2,
  C₁ = (−k₁ + 3k₃)/2, C₂ = −k₂ + 4k₄, C₃ = −2k₃, C₄ = −4k₄, C₅ = 0.
- **OPLS ↔ `periodic`**: one term per non-zero kₙ, `k = |kₙ|/2`, phase 0° at
  odd n and 180° at even n for kₙ > 0 (the other for kₙ < 0), and a
  periodicity-0 term for the constant those leave over.

**Outside the image.**

- `improper harmonic`, K(|φ| − chi0)², and LAMMPS `dihedral quadratic`,
  K(φ − φ0)², are not finite Fourier series and are refused (`improper
  harmonic` registers no form codec; `fit_form` projects onto it under a
  metric, with a residual). A periodic
  improper k[1 + cos(nφ − γ)] and a harmonic one agree only to second order
  about the minimum: **K = n²k/2** in LAMMPS's un-halved K (K = 2k for
  AMBER's n = 2, γ = 180°; the `k_h = n²k` sometimes quoted is the ½-form
  harmonic ½k_h χ²), with chi0 the minimum nearest 0 and, back,
  γ = n·chi0 + 180°. The quartic terms differ by −k n⁴ δ⁴/24.
- `dihedral class2`'s cross terms (`mbt`, `ebt`, `at`, `aat`, `bb13`) couple
  the torsion to bonds and angles: outside the Class-I IR.
- `improper fourier` (the core's `uff_inversion`) and `mmff_oop` price an
  out-of-plane (Wilson) angle, not a dihedral.

In Rust:

```rust
use molrs::ff::ir::torsion::{FourierSeries, MultiHarmonicForm, OplsForm, TorsionRefusal};
use molrs::ff::style_registry::torsion_series;

// A stored row of any torsion style → its series (exact, constant included),
// through the style's registered form codec.
let series = torsion_series("dihedral", "opls", style.params(), &row)?;
// Several rows on one quadruple → one series; compare without the constant.
let total: FourierSeries = rows.iter().map(|r| r.to_series()).sum();
assert_eq!(total.canonical(), other.canonical());
// Series → a form, or the reason it cannot be.
match MultiHarmonicForm::from_series(&series) {
    Ok(form) => form.to_params(),
    Err(TorsionRefusal::SineTerm { n, .. }) => todo!("phase off 0/180° at order {n}"),
    Err(other) => todo!("{other}"),
};
```

The per-form types (`PeriodicForm`, `CharmmForm`, `CosineTermForm`,
`SignedCosineForm`, `OplsForm`, `Class2Form`, `MultiHarmonicForm`,
`NHarmonicForm`, `RyckaertBellemansForm`, `ImproperHarmonicForm`; the `Form`
suffix tells a codec's form from the kernel's form of the same style, such
as `ff::potential::improper::Harmonic`) carry the maps one form at a time (`from_params`,
`to_params`, `to_series`, `from_series`, and `nearest` — the closest member
of the form's image, the start of a fit). `FourierSeries::chopped(tol)`
zeroes coefficients below a tolerance first, for input rounded on print. A
test evaluates every registered kernel on random geometries against the
series its codec embeds the row as, so the algebra, the codecs and the
kernels cannot drift.

## Converting between forms

A **form family** is a set of styles that are one function space
parametrised differently. Each member registers a *form codec* beside its
kernel: its family, and two exact maps between its parameters and those of
the family's one **canonical** style — `embed` (this style → canonical, the
same energy, constant included, or a refusal) and `project` (canonical →
this style, exact on its image, else a refusal naming the condition) — and
optionally a `seed`, the nearest member of its image, where a fit starts.

| Family | Canonical style | Members (exact image in the canonical style) |
|---|---|---|
| `torsion` | `dihedral periodic` | `dihedral` `charmm` (w = 0), `opls`, `multi/harmonic`, `nharmonic`, `harmonic`, `class2`, `rb`; `improper` `cvff`, `periodic` |
| `bond` | `bond harmonic` | `bond class2` (k3 = k4 = 0) |
| `angle` | `angle harmonic` | `angle class2` (k3 = k4 = 0), `angle charmm` (k_ub = 0) |
| `lj` | `pair lj/cut` | `pair lj/class2` = `lj/cut` with n = 9, m = 6, σ′ = (2/3)^⅓ σ (σ of `lj/class2` is the minimum, of `lj/cut` the zero) |

No codec, by design: `improper harmonic` and `morse` are in no family's
function space; `pair lj/charmm` always switches (its kernel requires
inner < cutoff), so it is never exactly `lj/cut`. The canonical torsion keeps
the constant as a periodicity-0 term, so `canonical()` is exact on energy,
not only on forces. Impropers share the torsion algebra (their φ is the same
signed dihedral) but stay impropers: the canonical style is a dihedral style,
and moving them would change which block prices them; `to_form("improper",
"periodic")` is the improper conversion.

Three operations, one job each, each returning a new force field:

- **`ForceField.canonical()`** maps every style of a family in the canonical
  style's category onto it, rows in their canonical spelling (for `periodic`:
  one term per order, k > 0, γ ∈ (−180°, 180°], the constant term first).
  Styles of no family stay, in their place; idempotent.
- **`ForceField.to_form(category, style)`** converts every style of
  `category` in `style`'s family to `style`, through the canonical
  parameters, exactly or not at all. A refusal names the source style, the
  type and the condition: `dihedral periodic type 'CT-CT-CT-CT' has no exact
  `dihedral rb` form: sin(2φ) coefficient 0.5 ≠ 0 …`.
- **`ForceField.fit_form(category, style, metric)`** is the projection that
  always answers: each row of every other style of `category` is fitted to
  `style` by least squares under a declared metric — sample points q of the
  coordinate (r; θ, φ in radians) with weights w, optionally Boltzmann
  factors exp(−(E − E_min)/kT) of the row's own energy, optionally a free
  constant offset — minimising Σᵢ wᵢ (E_fit(qᵢ) + c − E_src(qᵢ))². It sees
  every style only through its energy as the registry's kernel prices it, so
  it works for expression and Python styles as for native ones. A row in the
  target's image is converted exactly (`exact`); otherwise the fit starts at
  the codec's seed (else the source's same-named values) and varies the
  target's dimensioned parameters — a dimensionless one (a periodicity, a
  sign, a 1-4 weight) is a choice of form and stays. The residual is
  reported per row (`sum_sq` = the minimised Σ w r², `rms`, `max_abs`, the
  offset); `sum_sq` is monotone in the metric: pointwise larger weights, or
  more points, never lower it.

```python
import numpy as np
import molpy as mp

ff = mp.ff.forcefield.ForceField("demo", units="real")
ct = ff.def_style("atom", "full").def_type("CT", mass=12.011)
ff.def_style("dihedral", "periodic").def_type(
    "CT-CT-CT-CT", ct, ct, ct, ct, k1=1.0, periodicity1=2, phase1=0.0
)

canonical = ff.canonical()                    # exact
rb = ff.to_form("dihedral", "rb")             # exact, or ValueError naming why
phi = np.linspace(-np.pi, np.pi, 72, endpoint=False)
opls, residual = ff.fit_form("dihedral", "opls", phi, offset=True)
residual["rms"], residual["types"][0]["offset"]
```

A third party's style joins a family by registering a codec
(`molrs::ff::style_registry::register_form(category, style, FormCodec::new(family,
embed, project))`); from then on `canonical`, `to_form` and `fit_form` treat
it as a built-in. `fit_form` refuses a pair category (its unlike pairs follow
the style's mixing rule, which a per-row fit cannot hold).

## OpenMM XML

`OpenmmXmlReader` (`OpenmmXmlReader().read(source)`) reads OpenMM's
`<ForceField>` — its own CHARMM36, AMBER and OPLS-AA ports and the foyer /
molpy packs — and `OpenmmXmlWriter` (`OpenmmXmlWriter().write(sink, ff)`) writes
the inverse, each
number in the shortest form that reads back to the same `f64` unless a
`precision` is given. The IR's definitions are LAMMPS's; OpenMM's are
converted at the boundary:

| OpenMM | IR | Conversion |
|---|---|---|
| `<HarmonicBondForce><Bond length k>` | `bond harmonic` | `r0 = 10·length`, `k = k/(2·418.4)` (OpenMM's ½k) |
| `<HarmonicAngleForce><Angle angle k>` | `angle harmonic` | `theta0` in degrees, `k = k/(2·4.184)` |
| `<AmoebaUreyBradleyForce><UreyBradley k d>` | `angle charmm`, joined with its angle row | `k_ub = k/418.4`, `r_ub = 10·d` ([Urey–Bradley](#ureybradley)) |
| `<PeriodicTorsionForce><Proper k_m periodicity_m phase_m>` | `dihedral periodic` | `k_m/4.184`, phases in degrees; the writer also writes `dihedral charmm` (`w = 0`), `harmonic` and `class2` here, term for term, constant included |
| `<PeriodicTorsionForce><Proper c0..c3>` (CL&P / foyer) | `dihedral opls` | `k_n = c_{n−1}/4.184` |
| `<PeriodicTorsionForce><Improper>` | `improper periodic` (one term) | stored in the order OpenMM prices ([Improper atom order](#improper-atom-order)) |
| `<RBTorsionForce><Proper c0..c5>` | `dihedral multi/harmonic` (C₅ = 0) or `nharmonic` (N = 6) | `Aₙ₊₁ = (−1)ⁿ Cₙ/4.184`, constant included; the writer writes `multi/harmonic`, `nharmonic` (N ≤ 6) and `opls` here |
| `<CustomTorsionForce energy="k*(theta-theta0)^2">` `<Improper>` | `improper harmonic` | `k/4.184`; OpenMM's θ is signed and LAMMPS's χ = \|φ\|, which agree at `theta0 = 0` only — another `theta0` is refused (CHARMM36's two `theta0 = π` rows among them) |
| `<CustomTorsionForce energy="k*(abs(theta)-theta0)^2">` | `improper harmonic` | `chi0` = theta0 in degrees (the writer's form when some `chi0 ≠ 0`) |
| `<CMAPTorsionForce><Map>`, `<Torsion map>` | `cmap charmm` | [CMAP](#cmap) |
| `<NonbondedForce coulomb14scale lj14scale><Atom charge sigma epsilon>` | `pair lj/cut` (`mixing` = the root's foyer `combining_rule`, else `arithmetic`) + `pair coul/cut`; `charge` on `atom full`; the writer writes `coul/long/pme` here too | `sigma` × 10, `epsilon` ÷ 4.184; `special_bonds` `[0, 0, scale]` |
| `<LennardJonesForce lj14scale><Atom sigma epsilon [sigma14 epsilon14]>`, `<NBFixPair>` | `pair lj/charmm` (`arithmetic`, NBFIX as cross rows) + `pair coul/charmm`, the `<NonbondedForce>` beside it (its `epsilon` 0) giving the charges | 1-4: `special_bonds` `[0, 0, lj14scale]` / `[0, 0, coulomb14scale]`, and `one_four = "epsilon14"` when a type's 1-4 parameters differ ([1-4](#1-4-interactions)); an NBFIX row is OpenMM's 1-4 parameters for its pair too, LAMMPS's two-number cross row |

Rows key on `class{n}` or `type{n}` as written; OpenMM's wildcard (an empty
attribute) is `""`, and a row naming neither, which OpenMM ignores, is
refused. Every Coulomb style states OpenMM's constant (`ONE_4PI_EPS0` =
332.06371329919216 kcal·Å/(mol·e²), 9.9·10⁻⁹ above LAMMPS `real`'s
`qqr2e`); the writers do not carry it (each engine fixes its own). OpenMM's
cutoffs and switching are `createSystem` arguments, so no style read from a
file has a `cutoff` (or `inner`): `lj/cut` and `coul/cut` then price every
pair untruncated, as `NoCutoff` does ([Cutoffs](#pair-styles)), and the
caller states `inner` and `cutoff` of `lj/charmm` / `coul/charmm`, which
have no default.

The Coulomb style is written as the charges alone, the long-range
`coul/long/pme` as `coul/cut` is: whether OpenMM sums them with `NoCutoff`, a
cutoff, `Ewald` or `PME` is `createSystem`'s `nonbondedMethod` (with
`nonbondedCutoff` and `ewaldErrorTolerance`), chosen when the System is
built, as LAMMPS's `kspace_style` and GROMACS's `.mdp` `coulombtype` are. A
`coul/long/pme` that states its Ewald parameters (`alpha`, `order`,
`grid_*`) is refused: the XML has no place for them.

OpenMM's `createSystem` mixes every `<NonbondedForce>` pair
Lorentz-Berthelot, 1-4 exceptions included, and ignores foyer's
`combining_rule`; no tag of its schema states a geometric σ for both the
pairs and the 1-4 exceptions (a `<CustomNonbondedForce>` excludes whole
bonded neighbours and scales none; a `<LennardJonesForce>` mixes arithmetic
but for one `<NBFixPair>` per type pair). So a `geometric` `lj/cut` (OPLS-AA)
is written with the root's `combining_rule="geometric"` and a `<Script>`,
which `createSystem` runs last: it prices the Lennard-Jones in a
`CustomNonbondedForce` named `GeometricLennardJones` (√(σᵢσⱼ), √(εᵢεⱼ), the
`NonbondedForce`'s cutoff, switch and dispersion correction, periodic under
`CutoffPeriodic`, `Ewald` and `PME`; `LJPME` is refused), keeps the charges
in the `NonbondedForce` at ε = 0, and gives each 1-4 exception σ = √(σᵢσⱼ)
at its `lj14scale`·√(εᵢεⱼ) — foyer's patch of the System. The reader reads
that script, and only that one, as `mixing geometric`. Such an XML is for
OpenMM's `app.ForceField`: a tool that applies `combining_rule` itself
(foyer) is given the source file.

**Refused**, by name: `<Script>` (but the writer's geometric-mixing one) /
`<InitializationScript>`, every
`Custom*Force` other than the harmonic improper, a `<Proper>` under it,
an RB `<Improper>`, `ordering="smirnoff"`, a multi-term periodic improper,
an odd CMAP size, a wildcard Urey–Bradley row, an `<NBFixPair>` of a type
with itself, a `<NonbondedForce>` with non-zero `epsilon` beside a
`<LennardJonesForce>` (OpenMM prices both), and every other force (AMOEBA
multipoles, GBSA, Drude, …). A style outside the table that has an
expression is written as its category's `Custom*Force` ([Engine
codecs](#engine-codecs)). The writer refuses what OpenMM's tags cannot
hold: a style outside the table without an expression (`dihedral
mmff_torsion`, …),
`dihedral charmm` `w ≠ 0`, `special_bonds` other than `[0, 0, s]`,
`sixthpower` mixing or `geometric` with cross rows, a cross row with its own
1-4 parameters, `lj/charmm` under `one_four = "regular"` with 1-4 parameters
of its own at a non-zero weight (OpenMM would use them), a Coulomb style
with `dielectric ≠ 1` or `delta ≠ 0`, a `coul/long/pme` stating Ewald
parameters, charges on some atom types only, a
shifted or Mie `lj/cut`, a force field in units other than `real`, and two
types OpenMM's generator would match on the same labels with other
parameters (bonds, angles, propers and crossterms either way round,
impropers by their centre and the other three in any order; a proper's
periodic and RB rows on one quartet count, since OpenMM adds both) — the
same row twice is written once. A type without a `class` (a prmtop's, a
LAMMPS file's) is written as its own class, which OpenMM requires of every
`<Type>`. `<Residues>`, `<Patches>` and `<Info>` carry no parameters and are
skipped; placeholder atom types the reader makes for classes are not written.

A force-field XML is typed: OpenMM's generators find each term from the
atom types of a residue template. A system written for OpenMM therefore
needs residue templates whose charges are the atoms' (the field written
without type charges, so `<UseAttributeFromResidue name="charge"/>`), and
its rows must be what the generators find: a `dihedral` row on four atoms
that are no bonded chain (OPLS-AA's impropers as GROMACS funct 1) is the
same function as `improper periodic` over the same atoms and goes to OpenMM
as one; an `improper periodic` row's two outer atoms must be in the order
OpenMM's AMBER rule puts them (by element, then index). The equivalence
check ([Cross-engine equivalence](#cross-format-equivalence)) builds the
templates from the frame and holds OpenMM's energies to the core's.

`OpenmmXmlWriter().write_system(sink, ff, frame)` (Rust
`OpenmmXmlWriter::write_system`) writes those templates: the field as
`OpenmmXmlWriter().write(sink, ff)` writes it, plus one `<Residue>` per distinct
residue of the frame. The residues and atom names are the ones OpenMM's
`PDBFile` reads from `write_pdb` of the same frame (one rule, shared by
both writers): runs of consecutive atoms with the same `chain`, `res_id`,
`icode` and `res_name`; without `res_id`, each bonded molecule is a residue
named `MOL`; an atom without a `name` is its element numbered within the
residue (`C1`, `C2`, `H1`, …). Each template lists `<Atom name type
charge>`, a `<Bond>` per bond inside the residue and an `<ExternalBond>` per
bond leaving it. Charges are the frame's `charge` column, under
`<UseAttributeFromResidue name="charge"/>` (without the column, the types'
own charges, which every type must then carry), and every atom type the
frame uses gets an `element`, which OpenMM matches residues by. So
`ForceField(xml).createSystem(PDBFile(pdb).topology)` builds the System
directly. Refused, by name: a type the force field lacks, one type on two
elements, two atoms of one residue with the same name, no charges anywhere.
Two residues with the same bond graph and elements but other types or
charges become two templates OpenMM's matcher cannot choose between
(`createSystem` then needs `residueTemplates`).

## AMBER prmtop

AMBER is read, never written (no prmtop writer). The prmtop force-field
reader (`AmberPrmtopForcefieldReader`) and frame reader (`read_prmtop`) name
every type and row alike; a chamber prmtop (ParmEd's `chamber`, `%FLAG
CTITLE`) reads through the same pair.

| prmtop | IR |
|---|---|
| `BOND_*` `RK`, `ANGLE_*` `TK` (no ½), radians | `bond harmonic`, `angle harmonic` (`angle charmm` in a chamber file, see [Urey–Bradley](#ureybradley)) |
| `DIHEDRAL_*` `PK`, `PN`, phase; the rows of one quartet and each negative-`PN` chain are one torsion | `dihedral periodic` `k<m>`, `periodicity<m>`, `phase<m>` (degrees), terms sorted by periodicity; a second distinct set of terms on one type quartet (tleap reuses a quartet's first match, so two torsions of one quartet can differ) is the type `<quartet>@<n>` |
| an improper (negative 4th pointer) | `improper periodic` in AMBER's order; a multi-term one (several rows, or a chain) is one frame row and one type `<quartet>@<n>` per term, as LAMMPS `cvff` holds one term |
| a phase within 0.004 rad of ±π | ±180° exactly, as sander's `rdparm` (tleap writes π as `3.14159400`) |
| `CHARMM_IMPROPERS` `K_ψ (ψ − ψ₀)²` | `improper harmonic` `k = K_ψ`, `chi0 = ψ₀`, centre first; ψ₀ other than 0° / 180° refused (LAMMPS prices \|ψ\|) |
| `CHARMM_CMAP_*` / `CMAP_*` | `cmap charmm`, a type per map named by the five atom types (qualified `@<residue>` of the Cα when one name stands for two maps, as ff19SB's do); the frame's `cmaps` block |
| `LENNARD_JONES_ACOEF/BCOEF` via ICO | `lj/cut` (`lj/charmm` in a chamber file) self rows; a cross row where the entry is not Lorentz–Berthelot (NBFIX) |
| `LENNARD_JONES_14_ACOEF/BCOEF` (chamber) | `lj/charmm` `epsilon14` / `sigma14` (cross rows where not Lorentz–Berthelot) and `one_four = "epsilon14"` when the table differs from the regular one |
| `CHARGE` | ÷ 18.2223, `coul/cut` at 332.0522173; ÷ √332.0716 and `coul/charmm` at 332.0716 in a chamber file |
| `SCEE_SCALE_FACTOR` / `SCNB_SCALE_FACTOR` per torsion type | `special_bonds` 1-4 = 1/divisor most 1-4 rows carry; `AmberPrmtopForcefieldReader::read_system` (Python `AmberPrmtopForcefieldReader().read_system(source)`) writes the special pairs, with `coul_scale` / `lj_scale` on every pair weighted otherwise |
| `AMBER_ATOM_TYPE` | the type name; `<name>~<class>` where one name stands for two LJ classes or masses (a chamber file cuts CHARMM's types to four characters) |

sander prices a 1-4 pair once per proper row whose 3rd pointer is not
negative, at that row's `1/SCEE`, `1/SCNB` (never an improper's, whatever its
pointer). The field's weights are the divisors most such rows carry. The
frame's special pairs are every pair within three bonds with its `n_bonds`,
and the pairs whose summed weight differs from the field's carry their own
`coul_scale` / `lj_scale` (only the differing cells set): another divisor
(GLYCAM's 1.0 beside ff14SB's 1.2 / 2.0), a pair two rows list, a 1-4 pair
of the topology no row lists (weight 0, or 1 if the exclusion list leaves it
out), a pair `EXCLUDED_ATOMS_LIST` adds beyond three bonds (weight 0). A
chamber file's 1-4 Lennard-Jones (`one_four = "epsilon14"`) is priced at
`epsilon14` / `sigma14` by the compile doors directly.

sander clamps the cosine of a harmonic angle to ±0.999 before taking its
arccosine, so it prices an angle beyond 177.44° as 177.44°; LAMMPS and the core
price the IR's `k (θ − theta0)²` at the angle itself. A prmtop with a
near-linear angle (an `sp` carbon, θ₀ ≈ 180°) therefore prices its angle term
differently in sander, and only there.

Refused by name: polarizable (`IPOL > 0`), 12-6-4 (`LENNARD_JONES_CCOEF`),
non-zero 10-12 (`HBOND_ACOEF/BCOEF`, a negative ICO), perturbed, solvent-cap
and `IFBOX = 3` files; a 1-4 row on a negative-`PN` chain (sander prices the
pair once per chained term, and its Coulomb at a factor unlike any other 1-4
pair's); a 1-4 row on a bonded or angle-end pair; two terms of one improper
with one periodicity; a Urey–Bradley term on no angle or on several.

## GROMACS topologies

`GromacsTopForcefieldReader` reads a topology's directives into a force field
(`read`) or a whole `.top` into the force field and a typed frame
(`read_system`; Python `mp.io.gromacs.GromacsTopForcefieldReader(include=True).read_system(source)`);
the writer (`GromacsTopForcefieldWriter`) is the inverse of the directive map
(`write`) and of `read_system` (`write_system`). Every row is
exact; GROMACS's ½k forms are halved into LAMMPS's `K`, nm → Å, kJ → kcal,
degrees stay degrees. Every Coulomb style states GROMACS's own constant
(`mp.core.constants.GROMACS_ONE_4PI_EPS0`, its `ONE_4PI_EPS0` from
CODATA 2018, 138.93545764438196 kJ·nm/(mol·e²), which the unit registry
converts to 332.06371329919205 kcal·Å/(mol·e²): 9.9·10⁻⁹ above LAMMPS
`real`'s).

| GROMACS | IR |
|---|---|
| `[ defaults ]` comb-rule 2 / 3 | `mixing` `arithmetic` / `geometric` on the Lennard-Jones style (comb-rule 1, C6/C12, refused) |
| `[ defaults ]` gen-pairs, fudgeLJ, fudgeQQ | `special_bonds` `lj [0, 0, fudgeLJ]`, `coul [0, 0, fudgeQQ]` (gen-pairs `no`: `lj` 1-4 = 1, no pair is generated) |
| `[ atomtypes ]` V W | `atom full` type + Lennard-Jones self row `sigma = 10·V`, `epsilon = W/4.184` |
| `[ nonbond_params ]` funct 1 | explicit cross row |
| `[ pairtypes ]` funct 1 | `lj/charmm` + `coul/charmm`, `one_four = "epsilon14"`: `epsilon14 = ε/fudgeLJ`, `sigma14 = σ` on the self rows, and cross rows where the mix of the self rows would not give the pair GROMACS's 1-4 parameters (to 10⁻¹²) — so `special_bonds` × LJ(ε₁₄, σ₁₄) is GROMACS's 1-4 energy for every type pair. A pairtype equal to the generated pair changes nothing (`lj/cut` stays) |
| `[ bondtypes ]` funct 1 / 3 | `bond harmonic` (`k = k_b/2`) / `bond morse` |
| `[ angletypes ]` funct 1 / 5 | `angle harmonic` / `angle charmm` ([Urey–Bradley](#ureybradley)) |
| `[ dihedraltypes ]` funct 1 | `dihedral periodic`, one term |
| `[ dihedraltypes ]` funct 9 | `dihedral periodic`: the consecutive rows on equal labels are the terms `k<m>`, `periodicity<m>`, `phase<m>` of one type, in file order |
| `[ dihedraltypes ]` funct 3 (RB) | `dihedral multi/harmonic` `aₙ₊₁ = (−1)ⁿ Cₙ`, constant included; `dihedral nharmonic` (N = 6) when C₅ ≠ 0 |
| `[ dihedraltypes ]` funct 5 (Fourier) | `dihedral opls`, `kₙ = Cₙ` |
| `[ dihedraltypes ]` funct 2 | `improper harmonic`, `K = k_ξ/2`, `chi0 = ξ₀` ∈ {0°, 180°} (the signed GROMACS form equals `K(|φ| − chi0)²` there and nowhere else, so other ξ₀ are refused); atoms as written |
| `[ dihedraltypes ]` funct 4 | `improper periodic`; atoms as written (AMBER order) |
| `[ cmaptypes ]` funct 1 | `cmap charmm`, the grid unchanged in layout ([CMAP](#cmap)) |
| `X` | the empty wildcard |

The writer writes `dihedral periodic` with several terms as consecutive
funct-9 rows, `multi/harmonic` and `nharmonic` (N ≤ 6) as funct 3, `opls` as
funct 5, `dihedral harmonic` / `improper cvff` and `dihedral charmm` with
`w = 0` as periodic rows at phase 0° / 180°, `cmap charmm` as `[ cmaptypes ]`,
and the 1-4 parameters of an `lj/charmm` declared `one_four = "epsilon14"`
as `[ pairtypes ]` exactly where GROMACS would generate other ones. It
refuses `dihedral charmm` with `w > 0` (GROMACS prices a 1-4 pair by
`[ pairs ]`, never by a dihedral), `epsilon14` / `sigma14` on an `lj/charmm`
that does not price 1-4 pairs with them, two types GROMACS would read as
one (same labels, same function-code table), and a force field in units
other than `real`. `dihedral class2`'s torsion goes out as funct-9 rows at
phase φₙ + 180° (k[1 − cos(nφ − φₙ)] = k[1 + cos(nφ − φₙ − 180°)]). No
Coulomb constant is written: GROMACS prices at its own, as LAMMPS and OpenMM
do at theirs. No Coulomb style has a directive either — `coul/cut`,
`coul/charmm` and the long-range `coul/long/pme` alike: the `.top` holds the
charges, and how GROMACS sums them (`coulombtype = PME`, Ewald, cut-off,
with `rcoulomb`, `fourier-spacing`, `pme-order`, `ewald-rtol`) is the
`.mdp`'s, chosen at run time. A `coul/long/pme` stating its Ewald
parameters (`alpha`, `order`, `grid_*`) is refused.

**Systems.** `read_system` types each molecule row as GROMACS's own lookup
does — bonds, angles, Fourier dihedrals and cmaps by exact bond types
(cmaps forward only), the other dihedrals by the first row with the most
non-wildcard matches, either way, with all of a funct-9 row's terms; a row
with parameters of its own (an OPLS-AA `improper_*` macro, expanded) gets a
type of its own, `<labels>@gmx_<n>`. Its frame holds `atoms`, `bonds`,
`angles`, `dihedrals` (funct 1, 9, 3, 5), `impropers` (funct 2, 4), `cmaps`,
`constraints` (`[ constraints ]`, `[ settles ]`; `r0` in Å) and its special
pairs (`pairs`): per molecule, every pair within three chemical bonds (bonds
funct 1 and 3, constraints funct 1) with its `n_bonds`, weighed 0 where
`nrexcl` excludes it and no `[ pairs ]` row names it; a `[ pairs ]` row at
the fudges, or with parameters of its own carried as cells — funct 1
`sigma`, `epsilon`, `lj_scale` 1, `coul_scale` fudgeQQ; funct 2 also
`charge_product` and its own fudgeQQ — which LAMMPS cannot hold; an
`[ exclusions ]` pair weighed 0 at its own distance. Every pair of two
molecules is at full strength. Virtual sites, restraints, polarization and
free-energy B states have no IR form and are refused by name. A field read
with `one_four = "epsilon14"` (CHARMM's pairtypes) prices its 1-4 special
pairs at `epsilon14` / `sigma14`.

`write_system(sink, ff, frame)` writes the directives (no bonded
`[ *types ]` tables: each row carries its type's parameters on its line,
one funct-9 line per periodic term, so no lookup can pick another type;
`[ cmaptypes ]` stays, and a crossterm must be one GROMACS's lookup finds)
and one `[ moleculetype ]` (`nrexcl` 3) per molecule — a bond-graph
component, a run of consecutive atoms — with `[ atoms ]` (each atom's
charge and mass), `[ bonds ]`, `[ pairs ]`, `[ angles ]`, `[ dihedrals ]`
(propers and impropers), `[ cmap ]`, `[ exclusions ]` and `[ constraints ]`,
then `[ molecules ]`. `[ pairs ]` lists the frame's priced 1-4 special
pairs: funct 1, with `σ ε` when only the cells `epsilon` / `sigma` differ
from the generated pair, else funct 2 `fudgeQQ qᵢqⱼ 1 σ lj_scale·ε` — so
every cell holds. `[ exclusions ]` holds each special pair beyond three
bonds the frame weighs 0, so GROMACS prices every pair as the frame does.
Refused by name: a frame whose bonds have no special pairs, a pair within
three bonds the frame prices other than `nrexcl` 3 and the fudges would,
cells without `epsilon` and `sigma`, a crossterm
GROMACS would give another grid, a molecule whose atoms are not
consecutive or a row across two molecules, and every force-field refusal
above. A topology `read_system` reads from it is the system: written again,
it is the same file.

## GAFF and GAFF2

`GaffTypifier` (Python `mp.ff.typifier.GaffTypifier(parameter_set=
"gaff" | "gaff2")`) matches a molecule whose atoms carry GAFF types —
`AtdTypifier` with the same `parameter_set` stamps them — against the
`gaff.dat` / `gaff2.dat` tables compiled into the core (GAFF 1.81, GAFF2 2.2.30,
AmberTools 26.1), and writes the IR: `atom full` (mass), `pair lj/cut`
(σ = 2·R\*/2^(1/6), ε), `pair coul/cut` at AMBER's 332.0522173,
`bond harmonic` and `angle harmonic` (`k = K`, θ₀ in degrees),
`dihedral periodic` (`k<m> = PK/IDIVF`), `improper periodic` (AMBER order,
centre third) and `special_bonds` ½ / ⅚ (SCNB 2, SCEE 1.2). Charges are not
a GAFF parameter: a charge model writes `atoms.charge`.

What the table lacks is estimated as AmberTools estimates it, and each
estimate carries `estimated`, `estimate_penalty`, `estimate_method` and
`estimate_analog`:

- **Torsions** follow parmchk2's `chk_torsion`: equivalent-type rows, the
  wildcard row `X-j-k-X` (a parameter, not an estimate), equivalent-type
  wildcard rows, then the cheapest corresponding-type row and wildcard row,
  scored as parmchk2 scores them.
- **Impropers** follow parmchk2's `chk_improper` (estimates at the centres
  `PARMCHK.DAT` flags as planar) and then tleap: an improper wherever tleap
  finds a row for a triple of an atom's neighbours, its atoms in the order
  tleap gives them.
- **Bonds and angles** follow parmchk2's `chk_bond` / `chk_angle`:
  equivalent-type rows, the cheapest corresponding-type row (an angle within
  `THRESHOLD_BA`), then, for an angle, Wang's empirical `K_θ` and the mean
  θ₀ of the `A-B-A` and `C-B-C` rows, for its own, equivalent and
  corresponding types. A corresponding type scores parmchk2's columns:
  `bl` + `blf` at a bond end, `ba` + `baf` at an angle end, (`cba` +
  `cbaf`)·`WEIGHT_BA_CTR` at its vertex (caffeine's `c-cc-na` is
  `c2-cc-na` at 2.6).

An estimate's numbers are the frcmod's, at the precision parmchk2 prints
them (a bond's r₀ to three decimals: `ne-s6` copies `ne-sy`'s 1.6723 and
prices 1.672), since that is what tleap reads.

Every term is named by its types alone, as the prmtop reader names it: a
bond, angle or torsion in the orientation that reads smaller, type by type
(`c3-c3-oh-ho`, where `gaff.dat` writes `ho-oh-c3-c3`), its cosine terms
numbered by periodicity, then phase; an improper in AMBER's order. An
estimate is named the same way (`f-c3-sy`) and says what it is in its
params. So the native force field of a molecule and the one
`AmberPrmtopForcefieldReader().read` reads from tleap's prmtop of it merge
(`ForceField.merge`) into one, type for type.

parmchk2 searches the molecule in atom and bond order and reuses a name's
first estimate (an improper estimate is itself a row later impropers can
copy), so a name can be estimated differently in two molecules (aspirin's
ester carbon takes `c3-o -c -oh`, ethyl acetate's the `X -X -c -o` amide
term). Typed into one output, the first keeps the name and each further
estimate of it is `<types>@<n>` (`c3-o-c-os@2`), as the prmtop reader
numbers a further term set on one quartet.

**Which bond orders the types follow.** antechamber (its default `-j 4`)
ignores the bond orders of its input: `bondtype -j full` judges new ones from
the connectivity — each atom's penalty per valence (`APS.DAT`), valence states
tried in order of total penalty, and for each a search that settles forced
bonds and tries the first undecided bond in file order single, double, then
triple — and the atom types follow that structure. `AtdTypifier` does the
same by default (`bond_orders=BondOrderSource.Connectivity`; Rust
`BondOrderSource::Connectivity`):
it types a molecule as antechamber types the mol2 file holding the same atoms
and bonds in the same order, whatever orders the graph carries. Where a
molecule has more than one Kekulé structure (azulene, cyclooctatetraene,
polycyclic aromatics) the structure — and with it the `cc` / `cd`, `ce` /
`cf`, `nc` / `nd`, `cp` / `cq` colouring — is antechamber's, and is decided
by that order, as antechamber's is. Two things still read the input's bond
types, because `bondtype` reads them from its file: which rings count as
aromatic for the 7 / 8 bond types (a quinone drawn with its C=O is not; read
from a connectivity-only file, it is), and a molecule no valence state closes
(tropylium), where antechamber keeps the file's bonds. Every hydrogen must be
drawn. `bond_orders=BondOrderSource.Stated` keeps the graph's own orders instead (aromatic
bonds kekulized): a cyclooctatetraene drawn `C1=CC=CC=CC=C1` then types `cc
cd cd cc …`, the structure it was drawn with, where antechamber answers `cc cc
cd cd …`. `assign_bond_orders` writes the judged structure onto a
molecule, in place. (`BondOrderSource` is `mp.perceive.BondOrderSource`;
`assign_bcc_bond_types(mol, source)` takes the same choice.)

The rest follows `antechamber` as well: rings and ring classes (`AR1` …
`AR5`, `RG*`, `NR`) are its own (`ring.c`: every chordless ring of up to ten
atoms, the AM1-BCC indole rule under `bcc` / `abcg2`), not a smallest set of
smallest rings, and the colouring is its two post-typing passes (`atadjust`
for `cc` / `ce` / `cg` / `nc` / `ne` / `pc` / `pe`, swept in bond order;
`cpadjust` for the biphenyl bridge carbon, so o-terphenyl's middle ring is
`cp … cq`).

Checked against AmberTools 26.1:

- antechamber + parmchk2 + tleap + sander on 127 molecules (neutral and
  charged; aromatic, heteroaromatic, conjugated, strained, S / P / B /
  halogen chemistry) under both sets, with antechamber's atom types: every
  bond, angle, torsion and improper row (atoms, atom order, every parameter),
  every estimate's analog and penalty (all 634 frcmod rows), every atom's
  σ / ε, and every energy term at perturbed coordinates — bond, angle,
  dihedral with impropers, 1-4 and other van der Waals and Coulomb — agree.
  The angle energy agrees to the 4·10⁻⁷ by which tleap's π (3.141594) moves
  θ₀, except at a near-linear angle: sander clamps cos θ to ±0.999 (θ never
  above 177.44°), LAMMPS and the core do not (phenylacetylene's `c1` / `cg`
  angles, 0.055 kcal/mol).
- parmchk2 on 800 type graphs built to need bond and angle estimates: every
  bond and angle it writes — parameters, analog, penalty, and every `ATTN`
  (a missing term here) — agrees.
- `AtdTypifier` against antechamber's atom types, on the mol2 files
  antechamber read (connectivity only): identical on all 127 (and on
  phenylboronic acid, which antechamber types but cannot charge) under all
  seven `-at` tables (`gaff`, `gaff2`, `bcc`, `abcg2`, `amber`, `sybyl`,
  `gas`), with the bond types `bondtype -j full` writes identical bond for
  bond — and likewise on 173 further molecules built to stress the Kekulé
  search and the ring perception (polycyclic and non-alternant aromatics,
  annulenes, terphenyls, heteroaromatics, cations, anions and zwitterions,
  N-oxides, azides, molecules no valence state closes, drug-like molecules).

A term neither the table nor parmchk2's search reaches is an error where
parmchk2 writes a zero marked `ATTN, need revision`.

## Parameters as frame columns

`ForceField.materialize_params(frame, prefix=…)` (Rust
`ForceField::materialize_params`) writes the parameters this force field
gives each row of a typed frame next to the row, as columns
`<prefix><parameter>`:

| Block | Rows priced by | Columns (`lj/cut` + `harmonic` + `periodic` field) |
|---|---|---|
| `bonds`, `angles`, `dihedrals`, `impropers`, `cmaps` | the type its `type` names, under the one style of the category that defines it | `k`, `r0`; `k`, `theta0`; `k<m>`, `periodicity<m>`, `phase<m>`; `k`, `periodicity`, `phase` |
| `atoms` | `atoms.type`: every `atom` style's type, every `pair` style's self row | `mass`; `epsilon`, `sigma` |

Nothing in it is per style: the columns are the numeric parameters the
field stores for each type, in the IR's units (degrees, `K` without a ½, the
field's `units`), so a style added to the IR is written by the same call. A
row whose type lacks a parameter another row's type has — a two-term torsion
beside a three-term one, `estimate_penalty` on an estimated term only — is a
null cell (`Block.validity(column)`), not a zero. String and array
parameters, a pair style without per-type rows (`coul/cut`) and cross rows
(NBFIX, which have no per-atom form; the style's `mixing` combines the self
rows) are not written; charges are frame data unless the field stores them
per type. A column already present under a written name is replaced. It
returns block → columns written, and refuses a row whose type no style (or
two styles) of its category defines.

The use it is for is a **reference field**: type a molecule with one field
and read another's parameters beside the same rows — e.g. GAFF2's bonded
terms and Lennard-Jones as the reference of a learned Class-I field,

```python
mol, _ = mp.conformer.Conformer(add_hydrogens=True, seed=42).generate(
    mp.io.smiles.read_smiles("CCO").to_atomistic()
)
labelled = mp.ff.typifier.AtdTypifier(parameter_set="gaff2").typify(mol)
gaff2 = mp.ff.typifier.GaffTypifier(parameter_set="gaff2")
frame = gaff2.typify(labelled).to_frame()
gaff2.forcefield().materialize_params(frame, prefix="gaff2_")
frame["bonds"]["gaff2_k"], frame["atoms"]["gaff2_sigma"]
```

## Engine maps at a glance

| Engine | bond `k` | angle `k`, `theta0` | phases | impropers |
|---|---|---|---|---|
| LAMMPS | `K` | `K`, deg | deg | as written |
| GROMACS (`.top`/`.itp`) | `k_b/2`, kJ→kcal, nm→Å | `k_θ/2`, deg | deg | as written |
| OpenMM XML | `k/2`, kJ→kcal, nm→Å | `k/2`, rad→deg | rad→deg | (c1..c4) ↔ (c2, c3, c1, c4); see [OpenMM XML](#openmm-xml) |
| AMBER prmtop | `RK` | `TK`, rad→deg | rad→deg (±π snapped) | AMBER order |
| chamber prmtop | `RK` | `TK`, rad→deg (`angle charmm`) | rad→deg | AMBER order; `CHARMM_IMPROPERS` centre first |
| AMBER frcmod (writer) | `RK = k` | `TK = k`, deg | deg | AMBER order |
| GAFF / GAFF2 tables | `K` | `K`, deg | deg | AMBER order |
| OPLS-AA table (GROMACS `oplsaa.ff`) | `k_b/2` | `k_θ/2`, deg | — | — |

Each engine fixes its Coulomb constant, and each reader states its engine's
on the Coulomb style (`coulomb`, kcal·Å/(mol·e²)); no writer writes one, so
a field is priced by an engine at that engine's:

| Engine | Coulomb constant | Relative to LAMMPS `real` |
|---|---|---|
| LAMMPS `real` (`qqr2e`) | 332.06371 | — |
| OpenMM (`ONE_4PI_EPS0`, CODATA 2018) | 332.06371329919216 | + 9.9·10⁻⁹ |
| GROMACS 2025 (`ONE_4PI_EPS0`, CODATA 2018, its own expression) | 332.06371329919205 | + 9.9·10⁻⁹ (one ulp below OpenMM's) |
| AMBER (charges × 18.2223) | 332.05221729 | − 3.5·10⁻⁵ |
| CHARMM (chamber prmtop) | 332.0716 | + 2.4·10⁻⁵ |

## Engine codecs

Engine I/O follows the IR's protocol (`molrs::ff::ir::{Engine, EngineCodec}`), not a table
per writer: a style registered with a spec reads and writes in every engine
that can hold it with nothing else written, and is refused by name —
`IrError::NoEngineForm { engine, category, style, reason }` — by every
engine that cannot.

### LAMMPS: the style's `LammpsForm`

A `StyleSpec` carries `lammps: LammpsForm`:

- **`Positional`** (`LammpsForm::positional()`, or `positional_named("fene")`
  for another LAMMPS name) is derived from the spec: `<category>_style
  <name>` (`pair_style <name> <cutoff>`), `<category>_coeff <type> v₁ … vₙ`
  in `params` order, `mixing` as `pair_modify mix`. Each value is converted
  by its `ParamDimension` from the force field's units to the file's — `E/L^2` by
  energy/length², `E*L^6` by energy·length⁶, an angle value (exactly `A`)
  and a per-radian constant not at all — and a dimensionless integral value
  is written as an integer (LAMMPS reads multiplicities and signs with
  `inumeric`). Reading is the inverse: exactly `n` numbers, else an error
  naming the line and the missing parameter. A positional form refuses, at
  registration, a Text, Array or indexed parameter, a style parameter
  other than `cutoff` / `mixing`, and a category with no `*_style` command;
  at write, a row without a value and no default.
- **`Custom(Arc<dyn LammpsCodec>)`**: a codec of its own where the LAMMPS
  line is not positional.
- **`None`**: refused by name; an expression-only style because the
  installed LAMMPS has no `LEPTON` package.

| Built-in | LAMMPS form |
|---|---|
| `bond harmonic`, `morse`, `class2`; `angle harmonic`, `charmm`; `dihedral opls`, `multi/harmonic`; `improper harmonic`; `pair buck`, `morse` | positional |
| `dihedral periodic` | `fourier`: `m K1 n1 d1 …` (the term count, indexed parameters) |
| `dihedral nharmonic` | `N A1 … AN` |
| `dihedral charmm` | `K n d w`, `n` and `d` integers |
| `dihedral harmonic`, `improper cvff` | `K d n`, `d` = ±1 and `n` integers |
| `improper periodic` | `cvff` `K d n`, φ₀ = 0° / 180° as `d` = ±1 (it reads back as `improper cvff`, the same energy) |
| `angle class2`, `dihedral class2` | `class2` and its cross-term lines (`bb`, `ba`; `mbt`, `ebt`, `at`, `aat`, `bb13`) at zero; a data file's `BondBond Coeffs`, … sections; a non-zero cross term is refused on read |
| `pair lj/cut` | `epsilon sigma`; `shift` as `pair_modify shift yes`; only 12-6 |
| `pair lj/charmm` + `coul/charmm` | `lj/charmm/coul/charmm inner outer [inner2 outer2]`, `epsilon sigma epsilon14 sigma14` |
| `pair lj/class2` | `epsilon sigma`, mixing always `sixthpower` (LAMMPS's `init_one`) |
| `pair coul/cut`, `coul/long/pme` | the Coulomb half of `lj/cut/coul/cut` / `lj/cut/coul/long` (or of a `hybrid/overlay` with another style, the Coulomb on `* *`); `delta ≠ 0`, `dielectric ≠ 1` and stated Ewald parameters refused |
| `cmap charmm` | `fix cmap` and its grid file |
| `drude harmonic` | the DRUDE package's core-shell spring: `bond_style harmonic` `K 0` (`K = k`, both unhalved, `r0 = 0`); the data writer writes the `drudes` rows as `Bonds` rows of the same labels. `alpha` and `thole` go to a comment of the include, for the input's `pair_style thole` / `fix drude` |
| `dihedral rb`, MMFF, UFF, `thole`, `coul/tt` | none |

The reader resolves a LAMMPS style name through the registry
(`Registry::lammps_style`): the style of that name when its form writes it,
else the one style whose form does (`fourier` is `dihedral periodic`). A
run-time category, which LAMMPS has no `*_style` for, is refused by the
writer rather than dropped. Readers and writers read the process-wide
registry, or the one `with_registry` gives them.

Registering a LAMMPS form:

```rust
use molrs::ff::ir::{ParamDimension, LammpsForm, ParamSpec, StyleSpec};
use molrs::ff::style_registry::GlobalRegistry;

let fene = StyleSpec::new("bond", "fene")
    .params(vec![
        ParamSpec::new("k", "E/L^2".parse().unwrap()),
        ParamSpec::new("r0", ParamDimension::LENGTH),
        ParamSpec::new("epsilon", ParamDimension::ENERGY),
        ParamSpec::new("sigma", ParamDimension::LENGTH),
    ])
    .expression("-0.5*k*r0^2*log(1-(r/r0)^2)+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")
    .lammps(LammpsForm::positional());
GlobalRegistry.register_style(fene, None)?;
```

```python
from molpy.ff import ir, style_registry
registry = style_registry.GlobalRegistry()

FENE = ("-0.5*k*r0^2*log(1-(r/r0)^2)"
        "+step(2^(1/6)*sigma-r)*(4*epsilon*((sigma/r)^12-(sigma/r)^6)+epsilon)")
registry.register_style("bond", "fene", params={"k": "E/L^2", "r0": "L",
    "epsilon": "E", "sigma": "L"}, expression=FENE, lammps="positional")
registry.register_style("bond", "other", params={"k": "E/L^2", "r0": "L",
    "epsilon": "E", "sigma": "L"}, expression=FENE)  # no LAMMPS form yet
registry.register_engine_form("lammps", "bond", "other", "positional:fene")  # afterwards
[s.lammps for s in registry.styles("dihedral") if s.name == "periodic"]  # ['custom:fourier']
```

`register_engine_form(engine, category, style, form)` gives a style
registered without one its form; another engine is `NoEngineForm` (OpenMM
writes an expression style from its expression, GROMACS and AMBER hold
built-in styles only), a built-in `Sealed`.

### OpenMM XML: `Custom*Force` of the expression

A style with no native tag but an expression — registered, built in
(`bond morse`, `bond class2`, `angle class2`, `improper cvff`, …), or an
unregistered instance's own — is written as its category's custom force.
The parameters stay in IR units (`PerBondParameter`, …, style parameters
as `GlobalParameter`s); the expression is rewritten, exactly, through the
expression printer:

| Category | Force | Rewrite |
|---|---|---|
| bond (`r`) | `CustomBondForce` | `4.184*(E[r → 10*r])` |
| angle (`theta`) | `CustomAngleForce` | `4.184*(E)` |
| dihedral (`phi`) | `CustomTorsionForce` `<Proper>` | `phi` → `theta` |
| improper (`phi`, `chi`) | `CustomTorsionForce ordering="charmm"` `<Improper>`, the row's order (its centre first) | `phi` → `theta`, `chi` → `abs(theta)` |
| compound (a run-time category, or points in the expression) | `CustomCompoundBondForce`, built by a `<Script>` over OpenMM's bonds, angles or propers whose types match a row either way round | `distance(…)` → `10*distance(…)`; `r`, `theta`, `phi` as their functions of the points |
| pair | `CustomNonbondedForce` `bondCutoff` from `special_bonds` (`[0,0,0]` → 3, `[0,0,1]` → 2, …) | `r` → `10*r`; a bare parameter → its mixing rule of `x1`, `x2`; `q1`, `q2` → per-particle `charge1`, `charge2` |

OpenMM's ForceField XML has no compound tag (an unknown tag is ignored, its
energy lost), so the compound force is a `<Script>` OpenMM's
`createSystem` runs. Refused: a style with no expression, an indexed
parameter, a wildcard where OpenMM would re-order an improper or a
compound row, a compound category that is no chain of 2 to 4 atoms, two
custom forces stating one global parameter at different values, and for a
pair a cross row, a parameter that does not mix (`buck`, `morse`), an atom
type without a row, and 1-4 weights a `bondCutoff` cannot state. Reading a
`Custom*Force` is refused, but for the two harmonic-improper forms.

### GROMACS, AMBER

The GROMACS topology writer and the frcmod writer hold the built-in styles
they have directives (sections) for; every other style — built in, or
registered at run time, or an instance's expression — is `NoEngineForm`
("not a built-in style: GROMACS holds the built-in styles it has
directives for"). The GROMACS and prmtop readers produce built-in styles
only ([GROMACS topologies](#gromacs-topologies), [AMBER prmtop](#amber-prmtop)).

## Cross-format equivalence

One molecule per force-field family, read from its native format into the
IR and written by the core to every format that can hold it
(`ff::equivalence_check`):

| Source | Family, molecule | Native format |
|---|---|---|
| `ff14sb` | AMBER ff14SB, ACE-PHE-NME | prmtop |
| `gaff2` | GAFF2, the modXNA DMA fragment | prmtop |
| `chamber` | CHARMM36 (Urey–Bradley, CHARMM impropers, CMAP, 1-4 table), alanine dipeptide + α-D-glucose | chamber prmtop |
| `charmm36` | CHARMM36 (Urey–Bradley, harmonic impropers, CMAP, `sigma14`/`epsilon14`, an NBFIX row), ACE-ALA-NME | OpenMM XML |
| `oplsaa` | OPLS-AA (RB, geometric mixing, funct-1 impropers), ACE-ALA-ALA-NME | GROMACS `.top` |

Each source's IR is written as a LAMMPS data file and include, an OpenMM
`<ForceField>` XML (with residue templates built from the frame) and a
GROMACS topology (`write_system`). Every file written reads back into the
IR it was written from — written again it is the same file, and the core
prices it the same (`every_written_file_reads_back_as_written`) — and every
source's IR, and each format's form of it, persists through a record.

Where a format holds a field in another exact form, the core writes that form,
and prices it as the source to 10⁻¹² (`each_engine_form_prices_as_the_source`):
CHARMM's 1-4 table goes to LAMMPS as `special_bonds` 0 and one zero-`K`
`dihedral charmm` row of `w` = 1 per 1-4 pair; to OpenMM, OPLS-AA's funct-1
impropers (dihedral rows on atoms that are no bonded chain) go as the
`improper periodic` they are, and charges per atom in the residue
templates.

The core writes engine inputs; pricing them with LAMMPS, GROMACS, OpenMM or
sander is an out-of-tree script, and no engine number is kept here.

## Completeness

Every style the core registers a kernel for, and every field-level setting,
against every engine format the core reads or writes and the record (molrec
v2): **✓** exact — priced, read or written as the IR means it, held by the
tests `ff::completeness` names (a numbered note bounds a ✓ or states the
condition it holds under); **refused** — an error naming the style or
setting, never a silent drop or an approximation; **untested** — the format
holds it, but no test here holds it exact (its only check is pricing by the
engine itself, out of tree); **—** the format has no such thing. The table is generated: `ff::completeness` holds it with the tests
behind every cell, and fails when a registered kernel has no row, a row names
no kernel, a ✓ cites no test, or this table differs from its rendering (the
failure prints the table to put between the markers).

<!-- completeness:begin (generated by ff::completeness; do not edit) -->
| Style or setting | kernel | LAMMPS read | LAMMPS write | OpenMM read | OpenMM write | GROMACS read | GROMACS write | prmtop read | molrec v2 |
|---|---|---|---|---|---|---|---|---|---|
| `bond harmonic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | untested | ✓ |
| `bond morse` | ✓ | ✓ | ✓ | refused [1] | untested | ✓ | ✓ | — | ✓ |
| `bond class2` | ✓ | ✓ [2] | ✓ [2] | — | ✓ [3] | — | refused [4] | — | ✓ |
| `bond mmff_bond` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `bond uff_bond` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `angle harmonic` | ✓ | ✓ | ✓ | untested | ✓ | ✓ | ✓ | untested | ✓ |
| `angle charmm` | ✓ | ✓ | ✓ | ✓ [6] | ✓ | ✓ | ✓ | ✓ | ✓ |
| `angle class2` | ✓ | ✓ [2] | ✓ [2] | — | ✓ [3] | — | refused [4] | — | ✓ |
| `angle mmff_angle` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `angle mmff_stbn` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `angle uff_angle` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `dihedral periodic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `dihedral charmm` | ✓ | ✓ | ✓ | — | ✓ [7] | — | ✓ [8] | — | ✓ |
| `dihedral opls` | ✓ | ✓ | ✓ | ✓ | ✓ [9] | ✓ | ✓ | — | ✓ |
| `dihedral multi/harmonic` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — | ✓ |
| `dihedral nharmonic` | ✓ | ✓ | ✓ | ✓ | ✓ [10] | ✓ | ✓ [11] | — | ✓ |
| `dihedral harmonic` | ✓ | ✓ | ✓ | — | ✓ | — | ✓ | — | ✓ |
| `dihedral class2` | ✓ | ✓ [2] | ✓ [2] | — | ✓ | — | ✓ | — | ✓ |
| `dihedral mmff_torsion` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `dihedral uff_torsion` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `improper harmonic` | ✓ | ✓ | ✓ [12] | ✓ [13] | ✓ [14] | ✓ [15] | ✓ [16] | ✓ | ✓ |
| `improper cvff` | ✓ | ✓ | ✓ | — | ✓ [17] | — | ✓ | — | ✓ |
| `improper periodic` | ✓ | — | ✓ [18] | ✓ | ✓ [19] | ✓ | ✓ | ✓ | ✓ |
| `improper mmff_oop` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `improper uff_inversion` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `cmap charmm` | untested | ✓ | ✓ | ✓ [20] | ✓ [21] | ✓ | ✓ | ✓ | ✓ |
| `pair lj/cut` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pair lj/charmm` | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| `pair coul/cut` | ✓ | ✓ | ✓ [22] | untested | untested | ✓ | ✓ [23] | ✓ | ✓ |
| `pair coul/charmm` | ✓ | ✓ | untested | untested | untested | untested | untested | untested | ✓ |
| `pair coul/long/pme` | ✓ | ✓ [24] | ✓ [25] | — | ✓ [26] | — | ✓ [27] | — | ✓ |
| `pair lj/class2` | ✓ | ✓ [28] | ✓ [28] | — | refused [29] | — | refused [4] | — | ✓ |
| `pair buck` | ✓ | ✓ | ✓ | — | refused [30] | — | refused [4] | — | ✓ |
| `pair morse` | ✓ | ✓ | ✓ | — | refused [30] | — | refused [4] | — | ✓ |
| `pair thole` | ✓ | refused [31] | refused [31] | — | refused [31] | — | refused [31] | — | ✓ |
| `pair coul/tt` | ✓ | refused [31] | refused [31] | — | refused [31] | — | refused [31] | — | ✓ |
| `pair mmff_vdw` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `pair uff_lj` | ✓ | — | refused [5] | — | refused [5] | — | refused [5] | — | ✓ |
| `special_bonds` | ✓ | ✓ | ✓ | untested | ✓ [32] | ✓ | ✓ [32] | ✓ | ✓ |
| `per-pair overrides (pairs epsilon, sigma, lj_scale, charge_product, coul_scale)` | ✓ | — | refused [33] | — | — | ✓ | ✓ | untested | ✓ |
| `lj/charmm one_four = "epsilon14"` | ✓ | — | ✓ [34] | untested | ✓ | untested | ✓ | untested | ✓ |
| `mixing arithmetic` | ✓ | ✓ | ✓ | untested | untested | ✓ | ✓ | ✓ | ✓ |
| `mixing geometric` | ✓ | ✓ | ✓ | untested | ✓ [35] | ✓ | ✓ | — | ✓ |
| `mixing sixthpower` | ✓ | ✓ | ✓ | — | refused [36] | — | refused [37] | — | ✓ |
| `cross rows (NBFIX)` | ✓ | ✓ | ✓ | untested | ✓ [38] | ✓ | ✓ | ✓ | ✓ |
| `lj/cut shift` | ✓ | ✓ | ✓ | — | refused [39] | — | refused [40] | — | ✓ |
| `lj/cut n, m (Mie)` | ✓ | refused [41] | refused [42] | — | refused [43] | — | refused [44] | — | ✓ |
| `Coulomb constant` | ✓ | ✓ | — | untested | — | untested | — | ✓ | ✓ |
| `cutoff, inner (switch)` | ✓ | ✓ | ✓ | — | — | — | — | — | ✓ |
| `units presets (real, metal, lj)` | ✓ | ✓ | ✓ | — | refused [45] | — | refused [45] | — | ✓ |
| `run-time style with a positional LAMMPS form` | untested | ✓ | ✓ | refused [46] | ✓ | — | refused [47] | — | ✓ |
| `expression style without a LAMMPS form` | untested | refused [48] | refused [49] | refused [46] | ✓ | — | refused [47] | — | ✓ |
| `run-time compound category (Urey-Bradley)` | untested | — | refused [50] | — | untested | — | refused [47] | — | ✓ |

1. OpenMM has a Morse bond only as a CustomBondForce, which the reader refuses
2. through its codec (cross-term lines at zero, a non-zero one refused); not run by LAMMPS: the installed lmp has no CLASS2 package
3. the CustomBondForce / CustomAngleForce of its expression, rewritten exactly (held by OpenMM on bond fene, bond morse, angle class2)
4. GROMACS has no directive of its form (NoEngineForm)
5. the core's own definition (a typifier's), no engine style
6. a wildcard Urey–Bradley row is refused
7. w = 0 (a periodic term); w ≠ 0 refused
8. w = 0 (funct 9); w > 0 refused
9. as RB, which reads back as multi/harmonic: the same series, constant included
10. N ≤ 6 (RB's C0 … C5); above refused
11. N ≤ 6; above refused
12. LAMMPS clamps sin χ at 0.001 in its force: within 0.057° of planar its forces are not its energy's gradient
13. CustomTorsionForce k(θ−θ0)² at θ0 = 0, or k(|θ|−θ0)²; another signed θ0 refused
14. a wildcard endpoint refused (OpenMM would re-order the atoms)
15. funct 2 at ξ0 ∈ {0°, 180°}; another ξ0 refused
16. chi0 ∈ {0°, 180°}; another refused
17. the CustomTorsionForce of its expression with ordering="charmm", which prices the row's dihedral, its centre first
18. as cvff, at a phase of 0° or 180°; another refused
19. OpenMM orders the two outer atoms it finds first by element and index (its AMBER rule); a system whose stored order differs prices another dihedral
20. odd N refused; OpenMM takes its node slopes from periodic splines (≤ 10⁻⁷ kcal/mol off the IR's)
21. as OpenMM read: its interpolation ≤ 10⁻⁷ kcal/mol off the IR's
22. delta = 0 and dielectric = 1; the Coulomb constant is LAMMPS's own
23. the Coulomb constant is GROMACS's own
24. lj/cut/coul/long: cutoff and constant; its kspace_style states an accuracy, not an alpha, so pricing is refused until the Ewald parameters are stated
25. the real-space lj/cut/coul/long; stated Ewald parameters refused (the core's smooth PME is not LAMMPS's PPPM)
26. the charges; PME / Ewald is a createSystem argument; stated Ewald parameters refused
27. the charges; coulombtype = PME is an .mdp setting; stated Ewald parameters refused
28. through its codec, mixing sixthpower as LAMMPS's always does; not run by LAMMPS: the installed lmp has no CLASS2 package
29. its kernel prices a row per pair (cross rows), which a CustomNonbondedForce has not
30. its parameters do not mix, so a pair's are a cross row's, which a CustomNonbondedForce has not
31. polarizable (Drude) screening, outside the Class-I IR
32. [0, 0, s]; another refused
33. LAMMPS has no per-pair 1-4 parameters
34. refused as such; its exact LAMMPS form is special_bonds 0 and one zero-K dihedral charmm row of w = 1 per 1-4 pair
35. foyer's combining_rule and the <Script> that applies it (OpenMM's own app.ForceField ignores the attribute); with cross rows refused
36. OpenMM mixes arithmetically
37. no GROMACS comb-rule
38. a cross row with 1-4 parameters of its own refused (OpenMM prices an NBFIX 1-4 pair with the NBFIX row)
39. OpenMM's NonbondedForce has no shifted Lennard-Jones
40. a modifier is an .mdp setting
41. pair_style mie/cut is not read
42. pair_style mie/cut is not written
43. OpenMM's Lennard-Jones is 12-6
44. GROMACS's Lennard-Jones is 12-6
45. the conversions are from real
46. reading a Custom*Force is refused, but for the two harmonic impropers
47. not a built-in style (NoEngineForm)
48. no registered style reads its name (the installed LAMMPS has no LEPTON package)
49. no LAMMPS form: an expression needs LAMMPS's LEPTON package (NoEngineForm)
50. no `*_style` command (NoEngineForm)
<!-- completeness:end -->

The styles outside the Class-I IR (Thole, Tang–Toennies) and the core's own
typifier styles (MMFF94, UFF) are priced and persisted, and no engine
reader or writer maps them; the Class II styles, `pair buck` and `pair
morse` are LAMMPS's, read and written as the table says. Every Class-I style of molrec's registry is a style of the
same name in the core, with a native kernel, but two: `dihedral rb`, priced by its
expression (no reader produces it: each reads Ryckaert–Bellemans as the
`multi/harmonic` / `nharmonic` polynomial it is, constant included), and
`improper trefoil` (the SMIRNOFF average over three orderings), which the core
does not register and `compile` refuses by name (`NoKernel`; the OpenMM
reader refuses `ordering="smirnoff"`).

## How this is checked

- Every reader and writer against the IR: [Cross-format
  equivalence](#cross-format-equivalence) and [Completeness](#completeness).
- Each style has a hand-value test against the LAMMPS manual's formula.
- `ff::ir_invariance` holds the energies of GAFF-, OPLS-AA-,
  MMFF94- and UFF-typed acetanilide and of a GROMACS-, OpenMM- and
  LAMMPS-read hand molecule to pinned reference values, term by term, at
  1e-12 relative. The OpenMM improper equals the GROMACS-read value of the
  same improper and the hand value of OpenMM's formula; the OpenMM- and
  GROMACS-read Coulomb terms carry the engine's own constant.
- The 1-4 mechanisms against each other (`ff::one_four_check`):
  `special_bonds` ½ equals per-pair scales ½ equals per-pair parameters
  ε/2, qᵢqⱼ/2; `w` = 1 equals per-pair rows of ε₁₄, σ₁₄; an override cell
  beats the weight and a null cell keeps it; forces are the gradient inside
  the switches; `compile` equals `compile_typed`.
- Both compile doors truncate at the cutoff: `ff::style_registry::builtin_conformance`
  holds every built-in pair style and its expression twin to each other at
  both doors and `compile` to `compile_typed` (1e-12) on pairs straddling
  the cutoff (and CHARMM's switch); `ff::compile` holds
  `lj/cut` (shifted and not) and `coul/cut` to LAMMPS's formula by hand,
  and those with `lj/class2`, `buck` and `morse` to `compile_typed`, with a
  1-4 pair and a regular one on each side of the cutoff.
- GROMACS topologies (`io::gromacs::top_reader::write_read_check`): the
  charmm27, amber99sb-ildn and oplsaa fixtures written back by the GROMACS
  writer read as the same force field, and per-pair `[ pairs ]` parameters
  are refused by the LAMMPS writer by name.
- OpenMM XML (`ff::openmm_check`): CHARMM36, AMBER ff14SB and OPLS-AA
  fixtures read → write → read as the identity, and CHARMM's
  `dihedral charmm` form prices as its `one_four = "epsilon14"` form.
