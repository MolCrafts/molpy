//! GROMACS topology reader: force-field directives, and whole systems.
//!
//! Two products from one parser:
//!
//! - [`ForceFieldReader::read`] reads the force-field **directives** of a
//!   topology (`forcefield.itp` with its includes, or a `.top`'s directive
//!   sections) into a [`ForceField`];
//! - [`GromacsTopForcefieldReader::read_system`] reads a whole `.top` —
//!   directives **and** molecules — into the [`ForceField`] and a typed
//!   [`Frame`] (see [`system`](self#whole-systems)).
//!
//! Both take a [`Source`]: a path, whose `#include`s resolve against its
//! directory and then the include directories, or a whole file's bytes, whose
//! relative `#include`s resolve against the include directories only.
//!
//! The file speaks nm, kJ/mol, degrees and e; the force field is the molrs
//! force-field IR, whose definitions follow LAMMPS (`real`: Å, kcal/mol,
//! degrees for angle-valued parameters, e; LAMMPS's factors — no hidden ½).
//! Every conversion happens here, at the boundary.
//!
//! # Directives read
//!
//! - **`[ defaults ]`** `nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ`. Requires
//!   nbfunc 1 (Lennard-Jones). comb-rule 2 (σᵢⱼ = ½(σᵢ+σⱼ), εᵢⱼ = √(εᵢεⱼ))
//!   is `mixing` `arithmetic`, comb-rule 3 (σᵢⱼ = √(σᵢσⱼ), εᵢⱼ = √(εᵢεⱼ))
//!   `geometric`, declared on the Lennard-Jones pair style. The special-bond
//!   weights are `lj [0, 0, fudgeLJ]` and `coul [0, 0, fudgeQQ]` — GROMACS
//!   excludes 1-2 and 1-3 pairs and scales the generated 1-4 pairs. With
//!   gen-pairs `no` GROMACS generates no 1-4 Lennard-Jones parameters (every
//!   1-4 pair takes a `[ pairtypes ]` row or its own parameters, at full
//!   weight) and the LJ weight is `1`; fudgeLJ is then unused, as in GROMACS.
//! - **`[ atomtypes ]`** `name [bond_type] [at.num] mass charge ptype V W`.
//!   Columns resolve from the right: the last five are `mass charge ptype V W`
//!   (`ptype` one of `A S V D B`); the one to three leading tokens are `name`,
//!   an optional `bond_type` and an optional integer `at.num`. Each row defines
//!   - an `atom/full` type with `mass` (amu), `charge` (e), `atomic_number`
//!     (when present) and string `ptype` and `class` (= `bond_type`, when
//!     present);
//!   - a self row of the Lennard-Jones pair style with `sigma` = V·10 (Å) and
//!     `epsilon` = W/4.184 (kcal/mol) — V/W are σ/ε under comb-rules 2 and 3,
//!     so `[ atomtypes ]` requires `[ defaults ]`;
//!   - the Coulomb pair style, with `coulomb` = [`GROMACS_ONE_4PI_EPS0`](crate::core::constants::GROMACS_ONE_4PI_EPS0) in kcal·Å (GROMACS's
//!     own constant) and `dielectric` = 1 (vacuum).
//! - **`[ nonbond_params ]`** `i j func V W`, func 1: an explicit cross row of
//!   the Lennard-Jones style between the atom types `i` and `j` (σ, ε as for
//!   `[ atomtypes ]`), which the kernels use in place of the comb-rule. The row
//!   is stored with its two types in byte order and named
//!   [`TypeName::pair`](molrs::core::TypeName::pair) of them, so
//!   `j i` restating `i j` is the same row and a different one is an error.
//! - **`[ pairtypes ]`** `i j func V W`, func 1: the Lennard-Jones parameters
//!   of the 1-4 pairs of atom types `i`, `j` (see [1-4 pairs](#1-4-pairs)).
//! - **`[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]`**
//!   `labels… funct params…`, keyed by the row's labels (GROMACS bond types,
//!   the `class` of the atom types). GROMACS `X` becomes the empty-endpoint
//!   wildcard, so `X CT CT X` is the type `-CT-CT-`. The 2-name dihedraltypes
//!   form `a b funct …` is GROMACS's shorthand for `a X X b` (funct 2) and
//!   `X a b X` (every other funct).
//! - **`[ cmaptypes ]`** `a b c d e 1 N N v₁ … v_{N²}` (lines joined at a
//!   trailing `\`): a `cmap/charmm` type on the five labels, `grid` the N×N
//!   map in kcal/mol. GROMACS lists the map φ-major from −180° with step
//!   360°/N (`cmap_setup_grid_index`), which is the IR's layout, so value
//!   `i·N + j` is `grid[i][j]` unchanged.
//!
//! | Directive, funct | GROMACS form | molrs style | Conversion |
//! |---|---|---|---|
//! | bondtypes 1 | ½k_b(r−b₀)² | `bond/harmonic` {`r0`, `k`} | r0 = 10·b₀ Å; k = k_b/(2·418.4) kcal/mol/Å² |
//! | bondtypes 3 | D[1−e^{−β(r−b₀)}]² | `bond/morse` {`d0`, `alpha`, `r0`} | d0 = D/4.184; alpha = β/10 Å⁻¹; r0 = 10·b₀ |
//! | angletypes 1 | ½k_θ(θ−θ₀)² | `angle/harmonic` {`theta0`, `k`} | θ₀ as written; k = k_θ/(2·4.184) kcal/mol/rad² |
//! | angletypes 5 | ½k_θ(θ−θ₀)² + ½k_UB(r₁₃−r₁₃⁰)² | `angle/charmm` {`k`, `theta0`, `k_ub`, `r_ub`} | k = k_θ/(2·4.184); k_ub = k_UB/(2·418.4); r_ub = 10·r₁₃⁰ |
//! | dihedraltypes 1 | k[1+cos(nφ−φ_s)] | `dihedral/periodic` {`k`, `periodicity`, `phase`} | k/4.184; φ_s as written |
//! | dihedraltypes 9 | Σ of the consecutive rows with equal labels | `dihedral/periodic` {`k<m>`, `periodicity<m>`, `phase<m>`} (one row: as funct 1) | each term as funct 1, in file order |
//! | dihedraltypes 2 | ½k_ξ(ξ−ξ₀)², ξ signed | `improper/harmonic` {`k`, `chi0`} | k = k_ξ/(2·4.184); ξ₀ ∈ {0°, 180°} |
//! | dihedraltypes 3 | Σₙ₌₀⁵ Cₙ cosⁿ(φ−180°) | `dihedral/multi/harmonic` {`a1`..`a5`}, `dihedral/nharmonic` {`a1`..`a6`} when C₅ ≠ 0 | aₙ₊₁ = (−1)ⁿ Cₙ/4.184, constant included |
//! | dihedraltypes 4 | k[1+cos(nφ−φ_s)] | `improper/periodic` {`k`, `periodicity`, `phase`} | as funct 1 |
//! | dihedraltypes 5 | ½[C₁(1+cos φ) + C₂(1−cos 2φ) + C₃(1+cos 3φ) + C₄(1−cos 4φ)] | `dihedral/opls` {`k1`..`k4`} | kₙ = Cₙ/4.184 |
//! | cmaptypes 1 | CHARMM bicubic map | `cmap/charmm` {`grid`} | grid/4.184 |
//!
//! Every conversion is exact: RB is the polynomial `nharmonic` /
//! `multi/harmonic` (the torsion algebra, `ff::ir::torsion`), so no
//! RB row is refused for its C₅ or its ΣCₙ. ½k_ξ(ξ−ξ₀)² is signed and
//! molrs's `improper/harmonic` is `K(χ−χ₀)²` with χ = |φ|: they agree for ξ₀ =
//! 0° and ξ₀ = 180° and for no other ξ₀, which is refused.
//!
//! Improper rows keep their atom order: GROMACS prices the dihedral of the
//! listed order, as molrs does (an AMBER port lists the centre third, a CHARMM
//! port first).
//!
//! Rows repeated across sections or includes follow the conflict rule of
//! [`Style::def_type`](crate::ff::forcefield::Style::def_type): equal
//! parameters are one type, different ones an error. When two styles of one
//! category would name a type alike (funct 9 and funct 3 rows on the same four
//! labels), both are qualified with their function code (`-CT-CT-@3`).
//!
//! # 1-4 pairs
//!
//! GROMACS prices a 1-4 pair (a `[ pairs ]` row) at the `[ pairtypes ]` row of
//! its two atom types, at full weight, or — gen-pairs `yes` and no such row —
//! at the comb-rule (or `[ nonbond_params ]`) parameters scaled by fudgeLJ;
//! its Coulomb is scaled by fudgeQQ. The IR states the same thing with
//! LAMMPS's `lj/charmm` parameters: when a `[ pairtypes ]` row prices a pair
//! differently from that generated value, the Lennard-Jones style is
//! `lj/charmm` (with `coul/charmm`), declared `one_four = "epsilon14"`, and
//! each type pair's `epsilon14` / `sigma14` — a self row's, an explicit cross
//! row's, or the mix of the two self rows' — is the 1-4 pair's parameters
//! divided by the 1-4 weight (ε₁₄ = ε_pairtype / fudgeLJ; σ₁₄ = σ_pairtype),
//! so `special_bonds` × LJ(ε₁₄, σ₁₄) is GROMACS's 1-4 energy for every type
//! pair. Cross rows are written exactly where the mix of the self rows would
//! differ (to 10⁻¹² relative), with the pair's regular parameters beside
//! them. Without such a `[ pairtypes ]` row the styles are `lj/cut` and
//! `coul/cut`, as for AMBER and OPLS-AA. gen-pairs `yes` with fudgeLJ 0 and a
//! `[ pairtypes ]` row is refused: one weight cannot price the pairtype pairs
//! at 1 and the generated ones at 0.
//!
//! `lj/charmm` switches between `inner` and `cutoff`, run settings GROMACS
//! keeps in the `.mdp`: the reader declares neither, and a caller sets both
//! before compiling.
//!
//! # Whole systems
//!
//! [`GromacsTopForcefieldReader::read_system`] reads the molecule sections too and
//! returns the force field with a typed [`Frame`] (0-based atom indices):
//!
//! | Section | Frame |
//! |---|---|
//! | `[ atoms ]` | `atoms`: `type`, `charge`, `mass` (the atom type's when the row omits them), `name`, `res_id`, `res_name`, `mol_id` (1-based molecule instance) |
//! | `[ bonds ]` funct 1, 3 | `bonds` |
//! | `[ angles ]` funct 1, 5 | `angles` |
//! | `[ dihedrals ]` funct 1, 9, 3, 5 | `dihedrals` |
//! | `[ dihedrals ]` funct 2, 4 | `impropers` |
//! | `[ cmap ]` funct 1 | `cmaps` |
//! | `[ constraints ]` funct 1, 2; `[ settles ]` | `constraints` (`r0`, Å) |
//! | nrexcl, `[ exclusions ]` | `exclusions` |
//! | `[ pairs ]`, nrexcl | `pairs`: every pair GROMACS prices, `is_14` on the `[ pairs ]` rows |
//! | `[ molecules ]` | the molecule types repeated in order |
//!
//! Each relation row's `type` is the force-field type GROMACS's own lookup
//! picks: bonds, angles, Fourier dihedrals and cmaps match their atoms' bond
//! types exactly (cmaps forward only, the others either way), the other
//! dihedrals take the first row with the most non-wildcard matches, either
//! way — the funct-9 terms of that row come with it. A row with parameters
//! of its own defines a type of its own, named by its labels qualified
//! `@gmx_<n>`. `#define` macros are expanded in every row, so an OPLS-AA
//! `improper_Z_N_X_Y` row is a funct-1 row with those parameters.
//!
//! A `[ pairs ]` row with parameters is priced by them, at full Lennard-Jones
//! weight and fudgeQQ Coulomb (funct 1), or at its own fudgeQQ and charges
//! (funct 2): the per-pair override columns (`epsilon`, `sigma`, `lj_scale`,
//! `coul_scale`, `charge_product`) of `pairs`. A row without parameters is
//! priced by the force field (`special_bonds` and `epsilon14` above); with
//! gen-pairs `no` it needs a `[ pairtypes ]` row, as in GROMACS.
//!
//! `pairs` lists every pair GROMACS prices: per molecule, the pairs beyond
//! `nrexcl` chemical bonds (bonds funct 1 and 3, constraints funct 1) and not
//! in `[ exclusions ]`, and the `[ pairs ]` rows (which must be excluded, else
//! GROMACS would price them twice); and every pair of two molecules, up to
//! `MAX_ATOMS_FOR_A_FULL_PAIR_LIST` atoms (above it the list is the
//! intramolecular one, and a neighbour list — `compile_typed` — finds the
//! others). LAMMPS derives its exclusions and its
//! 1-4 list from the bonds alone; a topology whose `[ pairs ]` and exclusions
//! are those of `nrexcl` 3 (every `pdb2gmx` topology) is priced the same by
//! both.
//!
//! # Refusals
//!
//! Anything this reader does not model is an `Err` naming it, never a silent
//! drop:
//!
//! - function codes outside the tables above (bond 2, 4, 5–10; angle 2, 3, 4,
//!   6, 8, 10; dihedral 8, 10, 11; pair funct other than 1 and 2; constraint
//!   other than 1 and 2), dihedral 2 with ξ₀ ∉ {0°, 180°}, a non-integer
//!   multiplicity, a row with the wrong parameter count, a cmap grid that is
//!   not square, a row on the atoms of an earlier row of its function-code
//!   table (either way round) with other parameters (GROMACS warns and
//!   applies it over the first);
//! - comb-rule 1 (V/W are C6/C12), nbfunc 2 (Buckingham); a `[ nonbond_params ]`
//!   or `[ pairtypes ]` row with a func other than 1 or on an undefined atom
//!   type;
//! - `[ constrainttypes ]` when reading a force field (no force-field category
//!   holds a constraint; [`read_system`](GromacsTopForcefieldReader::read_system)
//!   reads it for `[ constraints ]`), `[ implicit_genborn_params ]`, and any
//!   unknown section;
//! - every molecule section when reading a force field (read the topology with
//!   `read_system`); in a system, the sections with no IR form: virtual sites,
//!   restraints, polarization, `[ pairs_nb ]`, `[ intermolecular_interactions ]`;
//! - in a system: B-state (free-energy) columns, an atom type or molecule type
//!   that is not defined, a row no type matches, atoms numbered out of order.
//!
//! A section the caller names with
//! [`GromacsTopForcefieldReader::with_skipped_directive`] is read past instead, rows
//! and all.
//!
//! # Preprocessor
//!
//! - `#include "file"` is followed when [`GromacsTopForcefieldReader::with_include`]
//!   is `true` (and ignored otherwise). It resolves relative to the including
//!   file, then against each [`with_include_dir`](GromacsTopForcefieldReader::with_include_dir)
//!   directory (GROMACS's share/top for `#include "charmm27.ff/forcefield.itp"`);
//!   a file already read is not read again.
//! - `#define NAME [body]` and `#undef NAME` maintain a define set; a row's
//!   whitespace-separated token that names a define is replaced by its body.
//! - `#ifdef` / `#ifndef` / `#else` / `#endif` select lines against that set,
//!   and nest.
//! - A line ending in `\` continues on the next.
//! - Text before the first `[ section ]` is not read (GROMACS ignores it too;
//!   charmm27's `forcefield.itp` opens with a banner).
//! - `#if`, `#elif` and any other directive are an `Err`.

mod system;

#[cfg(test)]
mod engine_check;

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::{Path, PathBuf};

use ndarray::ArrayD;

use crate::core::constants::VACUUM_DIELECTRIC;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::CMAP_GRID;
use crate::ff::ir::CombiningRule;
use crate::ff::ir::same_lj;
use crate::ff::ir::torsion::rb_polynomial;
use crate::ff::ir::{Params, SpecialBonds};
use crate::io::reader::ForceFieldReader;
use crate::io::{Source, invalid_data};
use molrs::core::Frame;
use molrs::core::TypeName;

use crate::core::unit_factors::{KCAL_ANGSTROM2_TO_KJ_NM2, KCAL_TO_KJ, NM_TO_ANGSTROM};

/// A Lennard-Jones `(ε, σ)` in molrs units.
type Lj = (f64, f64);
/// Two atom types, in byte order.
type TypePair = (String, String);
/// Lennard-Jones parameters per type pair (`[ nonbond_params ]`, `[ pairtypes ]`).
type CrossTable = BTreeMap<TypePair, Lj>;

/// Force-field directives: read in both modes.
const DIRECTIVES: &[&str] = &[
    "defaults",
    "atomtypes",
    "nonbond_params",
    "pairtypes",
    "bondtypes",
    "angletypes",
    "dihedraltypes",
    "cmaptypes",
];

/// Sections only [`GromacsTopForcefieldReader::read_system`] reads.
const SYSTEM_SECTIONS: &[&str] = &[
    "constrainttypes",
    "moleculetype",
    "atoms",
    "bonds",
    "pairs",
    "angles",
    "dihedrals",
    "cmap",
    "exclusions",
    "constraints",
    "settles",
    "system",
    "molecules",
];

/// Molecule sections GROMACS defines that have no form in the IR.
const NO_IR_FORM: &[&str] = &[
    "pairs_nb",
    "virtual_sites1",
    "virtual_sites2",
    "virtual_sites3",
    "virtual_sites4",
    "virtual_sitesn",
    "position_restraints",
    "distance_restraints",
    "dihedral_restraints",
    "orientation_restraints",
    "angle_restraints",
    "angle_restraints_z",
    "polarization",
    "water_polarization",
    "thole_polarization",
    "intermolecular_interactions",
    "implicit_genborn_params",
];

/// Reader for GROMACS topologies.
///
/// Configure with the builders, then read the force-field directives with
/// [`ForceFieldReader::read`], or a whole system with
/// [`read_system`](Self::read_system). The supported directives,
/// conversions and refusals are listed in the module documentation.
///
/// # Examples
///
/// ```
/// use molrs::io::{reader::ForceFieldReader, gromacs::GromacsTopForcefieldReader};
///
/// let text = "\
/// [ defaults ]
/// 1  3  yes  0.5  0.5
/// [ atomtypes ]
/// opls_135  CT  6  12.011  -0.18  A  0.35  0.276144
/// [ bondtypes ]
/// CT  HC  1  0.10900  284512.0
/// ";
/// let ff = GromacsTopForcefieldReader::new().read(text.as_bytes())?;
///
/// // comb-rule 3 is geometric mixing, declared on the lj/cut style.
/// let lj = ff.get_style("pair", "lj/cut").expect("lj/cut style");
/// assert_eq!(lj.params().get("mixing").and_then(|v| v.as_text()), Some("geometric"));
///
/// // b₀ = 0.109 nm is r0 = 1.09 Å.
/// let bonds = ff.get_bondtypes();
/// let r0 = bonds[0].params.get("r0").and_then(|v| v.as_num()).expect("r0");
/// assert!((r0 - 1.09).abs() < 1e-12);
/// # Ok::<(), std::io::Error>(())
/// ```
#[derive(Debug, Clone, Default)]
pub struct GromacsTopForcefieldReader {
    include: bool,
    include_dirs: Vec<PathBuf>,
    skipped: HashSet<String>,
}

impl GromacsTopForcefieldReader {
    /// A reader that ignores `#include` and skips no section.
    pub fn new() -> Self {
        Self::default()
    }

    /// Follow `#include` directives, resolved relative to the including file
    /// and then against the [`with_include_dir`](Self::with_include_dir)
    /// directories (default `false`: they are ignored).
    pub fn with_include(mut self, include: bool) -> Self {
        self.include = include;
        self
    }

    /// Resolve an `#include` the including file's directory does not hold
    /// against `dir` too (GROMACS's `-I`, or its `share/top` for
    /// `#include "charmm27.ff/forcefield.itp"`). Directories are tried in the
    /// order they are added. Has no effect unless `#include` is followed
    /// ([`with_include`](Self::with_include)).
    pub fn with_include_dir(mut self, dir: impl Into<PathBuf>) -> Self {
        self.include_dirs.push(dir.into());
        self
    }

    /// Read past every `[ name ]` section instead of refusing it. `name` is the
    /// directive name without brackets (`"constrainttypes"`, `"atoms"`);
    /// section names are case-insensitive.
    pub fn with_skipped_directive(mut self, name: &str) -> Self {
        self.skipped.insert(name.trim().to_ascii_lowercase());
        self
    }

    /// Read the whole topology `src` holds — its directives and its
    /// molecules — into the force field and a typed frame (module docs,
    /// "Whole systems").
    ///
    /// # Errors
    ///
    /// The OS error of a path that cannot be read;
    /// [`InvalidData`](std::io::ErrorKind::InvalidData) for every refusal of
    /// the module documentation, naming the file, line and row.
    pub fn read_system<'a>(
        &self,
        src: impl Into<Source<'a>>,
    ) -> std::io::Result<(ForceField, Frame)> {
        let scan = self.scan_source(&src.into(), true)?;
        self.system(scan).map_err(invalid_data)
    }

    /// The force field and frame of a preprocessed whole topology.
    fn system(&self, scan: Scan) -> Result<(ForceField, Frame), String> {
        let mut directives = scan.directives()?;
        let frame = system::build(&scan.rows, &mut directives)?;
        Ok((directives.ff, frame))
    }

    /// Whether the rows of `section` are read (`Ok(true)`) or skipped
    /// (`Ok(false)`); `Err` naming the section when it is refused. `system`:
    /// reading a whole topology.
    fn admits(&self, section: &str, system: bool) -> Result<bool, String> {
        if self.skipped.contains(section) {
            return Ok(false);
        }
        if DIRECTIVES.contains(&section) {
            return Ok(true);
        }
        if SYSTEM_SECTIONS.contains(&section) {
            if system {
                return Ok(true);
            }
            return Err(if section == "constrainttypes" {
                "[ constrainttypes ] has no force-field category (a constraint is not an \
                 energy term): read_system reads it for [ constraints ]; skip it here with \
                 with_skipped_directive(\"constrainttypes\")"
                    .to_owned()
            } else {
                format!(
                    "[ {section} ] is topology, not a force-field directive: read the whole \
                     topology with GromacsTopForcefieldReader::read_system, or skip it with \
                     with_skipped_directive(\"{section}\")"
                )
            });
        }
        if NO_IR_FORM.contains(&section) {
            return Err(format!(
                "[ {section} ] has no form in the molrs force-field IR; skip it with \
                 with_skipped_directive(\"{section}\")"
            ));
        }
        Err(format!(
            "[ {section} ] is not a GROMACS section this reader knows; skip it with \
             with_skipped_directive(\"{section}\")"
        ))
    }

    /// Preprocess the topology `src` holds: a path names itself in errors
    /// and resolves `#include` against its directory; bytes are `<bytes>`.
    fn scan_source(&self, src: &Source<'_>, system: bool) -> std::io::Result<Scan> {
        let text = src.text()?;
        let mut scan = Scan::default();
        let (origin, dir) = match src.path() {
            Some(p) => {
                scan.visited.insert(p.to_path_buf());
                (
                    p.display().to_string(),
                    Some(p.parent().unwrap_or(Path::new("."))),
                )
            }
            None => ("<bytes>".to_owned(), None),
        };
        self.scan(&text, &origin, dir, system, &mut scan)
            .map_err(invalid_data)?;
        Ok(scan)
    }

    /// Preprocess `text` (from `origin`, in directory `dir`) into `scan`.
    fn scan(
        &self,
        text: &str,
        origin: &str,
        dir: Option<&Path>,
        system: bool,
        scan: &mut Scan,
    ) -> Result<(), String> {
        let mut conds: Vec<Cond> = Vec::new();
        let lines: Vec<&str> = text.lines().collect();
        let mut next = 0;
        while next < lines.len() {
            let first = next;
            // One logical line: a trailing `\` continues it (GROMACS joins
            // before it strips the comment).
            let mut joined = String::new();
            loop {
                let raw = lines[next];
                next += 1;
                if let Some(body) = raw.trim_end().strip_suffix('\\') {
                    joined.push_str(body);
                    joined.push(' ');
                    if next < lines.len() {
                        continue;
                    }
                } else {
                    joined.push_str(raw);
                }
                break;
            }
            let at = format!("{origin}:{}", first + 1);
            let line = joined.split(';').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let active = conds.last().is_none_or(Cond::active);
            if let Some(rest) = line.strip_prefix('#') {
                let rest = rest.trim_start();
                let (name, arg) = rest
                    .split_once(char::is_whitespace)
                    .map_or((rest, ""), |(n, a)| (n, a.trim()));
                let symbol = arg.split_whitespace().next();
                match name {
                    "ifdef" | "ifndef" => {
                        let symbol = symbol.ok_or_else(|| format!("{at}: #{name} needs a name"))?;
                        let defined = scan.defines.contains_key(symbol);
                        conds.push(Cond {
                            parent: active,
                            taken: if name == "ifdef" { defined } else { !defined },
                            seen_else: false,
                        });
                    }
                    "else" => {
                        let cond = conds
                            .last_mut()
                            .ok_or_else(|| format!("{at}: #else without #ifdef / #ifndef"))?;
                        if cond.seen_else {
                            return Err(format!("{at}: second #else in one conditional"));
                        }
                        cond.taken = !cond.taken;
                        cond.seen_else = true;
                    }
                    "endif" => {
                        conds
                            .pop()
                            .ok_or_else(|| format!("{at}: #endif without #ifdef / #ifndef"))?;
                    }
                    "if" | "elif" => {
                        return Err(format!(
                            "{at}: #{name} is not supported (only #ifdef / #ifndef / #else / \
                             #endif are evaluated)"
                        ));
                    }
                    _ if !active => {}
                    "define" => {
                        let symbol = symbol.ok_or_else(|| format!("{at}: #define needs a name"))?;
                        let body = arg[symbol.len()..].trim();
                        scan.defines.insert(symbol.to_owned(), body.to_owned());
                    }
                    "undef" => {
                        let symbol = symbol.ok_or_else(|| format!("{at}: #undef needs a name"))?;
                        scan.defines.remove(symbol);
                    }
                    "include" => {
                        if self.include {
                            self.include_file(arg, &at, dir, system, scan)?;
                        }
                    }
                    other => {
                        return Err(format!("{at}: #{other} is not a supported directive"));
                    }
                }
                continue;
            }
            if !active {
                continue;
            }
            if let Some(header) = line.strip_prefix('[') {
                let name = header
                    .strip_suffix(']')
                    .ok_or_else(|| format!("{at}: unterminated section header '{line}'"))?
                    .trim()
                    .to_ascii_lowercase();
                let admitted = self
                    .admits(&name, system)
                    .map_err(|e| format!("{at}: {e}"))?;
                scan.section = Some((name, admitted));
                continue;
            }
            match &scan.section {
                // Text before the first section is not read, as GROMACS
                // does not read it (charmm27's forcefield.itp opens with a
                // banner of `*` lines).
                None => {}
                Some((_, false)) => {}
                Some((section, true)) => {
                    let text = scan.expand(line);
                    scan.rows.push(Row {
                        at,
                        section: section.clone(),
                        text,
                    });
                }
            }
        }
        if !conds.is_empty() {
            return Err(format!(
                "{origin}: {} #ifdef / #ifndef block(s) without #endif",
                conds.len()
            ));
        }
        Ok(())
    }

    /// Follow `#include <arg>` from the file at `at`, relative to `dir`, then
    /// to the include directories.
    fn include_file(
        &self,
        arg: &str,
        at: &str,
        dir: Option<&Path>,
        system: bool,
        scan: &mut Scan,
    ) -> Result<(), String> {
        let name = arg.trim_matches(|c| c == '"' || c == '<' || c == '>');
        if name.is_empty() {
            return Err(format!("{at}: #include names no file"));
        }
        let given = Path::new(name);
        let candidates: Vec<PathBuf> = if given.is_absolute() {
            vec![given.to_path_buf()]
        } else {
            dir.into_iter()
                .chain(self.include_dirs.iter().map(PathBuf::as_path))
                .map(|d| d.join(given))
                .collect()
        };
        let Some(path) = candidates.iter().find(|p| p.is_file()).cloned() else {
            if candidates.is_empty() {
                return Err(format!(
                    "{at}: #include \"{name}\" is relative, and a topology read from bytes \
                     has no directory to resolve it against (add one with with_include_dir)"
                ));
            }
            let tried: Vec<String> = candidates.iter().map(|p| p.display().to_string()).collect();
            return Err(format!(
                "{at}: could not resolve #include \"{name}\" (tried {})",
                tried.join(", ")
            ));
        };
        if !scan.visited.insert(path.clone()) {
            return Ok(());
        }
        let body = std::fs::read_to_string(&path)
            .map_err(|e| format!("{at}: #include {}: {e}", path.display()))?;
        self.scan(
            &body,
            &path.display().to_string(),
            path.parent(),
            system,
            scan,
        )
    }
}

impl ForceFieldReader for GromacsTopForcefieldReader {
    fn read<'a>(&self, src: impl Into<Source<'a>>) -> std::io::Result<ForceField> {
        let scan = self.scan_source(&src.into(), false)?;
        Ok(scan.directives().map_err(invalid_data)?.ff)
    }
}

// ---------------------------------------------------------------------------
// Preprocessed text
// ---------------------------------------------------------------------------

/// One open `#ifdef` / `#ifndef` block.
struct Cond {
    /// Whether the enclosing text is active.
    parent: bool,
    /// Whether the current branch of this block is taken.
    taken: bool,
    seen_else: bool,
}

impl Cond {
    fn active(&self) -> bool {
        self.parent && self.taken
    }
}

/// The preprocessor's state and output: the defines, the files read, the
/// current section, and every admitted data row in file order.
#[derive(Default)]
struct Scan {
    /// `#define` name → body.
    defines: HashMap<String, String>,
    visited: HashSet<PathBuf>,
    /// The current section and whether its rows are read.
    section: Option<(String, bool)>,
    rows: Vec<Row>,
}

impl Scan {
    /// `line` with every token that names a define replaced by its body.
    fn expand(&self, line: &str) -> String {
        line.split_whitespace()
            .map(|tok| self.defines.get(tok).map_or(tok, String::as_str))
            .filter(|tok| !tok.is_empty())
            .collect::<Vec<_>>()
            .join("  ")
    }
}

/// One admitted data row, with the section it sits in and where it came from.
pub(super) struct Row {
    /// `file:line`.
    pub(super) at: String,
    pub(super) section: String,
    pub(super) text: String,
}

impl Row {
    /// An error about this row, naming its place, section and text.
    pub(super) fn err(&self, why: &str) -> String {
        format!(
            "{}: [ {} ] row '{}': {why}",
            self.at, self.section, self.text
        )
    }

    pub(super) fn cols(&self) -> Vec<&str> {
        self.text.split_whitespace().collect()
    }

    pub(super) fn number(&self, tok: &str, what: &str) -> Result<f64, String> {
        tok.parse::<f64>()
            .map_err(|_| self.err(&format!("{what} is not a number: {tok}")))
    }

    /// `[ defaults ]`.
    fn defaults(&self) -> Result<Defaults, String> {
        let cols = self.cols();
        let (nbfunc, comb, gen_pairs, fudge_lj, fudge_qq) = match cols[..] {
            [a, b, c, d, e] => (a, b, c, d, e),
            // GROMACS defaults the trailing columns: gen-pairs no, fudges 1.
            [a, b] => (a, b, "no", "1.0", "1.0"),
            [a, b, c] => (a, b, c, "1.0", "1.0"),
            [a, b, c, d] => (a, b, c, d, "1.0"),
            _ => return Err(self.err("expected `nbfunc comb-rule gen-pairs fudgeLJ fudgeQQ`")),
        };
        if nbfunc != "1" {
            return Err(self.err(&format!(
                "nbfunc {nbfunc} is not supported: only 1 (Lennard-Jones) is modelled"
            )));
        }
        let mixing = match comb {
            "2" => CombiningRule::Arithmetic,
            "3" => CombiningRule::Geometric,
            other => {
                return Err(self.err(&format!(
                    "comb-rule {other} is not supported: only 2 (arithmetic) and 3 \
                     (geometric), under which V/W are sigma/epsilon"
                )));
            }
        };
        let gen_pairs = if gen_pairs.eq_ignore_ascii_case("yes") {
            true
        } else if gen_pairs.eq_ignore_ascii_case("no") {
            false
        } else {
            return Err(self.err(&format!("gen-pairs '{gen_pairs}' is neither yes nor no")));
        };
        Ok(Defaults {
            mixing,
            gen_pairs,
            fudge_lj: self.number(fudge_lj, "fudgeLJ")?,
            fudge_qq: self.number(fudge_qq, "fudgeQQ")?,
        })
    }

    /// `[ atomtypes ]` → `(name, bond type, atom/full params, (ε, σ))` in
    /// molrs units.
    ///
    /// Columns resolve from the right: the last five are `mass charge ptype V
    /// W`; the one to three leading tokens are `name`, then an optional
    /// `bond_type` and an optional integer `at.num`.
    fn atomtype(&self) -> Result<AtomRow<'_>, String> {
        let cols = self.cols();
        if cols.len() < 6 {
            return Err(self.err("expected `name [bond_type] [at.num] mass charge ptype V W`"));
        }
        let (lead, tail) = cols.split_at(cols.len() - 5);
        let is_int = |tok: &str| tok.parse::<i64>().is_ok();
        let (name, bond_type, at_num) = match *lead {
            [name] => (name, None, None),
            [name, second] if is_int(second) => (name, None, Some(second)),
            [name, second] => (name, Some(second), None),
            [name, bond_type, at_num] if is_int(at_num) => (name, Some(bond_type), Some(at_num)),
            [_, _, _] => return Err(self.err("at.num is not an integer")),
            _ => {
                return Err(self.err("more than three tokens before `mass charge ptype V W`"));
            }
        };
        let ptype = tail[2];
        if !matches!(ptype, "A" | "S" | "V" | "D" | "B") {
            return Err(self.err(&format!("ptype '{ptype}' is not one of A S V D B")));
        }
        let mut atom = Params::from_iter([
            ("mass", self.number(tail[0], "mass")?),
            ("charge", self.number(tail[1], "charge")?),
        ]);
        if let Some(tok) = at_num {
            atom.set("atomic_number", self.number(tok, "at.num")?);
        }
        atom.set("ptype", ptype);
        if let Some(bond_type) = bond_type {
            atom.set("class", bond_type);
        }
        let lj = (
            self.number(tail[4], "W (epsilon)")? / KCAL_TO_KJ.get(),
            self.number(tail[3], "V (sigma)")? * NM_TO_ANGSTROM.get(),
        );
        Ok(AtomRow {
            name,
            class: bond_type.unwrap_or(name),
            atom,
            lj,
        })
    }

    /// `[ nonbond_params ]` / `[ pairtypes ]` `i j func V W` → `((i, j) in
    /// byte order, (ε, σ))` in molrs units. Only func 1 (Lennard-Jones, V/W =
    /// σ/ε) is modelled.
    fn type_pair(&self) -> Result<(TypePair, Lj), String> {
        let cols = self.cols();
        let [i, j, func, v, w] = cols[..] else {
            return Err(self.err("expected `i j func V W`"));
        };
        if func != "1" {
            return Err(self.err(&format!(
                "func {func} is not supported: only 1 (Lennard-Jones) is modelled"
            )));
        }
        let lj = (
            self.number(w, "W (epsilon)")? / KCAL_TO_KJ.get(),
            self.number(v, "V (sigma)")? * NM_TO_ANGSTROM.get(),
        );
        let ends = if i <= j { (i, j) } else { (j, i) };
        Ok(((ends.0.to_owned(), ends.1.to_owned()), lj))
    }

    /// `[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]` → its labels
    /// (`X` mapped to the empty wildcard), function code and parameters.
    fn bonded(&self) -> Result<(Vec<String>, u32, Vec<f64>), String> {
        let (kind, n) = match self.section.as_str() {
            "bondtypes" => (Kind::Bond, 2),
            "angletypes" => (Kind::Angle, 3),
            "dihedraltypes" => (Kind::Dihedral, 4),
            other => return Err(self.err(&format!("[ {other} ] is not a bonded directive"))),
        };
        let cols = self.cols();
        let is_int = |tok: &str| tok.parse::<u32>().is_ok();
        // GROMACS's 2-name dihedraltypes form: `a b funct params`.
        let two_name = kind == Kind::Dihedral
            && cols.len() > 2
            && is_int(cols[2])
            && cols.get(4).is_none_or(|t| !is_int(t));
        let n_names = if two_name { 2 } else { n };
        if cols.len() <= n_names {
            return Err(self.err(&format!(
                "expected {n} type names, a function code and its parameters"
            )));
        }
        let funct: u32 = cols[n_names].parse().map_err(|_| {
            self.err(&format!(
                "function code '{}' is not an integer",
                cols[n_names]
            ))
        })?;
        let values = cols[n_names + 1..]
            .iter()
            .map(|tok| {
                tok.parse::<f64>()
                    .map_err(|_| self.err(&format!("parameter '{tok}' is not a number")))
            })
            .collect::<Result<Vec<f64>, String>>()?;
        let label = |l: &str| {
            if l == "X" {
                String::new()
            } else {
                l.to_owned()
            }
        };
        let labels = if two_name {
            let (a, b) = (label(cols[0]), label(cols[1]));
            if funct == 2 {
                vec![a, String::new(), String::new(), b]
            } else {
                vec![String::new(), a, b, String::new()]
            }
        } else {
            cols[..n].iter().map(|l| label(l)).collect()
        };
        Ok((labels, funct, values))
    }

    /// `[ cmaptypes ]` `a b c d e funct nx ny values…` → labels and the
    /// `N×N` grid in kcal/mol.
    fn cmaptype(&self) -> Result<(Vec<String>, ArrayD<f64>), String> {
        let cols = self.cols();
        if cols.len() < 8 {
            return Err(self.err("expected `a b c d e funct nx ny` and the grid values"));
        }
        if cols[5] != "1" {
            return Err(self.err(&format!(
                "function code {} is not supported (supported: 1)",
                cols[5]
            )));
        }
        let size = |tok: &str| {
            tok.parse::<usize>()
                .map_err(|_| self.err(&format!("grid size '{tok}' is not a count")))
        };
        let (nx, ny) = (size(cols[6])?, size(cols[7])?);
        if nx != ny || nx < 2 {
            return Err(self.err(&format!(
                "a {nx}×{ny} grid is not supported: GROMACS and the IR take a square map, \
                 N ≥ 2"
            )));
        }
        let values = cols[8..]
            .iter()
            .map(|tok| {
                self.number(tok, "a grid value")
                    .map(|v| v / KCAL_TO_KJ.get())
            })
            .collect::<Result<Vec<f64>, String>>()?;
        if values.len() != nx * ny {
            return Err(self.err(&format!(
                "a {nx}×{ny} grid takes {} values, got {}",
                nx * ny,
                values.len()
            )));
        }
        let grid =
            ArrayD::from_shape_vec(vec![nx, ny], values).map_err(|e| self.err(&e.to_string()))?;
        Ok((cols[..5].iter().map(|l| (*l).to_owned()).collect(), grid))
    }

    /// `[ constrainttypes ]` `i j funct b0` → `(labels, funct, r0 Å)`.
    fn constrainttype(&self) -> Result<([String; 2], u32, f64), String> {
        let cols = self.cols();
        let [i, j, funct, b0] = cols[..] else {
            return Err(self.err("expected `i j funct b0`"));
        };
        let funct = match funct {
            "1" => 1,
            "2" => 2,
            other => {
                return Err(self.err(&format!(
                    "function code {other} is not supported (supported: 1, 2)"
                )));
            }
        };
        Ok((
            [i.to_owned(), j.to_owned()],
            funct,
            self.number(b0, "b0")? * NM_TO_ANGSTROM.get(),
        ))
    }
}

// ---------------------------------------------------------------------------
// Function codes
// ---------------------------------------------------------------------------

/// The interaction kinds whose function codes this reader converts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Kind {
    Bond,
    Angle,
    Dihedral,
}

/// A GROMACS parameter table: the types one function code looks up in. funct
/// 1 and 9 share one, as in GROMACS (`F_PDIHS`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum Table {
    Bond(u32),
    Angle(u32),
    /// funct 1, 9
    Pdihs,
    /// funct 2
    Idihs,
    /// funct 4
    Pidihs,
    /// funct 3
    Rbdihs,
    /// funct 5
    Fourdihs,
    Cmap,
}

/// One converted row: where it goes and what it holds.
pub(super) struct Converted {
    pub(super) category: &'static str,
    pub(super) style: &'static str,
    pub(super) params: Params,
    pub(super) table: Table,
}

/// The `kind` row of function code `funct` with GROMACS parameters `v`, in
/// the IR (module docs, the conversion table). `Err` is the reason alone; the
/// caller names the row.
pub(super) fn convert(kind: Kind, funct: u32, v: &[f64]) -> Result<Converted, String> {
    let exactly = |n: usize| {
        if v.len() == n {
            Ok(())
        } else {
            Err(format!(
                "function code {funct} takes {n} parameters, got {}",
                v.len()
            ))
        }
    };
    let whole = |m: f64| {
        if m.fract() == 0.0 && m.is_finite() {
            Ok(m)
        } else {
            Err(format!("multiplicity {m} is not an integer"))
        }
    };
    let done = |category, style, params, table| {
        Ok(Converted {
            category,
            style,
            params,
            table,
        })
    };
    let kb_scale = KCAL_ANGSTROM2_TO_KJ_NM2.get();
    match (kind, funct) {
        (Kind::Bond, 1) => {
            exactly(2)?;
            done(
                "bond",
                "harmonic",
                // GROMACS ½k_b → LAMMPS K = k_b/2.
                Params::from_iter([
                    ("r0", v[0] * NM_TO_ANGSTROM.get()),
                    ("k", v[1] / kb_scale / 2.0),
                ]),
                Table::Bond(1),
            )
        }
        (Kind::Bond, 3) => {
            exactly(3)?;
            done(
                "bond",
                "morse",
                Params::from_iter([
                    ("d0", v[1] / KCAL_TO_KJ.get()),
                    ("alpha", v[2] / NM_TO_ANGSTROM.get()),
                    ("r0", v[0] * NM_TO_ANGSTROM.get()),
                ]),
                Table::Bond(3),
            )
        }
        (Kind::Angle, 1) => {
            exactly(2)?;
            done(
                "angle",
                "harmonic",
                // GROMACS ½k_θ → LAMMPS K = k_θ/2; θ₀ stays in degrees.
                Params::from_iter([("theta0", v[0]), ("k", v[1] / KCAL_TO_KJ.get() / 2.0)]),
                Table::Angle(1),
            )
        }
        (Kind::Angle, 5) => {
            exactly(4)?;
            done(
                "angle",
                "charmm",
                // Both GROMACS terms are ½k forms; LAMMPS's are not.
                Params::from_iter([
                    ("k", v[1] / KCAL_TO_KJ.get() / 2.0),
                    ("theta0", v[0]),
                    ("k_ub", v[3] / kb_scale / 2.0),
                    ("r_ub", v[2] * NM_TO_ANGSTROM.get()),
                ]),
                Table::Angle(5),
            )
        }
        (Kind::Dihedral, 1 | 9 | 4) => {
            exactly(3)?;
            let params = Params::from_iter([
                ("k", v[1] / KCAL_TO_KJ.get()),
                ("periodicity", whole(v[2])?),
                ("phase", v[0]),
            ]);
            if funct == 4 {
                done("improper", "periodic", params, Table::Pidihs)
            } else {
                done("dihedral", "periodic", params, Table::Pdihs)
            }
        }
        (Kind::Dihedral, 2) => {
            exactly(2)?;
            // ½k(ξ−ξ₀)² (ξ signed, the difference wrapped to ±180°) equals
            // K(|ξ|−χ₀)² exactly at ξ₀ = 0° and ±180°.
            let chi0 = match v[0] {
                0.0 => 0.0,
                x if x.abs() == 180.0 => 180.0,
                x => {
                    return Err(format!(
                        "function code 2 with xi0 = {x} deg has no IR form: the signed GROMACS \
                         harmonic improper equals improper/harmonic (chi = |phi|) only at \
                         xi0 = 0 and 180"
                    ));
                }
            };
            done(
                "improper",
                "harmonic",
                Params::from_iter([("k", v[1] / (2.0 * KCAL_TO_KJ.get())), ("chi0", chi0)]),
                Table::Idihs,
            )
        }
        (Kind::Dihedral, 3) => {
            exactly(6)?;
            let (style, params) = rb_polynomial(std::array::from_fn(|n| v[n] / KCAL_TO_KJ.get()));
            done("dihedral", style, params, Table::Rbdihs)
        }
        (Kind::Dihedral, 5) => {
            exactly(4)?;
            done(
                "dihedral",
                "opls",
                Params::from_iter([
                    ("k1", v[0] / KCAL_TO_KJ.get()),
                    ("k2", v[1] / KCAL_TO_KJ.get()),
                    ("k3", v[2] / KCAL_TO_KJ.get()),
                    ("k4", v[3] / KCAL_TO_KJ.get()),
                ]),
                Table::Fourdihs,
            )
        }
        _ => {
            let supported = match kind {
                Kind::Bond => "1 (harmonic), 3 (Morse)",
                Kind::Angle => "1 (harmonic), 5 (Urey-Bradley)",
                Kind::Dihedral => {
                    "1 (periodic), 2 (harmonic improper), 3 (Ryckaert-Bellemans), 4 (periodic \
                     improper), 5 (Fourier), 9 (multiple periodic)"
                }
            };
            Err(format!(
                "function code {funct} is not supported (supported: {supported})"
            ))
        }
    }
}

/// A multi-term `dihedral/periodic` params bag from its terms `(k, n, phase)`
/// (one term: the single-term spelling).
fn periodic_terms(terms: &[[f64; 3]]) -> Params {
    if let [[k, n, phase]] = terms {
        return Params::from_iter([("k", *k), ("periodicity", *n), ("phase", *phase)]);
    }
    let mut p = Params::new();
    for (m, [k, n, phase]) in terms.iter().enumerate() {
        p.set(&format!("k{}", m + 1), *k);
        p.set(&format!("periodicity{}", m + 1), *n);
        p.set(&format!("phase{}", m + 1), *phase);
    }
    p
}

// ---------------------------------------------------------------------------
// Directives → force field
// ---------------------------------------------------------------------------

/// `[ defaults ]`.
#[derive(Debug, Clone, Copy)]
pub(super) struct Defaults {
    pub(super) mixing: CombiningRule,
    pub(super) gen_pairs: bool,
    pub(super) fudge_lj: f64,
    pub(super) fudge_qq: f64,
}

/// One `[ atomtypes ]` row in molrs units.
struct AtomRow<'a> {
    name: &'a str,
    class: &'a str,
    atom: Params,
    /// `(ε, σ)`
    lj: (f64, f64),
}

/// A force-field type the system reader can look up: its labels (`""` the
/// wildcard) and name.
pub(super) struct LabelledType {
    pub(super) labels: Vec<String>,
    pub(super) name: String,
}

/// The directives as the IR, with what [`system`] needs to type a molecule.
pub(super) struct Directives {
    pub(super) ff: ForceField,
    pub(super) defaults: Option<Defaults>,
    /// atom type → its bond type (`class`, or its name).
    pub(super) classes: HashMap<String, String>,
    /// atom type → `(mass, charge)`.
    pub(super) atom_defaults: HashMap<String, (f64, f64)>,
    /// The `[ pairtypes ]` type pairs, in byte order.
    pub(super) pairtypes: HashSet<(String, String)>,
    /// Each parameter table's types in file order.
    pub(super) lookup: HashMap<Table, Vec<LabelledType>>,
    /// `[ constrainttypes ]`: `(labels, funct, r0 Å)` in file order.
    pub(super) constrainttypes: Vec<([String; 2], u32, f64)>,
}

/// A bonded type before it is named.
struct BondedDef<'r> {
    row: &'r Row,
    funct: u32,
    category: &'static str,
    style: &'static str,
    labels: Vec<String>,
    params: Params,
    table: Table,
    /// funct 9: the terms `(k, n, phase)` merged so far.
    terms: Vec<[f64; 3]>,
}

impl Scan {
    /// The force field the directive rows define, and the lookup tables.
    fn directives(&self) -> Result<Directives, String> {
        let mut ff = ForceField::new("GROMACS");
        let rows = |section: &'static str| self.rows.iter().filter(move |r| r.section == section);

        let mut defaults_rows = rows("defaults");
        let defaults = match defaults_rows.next() {
            Some(row) => {
                if let Some(extra) = defaults_rows.next() {
                    return Err(extra.err("a second [ defaults ] row"));
                }
                Some(row.defaults()?)
            }
            None => None,
        };
        let need_defaults = |row: &Row| {
            defaults.ok_or_else(|| {
                row.err(&format!(
                    "[ {} ] needs [ defaults ]: V/W are sigma/epsilon only under the \
                     comb-rule it declares",
                    row.section
                ))
            })
        };

        // Atom types, in file order.
        let mut atoms: Vec<AtomRow<'_>> = Vec::new();
        let mut atom_index: HashMap<&str, usize> = HashMap::new();
        for row in rows("atomtypes") {
            need_defaults(row)?;
            let a = row.atomtype()?;
            match atom_index.get(a.name) {
                Some(&i) if atoms[i].atom == a.atom && atoms[i].lj == a.lj => {}
                Some(_) => {
                    return Err(row.err(&format!(
                        "atom type '{}' is already defined with other parameters",
                        a.name
                    )));
                }
                None => {
                    atom_index.insert(a.name, atoms.len());
                    atoms.push(a);
                }
            }
        }
        // Cross tables, keyed by the type pair in byte order.
        let cross = |section: &'static str| -> Result<CrossTable, String> {
            let mut table = BTreeMap::new();
            for row in rows(section) {
                need_defaults(row)?;
                let (ends, lj) = row.type_pair()?;
                for end in [&ends.0, &ends.1] {
                    if !atom_index.contains_key(end.as_str()) {
                        return Err(row.err(&format!("'{end}' is no [ atomtypes ] type")));
                    }
                }
                match table.get(&ends) {
                    Some(&prev) if prev == lj => {}
                    Some(_) => {
                        return Err(row.err(&format!(
                            "{} {} is already given other parameters",
                            ends.0, ends.1
                        )));
                    }
                    None => {
                        table.insert(ends, lj);
                    }
                }
            }
            Ok(table)
        };
        let nonbond = cross("nonbond_params")?;
        let pairtypes = cross("pairtypes")?;

        if let Some(d) = defaults {
            let lj14 = if d.gen_pairs { d.fudge_lj } else { 1.0 };
            ff.set_special_bonds(SpecialBonds {
                lj: [0.0, 0.0, lj14],
                coul: [0.0, 0.0, d.fudge_qq],
            });
        }
        if let Some(d) = defaults
            && (!atoms.is_empty() || !nonbond.is_empty())
        {
            define_lj(&mut ff, d, &atoms, &nonbond, &pairtypes)?;
        }
        for a in &atoms {
            ff.def_style("atom", "full", Params::new())
                .and_then(|s| s.def_type(a.name, &[], a.atom.clone()))
                .map_err(|e| e.to_string())?;
        }

        let lookup = define_bonded(&mut ff, self)?;

        let mut constrainttypes = Vec::new();
        for row in rows("constrainttypes") {
            constrainttypes.push(row.constrainttype()?);
        }
        Ok(Directives {
            ff,
            defaults,
            classes: atoms
                .iter()
                .map(|a| (a.name.to_owned(), a.class.to_owned()))
                .collect(),
            atom_defaults: atoms
                .iter()
                .map(|a| {
                    (
                        a.name.to_owned(),
                        (
                            a.atom.get("mass").and_then(|v| v.as_num()).unwrap_or(0.0),
                            a.atom.get("charge").and_then(|v| v.as_num()).unwrap_or(0.0),
                        ),
                    )
                })
                .collect(),
            pairtypes: pairtypes.into_keys().collect(),
            lookup,
            constrainttypes,
        })
    }
}

/// The Lennard-Jones and Coulomb pair styles of `[ atomtypes ]`,
/// `[ nonbond_params ]` and `[ pairtypes ]` (module docs, "1-4 pairs").
fn define_lj(
    ff: &mut ForceField,
    d: Defaults,
    atoms: &[AtomRow<'_>],
    nonbond: &CrossTable,
    pairtypes: &CrossTable,
) -> Result<(), String> {
    let regular: HashMap<&str, (f64, f64)> = atoms.iter().map(|a| (a.name, a.lj)).collect();
    let key = |a: &str, b: &str| {
        if a <= b {
            (a.to_owned(), b.to_owned())
        } else {
            (b.to_owned(), a.to_owned())
        }
    };
    // The pair's regular parameters: its cross row, else the comb-rule.
    let nb = |a: &str, b: &str| -> (f64, f64) {
        nonbond
            .get(&key(a, b))
            .copied()
            .unwrap_or_else(|| d.mixing.combine(regular[a], regular[b]))
    };
    let weight = if d.gen_pairs { d.fudge_lj } else { 1.0 };
    if !pairtypes.is_empty() && weight == 0.0 {
        return Err(
            "[ defaults ] fudgeLJ 0 with [ pairtypes ]: GROMACS prices a pairtype pair at full \
             weight and a generated one at 0, and the IR's one 1-4 weight cannot do both"
                .to_owned(),
        );
    }
    // The 1-4 (ε, σ) the IR must give the pair, divided by the 1-4 weight.
    let target = |a: &str, b: &str| -> (f64, f64) {
        match pairtypes.get(&key(a, b)) {
            Some(&(eps, sigma)) => (eps / weight, sigma),
            None => nb(a, b),
        }
    };
    let mut self14: HashMap<&str, (f64, f64)> = HashMap::new();
    for a in atoms {
        let t = target(a.name, a.name);
        if !same_lj(t, a.lj) {
            self14.insert(a.name, t);
        }
    }
    let own14 = |a: &str| self14.get(a).copied().unwrap_or(regular[a]);
    // Cross rows: `[ nonbond_params ]`, then every pair the mix of the self
    // rows' 1-4 values would price wrong.
    let mut rows: BTreeMap<TypePair, (Lj, Option<Lj>)> = BTreeMap::new();
    for ((a, b), &lj) in nonbond {
        if a == b {
            // A self pair restates the [ atomtypes ] row, or contradicts it.
            if !same_lj(lj, regular[a.as_str()]) {
                return Err(format!(
                    "[ nonbond_params ] {a} {a} gives sigma/epsilon other than its [ atomtypes ] \
                     row: a self pair has one set of parameters"
                ));
            }
            continue;
        }
        let t = target(a, b);
        rows.insert((a.clone(), b.clone()), (lj, (!same_lj(t, lj)).then_some(t)));
    }
    let mut candidates: Vec<(String, String)> =
        pairtypes.keys().filter(|(a, b)| a != b).cloned().collect();
    for a in self14.keys() {
        for b in atoms {
            if *a != b.name {
                candidates.push(key(a, b.name));
            }
        }
    }
    for (a, b) in candidates {
        if rows.contains_key(&(a.clone(), b.clone())) {
            continue;
        }
        let t = target(&a, &b);
        if !same_lj(t, d.mixing.combine(own14(&a), own14(&b))) {
            let lj = nb(&a, &b);
            rows.insert((a, b), (lj, Some(t)));
        }
    }

    let uses_14 = !self14.is_empty() || rows.values().any(|(_, t)| t.is_some());
    let (lj_name, coul_name) = if uses_14 {
        ("lj/charmm", "coul/charmm")
    } else {
        ("lj/cut", "coul/cut")
    };
    let mut style_params = Params::new();
    style_params.set("mixing", d.mixing.name());
    if uses_14 {
        style_params.set("one_four", "epsilon14");
    }
    let lj_params = |(eps, sigma): (f64, f64), t: Option<(f64, f64)>| {
        let mut p = Params::from_iter([("epsilon", eps), ("sigma", sigma)]);
        if let Some((eps14, sigma14)) = t {
            p.set("epsilon14", eps14);
            p.set("sigma14", sigma14);
        }
        p
    };
    let style = ff
        .def_style("pair", lj_name, style_params)
        .map_err(|e| e.to_string())?;
    for a in atoms {
        style
            .def_type(
                a.name,
                &[a.name],
                lj_params(a.lj, self14.get(a.name).copied()),
            )
            .map_err(|e| e.to_string())?;
    }
    for ((a, b), (lj, t)) in &rows {
        let name = TypeName::pair(a, b)?;
        style
            .def_type(name.as_str(), &[a, b], lj_params(*lj, *t))
            .map_err(|e| format!("[ nonbond_params ] / [ pairtypes ] {a} {b}: {e}"))?;
    }
    if !atoms.is_empty() {
        ff.def_style(
            "pair",
            coul_name,
            Params::from_iter([
                ("coulomb", crate::core::constants::gromacs_coulomb_real()),
                ("dielectric", VACUUM_DIELECTRIC),
            ]),
        )
        .map_err(|e| e.to_string())?;
    }
    Ok(())
}

/// Define the bonded and cmap types of `scan`'s directive rows in `ff`, and
/// return each parameter table's types in file order.
fn define_bonded(
    ff: &mut ForceField,
    scan: &Scan,
) -> Result<HashMap<Table, Vec<LabelledType>>, String> {
    let mut defs: Vec<BondedDef<'_>> = Vec::new();
    // funct 9 continues the last F_PDIHS row when its labels are the same.
    let mut last_pdihs: Option<usize> = None;
    for row in &scan.rows {
        let kind = match row.section.as_str() {
            "bondtypes" => Kind::Bond,
            "angletypes" => Kind::Angle,
            "dihedraltypes" => Kind::Dihedral,
            _ => continue,
        };
        let (labels, funct, values) = row.bonded()?;
        let c = convert(kind, funct, &values).map_err(|e| row.err(&e))?;
        if c.table == Table::Pdihs {
            let term = [values[1] / KCAL_TO_KJ.get(), values[2], values[0]];
            if let Some(i) = last_pdihs
                && funct == 9
                && defs[i].funct == 9
                && defs[i].labels == labels
            {
                defs[i].terms.push(term);
                defs[i].params = periodic_terms(&defs[i].terms);
                continue;
            }
            last_pdihs = Some(defs.len());
        }
        defs.push(BondedDef {
            row,
            funct,
            category: c.category,
            style: c.style,
            terms: if c.table == Table::Pdihs {
                vec![[values[1] / KCAL_TO_KJ.get(), values[2], values[0]]]
            } else {
                Vec::new()
            },
            labels,
            params: c.params,
            table: c.table,
        });
    }

    // GROMACS finds a row in either orientation: a later row on the same
    // atoms either way with other parameters is a redefinition, which it
    // warns about and applies over the first. Refused here.
    let mut by_atoms: HashMap<(Table, Vec<String>), usize> = HashMap::new();
    for (i, def) in defs.iter().enumerate() {
        let reversed: Vec<String> = def.labels.iter().rev().cloned().collect();
        for key in [def.labels.clone(), reversed] {
            if let Some(&j) = by_atoms.get(&(def.table, key))
                && defs[j].params != def.params
            {
                return Err(def.row.err(&format!(
                    "the same atoms, either way, as {} with other parameters: GROMACS \
                     overrides that row with a warning; give the type once",
                    defs[j].row.at
                )));
            }
        }
        by_atoms.insert((def.table, def.labels.clone()), i);
    }

    // Names: the labels joined, qualified by the function code where two
    // styles of one category would share one.
    let mut styles_of: HashMap<(&str, String), HashSet<&str>> = HashMap::new();
    let mut bases = Vec::with_capacity(defs.len());
    for def in &defs {
        let refs: Vec<&str> = def.labels.iter().map(String::as_str).collect();
        let base = TypeName::join(&refs).map_err(|e| def.row.err(&e))?;
        styles_of
            .entry((def.category, base.as_str().to_owned()))
            .or_default()
            .insert(def.style);
        bases.push(base);
    }
    let mut lookup: HashMap<Table, Vec<LabelledType>> = HashMap::new();
    for (def, base) in defs.iter().zip(bases) {
        let name = if styles_of[&(def.category, base.as_str().to_owned())].len() > 1 {
            base.with_qualifier(&[&def.funct.to_string()])
                .map_err(|e| def.row.err(&e))?
        } else {
            base
        };
        let refs: Vec<&str> = def.labels.iter().map(String::as_str).collect();
        ff.def_style(def.category, def.style, Params::new())
            .and_then(|s| s.def_type(name.as_str(), &refs, def.params.clone()))
            .map_err(|e| def.row.err(&e.to_string()))?;
        lookup.entry(def.table).or_default().push(LabelledType {
            labels: def.labels.clone(),
            name: name.as_str().to_owned(),
        });
    }

    for row in scan.rows.iter().filter(|r| r.section == "cmaptypes") {
        let (labels, grid) = row.cmaptype()?;
        let refs: Vec<&str> = labels.iter().map(String::as_str).collect();
        let name = TypeName::join(&refs).map_err(|e| row.err(&e))?;
        let mut params = Params::new();
        params.set(CMAP_GRID, grid);
        ff.def_style("cmap", "charmm", Params::new())
            .and_then(|s| s.def_type(name.as_str(), &refs, params))
            .map_err(|e| row.err(&e.to_string()))?;
        lookup.entry(Table::Cmap).or_default().push(LabelledType {
            labels,
            name: name.as_str().to_owned(),
        });
    }
    Ok(lookup)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::VACUUM_DIELECTRIC;
    use crate::ff::forcefield::{AtomType, ForceField, PairType, Style, StyleDefs};
    use crate::ff::ir::Params;
    use molrs::core::constants::COULOMB_REAL;

    /// `nbfunc 1`, comb-rule 3 (OPLS-AA: geometric σ and ε), `gen-pairs yes`.
    const DEFAULTS: &str = "[ defaults ]\n1  3  yes  0.5  0.5\n";

    /// GROMACS OPLS-AA `ffnonbonded.itp` opls_135 (8-column form).
    const OPLS_135: &str = "opls_135  CT  6  12.011  -0.18  A  0.35  0.276144";

    fn read(text: &str) -> ForceField {
        GromacsTopForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("read: {e}"))
    }

    fn read_err(text: &str) -> String {
        GromacsTopForcefieldReader::new()
            .read(text.as_bytes())
            .expect_err("expected Err from read")
            .to_string()
    }

    /// `[ defaults ]` (comb-rule 3) then one `[ section ]` holding `rows`.
    fn with_section(section: &str, rows: &str) -> String {
        format!("{DEFAULTS}[ {section} ]\n{rows}\n")
    }

    fn style<'a>(ff: &'a ForceField, category: &str, name: &str) -> &'a Style {
        ff.get_style(category, name)
            .unwrap_or_else(|| panic!("no {category}/{name} style in {:?}", style_keys(ff)))
    }

    fn style_keys(ff: &ForceField) -> Vec<(String, String)> {
        ff.styles()
            .iter()
            .map(|s| (s.category().to_owned(), s.name().to_owned()))
            .collect()
    }

    fn atom_type<'a>(ff: &'a ForceField, name: &str) -> &'a AtomType {
        style(ff, "atom", "full")
            .get_atomtype(name)
            .unwrap_or_else(|| panic!("no atom type {name}"))
    }

    fn lj_self_row<'a>(ff: &'a ForceField, name: &str) -> &'a PairType {
        style(ff, "pair", "lj/cut")
            .get_pairtype(name, None)
            .unwrap_or_else(|| panic!("no lj/cut self row for {name}"))
    }

    /// Every bonded type of `s` as `(endpoints, params)`.
    fn bonded(s: &Style) -> Vec<(Vec<&str>, &Params)> {
        match s.defs() {
            StyleDefs::Bond(v) => v
                .iter()
                .map(|t| (vec![t.itom.as_str(), t.jtom.as_str()], &t.params))
                .collect(),
            StyleDefs::Angle(v) => v
                .iter()
                .map(|t| {
                    (
                        vec![t.itom.as_str(), t.jtom.as_str(), t.ktom.as_str()],
                        &t.params,
                    )
                })
                .collect(),
            StyleDefs::Dihedral(v) => v
                .iter()
                .map(|t| {
                    (
                        vec![
                            t.itom.as_str(),
                            t.jtom.as_str(),
                            t.ktom.as_str(),
                            t.ltom.as_str(),
                        ],
                        &t.params,
                    )
                })
                .collect(),
            StyleDefs::Improper(v) => v
                .iter()
                .map(|t| {
                    (
                        vec![
                            t.itom.as_str(),
                            t.jtom.as_str(),
                            t.ktom.as_str(),
                            t.ltom.as_str(),
                        ],
                        &t.params,
                    )
                })
                .collect(),
            other => panic!("{} is not a bonded style", other.category()),
        }
    }

    /// The params of the only type of `category/name`, with its endpoints.
    fn only_type<'a>(ff: &'a ForceField, category: &str, name: &str) -> (Vec<&'a str>, &'a Params) {
        let mut types = bonded(style(ff, category, name));
        assert_eq!(types.len(), 1, "{category}/{name}: {types:?}");
        types.pop().expect("one type")
    }

    fn assert_param(p: &Params, key: &str, want: f64, tol: f64) {
        let got = p
            .get(key)
            .and_then(|v| v.as_num())
            .unwrap_or_else(|| panic!("missing `{key}` in {p:?}"));
        assert!((got - want).abs() < tol, "`{key}`: got {got}, want {want}");
    }

    fn assert_names(err: &str, needles: &[&str]) {
        for needle in needles {
            assert!(err.contains(needle), "error should name `{needle}`: {err}");
        }
    }

    // -- [ defaults ] ----------------------------------------------------------

    #[test]
    fn comb_rule_3_declares_geometric_mixing_on_lj_cut() {
        let ff = read(&with_section("atomtypes", OPLS_135));
        assert_eq!(
            style(&ff, "pair", "lj/cut")
                .params()
                .get("mixing")
                .and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    #[test]
    fn comb_rule_2_declares_arithmetic_mixing_on_lj_cut() {
        let text = format!("[ defaults ]\n1  2  yes  0.5  0.5\n[ atomtypes ]\n{OPLS_135}\n");
        let ff = read(&text);
        assert_eq!(
            style(&ff, "pair", "lj/cut")
                .params()
                .get("mixing")
                .and_then(|v| v.as_text()),
            Some("arithmetic")
        );
    }

    /// 1-2 and 1-3 are excluded; fudgeLJ / fudgeQQ are the 1-4 weights.
    #[test]
    fn fudge_factors_are_the_1_4_special_bond_weights() {
        let ff = read("[ defaults ]\n1  3  yes  0.5  0.8333\n");
        let sb = ff.special_bonds();
        assert_eq!(sb.lj, [0.0, 0.0, 0.5]);
        assert_eq!(sb.coul, [0.0, 0.0, 0.8333]);
    }

    /// comb-rule 1 makes V/W C6/C12, which `lj/cut` does not take.
    #[test]
    fn comb_rule_1_is_an_error_naming_the_value() {
        let err = read_err("[ defaults ]\n1  1  yes  0.5  0.5\n");
        assert_names(&err, &["comb-rule", "1"]);
    }

    /// nbfunc 2 is Buckingham.
    #[test]
    fn nbfunc_2_is_an_error_naming_the_value() {
        let err = read_err("[ defaults ]\n2  3  yes  0.5  0.5\n");
        assert_names(&err, &["nbfunc", "2"]);
    }

    /// gen-pairs no generates no 1-4 Lennard-Jones: every 1-4 pair is priced
    /// at full weight by its own parameters, and fudgeLJ is unused.
    #[test]
    fn gen_pairs_no_prices_1_4_lj_at_full_weight() {
        let sb = *read("[ defaults ]\n1  3  no  0.5  0.8333\n").special_bonds();
        assert_eq!(sb.lj, [0.0, 0.0, 1.0]);
        assert_eq!(sb.coul, [0.0, 0.0, 0.8333]);
    }

    #[test]
    fn gen_pairs_neither_yes_nor_no_is_an_error() {
        let err = read_err("[ defaults ]\n1  3  maybe  0.5  0.5\n");
        assert_names(&err, &["gen-pairs", "maybe"]);
    }

    // -- [ atomtypes ] ---------------------------------------------------------

    /// opls_135: σ = 0.35 nm × 10 = 3.5 Å; ε = 0.276144 kJ/mol ÷ 4.184 =
    /// 0.066 kcal/mol, on the `lj/cut` self row, not on the atom type.
    #[test]
    fn atomtypes_row_splits_into_atom_full_and_an_lj_cut_self_row() {
        let ff = read(&with_section("atomtypes", OPLS_135));

        let a = &atom_type(&ff, "opls_135").params;
        assert_eq!(a.get("mass").and_then(|v| v.as_num()), Some(12.011));
        assert_eq!(a.get("charge").and_then(|v| v.as_num()), Some(-0.18));
        assert_eq!(a.get("atomic_number").and_then(|v| v.as_num()), Some(6.0));
        assert_eq!(a.get("class").and_then(|v| v.as_text()), Some("CT"));
        assert_eq!(a.get("ptype").and_then(|v| v.as_text()), Some("A"));
        assert_eq!(
            a.get("sigma").and_then(|v| v.as_num()),
            None,
            "σ belongs on lj/cut"
        );
        assert_eq!(
            a.get("epsilon").and_then(|v| v.as_num()),
            None,
            "ε belongs on lj/cut"
        );

        let lj = lj_self_row(&ff, "opls_135");
        assert_eq!(
            (lj.itom.as_str(), lj.jtom.as_str()),
            ("opls_135", "opls_135")
        );
        assert_param(&lj.params, "sigma", 3.5, 1e-12);
        assert_param(&lj.params, "epsilon", 0.066, 1e-12);
    }

    /// Atom types imply the Coulomb pair style with its constants declared.
    #[test]
    fn atomtypes_declare_coul_cut_with_its_constants() {
        let ff = read(&with_section("atomtypes", OPLS_135));
        let p = style(&ff, "pair", "coul/cut").params();
        assert_eq!(
            p.get("coulomb").and_then(|v| v.as_num()),
            Some(crate::core::constants::gromacs_coulomb_real())
        );
        // GROMACS's ONE_4PI_EPS0 is LAMMPS real's to 9.9e-9.
        assert!(
            (crate::core::constants::gromacs_coulomb_real() / COULOMB_REAL - 1.0 - 9.9e-9).abs()
                < 1e-10
        );
        assert_eq!(
            p.get("dielectric").and_then(|v| v.as_num()),
            Some(VACUUM_DIELECTRIC)
        );
    }

    #[test]
    fn atomtypes_six_column_row_has_no_bond_type_or_atomic_number() {
        let ff = read(&with_section(
            "atomtypes",
            "opls_135  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get("class").and_then(|v| v.as_text()), None);
        assert_eq!(p.get("atomic_number").and_then(|v| v.as_num()), None);
        assert_param(&lj_self_row(&ff, "opls_135").params, "sigma", 3.5, 1e-12);
    }

    /// Seven columns with an integer second token: `name at.num mass …`.
    #[test]
    fn atomtypes_seven_column_row_with_an_integer_reads_the_atomic_number() {
        let ff = read(&with_section(
            "atomtypes",
            "opls_135  6  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get("atomic_number").and_then(|v| v.as_num()), Some(6.0));
        assert_eq!(p.get("class").and_then(|v| v.as_text()), None);
        assert_eq!(p.get("mass").and_then(|v| v.as_num()), Some(12.011));
    }

    /// Seven columns with a non-integer second token: `name bond_type mass …`.
    #[test]
    fn atomtypes_seven_column_row_with_a_label_reads_the_bond_type() {
        let ff = read(&with_section(
            "atomtypes",
            "opls_135  CT  12.011  -0.18  A  0.35  0.276144",
        ));
        let p = &atom_type(&ff, "opls_135").params;
        assert_eq!(p.get("class").and_then(|v| v.as_text()), Some("CT"));
        assert_eq!(p.get("atomic_number").and_then(|v| v.as_num()), None);
        assert_eq!(p.get("mass").and_then(|v| v.as_num()), Some(12.011));
    }

    #[test]
    fn atomtypes_eight_column_row_with_a_non_integer_atomic_number_is_an_error() {
        let err = read_err(&with_section(
            "atomtypes",
            "opls_135  CT  C  12.011  -0.18  A  0.35  0.276144",
        ));
        assert_names(&err, &["opls_135"]);
    }

    #[test]
    fn atomtypes_row_with_four_leading_tokens_is_an_error() {
        let err = read_err(&with_section(
            "atomtypes",
            "opls_135  CT  6  extra  12.011  -0.18  A  0.35  0.276144",
        ));
        assert_names(&err, &["opls_135"]);
    }

    #[test]
    fn atomtypes_row_with_an_unknown_ptype_is_an_error() {
        let err = read_err(&with_section(
            "atomtypes",
            "opls_135  12.011  -0.18  X  0.35  0.276144",
        ));
        assert_names(&err, &["opls_135"]);
    }

    /// V/W mean σ/ε only under the comb-rule `[ defaults ]` declares.
    #[test]
    fn atomtypes_without_defaults_is_an_error() {
        let err = read_err(&format!("[ atomtypes ]\n{OPLS_135}\n"));
        assert_names(&err, &["defaults"]);
    }

    // -- [ bondtypes ] ---------------------------------------------------------

    /// b₀ = 0.109 nm × 10 = 1.09 Å; GROMACS ½k_b with k_b = 284512 kJ/mol/nm²
    /// is LAMMPS's K = k_b/2 = 284512 ÷ 418.4 ÷ 2 = 340 kcal/mol/Å².
    #[test]
    fn bondtypes_funct_1_is_bond_harmonic_in_molrs_units() {
        let ff = read(&with_section("bondtypes", "CT  HC  1  0.10900  284512.0"));
        let (ends, p) = only_type(&ff, "bond", "harmonic");
        assert_eq!(ends, ["CT", "HC"]);
        assert_param(p, "r0", 1.09, 1e-12);
        assert_param(p, "k", 340.0, 1e-9);
    }

    /// d0 = D = 400 kJ/mol ÷ 4.184 = 95.602294455066… kcal/mol; α = 20 nm⁻¹ ÷ 10 =
    /// 2 Å⁻¹; b₀ = 0.1529 nm × 10 = 1.529 Å.
    #[test]
    fn bondtypes_funct_3_is_bond_morse_in_molrs_units() {
        let ff = read(&with_section("bondtypes", "CT  CT  3  0.1529  400.0  20.0"));
        let (ends, p) = only_type(&ff, "bond", "morse");
        assert_eq!(ends, ["CT", "CT"]);
        assert_param(p, "d0", 95.602_294_455_066_9, 1e-9);
        assert_param(p, "alpha", 2.0, 1e-12);
        assert_param(p, "r0", 1.529, 1e-12);
    }

    #[test]
    fn bondtypes_funct_2_is_an_error_naming_section_and_code() {
        let err = read_err(&with_section("bondtypes", "CT  HC  2  0.10900  284512.0"));
        assert_names(&err, &["bondtypes", "2", "CT"]);
    }

    /// Two identical rows of one label pair are one type.
    #[test]
    fn two_equal_bondtypes_rows_define_one_type() {
        let ff = read(&with_section(
            "bondtypes",
            "CT  HC  1  0.10900  284512.0\nCT  HC  1  0.10900  284512.0",
        ));
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    /// Two rows of one label pair with different k: a conflict, not
    /// last-writer-wins.
    #[test]
    fn two_bondtypes_rows_of_one_pair_with_different_k_are_an_error() {
        let err = read_err(&with_section(
            "bondtypes",
            "CT  HC  1  0.10900  284512.0\nCT  HC  1  0.10900  300000.0",
        ));
        assert_names(&err, &["300000.0", "the same atoms"]);
    }

    /// `HC CT` restated as `CT HC` with other parameters is the redefinition
    /// GROMACS warns about and applies over the first; with the same ones it
    /// is a second name for the same row.
    #[test]
    fn a_reversed_restatement_with_other_parameters_is_an_error() {
        let err = read_err(&with_section(
            "bondtypes",
            "CT  HC  1  0.10900  284512.0\nHC  CT  1  0.10900  300000.0",
        ));
        assert_names(&err, &["HC  CT", "the same atoms"]);
        let ff = read(&with_section(
            "bondtypes",
            "CT  HC  1  0.10900  284512.0\nHC  CT  1  0.10900  284512.0",
        ));
        assert_eq!(ff.get_bondtypes().len(), 2);
    }

    // -- [ angletypes ] --------------------------------------------------------

    /// θ₀ = 107.8° stays degrees; GROMACS ½k_θ with k_θ = 276.144 kJ/mol/rad²
    /// is LAMMPS's K = 276.144 ÷ 4.184 ÷ 2 = 33 kcal/mol/rad².
    #[test]
    fn angletypes_funct_1_is_angle_harmonic_in_molrs_units() {
        let ff = read(&with_section(
            "angletypes",
            "HC  CT  HC  1  107.800  276.144",
        ));
        let (ends, p) = only_type(&ff, "angle", "harmonic");
        assert_eq!(ends, ["HC", "CT", "HC"]);
        assert_eq!(p.get("theta0").and_then(|v| v.as_num()), Some(107.8));
        assert_param(p, "k", 33.0, 1e-9);
    }

    /// CHARMM36 HA-CT2-HA in GROMACS: θ₀ = 109.5°, k_θ = 297.064 kJ/mol/rad²,
    /// r₁₃ = 0.1802 nm, k_UB = 4518.72 kJ/mol/nm². Both GROMACS terms are ½k:
    /// k = 297.064/(2·4.184) = 35.5, k_ub = 4518.72/(2·418.4) = 5.4, r_ub =
    /// 1.802 Å — CHARMM's own `HA CT2 HA 35.500 109.50 5.40 1.80200`.
    #[test]
    fn angletypes_funct_5_is_angle_charmm() {
        let ff = read(&with_section(
            "angletypes",
            "HA  CT2  HA  5  109.50  297.064  0.1802  4518.72",
        ));
        let (ends, p) = only_type(&ff, "angle", "charmm");
        assert_eq!(ends, ["HA", "CT2", "HA"]);
        assert_param(p, "k", 35.5, 1e-12);
        assert_param(p, "theta0", 109.5, 1e-12);
        assert_param(p, "k_ub", 5.4, 1e-12);
        assert_param(p, "r_ub", 1.802, 1e-12);
    }

    #[test]
    fn angletypes_funct_2_is_an_error_naming_section_and_code() {
        let err = read_err(&with_section("angletypes", "HC  CT  HC  2  107.8  276.144"));
        assert_names(&err, &["angletypes", "2", "HC"]);
    }

    // -- [ dihedraltypes ] -----------------------------------------------------

    /// GROMACS OPLS-AA HC-CT-CT-HC, RB in kJ/mol: Aₙ₊₁ = (−1)ⁿ Cₙ / 4.184, so
    /// a1 = 0.6276/4.184 = 0.15, a2 = −0.45, a3 = 0, a4 = 2.5104/4.184 = 0.6,
    /// a5 = 0 kcal/mol — the polynomial in cos φ, constant included.
    #[test]
    fn dihedraltypes_funct_3_is_dihedral_multi_harmonic() {
        let ff = read(&with_section(
            "dihedraltypes",
            "HC  CT  CT  HC  3  0.62760  1.88280  0.00000  -2.51040  0.00000  0.00000",
        ));
        let (ends, p) = only_type(&ff, "dihedral", "multi/harmonic");
        assert_eq!(ends, ["HC", "CT", "CT", "HC"]);
        for (key, want) in [
            ("a1", 0.15),
            ("a2", -0.45),
            ("a3", 0.0),
            ("a4", 0.6),
            ("a5", 0.0),
        ] {
            assert_param(p, key, want, 1e-12);
        }
    }

    /// C₅ ≠ 0 is order 5 in cos φ: `nharmonic` with six coefficients, a6 =
    /// −C₅/4.184.
    #[test]
    fn dihedraltypes_funct_3_with_c5_is_dihedral_nharmonic() {
        let ff = read(&with_section(
            "dihedraltypes",
            "CT  CT  CT  CT  3  4.184  0.0  0.0  0.0  0.0  8.368",
        ));
        let (_, p) = only_type(&ff, "dihedral", "nharmonic");
        assert_param(p, "a1", 1.0, 1e-12);
        assert_param(p, "a6", -2.0, 1e-12);
        assert_eq!(p.get("a7").and_then(|v| v.as_num()), None);
    }

    /// GROMACS `X` is the canonical empty-endpoint wildcard.
    #[test]
    fn x_endpoint_is_the_empty_wildcard() {
        let ff = read(&with_section(
            "dihedraltypes",
            "X  CT  CT  X  3  0.62760  1.88280  0.00000  -2.51040  0.00000  0.00000",
        ));
        let (ends, _) = only_type(&ff, "dihedral", "multi/harmonic");
        assert_eq!(ends, ["", "CT", "CT", ""]);
    }

    /// k = 4.184 kJ/mol ÷ 4.184 = 1 kcal/mol; n = 3; φ_s = 0.
    #[test]
    fn dihedraltypes_funct_1_is_dihedral_periodic() {
        let ff = read(&with_section(
            "dihedraltypes",
            "CT  CT  CT  CT  1  0.0  4.184  3",
        ));
        let (ends, p) = only_type(&ff, "dihedral", "periodic");
        assert_eq!(ends, ["CT", "CT", "CT", "CT"]);
        assert_param(p, "k", 1.0, 1e-12);
        assert_param(p, "periodicity", 3.0, 1e-12);
        assert_param(p, "phase", 0.0, 1e-12);
    }

    /// k = 10.46 kJ/mol ÷ 4.184 = 2.5 kcal/mol; n = 2; φ_s = 180°, kept, and the
    /// row keeps its GROMACS atom order.
    #[test]
    fn dihedraltypes_funct_4_is_improper_periodic() {
        let ff = read(&with_section(
            "dihedraltypes",
            "X  X  N  H  4  180.0  10.46  2",
        ));
        let (ends, p) = only_type(&ff, "improper", "periodic");
        assert_eq!(ends, ["", "", "N", "H"]);
        assert_param(p, "k", 2.5, 1e-12);
        assert_param(p, "periodicity", 2.0, 1e-12);
        assert_eq!(p.get("phase").and_then(|v| v.as_num()), Some(180.0));
    }

    /// ½k_ξ(ξ−0)² = K(χ−0)² with K = 167.36 ÷ (2·4.184) = 20 kcal/mol/rad²;
    /// χ₀ = 0.
    #[test]
    fn dihedraltypes_funct_2_at_zero_is_improper_harmonic() {
        let ff = read(&with_section("dihedraltypes", "X  X  C  O  2  0.0  167.36"));
        let (ends, p) = only_type(&ff, "improper", "harmonic");
        assert_eq!(ends, ["", "", "C", "O"]);
        assert_param(p, "k", 20.0, 1e-12);
        assert_eq!(p.get("chi0").and_then(|v| v.as_num()), Some(0.0));
    }

    /// ξ₀ = ±180°: |ξ − 180°| wrapped is 180° − |ξ|, so the signed form is
    /// K(|φ| − 180°)².
    #[test]
    fn dihedraltypes_funct_2_at_180_is_improper_harmonic() {
        for xi0 in ["180.0", "-180.0"] {
            let ff = read(&with_section(
                "dihedraltypes",
                &format!("X  X  C  O  2  {xi0}  167.36"),
            ));
            let (_, p) = only_type(&ff, "improper", "harmonic");
            assert_eq!(p.get("chi0").and_then(|v| v.as_num()), Some(180.0));
        }
    }

    /// Signed ½k(ξ−ξ₀)² and unsigned K(|φ|−χ₀)² agree only at ξ₀ = 0, 180°.
    #[test]
    fn dihedraltypes_funct_2_off_zero_is_an_error() {
        let err = read_err(&with_section(
            "dihedraltypes",
            "X  X  C  O  2  10.0  167.36",
        ));
        assert_names(&err, &["dihedraltypes", "2", "10"]);
    }

    /// AMBER ff99SB's X-C-N-X style rows: consecutive funct-9 rows with equal
    /// labels are the terms of one `dihedral/periodic` type, in file order.
    #[test]
    fn consecutive_funct_9_rows_are_one_multi_term_type() {
        let ff = read(&with_section(
            "dihedraltypes",
            "N  CT  C  N  9  180.0  1.88280  1\nN  CT  C  N  9  180.0  6.61072  2\n\
             N  CT  C  N  9  180.0  4.184  3\nCT  CT  C  N  9  0.0  0.4184  2",
        ));
        let types = bonded(style(&ff, "dihedral", "periodic"));
        assert_eq!(types.len(), 2, "{types:?}");
        let (ends, p) = &types[0];
        assert_eq!(*ends, ["N", "CT", "C", "N"]);
        for (m, (k, n)) in [(0.45, 1.0), (1.58, 2.0), (1.0, 3.0)].iter().enumerate() {
            assert_param(p, &format!("k{}", m + 1), *k, 1e-12);
            assert_param(p, &format!("periodicity{}", m + 1), *n, 1e-12);
            assert_param(p, &format!("phase{}", m + 1), 180.0, 1e-12);
        }
        assert_eq!(
            p.get("k").and_then(|v| v.as_num()),
            None,
            "multi-term spelling only"
        );
        assert_param(types[1].1, "k", 0.1, 1e-12);
    }

    /// A funct-9 block repeated after another row is a second definition of the
    /// same type: equal is one type, different is the conflict GROMACS refuses.
    #[test]
    fn a_non_consecutive_funct_9_restatement_follows_the_conflict_rule() {
        let block = "N  CT  C  N  9  180.0  1.88280  1\nN  CT  C  N  9  180.0  6.61072  2";
        let same = format!("{block}\nCT  CT  C  N  9  0.0  0.4184  2\n{block}");
        assert_eq!(
            bonded(style(
                &read(&with_section("dihedraltypes", &same)),
                "dihedral",
                "periodic"
            ))
            .len(),
            2
        );
        let differ =
            format!("{block}\nCT  CT  C  N  9  0.0  0.4184  2\nN  CT  C  N  9  0.0  1.0  3");
        assert_names(
            &read_err(&with_section("dihedraltypes", &differ)),
            &["N  CT  C  N  9  0.0  1.0  3", "the same atoms"],
        );
    }

    /// GROMACS funct 5 (Fourier) is LAMMPS `dihedral opls` term for term:
    /// kₙ = Cₙ/4.184.
    #[test]
    fn dihedraltypes_funct_5_is_dihedral_opls() {
        let ff = read(&with_section(
            "dihedraltypes",
            "CT  CT  OH  HO  5  4.184  -2.092  1.2552  0.0",
        ));
        let (_, p) = only_type(&ff, "dihedral", "opls");
        for (key, want) in [("k1", 1.0), ("k2", -0.5), ("k3", 0.3), ("k4", 0.0)] {
            assert_param(p, key, want, 1e-12);
        }
    }

    /// RB keeps its constant: ΣC = 1 kJ/mol is a1 = 1/4.184 kcal/mol, not a
    /// refusal.
    #[test]
    fn dihedraltypes_funct_3_with_a_nonzero_sum_keeps_its_constant() {
        let ff = read(&with_section(
            "dihedraltypes",
            "HC  CT  CT  HC  3  1.00000  0.00000  0.00000  0.00000  0.00000  0.00000",
        ));
        let (_, p) = only_type(&ff, "dihedral", "multi/harmonic");
        assert_param(p, "a1", 1.0 / 4.184, 1e-15);
    }

    /// GROMACS's 2-name form: `j k` of a proper is `X j k X`, `i l` of a funct-2
    /// improper is `i X X l`.
    #[test]
    fn two_name_dihedraltypes_rows_are_wildcard_rows() {
        let ff = read(&with_section(
            "dihedraltypes",
            "CT  CT  1  0.0  4.184  3\nC  O  2  0.0  167.36",
        ));
        assert_eq!(
            only_type(&ff, "dihedral", "periodic").0,
            ["", "CT", "CT", ""]
        );
        assert_eq!(only_type(&ff, "improper", "harmonic").0, ["C", "", "", "O"]);
    }

    /// funct 1 and funct 3 on the same labels: two styles of one category would
    /// share a name, so both carry their function code.
    #[test]
    fn a_name_two_dihedral_styles_share_is_qualified_by_funct() {
        let ff = read(&with_section(
            "dihedraltypes",
            "X  CT  CT  X  9  0.0  4.184  3\nX  CT  CT  X  3  1.0  0.0  0.0  0.0  0.0  0.0",
        ));
        let names = |s: &str| -> Vec<String> {
            style(&ff, "dihedral", s)
                .type_rows()
                .iter()
                .map(|r| r.0.to_owned())
                .collect()
        };
        assert_eq!(names("periodic"), ["-CT-CT-@9"]);
        assert_eq!(names("multi/harmonic"), ["-CT-CT-@3"]);
    }

    // -- [ cmaptypes ] -----------------------------------------------------------

    /// A 2×2 map split over continuation lines: φ-major, values in kcal/mol.
    #[test]
    fn cmaptypes_is_a_cmap_charmm_grid_in_kcal() {
        let ff = read(&with_section(
            "cmaptypes",
            "C NH1 CT1 C NH1 1 2 2\\\n4.184 8.368\\\n-4.184 0.0",
        ));
        let types = ff.get_cmaptypes();
        assert_eq!(types.len(), 1);
        let t = types[0];
        assert_eq!(
            [&t.itom, &t.jtom, &t.ktom, &t.ltom, &t.mtom],
            ["C", "NH1", "CT1", "C", "NH1"]
        );
        let grid = t
            .params
            .get("grid")
            .and_then(|v| v.as_array())
            .expect("grid");
        assert_eq!(grid.shape(), [2, 2]);
        assert_eq!(
            grid.iter().copied().collect::<Vec<_>>(),
            [1.0, 2.0, -1.0, 0.0]
        );
    }

    #[test]
    fn a_cmap_grid_of_the_wrong_size_is_an_error() {
        let err = read_err(&with_section(
            "cmaptypes",
            "C NH1 CT1 C NH1 1 2 2 1.0 2.0 3.0",
        ));
        assert_names(&err, &["cmaptypes", "4 values"]);
        let err = read_err(&with_section(
            "cmaptypes",
            "C NH1 CT1 C NH1 1 2 3 1 2 3 4 5 6",
        ));
        assert_names(&err, &["2×3"]);
    }

    // -- refused sections --------------------------------------------------------

    // -- [ pairtypes ] ----------------------------------------------------------

    /// CHARMM-style: comb-rule 2, fudge 1 1; CT1's 1-4 σ/ε differ from its
    /// regular ones, HA's have none.
    const CHARMM_TYPES: &str = "[ defaults ]\n1  2  yes  1.0  1.0\n[ atomtypes ]\n\
        CT1  6  12.011  0.0  A  0.4  0.4184\n\
        HA  1  1.008  0.0  A  0.2  0.1\n\
        OH1  8  15.999  0.0  A  0.3  0.6\n";

    /// A self pairtype is the self row's `epsilon14` / `sigma14`, and the style
    /// becomes `lj/charmm` declared `one_four = "epsilon14"`, with `coul/charmm`.
    #[test]
    fn a_self_pairtype_is_lj_charmm_epsilon14() {
        let ff = read(&format!(
            "{CHARMM_TYPES}[ pairtypes ]\nCT1  CT1  1  0.3  0.04184\n"
        ));
        assert!(ff.get_style("pair", "lj/cut").is_none());
        let lj = style(&ff, "pair", "lj/charmm");
        assert_eq!(
            lj.params().get("one_four").and_then(|v| v.as_text()),
            Some("epsilon14")
        );
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("arithmetic")
        );
        let row = lj.get_pairtype("CT1", None).expect("CT1 self row");
        assert_param(&row.params, "epsilon", 0.1, 1e-12);
        assert_param(&row.params, "sigma", 4.0, 1e-12);
        assert_param(&row.params, "epsilon14", 0.01, 1e-12);
        assert_param(&row.params, "sigma14", 3.0, 1e-12);
        let ha = lj.get_pairtype("HA", None).expect("HA self row");
        assert_eq!(
            ha.params.get("epsilon14").and_then(|v| v.as_num()),
            None,
            "HA's 1-4 is its regular LJ"
        );
        assert!(ff.get_style("pair", "coul/charmm").is_some());
    }

    /// CT1-HA has no pairtype: GROMACS generates it from the regular
    /// parameters, where the mix of CT1's ε₁₄ and HA's ε would not, so the
    /// reader writes the cross row; CT1-OH1's pairtype is the mix of the self
    /// rows and needs none.
    #[test]
    fn a_pair_the_self_rows_would_mix_wrong_gets_a_cross_row() {
        let ff = read(&format!(
            "{CHARMM_TYPES}[ pairtypes ]\nCT1  CT1  1  0.3  0.04184\nOH1  OH1  1  0.3  0.6\n\
             CT1  OH1  1  0.3  0.15846\n"
        ));
        let lj = style(&ff, "pair", "lj/charmm");
        let cross = lj
            .get_pairtype("CT1", Some("HA"))
            .expect("CT1-HA cross row");
        // Regular: σ = (4 + 2)/2 = 3 Å, ε = √(0.1 · 0.1/4.184) kcal/mol.
        let eps = (0.1_f64 * 0.1 / 4.184).sqrt();
        assert_param(&cross.params, "sigma", 3.0, 1e-12);
        assert_param(&cross.params, "epsilon", eps, 1e-15);
        assert_param(&cross.params, "sigma14", 3.0, 1e-12);
        assert_param(&cross.params, "epsilon14", eps, 1e-15);
        // The file's 0.15846 is not √(0.04184 · 0.6) = 0.158443…: a cross row
        // as well.
        assert!(lj.get_pairtype("CT1", Some("OH1")).is_some());
    }

    /// With fudgeLJ ½, a pairtype (priced at full weight) is ε₁₄ = ε/½ so
    /// that `special_bonds` ½ × LJ(ε₁₄) is GROMACS's energy.
    #[test]
    fn a_pairtype_under_fudge_lj_is_divided_by_it() {
        let text = "[ defaults ]\n1  2  yes  0.5  0.8333\n[ atomtypes ]\n\
                    A  1.0  0.0  A  0.3  0.4184\n[ pairtypes ]\nA  A  1  0.25  0.4184\n";
        let ff = read(text);
        let row = style(&ff, "pair", "lj/charmm")
            .get_pairtype("A", None)
            .expect("A");
        assert_param(&row.params, "epsilon14", 0.2, 1e-12);
        assert_param(&row.params, "sigma14", 2.5, 1e-12);
        assert_eq!(ff.special_bonds().lj, [0.0, 0.0, 0.5]);
    }

    /// Pairtypes that restate what GROMACS would generate change nothing: the
    /// style stays `lj/cut`. A restatement rounded in print is not one.
    #[test]
    fn pairtypes_equal_to_the_generated_pairs_keep_lj_cut() {
        let ff = read(&format!(
            "{CHARMM_TYPES}[ pairtypes ]\nCT1  CT1  1  0.4  0.4184\nCT1  HA  1  0.3  0.204548\n"
        ));
        // 0.204548 is √(0.4184·0.1) rounded: 1e-12 does not hold it equal.
        assert!(ff.get_style("pair", "lj/charmm").is_some());
        let ff = read(&format!(
            "{CHARMM_TYPES}[ pairtypes ]\nCT1  CT1  1  0.4  0.4184\n"
        ));
        assert!(ff.get_style("pair", "lj/cut").is_some());
        assert!(ff.get_style("pair", "lj/charmm").is_none());
    }

    #[test]
    fn a_pairtype_on_an_unknown_type_or_funct_is_an_error() {
        let err = read_err(&format!(
            "{CHARMM_TYPES}[ pairtypes ]\nCT1  ZZ  1  0.3  0.1\n"
        ));
        assert_names(&err, &["pairtypes", "ZZ"]);
        let err = read_err(&format!(
            "{CHARMM_TYPES}[ pairtypes ]\nCT1  HA  2  0.3  0.1\n"
        ));
        assert_names(&err, &["pairtypes", "func 2"]);
    }

    #[test]
    fn fudge_lj_0_with_pairtypes_is_an_error() {
        let text = "[ defaults ]\n1  2  yes  0.0  1.0\n[ atomtypes ]\n\
                    A  1.0  0.0  A  0.3  0.4184\n[ pairtypes ]\nA  A  1  0.25  0.4184\n";
        assert_names(&read_err(text), &["fudgeLJ 0", "pairtypes"]);
    }

    /// A `[ nonbond_params ]` row is an explicit `lj/cut` cross row, in molrs
    /// units: σ = 0.3 nm = 3 Å, ε = 0.4184 kJ/mol = 0.1 kcal/mol.
    #[test]
    fn nonbond_params_is_an_explicit_lj_cut_cross_row() {
        let text = format!(
            "{DEFAULTS}[ atomtypes ]\n{OPLS_135}\nopls_140  HC  1  1.008  0.06  A  0.25  0.12552\n\
             [ nonbond_params ]\nopls_140  opls_135  1  0.3  0.4184\n"
        );
        let ff = read(&text);
        let lj = style(&ff, "pair", "lj/cut");
        let cross = lj
            .get_pairtype("opls_135", Some("opls_140"))
            .expect("cross row");
        assert_eq!(
            (cross.itom.as_str(), cross.jtom.as_str()),
            ("opls_135", "opls_140"),
            "stored in byte order"
        );
        assert_eq!(cross.name, "opls_135-opls_140");
        assert_param(&cross.params, "sigma", 3.0, 1e-12);
        assert_param(&cross.params, "epsilon", 0.1, 1e-12);
        assert_eq!(
            lj.params().get("mixing").and_then(|v| v.as_text()),
            Some("geometric")
        );
    }

    /// `j i` restating `i j` is one row; a different restatement is an error.
    #[test]
    fn nonbond_params_restated_in_reverse_is_one_row() {
        let types = "[ atomtypes ]\nA  1.0  0.0  A  0.3  0.4\nB  1.0  0.0  A  0.3  0.4\n";
        let same =
            format!("{DEFAULTS}{types}[ nonbond_params ]\nA  B  1  0.3  0.4\nB  A  1  0.3  0.4\n");
        // Two self rows and one cross row.
        assert_eq!(style(&read(&same), "pair", "lj/cut").type_rows().len(), 3);
        let differ =
            format!("{DEFAULTS}{types}[ nonbond_params ]\nA  B  1  0.3  0.4\nB  A  1  0.3  0.5\n");
        assert_names(&read_err(&differ), &["nonbond_params", "B  A"]);
        let unknown = with_section("nonbond_params", "A  B  1  0.3  0.4");
        assert_names(
            &read_err(&unknown),
            &["nonbond_params", "no [ atomtypes ] type"],
        );
    }

    #[test]
    fn nonbond_params_func_other_than_1_is_an_error() {
        let err = read_err(&with_section("nonbond_params", "A  B  2  1.0  2.0  3.0"));
        assert_names(&err, &["nonbond_params", "i j func V W"]);
        let err = read_err(&with_section("nonbond_params", "A  B  2  1.0  2.0"));
        assert_names(&err, &["func 2"]);
    }

    #[test]
    fn nonbond_params_needs_defaults() {
        let err = read_err("[ nonbond_params ]\nA  B  1  0.3  0.4\n");
        assert_names(&err, &["[ defaults ]"]);
    }

    #[test]
    fn constrainttypes_is_an_error_naming_the_section() {
        let err = read_err(&with_section("constrainttypes", "CT  HC  1  0.109"));
        assert_names(&err, &["constrainttypes"]);
    }

    #[test]
    fn skipped_constrainttypes_is_read_past() {
        let text = format!(
            "{DEFAULTS}[ constrainttypes ]\nCT  HC  1  0.109\n\
             [ bondtypes ]\nCT  HC  1  0.10900  284512.0\n"
        );
        let ff = GromacsTopForcefieldReader::new()
            .with_skipped_directive("constrainttypes")
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("skipped section still refused: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    /// Skipping one section does not skip another.
    #[test]
    fn skipping_one_section_still_refuses_another() {
        let text = format!(
            "{DEFAULTS}[ constrainttypes ]\nCT  HC  1  0.109\n\
             [ implicit_genborn_params ]\nCT  0.1  1  1.9  0.875  0.72\n"
        );
        let err = GromacsTopForcefieldReader::new()
            .with_skipped_directive("constrainttypes")
            .read(text.as_bytes())
            .expect_err("implicit_genborn_params is not skipped");
        assert_names(&err.to_string(), &["implicit_genborn_params"]);
    }

    #[test]
    fn unknown_section_is_an_error_naming_it() {
        let err = read_err(&with_section("mystery_section", "a  b  c"));
        assert_names(&err, &["mystery_section"]);
    }

    /// `[ atoms ]` is topology: refused, naming the reader of whole topologies.
    #[test]
    fn atoms_section_is_an_error_naming_read_system() {
        let err = read_err(&with_section(
            "atoms",
            "1  opls_135  1  LIG  C1  1  -0.18  12.011",
        ));
        assert_names(&err, &["atoms", "read_system"]);
    }

    #[test]
    fn every_molecule_section_is_an_error_naming_read_system() {
        for (section, row) in [
            ("moleculetype", "LIG  3"),
            ("bonds", "1  2  1"),
            ("pairs", "1  4  1"),
            ("angles", "1  2  3  1"),
            ("dihedrals", "1  2  3  4  3"),
            ("exclusions", "1  2"),
            ("settles", "1  1  0.1  0.1633"),
            ("system", "test"),
            ("molecules", "LIG  1"),
        ] {
            let err = read_err(&with_section(section, row));
            assert_names(&err, &[section, "read_system"]);
        }
    }

    #[test]
    fn skipped_molecule_sections_are_read_past() {
        let text = format!(
            "{DEFAULTS}[ bondtypes ]\nCT  HC  1  0.10900  284512.0\n\
             [ moleculetype ]\nLIG  3\n\
             [ atoms ]\n1  opls_135  1  LIG  C1  1  -0.18  12.011\n"
        );
        let ff = GromacsTopForcefieldReader::new()
            .with_skipped_directive("moleculetype")
            .with_skipped_directive("atoms")
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("skipped sections still refused: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
        assert!(ff.get_atomtypes().is_empty(), "[ atoms ] defined a type");
    }

    // -- preprocessor ------------------------------------------------------------

    const ROW_A: &str = "CT  HC  1  0.10900  284512.0";
    const ROW_B: &str = "CT  CT  1  0.15290  224262.4";

    /// The bond label pairs read from `body`, which sits in `[ bondtypes ]`.
    fn bond_pairs(body: &str) -> Vec<(String, String)> {
        let ff = read(&format!("{DEFAULTS}[ bondtypes ]\n{body}\n"));
        ff.get_bondtypes()
            .iter()
            .map(|t| (t.itom.clone(), t.jtom.clone()))
            .collect()
    }

    fn ct_hc() -> (String, String) {
        ("CT".to_owned(), "HC".to_owned())
    }

    fn ct_ct() -> (String, String) {
        ("CT".to_owned(), "CT".to_owned())
    }

    #[test]
    fn ifdef_block_is_read_when_the_define_comes_first() {
        let got = bond_pairs(&format!("#define FOO\n#ifdef FOO\n{ROW_A}\n#endif"));
        assert_eq!(got, [ct_hc()]);
    }

    #[test]
    fn ifdef_block_is_skipped_without_the_define() {
        let got = bond_pairs(&format!("#ifdef FOO\n{ROW_A}\n#endif\n{ROW_B}"));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn ifdef_block_is_skipped_when_the_define_comes_after() {
        let got = bond_pairs(&format!(
            "#ifdef FOO\n{ROW_A}\n#endif\n#define FOO\n{ROW_B}"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn ifndef_block_is_read_without_the_define() {
        let got = bond_pairs(&format!("#ifndef FOO\n{ROW_A}\n#endif"));
        assert_eq!(got, [ct_hc()]);
    }

    #[test]
    fn ifndef_block_is_skipped_with_the_define() {
        let got = bond_pairs(&format!(
            "#define FOO\n#ifndef FOO\n{ROW_A}\n#endif\n{ROW_B}"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn else_takes_the_other_branch() {
        let block = format!("#ifdef FOO\n{ROW_A}\n#else\n{ROW_B}\n#endif");
        assert_eq!(bond_pairs(&block), [ct_ct()]);
        assert_eq!(bond_pairs(&format!("#define FOO\n{block}")), [ct_hc()]);
    }

    /// A false outer block hides a true inner one; a true outer block
    /// evaluates its inner one.
    #[test]
    fn conditionals_nest() {
        let got = bond_pairs(&format!(
            "#define FOO\n\
             #ifdef BAR\n#ifdef FOO\n{ROW_A}\n#endif\n#endif\n\
             #ifdef FOO\n#ifndef BAR\n{ROW_B}\n#endif\n#endif"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    #[test]
    fn undef_removes_a_define() {
        let got = bond_pairs(&format!(
            "#define FOO\n#undef FOO\n#ifdef FOO\n{ROW_A}\n#endif\n{ROW_B}"
        ));
        assert_eq!(got, [ct_ct()]);
    }

    /// A macro with a body is recorded, not refused.
    #[test]
    fn define_with_a_body_is_accepted() {
        let got = bond_pairs(&format!(
            "#define improper_Z_N_X_Y 180.0 4.6024 2\n#ifdef improper_Z_N_X_Y\n{ROW_A}\n#endif"
        ));
        assert_eq!(got, [ct_hc()]);
    }

    /// A row's token naming a define is its body, as GROMACS's cpp does it.
    #[test]
    fn a_define_is_expanded_in_a_row() {
        let ff = read(&format!(
            "{DEFAULTS}#define improper_Z_N_X_Y 180.0 4.6024 2\n[ dihedraltypes ]\n\
             X  X  N  H  4  improper_Z_N_X_Y\n"
        ));
        let (_, p) = only_type(&ff, "improper", "periodic");
        assert_param(p, "k", 1.1, 1e-12);
        assert_param(p, "periodicity", 2.0, 1e-12);
    }

    /// A trailing backslash continues the line.
    #[test]
    fn a_trailing_backslash_continues_the_row() {
        let got = bond_pairs("CT  HC  1 \\\n 0.10900  284512.0");
        assert_eq!(got, [ct_hc()]);
    }

    #[test]
    fn if_directive_is_an_error() {
        let err = read_err(&format!("{DEFAULTS}#if FOO\n#endif\n"));
        assert_names(&err, &["#if"]);
    }

    #[test]
    fn elif_directive_is_an_error() {
        let err = read_err(&format!("{DEFAULTS}#ifdef FOO\n#elif BAR\n#endif\n"));
        assert_names(&err, &["#elif"]);
    }

    /// `#include` resolves relative to the including file, at every depth.
    #[test]
    fn include_resolves_relative_to_the_including_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sub = dir.path().join("ff");
        std::fs::create_dir(&sub).expect("mkdir");
        std::fs::write(
            dir.path().join("forcefield.itp"),
            format!("{DEFAULTS}#include \"ff/ffbonded.itp\"\n"),
        )
        .expect("write top");
        std::fs::write(sub.join("ffbonded.itp"), "#include \"more.itp\"\n").expect("write mid");
        std::fs::write(sub.join("more.itp"), format!("[ bondtypes ]\n{ROW_A}\n"))
            .expect("write leaf");

        let path = dir.path().join("forcefield.itp");
        let ff = GromacsTopForcefieldReader::new()
            .with_include(true)
            .read(path.as_path())
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
    }

    /// An include the including file's directory does not hold resolves
    /// against an include directory (GROMACS's share/top).
    #[test]
    fn include_resolves_against_an_include_dir() {
        let dir = tempfile::tempdir().expect("tempdir");
        let share = dir.path().join("share");
        std::fs::create_dir_all(share.join("x.ff")).expect("mkdir");
        std::fs::write(
            share.join("x.ff").join("forcefield.itp"),
            format!("{DEFAULTS}[ bondtypes ]\n{ROW_A}\n"),
        )
        .expect("write ff");
        let top = dir.path().join("topol.top");
        std::fs::write(&top, "#include \"x.ff/forcefield.itp\"\n").expect("write top");
        let ff = GromacsTopForcefieldReader::new()
            .with_include(true)
            .with_include_dir(&share)
            .read(top.as_path())
            .unwrap_or_else(|e| panic!("read: {e}"));
        assert_eq!(ff.get_bondtypes().len(), 1);
        let err = GromacsTopForcefieldReader::new()
            .with_include(true)
            .read(top.as_path())
            .expect_err("no include dir");
        assert_names(&err.to_string(), &["x.ff/forcefield.itp"]);
    }
}
