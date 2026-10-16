//! Regenerate `molrs/src/ff/params/oplsaa.rs` from GROMACS `oplsaa.ff`.
//!
//! ```text
//! cargo mrs-gen-opls --gromacs <path/to/share/top/oplsaa.ff>
//! ```
//!
//! A tool, not a test, and not published (`exclude = ["examples/"]`). It
//! checks the three input files against the SHA-256s pinned below, reads them
//! with molrs's own [`GromacsTopForcefieldReader`] — the only GROMACS parser in the
//! repository, which does every unit conversion — and writes the typed table.
//! The same pinned inputs always produce the same bytes.
//!
//! The one thing it reads from the text itself is a count: data rows per
//! section, and `#define` macros, for the provenance header. It never reads a
//! value.

use std::collections::{BTreeMap, HashSet};
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use molrs::core::constants::{COULOMB_REAL, OPLS_COULOMB_14, OPLS_LJ_14};
use molrs::core::unit_factors::KCAL_TO_KJ;
use molrs::ff::forcefield::ForceField;
use molrs::ff::forcefield::StyleDefs;
use molrs::ff::ir::Params;
use molrs::ff::ir::torsion::{MultiHarmonicForm, OplsForm};
use molrs::io::{gromacs::GromacsTopForcefieldReader, reader::ForceFieldReader};
use sha2::{Digest, Sha256};

/// The pinned GROMACS release.
const GROMACS_TAG: &str = "v2026.3";
/// The commit `GROMACS_TAG` points at.
const GROMACS_COMMIT: &str = "42105e4672205b4aa951962b8e3cdb4c27890da1";
/// The directory of the three files inside the GROMACS source tree.
const GROMACS_DIR: &str = "share/top/oplsaa.ff";

/// The three input files and their pinned SHA-256s, in reading order.
const PINNED: [(&str, &str); 3] = [
    (
        "forcefield.itp",
        "6eb8f0d08684e1ac6f525bb323eae75ed197b2980ddfde975a24ca1a8c931d06",
    ),
    (
        "ffnonbonded.itp",
        "fdf072dac9900c039b58e21323ce621b2112dc86c46a4b5e49e1398ae5d3b9f0",
    ),
    (
        "ffbonded.itp",
        "145e6924f52fae7e93108fa3d674f31a53aea10c80b13dd89abf0b011adfd4fa",
    ),
];

/// The only styles the table has a row type for, in the order the reader
/// defines them. `pair/coul/cut` carries constants and no rows; the embedded
/// assembly declares it itself.
const WALKED: [(&str, &str); 6] = [
    ("atom", "full"),
    ("pair", "lj/cut"),
    ("pair", "coul/cut"),
    ("bond", "harmonic"),
    ("angle", "harmonic"),
    ("dihedral", "multi/harmonic"),
];

/// GROMACS prints RB coefficients with 5 decimals; a row whose ΣCₙ (its
/// energy at φ = 180°, which the OPLS form fixes at 0) is further from 0 than
/// six roundings has an offset the table's Fourier row cannot hold.
const RB_SUM_TOL_KJ: f64 = 1e-4;

/// The table's own path, independent of where cargo was invoked.
const OUTPUT: &str = "src/ff/params/oplsaa.rs";

fn main() -> ExitCode {
    match run() {
        Ok(summary) => {
            println!("{summary}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("gen_opls_params: {e}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<String, String> {
    let dir = gromacs_dir()?;
    let digests = verify_pinned(&dir)?;
    let counts = SourceCount::of(&dir)?;
    let ff = GromacsTopForcefieldReader::new()
        .with_include(true)
        .with_skipped_directive("constrainttypes")
        .read(dir.join("forcefield.itp").as_path())
        .map_err(|e| e.to_string())?;
    let table = Table::from_force_field(&ff)?;
    table.check_counts(&counts)?;
    let text = table.render(&digests, &counts);
    let out = Path::new(env!("CARGO_MANIFEST_DIR")).join(OUTPUT);
    std::fs::write(&out, text).map_err(|e| format!("write {}: {e}", out.display()))?;
    Ok(format!(
        "wrote {}\n{}",
        out.display(),
        table.summary(&counts)
    ))
}

/// The directory named by `--gromacs <dir>`.
fn gromacs_dir() -> Result<PathBuf, String> {
    let usage = "usage: cargo mrs-gen-opls --gromacs <path/to/share/top/oplsaa.ff>";
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.as_slice() {
        [flag, dir] if flag == "--gromacs" => Ok(PathBuf::from(dir)),
        _ => Err(usage.to_owned()),
    }
}

/// Check each input against its pinned SHA-256; the digests, in `PINNED` order.
fn verify_pinned(dir: &Path) -> Result<Vec<String>, String> {
    let mut digests = Vec::with_capacity(PINNED.len());
    for (file, pinned) in PINNED {
        let path = dir.join(file);
        let bytes = std::fs::read(&path).map_err(|e| format!("read {}: {e}", path.display()))?;
        let actual = hex(&Sha256::digest(&bytes));
        if actual != pinned {
            return Err(format!(
                "SHA-256 mismatch for {file} ({}): pinned {pinned}, found {actual}. The table \
                 is generated from GROMACS {GROMACS_TAG} ({GROMACS_COMMIT}) only; a new \
                 release is a new pin, reviewed as such",
                path.display()
            ));
        }
        digests.push(actual);
    }
    Ok(digests)
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().fold(String::new(), |mut s, b| {
        let _ = write!(s, "{b:02x}");
        s
    })
}

/// The `std::f64::consts` a float literal can collide with. A literal bitwise
/// equal to one of these trips `clippy::approx_constant`, so it is emitted as
/// the constant's path instead — the same bits, spelled by name.
const NAMED_CONSTS: [(f64, &str); 19] = [
    (std::f64::consts::PI, "PI"),
    (std::f64::consts::TAU, "TAU"),
    (std::f64::consts::FRAC_PI_2, "FRAC_PI_2"),
    (std::f64::consts::FRAC_PI_3, "FRAC_PI_3"),
    (std::f64::consts::FRAC_PI_4, "FRAC_PI_4"),
    (std::f64::consts::FRAC_PI_6, "FRAC_PI_6"),
    (std::f64::consts::FRAC_PI_8, "FRAC_PI_8"),
    (std::f64::consts::FRAC_1_PI, "FRAC_1_PI"),
    (std::f64::consts::FRAC_2_PI, "FRAC_2_PI"),
    (std::f64::consts::FRAC_2_SQRT_PI, "FRAC_2_SQRT_PI"),
    (std::f64::consts::SQRT_2, "SQRT_2"),
    (std::f64::consts::FRAC_1_SQRT_2, "FRAC_1_SQRT_2"),
    (std::f64::consts::E, "E"),
    (std::f64::consts::LOG2_E, "LOG2_E"),
    (std::f64::consts::LOG2_10, "LOG2_10"),
    (std::f64::consts::LOG10_E, "LOG10_E"),
    (std::f64::consts::LOG10_2, "LOG10_2"),
    (std::f64::consts::LN_2, "LN_2"),
    (std::f64::consts::LN_10, "LN_10"),
];

/// An `f64` as Rust source: `std::f64::consts::<NAME>` (sign kept) when
/// bitwise equal to a named constant, else the round-tripping `{:?}` literal.
struct Lit(f64);

impl std::fmt::Display for Lit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let magnitude = self.0.abs();
        match NAMED_CONSTS
            .iter()
            .find(|(value, _)| value.to_bits() == magnitude.to_bits())
        {
            Some((_, name)) => {
                let sign = if self.0.is_sign_negative() { "-" } else { "" };
                write!(f, "{sign}std::f64::consts::{name}")
            }
            None => write!(f, "{:?}", self.0),
        }
    }
}

// ---------------------------------------------------------------------------
// What the source holds (counted, never parsed)
// ---------------------------------------------------------------------------

/// Data rows per `[ section ]` and the `#define` macros, counted across
/// `forcefield.itp` and its includes under the same `#ifdef` selection the
/// reader applies.
#[derive(Default)]
struct SourceCount {
    rows: BTreeMap<String, usize>,
    improper_macros: usize,
    dihedral_macros: usize,
}

impl SourceCount {
    fn of(dir: &Path) -> Result<Self, String> {
        let mut count = Self::default();
        let mut defines = HashSet::new();
        let mut section = String::new();
        count.file(&dir.join("forcefield.itp"), &mut defines, &mut section)?;
        Ok(count)
    }

    fn file(
        &mut self,
        path: &Path,
        defines: &mut HashSet<String>,
        section: &mut String,
    ) -> Result<(), String> {
        let text =
            std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
        // (enclosing text active, this branch taken)
        let mut conds: Vec<(bool, bool)> = Vec::new();
        for raw in text.lines() {
            let line = raw.split(';').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            let active = conds.last().is_none_or(|&(parent, taken)| parent && taken);
            if let Some(rest) = line.strip_prefix('#') {
                let mut words = rest.split_whitespace();
                let directive = words.next().unwrap_or("");
                let arg = words.next().unwrap_or("");
                match directive {
                    "ifdef" => conds.push((active, defines.contains(arg))),
                    "ifndef" => conds.push((active, !defines.contains(arg))),
                    "else" => {
                        if let Some(cond) = conds.last_mut() {
                            cond.1 = !cond.1;
                        }
                    }
                    "endif" => {
                        conds.pop();
                    }
                    _ if !active => {}
                    "define" => {
                        if arg.starts_with("improper_") {
                            self.improper_macros += 1;
                        } else if arg.starts_with("dih_") {
                            self.dihedral_macros += 1;
                        }
                        defines.insert(arg.to_owned());
                    }
                    "include" => {
                        let name = arg.trim_matches('"');
                        let dir = path.parent().unwrap_or(Path::new("."));
                        self.file(&dir.join(name), defines, section)?;
                    }
                    _ => {}
                }
                continue;
            }
            if !active {
                continue;
            }
            if let Some(header) = line.strip_prefix('[') {
                *section = header.trim_end_matches(']').trim().to_ascii_lowercase();
                continue;
            }
            *self.rows.entry(section.clone()).or_default() += 1;
        }
        Ok(())
    }

    fn rows(&self, section: &str) -> usize {
        self.rows.get(section).copied().unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// The table
// ---------------------------------------------------------------------------

struct AtomRow {
    name: String,
    class: String,
    mass: f64,
    charge: f64,
    sigma: f64,
    epsilon: f64,
    dummy: bool,
}

struct BondRow {
    ends: [String; 2],
    k: f64,
    r0: f64,
}

struct AngleRow {
    ends: [String; 3],
    k: f64,
    theta0: f64,
}

struct DihedralRow {
    ends: [String; 4],
    f: [f64; 4],
}

/// The rows the force field holds, in the table's shape.
struct Table {
    atoms: Vec<AtomRow>,
    bonds: Vec<BondRow>,
    angles: Vec<AngleRow>,
    dihedrals: Vec<DihedralRow>,
}

/// The value of `key`, with a negative zero written as `0.0` (the RB
/// inversion negates zero coefficients; the two compare equal).
fn param(params: &Params, key: &str, what: &str) -> Result<f64, String> {
    params
        .get(key)
        .and_then(|v| v.as_num())
        .map(|v| v + 0.0)
        .ok_or_else(|| format!("{what} has no `{key}` param"))
}

impl Table {
    /// Walk the read force field. Any style but the table's, an atom type
    /// without its `lj/cut` self row, or a cross row is an error.
    fn from_force_field(ff: &ForceField) -> Result<Self, String> {
        let lj = ff
            .get_style("pair", "lj/cut")
            .ok_or("the force field has no pair/lj/cut style")?;
        match lj.params().get("mixing").and_then(|v| v.as_text()) {
            Some("geometric") => {}
            other => {
                return Err(format!(
                    "pair/lj/cut mixing is {other:?}; OPLS-AA is geometric (comb-rule 3)"
                ));
            }
        }
        let sb = ff
            .declared_special_bonds()
            .ok_or("the force field declares no special bonds ([ defaults ])")?;
        if sb.lj != [0.0, 0.0, OPLS_LJ_14] || sb.coul != [0.0, 0.0, OPLS_COULOMB_14] {
            return Err(format!(
                "special bonds are {sb:?}; OPLS-AA is [0, 0, {OPLS_LJ_14}] for LJ and \
                 [0, 0, {OPLS_COULOMB_14}] for Coulomb (core::constants)"
            ));
        }

        let mut table = Self {
            atoms: Vec::new(),
            bonds: Vec::new(),
            angles: Vec::new(),
            dihedrals: Vec::new(),
        };
        for style in ff.styles() {
            let key = (style.category(), style.name());
            if !WALKED.contains(&key) {
                return Err(format!(
                    "{}/{} has no row type in the OPLS-AA table",
                    key.0, key.1
                ));
            }
            match style.defs() {
                StyleDefs::Atom(types) => {
                    for t in types {
                        let what = format!("atom/full {}", t.name);
                        let pair = lj
                            .get_pairtype(&t.name, None)
                            .ok_or_else(|| format!("{what} has no pair/lj/cut self row"))?;
                        let row = AtomRow {
                            name: t.name.clone(),
                            class: t
                                .params
                                .get("class")
                                .and_then(|v| v.as_text())
                                .unwrap_or(&t.name)
                                .to_owned(),
                            mass: param(&t.params, "mass", &what)?,
                            charge: param(&t.params, "charge", &what)?,
                            sigma: param(&pair.params, "sigma", &what)?,
                            epsilon: param(&pair.params, "epsilon", &what)?,
                            dummy: t.params.get("ptype").and_then(|v| v.as_text()) != Some("A"),
                        };
                        if row.dummy && (row.mass != 0.0 || row.epsilon != 0.0) {
                            return Err(format!(
                                "{what} is a non-A ptype with mass or epsilon: the table \
                                 has no virtual-site kind to hold it"
                            ));
                        }
                        table.atoms.push(row);
                    }
                }
                StyleDefs::Pair(types) if style.name() == "lj/cut" => {
                    if let Some(t) = types.iter().find(|t| t.itom != t.jtom) {
                        return Err(format!(
                            "pair/lj/cut cross row {}-{} has no place in the table",
                            t.itom, t.jtom
                        ));
                    }
                }
                StyleDefs::Pair(types) => {
                    let p = style.params();
                    if !types.is_empty()
                        || p.get("coulomb").and_then(|v| v.as_num()) != Some(COULOMB_REAL)
                        || p.get("dielectric").and_then(|v| v.as_num()) != Some(1.0)
                    {
                        return Err(
                            "pair/coul/cut is not the row-less vacuum Coulomb the embedded \
                             assembly declares"
                                .to_owned(),
                        );
                    }
                }
                StyleDefs::Bond(types) => {
                    for t in types {
                        let what = format!("bond/harmonic {}", t.name);
                        table.bonds.push(BondRow {
                            ends: [t.itom.clone(), t.jtom.clone()],
                            k: param(&t.params, "k", &what)?,
                            r0: param(&t.params, "r0", &what)?,
                        });
                    }
                }
                StyleDefs::Angle(types) => {
                    for t in types {
                        let what = format!("angle/harmonic {}", t.name);
                        table.angles.push(AngleRow {
                            ends: [t.itom.clone(), t.jtom.clone(), t.ktom.clone()],
                            k: param(&t.params, "k", &what)?,
                            theta0: param(&t.params, "theta0", &what)?,
                        });
                    }
                }
                StyleDefs::Dihedral(types) => {
                    for t in types {
                        // The reader keeps RB exactly (`multi/harmonic`); the
                        // table holds its OPLS Fourier projection, exact when
                        // ΣCₙ = 0 (C₅ = 0 is what `multi/harmonic` means).
                        let what = format!("dihedral/multi/harmonic {}", t.name);
                        let a = [
                            param(&t.params, "a1", &what)?,
                            param(&t.params, "a2", &what)?,
                            param(&t.params, "a3", &what)?,
                            param(&t.params, "a4", &what)?,
                            param(&t.params, "a5", &what)?,
                        ];
                        let series = MultiHarmonicForm { a }.to_series();
                        let sum = series.energy(std::f64::consts::PI);
                        if sum.abs() * KCAL_TO_KJ.get() > RB_SUM_TOL_KJ {
                            return Err(format!(
                                "{what}: sum of C = {} kJ/mol is a constant offset the OPLS \
                                 Fourier row cannot hold",
                                sum * KCAL_TO_KJ.get()
                            ));
                        }
                        // The sum is zero to the tolerance above, so the
                        // cosines are the OPLS row (its constant is its own).
                        let f = OplsForm::nearest(&series).k;
                        table.dihedrals.push(DihedralRow {
                            ends: [
                                t.itom.clone(),
                                t.jtom.clone(),
                                t.ktom.clone(),
                                t.ltom.clone(),
                            ],
                            f,
                        });
                    }
                }
                // `StyleDefs` is non-exhaustive: improper, cmap and any later
                // category have no row type here.
                other => {
                    return Err(format!(
                        "{} styles have no row type in the OPLS-AA table",
                        other.category()
                    ));
                }
            }
        }
        Ok(table)
    }

    /// Every section emits at most the rows it read; the difference is the
    /// repeats the reader merged.
    fn check_counts(&self, counts: &SourceCount) -> Result<(), String> {
        for (section, emitted) in self.emitted() {
            let read = counts.rows(section);
            if read < emitted {
                return Err(format!(
                    "[ {section} ]: emitted {emitted} rows but counted only {read} in the source"
                ));
            }
        }
        Ok(())
    }

    fn emitted(&self) -> [(&'static str, usize); 4] {
        [
            ("atomtypes", self.atoms.len()),
            ("bondtypes", self.bonds.len()),
            ("angletypes", self.angles.len()),
            ("dihedraltypes", self.dihedrals.len()),
        ]
    }

    fn dummies(&self) -> Vec<&str> {
        self.atoms
            .iter()
            .filter(|a| a.dummy)
            .map(|a| a.name.as_str())
            .collect()
    }

    /// Rows read and emitted per section, one line each.
    fn summary(&self, counts: &SourceCount) -> String {
        let mut s = String::new();
        for (section, emitted) in self.emitted() {
            let _ = writeln!(
                s,
                "[ {section} ] read {}, emitted {emitted}",
                counts.rows(section)
            );
        }
        let _ = write!(
            s,
            "[ constrainttypes ] read past {} (not encoded)",
            counts.rows("constrainttypes")
        );
        s
    }

    fn render(&self, digests: &[String], counts: &SourceCount) -> String {
        let mut o = String::new();
        let merged = |section: &str, emitted: usize| counts.rows(section) - emitted;
        let dummies = self.dummies();
        let _ = write!(
            o,
            "\
//! OPLS-AA force-field parameters — GROMACS `oplsaa.ff`, in molrs units.
//!
//! DO NOT HAND-EDIT — regenerate with `cargo mrs-gen-opls --gromacs <path>`
//! (`molrs/examples/gen_opls_params.rs`), where `<path>` is the `oplsaa.ff`
//! directory of the GROMACS release below. The generator refuses any other
//! bytes. This is ordinary source, not a build artefact: how the table arrived
//! is recorded here, never in its name.
//!
//! # Source
//!
//! GROMACS {GROMACS_TAG}, commit `{GROMACS_COMMIT}`:
//!
//! ```text
"
        );
        for ((file, _), digest) in PINNED.iter().zip(digests) {
            let _ = writeln!(o, "//! {GROMACS_DIR}/{file}  sha256 {digest}");
        }
        let _ = write!(
            o,
            "\
//! ```
//!
//! Derived from GROMACS share/top/oplsaa.ff, © the GROMACS development team,
//! distributed under the GNU Lesser General Public License, version 2.1 or
//! later (LGPL-2.1-or-later); Abraham et al., SoftwareX 1–2, 19 (2015),
//! DOI 10.1016/j.softx.2015.06.001. OPLS-AA: Jorgensen, Maxwell, Tirado-Rives,
//! J. Am. Chem. Soc. 118, 11225 (1996), DOI 10.1021/ja9621760.
//!
//! # Conversions
//!
//! Every number is converted by `GromacsTopForcefieldReader`, the one GROMACS parser
//! in molrs; the generator only writes its result.
//!
//! | GROMACS | molrs |
//! |---|---|
//! | σ (nm), ε (kJ/mol) | σ × 10 (Å), ε / 4.184 (kcal/mol) |
//! | bond b₀ (nm), k_b (kJ/mol/nm², ½k_b form) | r0 = b₀ × 10 (Å), k = k_b / 418.4 / 2 (kcal/mol/Å², LAMMPS `K` form) |
//! | angle θ₀ (deg), k_θ (kJ/mol/rad², ½k_θ form) | θ₀ in degrees, k = k_θ / 4.184 / 2 (kcal/mol/rad², LAMMPS `K` form) |
//! | dihedral funct 3, Ryckaert–Bellemans C₀..C₅ (kJ/mol) | exact RB → OPLS Fourier f₁..f₄ (GROMACS manual Eqs. 200–201), / 4.184 (kcal/mol) |
//! | `[ defaults ] 1 3 yes 0.5 0.5` | geometric mixing, 1-4 scale LJ 0.5 / Coulomb 0.5 |
//!
//! Classes are the GROMACS `bond_type` column (a row without one is its own
//! class); a GROMACS `X` endpoint is the empty-string wildcard.
//!
//! # Rows
//!
//! Rows read per section (under the `#ifdef` selection of a plain
//! `forcefield.itp`, i.e. without `HEAVY_H`) and rows emitted. A difference is
//! exactly the repeated rows the reader's conflict rule merged (equal
//! parameters are one type).
//!
//! | Section | Read | Emitted | Merged |
//! |---|---|---|---|
//! | `[ atomtypes ]` | {} | {} | {} |
//! | `[ bondtypes ]` | {} | {} | {} |
//! | `[ angletypes ]` | {} | {} | {} |
//! | `[ dihedraltypes ]` | {} | {} | {} |
//!
//! # Not encoded
//!
//! - `[ constrainttypes ]` ({} rows): the virtual-site constraints of the
//!   rigid MNH3 / MNH2 / MCH3A / MCH3B groups and the angle-derived OH / SH
//!   constraints. molrs has no constraint category, so the section is read
//!   past.
//! - The {} improper `#define` macros (`improper_*`): GROMACS applies them by
//!   name from residue topologies, never by type lookup, and OPLS improper
//!   assignment is not modelled.
//! - The {} residue-specific dihedral `#define` macros (`dih_*`): likewise
//!   named explicitly by `.rtp` / `.top` files, never matched by type.
//! - `at.num`: nothing in molrs reads a per-type atomic number, and GROMACS's
//!   is wrong for at least `opls_009` (a united-atom CH₂ given 7). An element,
//!   when needed, derives from the mass.
//! - `ptype`: molrs has no virtual-site particle kind, so the {} `D`
//!   dummies are emitted as ordinary rows with GROMACS's own mass 0 and ε 0
//!   (a TIP4P M site keeps its charge): {}.

use crate::ff::params::{{OplsAngleRow, OplsAtomRow, OplsBondRow, OplsDihedralRow}};

/// The force field's own name.
pub const OPLSAA_NAME: &str = \"OPLS-AA\";

/// The combining rule — geometric in σ and ε (`[ defaults ]` comb-rule 3).
pub const OPLSAA_MIXING: &str = \"geometric\";

/// The {} `[ atomtypes ]` rows of `ffnonbonded.itp`, in file order.
#[rustfmt::skip]
pub const OPLSAA_ATOMS: &[OplsAtomRow] = &[
",
            counts.rows("atomtypes"),
            self.atoms.len(),
            merged("atomtypes", self.atoms.len()),
            counts.rows("bondtypes"),
            self.bonds.len(),
            merged("bondtypes", self.bonds.len()),
            counts.rows("angletypes"),
            self.angles.len(),
            merged("angletypes", self.angles.len()),
            counts.rows("dihedraltypes"),
            self.dihedrals.len(),
            merged("dihedraltypes", self.dihedrals.len()),
            counts.rows("constrainttypes"),
            counts.improper_macros,
            counts.dihedral_macros,
            dummies.len(),
            dummies.join(", "),
            self.atoms.len(),
        );
        for a in &self.atoms {
            let _ = writeln!(
                o,
                "    OplsAtomRow {{ name: {:?}, class: {:?}, mass: {}, charge: {}, sigma: {}, epsilon: {} }},",
                a.name,
                a.class,
                Lit(a.mass),
                Lit(a.charge),
                Lit(a.sigma),
                Lit(a.epsilon)
            );
        }
        let _ = write!(
            o,
            "\
];

/// The {} `[ bondtypes ]` funct-1 rows of `ffbonded.itp`, in file order.
///
/// `force_constant` is kcal/mol/Å² in molrs's — LAMMPS's — `k(r−r₀)²` form
/// (GROMACS's `k_b / 2`), and `r0` is Å.
#[rustfmt::skip]
pub const OPLSAA_BONDS: &[OplsBondRow] = &[
",
            self.bonds.len()
        );
        for b in &self.bonds {
            let _ = writeln!(
                o,
                "    OplsBondRow {{ i: {:?}, j: {:?}, force_constant: {}, r0: {} }},",
                b.ends[0],
                b.ends[1],
                Lit(b.k),
                Lit(b.r0)
            );
        }
        let _ = write!(
            o,
            "\
];

/// The {} `[ angletypes ]` funct-1 rows of `ffbonded.itp`, in file order.
///
/// `force_constant` is kcal/mol/rad² in molrs's — LAMMPS's — `k(θ−θ₀)²` form
/// (GROMACS's `k_θ / 2`), and `theta0` is degrees.
#[rustfmt::skip]
pub const OPLSAA_ANGLES: &[OplsAngleRow] = &[
",
            self.angles.len()
        );
        for a in &self.angles {
            let _ = writeln!(
                o,
                "    OplsAngleRow {{ i: {:?}, j: {:?}, k: {:?}, force_constant: {}, theta0: {} }},",
                a.ends[0],
                a.ends[1],
                a.ends[2],
                Lit(a.k),
                Lit(a.theta0)
            );
        }
        let _ = write!(
            o,
            "\
];

/// The {} `[ dihedraltypes ]` funct-3 rows of `ffbonded.itp`, in file order,
/// as OPLS Fourier coefficients (kcal/mol).
#[rustfmt::skip]
pub const OPLSAA_DIHEDRALS: &[OplsDihedralRow] = &[
",
            self.dihedrals.len()
        );
        for d in &self.dihedrals {
            let _ = writeln!(
                o,
                "    OplsDihedralRow {{ i: {:?}, j: {:?}, k: {:?}, l: {:?}, f1: {}, f2: {}, f3: {}, f4: {} }},",
                d.ends[0],
                d.ends[1],
                d.ends[2],
                d.ends[3],
                Lit(d.f[0]),
                Lit(d.f[1]),
                Lit(d.f[2]),
                Lit(d.f[3])
            );
        }
        o.push_str("];\n");
        o
    }
}
