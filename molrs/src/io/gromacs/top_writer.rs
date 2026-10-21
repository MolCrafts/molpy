//! GROMACS force-field directive writer.

use crate::ff::ir::Engine;
use crate::ff::style_registry::refuse_style;
use crate::io::null_cells::refuse_null_cells;
use crate::io::pdb::labels::{AtomLabels, LabelWidths};
use std::collections::{BTreeMap, HashMap, HashSet};

use crate::core::constants::VACUUM_DIELECTRIC;
use crate::core::unit_factors::{KCAL_ANGSTROM2_TO_KJ_NM2, KCAL_TO_KJ, NM_TO_ANGSTROM};
use crate::ff::forcefield::{AtomType, ForceField, Style};
use crate::ff::ir::CMAP_GRID;
use crate::ff::ir::CombiningRule;
use crate::ff::ir::Params;
use crate::ff::ir::same_lj;
use crate::ff::ir::torsion::nharmonic_coefficients;
use crate::ff::potential::pair::charmm::{charmm_mixing, charmm_pair_params};
use crate::ff::potential::{MAX_ATOMS_FOR_A_FULL_PAIR_LIST, intramolecular_pairs};
use crate::io::Sink;
use crate::io::writer::{ForceFieldWriteError, ForceFieldWriter, put_text};
use molrs::core::Frame;
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;

/// The Lennard-Jones styles, one of which carries `[ atomtypes ]` V/W.
const LJ_STYLES: [&str; 2] = ["lj/cut", "lj/charmm"];

/// Writer for GROMACS force-field directives.
///
/// `GromacsTopForcefieldWriter::new().with_precision(p)`, then
/// [`ForceFieldWriter::write`] (directives) or
/// [`write_system`](GromacsTopForcefieldWriter::write_system) (a whole
/// topology).
///
/// The inverse of
/// [`GromacsTopForcefieldReader`](crate::io::gromacs::GromacsTopForcefieldReader):
/// it writes a [`ForceField`] as GROMACS force-field **directives**, converting
/// the force-field IR (LAMMPS's definitions: `real` — Å, kcal/mol, degrees, e;
/// LAMMPS's un-halved `K`) to GROMACS's (nm, kJ/mol, degrees, e; ½k) at this
/// boundary only. No molecule section (`[ atoms ]`, `[ bonds ]`,
/// `[ angles ]`, `[ dihedrals ]`, `[ pairs ]`, …) is written: a force field
/// holds no molecule.
///
/// # Output
///
/// - **`[ defaults ]`** `1 <comb> yes <fudgeLJ> <fudgeQQ>`. comb is the
///   Lennard-Jones style's `mixing` — `arithmetic` → 2, `geometric` → 3 — or,
///   when none is declared, the rule the style is evaluated under (arithmetic,
///   2, for both `lj/cut` and `lj/charmm`). fudgeLJ / fudgeQQ are the 1-4
///   special-bond weights.
/// - **`[ atomtypes ]`** `name [bond_type] [at.num] mass charge ptype V W`, one
///   row per `atom/full` type: `mass` (amu), `charge` (e), `bond_type` (the
///   type's string param `class`) and `atomic_number` from the atom type
///   (choosing the 6-, 7- or 8-column form), `ptype` as declared or `A` (an
///   `atom/full` type is a real atom), and V = σ/10 (nm), W = ε·4.184 (kJ/mol)
///   from the type's self row of the Lennard-Jones style (`lj/cut` or
///   `lj/charmm`).
/// - **`[ nonbond_params ]`** `i j 1 V W`: every explicit `lj/cut` cross row
///   (CHARMM NBFIX and the like), and every `lj/charmm` cross row whose `epsilon`
///   / `sigma` are not the mix of the two self rows (to 10⁻¹² relative — the
///   reader adds such rows only to carry `epsilon14` / `sigma14`). Written only
///   when there is one.
/// - **`[ pairtypes ]`** `i j 1 V W`, from an `lj/charmm` declared
///   `one_four = "epsilon14"`: GROMACS prices a 1-4 pair of types `i`, `j` at
///   fudgeLJ × LJ(the comb-rule or `[ nonbond_params ]` parameters) unless a
///   pairtype gives its parameters; the IR prices it at the 1-4 weight ×
///   LJ(ε₁₄, σ₁₄) of that type pair. A pairtype (σ₁₄, fudgeLJ·ε₁₄) is written
///   for every type pair where the two differ — the inverse of the reader.
/// - **`[ bondtypes ]` / `[ angletypes ]` / `[ dihedraltypes ]` /
///   `[ cmaptypes ]`**, the inverse of the reader's function-code map:
///
/// | molrs style | Directive, funct | Columns (file units) |
/// |---|---|---|
/// | `bond/harmonic` | bondtypes 1 | b₀ = r0/10 nm; k_b = 2·k·418.4 kJ/mol/nm² |
/// | `bond/morse` | bondtypes 3 | b₀ = r0/10 nm; D = d0·4.184 kJ/mol; β = alpha·10 nm⁻¹ |
/// | `angle/harmonic` | angletypes 1 | θ₀ = theta0 (degrees); k_θ = 2·k·4.184 kJ/mol/rad² |
/// | `angle/charmm` | angletypes 5 | θ₀; k_θ = 2·k·4.184; r₁₃ = r_ub/10 nm; k_UB = 2·k_ub·418.4 kJ/mol/nm² |
/// | `dihedral/periodic`, one term | dihedraltypes 1 | φ_s = phase (degrees); k·4.184 kJ/mol; n |
/// | `dihedral/periodic`, m terms | dihedraltypes 9, m consecutive rows | each term as funct 1, in term order |
/// | `dihedral/charmm` with `w` = 0 | dihedraltypes 9 | as funct 1 |
/// | `dihedral/harmonic` k[1 + d cos nφ] | dihedraltypes 9 | φ_s = 0° (d = 1) or 180° (d = −1) |
/// | `dihedral/multi/harmonic`, `dihedral/nharmonic` (N ≤ 6) | dihedraltypes 3 | Cₙ = (−1)ⁿ·aₙ₊₁·4.184 kJ/mol (C₅ = 0 for multi/harmonic) |
/// | `dihedral/opls` | dihedraltypes 5 | Cₙ = kₙ·4.184 kJ/mol |
/// | `dihedral/class2` (its torsion; LAMMPS's cross terms are not IR) | dihedraltypes 9, a row per non-zero kₙ | φ_s = phiₙ + 180°; kₙ·4.184 kJ/mol; n |
/// | `improper/periodic` | dihedraltypes 4 | as funct 1; the atoms in the stored order |
/// | `improper/cvff` k[1 + d cos nφ] | dihedraltypes 4 | φ_s = 0° (d = 1) or 180° (d = −1) |
/// | `improper/harmonic` | dihedraltypes 2 | ξ₀ = chi0 (0° or 180°); k_ξ = 2·k·4.184 kJ/mol/rad² |
/// | `cmap/charmm` | cmaptypes 1 | `N N` and the grid ·4.184 kJ/mol, φ-major, 10 values a line |
///
/// Every row is exact: each prices the same energy, constant included, as the
/// style it comes from. The empty-endpoint wildcard is written as `X`.
///
/// A pair style's `cutoff` and `lj/charmm`'s `inner` are run settings (the
/// .mdp's `rvdw` / `rcoulomb` and switch), not force-field data, so they are
/// not written.
///
/// `pair/coul/cut`, `pair/coul/charmm` and the long-range
/// `pair/coul/long/pme` have no directive: the `.top` holds the charges
/// (`[ atomtypes ]`, `[ atoms ]`), and how GROMACS sums them — cut-off,
/// reaction field, Ewald, PME — is the `.mdp`'s `coulombtype` (with
/// `rcoulomb`, `fourier-spacing`, `pme-order`, `ewald-rtol`), chosen at run
/// time, as LAMMPS's `kspace_style` and OpenMM's `createSystem` arguments are.
/// GROMACS fixes its Coulomb constant (CODATA 2018, LAMMPS `real`'s × (1 +
/// 9.9·10⁻⁹)) as LAMMPS and OpenMM fix theirs, so the field's stated
/// `coulomb` is not written — an AMBER field's 332.0522173 is priced at
/// GROMACS's constant, 3.5·10⁻⁵ above it, as every engine but AMBER's own
/// prices it. Only dielectric 1 is accepted, and nothing is written; a
/// `coul/long/pme` stating its Ewald parameters (`alpha`, `order`,
/// `grid_*`, `.mdp` settings) is refused.
///
/// The force field must be in `real` units (the conversions above are from
/// Å and kcal/mol); another declared preset is refused.
///
/// # Refusals
///
/// What GROMACS force-field directives cannot express is an `Err` naming it,
/// never a silent drop or an invented value:
///
/// - any other style (`improper/mmff_oop`, `bond/class2`, …),
///   `dihedral/charmm` with `w` ≠ 0 (GROMACS prices a 1-4 pair by `[ pairs ]`,
///   never by a dihedral), `dihedral/nharmonic` with N > 6;
/// - `sixthpower` mixing; a non-zero 1-2 or 1-3 special-bond weight;
///   `lj/charmm` `epsilon14` / `sigma14` that no 1-4 pair is priced by (the
///   style is not `one_four = "epsilon14"`: LAMMPS prices them only inside
///   `dihedral charmm`);
/// - an atom type lacking `mass`, `charge` or its Lennard-Jones self row; a
///   Lennard-Jones row (self or cross) whose type is not an `atom/full` type,
///   or that lacks `sigma` or `epsilon`;
/// - a bonded type missing a parameter, carrying one with no column, with an
///   endpoint label that is neither an atom-type name nor a `class`, or on the
///   same labels as another type of its GROMACS table (GROMACS would read the
///   two as one); `improper/harmonic` with `chi0` ∉ {0°, 180°}; a non-integral
///   multiplicity or `sign` other than ±1.
///
/// # Systems
///
/// [`GromacsTopForcefieldWriter::write_system`] writes a force field **and** a
/// typed frame as one `.top`, the inverse of
/// [`GromacsTopForcefieldReader::read_system`](crate::io::gromacs::GromacsTopForcefieldReader::read_system):
/// `[ defaults ]`, `[ atomtypes ]`, `[ nonbond_params ]`, `[ pairtypes ]`
/// and `[ cmaptypes ]` as above, then one `[ moleculetype ]` (`nrexcl` 3) per
/// molecule (bond-graph component, a run of consecutive atoms), with each
/// `bonds` / `angles` / `dihedrals` /
/// `impropers` row written with its type's parameters on the line (one
/// funct-9 line per periodic term, a single term included), so no lookup can
/// pick another type; `cmaps` rows
/// are found by GROMACS's lookup, which must give the row's own grid.
/// The `[ atoms ]` residue number, residue name and atom name are the ones
/// [`write_gro`](crate::io::write_gro) writes for the same frame (the PDB
/// writer's labels at GROMACS's five columns): stated, else — no column, or
/// an empty or null cell — the molecule's number, `MOL`, and the element
/// numbered within its residue, so `grompp` finds the `.gro` and the `.top`
/// alike. A null `type`, `charge` or `mass` cell is refused by column and
/// rows.
/// `[ pairs ]` lists the frame's 1-4 pairs (funct 1, or with parameters of
/// their own from the override cells: funct 1 `σ ε` when only `epsilon` /
/// `sigma` differ, else funct 2 `fudgeQQ qᵢqⱼ 1 σ lj_scale·ε`), and
/// `[ exclusions ]` every pair of one molecule beyond three bonds the frame
/// does not price — so GROMACS prices exactly the frame's intramolecular
/// `pairs` (built by [`intramolecular_pairs`] when absent), and every pair
/// across molecules. Refused by name: a priced pair within three bonds that is
/// not a 1-4 pair (GROMACS excludes it) or a 1-4 pair beyond them, override cells
/// without `epsilon` and `sigma`, a crossterm GROMACS's lookup would give
/// another grid, a type name two styles of one category share, a molecule
/// whose atoms are not consecutive or a row across two molecules, more than
/// `MAX_ATOMS_FOR_A_FULL_PAIR_LIST` atoms, and the force-field refusals
/// above.
///
/// # Whole-FF serialization, not coefficient writing
///
/// molrs has two kinds of force-field writer. This one is **whole-FF
/// serialization**: it writes every type the [`ForceField`] holds, as a
/// force-field file, and takes no type labels. **Coefficient writing**
/// ([`LammpsForcefieldWriter`](crate::io::lammps::LammpsForcefieldWriter), LAMMPS only) answers "which coefficients
/// does this system's data file need" and is keyed by the system's
/// `TypeLabels`.
#[derive(Debug, Clone)]
pub struct GromacsTopForcefieldWriter {
    /// Decimal places for floating coefficients.
    pub precision: usize,
}

impl Default for GromacsTopForcefieldWriter {
    fn default() -> Self {
        Self { precision: 6 }
    }
}

/// One `[ *types ]` row's function code and values (file units). The
/// multiplicity of a periodic row is printed as an integer.
struct Line {
    funct: u32,
    values: Vec<f64>,
    /// Index of a value printed as an integer.
    integer: Option<usize>,
}

/// A GROMACS parameter table: directive and (funct 9 folded into 1) function
/// code.
type Table = (String, u32);

/// The GROMACS parameter table a row of `funct` is looked up in (funct 1 and
/// 9 share one): two types on the same labels in one table are one to GROMACS.
fn table_key(directive: &str, funct: u32) -> Table {
    let funct = if directive == "dihedraltypes" && funct == 9 {
        1
    } else {
        funct
    };
    (directive.to_owned(), funct)
}

impl GromacsTopForcefieldWriter {
    /// The writer, coefficients with 6 decimals.
    pub fn new() -> Self {
        Self::default()
    }

    /// Write floating coefficients with `precision` decimals.
    pub fn with_precision(mut self, precision: usize) -> Self {
        self.precision = precision;
        self
    }

    fn fmt_f(&self, v: f64) -> String {
        format!("{:.*}", self.precision, v)
    }

    /// The force field's one Lennard-Jones style, if any.
    fn lj_style<'f>(&self, ff: &'f ForceField) -> Result<Option<&'f Style>, String> {
        let styles: Vec<&Style> = LJ_STYLES
            .iter()
            .filter_map(|name| ff.get_style("pair", name))
            .collect();
        match styles[..] {
            [] => Ok(None),
            [one] => Ok(Some(one)),
            _ => Err("pair/lj/cut and pair/lj/charmm together: GROMACS has one \
                      Lennard-Jones table"
                .to_owned()),
        }
    }

    /// The `[ defaults ]` row: `1 comb yes fudgeLJ fudgeQQ`.
    fn defaults_row(&self, ff: &ForceField, lj: Option<&Style>) -> Result<String, String> {
        let mixing = match lj {
            Some(style) if style.name() == "lj/charmm" => {
                charmm_mixing(style.params()).map_err(|e| e.to_string())?
            }
            Some(style) => match style.params().get("mixing").and_then(|v| v.as_text()) {
                Some(name) => {
                    CombiningRule::parse(name).map_err(|e| format!("pair/lj/cut: {e}"))?
                }
                None => CombiningRule::UNDECLARED,
            },
            None => CombiningRule::UNDECLARED,
        };
        let comb = match mixing {
            CombiningRule::Arithmetic => 2,
            CombiningRule::Geometric => 3,
            CombiningRule::SixthPower => {
                return Err(format!(
                    "Lennard-Jones mixing '{}' has no GROMACS comb-rule (2 is arithmetic, 3 \
                     geometric)",
                    mixing.name()
                ));
            }
        };
        let sb = ff.special_bonds();
        for (idx, order) in [(0, "1-2"), (1, "1-3")] {
            if sb.lj[idx] != 0.0 || sb.coul[idx] != 0.0 {
                return Err(format!(
                    "special-bond {order} weights (lj {}, coul {}) are not zero: GROMACS \
                     excludes 1-2 and 1-3 pairs, and gen-pairs scales only 1-4 pairs",
                    sb.lj[idx], sb.coul[idx]
                ));
            }
        }
        Ok(format!(
            "  1  {comb}  yes  {}  {}\n",
            self.fmt_f(sb.lj[2]),
            self.fmt_f(sb.coul[2]),
        ))
    }

    /// The `[ atomtypes ]` row of `t`, with σ/ε from its Lennard-Jones self row.
    /// `system`: the `[ atoms ]` rows carry each atom's mass and charge, so a
    /// type without them is written 0.
    fn atomtypes_row(
        &self,
        t: &AtomType,
        lj: Option<&Style>,
        system: bool,
    ) -> Result<String, String> {
        let p = &t.params;
        let name = &t.name;
        let need = |key: &str| match p.get(key).and_then(|v| v.as_num()) {
            Some(v) => Ok(v),
            None if system => Ok(0.0),
            None => Err(format!("atom type '{name}' has no {key}")),
        };
        let (mass, charge) = (need("mass")?, need("charge")?);
        let lj_name = lj.map_or("lj/cut", Style::name);
        let lj_row = lj.and_then(|s| s.get_pairtype(name, None)).ok_or_else(|| {
            format!("atom type '{name}' has no pair/{lj_name} self row (sigma, epsilon)")
        })?;
        let lj_need = |key: &str| {
            lj_row
                .params
                .get(key)
                .and_then(|v| v.as_num())
                .ok_or_else(|| format!("pair/{lj_name} self row '{name}' has no {key}"))
        };
        let (sigma, epsilon) = (lj_need("sigma")?, lj_need("epsilon")?);

        let mut cols = vec![name.clone()];
        if let Some(bond_type) = p.get("class").and_then(|v| v.as_text()) {
            cols.push(bond_type.to_owned());
        }
        if let Some(z) = p.get("atomic_number").and_then(|v| v.as_num()) {
            if z.fract() != 0.0 || !z.is_finite() {
                return Err(format!(
                    "atom type '{name}': atomic_number {z} is not an integer"
                ));
            }
            cols.push(format!("{}", z as i64));
        }
        cols.extend([
            self.fmt_f(mass),
            self.fmt_f(charge),
            p.get("ptype")
                .and_then(|v| v.as_text())
                .unwrap_or("A")
                .to_owned(),
            self.fmt_f(sigma / NM_TO_ANGSTROM.get()),
            self.fmt_f(epsilon * KCAL_TO_KJ.get()),
        ]);
        Ok(format!("  {}\n", cols.join("  ")))
    }

    /// `[ nonbond_params ]` and `[ pairtypes ]` of the Lennard-Jones style.
    fn cross_sections(
        &self,
        ff: &ForceField,
        lj: &Style,
        type_names: &HashSet<&str>,
    ) -> Result<(String, String), String> {
        let lj_name = lj.name();
        let rows = lj.type_rows();
        for (name, ends, params) in &rows {
            if let Some(end) = ends.iter().find(|e| !type_names.contains(*e)) {
                return Err(format!(
                    "pair/{lj_name} row '{name}': '{end}' is no atom/full type"
                ));
            }
            for key in ["sigma", "epsilon"] {
                if params.get(key).and_then(|v| v.as_num()).is_none() {
                    return Err(format!("pair/{lj_name} row '{name}' has no {key}"));
                }
            }
        }
        let line = |a: &str, b: &str, (eps, sigma): (f64, f64)| {
            format!(
                "  {a}  {b}  1  {}  {}\n",
                self.fmt_f(sigma / NM_TO_ANGSTROM.get()),
                self.fmt_f(eps * KCAL_TO_KJ.get())
            )
        };
        let mut nonbond = String::new();
        if lj_name == "lj/cut" {
            for (name, ends, params) in &rows {
                if ends[0] == ends[1] {
                    continue;
                }
                let need = |key: &str| {
                    params
                        .get(key)
                        .and_then(|v| v.as_num())
                        .ok_or_else(|| format!("pair/lj/cut cross row '{name}' has no {key}"))
                };
                nonbond.push_str(&line(ends[0], ends[1], (need("epsilon")?, need("sigma")?)));
            }
            return Ok((nonbond, String::new()));
        }

        // lj/charmm: the kernel's own rows, keyed as it keys them (`pair_key`).
        let kernel_rows: HashMap<String, Params> =
            lj.defs().kernel_type_params()?.into_iter().collect();
        let by_key: HashMap<&str, &Params> =
            kernel_rows.iter().map(|(k, p)| (k.as_str(), p)).collect();
        let mixing = charmm_mixing(lj.params()).map_err(|e| e.to_string())?;
        let has_14 = |p: &Params| {
            p.get("epsilon14").and_then(|v| v.as_num()).is_some()
                || p.get("sigma14").and_then(|v| v.as_num()).is_some()
        };
        let one_four = lj.params().get("one_four").and_then(|v| v.as_text());
        if one_four != Some("epsilon14") && rows.iter().any(|r| has_14(r.2)) {
            return Err(format!(
                "pair/lj/charmm carries epsilon14/sigma14 without one_four = \"epsilon14\": \
                 they price 1-4 pairs only through dihedral charmm w (LAMMPS), which GROMACS \
                 cannot express (one_four is {one_four:?})"
            ));
        }
        // Regular cross rows that are not the mix: [ nonbond_params ].
        for (_, ends, params) in &rows {
            if ends[0] == ends[1] {
                continue;
            }
            let own = |t: &str| {
                let p = by_key[t];
                (
                    p.get("epsilon").and_then(|v| v.as_num()).unwrap_or(0.0),
                    p.get("sigma").and_then(|v| v.as_num()).unwrap_or(0.0),
                )
            };
            let row = (
                params
                    .get("epsilon")
                    .and_then(|v| v.as_num())
                    .unwrap_or(0.0),
                params.get("sigma").and_then(|v| v.as_num()).unwrap_or(0.0),
            );
            if !same_lj(row, mixing.combine(own(ends[0]), own(ends[1]))) {
                nonbond.push_str(&line(ends[0], ends[1], row));
            }
        }
        let mut pairtypes = String::new();
        let weight = ff.special_bonds().lj[2];
        if one_four == Some("epsilon14") && weight != 0.0 {
            // Every type pair whose 1-4 parameters can differ from the
            // generated ones: one end has its own, or a cross row.
            let with_14: Vec<&str> = rows
                .iter()
                .filter(|r| r.1[0] == r.1[1] && has_14(r.2))
                .map(|r| r.1[0])
                .collect();
            let mut atoms: Vec<&str> = type_names.iter().copied().collect();
            atoms.sort_unstable();
            let mut pairs: Vec<(&str, &str)> = Vec::new();
            for &a in &with_14 {
                for &b in &atoms {
                    pairs.push(if a <= b { (a, b) } else { (b, a) });
                }
            }
            for (_, ends, _) in &rows {
                if ends[0] != ends[1] {
                    pairs.push((ends[0], ends[1]));
                }
            }
            pairs.sort_unstable();
            pairs.dedup();
            for (a, b) in pairs {
                if !by_key.contains_key(a) || !by_key.contains_key(b) {
                    continue;
                }
                let (regular, one_four) =
                    charmm_pair_params(&by_key, mixing, a, b).map_err(|e| e.to_string())?;
                if !same_lj(regular, one_four) {
                    pairtypes.push_str(&line(a, b, (one_four.0 * weight, one_four.1)));
                }
            }
        }
        Ok((nonbond, pairtypes))
    }

    /// A line's function code and values, as written after its labels.
    fn render_values(&self, line: &Line) -> String {
        let mut text = line.funct.to_string();
        for (i, v) in line.values.iter().enumerate() {
            text.push_str("  ");
            text.push_str(&if line.integer == Some(i) {
                format!("{}", *v as i64)
            } else {
                self.fmt_f(*v)
            });
        }
        text
    }

    /// The `[ *types ]` lines of one bonded type of `style`.
    fn bonded_lines(
        &self,
        style: &Style,
        name: &str,
        p: &Params,
    ) -> Result<Vec<Line>, ForceFieldWriteError> {
        let what = format!("{}/{} type '{name}'", style.category(), style.name());
        let allowed = |keys: &[&str]| -> Result<(), String> {
            match p.iter_parameters().find(|(key, _)| !keys.contains(key)) {
                Some((key, _)) => Err(format!("{what}: parameter '{key}' has no GROMACS column")),
                None => Ok(()),
            }
        };
        let need = |key: &str| {
            p.get(key)
                .and_then(|v| v.as_num())
                .ok_or_else(|| format!("{what} has no {key}"))
        };
        let whole = |key: &str| -> Result<f64, String> {
            let n = need(key)?;
            if n.fract() != 0.0 || !n.is_finite() {
                return Err(format!("{what}: {key} {n} is not an integer"));
            }
            Ok(n)
        };
        let sign_phase = |key: &str| -> Result<f64, String> {
            match need(key)? {
                1.0 => Ok(0.0),
                -1.0 => Ok(180.0),
                d => Err(format!("{what}: {key} {d} is not ±1")),
            }
        };
        let one = |funct: u32, values: Vec<f64>, integer: Option<usize>| {
            Ok(vec![Line {
                funct,
                values,
                integer,
            }])
        };
        let kb = KCAL_ANGSTROM2_TO_KJ_NM2.get();
        match (style.category(), style.name()) {
            ("bond", "harmonic") => {
                allowed(&["r0", "k"])?;
                // LAMMPS K → GROMACS ½k_b: k_b = 2K.
                one(
                    1,
                    vec![need("r0")? / NM_TO_ANGSTROM.get(), 2.0 * need("k")? * kb],
                    None,
                )
            }
            ("bond", "morse") => {
                allowed(&["d0", "alpha", "r0"])?;
                one(
                    3,
                    vec![
                        need("r0")? / NM_TO_ANGSTROM.get(),
                        need("d0")? * KCAL_TO_KJ.get(),
                        need("alpha")? * NM_TO_ANGSTROM.get(),
                    ],
                    None,
                )
            }
            ("angle", "harmonic") => {
                allowed(&["theta0", "k"])?;
                // LAMMPS K → GROMACS ½k_θ: k_θ = 2K.
                one(
                    1,
                    vec![need("theta0")?, 2.0 * need("k")? * KCAL_TO_KJ.get()],
                    None,
                )
            }
            ("angle", "charmm") => {
                allowed(&["k", "theta0", "k_ub", "r_ub"])?;
                one(
                    5,
                    vec![
                        need("theta0")?,
                        2.0 * need("k")? * KCAL_TO_KJ.get(),
                        need("r_ub")? / NM_TO_ANGSTROM.get(),
                        2.0 * need("k_ub")? * kb,
                    ],
                    None,
                )
            }
            ("dihedral" | "improper", "periodic") => {
                if p.get("k1").and_then(|v| v.as_num()).is_none() {
                    allowed(&["k", "periodicity", "phase"])?;
                    let funct = if style.category() == "dihedral" { 1 } else { 4 };
                    return one(
                        funct,
                        vec![
                            p.get("phase").and_then(|v| v.as_num()).unwrap_or(0.0),
                            need("k")? * KCAL_TO_KJ.get(),
                            whole("periodicity")?,
                        ],
                        Some(2),
                    );
                }
                if style.category() == "improper" {
                    return Err(format!(
                        "{what} has several terms: dihedraltypes funct 4 holds one"
                    )
                    .into());
                }
                let mut lines = Vec::new();
                let mut m = 1;
                while p.get(&format!("k{m}")).and_then(|v| v.as_num()).is_some() {
                    lines.push(Line {
                        funct: 9,
                        values: vec![
                            p.get(&format!("phase{m}"))
                                .and_then(|v| v.as_num())
                                .unwrap_or(0.0),
                            need(&format!("k{m}"))? * KCAL_TO_KJ.get(),
                            whole(&format!("periodicity{m}"))?,
                        ],
                        integer: Some(2),
                    });
                    m += 1;
                }
                let terms = m - 1;
                if let Some((key, _)) = p.iter_parameters().find(|(key, _)| {
                    ["k", "periodicity", "phase"].iter().all(|prefix| {
                        key.strip_prefix(prefix)
                            .and_then(|i| i.parse::<usize>().ok())
                            .is_none_or(|i| i == 0 || i > terms)
                    })
                }) {
                    return Err(format!("{what}: parameter '{key}' has no GROMACS column").into());
                }
                Ok(lines)
            }
            ("dihedral", "charmm") => {
                allowed(&["k", "periodicity", "phase", "w"])?;
                if p.get("w").and_then(|v| v.as_num()).unwrap_or(0.0) != 0.0 {
                    return Err(format!(
                        "{what}: w = {} prices the dihedral's 1-4 pair, which GROMACS prices \
                         by [ pairs ], never by a dihedral",
                        need("w")?
                    )
                    .into());
                }
                one(
                    9,
                    vec![
                        p.get("phase").and_then(|v| v.as_num()).unwrap_or(0.0),
                        need("k")? * KCAL_TO_KJ.get(),
                        whole("periodicity")?,
                    ],
                    Some(2),
                )
            }
            ("dihedral", "harmonic") | ("improper", "cvff") => {
                allowed(&["k", "sign", "periodicity"])?;
                let funct = if style.category() == "dihedral" { 9 } else { 4 };
                one(
                    funct,
                    vec![
                        sign_phase("sign")?,
                        need("k")? * KCAL_TO_KJ.get(),
                        whole("periodicity")?,
                    ],
                    Some(2),
                )
            }
            // k[1 − cos(nφ − φₙ)] = k[1 + cos(nφ − φₙ − 180°)]: one funct-9 row
            // per non-zero term, constant included.
            ("dihedral", "class2") => {
                allowed(&["k1", "phi1", "k2", "phi2", "k3", "phi3"])?;
                let lines: Vec<Line> = (1..=3)
                    .filter_map(|n| {
                        let k = p
                            .get(&format!("k{n}"))
                            .and_then(|v| v.as_num())
                            .unwrap_or(0.0);
                        (k != 0.0).then(|| Line {
                            funct: 9,
                            values: vec![
                                p.get(&format!("phi{n}"))
                                    .and_then(|v| v.as_num())
                                    .unwrap_or(0.0)
                                    + 180.0,
                                k * KCAL_TO_KJ.get(),
                                n as f64,
                            ],
                            integer: Some(2),
                        })
                    })
                    .collect();
                if lines.is_empty() {
                    return one(9, vec![0.0, 0.0, 1.0], Some(2));
                }
                Ok(lines)
            }
            ("dihedral", "multi/harmonic" | "nharmonic") => {
                let a: Vec<f64> = if style.name() == "nharmonic" {
                    nharmonic_coefficients(p).map_err(|e| format!("{what}: {e}"))?
                } else {
                    allowed(&["a1", "a2", "a3", "a4", "a5"])?;
                    (1..=5)
                        .map(|i| {
                            p.get(&format!("a{i}"))
                                .and_then(|v| v.as_num())
                                .unwrap_or(0.0)
                        })
                        .collect()
                };
                if a.len() > 6 {
                    return Err(format!(
                        "{what}: N = {} is above Ryckaert-Bellemans' C0..C5 (N ≤ 6)",
                        a.len()
                    )
                    .into());
                }
                let mut c = [0.0; 6];
                for (n, &x) in a.iter().enumerate() {
                    c[n] = if n % 2 == 0 { x } else { -x } * KCAL_TO_KJ.get();
                }
                one(3, c.to_vec(), None)
            }
            ("dihedral", "opls") => {
                allowed(&["k1", "k2", "k3", "k4"])?;
                // An absent k_n is a zero term, as the kernel reads it.
                let f = |key: &str| {
                    p.get(key).and_then(|v| v.as_num()).unwrap_or(0.0) * KCAL_TO_KJ.get()
                };
                one(5, vec![f("k1"), f("k2"), f("k3"), f("k4")], None)
            }
            ("improper", "harmonic") => {
                allowed(&["k", "chi0"])?;
                let chi0 = p.get("chi0").and_then(|v| v.as_num()).unwrap_or(0.0);
                if chi0 != 0.0 && chi0 != 180.0 {
                    return Err(format!(
                        "{what}: chi0 = {chi0} deg; dihedraltypes funct 2 is signed and agrees \
                         with K(|phi| - chi0)^2 only at chi0 = 0 and 180"
                    )
                    .into());
                }
                one(2, vec![chi0, 2.0 * need("k")? * KCAL_TO_KJ.get()], None)
            }
            (category, style) => Err(refuse_style(Engine::Gromacs, category, style).into()),
        }
    }
}

/// The styles this writer reads: `Ok` when `style` is one, `Err` naming it
/// otherwise. A style parameter stated at its spec default (`coul/cut`'s
/// `delta = 0`) prices as its absence does and needs no directive
/// ([`StyleSpec::style_params_off_default`]); any other value a directive
/// cannot carry is refused.
///
/// [`StyleSpec::style_params_off_default`]: crate::ff::ir::StyleSpec::style_params_off_default
fn check_style(style: &Style) -> Result<(), ForceFieldWriteError> {
    let declared = |key: &str, value: f64| {
        style
            .params()
            .get(key)
            .and_then(|v| v.as_num())
            .is_none_or(|v| v == value)
    };
    let stated = crate::ff::style_registry::with_global_registry(|r| {
        r.style(style.category(), style.name())
            .map(|(spec, _)| spec.style_params_off_default(style.params()))
    })
    .unwrap_or_else(|| style.params().clone());
    let extra_params = |keys: &[&str]| {
        stated
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
            .map(|(k, _)| k)
            .chain(stated.iter().filter_map(|(k, v)| v.as_text().map(|_| k)))
            .find(|k| !keys.contains(k))
            .map(str::to_owned)
    };
    match (style.category(), style.name()) {
        ("atom", "full")
        | ("bond", "harmonic" | "morse")
        | ("angle", "harmonic" | "charmm")
        | (
            "dihedral",
            "periodic" | "opls" | "multi/harmonic" | "nharmonic" | "harmonic" | "charmm" | "class2",
        )
        | ("improper", "periodic" | "harmonic" | "cvff")
        | ("cmap", "charmm") => Ok(()),
        // A pair `cutoff` (and the CHARMM switch's `inner`) is a run setting
        // (GROMACS keeps it in the .mdp), not force-field data.
        ("pair", "lj/cut") => match extra_params(&["cutoff", "mixing"]) {
            Some(key) => {
                Err(format!("pair/lj/cut style param '{key}' has no GROMACS directive").into())
            }
            None => Ok(()),
        },
        ("pair", "lj/charmm") => match extra_params(&["cutoff", "inner", "mixing", "one_four"]) {
            Some(key) => {
                Err(format!("pair/lj/charmm style param '{key}' has no GROMACS directive").into())
            }
            None => Ok(()),
        },
        // The Coulomb constant is GROMACS's own (as LAMMPS's is LAMMPS's):
        // the field's stated `coulomb` is not written, whatever it is.
        ("pair", name @ ("coul/cut" | "coul/charmm")) => {
            let extra = extra_params(&["coulomb", "dielectric", "cutoff", "inner"]);
            if extra.is_some()
                || !declared("dielectric", VACUUM_DIELECTRIC)
                || !style.type_rows().is_empty()
            {
                return Err(format!(
                    "pair/{name} {:?} has no GROMACS directive: only dielectric = \
                     {VACUUM_DIELECTRIC}, with no types, is implied by the directives",
                    style.params()
                )
                .into());
            }
            Ok(())
        }
        // A long-range Coulomb is the charges too: the method (PME, Ewald)
        // and its accuracy are the .mdp's.
        ("pair", "coul/long/pme") => {
            if let Some(key) = extra_params(&["coulomb", "cutoff"]) {
                return Err(format!(
                    "pair/coul/long/pme states its Ewald '{key}', which a .top has no place for: \
                     the long-range method and its accuracy are .mdp settings (coulombtype = PME, \
                     fourier-spacing, pme-order, ewald-rtol)"
                )
                .into());
            }
            if !style.type_rows().is_empty() {
                return Err("pair/coul/long/pme has types: the charges are the atom types'".into());
            }
            Ok(())
        }
        (category, name) => Err(refuse_style(Engine::Gromacs, category, name).into()),
    }
}

impl GromacsTopForcefieldWriter {
    /// The file's text: what [`write`](ForceFieldWriter::write) puts into its sink.
    fn text(&self, ff: &ForceField) -> Result<String, ForceFieldWriteError> {
        self.directives(ff, false)
    }
}

impl ForceFieldWriter for GromacsTopForcefieldWriter {
    fn write<'a>(
        &self,
        sink: impl Into<Sink<'a>>,
        ff: &ForceField,
    ) -> Result<(), ForceFieldWriteError> {
        put_text(sink.into(), &self.text(ff)?)
    }
}

impl GromacsTopForcefieldWriter {
    /// The force-field directives of `ff`. `system`: the directives of
    /// [`Self::write_system`] — no bonded `[ *types ]` tables but
    /// `[ cmaptypes ]` (its rows carry their parameters), and atom types
    /// without a mass or charge written 0.
    fn directives(&self, ff: &ForceField, system: bool) -> Result<String, ForceFieldWriteError> {
        if let Some(units) = ff.declared_units()
            && units != "real"
        {
            return Err(format!(
                "force field units '{units}': the GROMACS writer converts from real units \
                 (Å, kcal/mol)"
            )
            .into());
        }
        for style in ff.styles() {
            check_style(style)?;
        }
        let lj = self.lj_style(ff)?;

        let mut out = String::from("; Generated by molrs\n\n");
        out.push_str("[ defaults ]\n");
        out.push_str("; nbfunc  comb-rule  gen-pairs  fudgeLJ  fudgeQQ\n");
        out.push_str(&self.defaults_row(ff, lj)?);
        out.push('\n');

        let atom_types: Vec<&AtomType> = ff.get_atomtypes();
        let type_names: HashSet<&str> = atom_types.iter().map(|t| t.name.as_str()).collect();
        let (nonbond_params, pairtypes) = match lj {
            Some(lj) => self.cross_sections(ff, lj, &type_names)?,
            None => (String::new(), String::new()),
        };
        if !atom_types.is_empty() {
            out.push_str("[ atomtypes ]\n");
            out.push_str("; name  [bond_type]  [at.num]  mass  charge  ptype  sigma  epsilon\n");
            for t in &atom_types {
                out.push_str(&self.atomtypes_row(t, lj, system)?);
            }
            out.push('\n');
        }
        if !nonbond_params.is_empty() {
            out.push_str("[ nonbond_params ]\n; i  j  func  sigma  epsilon\n");
            out.push_str(&nonbond_params);
            out.push('\n');
        }
        if !pairtypes.is_empty() {
            out.push_str("[ pairtypes ]\n; i  j  func  sigma14  epsilon14\n");
            out.push_str(&pairtypes);
            out.push('\n');
        }

        // A bonded endpoint is the wildcard, an atom-type name or a class
        // (GROMACS `bond_type`).
        let mut labels = type_names;
        labels.extend(
            atom_types
                .iter()
                .filter_map(|t| t.params.get("class").and_then(|v| v.as_text())),
        );
        let resolve = |style: &Style, name: &str, ends: &[&str]| -> Result<Vec<String>, String> {
            ends.iter()
                .map(|end| {
                    if end.is_empty() {
                        Ok("X".to_owned())
                    } else if labels.contains(end) {
                        Ok((*end).to_owned())
                    } else {
                        Err(format!(
                            "{}/{} type '{name}': endpoint '{end}' is neither an atom-type name \
                             nor a bond_type",
                            style.category(),
                            style.name()
                        ))
                    }
                })
                .collect()
        };
        // Labels per GROMACS table, so that two types GROMACS would read as one
        // are refused.
        let mut seen: HashMap<(Table, Vec<String>), (String, Vec<String>)> = HashMap::new();
        let mut written: HashSet<(Table, Vec<String>, Vec<String>)> = HashSet::new();
        // A system's rows carry their parameters: no table to look them up in.
        let tables: &[(&str, &[&str], &str)] = if system {
            &[]
        } else {
            &[
                ("bondtypes", &["bond"], "; i  j  func  b0  kb"),
                (
                    "angletypes",
                    &["angle"],
                    "; i  j  k  func  th0  cth  [r13  kub]",
                ),
                (
                    "dihedraltypes",
                    &["dihedral", "improper"],
                    "; i  j  k  l  func  params",
                ),
            ]
        };
        for &(directive, categories, header) in tables {
            let mut rows = String::new();
            for style in ff
                .styles()
                .iter()
                .filter(|s| categories.contains(&s.category()))
            {
                for (name, ends, params) in style.type_rows() {
                    let cols = resolve(style, name, &ends)?;
                    let lines = self.bonded_lines(style, name, params)?;
                    let table = table_key(directive, lines[0].funct);
                    let values: Vec<String> =
                        lines.iter().map(|line| self.render_values(line)).collect();
                    // GROMACS finds a row in either orientation, so two rows
                    // on the same labels either way are one; equal ones are a
                    // harmless restatement.
                    let reversed: Vec<String> = cols.iter().rev().cloned().collect();
                    // The same rows on the same labels again would be read
                    // as more funct-9 terms of the first: written once.
                    if !written.insert((table.clone(), cols.clone(), values.clone())) {
                        continue;
                    }
                    for key in [cols.clone(), reversed] {
                        let entry = (name.to_owned(), values.clone());
                        if let Some((other, other_values)) =
                            seen.insert((table.clone(), key), entry)
                            && other != name
                            && other_values != values
                        {
                            return Err(format!(
                                "{}/{} type '{name}' and type '{other}' are on the same labels \
                                 {} of [ {directive} ] funct {}: GROMACS would read them as one",
                                style.category(),
                                style.name(),
                                cols.join(" "),
                                table.1
                            )
                            .into());
                        }
                    }
                    for value in values {
                        rows.push_str(&format!("  {}  {value}\n", cols.join("  ")));
                    }
                }
            }
            if !rows.is_empty() {
                out.push_str(&format!("[ {directive} ]\n{header}\n{rows}\n"));
            }
        }

        if let Some(cmap) = ff.get_style("cmap", "charmm") {
            let mut rows = String::new();
            for (name, ends, params) in cmap.type_rows() {
                let cols = resolve(cmap, name, &ends)?;
                if let Some((key, _)) = params.iter_parameters().next() {
                    return Err(format!(
                        "cmap/charmm type '{name}': parameter '{key}' has no GROMACS column"
                    )
                    .into());
                }
                let grid = params
                    .get(CMAP_GRID)
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("cmap/charmm type '{name}' has no grid"))?;
                let n = match grid.shape() {
                    [a, b] if a == b => *a,
                    shape => {
                        return Err(format!(
                            "cmap/charmm type '{name}': grid of shape {shape:?} is not N×N"
                        )
                        .into());
                    }
                };
                rows.push_str(&format!("{} 1 {n} {n}\\\n", cols.join(" ")));
                let values: Vec<String> = grid
                    .iter()
                    .map(|v| self.fmt_f(v * KCAL_TO_KJ.get()))
                    .collect();
                let chunks: Vec<String> = values.chunks(10).map(|c| c.join(" ")).collect();
                rows.push_str(&chunks.join("\\\n"));
                rows.push_str("\n\n");
            }
            if !rows.is_empty() {
                out.push_str(&format!("[ cmaptypes ]\n\n{rows}"));
            }
        }
        Ok(out)
    }

    /// Write `ff` and the typed `frame` into `sink` as one GROMACS topology:
    /// the force-field directives and one molecule type per molecule (module
    /// docs, "Systems"); the inverse of
    /// [`GromacsTopForcefieldReader::read_system`](crate::io::gromacs::GromacsTopForcefieldReader::read_system).
    /// Nothing is written when the topology is refused.
    ///
    /// # Errors
    ///
    /// Every refusal of the module documentation, and the sink's OS error.
    pub fn write_system<'a>(
        &self,
        sink: impl Into<Sink<'a>>,
        ff: &ForceField,
        frame: &Frame,
    ) -> Result<(), ForceFieldWriteError> {
        put_text(sink.into(), &self.system_text(ff, frame)?)
    }

    /// The topology [`write_system`](Self::write_system) writes.
    fn system_text(&self, ff: &ForceField, frame: &Frame) -> Result<String, ForceFieldWriteError> {
        let mut out = self.directives(ff, true)?;
        let atoms = frame.get("atoms").ok_or("frame has no atoms block")?;
        let n = atoms.n_rows().unwrap_or(0);
        if n > MAX_ATOMS_FOR_A_FULL_PAIR_LIST {
            return Err(format!(
                "{n} atoms: the topology states the frame's pairs molecule by molecule, \
                 which is not done above {MAX_ATOMS_FOR_A_FULL_PAIR_LIST} atoms"
            )
            .into());
        }
        let strings = |key: &str| atoms.get(key).and_then(|c| c.get::<String>());
        let types = strings("type").ok_or("atoms has no string type column")?;
        let charges = atoms
            .get("charge")
            .and_then(|c| c.get::<f64>())
            .ok_or("atoms has no charge column")?;
        let masses = atoms.get("mass").and_then(|c| c.get::<f64>());
        // A stated type, charge or mass has no stand-in; an unset name is
        // generated, as the `.gro` writer generates it (`null_cells`).
        refuse_null_cells(frame, "atoms", &["type", "charge", "mass"])
            .map_err(|e| e.to_string())?;
        let labels =
            AtomLabels::with_widths(frame, n, LabelWidths::GROMACS).map_err(|e| e.to_string())?;
        let atom_types: HashMap<&str, &AtomType> = ff
            .get_atomtypes()
            .into_iter()
            .map(|t| (t.name.as_str(), t))
            .collect();
        let full = |v: f64| format!("{v:?}");
        let column = |block: &str, key: &str| -> Result<Vec<usize>, String> {
            frame
                .get(block)
                .and_then(|b| b.get(key))
                .and_then(|c| c.get::<u64>())
                .map(|c| c.iter().map(|&v| v as usize).collect())
                .ok_or_else(|| format!("{block} has no index column {key}"))
        };
        const KEYS: [&str; 5] = ["atomi", "atomj", "atomk", "atoml", "atomm"];
        let rows_of = |block: &str, arity: usize| -> Result<Vec<(Vec<usize>, String)>, String> {
            let Some(b) = frame.get(block) else {
                return Ok(Vec::new());
            };
            let names = b
                .get("type")
                .and_then(|c| c.get::<String>())
                .ok_or_else(|| format!("{block} has no string type column"))?;
            let cols = KEYS[..arity]
                .iter()
                .map(|k| column(block, k))
                .collect::<Result<Vec<_>, _>>()?;
            Ok((0..names.len())
                .map(|r| (cols.iter().map(|c| c[r]).collect(), names[[r]].clone()))
                .collect())
        };

        // The molecules: bond-graph components (bonds, constraints), each a
        // run of consecutive atoms, as a GROMACS molecule is.
        let mut adjacent = vec![Vec::new(); n];
        for block in ["bonds", "constraints"] {
            if frame.get(block).is_some() {
                for (i, j) in column(block, "atomi")?
                    .into_iter()
                    .zip(column(block, "atomj")?)
                {
                    adjacent[i].push(j);
                    adjacent[j].push(i);
                }
            }
        }
        let mut molecule = vec![usize::MAX; n];
        let mut runs: Vec<(usize, usize)> = Vec::new();
        for start in 0..n {
            if molecule[start] != usize::MAX {
                continue;
            }
            let m = runs.len();
            molecule[start] = m;
            let (mut lo, mut hi) = (start, start);
            let mut stack = vec![start];
            while let Some(a) = stack.pop() {
                for &b in &adjacent[a] {
                    if molecule[b] == usize::MAX {
                        molecule[b] = m;
                        (lo, hi) = (lo.min(b), hi.max(b));
                        stack.push(b);
                    }
                }
            }
            runs.push((lo, hi + 1));
        }
        for (m, &(lo, hi)) in runs.iter().enumerate() {
            if let Some(a) = (lo..hi).find(|&a| molecule[a] != m) {
                return Err(format!(
                    "atom {} lies inside molecule {} (atoms {}..{}) but is not bonded to it: \
                     a GROMACS molecule is a run of consecutive atoms",
                    a + 1,
                    m + 1,
                    lo + 1,
                    hi
                )
                .into());
            }
        }
        let molecule_of = |what: &str, atoms_of: &[usize]| -> Result<usize, String> {
            let m = molecule[atoms_of[0]];
            if atoms_of.iter().any(|&a| molecule[a] != m) {
                return Err(format!(
                    "{what} {:?} spans two molecules",
                    atoms_of.iter().map(|a| a + 1).collect::<Vec<_>>()
                ));
            }
            Ok(m)
        };

        // Every bonded row, written with its type's parameters.
        let mut typed: HashMap<(&str, &str), (&Style, &Params)> = HashMap::new();
        for style in ff.styles() {
            for (name, _, params) in style.type_rows() {
                if typed
                    .insert((style.category(), name), (style, params))
                    .is_some()
                {
                    return Err(format!(
                        "{} type '{name}' is defined by two styles: a frame row naming it is \
                         ambiguous",
                        style.category()
                    )
                    .into());
                }
            }
        }
        let mut sections: Vec<BTreeMap<&str, String>> = vec![BTreeMap::new(); runs.len()];
        let local = |m: usize, atoms_of: &[usize]| -> String {
            atoms_of
                .iter()
                .map(|a| (a - runs[m].0 + 1).to_string())
                .collect::<Vec<_>>()
                .join("  ")
        };
        for (section, blocks) in [
            ("bonds", &[("bonds", "bond", 2)][..]),
            ("angles", &[("angles", "angle", 3)][..]),
            (
                "dihedrals",
                &[("dihedrals", "dihedral", 4), ("impropers", "improper", 4)][..],
            ),
        ] {
            for &(block, category, arity) in blocks {
                for (atoms_of, name) in rows_of(block, arity)? {
                    let (style, params) =
                        typed.get(&(category, name.as_str())).ok_or_else(|| {
                            format!(
                                "{block} row names type '{name}', which no {category} style has"
                            )
                        })?;
                    let m = molecule_of(block, &atoms_of)?;
                    let text = sections[m].entry(section).or_default();
                    for mut line in self.bonded_lines(style, &name, params)? {
                        // One periodic proper per line, as funct 9 writes
                        // each term of several: one form for both.
                        if line.funct == 1 && category == "dihedral" {
                            line.funct = 9;
                        }
                        text.push_str(&format!(
                            "  {}  {}\n",
                            local(m, &atoms_of),
                            self.render_values(&line)
                        ));
                    }
                }
            }
        }

        // Crossterms by GROMACS's lookup: exact, forward, first row.
        let cmap_rows = rows_of("cmaps", 5)?;
        if !cmap_rows.is_empty() {
            let style = ff
                .get_style("cmap", "charmm")
                .ok_or("cmaps rows without a cmap/charmm style")?;
            let label = |t: &str| -> String {
                atom_types
                    .get(t)
                    .and_then(|ty| ty.params.get("class").and_then(|v| v.as_text()))
                    .unwrap_or(t)
                    .to_owned()
            };
            for (atoms_of, name) in &cmap_rows {
                let classes: Vec<String> = atoms_of.iter().map(|&a| label(&types[[a]])).collect();
                let found = style
                    .type_rows()
                    .into_iter()
                    .find(|(_, ends, _)| ends.iter().zip(&classes).all(|(e, c)| e == c))
                    .map(|(found, _, _)| found);
                if found != Some(name.as_str()) {
                    return Err(format!(
                        "cmaps row {:?} of type '{name}': GROMACS's lookup by the bond types \
                         {} finds {found:?}",
                        atoms_of.iter().map(|a| a + 1).collect::<Vec<_>>(),
                        classes.join(" ")
                    )
                    .into());
                }
                let m = molecule_of("cmaps", atoms_of)?;
                sections[m]
                    .entry("cmap")
                    .or_default()
                    .push_str(&format!("  {}  1\n", local(m, atoms_of)));
            }
        }

        // The pairs GROMACS prices: nrexcl 3 excludes every pair within
        // three bonds, `[ pairs ]` prices the 1-4 ones, `[ exclusions ]`
        // removes the others of the molecule the frame does not price.
        let pairs = match frame.get("pairs") {
            Some(p) => p.clone(),
            None => intramolecular_pairs(frame, ff.special_bonds(), None)?,
        };
        // A frame whose every pair is excluded (a water) has a column-less
        // `pairs` block: no rows, not a missing column.
        let endpoints = |key: &str| -> Result<Vec<usize>, String> {
            if pairs.is_empty() {
                return Ok(Vec::new());
            }
            pairs
                .get(key)
                .and_then(|c| c.get::<u64>())
                .map(|c| c.iter().map(|&v| v as usize).collect())
                .ok_or_else(|| format!("pairs has no {key}"))
        };
        let (pi, pj) = (endpoints("atomi")?, endpoints("atomj")?);
        let is_14 = pairs.get("is_14").and_then(|c| c.get::<bool>());
        let cell = |key: &str, r: usize| -> Option<f64> {
            let col = pairs.get(key)?.get::<f64>()?;
            pairs.validity(key).is_none_or(|m| m[r]).then(|| col[[r]])
        };
        let mut near = HashSet::new();
        for start in 0..n {
            let mut depth = vec![usize::MAX; n];
            depth[start] = 0;
            let mut queue = std::collections::VecDeque::from([start]);
            while let Some(a) = queue.pop_front() {
                if depth[a] == 3 {
                    continue;
                }
                for &b in &adjacent[a] {
                    if depth[b] == usize::MAX {
                        depth[b] = depth[a] + 1;
                        queue.push_back(b);
                    }
                }
            }
            near.extend(
                (start + 1..n)
                    .filter(|&b| depth[b] != usize::MAX)
                    .map(|b| (start, b)),
            );
        }
        let sb = ff.special_bonds();
        let mut priced = HashSet::new();
        for r in 0..pi.len() {
            let (a, b) = (pi[r], pj[r]);
            let key = (a.min(b), a.max(b));
            let one_four = is_14.is_some_and(|f| f[[r]]);
            if one_four != near.contains(&key) {
                return Err(format!(
                    "pair {} {}: {} — GROMACS (nrexcl 3) prices a pair within three bonds \
                     only through [ pairs ] and one beyond them only as a regular pair",
                    a + 1,
                    b + 1,
                    if one_four {
                        "a 1-4 pair beyond three bonds"
                    } else {
                        "a regular pair within three bonds"
                    }
                )
                .into());
            }
            priced.insert(key);
            if !one_four {
                continue;
            }
            let [eps, sigma, qq, lj_w, coul_w] = PAIR_OVERRIDE_COLUMNS.map(|k| cell(k, r));
            let (i, j) = (a + 1, b + 1);
            let m = molecule[a];
            let ij = local(m, &[a, b]);
            let line = if [eps, sigma, qq, lj_w, coul_w].iter().all(Option::is_none) {
                format!("  {ij}  1\n")
            } else {
                let (Some(eps), Some(sigma)) = (eps, sigma) else {
                    return Err(format!(
                        "pair {i} {j}: override cells without epsilon and sigma — a \
                         [ pairs ] row with parameters states both"
                    )
                    .into());
                };
                let (v, w) = (full(sigma / NM_TO_ANGSTROM.get()), |e: f64| {
                    full(e * KCAL_TO_KJ.get())
                });
                if qq.is_none()
                    && lj_w.is_none_or(|x| x == 1.0)
                    && coul_w.is_none_or(|x| x == sb.coul[2])
                {
                    format!("  {ij}  1  {v}  {}\n", w(eps))
                } else {
                    format!(
                        "  {ij}  2  {}  {}  1.0  {v}  {}\n",
                        full(coul_w.unwrap_or(sb.coul[2])),
                        full(qq.unwrap_or(charges[[a]] * charges[[b]])),
                        w(eps * lj_w.unwrap_or(1.0))
                    )
                }
            };
            sections[m].entry("pairs").or_default().push_str(&line);
        }
        for (m, &(lo, hi)) in runs.iter().enumerate() {
            for i in lo..hi {
                let others: Vec<String> = (i + 1..hi)
                    .filter(|&j| !near.contains(&(i, j)) && !priced.contains(&(i, j)))
                    .map(|j| (j - lo + 1).to_string())
                    .collect();
                if !others.is_empty() {
                    sections[m]
                        .entry("exclusions")
                        .or_default()
                        .push_str(&format!("  {}  {}\n", i - lo + 1, others.join("  ")));
                }
            }
        }
        if frame.get("constraints").is_some() {
            let r0 = frame
                .get("constraints")
                .and_then(|b| b.get("r0"))
                .and_then(|c| c.get::<f64>())
                .ok_or("constraints has no r0 column")?;
            for (k, (i, j)) in column("constraints", "atomi")?
                .into_iter()
                .zip(column("constraints", "atomj")?)
                .enumerate()
            {
                let m = molecule[i];
                sections[m]
                    .entry("constraints")
                    .or_default()
                    .push_str(&format!(
                        "  {}  1  {}\n",
                        local(m, &[i, j]),
                        full(r0[[k]] / NM_TO_ANGSTROM.get())
                    ));
            }
        }

        for (m, &(lo, hi)) in runs.iter().enumerate() {
            out.push_str(&format!(
                "[ moleculetype ]\n; name  nrexcl\n  M{}  3\n\n[ atoms ]\n\
                 ; nr  type  resnr  residue  atom  cgnr  charge  mass\n",
                m + 1
            ));
            for i in lo..hi {
                let t = types[[i]].as_str();
                let ty = atom_types
                    .get(t)
                    .ok_or_else(|| format!("atom {}: type '{t}' is no atom/full type", i + 1))?;
                let mass = match masses {
                    Some(mass) => mass[[i]],
                    None => ty
                        .params
                        .get("mass")
                        .and_then(|v| v.as_num())
                        .ok_or_else(|| {
                            format!("atom {}: no mass column and type '{t}' has no mass", i + 1)
                        })?,
                };
                out.push_str(&format!(
                    "  {}  {t}  {}  {}  {}  {}  {}  {}\n",
                    i - lo + 1,
                    labels.res_seqs[i],
                    labels.res_names[i],
                    labels.names[i],
                    i - lo + 1,
                    full(charges[[i]]),
                    full(mass)
                ));
            }
            out.push('\n');
            for section in [
                "bonds",
                "pairs",
                "angles",
                "dihedrals",
                "cmap",
                "exclusions",
                "constraints",
            ] {
                if let Some(text) = sections[m].get(section) {
                    out.push_str(&format!("[ {section} ]\n{text}\n"));
                }
            }
        }
        out.push_str("[ system ]\nmolrs\n\n[ molecules ]\n");
        for m in 0..runs.len() {
            out.push_str(&format!("M{}  1\n", m + 1));
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::constants::VACUUM_DIELECTRIC;
    use crate::ff::forcefield::{ForceField, Style};
    use crate::ff::ir::{Params, SpecialBonds};
    use crate::ff::potential::fixtures::OneShot;
    use crate::io::writer::WrittenText;
    use crate::io::{gromacs::GromacsTopForcefieldReader, reader::ForceFieldReader};

    /// What `writer` writes as the topology of `ff` and `frame`.
    fn system(
        writer: &GromacsTopForcefieldWriter,
        ff: &ForceField,
        frame: &Frame,
    ) -> Result<String, ForceFieldWriteError> {
        let mut out = Vec::new();
        writer.write_system(&mut out, ff, frame)?;
        Ok(String::from_utf8(out).expect("the writer writes UTF-8"))
    }

    // -- fixtures (molrs units: Å, kcal/mol, degrees) --------------------------------

    fn atom_params(mass: f64, charge: f64, z: f64, bond_type: &str) -> Params {
        let mut p = Params::from_iter([("mass", mass), ("charge", charge), ("atomic_number", z)]);
        p.set("class", bond_type);
        p.set("ptype", "A");
        p
    }

    /// opls_135 (CT) and opls_140 (HC), each with its `lj/cut` self row;
    /// `mixing` declared when `Some`. 1-4 weights 0.5 / 0.5.
    fn opls_ff(mixing: Option<&str>) -> ForceField {
        let mut ff = ForceField::new("gmx");
        ff.def_style("atom", "full", Params::new())
            .unwrap()
            .def_type("opls_135", &[], atom_params(12.011, -0.18, 6.0, "CT"))
            .unwrap()
            .def_type("opls_140", &[], atom_params(1.008, 0.06, 1.0, "HC"))
            .unwrap();
        let mut lj_params = Params::new();
        if let Some(rule) = mixing {
            lj_params.set("mixing", rule);
        }
        ff.def_style("pair", "lj/cut", lj_params)
            .unwrap()
            .def_type(
                "opls_135",
                &["opls_135"],
                Params::from_iter([("sigma", 3.5), ("epsilon", 0.066)]),
            )
            .unwrap()
            .def_type(
                "opls_140",
                &["opls_140"],
                Params::from_iter([("sigma", 2.5), ("epsilon", 0.03)]),
            )
            .unwrap();
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        ff
    }

    /// `opls_ff(Some("geometric"))` plus one `category/name` type.
    fn with_type(
        category: &str,
        name: &str,
        type_name: &str,
        endpoints: &[&str],
        params: Params,
    ) -> ForceField {
        let mut ff = opls_ff(Some("geometric"));
        ff.def_style(category, name, Params::new())
            .unwrap()
            .def_type(type_name, endpoints, params)
            .unwrap();
        ff
    }

    fn write(ff: &ForceField) -> String {
        GromacsTopForcefieldWriter::new()
            .written(ff)
            .unwrap_or_else(|e| panic!("write_str: {e}"))
    }

    fn write_err(ff: &ForceField) -> String {
        GromacsTopForcefieldWriter::new()
            .written(ff)
            .expect_err("expected Err from write_str")
            .into()
    }

    /// The whitespace-split data rows of `[ section ]` (comments skipped).
    fn section_rows<'a>(text: &'a str, section: &str) -> Vec<Vec<&'a str>> {
        let header = format!("[ {section} ]");
        let mut rows = Vec::new();
        let mut inside = false;
        for line in text.lines() {
            let t = line.trim();
            if t.starts_with('[') {
                inside = t == header;
                continue;
            }
            if inside && !t.is_empty() && !t.starts_with(';') {
                rows.push(t.split_whitespace().collect());
            }
        }
        rows
    }

    /// The only row of `[ section ]` whose leading tokens are `labels`.
    fn row<'a>(text: &'a str, section: &str, labels: &[&str]) -> Vec<&'a str> {
        let rows = section_rows(text, section);
        let mut found: Vec<Vec<&str>> = rows
            .into_iter()
            .filter(|r| r.len() >= labels.len() && r[..labels.len()] == *labels)
            .collect();
        assert_eq!(
            found.len(),
            1,
            "[ {section} ] rows for {labels:?} in:\n{text}"
        );
        found.pop().expect("one row")
    }

    fn number(token: &str) -> f64 {
        token
            .parse()
            .unwrap_or_else(|e| panic!("{token:?} is not a number: {e}"))
    }

    /// `tokens` (after the labels) are `code` then `values`, numerically.
    fn assert_row_values(tokens: &[&str], code: &str, values: &[f64]) {
        assert_eq!(tokens[0], code, "function code in {tokens:?}");
        assert_eq!(tokens.len(), 1 + values.len(), "{tokens:?}");
        for (tok, want) in tokens[1..].iter().zip(values) {
            assert!(
                (number(tok) - want).abs() < 1e-9,
                "{tok} != {want} in {tokens:?}"
            );
        }
    }

    fn assert_names(err: &str, needles: &[&str]) {
        for needle in needles {
            assert!(err.contains(needle), "error should name `{needle}`: {err}");
        }
    }

    // -- [ defaults ] ------------------------------------------------------------

    #[test]
    fn geometric_mixing_writes_comb_rule_3() {
        let text = write(&opls_ff(Some("geometric")));
        let rows = section_rows(&text, "defaults");
        assert_eq!(rows.len(), 1, "{text}");
        assert_eq!(rows[0][..3], ["1", "3", "yes"]);
        assert!((number(rows[0][3]) - 0.5).abs() < 1e-12);
        assert!((number(rows[0][4]) - 0.5).abs() < 1e-12);
    }

    #[test]
    fn arithmetic_mixing_writes_comb_rule_2() {
        let text = write(&opls_ff(Some("arithmetic")));
        assert_eq!(section_rows(&text, "defaults")[0][..3], ["1", "2", "yes"]);
    }

    /// No declared rule is `CombiningRule::UNDECLARED`, Lorentz-Berthelot: comb-rule 2.
    #[test]
    fn undeclared_mixing_writes_comb_rule_2() {
        let text = write(&opls_ff(None));
        assert_eq!(section_rows(&text, "defaults")[0][..3], ["1", "2", "yes"]);
    }

    #[test]
    fn empty_force_field_writes_only_defaults() {
        let text = write(&ForceField::new("x"));
        assert_eq!(section_rows(&text, "defaults")[0][..3], ["1", "2", "yes"]);
        assert!(section_rows(&text, "atomtypes").is_empty(), "{text}");
    }

    /// GROMACS has no comb-rule for sixth-power mixing.
    #[test]
    fn sixthpower_mixing_is_an_error() {
        let err = write_err(&opls_ff(Some("sixthpower")));
        assert_names(&err, &["sixthpower"]);
    }

    /// gen-pairs 1-4 scaling cannot say "keep 1-2 neighbours".
    #[test]
    fn nonzero_1_2_special_bond_weight_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.set_special_bonds(SpecialBonds {
            lj: [0.5, 0.0, 0.5],
            coul: [0.0, 0.0, 0.5],
        });
        let err = write_err(&ff);
        assert_names(&err, &["1-2"]);
    }

    #[test]
    fn nonzero_1_3_special_bond_weight_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.5, 0.5],
        });
        let err = write_err(&ff);
        assert_names(&err, &["1-3"]);
    }

    // -- [ atomtypes ] -----------------------------------------------------------

    /// opls_135 in file units: V = 3.5 Å ÷ 10 = 0.35 nm; W = 0.066 kcal/mol ×
    /// 4.184 = 0.276144 kJ/mol, from the `lj/cut` self row.
    #[test]
    fn atomtypes_row_joins_atom_full_and_the_lj_cut_self_row() {
        let text = write(&opls_ff(Some("geometric")));
        let r = row(&text, "atomtypes", &["opls_135"]);
        assert_eq!(r.len(), 8, "{r:?}");
        assert_eq!(r[1], "CT");
        assert_eq!(r[2], "6");
        assert!((number(r[3]) - 12.011).abs() < 1e-9);
        assert!((number(r[4]) - -0.18).abs() < 1e-9);
        assert_eq!(r[5], "A");
        assert!((number(r[6]) - 0.35).abs() < 1e-9);
        assert!((number(r[7]) - 0.276144).abs() < 1e-9);
    }

    /// Neither `bond_type` nor `atomic_number`: the 6-column form.
    #[test]
    fn atomtypes_row_without_bond_type_or_atomic_number_has_six_columns() {
        let mut ff = opls_ff(Some("geometric"));
        let atoms = ff.get_style_mut("atom", "full").unwrap();
        atoms.del_type("opls_135");
        let mut p = Params::from_iter([("mass", 12.011), ("charge", -0.18)]);
        p.set("ptype", "A");
        atoms.def_type("opls_135", &[], p).unwrap();
        let text = write(&ff);
        let r = row(&text, "atomtypes", &["opls_135"]);
        assert_eq!(r.len(), 6, "{r:?}");
        assert!((number(r[1]) - 12.011).abs() < 1e-9);
        assert_eq!(r[3], "A");
        assert!((number(r[4]) - 0.35).abs() < 1e-9);
    }

    /// molrs `atom/full` types are real atoms: no `ptype` is written `A`.
    #[test]
    fn missing_ptype_is_written_a() {
        let mut ff = opls_ff(Some("geometric"));
        let atoms = ff.get_style_mut("atom", "full").unwrap();
        atoms.del_type("opls_135");
        let mut p =
            Params::from_iter([("mass", 12.011), ("charge", -0.18), ("atomic_number", 6.0)]);
        p.set("class", "CT");
        atoms.def_type("opls_135", &[], p).unwrap();
        let text = write(&ff);
        assert_eq!(row(&text, "atomtypes", &["opls_135"])[5], "A");
    }

    #[test]
    fn declared_ptype_is_written_as_declared() {
        let mut ff = opls_ff(Some("geometric"));
        assert!(
            ff.get_style_mut("atom", "full")
                .unwrap()
                .set_type_param("opls_135", "ptype", "S")
        );
        let text = write(&ff);
        assert_eq!(row(&text, "atomtypes", &["opls_135"])[5], "S");
    }

    fn opls_135_without(key: &str) -> ForceField {
        let mut ff = opls_ff(Some("geometric"));
        let atoms = ff.get_style_mut("atom", "full").unwrap();
        atoms.del_type("opls_135");
        let mut p = Params::new();
        for (k, v) in [("mass", 12.011), ("charge", -0.18), ("atomic_number", 6.0)] {
            if k != key {
                p.set(k, v);
            }
        }
        p.set("class", "CT");
        p.set("ptype", "A");
        atoms.def_type("opls_135", &[], p).unwrap();
        ff
    }

    #[test]
    fn atom_type_without_mass_is_an_error() {
        let err = write_err(&opls_135_without("mass"));
        assert_names(&err, &["opls_135", "mass"]);
    }

    #[test]
    fn atom_type_without_charge_is_an_error() {
        let err = write_err(&opls_135_without("charge"));
        assert_names(&err, &["opls_135", "charge"]);
    }

    #[test]
    fn atom_type_without_an_lj_cut_self_row_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .del_type("opls_135");
        let err = write_err(&ff);
        assert_names(&err, &["opls_135", "lj/cut"]);
    }

    /// A cross row is a `[ nonbond_params ]` row (σ = 3 Å = 0.3 nm, ε = 0.05
    /// kcal/mol = 0.2092 kJ/mol) and reads back as the same cross row. It used
    /// to be refused.
    #[test]
    fn explicit_lj_cut_cross_row_is_a_nonbond_params_row() {
        let mut ff = opls_ff(Some("geometric"));
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .def_type(
                "opls_135-opls_140",
                &["opls_135", "opls_140"],
                Params::from_iter([("sigma", 3.0), ("epsilon", 0.05)]),
            )
            .unwrap();
        let text = write(&ff);
        let r = row(&text, "nonbond_params", &["opls_135", "opls_140"]);
        assert_row_values(&r[2..], "1", &[0.3, 0.2092]);
        let back = GromacsTopForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap();
        let cross = back
            .get_style("pair", "lj/cut")
            .unwrap()
            .get_pairtype("opls_135", Some("opls_140"))
            .expect("cross row read back");
        assert!((cross.params.get("sigma").and_then(|v| v.as_num()).unwrap() - 3.0).abs() < 1e-9);
        assert!(
            (cross
                .params
                .get("epsilon")
                .and_then(|v| v.as_num())
                .unwrap()
                - 0.05)
                .abs()
                < 1e-9
        );
    }

    /// No cross row, no `[ nonbond_params ]` section.
    #[test]
    fn no_cross_row_writes_no_nonbond_params() {
        assert!(!write(&opls_ff(Some("geometric"))).contains("nonbond_params"));
    }

    #[test]
    fn a_cross_row_on_an_unknown_type_is_an_error() {
        let mut ff = opls_ff(Some("geometric"));
        ff.get_style_mut("pair", "lj/cut")
            .unwrap()
            .def_type(
                "opls_135-ZZ",
                &["opls_135", "ZZ"],
                Params::from_iter([("sigma", 3.0), ("epsilon", 0.05)]),
            )
            .unwrap();
        assert_names(&write_err(&ff), &["opls_135-ZZ", "ZZ"]);
    }

    // -- bonded directives -------------------------------------------------------

    /// r0 = 1.09 Å → 0.109 nm; LAMMPS K = 340 kcal/mol/Å² is GROMACS's
    /// ½k_b with k_b = 2 × 340 × 418.4 = 284512 kJ/mol/nm².
    #[test]
    fn bond_harmonic_is_bondtypes_code_1() {
        let ff = with_type(
            "bond",
            "harmonic",
            "CT-HC",
            &["CT", "HC"],
            Params::from_iter([("r0", 1.09), ("k", 340.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "bondtypes", &["CT", "HC"]);
        assert_row_values(&r[2..], "1", &[0.109, 284512.0]);
    }

    /// b₀ = 0.1529 nm; D = d0 = 95.602294455066… × 4.184 = 400 kJ/mol; β = 2 × 10 =
    /// 20 nm⁻¹.
    #[test]
    fn bond_morse_is_bondtypes_code_3() {
        let ff = with_type(
            "bond",
            "morse",
            "CT-CT",
            &["CT", "CT"],
            Params::from_iter([("d0", 95.602_294_455_066_9), ("alpha", 2.0), ("r0", 1.529)]),
        );
        let text = write(&ff);
        let r = row(&text, "bondtypes", &["CT", "CT"]);
        assert_row_values(&r[2..], "3", &[0.1529, 400.0, 20.0]);
    }

    /// θ₀ = 107.8° as stored; LAMMPS K = 33 is GROMACS's ½k_θ with
    /// k_θ = 2 × 33 × 4.184 = 276.144 kJ/mol/rad².
    #[test]
    fn angle_harmonic_is_angletypes_code_1() {
        let ff = with_type(
            "angle",
            "harmonic",
            "HC-CT-HC",
            &["HC", "CT", "HC"],
            Params::from_iter([("theta0", 107.8), ("k", 33.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "angletypes", &["HC", "CT", "HC"]);
        assert_row_values(&r[3..], "1", &[107.8, 276.144]);
    }

    /// φ_s = 0°; k = 1 × 4.184 = 4.184 kJ/mol; n = 3.
    #[test]
    fn dihedral_periodic_is_dihedraltypes_code_1() {
        let ff = with_type(
            "dihedral",
            "periodic",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_iter([("k", 1.0), ("periodicity", 3.0), ("phase", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["CT", "CT", "CT", "CT"]);
        assert_row_values(&r[4..], "1", &[0.0, 4.184, 3.0]);
    }

    /// LAMMPS `opls` is GROMACS's Fourier dihedral term for term: C3 = 0.3 ×
    /// 4.184 = 1.2552 kJ/mol.
    #[test]
    fn dihedral_opls_is_dihedraltypes_code_5() {
        let ff = with_type(
            "dihedral",
            "opls",
            "HC-CT-CT-HC",
            &["HC", "CT", "CT", "HC"],
            Params::from_iter([("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["HC", "CT", "CT", "HC"]);
        assert_row_values(&r[4..], "5", &[0.0, 0.0, 1.2552, 0.0]);
    }

    /// `multi/harmonic` is Ryckaert-Bellemans with Cₙ = (−1)ⁿ aₙ₊₁: a = (0.15,
    /// −0.45, 0, 0.6, 0) kcal/mol is C = (0.6276, 1.8828, 0, −2.5104, 0, 0) kJ/mol.
    #[test]
    fn dihedral_multi_harmonic_is_dihedraltypes_code_3() {
        let ff = with_type(
            "dihedral",
            "multi/harmonic",
            "HC-CT-CT-HC",
            &["HC", "CT", "CT", "HC"],
            Params::from_iter([
                ("a1", 0.15),
                ("a2", -0.45),
                ("a3", 0.0),
                ("a4", 0.6),
                ("a5", 0.0),
            ]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["HC", "CT", "CT", "HC"]);
        assert_row_values(&r[4..], "3", &[0.6276, 1.8828, 0.0, -2.5104, 0.0, 0.0]);
    }

    /// `nharmonic` with six coefficients fills C5; seven is above RB.
    #[test]
    fn dihedral_nharmonic_is_dihedraltypes_code_3_up_to_six_terms() {
        let six: Vec<(String, f64)> = (1..=6).map(|i| (format!("a{i}"), 1.0)).collect();
        let pairs: Vec<(&str, f64)> = six.iter().map(|(k, v)| (k.as_str(), *v)).collect();
        let ff = with_type(
            "dihedral",
            "nharmonic",
            "CT-CT-CT-CT",
            &["CT"; 4],
            Params::from_iter(pairs.iter().copied()),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["CT", "CT", "CT", "CT"]);
        let c = 4.184;
        assert_row_values(&r[4..], "3", &[c, -c, c, -c, c, -c]);
        let mut seven = pairs.clone();
        seven.push(("a7", 1.0));
        let ff = with_type(
            "dihedral",
            "nharmonic",
            "CT-CT-CT-CT",
            &["CT"; 4],
            Params::from_iter(seven.iter().copied()),
        );
        assert_names(&write_err(&ff), &["N = 7"]);
    }

    /// θ₀ = 109.5°; k_θ = 2·35.5·4.184 = 297.064; r₁₃ = 0.1802 nm; k_UB =
    /// 2·5.4·418.4 = 4518.72 — CHARMM36 HA-CT2-HA as GROMACS writes it.
    #[test]
    fn angle_charmm_is_angletypes_code_5() {
        let ff = with_type(
            "angle",
            "charmm",
            "HC-CT-HC",
            &["HC", "CT", "HC"],
            Params::from_iter([
                ("k", 35.5),
                ("theta0", 109.5),
                ("k_ub", 5.4),
                ("r_ub", 1.802),
            ]),
        );
        let text = write(&ff);
        let r = row(&text, "angletypes", &["HC", "CT", "HC"]);
        assert_row_values(&r[3..], "5", &[109.5, 297.064, 0.1802, 4518.72]);
    }

    /// `dihedral harmonic` k[1 + d cos nφ] is one periodic term at phase 0°
    /// (d = 1) or 180° (d = −1); `improper cvff` the same as funct 4.
    #[test]
    fn signed_cosines_are_periodic_rows_at_0_or_180() {
        let ff = with_type(
            "dihedral",
            "harmonic",
            "CT-CT-CT-CT",
            &["CT"; 4],
            Params::from_iter([("k", 1.0), ("sign", -1.0), ("periodicity", 2.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["CT", "CT", "CT", "CT"]);
        assert_row_values(&r[4..], "9", &[180.0, 4.184, 2.0]);
        let ff = with_type(
            "improper",
            "cvff",
            "HC-CT-CT-CT",
            &["HC", "CT", "CT", "CT"],
            Params::from_iter([("k", 1.0), ("sign", 1.0), ("periodicity", 2.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["HC", "CT", "CT", "CT"]);
        assert_row_values(&r[4..], "4", &[0.0, 4.184, 2.0]);
    }

    /// A 2×2 cmap is `[ cmaptypes ]` in kJ/mol, φ-major, and reads back.
    #[test]
    fn cmap_charmm_is_a_cmaptypes_row() {
        let mut ff = opls_ff(Some("geometric"));
        let mut p = Params::new();
        p.set(
            "grid",
            ndarray::ArrayD::from_shape_vec(vec![2, 2], vec![1.0, 2.0, -1.0, 0.0]).unwrap(),
        );
        ff.def_style("cmap", "charmm", Params::new())
            .unwrap()
            .def_type("CT-HC-CT-CT-HC", &["CT", "HC", "CT", "CT", "HC"], p)
            .unwrap();
        let text = write(&ff);
        assert!(text.contains("[ cmaptypes ]"), "{text}");
        assert!(
            text.contains("CT HC CT CT HC 1 2 2\\\n4.184000 8.368000 -4.184000 0.000000"),
            "{text}"
        );
        let back = GromacsTopForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap();
        let grid = back.get_cmaptypes()[0]
            .params
            .get("grid")
            .and_then(|v| v.as_array())
            .unwrap()
            .clone();
        assert_eq!(
            grid.iter().copied().collect::<Vec<_>>(),
            [1.0, 2.0, -1.0, 0.0]
        );
    }

    /// Two periodic types on the same labels are one to GROMACS.
    #[test]
    fn two_types_on_the_same_labels_of_one_table_are_an_error() {
        let mut ff = with_type(
            "dihedral",
            "periodic",
            "CT-CT-CT-CT",
            &["CT"; 4],
            Params::from_iter([("k", 1.0), ("periodicity", 3.0), ("phase", 0.0)]),
        );
        ff.get_style_mut("dihedral", "periodic")
            .unwrap()
            .def_type(
                "CT-CT-CT-CT@gmx_1",
                &["CT"; 4],
                Params::from_iter([("k", 2.0), ("periodicity", 3.0), ("phase", 0.0)]),
            )
            .unwrap();
        assert_names(&write_err(&ff), &["CT-CT-CT-CT@gmx_1", "one"]);
    }

    #[test]
    fn a_pair_cutoff_is_a_run_setting_and_not_written() {
        let mut ff = ForceField::new("t");
        ff.def_style("pair", "lj/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        ff.def_style("pair", "coul/cut", Params::from_iter([("cutoff", 10.0)]))
            .unwrap();
        let text = write(&ff);
        assert!(!text.contains("10"), "{text}");
    }

    /// The empty endpoint wildcard is written as GROMACS `X`.
    #[test]
    fn empty_endpoint_is_written_x() {
        let ff = with_type(
            "dihedral",
            "opls",
            "-CT-CT-",
            &["", "CT", "CT", ""],
            Params::from_iter([("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["X", "CT", "CT", "X"]);
        assert_eq!(r[4], "5");
    }

    /// k = 2.5 × 4.184 = 10.46 kJ/mol; φ_s = 180°; n = 2.
    #[test]
    fn improper_periodic_is_dihedraltypes_code_4() {
        let ff = with_type(
            "improper",
            "periodic",
            "--CT-HC",
            &["", "", "CT", "HC"],
            Params::from_iter([("k", 2.5), ("periodicity", 2.0), ("phase", 180.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["X", "X", "CT", "HC"]);
        assert_row_values(&r[4..], "4", &[180.0, 10.46, 2.0]);
    }

    /// K(χ)² = ½k_ξ(ξ)²: k_ξ = 2 · 20 · 4.184 = 167.36 kJ/mol/rad²; ξ₀ = 0.
    #[test]
    fn improper_harmonic_is_dihedraltypes_code_2() {
        let ff = with_type(
            "improper",
            "harmonic",
            "--CT-HC",
            &["", "", "CT", "HC"],
            Params::from_iter([("k", 20.0), ("chi0", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["X", "X", "CT", "HC"]);
        assert_row_values(&r[4..], "2", &[0.0, 167.36]);
    }

    /// Directives only: no molecule section is written.
    #[test]
    fn no_molecule_section_is_written() {
        let mut ff = with_type(
            "bond",
            "harmonic",
            "CT-HC",
            &["CT", "HC"],
            Params::from_iter([("r0", 1.09), ("k", 340.0)]),
        );
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                Params::from_iter([("theta0", 108.9), ("k", 33.0)]),
            )
            .unwrap();
        let text = write(&ff);
        for section in [
            "atoms",
            "bonds",
            "angles",
            "dihedrals",
            "pairs",
            "moleculetype",
        ] {
            assert!(
                !text.contains(&format!("[ {section} ]")),
                "[ {section} ] written:\n{text}"
            );
        }
    }

    // -- refusals ----------------------------------------------------------------

    #[test]
    fn dihedral_charmm_is_an_error() {
        let ff = with_type(
            "dihedral",
            "charmm",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_iter([("k", 1.0), ("periodicity", 3.0), ("phase", 0.0), ("w", 0.5)]),
        );
        let err = write_err(&ff);
        assert_names(&err, &["charmm"]);
    }

    /// `dihedral charmm` with w = 0 is the periodic term alone: funct 9.
    #[test]
    fn dihedral_charmm_with_w_0_is_dihedraltypes_code_9() {
        let ff = with_type(
            "dihedral",
            "charmm",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_iter([("k", 1.0), ("periodicity", 3.0), ("phase", 0.0), ("w", 0.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "dihedraltypes", &["CT", "CT", "CT", "CT"]);
        assert_row_values(&r[4..], "9", &[0.0, 4.184, 3.0]);
    }

    /// Several periodic terms on one type are consecutive funct-9 rows.
    #[test]
    fn multi_term_dihedral_periodic_is_funct_9_rows() {
        let ff = with_type(
            "dihedral",
            "periodic",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_iter([
                ("k1", 1.0),
                ("periodicity1", 1.0),
                ("phase1", 0.0),
                ("k2", 0.5),
                ("periodicity2", 3.0),
                ("phase2", 0.0),
            ]),
        );
        let text = write(&ff);
        let rows: Vec<Vec<&str>> = section_rows(&text, "dihedraltypes");
        assert_eq!(rows.len(), 2, "{text}");
        assert_row_values(&rows[0][4..], "9", &[0.0, 4.184, 1.0]);
        assert_row_values(&rows[1][4..], "9", &[0.0, 2.092, 3.0]);
    }

    /// `ZZ` is neither an atom-type name nor any type's `class`.
    #[test]
    fn unresolvable_bonded_endpoint_is_an_error() {
        let ff = with_type(
            "bond",
            "harmonic",
            "ZZ-CT",
            &["ZZ", "CT"],
            Params::from_iter([("r0", 1.09), ("k", 340.0)]),
        );
        let err = write_err(&ff);
        assert_names(&err, &["ZZ"]);
    }

    /// An atom-type name is a resolvable endpoint too.
    #[test]
    fn atom_type_name_endpoint_is_written() {
        let ff = with_type(
            "bond",
            "harmonic",
            "opls_135-opls_140",
            &["opls_135", "opls_140"],
            Params::from_iter([("r0", 1.09), ("k", 340.0)]),
        );
        let text = write(&ff);
        let r = row(&text, "bondtypes", &["opls_135", "opls_140"]);
        assert_row_values(&r[2..], "1", &[0.109, 284512.0]);
    }

    // -- read(write(ff)) == ff -----------------------------------------------------

    /// A force field holding every style the GROMACS directives express.
    fn every_supported_style() -> ForceField {
        let mut ff = opls_ff(Some("geometric"));
        ff.def_style(
            "pair",
            "coul/cut",
            Params::from_iter([
                ("coulomb", crate::core::constants::gromacs_coulomb_real()),
                ("dielectric", VACUUM_DIELECTRIC),
            ]),
        )
        .unwrap();
        // (category, style, type name, endpoints, params) of one type definition.
        type TypeDef<'a> = (
            &'a str,
            &'a str,
            &'a str,
            &'a [&'a str],
            &'a [(&'a str, f64)],
        );
        let defs: &[TypeDef] = &[
            (
                "bond",
                "harmonic",
                "CT-HC",
                &["CT", "HC"],
                &[("r0", 1.09), ("k", 340.0)],
            ),
            (
                "bond",
                "morse",
                "CT-CT",
                &["CT", "CT"],
                &[("d0", 95.602_294_455_066_9), ("alpha", 2.0), ("r0", 1.529)],
            ),
            (
                "angle",
                "harmonic",
                "HC-CT-HC",
                &["HC", "CT", "HC"],
                &[("theta0", 107.8), ("k", 33.0)],
            ),
            (
                "dihedral",
                "periodic",
                "CT-CT-CT-CT",
                &["CT", "CT", "CT", "CT"],
                &[("k", 1.0), ("periodicity", 3.0), ("phase", 0.0)],
            ),
            (
                "dihedral",
                "opls",
                "HC-CT-CT-HC",
                &["HC", "CT", "CT", "HC"],
                &[("k1", 0.0), ("k2", 0.0), ("k3", 0.3), ("k4", 0.0)],
            ),
            (
                "dihedral",
                "opls",
                "-CT-CT-",
                &["", "CT", "CT", ""],
                &[("k1", 1.3), ("k2", -0.05), ("k3", 0.2), ("k4", 0.1)],
            ),
            (
                "improper",
                "periodic",
                "--CT-HC",
                &["", "", "CT", "HC"],
                &[("k", 2.5), ("periodicity", 2.0), ("phase", 180.0)],
            ),
            (
                "improper",
                "harmonic",
                "--HC-CT",
                &["", "", "HC", "CT"],
                &[("k", 20.0), ("chi0", 0.0)],
            ),
            (
                "angle",
                "harmonic",
                "CT-CT-HC",
                &["CT", "CT", "HC"],
                &[("theta0", 108.9), ("k", 37.5)],
            ),
            (
                "angle",
                "charmm",
                "HC-CT-CT",
                &["HC", "CT", "CT"],
                &[
                    ("k", 35.5),
                    ("theta0", 109.5),
                    ("k_ub", 5.4),
                    ("r_ub", 1.802),
                ],
            ),
            (
                "dihedral",
                "periodic",
                "HC-CT-CT-CT",
                &["HC", "CT", "CT", "CT"],
                &[
                    ("k1", 0.2),
                    ("periodicity1", 1.0),
                    ("phase1", 180.0),
                    ("k2", 0.25),
                    ("periodicity2", 2.0),
                    ("phase2", 37.5),
                ],
            ),
            (
                "dihedral",
                "multi/harmonic",
                "CT-CT-CT-HC",
                &["CT", "CT", "CT", "HC"],
                &[
                    ("a1", 0.15),
                    ("a2", -0.45),
                    ("a3", 0.1),
                    ("a4", 0.6),
                    ("a5", 0.0),
                ],
            ),
            (
                "dihedral",
                "nharmonic",
                "HC-HC-CT-HC",
                &["HC", "HC", "CT", "HC"],
                &[
                    ("a1", 0.1),
                    ("a2", 0.2),
                    ("a3", 0.3),
                    ("a4", 0.4),
                    ("a5", 0.5),
                    ("a6", 0.6),
                ],
            ),
            (
                "improper",
                "harmonic",
                "CT-HC-HC-CT",
                &["CT", "HC", "HC", "CT"],
                &[("k", 12.0), ("chi0", 180.0)],
            ),
        ];
        for &(category, style, name, endpoints, params) in defs {
            ff.def_style(category, style, Params::new())
                .unwrap()
                .def_type(name, endpoints, Params::from_iter(params.iter().copied()))
                .unwrap();
        }
        ff
    }

    type TypeRow = (String, Vec<String>, Params);

    /// `(category, name)` → (style params, types sorted by name).
    fn snapshot(ff: &ForceField) -> Vec<((String, String), Params, Vec<TypeRow>)> {
        let mut out: Vec<((String, String), Params, Vec<TypeRow>)> = ff
            .styles()
            .iter()
            .map(|s: &Style| {
                let mut types: Vec<TypeRow> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, params)| {
                        let ends = s.type_endpoints(&name).expect("type has endpoints");
                        (name, ends, params)
                    })
                    .collect();
                types.sort_by(|a, b| a.0.cmp(&b.0));
                (
                    (s.category().to_owned(), s.name().to_owned()),
                    s.params().clone(),
                    types,
                )
            })
            .collect();
        out.sort_by(|a, b| a.0.cmp(&b.0));
        out
    }

    fn assert_params_close(what: &str, got: &Params, want: &Params) {
        let mut got_keys: Vec<&str> = got
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
            .map(|(k, _)| k)
            .collect();
        let mut want_keys: Vec<&str> = want
            .iter()
            .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
            .map(|(k, _)| k)
            .collect();
        got_keys.sort_unstable();
        want_keys.sort_unstable();
        assert_eq!(got_keys, want_keys, "{what}: numeric keys");
        for (k, w) in want.iter().filter_map(|(k, v)| v.as_num().map(|x| (k, x))) {
            let g = got.get(k).and_then(|v| v.as_num()).expect("key present");
            assert!((g - w).abs() < 1e-9, "{what}.{k}: got {g}, want {w}");
        }
        let mut got_strs: Vec<(&str, &str)> = got
            .iter()
            .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
            .collect();
        let mut want_strs: Vec<(&str, &str)> = want
            .iter()
            .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
            .collect();
        got_strs.sort_unstable();
        want_strs.sort_unstable();
        assert_eq!(got_strs, want_strs, "{what}: string params");
    }

    #[test]
    fn reading_what_is_written_gives_back_the_force_field() {
        let ff = every_supported_style();
        let text = GromacsTopForcefieldWriter::new()
            .with_precision(10)
            .written(&ff)
            .unwrap_or_else(|e| panic!("write_str: {e}"));
        let back = GromacsTopForcefieldReader::new()
            .read(text.as_bytes())
            .unwrap_or_else(|e| panic!("read_str: {e}\n{text}"));

        assert_eq!(back.special_bonds(), ff.special_bonds());
        let (want, got) = (snapshot(&ff), snapshot(&back));
        let keys = |s: &[((String, String), Params, Vec<TypeRow>)]| {
            s.iter().map(|(k, _, _)| k.clone()).collect::<Vec<_>>()
        };
        assert_eq!(keys(&got), keys(&want), "styles");
        for ((key, want_params, want_types), (_, got_params, got_types)) in want.iter().zip(&got) {
            let what = format!("{}/{}", key.0, key.1);
            assert_params_close(&what, got_params, want_params);
            let names = |t: &[TypeRow]| t.iter().map(|r| r.0.clone()).collect::<Vec<_>>();
            assert_eq!(names(got_types), names(want_types), "{what}: type names");
            for ((name, want_ends, want_p), (_, got_ends, got_p)) in
                want_types.iter().zip(got_types)
            {
                assert_eq!(got_ends, want_ends, "{what} {name}: endpoints");
                assert_params_close(&format!("{what} {name}"), got_p, want_p);
            }
        }
    }

    // -- systems ------------------------------------------------------------------

    /// Butane-ish C1-C2-C3-C4 with an H on C1 — one `[ pairs ]` row with
    /// parameters (funct 1), one funct 2 — and two waters (settles).
    const SYSTEM: &str = "\
[ defaults ]
1  3  yes  0.5  0.8333
[ atomtypes ]
opls_135  CT  6  12.011  -0.18  A  0.35  0.276144
opls_140  HC  1   1.008   0.06  A  0.25  0.12552
OW        OW  8  15.999  -0.834 A  0.315  0.6364
HW        HW  1   1.008   0.417 A  0.0    0.0
[ bondtypes ]
HC  CT  1  0.109  284512.0
CT  CT  1  0.1529 224262.4
[ angletypes ]
CT  CT  CT  1  112.7  488.273
HC  CT  CT  1  110.7  313.800
[ dihedraltypes ]
HC  CT  CT  CT  9  0.0  2.0  3
HC  CT  CT  CT  9  180.0  0.5  1
CT  CT  CT  CT  3  2.9288  -1.4644  0.2092  -1.6736  0.0  0.0
[ moleculetype ]
BUT  3
[ atoms ]
1  opls_135  1  BUT  C1  1
2  opls_135  1  BUT  C2  1  -0.12  12.011
3  opls_135  1  BUT  C3  1  -0.12
4  opls_135  1  BUT  C4  1
5  opls_140  1  BUT  H1  1
[ bonds ]
1  2  1
2  3  1
3  4  1
5  1  1
[ pairs ]
1  4  1  0.3  0.5
5  3  2  0.8  0.2  -0.3  0.31  0.4
[ angles ]
1  2  3  1
2  3  4  1
5  1  2  1
[ dihedrals ]
1  2  3  4  3
5  1  2  3  9
[ moleculetype ]
SOL  2
[ atoms ]
1  OW  1  SOL  OW  1
2  HW  1  SOL  HW1 1
3  HW  1  SOL  HW2 1
[ settles ]
1  1  0.09572  0.15139
[ exclusions ]
1  2  3
2  1  3
3  1  2
[ system ]
test
[ molecules ]
BUT  1
SOL  2
";

    fn read_system(text: &str) -> (ForceField, molrs::core::Frame) {
        GromacsTopForcefieldReader::new()
            .read_system(text.as_bytes())
            .unwrap_or_else(|e| panic!("{e}\n{text}"))
    }

    /// Coordinates for `n` atoms, none on another.
    fn coords(n: usize) -> Vec<f64> {
        (0..3 * n)
            .map(|i| 1.3 * (i / 3) as f64 + 0.37 * ((i * 7 + 3) % 5) as f64)
            .collect()
    }

    fn energy(ff: &ForceField, frame: &molrs::core::Frame, x: &[f64]) -> f64 {
        let mut ff = ff.clone();
        for name in ["lj/cut", "coul/cut"] {
            if let Some(s) = ff.get_style_mut("pair", name) {
                s.set_param("cutoff", 100.0);
            }
        }
        crate::ff::compile::PotentialCompiler::new(&ff)
            .compile(frame)
            .unwrap()
            .energy_at(x)
    }

    /// The (i, j, is_14, override cells) of a frame's pairs.
    fn pairs_of(frame: &molrs::core::Frame) -> Vec<(u64, u64, bool, Vec<Option<f64>>)> {
        let p = frame.get("pairs").unwrap();
        let (i, j) = (
            p.get("atomi").unwrap().get::<u64>().unwrap(),
            p.get("atomj").unwrap().get::<u64>().unwrap(),
        );
        let f = p.get("is_14").unwrap().get::<bool>().unwrap();
        (0..i.len())
            .map(|r| {
                let cells = molrs::core::schema::PAIR_OVERRIDE_COLUMNS
                    .iter()
                    .map(|k| {
                        let col = p.get(k)?.get::<f64>()?;
                        p.validity(k).is_none_or(|m| m[r]).then(|| col[[r]])
                    })
                    .collect();
                (i[[r]], j[[r]], f[[r]], cells)
            })
            .collect()
    }

    /// A system written and read back is the system: every pair GROMACS
    /// prices and its 1-4 flag, the `[ pairs ]` rows' own parameters (funct 1
    /// and 2), the constraints, and the energy molrs prices; writing the read
    /// system again gives the same topology.
    #[test]
    fn a_system_reads_back_as_written() {
        let (ff, frame) = read_system(SYSTEM);
        let writer = GromacsTopForcefieldWriter::new().with_precision(17);
        let top = system(&writer, &ff, &frame).unwrap();
        for section in [
            "[ moleculetype ]",
            "[ pairs ]",
            "[ constraints ]",
            "[ dihedrals ]",
        ] {
            assert!(top.contains(section), "{section}:\n{top}");
        }
        let (back, back_frame) = read_system(&top);
        let (a, b) = (pairs_of(&frame), pairs_of(&back_frame));
        assert_eq!(a.len(), b.len());
        for (p, q) in a.iter().zip(&b) {
            assert_eq!((p.0, p.1, p.2), (q.0, q.1, q.2));
            for (x, y) in p.3.iter().zip(&q.3) {
                match (x, y) {
                    (Some(x), Some(y)) => assert!((x - y).abs() <= 1e-12 * x.abs().max(1.0)),
                    (x, y) => assert_eq!(x.is_some(), y.is_some(), "{p:?} vs {q:?}"),
                }
            }
        }
        let c = |f: &molrs::core::Frame| f.get("constraints").unwrap().n_rows();
        assert_eq!(c(&frame), c(&back_frame));
        let n = frame.get("atoms").unwrap().n_rows().unwrap();
        let x = coords(n);
        let (e, e_back) = (energy(&ff, &frame, &x), energy(&back, &back_frame, &x));
        assert!((e - e_back).abs() <= 1e-12 * e.abs(), "{e} vs {e_back}");
        let again = system(&writer, &back, &back_frame).unwrap();
        let tokens = |t: &str| t.split_whitespace().map(str::to_owned).collect::<Vec<_>>();
        assert_eq!(tokens(&top).len(), tokens(&again).len());
        for (u, v) in tokens(&top).iter().zip(tokens(&again)) {
            match (u.parse::<f64>(), v.parse::<f64>()) {
                (Ok(a), Ok(b)) => assert!((a - b).abs() <= 1e-14 * a.abs().max(b.abs())),
                _ => assert_eq!(u, &v),
            }
        }
    }

    /// A water: every pair of it is within three bonds, so its pair list has
    /// no row (a column-less `pairs` block). It writes, the reader's frame or
    /// one whose pair list the writer builds itself.
    #[test]
    fn a_molecule_with_every_pair_excluded_writes() {
        let (ff, frame) = read_system(
            "\
[ defaults ]
1  2  yes  0.5  0.8333
[ atomtypes ]
OW  OW  8  15.999  -0.834  A  0.315  0.6364
HW  HW  1   1.008   0.417  A  0.0    0.0
[ bondtypes ]
OW  HW  1  0.09572  502416.0
[ angletypes ]
HW  OW  HW  1  104.52  628.02
[ moleculetype ]
SOL  2
[ atoms ]
1  OW  1  SOL  OW   1
2  HW  1  SOL  HW1  1
3  HW  1  SOL  HW2  1
[ bonds ]
1  2  1
1  3  1
[ angles ]
2  1  3  1
[ system ]
water
[ molecules ]
SOL  1
",
        );
        let mut bare = frame.clone();
        bare.remove("pairs");
        let writer = GromacsTopForcefieldWriter::new();
        for f in [&frame, &bare] {
            let top = system(&writer, &ff, f).unwrap();
            assert!(top.contains("[ molecules ]"), "{top}");
            assert!(!top.contains("[ pairs ]"), "{top}");
        }
    }

    /// A pair the frame prices that GROMACS cannot (a regular pair within
    /// three bonds, a 1-4 pair beyond them), override cells without their
    /// Lennard-Jones parameters, and a frame pair GROMACS would exclude by
    /// default are refused or written out as exclusions.
    #[test]
    fn the_frame_s_pairs_are_what_gromacs_prices_or_refused() {
        let (ff, frame) = read_system(SYSTEM);
        let writer = GromacsTopForcefieldWriter::new().with_precision(17);
        let with_pairs = |edit: &dyn Fn(&mut molrs::core::Block)| {
            let mut f = frame.clone();
            edit(f.get_mut("pairs").unwrap());
            system(&writer, &ff, &f)
        };
        // C1-C4, a 1-4 pair, flagged regular.
        let err = with_pairs(&|p| {
            let mut flags = p.get("is_14").unwrap().get::<bool>().unwrap().to_owned();
            let r = flags.iter().position(|&b| b).unwrap();
            flags[[r]] = false;
            p.insert("is_14", flags, None).unwrap();
        })
        .unwrap_err();
        assert!(err.contains("within three bonds"), "{err}");
        // An override cell alone.
        let err = with_pairs(&|p| {
            p.remove("epsilon");
        })
        .unwrap_err();
        assert!(err.contains("epsilon and sigma"), "{err}");
        // A regular pair the frame does not price: an [ exclusions ] row.
        let top = with_pairs(&|p| {
            let keep: Vec<usize> = {
                let i = p.get("atomi").unwrap().get::<u64>().unwrap();
                let j = p.get("atomj").unwrap().get::<u64>().unwrap();
                (0..i.len())
                    .filter(|&r| (i[[r]], j[[r]]) != (3, 4))
                    .collect()
            };
            *p = p.select_rows(&keep).unwrap();
        })
        .unwrap();
        assert!(top.contains("[ exclusions ]\n  4  5\n"), "{top}");
        let (_, back) = read_system(&top);
        assert!(!pairs_of(&back).iter().any(|p| (p.0, p.1) == (3, 4)));
        // A pair of two molecules is no exclusion: GROMACS prices it.
        assert_eq!(top.matches("[ exclusions ]").count(), 1, "{top}");
    }

    /// The Coulomb constant is the engine's, not written: a field stating
    /// AMBER's is written as one stating LAMMPS's.
    #[test]
    fn a_stated_coulomb_constant_is_not_written() {
        let mut a = every_supported_style();
        let b = a.clone();
        a.get_style_mut("pair", "coul/cut")
            .unwrap()
            .set_param("coulomb", 332.0522173);
        assert_eq!(write(&a), write(&b));
    }

    /// A long-range Coulomb has no directive, as a cut one has none: the
    /// `.top` holds the charges, and PME / Ewald is the `.mdp`'s
    /// `coulombtype`. The `.top` is the `coul/cut` one; stated Ewald
    /// parameters (`fourier-spacing`, `pme-order`, `ewald-rtol` in the
    /// `.mdp`) are refused.
    #[test]
    fn a_long_range_coulomb_writes_the_coul_cut_top() {
        let field = |coul: &str, params: Params| {
            let mut ff = opls_ff(Some("geometric"));
            ff.def_style("pair", coul, params).unwrap();
            ff
        };
        let cut = write(&field(
            "coul/cut",
            Params::from_iter([("cutoff", 10.0)].iter().copied()),
        ));
        let long = field(
            "coul/long/pme",
            Params::from_iter(
                [
                    ("cutoff", 10.0),
                    ("coulomb", crate::core::constants::gromacs_coulomb_real()),
                ]
                .iter()
                .copied(),
            ),
        );
        assert_eq!(write(&long), cut);
        for key in ["alpha", "order", "grid_z"] {
            let ff = field(
                "coul/long/pme",
                Params::from_iter([("cutoff", 10.0), (key, 4.0)].iter().copied()),
            );
            assert_names(&write_err(&ff), &["coul/long/pme", key, ".mdp"]);
        }
    }

    /// The writer converts from real units; another declared preset is
    /// refused, as is an atom type without a charge outside a system.
    #[test]
    fn units_other_than_real_are_refused() {
        let mut ff = every_supported_style();
        ff.set_units("metal");
        assert_names(&write_err(&ff), &["metal"]);
    }

    /// `dihedral class2`'s torsion is funct 9, k[1 − cos(nφ − φₙ)] =
    /// k[1 + cos(nφ − φₙ − 180°)]: the same energy as read back.
    #[test]
    fn dihedral_class2_is_funct_9_at_its_phase_plus_180() {
        let ff = with_type(
            "dihedral",
            "class2",
            "CT-CT-CT-CT",
            &["CT", "CT", "CT", "CT"],
            Params::from_iter([
                ("k1", 1.5),
                ("phi1", 10.0),
                ("k2", 0.0),
                ("phi2", 0.0),
                ("k3", -0.25),
                ("phi3", 30.0),
            ]),
        );
        let text = write(&ff);
        let rows = section_rows(&text, "dihedraltypes");
        assert_eq!(rows.len(), 2, "{text}");
        assert_row_values(&rows[0][4..], "9", &[190.0, 1.5 * 4.184, 1.0]);
        assert_row_values(&rows[1][4..], "9", &[210.0, -0.25 * 4.184, 3.0]);
    }
}
