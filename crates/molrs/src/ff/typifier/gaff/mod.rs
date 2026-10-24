//! GAFF / GAFF2 — a [`Typifier`] over the compiled `parm` tables.

use std::collections::{BTreeSet, HashMap, HashSet};
use std::sync::OnceLock;

use molrs::core::RelationId;
use molrs::core::TypeName;
use molrs::core::keys;
use molrs::core::schema::block_names::{ANGLES, BONDS, DIHEDRALS, IMPROPERS};
use molrs::core::{Atomistic, NodeId, Props};

use crate::core::NeighborKinds;
use crate::core::constants::AMBER_COULOMB;
use crate::core::constants::VACUUM_DIELECTRIC;
use crate::core::constants::{AMBER_SCEE, AMBER_SCNB};
use crate::ff::forcefield::{DefError, ForceField};
use crate::ff::ir::{CombiningRule, Params, SpecialBonds};
use crate::ff::params::{
    GAFF, GAFF2, PARMCHK, ParmAngleRow, ParmBondRow, ParmDihedralRow, ParmImproperRow,
    ParmNonbondedRow, ParmTable, ParmType,
};
use crate::ff::typifier::BondedTerm;
use crate::ff::typifier::{Annotation, TypeAssignment, Typifier, TypingError};
use crate::ff::typifier::{EmpiricalSet, Provenance};

mod analog;
mod improper;
mod torsion;

/// Which AMBER `parm` force field to match against.
///
/// Pairs one-to-one with the atom-type table that produces the labels:
/// [`AtdParameterSet::Gff`](crate::ff::typifier::AtdParameterSet::Gff) types a
/// molecule for [`Gaff`](Self::Gaff), `Gff2` for [`Gaff2`](Self::Gaff2). Typing
/// with one and parameterising with the other is a category error no signature
/// can catch — `gaff2.dat` declares atom types (`cs`, `sq`, …) that `gaff.dat`
/// has no row for at all.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GaffParameterSet {
    /// `gaff.dat` — GAFF 1.81.
    Gaff,
    /// `gaff2.dat` — GAFF 2.
    Gaff2,
}

impl GaffParameterSet {
    /// Both sets.
    pub const ALL: [GaffParameterSet; 2] = [Self::Gaff, Self::Gaff2];

    /// The set [`name`](Self::name) names.
    ///
    /// # Errors
    ///
    /// An unknown name; there is no default set.
    pub fn from_name(name: &str) -> Result<Self, String> {
        Self::ALL
            .into_iter()
            .find(|set| set.name() == name)
            .ok_or_else(|| {
                let known: Vec<&str> = Self::ALL.iter().map(|set| set.name()).collect();
                format!(
                    "unknown GAFF parameter set {name:?}; expected one of {}",
                    known.join(", ")
                )
            })
    }

    /// The compile-time table this set names.
    pub fn table(self) -> ParmTable {
        match self {
            Self::Gaff => GAFF,
            Self::Gaff2 => GAFF2,
        }
    }

    /// The name of this set's library, and so of every typing output seeded
    /// from it.
    pub fn name(self) -> &'static str {
        match self {
            Self::Gaff => "gaff",
            Self::Gaff2 => "gaff2",
        }
    }

    /// The empirical bond / angle constants that pair with this table.
    fn empirical(self) -> EmpiricalSet {
        match self {
            Self::Gaff => EmpiricalSet::Gaff,
            Self::Gaff2 => EmpiricalSet::Gaff2,
        }
    }

    /// This set's candidate library, built on first use and shared after.
    ///
    /// `gaff.dat` transcribes to some 5,900 candidate rows, and it is the *same*
    /// 5,900 rows for every molecule ever parameterised; rebuilding them per call
    /// cost more than the whole rest of the typing put together. The library is
    /// named after the set ([`name`](Self::name)).
    fn forcefield(self) -> &'static ForceField {
        static GAFF_LIBRARY: OnceLock<ForceField> = OnceLock::new();
        static GAFF2_LIBRARY: OnceLock<ForceField> = OnceLock::new();

        let cell = match self {
            Self::Gaff => &GAFF_LIBRARY,
            Self::Gaff2 => &GAFF2_LIBRARY,
        };
        cell.get_or_init(|| {
            let mut ff = candidate_forcefield(self.table());
            ff.name = self.name().to_owned();
            ff
        })
    }
}

/// Transcribe a whole `parm` table into a [`ForceField`] of candidate rows.
///
/// An input-free constructor over a compiled `ff/params` table: the one
/// definition result it can meet is the table's own, and
/// `tests::candidate_forcefield_defines_without_conflict` proves it `Ok` for
/// GAFF and GAFF2.
fn candidate_forcefield(table: ParmTable) -> ForceField {
    try_candidate_forcefield(table).expect(
        "GAFF/GAFF2 parm table defines without conflict — proved by \
         ff::typifier::gaff::tests::candidate_forcefield_defines_without_conflict",
    )
}

/// The fallible body of [`candidate_forcefield`].
///
/// Every row, wildcards and all, `X` filling a wildcard slot: the library is
/// what [`Typifier::source_forcefield`] hands out, and the output declares its styles.
///
/// The library declares AMBER's 1-4 handling — 1-2 / 1-3 excluded outright, 1-4
/// scaled by 1/SCNB = 1/2 (LJ) and 1/SCEE = 1/1.2 (Coulomb) — so every typing
/// output seeded from it carries the weights.
fn try_candidate_forcefield(table: ParmTable) -> Result<ForceField, DefError> {
    let mut ff = ForceField::new(table.name);
    // The GAFF / GAFF2 tables are kcal/mol and Å (parm*.dat).
    ff.set_units("real");
    ff.set_special_bonds(SpecialBonds {
        lj: [0.0, 0.0, 1.0 / AMBER_SCNB],
        coul: [0.0, 0.0, 1.0 / AMBER_SCEE],
    });

    let atoms = ff.def_style("atom", "full", Params::new())?;
    for row in table.masses {
        atoms.def_type(row.atom_type, &[], Params::from_iter([("mass", row.mass)]))?;
    }

    // AMBER mixes Lennard-Jones by Lorentz–Berthelot; the style states it, as
    // the prmtop reader's does, rather than lean on the kernel's default.
    let mut mixing = Params::new();
    mixing.set("mixing", CombiningRule::Arithmetic.name());
    let lj = ff.def_style("pair", "lj/cut", mixing)?;
    for row in table.nonbonded {
        let name = table.name_of(row.atom_type);
        lj.def_type(
            name,
            &[name],
            Params::from_iter(lj_params(row).iter().copied()),
        )?;
    }
    // GAFF/AMBER uses the unbuffered Coulomb form. The constant is force-field
    // data (and differs measurably from CODATA and MMFF), while `delta = 0`
    // explicitly selects the unbuffered branch of the shared `coul/cut` kernel.
    ff.def_style(
        "pair",
        "coul/cut",
        Params::from_iter([
            ("coulomb", AMBER_COULOMB),
            ("dielectric", VACUUM_DIELECTRIC),
            ("delta", 0.0),
        ]),
    )?;

    let bonds = ff.def_style("bond", "harmonic", Params::new())?;
    for row in table.bonds {
        let ends = [row.i, row.j].map(|ty| table.name_of(ty));
        bonds.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter(bond_params(row).iter().copied()),
        )?;
    }

    let angles = ff.def_style("angle", "harmonic", Params::new())?;
    for row in table.angles {
        let ends = [row.i, row.j, row.k].map(|ty| table.name_of(ty));
        angles.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter(angle_params(row).iter().copied()),
        )?;
    }

    // The DIHE section writes one row per cosine term; a torsion is the whole group
    // of rows sharing a quartet, so the group is ONE candidate.
    let dihedrals = ff.def_style("dihedral", "periodic", Params::new())?;
    let mut groups: Vec<([Option<ParmType>; 4], Vec<&ParmDihedralRow>)> = Vec::new();
    for row in table.dihedrals {
        let slots = [row.i, row.j, row.k, row.l];
        match groups.iter_mut().find(|(seen, _)| *seen == slots) {
            Some((_, rows)) => rows.push(row),
            None => groups.push((slots, vec![row])),
        }
    }
    for (slots, rows) in &groups {
        let ends = slots.map(|slot| slot_name(&table, &slot));
        let params = dihedral_params(rows);
        dihedrals.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter(params.iter().cloned()),
        )?;
    }

    let impropers = ff.def_style("improper", "periodic", Params::new())?;
    for row in table.impropers {
        let ends = [row.i, row.j, row.k, row.l].map(|slot| slot_name(&table, &slot));
        impropers.def_type(
            TypeName::join(&ends).map_err(DefError::Name)?.as_str(),
            &ends,
            Params::from_iter(improper_params(row).iter().copied()),
        )?;
    }

    Ok(ff)
}

/// A row slot's atom-type name, or `X` where the file wrote a wildcard.
fn slot_name(table: &ParmTable, slot: &Option<ParmType>) -> &'static str {
    slot.map_or("X", |ty| table.name_of(ty))
}

/// One `NONBON` row as `lj/cut` params: ε as written, σ = 2·R\*/2^(1/6).
///
/// R\* is **half** the separation at the Lennard-Jones minimum, so the minimum
/// itself is `2·R*` and σ — where the potential crosses zero — is that over the
/// sixth root of two. Reading R\* as σ would inflate every radius by 78%.
///
/// A row with no van der Waals term (`ho`: R\* and ε both zero) is the
/// non-interacting type molrs writes for one everywhere, `σ = 1, ε = 0`
/// ([`lj_ab_to_sigma_epsilon`](crate::ff::potential::pair::lj_ab_to_sigma_epsilon),
/// which reads it from a prmtop's zero `A`/`B`): ε = 0 prices nothing, and σ
/// is non-zero so no consumer divides by zero.
fn lj_params(row: &ParmNonbondedRow) -> [(&'static str, f64); 2] {
    if row.epsilon == 0.0 || row.r_min_half == 0.0 {
        return [("epsilon", 0.0), ("sigma", 1.0)];
    }
    [
        ("epsilon", row.epsilon),
        ("sigma", 2.0 * row.r_min_half / 2f64.powf(1.0 / 6.0)),
    ]
}

/// One `BOND` row as candidate params, in molrs's convention: AMBER's
/// `E = K(r−r₀)²` is LAMMPS's `bond harmonic`, so `k = K`.
fn bond_params(row: &ParmBondRow) -> [(&'static str, f64); 2] {
    [("k", row.force_constant), ("r0", row.length)]
}

/// One `ANGLE` row as candidate params: `k = K`, θ₀ in degrees, as written.
fn angle_params(row: &ParmAngleRow) -> [(&'static str, f64); 2] {
    [("k", row.force_constant), ("theta0", row.angle_deg)]
}

/// The cosine terms of one torsion, in the `k{m}` / `periodicity{m}` / `phase{m}` encoding
/// [`dihedral::Periodic`](crate::ff::potential::dihedral::Periodic)
/// scans upward from `m = 1`. `k = PK / IDIVF`; phases in degrees.
///
/// The terms are numbered in one canonical order — by periodicity, then
/// phase, then `k` — whatever order the `.dat` rows list them in
/// (`ho-oh-c3-c3` writes its `n = 3` term first): the order the prmtop reader
/// numbers a torsion's terms in (`io::amber`'s `canonical_terms`), so one
/// torsion has one encoding wherever it was read.
fn dihedral_params(rows: &[&ParmDihedralRow]) -> Vec<(String, f64)> {
    let mut terms: Vec<(f64, f64, f64)> = rows
        .iter()
        .map(|row| {
            (
                f64::from(row.periodicity),
                row.phase_deg,
                row.barrier / f64::from(row.divisor),
            )
        })
        .collect();
    terms.sort_by(|a, b| {
        a.0.total_cmp(&b.0)
            .then(a.1.total_cmp(&b.1))
            .then(a.2.total_cmp(&b.2))
    });
    let mut out = Vec::with_capacity(terms.len() * 3);
    for (m, (periodicity, phase, k)) in terms.into_iter().enumerate() {
        let m = m + 1;
        out.push((format!("k{m}"), k));
        out.push((format!("periodicity{m}"), periodicity));
        out.push((format!("phase{m}"), phase));
    }
    out
}

/// [`dihedral_params`] with each row's phase rewritten onto `stored`
/// ([`TypeName::phase_for_stored_order`]). `authored` is the row's own
/// spelling, wildcards included; every term of one torsion shares it.
fn dihedral_params_for(
    rows: &[&ParmDihedralRow],
    authored: &[&str],
    stored: &[String],
) -> Vec<(String, f64)> {
    let stored_ref: Vec<&str> = stored.iter().map(String::as_str).collect();
    let adjusted: Vec<ParmDihedralRow> = rows
        .iter()
        .map(|row| ParmDihedralRow {
            phase_deg: TypeName::phase_for_stored_order(authored, &stored_ref, row.phase_deg),
            ..**row
        })
        .collect();
    let refs: Vec<&ParmDihedralRow> = adjusted.iter().collect();
    dihedral_params(&refs)
}

/// The four slot names of `row`, `X` where the file wrote a wildcard.
fn row_slot_names(index: &TableIndex, row: &ParmDihedralRow) -> [&'static str; 4] {
    [&row.i, &row.j, &row.k, &row.l].map(|slot| slot_name(&index.table, slot))
}

/// `value` as parmchk2 writes it into the frcmod, `%.<decimals>f`: the number
/// tleap reads back and prices. An estimate is a frcmod row, so its
/// parameters carry the frcmod's precision, not the analog's (`ne-s6` copies
/// `ne-sy`'s r₀ = 1.6723 and the frcmod says `1.672`).
fn as_printed(value: f64, decimals: usize) -> f64 {
    format!("{value:.decimals$}")
        .parse()
        .expect("a formatted finite number parses")
}

/// `params` with each of `decimals`' keys [as parmchk2 prints it](as_printed).
fn as_frcmod(params: &Params, decimals: &[(&str, usize)]) -> Params {
    let mut out = params.clone();
    for &(key, places) in decimals {
        if let Some(value) = params.get(key).and_then(|v| v.as_num()) {
            out.set(key, as_printed(value, places));
        }
    }
    out
}

/// parmchk2's frcmod precision (`%8.2lf%8.3lf`): bond `k`, `r0`.
const FRCMOD_BOND: [(&str, usize); 2] = [("k", 2), ("r0", 3)];
/// parmchk2's frcmod precision (`%9.3lf%12.3lf`): angle `k`, θ₀.
const FRCMOD_ANGLE: [(&str, usize); 2] = [("k", 3), ("theta0", 3)];
/// parmchk2's frcmod precision (`%11.1lf%15.1lf%12.1lf`): improper `k`,
/// phase, periodicity.
const FRCMOD_IMPROPER: [(&str, usize); 3] = [("k", 1), ("phase", 1), ("periodicity", 1)];

/// A torsion row as parmchk2 writes it into the frcmod
/// (`%4d%9.3lf%14.3lf%16.3lf`: `IDIVF`, `PK`, phase, `PN`).
fn torsion_as_frcmod(row: &ParmDihedralRow) -> ParmDihedralRow {
    ParmDihedralRow {
        barrier: as_printed(row.barrier, 3),
        phase_deg: as_printed(row.phase_deg, 3),
        ..*row
    }
}

/// One `IMPROPER` row as candidate params. An improper's barrier is never divided.
fn improper_params(row: &ParmImproperRow) -> [(&'static str, f64); 3] {
    [
        ("k", row.barrier),
        ("periodicity", f64::from(row.periodicity)),
        ("phase", row.phase_deg),
    ]
}

/// A term of the molecule that neither an exact row nor parmchk2 covers.
///
/// Atom types are the GAFF labels of the term's atoms, in the molecule's own
/// order. Every miss is collected before returning, so one call names every
/// parameter the table is short of, rather than only the first.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum MissingTerm {
    /// The table declares no such atom type: it has no `MASS` row.
    Mass(String),
    /// No `NONBON` row for this atom type. `gaff2.dat` has none for `ow` / `hw`:
    /// the water types take their Lennard-Jones parameters from a water model,
    /// not from the force field.
    Nonbonded(String),
    /// No `BOND` row for this pair of atom types.
    Bond([String; 2]),
    /// No `ANGLE` row for this triple (vertex in the middle).
    Angle([String; 3]),
    /// No `DIHE` row for this quartet.
    Dihedral([String; 4]),
}

impl std::fmt::Display for MissingTerm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Mass(t) => write!(f, "MASS {t}"),
            Self::Nonbonded(t) => write!(f, "NONBON {t}"),
            Self::Bond(t) => write!(f, "BOND {}", t.join("-")),
            Self::Angle(t) => write!(f, "ANGLE {}", t.join("-")),
            Self::Dihedral(t) => write!(f, "DIHE {}", t.join("-")),
        }
    }
}

/// Why a molecule could not be matched against a GAFF table. Reaches callers
/// as its `Display` text, through [`Typifier::assign`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum GaffError {
    /// An atom carries no [`keys::TYPE`] label: the molecule was never typed.
    Untyped {
        /// The atom's 0-based index in graph atom order.
        atom: usize,
    },
    /// Neither the table nor parmchk2 covers one or more of the
    /// molecule's terms.
    ///
    /// An atom type the table does not declare at all ([`MissingTerm::Mass`])
    /// short-circuits: every term touching it would be missing too, and listing
    /// them would bury the one fact that matters.
    Missing {
        /// The upstream table that was searched, e.g. `gaff.dat`.
        table: &'static str,
        /// Every uncovered term, deduplicated, in a stable order.
        terms: Vec<MissingTerm>,
    },
    /// The graph could not be walked — a defect in the input.
    Malformed {
        /// What the graph layer said.
        detail: String,
    },
}

impl From<GaffError> for TypingError {
    fn from(err: GaffError) -> Self {
        let detail = err.to_string();
        match err {
            GaffError::Untyped { .. } | GaffError::Malformed { .. } => {
                TypingError::Malformed { detail }
            }
            GaffError::Missing { .. } => TypingError::MissingTerms { detail },
        }
    }
}

impl std::fmt::Display for GaffError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Untyped { atom } => {
                write!(f, "atom {atom} carries no `{}` label", keys::TYPE)
            }
            Self::Missing { table, terms } => {
                let listed: Vec<String> = terms.iter().map(ToString::to_string).collect();
                write!(
                    f,
                    "{table} has no exact parameter for {} term(s): {}",
                    terms.len(),
                    listed.join(", ")
                )
            }
            Self::Malformed { detail } => write!(f, "{detail}"),
        }
    }
}

impl std::error::Error for GaffError {}

/// The GAFF / GAFF2 typifier: matches a molecule whose atoms already carry GAFF
/// atom types against one `parm` table.
///
/// Run it through [`Typing`](crate::ff::typifier::Typing); its output holds the
/// types the molecules typed so far assigned, under the AMBER 1-4 weights its
/// library declares.
///
/// # Example
///
/// Methane, end to end — perceive + type atoms, match terms, evaluate:
///
/// ```
/// use molrs::core::Atomistic;
/// use molrs::ff::compile::PotentialCompiler;
/// use molrs::ff::potential::intramolecular_pairs;
/// use molrs::ff::typifier::Typing;
/// use molrs::ff::typifier::{AtdParameterSet, AtdTypifier};
/// use molrs::ff::typifier::{GaffParameterSet, GaffTypifier};
/// use molrs::core::{Element, PropValue, keys};
/// use molrs::props;
///
/// # fn main() -> Result<(), Box<dyn std::error::Error>> {
/// let mut mol = Atomistic::new();
/// let c = mol.add_atom(props! {
///     keys::ATOMIC_NUMBER => Element::C,
///     keys::X => 0.000,
///     keys::Y => 0.000,
///     keys::Z => 0.000,
/// })?;
/// for (x, y, z) in [
///     (0.629, 0.629, 0.629),
///     (-0.629, -0.629, 0.629),
///     (-0.629, 0.629, -0.629),
///     (0.629, -0.629, -0.629),
/// ] {
///     let h = mol.add_atom(props! {
///         keys::ATOMIC_NUMBER => Element::H,
///         keys::X => x,
///         keys::Y => y,
///         keys::Z => z,
///     })?;
///     mol.add_bond(c, h, props! {})?;
/// }
///
/// // ATD types the atoms (perceiving the BCC bond types on the way through) …
/// let labelled = Typing::new(AtdTypifier::new(AtdParameterSet::Gff)).typify(&mol)?;
/// // … and GAFF matches the bonded terms of the labelled molecule.
/// let mut gaff = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
/// let typed = gaff.typify(&labelled)?;
/// let ff = gaff.forcefield();
///
/// // Every bond's `type` is now the force field's NAME, not a perceived integer.
/// let name = PropValue::Str("c3-hc".to_owned());
/// for (_, bond) in typed.bonds() {
///     assert_eq!(bond.props.get(keys::TYPE), Some(&name));
/// }
///
/// let mut frame = typed.to_frame()?;
/// let special = ff.special_bonds().ok_or("the force field states no special_bonds")?;
/// let pairs = intramolecular_pairs(&frame, special, None)?;
/// frame.insert("pairs", pairs);
/// let potentials = PotentialCompiler::new(ff).compile(&frame)?;
/// # Ok(())
/// # }
/// ```
///
/// [`GaffTypifier`] matches a molecule whose atoms already carry GAFF atom types
/// (what [`AtdTypifier`](crate::ff::typifier::AtdTypifier) with
/// [`AtdParameterSet::Gff`](crate::ff::typifier::AtdParameterSet::Gff) stamps, or
/// labels written by any other means): it enumerates the molecule's bonded terms
/// and looks every one of them up in the [`GAFF`] / [`GAFF2`] static table.
/// Nothing is parsed: the tables are `&'static` Rust data (see
/// [`crate::ff::params`]). Typing atoms is not this typifier's job; the caller
/// composes the two:
///
/// ```ignore
/// let labelled = Typing::new(AtdTypifier::new(AtdParameterSet::Gff)).typify(&mol)?;
/// let mut gaff = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
/// let typed = gaff.typify(&labelled)?;
/// ```
///
/// # Exact rows first, then parmchk2
///
/// A term is looked up first **only** against rows whose every slot is a
/// concrete atom type. What no such row covers is estimated exactly as
/// AmberTools' parmchk2 estimates it: bonds and angles by analogy, torsions
/// by parmchk2's torsion search (whose estimate tleap prefers, then the wildcard row
/// `X-j-k-X`), impropers by its improper search. A wildcard row that covers a term is
/// a parameter, anything reached by analogy or formula is an estimate and says
/// so (see `Typifier::assign` on [`GaffTypifier`]). A term parmchk2 cannot
/// estimate either (it writes a zero marked `ATTN, need revision`) is missing,
/// and every missing term is reported at once.
///
/// Matching a term in either orientation (`c3-c3-oh` covers `oh-c3-c3`) is not a
/// fallback but the undirected nature of a bonded term, and the generator
/// guarantees no table holds both a term and its reverse as separate rows.
///
/// Impropers are the one exception to "missing is an error": which atoms carry an improper, in which
/// order, and at which barrier is decided by parmchk2's improper search and
/// tleap's improper matching, and this typifier reproduces both exactly. An
/// improper exists where tleap finds a row (the table's, or parmchk2's
/// estimate at a centre `PARMCHK.DAT` flags as planar), and nowhere else.
///
/// # Units
///
/// The table holds what `gaff.dat` says; the kernels want molrs's convention,
/// which is LAMMPS's and, for every bonded term, AMBER's own: un-halved `K`,
/// degrees. The candidate library ([`Typifier::source_forcefield`]) is the boundary
/// between the two, and the only place a value changes:
///
/// | upstream | molrs |
/// |---|---|
/// | `E = K(r−r₀)²` | `bond harmonic`, `k = K` |
/// | `E = K(θ−θ₀)²`, θ₀ in degrees | `angle harmonic`, `k = K`, `theta0` in degrees |
/// | phases in degrees | degrees |
/// | one `PK` shared by `IDIVF` torsions | one `k` per torsion: `k = PK/IDIVF` |
/// | R\*, half the LJ minimum separation | σ = 2·R\*/2^(1/6) |
/// | an improper in AMBER's atom order, centre third | the same order (see [`improper::Periodic`](crate::ff::potential::improper::Periodic)) |
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GaffTypifier {
    set: GaffParameterSet,
}

impl GaffTypifier {
    /// A typifier over `set`'s table. The library it reads is shared per set,
    /// so construction is free.
    pub fn new(set: GaffParameterSet) -> Self {
        Self { set }
    }

    /// The body of [`Typifier::assign`], with the typed error.
    fn match_terms(&self, graph: &mut Atomistic) -> Result<TypeAssignment, GaffError> {
        let index = TableIndex::new(self.set.table());
        let type_of = index.intern_atoms(graph)?;
        let mut missed = Misses::default();

        // Topology first, so every positional vector below is read off the
        // graph `TypeAssignment::apply_to` stamps: angles and dihedrals regenerated
        // from the bond graph, impropers rebuilt as AmberTools builds them.
        graph
            .generate_topology(true, true, false, true)
            .map_err(malformed)?;
        let mut improper_terms = add_impropers(graph, &index, &type_of)?;

        let mut m = TypeAssignment::default();

        // --- atoms: mass (atom/full) + Lennard-Jones (pair rows) ---
        let used: BTreeSet<ParmType> = type_of.values().copied().collect();
        for &ty in &used {
            // Interning proved the MASS row exists. The NONBON row is a separate
            // section and can be absent (gaff2's `ow` / `hw`).
            if index.nonbonded(ty).is_none() {
                missed.note(MissingTerm::Nonbonded(index.name_of(ty).to_owned()));
            }
        }
        let library = self.set.forcefield();
        m.nodes = graph
            .atoms()
            .map(|(id, _)| {
                let name = index.name_of(type_of[&id]);
                let mass = index.table.masses[usize::from(type_of[&id].0)].mass;
                vec![(
                    keys::TYPE.to_owned(),
                    Annotation::Type {
                        style: "full".to_owned(),
                        name: name.to_owned(),
                        endpoints: Vec::new(),
                        params: Params::from_iter([("mass", mass)]),
                    },
                )]
            })
            .collect();

        // parmchk2 walks the molecule in atom and bond order; its estimates for
        // the bonds and angles the table lacks come first.
        let order: Vec<NodeId> = graph.atoms().map(|(id, _)| id).collect();
        let neighbours: Vec<Vec<NodeId>> = order
            .iter()
            .map(|&a| {
                graph
                    .neighbors(a, NeighborKinds::BONDS)
                    .map(|(n, _)| n)
                    .collect()
            })
            .collect();
        let names: HashMap<NodeId, &'static str> = type_of
            .iter()
            .map(|(&atom, &ty)| (atom, index.name_of(ty)))
            .collect();
        let analogs = analog::Analogs::new(
            &order,
            &neighbours,
            &names,
            index.table,
            &PARMCHK,
            self.set.empirical().table(),
        );

        // --- bonds ---
        let bonds: Vec<(NodeId, NodeId)> = graph
            .bonds()
            .map(|(_, b)| (b.nodes[0], b.nodes[1]))
            .collect();
        for (i, j) in bonds {
            let (ti, tj) = (type_of[&i], type_of[&j]);
            let resolved = match index.bond(ti, tj) {
                Some(row) => Some((
                    oriented(BondedTerm::Bond(index.names([row.i, row.j]))),
                    Bonded::matched(Params::from_iter(bond_params(row))),
                )),
                None => {
                    let names = index.names([ti, tj]);
                    match analogs.bond([index.name_of(ti), index.name_of(tj)]) {
                        Some(estimate) => Some(Bonded::estimated(
                            &BondedTerm::Bond(names),
                            as_frcmod(&estimate.params, &FRCMOD_BOND),
                            estimate.provenance.clone(),
                        )),
                        None => {
                            missed.note(MissingTerm::Bond(names));
                            None
                        }
                    }
                }
            };
            m.link_mut(BONDS)
                .push(Bonded::annotation("harmonic", resolved)?);
        }

        // --- angles ---
        let angles: Vec<[NodeId; 3]> = graph
            .angles()
            .map(|(_, a)| [a.nodes[0], a.nodes[1], a.nodes[2]])
            .collect();
        for [i, j, k] in angles {
            let (ti, tj, tk) = (type_of[&i], type_of[&j], type_of[&k]);
            let resolved = match index.angle(ti, tj, tk) {
                Some(row) => Some((
                    oriented(BondedTerm::Angle(index.names([row.i, row.j, row.k]))),
                    Bonded::matched(Params::from_iter(angle_params(row))),
                )),
                None => {
                    let names = index.names([ti, tj, tk]);
                    let types = [ti, tj, tk].map(|ty| index.name_of(ty));
                    match analogs.angle(types) {
                        Some(estimate) => Some(Bonded::estimated(
                            &BondedTerm::Angle(names),
                            as_frcmod(&estimate.params, &FRCMOD_ANGLE),
                            estimate.provenance.clone(),
                        )),
                        None => {
                            missed.note(MissingTerm::Angle(names));
                            None
                        }
                    }
                }
            };
            m.link_mut(ANGLES)
                .push(Bonded::annotation("harmonic", resolved)?);
        }

        // --- dihedrals ---
        // A torsion the exact index misses is not yet missing: parmchk2 may
        // estimate it (a specific frcmod row, which tleap prefers), else the
        // wildcard rows — most of the DIHE section — may cover it.
        let torsions = torsion::Torsions::new(&order, &neighbours, &names, index.table, &PARMCHK);
        let dihedrals: Vec<[NodeId; 4]> = graph
            .dihedrals()
            .map(|(_, d)| [d.nodes[0], d.nodes[1], d.nodes[2], d.nodes[3]])
            .collect();
        for atoms in dihedrals {
            let quartet = atoms.map(|a| type_of[&a]);
            let query_names = index.names(quartet);
            let term = BondedTerm::Dihedral(query_names.clone());
            // The type, and the phase on it, are this orientation. The graph
            // quartet is moved to it after every term has resolved.
            let stored: Vec<String> = term.endpoints().into_iter().map(str::to_owned).collect();
            let resolved = match index.dihedral(quartet) {
                Some(rows) => {
                    let authored = row_slot_names(&index, rows[0]);
                    Some((
                        stored.clone(),
                        Bonded::matched(Params::from_iter(dihedral_params_for(
                            rows, &authored, &stored,
                        ))),
                    ))
                }
                None => {
                    let canon = torsion::canonical(quartet.map(|ty| index.name_of(ty)));
                    if let Some(estimate) = torsions.estimate(&canon) {
                        let authored = row_slot_names(&index, estimate.rows[0]);
                        let printed: Vec<ParmDihedralRow> = estimate
                            .rows
                            .iter()
                            .map(|row| torsion_as_frcmod(row))
                            .collect();
                        let printed: Vec<&ParmDihedralRow> = printed.iter().collect();
                        Some(Bonded::estimated(
                            &term,
                            Params::from_iter(dihedral_params_for(&printed, &authored, &stored)),
                            estimate.provenance.clone(),
                        ))
                    } else if let Some(rows) = torsions.general(canon[1], canon[2]) {
                        let authored = row_slot_names(&index, rows[0]);
                        Some((
                            stored.clone(),
                            Bonded::matched(Params::from_iter(dihedral_params_for(
                                &rows, &authored, &stored,
                            ))),
                        ))
                    } else {
                        missed.note(MissingTerm::Dihedral(query_names));
                        None
                    }
                }
            };
            m.link_mut(DIHEDRALS)
                .push(Bonded::annotation("periodic", resolved)?);
        }

        // --- impropers: positional against the rows `add_impropers` left ---
        let improper_ids: Vec<RelationId> = graph.impropers().map(|(id, _)| id).collect();
        for id in improper_ids {
            m.link_mut(IMPROPERS)
                .push(Bonded::annotation("periodic", improper_terms.remove(&id))?);
        }

        let terms = missed.into_terms();
        if !terms.is_empty() {
            return Err(GaffError::Missing {
                table: index.table.name,
                terms,
            });
        }

        // Every proper resolved, so its phase is the phase of the oriented
        // type. Perception left the quartet at `j < k`; store that same
        // orientation or the kernel prices φ with the wrong sign.
        store_propers_in_type_order(graph, &type_of, &index)?;

        let used: HashSet<&str> = used.iter().map(|&ty| index.name_of(ty)).collect();
        m.declare_styles_of(library);
        m.add_pairs_among(library, &used);
        Ok(m)
    }
}

impl Typifier for GaffTypifier {
    /// Match the bonded terms of a molecule whose atoms carry GAFF types.
    ///
    /// Angles and dihedrals are regenerated from the bond graph onto `graph`;
    /// impropers are rebuilt as parmchk2 + tleap build them (see the `improper` stage of this typifier),
    /// in AMBER's central-atom-third order. Atom order and bond order are read
    /// as antechamber reads a mol2 file's: they decide parmchk2's improper
    /// estimates and tleap's atom order, as they do in AmberTools. Atoms get `type` as a
    /// type under `atom/full` with the table's `mass`; every bond, angle,
    /// dihedral and improper gets `type` as a type under `bond/harmonic`,
    /// `angle/harmonic`, `dihedral/periodic` or `improper/periodic`. A term an
    /// bond, angle or dihedral is named by [`BondedTerm::type_name`], its
    /// types in [`TypeName::orient`]'s orientation (`c3-c3-oh-ho`, however
    /// the covering `.dat` row is written). A proper is stored in that same
    /// atom order, and its phase is the phase of that order
    /// ([`TypeName::phase_for_stored_order`]: the covering row's phase, negated
    /// when the row is written the other way; a half-turn stays `180`). An
    /// improper is named by its types in AMBER's order; a dihedral's cosine
    /// terms are numbered by periodicity, then phase, then `k`. A term estimated rather than covered by a row
    /// has the same name, its params carry the provenance keys `estimated`,
    /// `estimate_penalty`, `estimate_method` and `estimate_analog`, and its
    /// numbers are the frcmod's, at the precision parmchk2 prints them. These
    /// are the names and numbers the prmtop reader gives a prmtop tleap built
    /// for the molecule, so the two force fields merge
    /// ([`ForceField::merge`]). Styles: every library style, in library
    /// order; `lj/cut` states `mixing = arithmetic`. Pairs: the library's
    /// `lj/cut` rows of the atom types used.
    ///
    /// # The bond `type` is the force field's
    ///
    /// Perception keeps its answer in its own
    /// [`BCC_BOND_TYPE`](molrs::core::keys::BCC_BOND_TYPE) prop, so the
    /// bond's [`keys::TYPE`] is free for the force-field type *name* (`c3-hc`),
    /// which `to_frame` writes to the `bonds` block's `type` column and every
    /// bonded kernel resolves its parameters by.
    ///
    /// # Errors
    ///
    /// An atom that carries no [`keys::TYPE`]; a table that declares none of an
    /// atom's type, or covers — exactly, by wildcard or by estimate — none of a
    /// term, listing **every** such term; a graph that cannot be walked.
    fn assign(&self, graph: &mut Atomistic) -> Result<TypeAssignment, TypingError> {
        Ok(self.match_terms(graph)?)
    }

    /// The set's candidate library: every row of the table (wildcards
    /// included) in molrs's units, the `atom/full`, `pair/lj/cut`,
    /// `pair/coul/cut`, `bond/harmonic`, `angle/harmonic`, `dihedral/periodic`
    /// and `improper/periodic` styles, and AMBER's special_bonds. Built once per
    /// set.
    fn source_forcefield(&self) -> &ForceField {
        self.set.forcefield()
    }
}

/// One bonded term of the molecule: its parameters, and — when they were estimated
/// rather than looked up — what that cost.
///
/// The parameters are in the library's convention whether they came from a row
/// of the table or from parmchk2's search: that is the point of the convention, and
/// it is why one struct serves all four arities. The provenance rides with them
/// rather than being reconstructed afterwards — an estimate that does not say so
/// is not auditable.
struct Bonded {
    params: Params,
    estimate: Option<Provenance>,
}

impl Bonded {
    fn matched(params: Params) -> Self {
        Self {
            params,
            estimate: None,
        }
    }

    /// A term parmchk2 estimated: its [`BondedTerm::endpoints`], `params`
    /// and the provenance.
    fn estimated(term: &BondedTerm, params: Params, provenance: Provenance) -> (Vec<String>, Self) {
        (
            term.endpoints().into_iter().map(str::to_owned).collect(),
            Self {
                params,
                estimate: Some(provenance),
            },
        )
    }

    /// The annotations of one term: `type` → the term's type under `style`,
    /// on its endpoints and named by their [`TypeName::join`], with the
    /// provenance keys written into its params when it was estimated. Nothing
    /// for an unresolved term (a miss, which fails the match).
    ///
    /// The name is the endpoints' alone, estimated or not: an estimate says
    /// so in its params, never in its name, so a term has the name every
    /// other source of the force field gives it (a prmtop of the same
    /// molecule names `f-c3-sy` `f-c3-sy`). Where a later molecule estimates
    /// a name differently, the output names that one `<name>@<n>`
    /// ([`TypeAssignment::apply_to`]).
    fn annotation(
        style: &str,
        resolved: Option<(Vec<String>, Self)>,
    ) -> Result<Vec<(String, Annotation)>, GaffError> {
        let Some((endpoints, term)) = resolved else {
            return Ok(Vec::new());
        };
        let ends: Vec<&str> = endpoints.iter().map(String::as_str).collect();
        let name = TypeName::join(&ends).map_err(malformed)?;
        let Bonded {
            mut params,
            estimate,
        } = term;
        if let Some(provenance) = &estimate {
            provenance.apply_to(&mut params);
        }
        Ok(vec![(
            keys::TYPE.to_owned(),
            Annotation::Type {
                style: style.to_owned(),
                name: name.to_string(),
                endpoints,
                params,
            },
        )])
    }
}

/// A proper term's endpoints in the orientation its type is stored in
/// ([`BondedTerm::endpoints`]: [`TypeName::orient`]'s), whichever way the
/// `.dat` row that covers it is written (`ho-oh-c3-c3` is `c3-c3-oh-ho`).
fn oriented(term: BondedTerm) -> Vec<String> {
    term.endpoints().into_iter().map(str::to_owned).collect()
}

/// Store each proper in the orientation its type name uses
/// ([`TypeName::orient`] of the query types). Perception emits `j < k`; a
/// quartet whose types read reversed is stored reversed, so the signed φ
/// the kernel prices is the φ of the type's phase. The relation sequence is
/// unchanged, which is the sequence the assignment's dihedral rows follow.
fn store_propers_in_type_order(
    graph: &mut Atomistic,
    type_of: &HashMap<NodeId, ParmType>,
    index: &TableIndex,
) -> Result<(), GaffError> {
    let ids: Vec<_> = graph.dihedral_ids().collect();
    let stored = graph.del_dihedrals(&ids).map_err(malformed)?;
    for rel in stored {
        let mut nodes = rel.nodes;
        let names: Vec<&str> = nodes.iter().map(|id| index.name_of(type_of[id])).collect();
        if TypeName::reads_reversed(&names) {
            nodes.reverse();
        }
        graph
            .add_dihedral(nodes[0], nodes[1], nodes[2], nodes[3], Props::new())
            .map_err(malformed)?;
    }
    Ok(())
}

/// Rebuild the impropers of `graph` as AmberTools would (see [`improper`]) and
/// resolve each one, by id.
///
/// Every improper is named by its four types in AMBER's order, centre third;
/// one parmchk2 estimated (by analogy, or its default) carries the
/// provenance keys and the frcmod's precision.
fn add_impropers(
    graph: &mut Atomistic,
    index: &TableIndex,
    type_of: &HashMap<NodeId, ParmType>,
) -> Result<HashMap<RelationId, (Vec<String>, Bonded)>, GaffError> {
    // Rebuild from scratch: a pre-existing improper would survive with a label
    // this force field never defines.
    let existing: Vec<_> = graph.improper_ids().collect();
    graph.del_impropers(&existing).map_err(malformed)?;
    let names: HashMap<NodeId, &'static str> = type_of
        .iter()
        .map(|(&atom, &ty)| (atom, index.name_of(ty)))
        .collect();
    let mut terms = HashMap::new();
    for term in improper::impropers(graph, index.table, &PARMCHK, &names) {
        let [i, j, k, l] = term.atoms;
        let id = graph
            .add_improper(i, j, k, l, Props::new())
            .map_err(malformed)?;
        let params = match term.estimate {
            Some(_) => as_frcmod(&term.params, &FRCMOD_IMPROPER),
            None => term.params,
        };
        let bonded = Bonded {
            params,
            estimate: term.estimate,
        };
        terms.insert(id, (term.types.to_vec(), bonded));
    }
    Ok(terms)
}

fn malformed(e: impl std::fmt::Display) -> GaffError {
    GaffError::Malformed {
        detail: e.to_string(),
    }
}

/// The missing terms of one molecule, deduplicated and kept in first-seen order.
#[derive(Default)]
struct Misses {
    seen: BTreeSet<MissingTerm>,
    terms: Vec<MissingTerm>,
}

impl Misses {
    fn note(&mut self, term: MissingTerm) {
        if self.seen.insert(term.clone()) {
            self.terms.push(term);
        }
    }

    fn into_terms(self) -> Vec<MissingTerm> {
        self.terms
    }
}

// ---------------------------------------------------------------------------
// The exact-match index
// ---------------------------------------------------------------------------

/// Exact-match lookups over one [`ParmTable`], built once per call.
///
/// Only rows with no wildcard slot are indexed. Keys are the table's own
/// [`ParmType`] indices — one byte each, so a bond key is two bytes and a
/// dihedral key four. They are stored in the row's own orientation and looked up
/// in both, which is sound precisely because the generator rejects a table that
/// holds both a term and its reverse.
struct TableIndex {
    table: ParmTable,
    /// Atom-type name -> its index. The only string keying in the whole path.
    names: HashMap<&'static str, ParmType>,
    nonbonded: HashMap<ParmType, &'static ParmNonbondedRow>,
    bonds: HashMap<[ParmType; 2], &'static ParmBondRow>,
    angles: HashMap<[ParmType; 3], &'static ParmAngleRow>,
    /// Every cosine term of a quartet, in file order.
    dihedrals: HashMap<[ParmType; 4], Vec<&'static ParmDihedralRow>>,
}

impl TableIndex {
    fn new(table: ParmTable) -> Self {
        let mut dihedrals: HashMap<[ParmType; 4], Vec<&'static ParmDihedralRow>> = HashMap::new();
        for row in table.dihedrals {
            // A wildcard row is the fallback matcher's business, not this one's.
            if let (Some(i), Some(j), Some(k), Some(l)) = (row.i, row.j, row.k, row.l) {
                dihedrals.entry([i, j, k, l]).or_default().push(row);
            }
        }

        Self {
            table,
            names: table
                .masses
                .iter()
                .enumerate()
                .map(|(idx, row)| (row.atom_type, ParmType(idx as u8)))
                .collect(),
            nonbonded: table.nonbonded.iter().map(|r| (r.atom_type, r)).collect(),
            bonds: table.bonds.iter().map(|r| ([r.i, r.j], r)).collect(),
            angles: table.angles.iter().map(|r| ([r.i, r.j, r.k], r)).collect(),
            dihedrals,
        }
    }

    /// Resolve every atom's label of `typed` to the table's [`ParmType`].
    ///
    /// Interning up front is what lets every later lookup key on a one-byte index
    /// rather than on a string. It is also the one check that short-circuits: an
    /// atom type the table never declares makes every term touching it missing,
    /// so reporting those terms as well would bury the actual fault.
    fn intern_atoms(&self, typed: &Atomistic) -> Result<HashMap<NodeId, ParmType>, GaffError> {
        let mut type_of = HashMap::new();
        let mut undeclared: BTreeSet<String> = BTreeSet::new();

        for (position, (id, atom)) in typed.atoms().enumerate() {
            let label = atom
                .get_str(keys::TYPE)
                .ok_or(GaffError::Untyped { atom: position })?;
            match self.names.get(label) {
                Some(&interned) => {
                    type_of.insert(id, interned);
                }
                None => {
                    undeclared.insert(label.to_owned());
                }
            }
        }

        if !undeclared.is_empty() {
            return Err(GaffError::Missing {
                table: self.table.name,
                terms: undeclared.into_iter().map(MissingTerm::Mass).collect(),
            });
        }
        Ok(type_of)
    }

    fn name_of(&self, ty: ParmType) -> &'static str {
        self.table.name_of(ty)
    }

    /// The type names of a term, for a [`MissingTerm`] or a [`BondedTerm`].
    fn names<const N: usize>(&self, types: [ParmType; N]) -> [String; N] {
        types.map(|ty| self.name_of(ty).to_owned())
    }

    fn nonbonded(&self, ty: ParmType) -> Option<&'static ParmNonbondedRow> {
        self.nonbonded.get(&ty).copied()
    }

    fn bond(&self, i: ParmType, j: ParmType) -> Option<&'static ParmBondRow> {
        self.bonds
            .get(&[i, j])
            .or_else(|| self.bonds.get(&[j, i]))
            .copied()
    }

    fn angle(&self, i: ParmType, j: ParmType, k: ParmType) -> Option<&'static ParmAngleRow> {
        self.angles
            .get(&[i, j, k])
            .or_else(|| self.angles.get(&[k, j, i]))
            .copied()
    }

    fn dihedral(&self, [i, j, k, l]: [ParmType; 4]) -> Option<&[&'static ParmDihedralRow]> {
        self.dihedrals
            .get(&[i, j, k, l])
            .or_else(|| self.dihedrals.get(&[l, k, j, i]))
            .map(Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    //! `Typing::typify` with a `GaffTypifier`, on hand-built molecules whose atom
    //! `type`s are stamped by hand. No ATD runs here: typing atoms is ATD's job,
    //! and this typifier starts from labels however they were written. Every
    //! expectation is read off the table by hand (`ff/params/gaff.rs`,
    //! `ff/params/gaff2.rs`), never captured from another program.

    use crate::props;
    use molrs::core::Element;
    use std::collections::{BTreeMap, BTreeSet};

    use molrs::core::Atomistic;
    use molrs::core::keys;

    use super::{
        GaffParameterSet, GaffTypifier, angle_params, bond_params, try_candidate_forcefield,
    };
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::SpecialBonds;
    use crate::ff::typifier::Typing;

    // -- fixtures ----------------------------------------------------------------

    /// Methane, `C` + 4 `H`, four C-H bonds and nothing else: no angle is
    /// written, so every angle the output holds was enumerated by the typifier.
    /// `h_types` are stamped on the hydrogens in order; `None` leaves that
    /// hydrogen without a `type`.
    fn methane(h_types: [Option<&str>; 4]) -> Atomistic {
        let mut mol = Atomistic::new();
        let c = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::C })
            .unwrap();
        mol.set_atom(c, keys::TYPE, "c3").expect("stamp c3");
        for h_type in h_types {
            let h = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                .unwrap();
            if let Some(label) = h_type {
                mol.set_atom(h, keys::TYPE, label).expect("stamp hc");
            }
            mol.add_bond(c, h, props! {}).expect("C-H bond");
        }
        mol
    }

    /// A two-atom molecule `a-b`, each atom stamped with its given type.
    fn diatomic((sym_a, type_a): (&str, &str), (sym_b, type_b): (&str, &str)) -> Atomistic {
        let mut mol = Atomistic::new();
        let a = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::by_symbol(sym_a).unwrap() })
            .unwrap();
        mol.set_atom(a, keys::TYPE, type_a).expect("stamp a");
        let b = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::by_symbol(sym_b).unwrap() })
            .unwrap();
        mol.set_atom(b, keys::TYPE, type_b).expect("stamp b");
        mol.add_bond(a, b, props! {}).expect("a-b bond");
        mol
    }

    /// Water `hw-ow-hw`: two O-H bonds, no angle written.
    fn water() -> Atomistic {
        let mut mol = Atomistic::new();
        let o = mol
            .add_atom(props! { keys::ATOMIC_NUMBER => Element::O })
            .unwrap();
        mol.set_atom(o, keys::TYPE, "ow").expect("stamp ow");
        for _ in 0..2 {
            let h = mol
                .add_atom(props! { keys::ATOMIC_NUMBER => Element::H })
                .unwrap();
            mol.set_atom(h, keys::TYPE, "hw").expect("stamp hw");
            mol.add_bond(o, h, props! {}).expect("O-H bond");
        }
        mol
    }

    /// Per `(category, style)` of `ff`, the set of type names it defines;
    /// styles holding no type are left out.
    fn output_names(ff: &ForceField) -> BTreeMap<(String, String), BTreeSet<String>> {
        ff.styles()
            .iter()
            .filter_map(|s| {
                let names: BTreeSet<String> = s
                    .defs()
                    .collect_type_params()
                    .into_iter()
                    .map(|(name, _)| name)
                    .collect();
                (!names.is_empty()).then(|| ((s.category().to_owned(), s.name().to_owned()), names))
            })
            .collect()
    }

    fn names(list: &[&str]) -> BTreeSet<String> {
        list.iter().map(|s| (*s).to_owned()).collect()
    }

    // -- tests -------------------------------------------------------------------

    /// Hand-stamped methane (`c3`, 4 x `hc`) is covered by exact `gaff.dat`
    /// rows: MASS/NONBON `c3` and `hc`, BOND `c3-hc`, ANGLE `hc-c3-hc`. It has
    /// no torsion, and `c3` is sp3 with four neighbours, so no improper. The
    /// output holds exactly those types, each in the style GAFF defines it
    /// under, and nothing else.
    #[test]
    fn gaff_typing_hand_stamped_methane_outputs_exactly_its_types() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        typing
            .typify(&methane([Some("hc"); 4]))
            .expect("hand-stamped methane is fully covered by gaff.dat");

        let key = |c: &str, s: &str| (c.to_owned(), s.to_owned());
        let expected = BTreeMap::from([
            (key("atom", "full"), names(&["c3", "hc"])),
            (key("pair", "lj/cut"), names(&["c3", "hc"])),
            (key("bond", "harmonic"), names(&["c3-hc"])),
            (key("angle", "harmonic"), names(&["hc-c3-hc"])),
        ]);
        assert_eq!(output_names(typing.forcefield()), expected);
    }

    /// The typifier types terms from atom labels it is handed; an atom that
    /// carries none cannot be matched and is an error, not a guess.
    #[test]
    fn gaff_typing_an_atom_without_a_type_is_an_error() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        let result = typing.typify(&methane([Some("hc"), Some("hc"), Some("hc"), None]));
        assert!(
            result.is_err(),
            "an atom with no `type` must fail, got {result:?}"
        );
    }

    /// AMBER excludes 1-2 / 1-3 and scales 1-4 by 1/SCNB = 1/2 (LJ) and
    /// 1/SCEE = 1/1.2 (Coulomb). The library declares them, so the output a
    /// fresh `Typing` seeds from it carries them before any molecule is typed.
    #[test]
    fn gaff_typing_output_declares_the_amber_special_bonds() {
        let typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        let amber = SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 1.0 / 1.2],
        };
        assert_eq!(typing.forcefield().special_bonds(), Some(&amber));
    }

    /// `gaff2.dat` declares `ow` and `hw` (MASS rows) but gives neither a NONBON
    /// row — the water types take their Lennard-Jones terms from a water model.
    /// Its BOND `ow-hw` and ANGLE `hw-ow-hw` rows exist. Both nonbonded misses
    /// are reported together, not just the first.
    #[test]
    fn gaff_typing_a_table_miss_lists_every_missing_term() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff2));
        let err = typing
            .typify(&water())
            .expect_err("gaff2.dat has no NONBON row for ow or hw")
            .to_string();
        for term in ["NONBON ow", "NONBON hw"] {
            assert!(
                err.contains(term),
                "the error must list `{term}`, got: {err}"
            );
        }
    }

    /// `gaff.dat` has no BOND row for `hc-br` (no GAFF-typed molecule bonds H to
    /// Br), and parmchk2 finds no analog for one: it writes the bond with a zero
    /// length, `ATTN, need revision`. molrs reports it missing instead.
    #[test]
    fn a_bond_parmchk2_cannot_estimate_is_missing() {
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        let err = typing
            .typify(&diatomic(("H", "hc"), ("Br", "br")))
            .expect_err("parmchk2 has no analog for hc-br")
            .to_string();
        assert!(err.contains("BOND hc-br"), "{err}");
    }

    /// Caffeine under GAFF (antechamber's types, its bond order): parmchk2
    /// (AmberTools 26.1) writes `c -cc-na   68.700  123.270   same as
    /// c2-cc-na, penalty score=  2.6` — `c` → `c2` at an angle end costs
    /// `0.5·ba + 0.5·baf` of its `CORR` row (2.9, 2.3). The estimate says so.
    #[test]
    fn caffeines_c_cc_na_angle_is_parmchk2s_estimate() {
        let caffeine = molecule(
            &[
                ("C", "c3"),
                ("N", "na"),
                ("C", "cc"),
                ("N", "nd"),
                ("C", "cd"),
                ("C", "cc"),
                ("C", "c"),
                ("O", "o"),
                ("N", "n"),
                ("C", "c3"),
                ("C", "c"),
                ("O", "o"),
                ("N", "n"),
                ("C", "c3"),
                ("H", "h1"),
                ("H", "h1"),
                ("H", "h1"),
                ("H", "h5"),
                ("H", "h1"),
                ("H", "h1"),
                ("H", "h1"),
                ("H", "h1"),
                ("H", "h1"),
                ("H", "h1"),
            ],
            &[
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),
                (4, 5),
                (1, 5),
                (5, 6),
                (6, 7),
                (6, 8),
                (8, 9),
                (8, 10),
                (10, 11),
                (10, 12),
                (4, 12),
                (12, 13),
                (0, 14),
                (0, 15),
                (0, 16),
                (2, 17),
                (9, 18),
                (9, 19),
                (9, 20),
                (13, 21),
                (13, 22),
                (13, 23),
            ],
        );
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        typing.typify(&caffeine).expect("caffeine types");
        let angles = params_of(typing.forcefield(), "angle");
        let estimated: Vec<_> = angles
            .iter()
            .filter(|(_, p)| p.get("estimated").and_then(|v| v.as_num()).is_some())
            .collect();
        assert_eq!(estimated.len(), 1, "{angles:?}");
        let (name, params) = estimated[0];
        // Named by its types alone; the estimate says so in its params.
        assert_eq!(name, "c-cc-na");
        assert_eq!(params.get("k").and_then(|v| v.as_num()), Some(68.7));
        assert_eq!(params.get("theta0").and_then(|v| v.as_num()), Some(123.27));
        assert_eq!(
            params.get("estimate_method").and_then(|v| v.as_text()),
            Some("analogy")
        );
        assert_eq!(
            params.get("estimate_analog").and_then(|v| v.as_text()),
            Some("c2-cc-na")
        );
        let penalty = params
            .get("estimate_penalty")
            .and_then(|v| v.as_num())
            .unwrap();
        assert!((penalty - 2.6).abs() < 1e-12, "{penalty}");
    }

    // -- the candidate library and its estimator ---------------------------------

    /// AMBER writes `E = K(r−r₀)²`; molrs's kernel writes `E = ½k(r−r₀)²`. The
    /// doubling belongs at this boundary, so a candidate row and an estimate
    /// drawn from it reach every consumer in one convention.
    #[test]
    fn bond_params_keep_ambers_force_constant() {
        let row = &GaffParameterSet::Gaff.table().bonds[0];
        let params = bond_params(row);
        assert_eq!(params[0].0, "k", "the canonical key is `k`, not `k0`");
        assert_eq!(params[0].1, row.force_constant);
        assert!((params[1].1 - row.length).abs() < 1e-12);
    }

    /// Both shipped `parm` tables (GAFF, GAFF2) transcribe into a candidate
    /// force field without a conflict. `candidate_forcefield` `expect`s this
    /// result and names this test.
    #[test]
    fn candidate_forcefield_defines_without_conflict() {
        for set in [GaffParameterSet::Gaff, GaffParameterSet::Gaff2] {
            assert_eq!(
                try_candidate_forcefield(set.table()).err(),
                None,
                "{}",
                set.name()
            );
        }
    }

    #[test]
    fn angle_params_keep_the_constant_and_the_degrees() {
        let row = &GaffParameterSet::Gaff.table().angles[0];
        let params = angle_params(row);
        assert_eq!(params[0].0, "k");
        assert_eq!(params[0].1, row.force_constant);
        assert_eq!(params[1].1, row.angle_deg);
    }

    // -- impropers and torsions as AmberTools builds them ----------------------

    /// A molecule of `(element, type)` atoms and `bonds` (in this order).
    fn molecule(atoms: &[(&str, &str)], bonds: &[(usize, usize)]) -> Atomistic {
        let mut mol = Atomistic::new();
        let ids: Vec<_> = atoms
            .iter()
            .map(|&(el, ty)| {
                let id = mol
                    .add_atom(props! { keys::ATOMIC_NUMBER => Element::by_symbol(el).unwrap() })
                    .unwrap();
                mol.set_atom(id, keys::TYPE, ty).expect("stamp type");
                id
            })
            .collect();
        for &(a, b) in bonds {
            mol.add_bond(ids[a], ids[b], props! {}).expect("bond");
        }
        mol
    }

    /// The params of every `category` type of `ff`, by name.
    fn params_of(ff: &ForceField, category: &str) -> BTreeMap<String, crate::ff::ir::Params> {
        ff.get_styles(category)
            .into_iter()
            .flat_map(|s| s.defs().collect_type_params())
            .collect()
    }

    /// Ethylene's two sp2 carbons: parmchk2 (AmberTools 26.1) writes
    /// `c2-ha-c2-ha 1.1 Same as X -X -ca-ha, penalty score= 47.1` under GAFF and
    /// `c2-ha-c2-ha 10.5 Same as X -X -cc-X , penalty score= 40.3` under GAFF2,
    /// whose 2.2.30 table added the `X -X -cc-X` row; tleap gives each carbon
    /// that improper.
    #[test]
    fn ethylene_impropers_are_parmchk2s_estimates() {
        let ethylene = molecule(
            &[
                ("C", "c2"),
                ("C", "c2"),
                ("H", "ha"),
                ("H", "ha"),
                ("H", "ha"),
                ("H", "ha"),
            ],
            &[(0, 1), (0, 2), (0, 3), (1, 4), (1, 5)],
        );
        for (set, k, analog, penalty) in [
            (GaffParameterSet::Gaff, 1.1, "X-X-ca-ha", 47.1),
            (GaffParameterSet::Gaff2, 10.5, "X-X-cc-X", 40.3),
        ] {
            let mut typing = Typing::new(GaffTypifier::new(set));
            let typed = typing.typify(&ethylene).expect("ethylene types");
            assert_eq!(typed.impropers().count(), 2, "{}", set.name());
            let impropers = params_of(typing.forcefield(), "improper");
            assert_eq!(impropers.len(), 1, "{}: {impropers:?}", set.name());
            let params = impropers.values().next().unwrap();
            assert_eq!(
                params.get("k").and_then(|v| v.as_num()),
                Some(k),
                "{}",
                set.name()
            );
            assert_eq!(
                params.get("estimate_analog").and_then(|v| v.as_text()),
                Some(analog)
            );
            let got = params
                .get("estimate_penalty")
                .and_then(|v| v.as_num())
                .unwrap();
            assert!((got - penalty).abs() < 1e-9, "{}: {got}", set.name());
        }
    }

    /// `gaff.dat` writes `ho-oh-c3-c3`. Perception stores it with `j < k`, which
    /// is that spelling when the atoms are added in that order, and the reverse
    /// spelling when they are added as `c3-c3-oh-ho`. Both are stored as
    /// `c3-c3-oh-ho`: the type name's order, which is the order its phase
    /// belongs to.
    #[test]
    fn a_proper_is_stored_in_its_type_name_order() {
        let reversed = molecule(
            &[("H", "ho"), ("O", "oh"), ("C", "c3"), ("C", "c3")],
            &[(0, 1), (1, 2), (2, 3)],
        );
        let forward = molecule(
            &[("C", "c3"), ("C", "c3"), ("O", "oh"), ("H", "ho")],
            &[(0, 1), (1, 2), (2, 3)],
        );
        for mol in [reversed, forward] {
            let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
            let typed = typing
                .typify(&mol)
                .expect("ho-oh-c3-c3 is a gaff.dat torsion");
            let dihedrals: Vec<_> = typed.dihedrals().collect();
            assert_eq!(dihedrals.len(), 1);
            let labels: Vec<String> = dihedrals[0]
                .1
                .nodes
                .iter()
                .map(|&id| {
                    typed
                        .get_atom(id)
                        .unwrap()
                        .get_str(keys::TYPE)
                        .unwrap()
                        .to_owned()
                })
                .collect();
            assert_eq!(labels, ["c3", "c3", "oh", "ho"]);
        }
    }

    /// A covering row written the other way from the stored quartet negates a
    /// phase that is not a half-turn. `180` stays `180`, so a merge with a
    /// table that spells the half-turn positive still compares equal.
    #[test]
    fn dihedral_params_for_negates_a_phase_onto_the_stored_order() {
        let row = crate::ff::params::ParmDihedralRow {
            i: None,
            j: None,
            k: None,
            l: None,
            divisor: 1,
            barrier: 1.0,
            phase_deg: 60.0,
            periodicity: 1,
            more_terms: false,
        };
        let stored = ["c3", "c3", "oh", "ho"].map(str::to_owned);
        let phase = |row: &crate::ff::params::ParmDihedralRow| {
            super::dihedral_params_for(&[row], &["ho", "oh", "c3", "c3"], &stored)
                .into_iter()
                .find(|(key, _)| key == "phase1")
                .unwrap()
                .1
        };
        assert_eq!(phase(&row), -60.0);
        assert_eq!(
            phase(&crate::ff::params::ParmDihedralRow {
                phase_deg: 180.0,
                ..row
            }),
            180.0
        );
    }

    /// Guanidinium: parmchk2 writes `nh-cz-nh-hn 4 2.700 180.000 2.000 same as
    /// X -c2-nh-X , penalty score=462.5` — `cz` stands in for the `c2` of a
    /// wildcard row through `cz`'s own `CORR c2` line.
    #[test]
    fn guanidinium_torsion_is_parmchk2s_corresponding_wildcard_row() {
        let guanidinium = molecule(
            &[
                ("C", "cz"),
                ("N", "nh"),
                ("N", "nh"),
                ("N", "nh"),
                ("H", "hn"),
                ("H", "hn"),
                ("H", "hn"),
                ("H", "hn"),
                ("H", "hn"),
                ("H", "hn"),
            ],
            &[
                (0, 1),
                (0, 2),
                (0, 3),
                (1, 4),
                (1, 5),
                (2, 6),
                (2, 7),
                (3, 8),
                (3, 9),
            ],
        );
        let mut typing = Typing::new(GaffTypifier::new(GaffParameterSet::Gaff));
        typing.typify(&guanidinium).expect("guanidinium types");
        let dihedrals = params_of(typing.forcefield(), "dihedral");
        let estimated: Vec<_> = dihedrals
            .iter()
            .filter(|(_, p)| p.get("estimated").and_then(|v| v.as_num()).is_some())
            .collect();
        assert_eq!(estimated.len(), 1, "{dihedrals:?}");
        let (name, params) = estimated[0];
        assert_eq!(name, "hn-nh-cz-nh");
        assert_eq!(
            params.get("estimate_analog").and_then(|v| v.as_text()),
            Some("X-c2-nh-X")
        );
        assert_eq!(
            params.get("estimate_penalty").and_then(|v| v.as_num()),
            Some(462.5)
        );
        assert_eq!(params.get("k1").and_then(|v| v.as_num()), Some(2.7 / 4.0));
        assert_eq!(
            params.get("periodicity1").and_then(|v| v.as_num()),
            Some(2.0)
        );
        assert_eq!(params.get("phase1").and_then(|v| v.as_num()), Some(180.0));
    }
}
