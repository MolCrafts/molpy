//! Every force-field parameter table molrs ships, as typed, compile-time Rust
//! data — **one place, one form**.
//!
//! molrs parses **no** parameter text at runtime. The upstream `.DAT` / `.DEF`
//! tables are transcribed into the `const`s in the sibling modules here by
//! `scripts/gen_param_tables.py`, which reads them from `$AMBERHOME` (the
//! committed fourteen are AmberTools 26.1's; `--check` verifies them byte for
//! byte); [`mmff`]
//! is ported from RDKit's `Params.cpp` and merged with what MMFF's XML
//! parameter file carries; the OPLS-AA tables ([`OPLSAA_ATOMS`], …) are generated from GROMACS `share/top/oplsaa.ff` (a
//! pinned release, LGPL-2.1-or-later) by `cargo mrs-gen-opls --gromacs <dir>`,
//! through molrs's own GROMACS reader, while the OPLS-AA SMARTS typing rules in
//! [`OPLSAA_TYPING`] are molrs-owned and hand-maintained — GROMACS has none,
//! and the generator never touches them. The AMBER 1-4 divisors, which are
//! not `gaff.dat` rows, are engine constants (`core::constants::AMBER_SCEE`,
//! `AMBER_SCNB`). The committed `.rs` is the
//! single in-repo source of truth; a malformed table is therefore a **compile**
//! error, not a runtime one, and the tables can be grepped, diffed and stepped
//! through like any other code.
//!
//! This module owns the row types of the tables it emits, and the sibling
//! modules own only data. (The RDKit port keeps its own row structs next to the
//! rows they describe — it is a transcription of one upstream file and is moved
//! verbatim, not re-shaped.) They are all ordinary source files, not build
//! artefacts: how a table *arrived* belongs in its header doc, never in its name.
//!
//! # The ATD / WILDATOM grammar
//!
//! All seven `ATOMTYPE_*.DEF` files share one grammar. An [`AtdRule`] is a
//! conjunction of constraints on a candidate atom — atomic number, degree,
//! attached-hydrogen count, an [`AtomProp`] expression, and a neighbourhood
//! [`AtomPattern`] tree. Rules are tried **in file order** and the first match
//! wins.
//!
//! The environment and property mini-languages are **pre-parsed** into the
//! static AST below, so the typifier walks a tree instead of re-parsing a
//! string for every atom and every candidate rule.
//!
//! # The `parm` force fields
//!
//! [`GAFF`] and [`GAFF2`] are the same idea applied to AMBER's `parm` format:
//! `gaff.dat` and `gaff2.dat` become [`ParmTable`]s of MASS / BOND / ANGLE /
//! DIHE / IMPROPER / NONBON rows. Values are kept in the **upstream's own units
//! and conventions** — degrees, and AMBER's un-halved force constants — because
//! the table is a transcription of the file, not a force field. That is also
//! what the force-field IR (adopts the LAMMPS standard) holds for every
//! bonded term; what the code that populates a
//! [`ForceField`](crate::ff::forcefield::ForceField) from it still converts
//! (`IDIVF`, R\*/2 → σ) is in [`crate::ff::typifier::GaffTypifier`].

pub mod atomtype_abcg2;
pub mod atomtype_amber;
pub mod atomtype_bcc;
pub mod atomtype_gas;
pub mod atomtype_gff;
pub mod atomtype_gff2;
pub mod atomtype_sybyl;
mod bccparm;
mod bccparm_abcg2;
mod clpol;
mod gaff;
mod gaff2;
mod gaff_empirical;
mod gasparm;
pub mod mmff;
mod oplsaa;
mod oplsaa_typing;
mod parmchk;
pub mod uff;

pub use bccparm::{BCC_ALIASES, BCC_CORRECTIONS};
pub use bccparm_abcg2::{ABCG2_ALIASES, ABCG2_CORRECTIONS};
pub use clpol::{CLPOL_FRAGMENTS, CLPOL_POLARIZABILITY, ClpolPolarizability, clpol_polarizability};
pub use gaff::GAFF;
pub use gaff_empirical::{EMPIRICAL_GAFF, EMPIRICAL_GAFF2};
pub use gaff2::GAFF2;
pub use gasparm::GASTEIGER_PARAMS;
pub use oplsaa::{
    OPLSAA_ANGLES, OPLSAA_ATOMS, OPLSAA_BONDS, OPLSAA_DIHEDRALS, OPLSAA_MIXING, OPLSAA_NAME,
};
pub use oplsaa_typing::OPLSAA_TYPING;
pub use parmchk::{PARMCHK, PARMCHK_TYPES, PARMCHK_WEIGHTS};

/// One oriented bond charge correction from a `BCCPARM*.DAT` table.
///
/// The bond `left|right|bond_type` adds `+delta` to `left` and `-delta` to
/// `right`; the same bond seen in the reverse orientation applies `-delta` /
/// `+delta`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BccCorrectionRow {
    /// BCC atom type on the positive side of the correction.
    pub left: &'static str,
    /// BCC atom type on the negative side of the correction.
    pub right: &'static str,
    /// BCC bond type (1 single, 2 double, 3 triple, 7/8/10 aromatic, 9 delocalized).
    pub bond_type: i32,
    /// Charge transferred from `right` to `left`, in elementary charge units.
    pub delta: f64,
}

/// A `CORR` row: `atom_type` borrows the corrections of `reference`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BccAlias {
    /// The atom type that carries no corrections of its own.
    pub atom_type: &'static str,
    /// The atom type whose corrections it borrows.
    pub reference: &'static str,
}

/// One row of `GASPARM.DAT` — the Gasteiger–Marsili PEOE parameters.
///
/// `a`, `b` and `c` are the electronegativity polynomial coefficients:
/// `chi(q) = a + b*q + c*q^2`. `chi_plus` is the normalisation denominator
/// (the cation electronegativity), **not** a quartic coefficient — the
/// upstream file calls that column `d`. `seed_charge` is the `formal_charge`
/// column, the initial charge q0 the iteration starts from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GasteigerRow {
    /// Gasteiger atom type (`h`, `c3`, `o-1`, `na+`, …).
    pub atom_type: &'static str,
    /// Constant term of the electronegativity polynomial.
    pub a: f64,
    /// Linear coefficient of the electronegativity polynomial.
    pub b: f64,
    /// Quadratic coefficient of the electronegativity polynomial.
    pub c: f64,
    /// Cation electronegativity chi+, the damping denominator (column `d`).
    pub chi_plus: f64,
    /// Seed charge q0 (column `formal_charge`).
    pub seed_charge: f64,
}

/// One alternative of a `WILDATOM` alias: an element, optionally at a fixed degree.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildAtomSpec {
    /// Atomic number.
    pub z: u8,
    /// Required connectivity, or `None` when the alias accepts any degree.
    pub degree: Option<usize>,
}

/// A `WILDATOM` alias — a named set of [`WildAtomSpec`] alternatives.
///
/// The set is file-local: `XB` is `C3 N2 N3 O2 S2 P2` in `ATOMTYPE_BCC.DEF`
/// but `N P` in `ATOMTYPE_GFF.DEF`. Pattern names are resolved against the
/// owning file at generation time, so this table is documentation rather than
/// a runtime lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WildAtom {
    /// Alias name as it appears in the `.DEF` (`XX`, `XA`, …).
    pub name: &'static str,
    /// The element alternatives it stands for.
    pub specs: &'static [WildAtomSpec],
}

/// A topological property an atom can be counted against.
///
/// The lowercase source tokens (`sb` / `db` / `tb`) count aromatic and
/// delocalized bonds as well; the uppercase ones (`SB` / `DB` / `TB`) are
/// strict. That distinction is load-bearing, so the two spellings map to
/// distinct variants rather than to one case-folded name.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomProp {
    /// `RG` — member of any ring.
    Rg,
    /// `RG3` — member of a 3-membered ring.
    Rg3,
    /// `RG4` — member of a 4-membered ring.
    Rg4,
    /// `RG5` — member of a 5-membered ring.
    Rg5,
    /// `RG6` — member of a 6-membered ring.
    Rg6,
    /// `RG7` — member of a 7-membered ring.
    Rg7,
    /// `RG8` — member of an 8-membered ring.
    Rg8,
    /// `RG9` — member of a 9-membered ring.
    Rg9,
    /// `RG10` — member of a 10-membered ring.
    Rg10,
    /// `NR` — in no ring at all.
    Nr,
    /// `AR1` — pure aromatic ring (benzene, pyridine).
    Ar1,
    /// `AR2` — planar ring with two continuous single bonds and at least two
    /// double bonds (imidazole, thiophene, pyrrole).
    Ar2,
    /// `AR3` — planar ring whose double bonds are formed between ring atoms and
    /// non-ring atoms (a quinone, a pyridone).
    Ar3,
    /// `AR4` — a ring that is none of `AR1`, `AR2`, `AR3` or `AR5`.
    Ar4,
    /// `AR5` — pure aliphatic ring, made of sp3 carbon.
    Ar5,
    /// `SB` — single bonds, strictly.
    SbStrict,
    /// `sb` — single bonds, counting aromatic and delocalized ones.
    SbAny,
    /// `DB` — double bonds, strictly.
    DbStrict,
    /// `db` — double bonds, counting aromatic and delocalized ones.
    DbAny,
    /// `TB` — triple bonds, strictly.
    TbStrict,
    /// `tb` — triple bonds, counting aromatic ones.
    TbAny,
    /// `AB` — aromatic bonds.
    Ab,
    /// `DL` — delocalized bonds.
    Dl,
}

/// The `'` / `''` suffix on a property unit.
///
/// It constrains the bond back to the atom the pattern arrived from, so it is
/// only meaningful inside an [`AtomPattern`], never on a rule's top-level
/// property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropRelation {
    /// `'` — the bond to the previous atom must itself be of this type.
    BondedToPrev,
    /// `''` — the bond to the previous atom must NOT be of this type.
    NotBondedToPrev,
}

/// One `[count]PROP['|'']` unit, e.g. `2DL`, `AR1`, `db'`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropUnit {
    /// Exact count required; `None` means "at least one".
    pub count: Option<usize>,
    /// The property being counted.
    pub prop: AtomProp,
    /// Optional constraint on the bond back to the previous atom.
    pub relation: Option<PropRelation>,
}

/// A comma-group of a property expression: satisfied when **any** unit matches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropConstraint {
    /// Dot-separated alternatives (`AR1.AR2` = AR1 **or** AR2).
    pub units: &'static [PropUnit],
}

/// A pre-parsed `[...]` atom-property expression: an AND of ORs.
///
/// `[RG5.RG6,AR1.AR2.AR3]` is "(5- or 6-ring) **and** (AR1, AR2 or AR3)".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PropExpr {
    /// Comma-separated constraints, **all** of which must hold.
    pub constraints: &'static [PropConstraint],
}

/// What an [`AtomPattern`] matches, resolved at generation time.
///
/// The `.DEF` files name pattern atoms with a bare symbol; that name is looked
/// up as `EW`, then as a `WILDATOM` of the owning file, then as an element.
/// Doing that here rather than at match time is what lets the typifier run
/// without a name table — and makes an unknown name a generator error instead
/// of a rule that silently never fires.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PatternAtom {
    /// `EW` — any electron-withdrawing atom (N, O, F, S, Cl, Br, I).
    ElectronWithdrawing,
    /// A `WILDATOM` alias, already expanded to its alternatives.
    Wild(&'static [WildAtomSpec]),
    /// A concrete element, by atomic number.
    Element(u8),
}

/// One node of a pre-parsed environment expression, e.g. `C3[AR1](O1,O1)`.
///
/// `children` are neighbours of *this* atom, excluding the atom the match
/// arrived from — so the tree is walked outward, never back on itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtomPattern {
    /// Which atoms this node can match.
    pub atom: PatternAtom,
    /// Required connectivity, or `None` for any.
    pub degree: Option<usize>,
    /// Optional `[...]` property expression.
    pub property: Option<PropExpr>,
    /// Optional `<label>`, referenced by an [`EnvBond`].
    pub label: Option<&'static str>,
    /// Neighbour patterns that must match, one atom each.
    pub children: &'static [AtomPattern],
}

/// A bond type an [`EnvBond`] can require.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EnvBondType {
    /// `any` — the two atoms need only be bonded.
    Any,
    /// A single bond (aromatic and delocalized single bonds count).
    Single,
    /// A double bond (aromatic and delocalized double bonds count).
    Double,
    /// A triple bond.
    Triple,
    /// An aromatic bond.
    Aromatic,
}

/// An `a:b:TYPE` constraint between two `<label>`ed atoms of an environment.
///
/// This is how the `.DEF` closes a ring: it demands that two atoms the pattern
/// reached by different branches are themselves bonded.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnvBond {
    /// Label of the first atom.
    pub a: &'static str,
    /// Label of the second atom.
    pub b: &'static str,
    /// The bond type required between them.
    pub bond: EnvBondType,
}

/// The phase-2 partner of a rule's atom type, and the colouring pass that
/// renames to it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Alternate {
    /// The phase-2 name (`cd`, `cf`, `cq`, …).
    pub atom_type: &'static str,
    /// Which of antechamber's two colouring passes owns the pair.
    pub pass: AlternatePass,
}

/// antechamber's two post-typing colouring passes (`atomtype.c`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AlternatePass {
    /// `atadjust`: the conjugated chain and ring names `cc` / `ce` / `cg` /
    /// `nc` / `ne` / `pc` / `pe`, coloured together.
    Conjugated,
    /// `cpadjust`: the biphenyl bridge carbons `cp`, coloured on their own.
    Bridge,
}

/// One `ATD` row: a conjunction of constraints on a candidate atom.
///
/// Every `Option` field is an unconstrained `*` / `&` column in the source.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtdRule {
    /// The atom type assigned when the rule matches.
    pub atom_type: &'static str,
    /// The conjugated partner of [`atom_type`](Self::atom_type), when it has one.
    ///
    /// An `ATOMTYPE_GFF*.DEF` row only ever emits the **phase-1** name of a
    /// conjugated system — `cc`, `ce`, `cg`, `nc`, `ne`, `pc`, `pe`. antechamber
    /// then 2-colours each conjugated system and renames one colour to its
    /// partner (`cd`, `cf`, `ch`, `nd`, `nf`, `pd`, `pf`), names that appear in
    /// no `.DEF` row at all. That pairing is upstream **data**, not an engine
    /// invention: `PARMCHK.DAT` declares it in a per-type `equivalent_flag`
    /// column (`1` for the phase-1 names, `2` for the phase-2 names, `0` for
    /// everything else). Hence a column of the rule rather than a `match` in the
    /// typifier.
    ///
    /// The biphenyl bridge carbon `cp` pairs the same way with `cq` (`-1` / `-2`
    /// in the column, though the header does not list them), and antechamber
    /// colours those in a pass of their own — hence [`Alternate::pass`].
    ///
    /// `None` whenever that column says `0` — including `ATOMTYPE_AMBER.DEF`'s
    /// `CC` / `CD`, which are parm94's histidine carbons and **not** a conjugated
    /// pair despite the spelling.
    pub alternate: Option<Alternate>,
    /// Residue name, or `*` for any.
    pub residue: &'static str,
    /// Required atomic number.
    pub atomic_number: Option<u8>,
    /// Required connectivity.
    pub degree: Option<usize>,
    /// Required number of attached hydrogens.
    pub hydrogen_count: Option<usize>,
    /// Required number of electron-withdrawing atoms around the attachment point.
    pub ewd_count: Option<usize>,
    /// Required `[...]` properties of the atom itself.
    pub atom_property: Option<PropExpr>,
    /// Required neighbourhood, as a pre-parsed pattern forest.
    pub environment: Option<&'static [AtomPattern]>,
    /// Required bonds between `<label>`ed environment atoms.
    pub environment_bonds: Option<&'static [EnvBond]>,
}

/// One `ATOMTYPE_*.DEF` file as a typed table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AtdTable {
    /// Upstream file name, e.g. `ATOMTYPE_BCC.DEF`.
    pub name: &'static str,
    /// The `WILDATOM` aliases the file declares.
    pub wildatoms: &'static [WildAtom],
    /// The `ATD` rules, in file order. The first match wins.
    pub rules: &'static [AtdRule],
}

// ---------------------------------------------------------------------------
// AMBER `parm` force fields — gaff.dat / gaff2.dat
// ---------------------------------------------------------------------------

/// An atom type of a [`ParmTable`], as an index into its [`masses`](ParmTable::masses).
///
/// **Why an index and not a `&'static str`.** The two `parm` tables hold about
/// 60,000 atom-type slots between them, and a `&str` is a 16-byte fat pointer
/// that also needs a load-time relocation. Spelling every slot out cost **1416
/// KB** of stripped release binary, measured; interning to one byte costs a
/// fraction of that, and the table stays just as readable because each type gets
/// a named `const` (`T_C3`), never a bare number.
///
/// The index is the type's row in the `MASS` section — the section that
/// *declares* the atom types — so [`ParmTable::name_of`] is an array read, and a
/// type with no `MASS` row is a generator error rather than a dangling index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ParmType(pub u8);

/// One `MASS` row of an AMBER `parm` table.
///
/// The row order of this section **is** the [`ParmType`] numbering.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmMassRow {
    /// GAFF atom type (`c3`, `ha`, `os`, …).
    pub atom_type: &'static str,
    /// Atomic mass, in amu.
    pub mass: f64,
    /// Atomic polarizability, in Å³. Read by the polarizable force fields only;
    /// a fixed-charge GAFF run never touches it.
    pub polarizability: f64,
}

/// One `BOND` row: `E = force_constant · (r − length)²`.
///
/// That is **AMBER's** convention and it carries no ½ — unlike molrs's
/// [`BondHarmonic`](crate::ff::potential::bond::BondHarmonic), whose
/// `k` is `2 · force_constant`. The factor is applied where the units are
/// normalised (the [`GaffTypifier`](crate::ff::typifier::GaffTypifier) candidate library), never here:
/// this row is what the file says.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmBondRow {
    /// Atom type of the first endpoint.
    pub i: ParmType,
    /// Atom type of the second endpoint.
    pub j: ParmType,
    /// Force constant, in kcal/mol/Å².
    pub force_constant: f64,
    /// Equilibrium bond length, in Å.
    pub length: f64,
}

/// One `ANGLE` row: `E = force_constant · (θ − angle_deg)²`, the force
/// constant per radian².
///
/// AMBER's un-halved convention, with the equilibrium angle in the **degrees**
/// the file writes it in — which is LAMMPS's `angle_style harmonic`, and so
/// molrs's, exactly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmAngleRow {
    /// Atom type of the first leg.
    pub i: ParmType,
    /// Atom type of the vertex.
    pub j: ParmType,
    /// Atom type of the second leg.
    pub k: ParmType,
    /// Force constant, in kcal/mol/rad².
    pub force_constant: f64,
    /// Equilibrium angle, in **degrees**.
    pub angle_deg: f64,
}

/// One `DIHE` row — a single cosine term of a proper torsion:
///
/// ```text
/// E = (barrier / divisor) · [1 + cos(periodicity · φ − phase_deg)]
/// ```
///
/// A slot is `None` where the file writes the `X` wildcard, which matches any
/// atom type. Modelling that as `Option` rather than as a distinguished index is
/// what makes a wildcard impossible to read as a concrete type by accident:
/// there are 607 such rows per table, and the exact-match population path has to
/// skip every one of them while the parmchk2-style fallback lives on them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmDihedralRow {
    /// Atom type of the first atom, or `None` for the `X` wildcard.
    pub i: Option<ParmType>,
    /// Atom type of the second (bonded-pair) atom, or `None`.
    pub j: Option<ParmType>,
    /// Atom type of the third (bonded-pair) atom, or `None`.
    pub k: Option<ParmType>,
    /// Atom type of the fourth atom, or `None`.
    pub l: Option<ParmType>,
    /// `IDIVF` — the number of torsions sharing this barrier. The per-torsion
    /// force constant is `barrier / divisor`.
    pub divisor: u32,
    /// `PK` — the **total** barrier height, in kcal/mol, before division.
    pub barrier: f64,
    /// Phase, in **degrees**.
    pub phase_deg: f64,
    /// `|PN|` — the multiplicity.
    pub periodicity: u32,
    /// Upstream wrote `PN` negative: another cosine term for the *same* quartet
    /// follows on the next row. The sign is a structural fact about the file
    /// rather than a number, so it becomes a flag of its own and `periodicity`
    /// stays a magnitude no kernel can be handed a negative multiplicity from.
    pub more_terms: bool,
}

/// One `IMPROPER` row: `E = barrier · [1 + cos(periodicity · φ − phase_deg)]`.
///
/// **The central atom is [`k`](Self::k) — the third** — which is AMBER's
/// convention, not molrs's (`Atomistic::add_improper` documents `i` as
/// conventionally central, and MMFF's Wilson out-of-plane puts it second). φ is
/// the I-J-K-L dihedral, so the slot order is load-bearing: an improper relation
/// built from this row must be written in it.
///
/// There is no `divisor`: an improper's barrier is never divided.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmImproperRow {
    /// Atom type of the first peripheral atom, or `None` for the `X` wildcard.
    pub i: Option<ParmType>,
    /// Atom type of the second peripheral atom, or `None`.
    pub j: Option<ParmType>,
    /// Atom type of the **central** atom, or `None`.
    pub k: Option<ParmType>,
    /// Atom type of the third peripheral atom, or `None`.
    pub l: Option<ParmType>,
    /// Barrier height, in kcal/mol.
    pub barrier: f64,
    /// Phase, in **degrees** (180 for every GAFF improper).
    pub phase_deg: f64,
    /// Multiplicity (2 for every GAFF improper).
    pub periodicity: u32,
}

/// One `NONBON` row, under the `MOD4  RE` label: R\* and ε.
///
/// `RE` is what declares these two columns to be a radius and a well depth;
/// under an `AC` label they would be the A/C coefficients of the same 12-6 form
/// instead. The generator refuses any other label rather than reinterpret them.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmNonbondedRow {
    /// The atom type these parameters belong to.
    pub atom_type: ParmType,
    /// R\* — **half** the LJ minimum-energy separation, in Å. Not σ:
    /// `σ = 2·R* / 2^(1/6)`.
    pub r_min_half: f64,
    /// Well depth ε, in kcal/mol.
    pub epsilon: f64,
}

/// One AMBER `parm` force field (`gaff.dat`, `gaff2.dat`) as a typed table.
///
/// Rows keep file order in every section, which is load-bearing twice over: for
/// `dihedrals`, consecutive rows sharing a quartet are the cosine terms of one
/// torsion; for `impropers`, the first matching row wins.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmTable {
    /// Upstream file name, e.g. `gaff.dat`.
    pub name: &'static str,
    /// The `MASS` section. Its row order is the [`ParmType`] numbering.
    pub masses: &'static [ParmMassRow],
    /// The `BOND` section.
    pub bonds: &'static [ParmBondRow],
    /// The `ANGLE` section.
    pub angles: &'static [ParmAngleRow],
    /// The `DIHE` section.
    pub dihedrals: &'static [ParmDihedralRow],
    /// The `IMPROPER` section.
    pub impropers: &'static [ParmImproperRow],
    /// The `NONBON` section.
    pub nonbonded: &'static [ParmNonbondedRow],
}

// ---------------------------------------------------------------------------
// parmchk2's substitution table — PARMCHK.DAT
// ---------------------------------------------------------------------------

/// The nine penalty columns of an `EQUA` / `CORR` row, in `PARMCHK.DAT`'s order.
///
/// The order is the one parmchk2 reads them in (`read_parmchk_parm`:
/// `bl blf cba cbaf ba baf ctor tor ps`): the angle's CENTRE columns come
/// before its END columns, and the torsion's INNER column before its OUTER
/// one. The file's own note pins the last — *"GENERAL_SIMILARITY is listed in
/// the 11th colume of CORR lines"*. `-1.0` means **not tabulated**: parmchk2
/// substitutes the matching `DEFAULT_*` (and a `CORR` line with no columns at
/// all, which this table also writes as `-1.0` throughout, reads as zeros).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParmchkPenalty {
    /// Bond length (`bl`).
    BondLength = 0,
    /// Bond force constant (`blf`).
    BondForce = 1,
    /// Angle, at the CENTRE atom (`cba`).
    AngleCentre = 2,
    /// Angle force constant, at the centre atom (`cbaf`).
    AngleCentreForce = 3,
    /// Angle, at an END atom (`ba`).
    Angle = 4,
    /// Angle force constant, at an end atom (`baf`).
    AngleForce = 5,
    /// Torsion, at an INNER atom (`ctor`).
    TorsionCentre = 6,
    /// Torsion, at an OUTER atom (`tor`).
    Torsion = 7,
    /// The row's overall similarity score — the 11th column (`ps`).
    Similarity = 8,
}

/// One `EQUA` / `CORR` row: the atom type it maps to, and its nine penalties.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmchkCorr {
    /// The atom type this row substitutes.
    pub to: &'static str,
    /// The nine penalty columns, indexed by [`ParmchkPenalty`]. `-1.0` = absent.
    pub penalties: [f64; 9],
}

impl ParmchkCorr {
    /// One penalty column, or `None` where the row leaves it untabulated (`-1`).
    pub fn get(&self, column: ParmchkPenalty) -> Option<f64> {
        let value = self.penalties[column as usize];
        (value >= 0.0).then_some(value)
    }
}

/// One `PARM` block of `PARMCHK.DAT` — an atom type and how it may be substituted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmchkType {
    /// The atom type, e.g. `c3`.
    pub name: &'static str,
    /// The `improper_flag` column: may an atom of this type be the **centre** of
    /// an improper at all?
    ///
    /// This is what separates `ca` (yes: benzene's ring carbon is planar) from
    /// `c3` (no: an sp3 carbon has no planarity to enforce), and it is a column
    /// of the table rather than a hybridisation the engine re-derives.
    pub improper: bool,
    /// The `group_id` column: crossing a group boundary costs
    /// [`ParmchkWeights::weight_group`].
    pub group: i32,
    /// Atomic mass, in amu.
    pub mass: f64,
    /// The `equivalent_flag` column: `±1` for a phase-1 conjugated name
    /// (`cc`/`ce`/…), `±2` for its phase-2 partner (`cd`/`cf`/…), `0` otherwise.
    pub equivalent_flag: i32,
    /// Atomic number.
    pub atomic_number: u8,
    /// The phase-2 name antechamber renames this type to on the other colour of a
    /// conjugated system (`cc` → `cd`), or `None`.
    pub alternate: Option<&'static str>,
    /// `EQUA` rows — **equivalent** types. Substituting one costs nothing.
    pub equivalent: &'static [&'static str],
    /// `CORR` rows — **corresponding** types, each with its penalty columns.
    pub corresponding: &'static [ParmchkCorr],
}

/// `PARMCHK.DAT`'s `WEIGHT_*` and `DEFAULT_*` scalars.
///
/// Every field is a column of the file; none is a constant this crate invented.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmchkWeights {
    /// `WEIGHT_BL` — bond-length penalty weight.
    pub weight_bond_length: f64,
    /// `WEIGHT_BLF` — bond force-constant penalty weight.
    pub weight_bond_force: f64,
    /// `WEIGHT_BA` — angle penalty weight.
    pub weight_angle: f64,
    /// `WEIGHT_BAF` — angle force-constant penalty weight.
    pub weight_angle_force: f64,
    /// `WEIGHT_X` — an `X` in slot 1, 2 or 4 of an improper row.
    pub weight_wildcard: f64,
    /// `WEIGHT_X3` — an `X` in the improper's CENTRE slot.
    pub weight_wildcard_centre: f64,
    /// `WEIGHT_BA_CTR` — inner-atom multiplier for an angle.
    pub weight_angle_centre: f64,
    /// `WEIGHT_TOR_CTR` — inner-atom multiplier for a torsion.
    pub weight_torsion_centre: f64,
    /// `WEIGHT_IMPROPER` — surcharge for substituting into an improper row.
    pub weight_improper: f64,
    /// `WEIGHT_GROUP` — surcharge for crossing an atom-type group boundary.
    pub weight_group: f64,
    /// `WEIGHT_EQUTYPE` — surcharge for substituting an equivalent type.
    pub weight_equivalent: f64,
    /// `DEFAULT_BL`.
    pub default_bond_length: f64,
    /// `DEFAULT_BLF`.
    pub default_bond_force: f64,
    /// `DEFAULT_BA`.
    pub default_angle: f64,
    /// `DEFAULT_BAF`.
    pub default_angle_force: f64,
    /// `DEFAULT_BA_CTR`.
    pub default_angle_centre: f64,
    /// `DEFAULT_BAF_CTR`.
    pub default_angle_centre_force: f64,
    /// `DEFAULT_TOR`.
    pub default_torsion: f64,
    /// `DEFAULT_TOR_CTR`.
    pub default_torsion_centre: f64,
    /// `DEFAULT_FRACT1`.
    pub default_fraction_1: f64,
    /// `DEFAULT_FRACT2`.
    pub default_fraction_2: f64,
    /// `THRESHOLD_BA`.
    pub threshold_angle: f64,
}

/// `PARMCHK.DAT` as one typed table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParmchkTable {
    /// Upstream file name.
    pub name: &'static str,
    /// The `PARM` blocks, in file order.
    pub types: &'static [ParmchkType],
    /// The `WEIGHT_*` / `DEFAULT_*` scalars.
    pub weights: ParmchkWeights,
}

impl ParmchkTable {
    /// The block declaring `name`, or `None` if the table does not know the type.
    pub fn get(&self, name: &str) -> Option<&'static ParmchkType> {
        self.types.iter().find(|t| t.name == name)
    }
}

// ---------------------------------------------------------------------------
// The empirical bond / angle constants — PARM_BLBA_GAFF*.DAT
// ---------------------------------------------------------------------------

/// One `BL` row: the reference length and `ln(Kij)` of an element **pair**.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalBondRow {
    /// Atomic number of the first element.
    pub z1: u8,
    /// Atomic number of the second element.
    pub z2: u8,
    /// Reference bond length, in Å (Wang2004 Table 3 `r_ref`).
    pub r_ref: f64,
    /// `ln(Kij)` (Wang2004 Table 3); the force constant is `exp(ln_k) / r^m`.
    pub ln_k: f64,
}

/// One `BA` row: the angle `C` and `Z` factors of an element.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalAngleRow {
    /// Atomic number.
    pub z: u8,
    /// The `C` factor — used when the element is the angle's CENTRE.
    pub c: f64,
    /// The `Z` factor — used when the element is an angle END atom.
    pub z_factor: f64,
}

/// One `PARM_BLBA_GAFF*.DAT` as a typed table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EmpiricalTable {
    /// Upstream file name.
    pub name: &'static str,
    /// The Badger exponent `m` (`PARM PC`; 4.5 upstream).
    pub bond_power: f64,
    /// The `BL` rows.
    pub bonds: &'static [EmpiricalBondRow],
    /// The `BA` rows.
    pub angles: &'static [EmpiricalAngleRow],
}

impl EmpiricalTable {
    /// The `BL` row for an unordered element pair, if tabulated.
    pub fn bond(&self, z1: u8, z2: u8) -> Option<&'static EmpiricalBondRow> {
        self.bonds
            .iter()
            .find(|r| (r.z1 == z1 && r.z2 == z2) || (r.z1 == z2 && r.z2 == z1))
    }

    /// The `BA` row for an element, if tabulated.
    pub fn angle(&self, z: u8) -> Option<&'static EmpiricalAngleRow> {
        self.angles.iter().find(|r| r.z == z)
    }
}

impl ParmTable {
    /// The name of an interned atom type, e.g. `c3`.
    ///
    /// # Panics
    ///
    /// If `ty` is not a type of this table. Every [`ParmType`] in a table's rows
    /// indexes that table's own `MASS` section by construction, so this can only
    /// fire if one table's type is used against another's.
    pub fn name_of(&self, ty: ParmType) -> &'static str {
        self.masses[usize::from(ty.0)].atom_type
    }

    /// The interned form of a type name, or `None` if this table has no such type.
    ///
    /// Linear over the `MASS` section (83 / 99 rows). Callers that resolve many
    /// labels should build a map once instead of calling this per atom.
    pub fn type_of(&self, name: &str) -> Option<ParmType> {
        self.masses
            .iter()
            .position(|row| row.atom_type == name)
            .map(|idx| ParmType(idx as u8))
    }
}

// ---------------------------------------------------------------------------
// OPLS-AA
// ---------------------------------------------------------------------------
//
// The rows of [`oplsaa`], in **molrs's convention** (LAMMPS `real`: Å,
// kcal/mol, degrees, e; un-halved `K`) — what the kernels read. GROMACS's `oplsaa.ff` speaks nm, kJ/mol and
// Ryckaert–Bellemans torsions; the conversion happens once, in the generator
// (through `GromacsTopForcefieldReader`), and its result is what is committed. The two
// vocabularies of the source survive intact: bonded rows key on the GROMACS
// `bond_type` (the **class**, `CT`, `HC`), atoms and pairs on the **type**
// (`opls_NNN`). The typing rules are not GROMACS's and live apart, in
// [`oplsaa_typing`].

/// One OPLS-AA atom type: its potential parameters, as GROMACS
/// `ffnonbonded.itp` `[ atomtypes ]` states them.
///
/// Parameters only. Whether and how an atom is *typed* as this type is the
/// molrs-owned [`OplsRuleRow`] of the same `name`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OplsAtomRow {
    /// Type name (`opls_135`) — the nonbonded / atom-style key.
    pub name: &'static str,
    /// Chemical class — the GROMACS `bond_type` column (`CT`), and the key of
    /// every bonded table. A row without a `bond_type` column is its own class.
    pub class: &'static str,
    /// Atomic mass (amu).
    pub mass: f64,
    /// Partial charge (e).
    pub charge: f64,
    /// Lennard-Jones σ (Å); `0` for the ε = 0 types, as GROMACS writes it.
    pub sigma: f64,
    /// Lennard-Jones ε (kcal/mol).
    pub epsilon: f64,
}

/// One OPLS-AA typing rule — the **static, compile-time** rule record.
///
/// Its runtime counterpart is
/// [`OplsTypeRow`](crate::ff::typifier::OplsTypeRow): owned strings, plus
/// the class, an explicit priority and an overlay layer. The embedded OPLS-AA
/// typifier builds one `OplsTypeRow` from each `OplsRuleRow`, taking the class
/// from the [`OplsAtomRow`] of the same `name` and layer 0; an XML force field
/// builds them from its `<Type>` attributes instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OplsRuleRow {
    /// Type name (`opls_135`) the rule assigns; names an [`OplsAtomRow`].
    pub name: &'static str,
    /// Daylight SMARTS for the typed atom (query atom 0), read with standard
    /// semantics against a graph with **explicit hydrogens** whose aromaticity
    /// the typifier perceives first: every bond written (`-`, `=`, `#`, `:`),
    /// hydrogen atoms as `[#1]`, `H<n>` a hydrogen count, aromatic atoms
    /// lowercase, `r<n>` the smallest ring. A `%opls_NNN` predicate (a molrs
    /// extension) reads the type already assigned to that neighbour.
    pub def: &'static str,
    /// Type names this one dominates when both match on the same layer —
    /// directly, and through their own overrides transitively; each names
    /// another rule.
    pub overrides: &'static [&'static str],
}

/// One GROMACS `[ bondtypes ]` funct-1 row, `½k_b(r − r₀)²` in the file, as LAMMPS `K = k_b/2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OplsBondRow {
    /// Class of the first atom.
    pub i: &'static str,
    /// Class of the second atom.
    pub j: &'static str,
    /// Force constant (kcal/mol/Å²) in molrs's — LAMMPS's — `E = k(r−r0)²`
    /// convention (GROMACS's `k_b / 2`); emitted under the param key `k`.
    pub force_constant: f64,
    /// Equilibrium length (Å).
    pub r0: f64,
}

/// One GROMACS `[ angletypes ]` funct-1 row, `½k_θ(θ − θ₀)²` in the file, as LAMMPS `K = k_θ/2`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OplsAngleRow {
    /// Class of the first atom.
    pub i: &'static str,
    /// Class of the central atom.
    pub j: &'static str,
    /// Class of the third atom.
    pub k: &'static str,
    /// Force constant (kcal/mol/rad²) in molrs's — LAMMPS's — `E = k(θ−θ0)²`
    /// convention (GROMACS's `k_θ / 2`); emitted under the param key `k`. Not
    /// named `k` here because this struct's `k` is already the third atom's
    /// class.
    pub force_constant: f64,
    /// Equilibrium angle (**degrees**).
    pub theta0: f64,
}

/// One GROMACS `[ dihedraltypes ]` funct-3 row, as the OPLS 4-cosine Fourier
/// series the `dihedral:opls` kernel evaluates:
/// `V = ½[f₁(1+cosφ) + f₂(1−cos2φ) + f₃(1+cos3φ) + f₄(1−cos4φ)]`.
///
/// The source stated it in Ryckaert–Bellemans form (`c0..c5`, kJ/mol); the
/// generator applies GROMACS Eqs. 200–201 — the exact analytic inversion, which
/// is independent of `c0` and `c5` — and the kcal/mol conversion. A GROMACS `X`
/// endpoint is the empty-string wildcard.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OplsDihedralRow {
    /// Class of the first atom.
    pub i: &'static str,
    /// Class of the second atom.
    pub j: &'static str,
    /// Class of the third atom.
    pub k: &'static str,
    /// Class of the fourth atom.
    pub l: &'static str,
    /// 1-fold coefficient (kcal/mol).
    pub f1: f64,
    /// 2-fold coefficient (kcal/mol).
    pub f2: f64,
    /// 3-fold coefficient (kcal/mol).
    pub f3: f64,
    /// 4-fold coefficient (kcal/mol).
    pub f4: f64,
}
