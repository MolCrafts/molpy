pub(crate) mod candidate;
mod cascade;
pub(crate) mod empirical;
mod provenance;
mod tables;
mod term;

use std::collections::HashMap;

use molrs::core::Element;

use crate::ff::forcefield::ForceField;
use crate::ff::ir::Params;
use crate::ff::params::{EmpiricalTable, ParmchkTable};

use candidate::CandidateSet;
pub(crate) use cascade::DEFAULT_IMPROPER;
pub use provenance::{Estimate, EstimateMethod, PenaltyTier, Provenance};
pub use tables::EmpiricalSet;
pub use term::BondedTerm;

use cascade::Arity;

/// Generic interpolation seam for typifier parameter families.
///
/// `Term` is intentionally an associated type: bonded parameters use
/// [`BondedTerm`], while future typifiers can introduce their own query structs
/// for atom, pair, stretch-bend, or charge-correction parameters without changing
/// this trait.
pub trait ParameterInterpolator {
    /// Query type understood by this interpolator.
    type Term;

    /// Interpolate parameters for `term`, or return `Ok(None)` to decline.
    fn interpolate(&self, term: &Self::Term) -> Result<Option<Params>, String>;
}

/// Reusable atom-type metadata the cascade needs from a typifier.
///
/// Two pieces, both force-field agnostic: a type-to-class map for class-keyed
/// force fields (OPLS bonded forces key on `CT`, not on `opls_135`), and a
/// type-to-element map for the empirical fallbacks. Keeping them here is what lets
/// a non-OPLS typifier build the same estimator without pretending to be OPLS.
#[derive(Debug, Clone, Default)]
pub struct EstimationInputs {
    type_to_class: HashMap<String, String>,
    type_to_element: HashMap<String, Element>,
}

/// An input set of `(type_name, class_name)` pairs, no element known yet.
impl<K: Into<String>, V: Into<String>> FromIterator<(K, V)> for EstimationInputs {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(classes: I) -> Self {
        let mut out = Self::new();
        for (name, class) in classes {
            out.insert_class(name, class);
        }
        out
    }
}

impl EstimationInputs {
    /// Create an empty interpolation input set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or replace a type-to-class entry.
    pub fn insert_class(&mut self, name: impl Into<String>, class: impl Into<String>) {
        self.type_to_class.insert(name.into(), class.into());
    }

    /// Insert or replace a type-to-element entry.
    pub fn insert_element(&mut self, name: impl Into<String>, element: Element) {
        self.type_to_element.insert(name.into(), element);
    }

    /// Add element inference from a force field's atom masses.
    ///
    /// The empirical formulas need a per-atom **element**, but a force-field reader
    /// keeps only `name` + `mass` per type. Rather than plumb a new element channel
    /// through every reader, each type's element is inferred from its tabulated mass
    /// by nearest standard-atomic-mass match ([`molrs::core::Element`]). This is
    /// force-field agnostic, and the analogy cascade itself never needs an element —
    /// it works on type / class names.
    ///
    /// A class-keyed force field's bonded rows name **classes** (`CA`, `OS`), and a
    /// class is never an atom type with a mass. So every class already known to this
    /// input set also takes the mass-derived element of its member types; a class with
    /// no member type falls back on reading its name (`element_from_token`).
    pub fn with_forcefield_elements(mut self, ff: &ForceField) -> Self {
        for at in ff.get_atomtypes() {
            if let Some(mass) = at.params.get("mass").and_then(|v| v.as_num())
                && let Some(element) = element_from_mass(mass)
            {
                self.type_to_element.insert(at.name.clone(), element);
            }
        }
        let mut class_elements: HashMap<String, Element> = HashMap::new();
        for (name, class) in &self.type_to_class {
            if let Some(&element) = self.type_to_element.get(name) {
                let first = *class_elements.entry(class.clone()).or_insert(element);
                debug_assert_eq!(
                    first, element,
                    "class {class} has member types of different elements"
                );
            }
        }
        for (class, element) in class_elements {
            self.type_to_element.entry(class).or_insert(element);
        }
        self
    }

    fn class_of(&self, name: &str) -> Option<&str> {
        self.type_to_class.get(name).map(String::as_str)
    }

    fn element_of(&self, name: &str) -> Option<Element> {
        self.type_to_element
            .get(name)
            .copied()
            .or_else(|| element_from_token(name))
    }
}

/// The missing-parameter estimator: parmchk2's cascade over one candidate table.
///
/// Built once from a force field plus typifier metadata, it caches the candidate
/// rows and the per-type class / element maps, so each
/// [`interpolate`](ParameterInterpolator::interpolate) call is a scan and nothing
/// more.
///
/// # It is named after parmchk2 because it *is* parmchk2's algorithm
///
/// The tiers, the penalty weights, the equivalence and correspondence tables and
/// the empirical formulas are all AmberTools' GAFF data and AmberTools' ordering,
/// and `parmchk2` is the oracle every one of them is checked against
/// (`tests/ff/typifier/parmchk2_oracle.rs`: 37 molecules × {gaff, gaff2} — the same
/// estimated-term set, the same values, the same confidence bands). A force field
/// that is not GAFF may still use it, and the OPLS typifier does, but it borrows
/// GAFF's tables to do so and degrades where they cannot speak its type names. See
/// [below](Self#the-tables-are-gaffs-that-is-a-limitation-and-it-is-deliberate).
///
/// The generic missing-parameter estimator: a parmchk2-style cascade any force
/// field can borrow.
///
/// A force field's tables never cover every term of every molecule. What fills the
/// gap for a force field with no estimator of its own is [`Parmchk2Estimator`] —
/// a cascade modelled on parmchk2's (exact → equivalent → **wildcard row** →
/// corresponding, plus an additive penalty with inner atoms weighted ×10) backed
/// by the GAFF empirical formulas (Badger bond `k`, mean-of-neighbours θ₀, the
/// Wang2004 Eq. 5 angle `K_θ`, and a never-fabricate rule for torsions). **No
/// ab-initio / QM fitting is ever performed.**
///
/// It is **not** parmchk2. GAFF itself does not use it:
/// [`GaffTypifier`](crate::ff::typifier::GaffTypifier) reproduces parmchk2's own
/// searches exactly, quirks and all (`gaff::analog`, `gaff::torsion`,
/// `gaff::improper`), because GAFF's estimates are AmberTools' by definition.
/// This cascade keeps its own, simpler scoring, which the OPLS-AA typifier's
/// estimates and tests are built on:
///
/// | Term | Column read per substituted atom (`PARMCHK.DAT`) | Weight |
/// |---|---|---|
/// | bond | `bl` | `WEIGHT_BL` |
/// | angle | `cba` (the centre column), at every atom | `WEIGHT_BA`, ×`WEIGHT_BA_CTR` at the vertex |
/// | torsion, inner atom | `tor`, else ½·`ctor` + ½·similarity | ×`WEIGHT_TOR_CTR` |
/// | torsion, outer atom | `ctor`, else `DEFAULT_TOR` | 1 |
///
/// parmchk2 scores an angle end by `ba` + `baf` and its vertex by `cba` +
/// `cbaf`, a bond by `bl` + `blf`, a torsion's inner atom by `ctor` and its
/// outer by `tor`, and adds group and conjugation penalties; a penalty this
/// cascade reports is therefore its own, not parmchk2's.
///
/// It reaches its callers as an interpolation seam: it implements
/// [`ParameterInterpolator`] for [`BondedTerm`] and is injected into the OPLS
/// bonded matcher via
/// [`OplsAaTypifier::with_estimator`](super::opls::OplsAaTypifier::with_estimator). Exact matches
/// always win first; with `strict=true` the interpolator is never consulted; with
/// none attached the assign path is byte-identical to pre-interpolator behaviour.
/// [`Parmchk2Estimator::estimate`] keeps the [`Covered`](Estimate::Covered) /
/// [`Estimated`](Estimate::Estimated) distinction for a caller that reads a
/// table directly.
///
/// # The cascade: exact → equivalent → wildcard row → corresponding → empirical
///
/// One set of tiers, modelled on parmchk2's order, for every arity (the
/// generic estimator — GAFF itself reproduces parmchk2 exactly in
/// `typifier::gaff`). The tiers are what the whole estimator is:
///
/// 1. **exact / class match** — the row is written for these very types. Penalty 0.
/// 2. **equivalent-type substitution** (`EQUA`) — `gaff2`'s `ns` *is* `n`, so the
///    specific row `o-c-n -hn` covers `o-c-ns-hn` and costs **nothing**. This tier
///    outranks the next: a specific row for an equivalent type beats a generic row
///    for this one.
/// 3. **a wildcard row that matches outright** (`X -c3-c3-X ` over ethane's
///    `hc-c3-c3-hc`) — the table *covers* the term, so this is a **parameter, not
///    an estimate**: nothing is substituted, no penalty is charged, and the result
///    carries no provenance ([`Estimate::covered`]). This is the half that stops an
///    estimator fabricating an analogy for the ~145 terms per molecule AMBER simply
///    looks up.
/// 4. **corresponding-type substitution** (`CORR`) — scored. Specific rows are
///    searched before wildcard ones, then by how badly the substitution disturbs
///    the term's inner atoms, then in file order.
/// 5. **empirical formula** — nothing left to copy; Wang et al. 2004's formulas (see Units).
///
/// ## Impropers are not a fourth kind of proper term
///
/// A proper term is a walk along bonds and reads the same backwards, so a row is
/// tried in both orientations. An improper is a planarity constraint on a
/// **centre** whose three peripherals are an unordered *set*, so its row is tried
/// against all six assignments of peripherals to slots and the cheapest is paid
/// for. And a **specific** improper row is an exact match or nothing — so
/// methyl methacrylate's `c -c2-ce-c3` comes out as the default rather than as
/// a substituted `c -c2-c2-c3`.
///
/// # The tables are GAFF's. That is a limitation, and it is deliberate.
///
/// Every constant the cascade scores with comes from AmberTools' GAFF data: the
/// atom-type equivalences and correspondences (`PARMCHK.DAT`), the penalty weights
/// and defaults (its `WEIGHT_*` / `DEFAULT_*` block), and the empirical bond /
/// angle constants (`PARM_BLBA_GAFF*.DAT`). They are keyed by **GAFF atom-type
/// names** — `c3`, `os`, `ca`.
///
/// A GAFF-typed term therefore gets the full cascade. **Any other force field
/// borrows it and degrades**: `opls_135` appears in no row of the substitution
/// table, so no equivalence and no correspondence can ever be found for it, and the
/// estimator falls back on what it *can* still say — a type-name / class-name match
/// (penalty 0), element compatibility at the arity's default penalty, and the
/// element-keyed empirical formulas. That is a real floor, and an honest one: an
/// OPLS estimate is a coarser thing than a GAFF estimate, and its penalty says so.
///
/// # Provenance
///
/// Every estimated term carries the four provenance keys of
/// [`Provenance::apply_to`] (`estimated`, `estimate_penalty`, `estimate_method`,
/// `estimate_analog`) so a consumer can audit and tier it. A term a wildcard row
/// *covers* carries none of them — it is a parameter, not an estimate.
///
/// # Units
///
/// molrs's convention, LAMMPS's: angles (θ₀, phases) in **degrees**, lengths Å,
/// harmonic force constants un-halved (`E = K·(x − x₀)²`) — what a candidate
/// table must present. Force constants are copied **verbatim** from the
/// candidate table, and the empirical formulas produce the same `K` that
/// `gaff.dat` itself tabulates (see the `empirical` stage).
pub struct Parmchk2Estimator {
    /// The rows the cascade scans, by arity.
    candidates: CandidateSet,
    /// Typifier-side type metadata (class + element).
    inputs: EstimationInputs,
    /// `PARMCHK.DAT`: equivalences, correspondences, penalty weights, and the
    /// improper-centre column.
    substitutions: ParmchkTable,
    /// The Badger / angle empirical constants of one force field.
    empirical: EmpiricalTable,
}

impl Parmchk2Estimator {
    /// Build an estimator from a force field and its interpolation input set.
    ///
    /// `inputs` carries the typifier's type metadata: for OPLS-AA the
    /// `type → class` map of its
    /// [`OplsTypingMetadata`](crate::ff::typifier::OplsTypingMetadata)
    /// (`meta.iter().map(|(name, row)| (name.clone(), row.class.clone())).collect()`)
    /// with each type's element inferred from its tabulated mass
    /// ([`EstimationInputs::with_forcefield_elements`]).
    ///
    /// The candidate rows are flattened out of every bonded style the force field
    /// declares, by style **kind** and never by style *name*: GAFF's dihedral style
    /// is `periodic` and OPLS's is `opls`, and an extractor that asks for one by
    /// name is an extractor with an empty table for the other.
    pub fn new(ff: &ForceField, inputs: EstimationInputs) -> Self {
        Self {
            candidates: CandidateSet::from_forcefield(ff),
            inputs,
            substitutions: tables::substitution_table(),
            empirical: EmpiricalSet::Gaff.table(),
        }
    }

    /// Choose the empirical constant set (`PARM_BLBA_GAFF.DAT` vs `…GAFF2.DAT`).
    ///
    /// The two files differ, so a GAFF2 force field must say so; everything else
    /// keeps the GAFF set, whose constants are element-keyed and generic.
    pub fn with_empirical(mut self, set: EmpiricalSet) -> Self {
        self.empirical = set.table();
        self
    }

    /// May an atom of this type be the CENTRE of an improper at all?
    ///
    /// `PARMCHK.DAT`'s `improper_flag` column: `ca` and `c` and `na` carry a
    /// planarity term, `c3` and `n3` do not. That is upstream **data**, not a
    /// hybridisation the engine re-derives — and it is why benzene gets its
    /// ring-planarity improper while methylamine's sp3 nitrogen gets none.
    pub fn is_improper_centre(&self, atom_type: &str) -> bool {
        self.substitutions.is_improper_centre(atom_type)
    }

    // -- the cascade, as a table reader -------------------------------------

    /// Run the cascade for one term.
    ///
    /// The primitive both callers share, and the one that keeps the distinction
    /// that matters: [`Estimate::Covered`] means a row of the table covers the term
    /// outright (nothing estimated, nothing charged), [`Estimate::Estimated`] means
    /// it had to be reached by analogy or by formula. `None` means nothing could
    /// produce it — **no barrier is ever fabricated here**.
    pub fn estimate(&self, term: &BondedTerm) -> Option<Estimate> {
        match term {
            BondedTerm::Bond(types) => self.bond(types),
            BondedTerm::Angle(types) => self.angle(types),
            BondedTerm::Dihedral(types) => self.torsion(refs(types)),
            BondedTerm::Improper(types) => {
                Some(self.improper(&types[2], [&types[0], &types[1], &types[3]]))
            }
        }
    }

    fn bond(&self, types: &[String; 2]) -> Option<Estimate> {
        self.analogy(
            &self.candidates.bonds,
            &[&types[0], &types[1]],
            &[false, false],
            Arity::Bond,
        )
        .or_else(|| self.empirical_bond(types))
    }

    fn angle(&self, types: &[String; 3]) -> Option<Estimate> {
        // The vertex (index 1) is the inner atom → ×10 weighting.
        self.analogy(
            &self.candidates.angles,
            &[&types[0], &types[1], &types[2]],
            &[false, true, false],
            Arity::Angle,
        )
        .or_else(|| self.empirical_angle(types))
    }

    /// Element of an atom type — force-field metadata, not a frame species: the
    /// typifier's map (mass inference), then the type name read as an element
    /// token (`c3` → C), then `PARMCHK.DAT`'s own atomic-number column.
    pub(crate) fn element_of(&self, name: &str) -> Option<Element> {
        self.inputs
            .element_of(name)
            .or_else(|| self.substitutions.element(name))
    }
}

impl ParameterInterpolator for Parmchk2Estimator {
    type Term = BondedTerm;

    /// The seam: [`estimate`](Parmchk2Estimator::estimate) a missing bonded
    /// term, with the provenance convention written onto the params — bond
    /// `k` / `r0`, angle `k` / `theta0` (degrees), improper `k` / `n` / `d`
    /// for a planar centre given in AMBER slot order (**centre third**).
    ///
    /// A dihedral never gets a fabricated rigid barrier: an analog (the whole
    /// multi-periodicity group, copied as one), else a generic wildcard term,
    /// else a **near-zero barrier** carrying a poor-tier penalty that says
    /// there is no torsion here — where `estimate` answers `None`.
    fn interpolate(&self, term: &BondedTerm) -> Result<Option<Params>, String> {
        if let BondedTerm::Dihedral(types) = term
            && self.torsion(refs(types)).is_none()
        {
            let mut params = self.no_torsion();
            Provenance::wildcard(self.no_torsion_penalty(), "").apply_to(&mut params);
            return Ok(Some(params));
        }
        Ok(self.estimate(term).map(Estimate::into_params))
    }
}

/// Borrow a quartet of owned names as a quartet of `&str`.
fn refs(types: &[String; 4]) -> [&str; 4] {
    [&types[0], &types[1], &types[2], &types[3]]
}

/// Nearest standard-atomic-mass element for a mass (amu). `None` for a
/// non-physical mass (≤ 0).
fn element_from_mass(mass: f64) -> Option<Element> {
    if mass <= 0.0 {
        return None;
    }
    let mut best: Option<(f64, Element)> = None;
    for &e in Element::ALL {
        let diff = (e.atomic_mass() as f64 - mass).abs();
        if best.is_none_or(|(d, _)| diff < d) {
            best = Some((diff, e));
        }
    }
    best.map(|(_, e)| e)
}

/// Element from an atom-type token (e.g. `c3` → `C`, `cl` → `Cl`).
///
/// GAFF lowercase atom types encode the element as the **leading letter**
/// (`c3`/`ca`/`cc` → C, `os`/`oh` → O), with only the genuine two-letter halogens
/// written two-letter (`cl` → Cl, `br` → Br). So the single leading letter is tried
/// first (correctly mapping `os` → O, not Osmium); the two-letter form is the
/// fallback for tokens whose single letter is not an element. Type names that are
/// real element symbols (`Cl`, `Br`) still resolve. All-caps tokens (OPLS classes
/// `CA`, `OS`, `NB`) follow the same leading-letter rule: `CA` is carbon, not
/// calcium.
fn element_from_token(token: &str) -> Option<Element> {
    let base: String = token
        .chars()
        .take_while(|c| c.is_ascii_alphabetic())
        .collect();
    if base.is_empty() {
        return None;
    }
    // An explicitly title-cased multi-letter token (`Cl`, `Br`) is a real element
    // symbol — honour it before the GAFF leading-letter convention. Only title
    // case qualifies: `Element::by_symbol` is case-insensitive, so an all-caps
    // `CA` would otherwise read as calcium.
    let bytes = base.as_bytes();
    if bytes.len() >= 2
        && bytes[0].is_ascii_uppercase()
        && bytes[1].is_ascii_lowercase()
        && let Some(element) = Element::by_symbol(&base)
    {
        return Some(element);
    }
    // GAFF writes the genuine two-letter halogens lowercase (`cl` / `br`); these
    // must win over the leading-letter rule (which would read `cl` as carbon).
    let lower = base.to_ascii_lowercase();
    if lower == "cl" {
        return Some(Element::Cl);
    }
    if lower == "br" {
        return Some(Element::Br);
    }
    // GAFF / OPLS convention: the leading letter is the element (`c3` / `os` /
    // `CA`).
    if let Some(element) = Element::by_symbol(&base[..1]) {
        return Some(element);
    }
    // Fallback: any other genuine two-letter element.
    if base.len() >= 2 {
        return Element::by_symbol(&base[..2]);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::OnceLock;

    use crate::ff::typifier::Typifier;
    use crate::ff::typifier::{GaffParameterSet, GaffTypifier};

    /// The cascade over a GAFF table's candidate rows and empirical constants.
    fn over(set: GaffParameterSet) -> &'static Parmchk2Estimator {
        static GAFF: OnceLock<Parmchk2Estimator> = OnceLock::new();
        static GAFF2: OnceLock<Parmchk2Estimator> = OnceLock::new();
        let (cell, empirical) = match set {
            GaffParameterSet::Gaff => (&GAFF, EmpiricalSet::Gaff),
            GaffParameterSet::Gaff2 => (&GAFF2, EmpiricalSet::Gaff2),
        };
        cell.get_or_init(|| {
            let gaff = GaffTypifier::new(set);
            let candidates = gaff.source_forcefield();
            let inputs = EstimationInputs::new().with_forcefield_elements(candidates);
            Parmchk2Estimator::new(candidates, inputs).with_empirical(empirical)
        })
    }

    /// The cascade over `gaff.dat`.
    fn gaff() -> &'static Parmchk2Estimator {
        over(GaffParameterSet::Gaff)
    }

    fn types<const N: usize>(names: [&str; N]) -> [String; N] {
        names.map(str::to_owned)
    }

    // --- the tiers, in order (ac-004) --------------------------------------

    #[test]
    fn tier_wildcard_row_is_a_parameter_not_an_estimate() {
        // `X -c3-c3-X ` covers ethane's H-C-C-H: LEaP finds it, parmchk2 is silent.
        let estimate = gaff()
            .estimate(&BondedTerm::Dihedral(types(["hc", "c3", "c3", "hc"])))
            .expect("a match");
        assert!(
            estimate.provenance().is_none(),
            "a term the table covers is not an estimate"
        );
        assert!(
            estimate
                .params()
                .get("k1")
                .and_then(|v| v.as_num())
                .is_some()
        );
    }

    #[test]
    fn tier_equivalent_type_substitutes_free_of_charge() {
        // gaff2's `ns` is `n`: the specific row `o-c-n -hn` covers `o-c-ns-hn`.
        let estimate = over(GaffParameterSet::Gaff2)
            .estimate(&BondedTerm::Dihedral(types(["o", "c", "ns", "hn"])))
            .expect("an equivalent-type match");
        let provenance = estimate.provenance().expect("estimated");
        assert_eq!(provenance.analog, "o-c-n-hn");
        assert_eq!(provenance.method, EstimateMethod::Analogy);
        assert!(
            provenance.penalty.abs() < 1e-12,
            "EQUA costs nothing, got {}",
            provenance.penalty
        );
    }

    #[test]
    fn tier_corresponding_type_substitution_is_scored() {
        // Thiophene's `cc-cd-ss-cd`: no row covers it, so `cd` stands in for `c2`
        // on the inner atom of the wildcard row `X -c2-ss-X `.
        let estimate = gaff()
            .estimate(&BondedTerm::Dihedral(types(["cc", "cd", "ss", "cd"])))
            .expect("a match");
        let provenance = estimate.provenance().expect("estimated");
        assert_eq!(provenance.analog, "X-c2-ss-X");
        assert!(
            (provenance.penalty - 232.0).abs() < 0.05,
            "parmchk2 charges 232.0, got {}",
            provenance.penalty
        );
        assert_eq!(provenance.tier(), PenaltyTier::Poor);
    }

    /// The empirical tier — the one the parmchk2 oracle **cannot** reach.
    ///
    /// Every bond and angle of all 37 oracle molecules is an exact hit in both
    /// tables, so nothing there exercises the Badger / Eq. 5 formulas: the oracle's
    /// green says nothing whatsoever about them, and no oracle case may be
    /// fabricated to pretend otherwise (`tests/ff/typifier/parmchk2_oracle.rs` says
    /// so in its own module docs). So the tier is driven here, directly.
    ///
    /// **H–Br is the hole.** `gaff.dat` has a bond row for `br` to every heavy atom
    /// it can reach and for `br-br` itself, but none for hydrogen bonded to bromine
    /// — no GAFF-typed molecule has one — so no row exists whose two ends are even
    /// the right *elements*, and the analogy tier cannot reach one by substitution.
    /// `PARM_BLBA_GAFF.DAT`, keyed by element rather than by atom type, does carry
    /// the H–Br pair. That is precisely the gap the empirical tier exists to fill.
    #[test]
    fn tier_empirical_formula_is_the_last_resort() {
        let estimator = gaff();

        let bond = estimator
            .estimate(&BondedTerm::Bond(types(["hc", "br"])))
            .expect("the empirical formula produces a bond");
        let provenance = bond.provenance().expect("estimated");
        assert_eq!(
            provenance.method,
            EstimateMethod::Empirical,
            "no row and no analog: the tier below analogy is the only one left"
        );
        assert_eq!(provenance.analog, "", "a formula has no row to name");
        assert_eq!(
            provenance.tier(),
            PenaltyTier::Caution,
            "an empirical bond is charged DEFAULT_BL — read it with care"
        );

        // The estimate must be Badger's rule (Wang2004 Eq. 3) evaluated on exactly
        // the H–Br row of the empirical table, at the equilibrium length it gives.
        let ln_k = estimator
            .empirical
            .bond_ln_k(Element::H, Element::Br)
            .expect("tabulated");
        let want_r = estimator
            .empirical
            .bond_length(Element::H, Element::Br)
            .expect("tabulated");
        // The formula yields AMBER's un-halved `K`, molrs's (LAMMPS's) `k`.
        let want_k = empirical::bond_k(ln_k, want_r, estimator.empirical.bond_power);
        let r0 = bond
            .params()
            .get("r0")
            .and_then(|v| v.as_num())
            .expect("r0");
        let k = bond.params().get("k").and_then(|v| v.as_num()).expect("k");
        assert!((r0 - want_r).abs() < 1e-12, "r₀ is the reference length");
        assert!((k - want_k).abs() < 1e-9, "k = exp(ln Kij) / r^m");
        assert!(k > 0.0, "an empirical force constant is positive");
    }

    // --- impropers ---------------------------------------------------------

    #[test]
    fn benzene_gets_the_ring_planarity_improper_off_a_wildcard_row() {
        let estimate = gaff().improper("ca", ["ca", "ca", "ha"]);
        let provenance = estimate.provenance().expect("estimated");
        assert_eq!(provenance.analog, "X-X-ca-ha");
        assert_eq!(provenance.method, EstimateMethod::GenericWildcard);
        assert!(
            (estimate
                .params()
                .get("k")
                .and_then(|v| v.as_num())
                .expect("k")
                - 1.1)
                .abs()
                < 1e-12
        );
        assert!(
            (provenance.penalty - 6.0).abs() < 0.05,
            "two wildcards, 3.0 each"
        );
    }

    #[test]
    fn the_amide_improper_needs_a_planar_neighbour() {
        // N-methylacetamide's carbonyl: `n` is planar, so the 10.5 amide term applies.
        let amide = gaff().improper("c", ["c3", "o", "n"]);
        assert!(
            (amide.params().get("k").and_then(|v| v.as_num()).expect("k") - 10.5).abs() < 1e-12
        );

        // Acetone's carbonyl has the same shape and NO planar neighbour, so
        // parmchk2 falls back on the default 1.1 rather than calling it an amide.
        let ketone = gaff().improper("c", ["c3", "c3", "o"]);
        let (barrier, ..) = DEFAULT_IMPROPER;
        assert!(
            (ketone
                .params()
                .get("k")
                .and_then(|v| v.as_num())
                .expect("k")
                - barrier)
                .abs()
                < 1e-12
        );
        assert!(
            ketone.provenance().is_some(),
            "a default is still an estimate"
        );
    }

    #[test]
    fn an_sp3_centre_carries_no_improper_at_all() {
        let estimator = gaff();
        assert!(estimator.is_improper_centre("ca"));
        assert!(estimator.is_improper_centre("c"));
        assert!(!estimator.is_improper_centre("c3"));
        assert!(!estimator.is_improper_centre("n3"), "methylamine's amine N");
    }

    // --- orientation symmetry of a dihedral estimate ------------------------

    /// A one-row OPLS-shaped library, `CZ-CT-CT-CW` under `dihedral/opls`,
    /// and an input set for four carbon types: `ta` (class `CZ`), `tb` / `tc`
    /// (class `CT`) and `td` (class `CY`). No class is a GAFF type, so every
    /// substitution is priced by element compatibility at `DEFAULT_TOR`.
    fn one_row_dihedral_estimator() -> Parmchk2Estimator {
        let mut ff = ForceField::new("one-row");
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type(
                "CZ-CT-CT-CW",
                &["CZ", "CT", "CT", "CW"],
                Params::from_iter([("k1", 1.0), ("k2", 0.0), ("k3", 0.5), ("k4", 0.0)]),
            )
            .unwrap();
        let mut inputs =
            EstimationInputs::from_iter([("ta", "CZ"), ("tb", "CT"), ("tc", "CT"), ("td", "CY")]);
        for name in ["ta", "tb", "tc", "td"] {
            inputs.insert_element(name, Element::C);
        }
        Parmchk2Estimator::new(&ff, inputs)
    }

    /// A proper torsion reads the same backwards (`BondedTerm::Dihedral`'s
    /// contract, and why `type_name` canonicalises it), so `ta-tb-tc-td` and
    /// `td-tc-tb-ta` must get one estimate — provenance included.
    ///
    /// Hand-derived: against `CZ-CT-CT-CW` the forward query needs one
    /// substitution (`CW` for `td`), the row read backwards two (`CW` for `ta`,
    /// `CZ` for `td`); both leave the inner pair untouched, so they tie on the
    /// inner score the tier-4 search ranks by. The cheaper reading (one
    /// substitution) is the same torsion whichever end the query starts from.
    #[test]
    fn a_dihedral_estimate_is_the_same_read_from_either_end() {
        let estimator = one_row_dihedral_estimator();
        let forward = estimator
            .interpolate(&BondedTerm::Dihedral(types(["ta", "tb", "tc", "td"])))
            .unwrap()
            .expect("estimated");
        let backward = estimator
            .interpolate(&BondedTerm::Dihedral(types(["td", "tc", "tb", "ta"])))
            .unwrap()
            .expect("estimated");
        assert_eq!(
            forward,
            backward,
            "one torsion, one estimate: penalty {:?} vs {:?}",
            forward.get("estimate_penalty").and_then(|v| v.as_num()),
            backward.get("estimate_penalty").and_then(|v| v.as_num())
        );
    }

    // --- element inference -------------------------------------------------

    #[test]
    fn element_from_mass_picks_nearest() {
        assert_eq!(element_from_mass(12.011), Some(Element::C));
        assert_eq!(element_from_mass(1.008), Some(Element::H));
        assert_eq!(element_from_mass(15.999), Some(Element::O));
        assert_eq!(element_from_mass(0.0), None);
    }

    #[test]
    fn element_from_token_reduces_gaff_types() {
        assert_eq!(element_from_token("c3"), Some(Element::C));
        assert_eq!(element_from_token("hc"), Some(Element::H));
        assert_eq!(element_from_token("cl"), Some(Element::Cl));
        assert_eq!(element_from_token("os"), Some(Element::O));
    }

    /// OPLS row classes are written all-caps (`CA`, `OS`, `NB`). A class is never a
    /// key of the mass map, so it resolves through `element_from_token`, and the
    /// OPLS convention is the same as GAFF's: the leading letter is the element.
    /// `CA` is aromatic carbon, not calcium; `OS` ether oxygen, not osmium.
    #[test]
    fn element_from_token_reads_all_caps_opls_classes_by_their_leading_letter() {
        for class in ["CA", "CM", "CN", "CO", "CR", "CS", "CU"] {
            assert_eq!(
                element_from_token(class),
                Some(Element::C),
                "OPLS class {class} is a carbon"
            );
        }
        for class in ["NA", "NB", "NO"] {
            assert_eq!(
                element_from_token(class),
                Some(Element::N),
                "OPLS class {class} is a nitrogen"
            );
        }
        assert_eq!(
            element_from_token("OS"),
            Some(Element::O),
            "OPLS class OS is an oxygen"
        );
        for class in ["HO", "HS"] {
            assert_eq!(
                element_from_token(class),
                Some(Element::H),
                "OPLS class {class} is a hydrogen"
            );
        }
    }

    #[test]
    fn element_from_token_keeps_title_case_symbols_and_gaff_types() {
        assert_eq!(element_from_token("Cl"), Some(Element::Cl));
        assert_eq!(element_from_token("Br"), Some(Element::Br));
        assert_eq!(element_from_token("ca"), Some(Element::C));
        assert_eq!(element_from_token("cl"), Some(Element::Cl));
        assert_eq!(element_from_token("na"), Some(Element::N));
        assert_eq!(element_from_token("os"), Some(Element::O));
    }

    // --- all-caps OPLS classes through the cascade ---------------------------

    /// An OPLS-shaped estimator: `ff` carries the bonded rows, and each
    /// `(type, class, mass)` becomes an `atom/full` type whose element the input set
    /// infers from its mass — exactly how OPLS-AA's estimator inputs are built.
    fn opls_class_estimator(
        mut ff: ForceField,
        atom_types: &[(&str, &str, f64)],
    ) -> Parmchk2Estimator {
        let atoms = ff.def_style("atom", "full", Params::new()).unwrap();
        for (name, _, mass) in atom_types {
            atoms
                .def_type(
                    name,
                    &[],
                    Params::from_iter([("mass", *mass), ("charge", 0.0)]),
                )
                .unwrap();
        }
        let inputs = atom_types
            .iter()
            .map(|(name, class, _)| (*name, *class))
            .collect::<EstimationInputs>()
            .with_forcefield_elements(&ff);
        Parmchk2Estimator::new(&ff, inputs)
    }

    /// One bond row `CA-CT`; types `ta` (class `CW`) and `tb` (class `CT`), both
    /// carbon by mass. Hand-derived: `tb` matches `CT` by class (0); `ta` against
    /// `CA` is carbon for carbon with nothing tabulated, so it costs `DEFAULT_BL`
    /// (`PARMCHK.DAT`: 20.0). The reversed reading pays that twice. So the bond is
    /// an analogy off `CA-CT` at 20.0 — never the empirical formula, which is what
    /// reading `CA` as calcium forces.
    #[test]
    fn an_all_caps_opls_bond_row_is_reached_by_element_analogy() {
        let mut ff = ForceField::new("one-bond-row");
        ff.def_style("bond", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CA-CT",
                &["CA", "CT"],
                Params::from_iter([("k", 634.0), ("r0", 1.51)]),
            )
            .unwrap();
        let estimator = opls_class_estimator(ff, &[("ta", "CW", 12.011), ("tb", "CT", 12.011)]);

        let estimate = estimator
            .estimate(&BondedTerm::Bond(types(["ta", "tb"])))
            .expect("a carbon-carbon bond has an analog");
        let provenance = estimate.provenance().expect("estimated, not covered");
        assert_eq!(provenance.method, EstimateMethod::Analogy);
        assert_eq!(provenance.analog, "CA-CT");
        assert!(
            (provenance.penalty - 20.0).abs() < 1e-12,
            "one element substitution at DEFAULT_BL = 20.0, got {}",
            provenance.penalty
        );
        assert!(
            (estimate
                .params()
                .get("r0")
                .and_then(|v| v.as_num())
                .expect("r0")
                - 1.51)
                .abs()
                < 1e-12
        );
        assert!(
            (estimate
                .params()
                .get("k")
                .and_then(|v| v.as_num())
                .expect("k")
                - 634.0)
                .abs()
                < 1e-10
        );
    }

    /// One dihedral row `CT-CA-OS-CT`; query types of classes `CT-CW-OH-CT`
    /// (carbon, carbon, oxygen, carbon by mass). Hand-derived: the outer `CT`s match
    /// by class; the inner `CW` for `CA` and `OH` for `OS` are element-compatible
    /// non-GAFF substitutions at `DEFAULT_TOR` (87.0) each → 174.0. The reversed
    /// row puts `OS` against a carbon and is refused. Reading `CA` as calcium and
    /// `OS` as osmium refuses both inner slots and hands back the `no_torsion`
    /// placeholder instead.
    #[test]
    fn an_all_caps_opls_torsion_row_is_reached_by_element_analogy() {
        let row = Params::from_iter([("k1", 0.0), ("k2", 3.0), ("k3", 0.0), ("k4", 0.0)]);
        let mut ff = ForceField::new("one-dihedral-row");
        ff.def_style("dihedral", "opls", Params::new())
            .unwrap()
            .def_type("CT-CA-OS-CT", &["CT", "CA", "OS", "CT"], row.clone())
            .unwrap();
        let estimator = opls_class_estimator(
            ff,
            &[
                ("t1", "CT", 12.011),
                ("t2", "CW", 12.011),
                ("t3", "OH", 15.999),
                ("t4", "CT", 12.011),
            ],
        );

        let estimate = estimator
            .estimate(&BondedTerm::Dihedral(types(["t1", "t2", "t3", "t4"])))
            .expect("an element-compatible analog exists: not the no_torsion placeholder");
        let provenance = estimate.provenance().expect("estimated, not covered");
        assert_eq!(provenance.method, EstimateMethod::Analogy);
        assert_eq!(provenance.analog, "CT-CA-OS-CT");
        assert!(
            (provenance.penalty - 174.0).abs() < 1e-12,
            "two inner substitutions at DEFAULT_TOR = 87.0, got {}",
            provenance.penalty
        );
        for key in ["k1", "k2", "k3", "k4"] {
            assert_eq!(
                estimate.params().get(key).and_then(|v| v.as_num()),
                row.get(key).and_then(|v| v.as_num()),
                "the row's {key} is copied, not a near-zero barrier"
            );
        }
    }
}
