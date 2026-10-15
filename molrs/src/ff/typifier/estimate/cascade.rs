//! The cascade: exact → equivalent → wildcard row → corresponding → empirical.

use crate::ff::ir::Params;
use crate::ff::params::ParmchkPenalty;

use super::Parmchk2Estimator;
use super::candidate::{Candidate, is_wildcard};
use super::empirical;
use super::provenance::{Estimate, Provenance};

/// parmchk2's improper default: `1.1 kcal/mol`, phase 180°, periodicity 2.
///
/// Not a row of any table — it is the value parmchk2 falls back on when nothing
/// matches, and it prints no penalty for it ("a default is not a substitution, so
/// there is nothing to score").
pub const DEFAULT_IMPROPER: (f64, f64, u32) = (1.1, 180.0, 2);

/// What an `X` in a peripheral slot of an improper row costs: **3.0**, measured.
///
/// `PARMCHK.DAT` declares `WEIGHT_X 10`, and parmchk2 does not charge it: every
/// improper it scores off a one-wildcard row comes to 3.0 (acetate's `X -o -c -o `)
/// and off a two-wildcard row to 6.0 (benzene's `X -X -ca-ha`, N-methylacetamide's
/// `X -X -c -o `), with the substitution penalties adding on top of that. So the
/// weight column is not the number the tool uses, and the tool — not the column —
/// is this crate's oracle. `WEIGHT_X3` (an `X` in the CENTRE slot) is unreachable:
/// neither `gaff.dat` nor `gaff2.dat` has such a row.
const IMPROPER_WILDCARD: f64 = 3.0;

/// The penalty charged for a dihedral no tier could reach: near-zero barrier.
///
/// High enough to land in [`PenaltyTier::Poor`](super::PenaltyTier::Poor) — the
/// parameter is a placeholder that says "there is no torsion here", and a caller
/// must be told so rather than handed a fabricated barrier.
const NO_TORSION_PENALTY: f64 = 99.0;

/// Which **harmonic** arity a substitution is being scored for: it selects the
/// penalty column, the per-arity default, and the inner-atom weight.
///
/// Only the two arities [`analogy`](Parmchk2Estimator::analogy) serves. A torsion
/// is not a third variant here because parmchk2 does not score it as one: its
/// penalty reads a different column for an inner atom than for an outer one, falls
/// back on an interpolation between two more when that column is blank, and refuses
/// the substitution outright for a type with no `CORR` row at all. That is
/// [`torsion_penalty`](Parmchk2Estimator::torsion_penalty)'s own rule, not a column
/// choice; an improper's is [`improper_score`](Parmchk2Estimator::improper_score)'s.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum Arity {
    /// Two atoms; no inner atom.
    Bond,
    /// Three atoms; the vertex is the inner atom (`WEIGHT_BA_CTR`, ×10).
    Angle,
}

/// Which substitutions a pass is allowed to make.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Substitution {
    /// None: every concrete slot must match outright (wildcard slots aside).
    None,
    /// `EQUA` rows only — free, because an equivalent type *is* the same type.
    Equivalent,
    /// `CORR` rows, scored.
    Corresponding,
}

/// What a torsion substitution costs, split so the inner pair can be preferred.
#[derive(Debug, Clone, Copy, Default)]
struct Score {
    /// The two inner (bonded-pair) atoms — what a torsion is actually about.
    inner: f64,
    /// Every atom.
    total: f64,
}

impl Parmchk2Estimator {
    // -- bonds and angles ---------------------------------------------------

    /// The analogy tiers over a bond / angle table: the lowest-penalty specific
    /// row, else a wildcard row that covers the term outright.
    ///
    /// `inner[i]` marks an inner atom (×10 weight). Returns `None` when no row of
    /// the right shape can be reached at all — the caller then falls to the
    /// empirical formulas.
    pub(super) fn analogy(
        &self,
        table: &[Candidate],
        query: &[&str],
        inner: &[bool],
        arity: Arity,
    ) -> Option<Estimate> {
        // Tiers 1, 2 and 4 at once: a specific row, cheapest first. An exact match
        // scores 0, an equivalent type the equtype weight, a corresponding type its
        // tabulated penalty — so "cheapest" IS the tier order.
        let mut best: Option<(f64, &Candidate)> = None;
        for cand in table {
            if cand.pattern.len() != query.len() || cand.wildcards() > 0 {
                continue;
            }
            let Some(penalty) = self.sequence_penalty(&cand.pattern, query, inner, arity) else {
                continue;
            };
            if best.is_none_or(|(p, _)| penalty < p) {
                best = Some((penalty, cand));
            }
        }
        if let Some((penalty, cand)) = best {
            return Some(Estimate::estimated(
                cand.params.clone(),
                Provenance::analogy(penalty, &cand.name),
            ));
        }

        // Tier 3: a generic row the table covers this term with is a PARAMETER.
        let covered = table.iter().find(|cand| {
            cand.pattern.len() == query.len() && cand.wildcards() > 0 && self.covers(cand, query)
        })?;
        Some(Estimate::covered(covered.params.clone(), &covered.name))
    }

    /// Whether a candidate matches `query` outright — every concrete slot equal
    /// (by type name or class), wildcard slots free — in either orientation.
    fn covers(&self, cand: &Candidate, query: &[&str]) -> bool {
        let matches = |forward: bool| {
            cand.pattern.iter().enumerate().all(|(at, pattern)| {
                let q = if forward {
                    query[at]
                } else {
                    query[query.len() - 1 - at]
                };
                is_wildcard(pattern) || self.same(pattern, q)
            })
        };
        matches(true) || matches(false)
    }

    /// Whether a pattern names this very atom type — by type name, or by the class
    /// the typifier resolved it to (OPLS keys its bonded forces on class).
    fn same(&self, pattern: &str, query: &str) -> bool {
        pattern == query || self.inputs.class_of(query) == Some(pattern)
    }

    /// Total substitution penalty of `pattern` against `query`, trying both
    /// orientations (proper terms are reversal-symmetric) and taking the smaller.
    /// `None` if neither orientation is a valid analog.
    fn sequence_penalty(
        &self,
        pattern: &[String],
        query: &[&str],
        inner: &[bool],
        arity: Arity,
    ) -> Option<f64> {
        let forward = self.oriented_penalty(pattern.iter(), query, inner, arity);
        let reversed = self.oriented_penalty(pattern.iter().rev(), query, inner, arity);
        match (forward, reversed) {
            (Some(a), Some(b)) => Some(a.min(b)),
            (Some(a), None) | (None, Some(a)) => Some(a),
            (None, None) => None,
        }
    }

    /// Penalty of one orientation: the per-end penalties with the inner-atom
    /// multiplier applied. `None` if any end is incompatible.
    fn oriented_penalty<'a, I>(
        &self,
        pattern: I,
        query: &[&str],
        inner: &[bool],
        arity: Arity,
    ) -> Option<f64>
    where
        I: Iterator<Item = &'a String>,
    {
        let mut total = 0.0;
        for ((pat, q), is_inner) in pattern.zip(query.iter()).zip(inner.iter()) {
            let base = self.end_penalty(pat, q, arity)?;
            let mult = if *is_inner {
                self.inner_multiplier(arity)
            } else {
                1.0
            };
            total += base * mult;
        }
        Some(total)
    }

    /// Substitution penalty for matching one endpoint `pattern` against one query
    /// atom type `q`.
    ///
    /// - exact type match or class match → `0`;
    /// - wildcard → the arity's default (a wildcard end is a weak analog *source*;
    ///   a wildcard row that COVERS the term never reaches this path);
    /// - equivalent type (`EQUA`) → the equtype weight;
    /// - corresponding type (`CORR`) → the tabulated penalty × weight;
    /// - same element, nothing tabulated → the arity default penalty;
    /// - different element, nothing tabulated → `None` (incompatible).
    fn end_penalty(&self, pattern: &str, q: &str, arity: Arity) -> Option<f64> {
        if is_wildcard(pattern) {
            return Some(self.default_penalty(arity));
        }
        if self.same(pattern, q) {
            return Some(0.0);
        }
        if let Some(p) = self.substitution_penalty(q, pattern, arity) {
            return Some(p);
        }
        // Element-based compatibility: same element ⇒ the arity's default price.
        match (self.element_of(q), self.element_of(pattern)) {
            (Some(a), Some(b)) if a == b => Some(self.default_penalty(arity)),
            _ => None,
        }
    }

    /// The substitution penalty for replacing query type `from` with pattern type
    /// `to` (or vice versa), from `PARMCHK.DAT`.
    fn substitution_penalty(&self, from: &str, to: &str, arity: Arity) -> Option<f64> {
        if self.substitutions.equivalent(from, to) {
            return Some(self.substitutions.weights.weight_equivalent);
        }
        let (column, default, weight) = self.penalty_columns(arity);
        let try_one = |a: &str, b: &str| -> Option<f64> {
            let row = self.substitutions.correspondence(a, b)?;
            Some(row.get(column).unwrap_or(default) * weight)
        };
        try_one(from, to).or_else(|| try_one(to, from))
    }

    /// The `CORR` column, default and weight one arity reads.
    fn penalty_columns(&self, arity: Arity) -> (ParmchkPenalty, f64, f64) {
        let w = &self.substitutions.weights;
        match arity {
            Arity::Angle => (ParmchkPenalty::AngleCentre, w.default_angle, w.weight_angle),
            Arity::Bond => (
                ParmchkPenalty::BondLength,
                w.default_bond_length,
                w.weight_bond_length,
            ),
        }
    }

    /// Inner-atom penalty multiplier (`WEIGHT_BA_CTR`, ×10). A torsion's inner pair is
    /// weighted by `WEIGHT_TOR_CTR` in `torsion_penalty`, where its column is chosen.
    fn inner_multiplier(&self, arity: Arity) -> f64 {
        match arity {
            Arity::Angle => self.substitutions.weights.weight_angle_centre,
            Arity::Bond => 1.0,
        }
    }

    /// The arity's default substitution penalty (`DEFAULT_BL` / `DEFAULT_BA`),
    /// charged when nothing more specific is tabulated.
    pub(super) fn default_penalty(&self, arity: Arity) -> f64 {
        let w = &self.substitutions.weights;
        match arity {
            Arity::Angle => w.default_angle,
            Arity::Bond => w.default_bond_length,
        }
    }

    // -- torsions -----------------------------------------------------------

    /// The torsion cascade, in parmchk2's own tier order.
    ///
    /// The caller has already failed to match the quartet against the table's
    /// wildcard-free rows, so tier 1 (an exact row) cannot fire here and is not
    /// re-tried.
    pub(super) fn torsion(&self, quartet: [&str; 4]) -> Option<Estimate> {
        let groups = &self.candidates.dihedrals;

        // Tier 2: equivalent-type substitution. Specific rows first, then the
        // fewest substitutions, then file order. A group that matches with NO
        // substitution is not an estimate — it belongs to tier 3.
        let equivalent = groups
            .iter()
            .filter(|group| group.pattern.len() == 4)
            .filter_map(|group| {
                let (subs, _) =
                    self.torsion_orientations(group, quartet, Substitution::Equivalent)?;
                (subs > 0).then_some((group.wildcards() > 0, subs, group))
            })
            .min_by_key(|(wild, subs, _)| (*wild, *subs));
        if let Some((_, _, group)) = equivalent {
            return Some(Estimate::estimated(
                group.params.clone(),
                Provenance::analogy(0.0, &group.name),
            ));
        }

        // Tier 3: a wildcard row that already matches — a parameter, not an estimate.
        if let Some(group) = groups.iter().find(|group| {
            group.pattern.len() == 4
                && group.wildcards() > 0
                && self
                    .torsion_orientations(group, quartet, Substitution::None)
                    .is_some()
        }) {
            return Some(Estimate::covered(group.params.clone(), &group.name));
        }

        // Tier 4: corresponding-type substitution, scored.
        let (_, _, group, penalty) = groups
            .iter()
            .filter(|group| group.pattern.len() == 4)
            .filter_map(|group| {
                let (_, score) =
                    self.torsion_orientations(group, quartet, Substitution::Corresponding)?;
                Some((
                    group.wildcards() > 0,
                    ordered(score.inner),
                    group,
                    score.total,
                ))
            })
            .min_by(|a, b| (a.0, a.1).cmp(&(b.0, b.1)))?;
        Some(Estimate::estimated(
            group.params.clone(),
            Provenance::analogy(penalty, &group.name),
        ))
    }

    /// The cheaper of a torsion row's two orientations against `quartet`, if either
    /// is a valid substitution under `rule`.
    ///
    /// "Cheaper" is the whole outcome — inner-pair score, then total score, then
    /// the number of substitutions — never the inner score alone with the
    /// forward reading kept on a tie. Reading the quartet from its other end
    /// swaps the two orientations, so only a comparison over everything an
    /// orientation reports picks the same outcome either way: when all three
    /// tie, the two readings report identical values and either may win.
    fn torsion_orientations(
        &self,
        group: &Candidate,
        quartet: [&str; 4],
        rule: Substitution,
    ) -> Option<(usize, Score)> {
        let forward: Vec<&str> = group.pattern.iter().map(String::as_str).collect();
        let mut reversed = forward.clone();
        reversed.reverse();
        [forward, reversed]
            .into_iter()
            .filter_map(|oriented| self.torsion_orientation(&oriented, quartet, rule))
            .min_by_key(|(subs, score)| (ordered(score.inner), ordered(score.total), *subs))
    }

    /// One orientation: the number of substitutions and what they cost, or `None`
    /// if a slot cannot legally be substituted under `rule`.
    fn torsion_orientation(
        &self,
        slots: &[&str],
        quartet: [&str; 4],
        rule: Substitution,
    ) -> Option<(usize, Score)> {
        let mut subs = 0usize;
        let mut score = Score::default();
        for (position, (slot, want)) in slots.iter().zip(quartet).enumerate() {
            if is_wildcard(slot) || self.same(slot, want) {
                continue; // a wildcard slot is free; an exact slot costs nothing
            }
            subs += 1;
            let inner = position == 1 || position == 2;
            match rule {
                Substitution::None => return None,
                Substitution::Equivalent => {
                    if !self.substitutions.equivalent(slot, want) {
                        return None;
                    }
                }
                Substitution::Corresponding => {
                    let penalty = self.torsion_penalty(slot, want, inner)?;
                    if inner {
                        score.inner += penalty;
                    }
                    score.total += penalty;
                }
            }
        }
        Some((subs, score))
    }

    /// What this cascade charges for replacing the row's `from` with the
    /// molecule's `to` in a torsion, or `None` when it will not make that
    /// substitution (its own rule, see the [module](super) table; not parmchk2's).
    ///
    /// An **inner** atom needs a tabulated correspondence and is weighted
    /// `WEIGHT_TOR_CTR` (×10). An **outer** atom may fall back on `DEFAULT_TOR`,
    /// but only for a type that takes part in the `CORR` table at all: `c3` has no
    /// `CORR` row anywhere, and parmchk2 will not substitute it for `cd` even at
    /// the default price.
    ///
    /// A type the substitution table has never heard of — every `opls_NNN` — has
    /// no correspondences to read, so the rule degrades to element compatibility at
    /// the default price. That is the honest floor for a non-GAFF force field, not
    /// a silent refusal to estimate.
    fn torsion_penalty(&self, from: &str, to: &str, inner: bool) -> Option<f64> {
        if self.substitutions.equivalent(from, to) {
            return Some(0.0);
        }
        let weights = &self.substitutions.weights;
        let (Some(zf), Some(zt)) = (
            self.substitutions.atomic_number(from),
            self.substitutions.atomic_number(to),
        ) else {
            // Not GAFF types: fall back on the element-based rule.
            return match (self.element_of(from), self.element_of(to)) {
                (Some(a), Some(b)) if a == b => Some(weights.default_torsion),
                _ => None,
            };
        };
        if zf != zt {
            return None;
        }
        let row = self.substitutions.correspondence(from, to);
        if inner {
            let row = row?;
            let centre = row.get(ParmchkPenalty::Torsion).unwrap_or_else(|| {
                // Untabulated: half the `ctor` column, half the row's overall
                // similarity.
                let outer = row
                    .get(ParmchkPenalty::TorsionCentre)
                    .unwrap_or(weights.default_torsion);
                let similarity = row
                    .get(ParmchkPenalty::Similarity)
                    .unwrap_or(weights.default_torsion);
                weights.default_fraction_1 * outer + weights.default_fraction_2 * similarity
            });
            return Some(centre * weights.weight_torsion_centre);
        }
        match row {
            Some(row) => Some(
                row.get(ParmchkPenalty::TorsionCentre)
                    .unwrap_or(weights.default_torsion),
            ),
            None => self
                .substitutions
                .substitutable(from)
                .then_some(weights.default_torsion),
        }
    }

    /// The last resort for a torsion nothing covers: a **near-zero barrier**, at a
    /// penalty that says so. Never a fabricated barrier.
    ///
    /// Only the interpolation seam reaches this: a caller reading a parameter table
    /// directly ([`Parmchk2Estimator::estimate`]) wants to hear that the torsion
    /// is missing, not to be handed a placeholder for it.
    pub(super) fn no_torsion(&self) -> Params {
        Params::from_iter([("k1", 0.0), ("k2", 0.0), ("k3", 0.0), ("k4", 0.0)])
    }

    /// The penalty charged for [`no_torsion`](Self::no_torsion).
    pub(super) fn no_torsion_penalty(&self) -> f64 {
        NO_TORSION_PENALTY
    }

    // -- impropers ----------------------------------------------------------

    /// The improper of a 3-coordinate centre: a table row, or parmchk2's default.
    ///
    /// Always produces something — an improper is a planarity term on a centre the
    /// table has already declared planar (the `improper_flag` column), so "no row"
    /// means "take the default", not "give up".
    pub(super) fn improper(&self, centre: &str, peripherals: [&str; 3]) -> Estimate {
        // An exact specific row is a parameter (the caller usually matched it
        // first; this is here so the cascade is complete on its own terms).
        if let Some(exact) = self.candidates.impropers.iter().find(|cand| {
            cand.pattern.len() == 4
                && cand.wildcards() == 0
                && self.improper_score(cand, centre, peripherals) == Some((0.0, false))
        }) {
            return Estimate::covered(exact.params.clone(), &exact.name);
        }

        // Only WILDCARD rows may be substituted into: a specific row is an exact
        // match or nothing.
        let best = self
            .candidates
            .impropers
            .iter()
            .filter(|cand| cand.pattern.len() == 4 && cand.wildcards() > 0)
            .filter_map(|cand| {
                let (score, substituted) = self.improper_score(cand, centre, peripherals)?;
                let concrete = 4 - cand.wildcards();
                Some((concrete, ordered(score), cand, score, substituted))
            })
            // Most specific row first (`X -o -c -o ` before `X -X -c -o `), then
            // cheapest — which is why methyl methacrylate's ester improper is the
            // 1.1 kcal/mol carboxyl term and not the 10.5 amide one.
            .min_by_key(|(concrete, ordered, ..)| (std::cmp::Reverse(*concrete), *ordered));

        match best {
            Some((_, _, cand, penalty, substituted)) => {
                let provenance = if substituted {
                    Provenance::analogy(penalty, &cand.name)
                } else {
                    Provenance::wildcard(penalty, &cand.name)
                };
                Estimate::estimated(cand.params.clone(), provenance)
            }
            None => {
                let (barrier, phase_deg, periodicity) = DEFAULT_IMPROPER;
                let params = Params::from_iter([
                    ("k", barrier),
                    ("periodicity", f64::from(periodicity)),
                    ("phase", phase_deg),
                ]);
                Estimate::estimated(params, Provenance::wildcard(0.0, ""))
            }
        }
    }

    /// One improper row against one centre + peripheral set: the total penalty and
    /// whether any concrete slot had to be substituted, or `None` if it cannot match.
    fn improper_score(
        &self,
        cand: &Candidate,
        centre: &str,
        peripherals: [&str; 3],
    ) -> Option<(f64, bool)> {
        let weights = &self.substitutions.weights;
        let mut substituted = false;

        let centre_slot = cand.pattern[2].as_str();
        let base = if is_wildcard(centre_slot) {
            weights.weight_wildcard_centre
        } else {
            if !self.same(centre_slot, centre) {
                substituted = true;
            }
            self.substitutions.similarity(centre_slot, centre)?
        };

        // The peripherals are a SET: the row's slots may take them in any order,
        // and the cascade pays for the cheapest assignment (parmchk2 itself
        // tries only the bond order; `typifier::gaff::improper`).
        let slots = [
            cand.pattern[0].as_str(),
            cand.pattern[1].as_str(),
            cand.pattern[3].as_str(),
        ];
        let mut best: Option<(f64, bool)> = None;
        for order in PERMUTATIONS {
            let mut total = base;
            let mut subs = substituted;
            let mut wild: Vec<&str> = Vec::new();
            let mut ok = true;
            for (slot, index) in slots.iter().zip(order) {
                let mol = peripherals[index];
                if is_wildcard(slot) {
                    total += IMPROPER_WILDCARD;
                    wild.push(mol);
                    continue;
                }
                match self.substitutions.similarity(slot, mol) {
                    Some(penalty) => {
                        total += penalty;
                        if !self.same(slot, mol) {
                            subs = true;
                        }
                    }
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            // A row that leaves TWO peripherals to wildcards is the generic planar
            // term; it applies only where the planarity is real, i.e. where at
            // least one of those two atoms is itself an improper centre. That is
            // the difference between N-methylacetamide's amide carbon, which takes
            // the 10.5 kcal/mol term (`n` is planar), and acetone's carbonyl
            // carbon, which does not (a second `c3` is not).
            if ok && wild.len() >= 2 && !wild.iter().any(|ty| self.is_improper_centre(ty)) {
                ok = false;
            }
            if !ok {
                continue;
            }
            if subs {
                total += weights.weight_improper;
            }
            if best.is_none_or(|(score, _)| total < score) {
                best = Some((total, subs));
            }
        }
        best
    }

    // -- empirical ----------------------------------------------------------

    /// Badger empirical bond `k` + the reference length, or `None` if either
    /// element is unknown or the pair is not tabulated.
    pub(super) fn empirical_bond(&self, types: &[String; 2]) -> Option<Estimate> {
        let e1 = self.element_of(&types[0])?;
        let e2 = self.element_of(&types[1])?;
        let ln_kij = self.empirical.bond_ln_k(e1, e2)?;
        let rref = self.empirical.bond_length(e1, e2)?;
        // The formula reproduces `gaff.dat`'s own un-halved `K`, which is
        // molrs's (LAMMPS's) `k` as it stands.
        let k = empirical::bond_k(ln_kij, rref, self.empirical.bond_power);
        Some(Estimate::estimated(
            Params::from_iter([("k", k), ("r0", rref)]),
            Provenance::empirical(self.default_penalty(Arity::Bond)),
        ))
    }

    /// Empirical angle: θ₀ = mean of the existing `A-B-A` and `C-B-C` angles
    /// sharing the centre `B`; `K_θ` from Eq. 5. `None` if the neighbour angles or
    /// the Z / C factors / reference lengths are unavailable.
    pub(super) fn empirical_angle(&self, types: &[String; 3]) -> Option<Estimate> {
        let (a, b, c) = (&types[0], &types[1], &types[2]);
        let theta_aba = self.existing_angle_theta0(a, b, a)?;
        let theta_cbc = self.existing_angle_theta0(c, b, c)?;
        let theta0 = empirical::angle_theta0(theta_aba, theta_cbc);

        let (ea, eb, ec) = (
            self.element_of(a)?,
            self.element_of(b)?,
            self.element_of(c)?,
        );
        // Un-halved `K`, molrs's (LAMMPS's) `k`; the formula wants θ₀ in radians.
        let k = empirical::angle_k(
            self.empirical.angle_z(ea)?,
            self.empirical.angle_c(eb)?,
            self.empirical.angle_z(ec)?,
            self.empirical.bond_length(ea, eb)?,
            self.empirical.bond_length(eb, ec)?,
            theta0.to_radians(),
        );
        Some(Estimate::estimated(
            Params::from_iter([("k", k), ("theta0", theta0)]),
            Provenance::empirical(self.default_penalty(Arity::Angle)),
        ))
    }

    /// θ₀ (degrees) of an existing `i-j-k` angle in the candidate table, if one
    /// matches by type or class in either orientation.
    fn existing_angle_theta0(&self, i: &str, j: &str, k: &str) -> Option<f64> {
        let query = [i, j, k];
        let inner = [false, true, false];
        self.candidates.angles.iter().find_map(|cand| {
            (cand.pattern.len() == 3
                && self
                    .sequence_penalty(&cand.pattern, &query, &inner, Arity::Angle)
                    .is_some())
            .then(|| cand.params.get("theta0").and_then(|v| v.as_num()))
            .flatten()
        })
    }
}

/// The six orders three peripherals can fill an improper row's three slots in.
const PERMUTATIONS: [[usize; 3]; 6] = [
    [0, 1, 2],
    [0, 2, 1],
    [1, 0, 2],
    [1, 2, 0],
    [2, 0, 1],
    [2, 1, 0],
];

/// A total order over a penalty, so scores can be compared without a `NaN` branch.
/// Penalties are sums of tabulated non-negative numbers; none can be `NaN`.
fn ordered(score: f64) -> u64 {
    score.to_bits()
}
