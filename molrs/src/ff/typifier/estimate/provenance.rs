//! What an estimate cost, and how it was reached.

use crate::ff::ir::Params;

/// Penalty tier for an estimate, following the CGenFF confidence bands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PenaltyTier {
    /// `penalty < 10`: reliable.
    Reliable,
    /// `10 ≤ penalty ≤ 50`: use with caution, manual review advised.
    Caution,
    /// `penalty > 50`: poor, needs optimization.
    Poor,
}

impl PenaltyTier {
    /// Classify a total penalty into the CGenFF bands (`<10` / `10–50` / `>50`).
    /// The boundaries are inclusive on the lower band edge (`10.0 → Caution`,
    /// `50.0 → Caution`).
    pub fn of(penalty: f64) -> Self {
        if penalty < 10.0 {
            Self::Reliable
        } else if penalty <= 50.0 {
            Self::Caution
        } else {
            Self::Poor
        }
    }
}

/// How a missing parameter was produced.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EstimateMethod {
    /// Copied verbatim from the nearest analog in the parameter table, reached by
    /// substituting an atom type for an equivalent or corresponding one.
    Analogy,
    /// Computed from an empirical formula (Badger bond `k`, mean-of-neighbours
    /// θ₀, the Wang2004 Eq. 5 angle `K_θ`).
    Empirical,
    /// Copied from a generic wildcard term (`X -c3-c3-X `).
    GenericWildcard,
}

impl EstimateMethod {
    /// The provenance string written onto the term (`estimate_method`).
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Analogy => "analogy",
            Self::Empirical => "empirical",
            Self::GenericWildcard => "generic-wildcard",
        }
    }

    /// The method an `estimate_method` string names, or `None`.
    pub fn parse(text: &str) -> Option<Self> {
        [Self::Analogy, Self::Empirical, Self::GenericWildcard]
            .into_iter()
            .find(|method| method.as_str() == text)
    }
}

/// How an estimated term was produced, and what it cost.
///
/// An estimated parameter that does not say it *is* one cannot be audited, so
/// every estimate the estimator produces carries a [`Provenance`]: the penalty
/// parmchk2's weight table charges for it, the [`PenaltyTier`] that lands it in,
/// the analog it was copied from, and how ([`EstimateMethod`]).
///
/// # The provenance convention
///
/// [`Provenance::apply_to`] writes four keys onto an estimated term's
/// [`Params`], and every consumer (the OPLS assign seam, the GAFF typifier,
/// the parmchk2 oracle test) reads the same four:
///
/// | key | type | meaning |
/// |---|---|---|
/// | `estimated` | numeric `1.0` | flag: this term was estimated, not matched |
/// | `estimate_penalty` | numeric | total additive penalty (f64) |
/// | `estimate_method` | string | `"analogy"`, `"empirical"`, or `"generic-wildcard"` |
/// | `estimate_analog` | string | source type name copied from, or `""` |
///
/// A term the table covered outright carries **none** of them — see [`Estimate`].
#[derive(Debug, Clone, PartialEq)]
pub struct Provenance {
    /// The additive penalty score charged for the substitutions made.
    pub penalty: f64,
    /// Which tier of the cascade produced the value.
    pub method: EstimateMethod,
    /// The table row it was copied from (`X-X-ca-ha`), or empty for a formula or
    /// a default.
    pub analog: String,
}

impl Provenance {
    /// An estimate reached by analogy: a row copied after substituting a type.
    pub fn analogy(penalty: f64, analog: impl Into<String>) -> Self {
        Self {
            penalty,
            method: EstimateMethod::Analogy,
            analog: analog.into(),
        }
    }

    /// An estimate copied from a generic wildcard term.
    pub fn wildcard(penalty: f64, analog: impl Into<String>) -> Self {
        Self {
            penalty,
            method: EstimateMethod::GenericWildcard,
            analog: analog.into(),
        }
    }

    /// An estimate computed from an empirical formula: no analog to name.
    pub fn empirical(penalty: f64) -> Self {
        Self {
            penalty,
            method: EstimateMethod::Empirical,
            analog: String::new(),
        }
    }

    /// The confidence band this penalty lands in.
    pub fn tier(&self) -> PenaltyTier {
        PenaltyTier::of(self.penalty)
    }

    /// The four provenance keys, in the order the module doc lists them:
    /// the style table's [`PROVENANCE_COLUMNS`]. Metadata, not parameters:
    /// they price nothing ([`is_unpriced_column`]), so every writer skips
    /// them and [`ForceField::merge`] ignores them.
    ///
    /// [`PROVENANCE_COLUMNS`]: crate::ff::ir::PROVENANCE_COLUMNS
    /// [`is_unpriced_column`]: crate::ff::ir::is_unpriced_column
    /// [`ForceField::merge`]: crate::ff::forcefield::ForceField::merge
    pub const KEYS: [&'static str; 4] = crate::ff::ir::PROVENANCE_COLUMNS;

    /// Write the four provenance keys onto an estimated term's params.
    pub fn apply_to(&self, params: &mut Params) {
        params.set(Self::KEYS[0], 1.0);
        params.set(Self::KEYS[1], self.penalty);
        params.set(Self::KEYS[2], self.method.as_str());
        params.set(Self::KEYS[3], self.analog.as_str());
    }

    /// The provenance `params` carry: the inverse of
    /// [`apply_to`](Self::apply_to). `None` for a term that was matched,
    /// not estimated (no `estimated` key), or whose `estimate_method` is not
    /// one this module writes.
    pub fn from_params(params: &Params) -> Option<Self> {
        params.get(Self::KEYS[0]).and_then(|v| v.as_num())?;
        let method = EstimateMethod::parse(params.get(Self::KEYS[2]).and_then(|v| v.as_text())?)?;
        Some(Self {
            penalty: params
                .get(Self::KEYS[1])
                .and_then(|v| v.as_num())
                .unwrap_or(0.0),
            method,
            analog: params
                .get(Self::KEYS[3])
                .and_then(|v| v.as_text())
                .unwrap_or_default()
                .to_owned(),
        })
    }
}

/// What the cascade found for one term.
///
/// The two variants are the load-bearing distinction of the whole cascade. A term
/// a **wildcard row** covers (`X -c3-c3-X ` over ethane's `hc-c3-c3-hc`) is
/// [`Covered`](Self::Covered): a *parameter*, not an estimate. LEaP finds it in
/// the table and parmchk2 stays silent about it, so nothing is substituted and no
/// penalty is charged. Collapse the two and an estimator fabricates an analogy for
/// the ~145 terms per molecule the table already answers — plausible numbers that
/// silently disagree with AMBER on most of the molecule.
#[derive(Debug, Clone)]
pub enum Estimate {
    /// The table covers this term outright. Nothing was estimated.
    Covered {
        /// The parameters, in the candidate table's own units and key names.
        params: Params,
        /// The row that covers it (`X-c3-c3-X`).
        analog: String,
    },
    /// No row covered the term; these parameters had to be estimated.
    Estimated {
        /// The parameters, in the candidate table's own units and key names.
        params: Params,
        /// What it cost, and how it was reached.
        provenance: Provenance,
    },
}

impl Estimate {
    /// A term the table covers outright: parameters, and nothing to declare.
    pub fn covered(params: Params, analog: impl Into<String>) -> Self {
        Self::Covered {
            params,
            analog: analog.into(),
        }
    }

    /// A term whose parameters had to be estimated.
    pub fn estimated(params: Params, provenance: Provenance) -> Self {
        Self::Estimated { params, provenance }
    }

    /// The parameters, whichever way they were reached.
    pub fn params(&self) -> &Params {
        match self {
            Self::Covered { params, .. } | Self::Estimated { params, .. } => params,
        }
    }

    /// What the estimate cost, or `None` when the table covered the term.
    pub fn provenance(&self) -> Option<&Provenance> {
        match self {
            Self::Covered { .. } => None,
            Self::Estimated { provenance, .. } => Some(provenance),
        }
    }

    /// The params with the provenance convention written onto them.
    ///
    /// A [`Covered`](Self::Covered) term is reported as a
    /// [`GenericWildcard`](EstimateMethod::GenericWildcard) estimate at penalty 0:
    /// a caller that reaches the *interpolation seam* has, by definition, already
    /// failed to match the term against its own tables, so from its point of view
    /// the generic row is a fallback and it needs to be told so. A caller that
    /// reads the parameter table itself
    /// ([`GaffTypifier`](crate::ff::typifier::GaffTypifier)) matches on [`Estimate`]
    /// instead and keeps the distinction, which is what the parmchk2 oracle
    /// demands of it.
    pub fn into_params(self) -> Params {
        let (mut params, provenance) = match self {
            Self::Covered { params, analog } => (params, Provenance::wildcard(0.0, analog)),
            Self::Estimated { params, provenance } => (params, provenance),
        };
        provenance.apply_to(&mut params);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn penalty_tiers_classify_at_boundaries() {
        assert_eq!(PenaltyTier::of(0.0), PenaltyTier::Reliable);
        assert_eq!(PenaltyTier::of(9.999), PenaltyTier::Reliable);
        assert_eq!(PenaltyTier::of(10.0), PenaltyTier::Caution);
        assert_eq!(PenaltyTier::of(50.0), PenaltyTier::Caution);
        assert_eq!(PenaltyTier::of(50.0001), PenaltyTier::Poor);
        assert_eq!(PenaltyTier::of(100.0), PenaltyTier::Poor);
    }

    #[test]
    fn a_covered_term_carries_no_provenance() {
        let covered = Estimate::covered(Params::from_iter([("k", 1.0)]), "X-c3-c3-X");
        assert!(
            covered.provenance().is_none(),
            "a wildcard row is a parameter, not an estimate"
        );
        // …but a consumer at the interpolation seam is still told where it came from.
        let params = covered.into_params();
        assert_eq!(params.get("estimated").and_then(|v| v.as_num()), Some(1.0));
        assert_eq!(
            params.get("estimate_penalty").and_then(|v| v.as_num()),
            Some(0.0)
        );
    }

    #[test]
    fn from_params_inverts_apply_to() {
        let written = Provenance::analogy(2.5, "c3-oh");
        let mut params = Params::from_iter([("k", 300.9)]);
        written.apply_to(&mut params);
        assert_eq!(Provenance::from_params(&params), Some(written));
    }

    #[test]
    fn a_matched_term_reads_as_no_provenance() {
        let params = Params::from_iter([("k", 300.9)]);
        assert_eq!(Provenance::from_params(&params), None);
    }

    #[test]
    fn provenance_writes_the_four_keys() {
        let mut params = Params::from_iter([("k", 300.9)]);
        Provenance::analogy(2.5, "c3-oh").apply_to(&mut params);
        assert_eq!(params.get("estimated").and_then(|v| v.as_num()), Some(1.0));
        assert_eq!(
            params.get("estimate_penalty").and_then(|v| v.as_num()),
            Some(2.5)
        );
        let strings: Vec<(&str, &str)> = params
            .iter()
            .filter_map(|(k, v)| v.as_text().map(|s| (k, s)))
            .collect();
        assert!(strings.contains(&("estimate_method", "analogy")));
        assert!(strings.contains(&("estimate_analog", "c3-oh")));
    }
}
