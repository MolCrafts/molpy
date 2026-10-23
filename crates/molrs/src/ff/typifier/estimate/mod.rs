//! What GAFF's missing-parameter estimate is made of: parmchk2's tables
//! (equivalence, correspondence, the empirical constants — [`EmpiricalSet`]),
//! the bonded-term query ([`BondedTerm`]), the empirical formulas
//! ([`empirical`]) and an estimate's provenance ([`Provenance`]). The search
//! that uses them is GAFF's own reproduction of parmchk2
//! (`gaff::{analog, torsion, improper}`); there is no second estimator.

pub(crate) mod empirical;
mod provenance;
mod tables;
mod term;

pub use provenance::{Estimate, EstimateMethod, PenaltyTier, Provenance};
pub use tables::EmpiricalSet;
pub use term::BondedTerm;
