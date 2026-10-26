//! The 1-4 semantics a `lj/charmm` style declares with its string param
//! `one_four` ([`OneFour`]); what each means, and how the compiler prices it,
//! is [`crate::ff::compile`]'s 1-4 exceptions.

use crate::ff::ir::Params;

/// The `lj/charmm` style param naming the 1-4 semantics.
pub const ONE_FOUR: &str = "one_four";
/// LAMMPS's semantics: `special_bonds` 1-4 pairs at the regular `epsilon` / `sigma`.
pub const ONE_FOUR_REGULAR: &str = "regular";
/// `special_bonds` 1-4 pairs at `epsilon14` / `sigma14`.
pub const ONE_FOUR_EPSILON14: &str = "epsilon14";
/// Every value [`ONE_FOUR`] may take.
pub const ONE_FOUR_VALUES: [&str; 2] = [ONE_FOUR_REGULAR, ONE_FOUR_EPSILON14];

/// The 1-4 semantics of a `lj/charmm` style.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OneFour {
    /// `"regular"` (or absent).
    Regular,
    /// `"epsilon14"`.
    Epsilon14,
}

impl OneFour {
    /// The `one_four` of a style's params.
    ///
    /// # Errors
    ///
    /// A value other than `"regular"` and `"epsilon14"`, or a number.
    pub fn of(style: &Params) -> Result<Self, String> {
        match (
            style.get(ONE_FOUR).and_then(|v| v.as_text()),
            style.get(ONE_FOUR).and_then(|v| v.as_num()),
        ) {
            (None, None) => Ok(Self::Regular),
            (Some(ONE_FOUR_REGULAR), _) => Ok(Self::Regular),
            (Some(ONE_FOUR_EPSILON14), _) => Ok(Self::Epsilon14),
            (Some(other), _) => Err(format!(
                "lj/charmm: one_four = {other:?}; it is \"{ONE_FOUR_REGULAR}\" or \
                 \"{ONE_FOUR_EPSILON14}\""
            )),
            (None, Some(v)) => Err(format!(
                "lj/charmm: one_four = {v} is a number; it is \"{ONE_FOUR_REGULAR}\" or \
                 \"{ONE_FOUR_EPSILON14}\""
            )),
        }
    }
}

/// Whether a `lj/charmm` row has 1-4 parameters other than its regular ones.
pub(crate) fn has_own_one_four(p: &Params) -> bool {
    let differs = |k14: &str, k: &str| {
        p.get(k14)
            .and_then(|v| v.as_num())
            .is_some_and(|v| Some(v) != p.get(k).and_then(|v| v.as_num()))
    };
    differs("epsilon14", "epsilon") || differs("sigma14", "sigma")
}
