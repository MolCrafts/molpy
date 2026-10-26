//! Improper forms: each built-in improper style's energy as a
//! [`ScalarForm`] of the signed dihedral of its stored order (priced by
//! [`ScalarBonded`]), or — an out-of-plane angle (MMFF's Wilson angle, UFF's
//! inversion) — as a [`CompoundForm`] of the four positions (priced by
//! [`CompoundTerms`]). [`ImproperDistance`] is the ETKDG planarity term, a
//! kernel of explicit quadruples no force field registers.
//!
//! [`ScalarForm`]: crate::ff::potential::form_kernel::ScalarForm
//! [`CompoundForm`]: crate::ff::potential::form_kernel::CompoundForm
//! [`ScalarBonded`]: crate::ff::potential::form_kernel::ScalarBonded
//! [`CompoundTerms`]: crate::ff::potential::form_kernel::CompoundTerms

pub(crate) mod cvff;
mod distance;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod periodic;
pub(crate) mod uff;

pub use cvff::Cvff;
pub use distance::ImproperDistance;
pub use harmonic::Harmonic;
pub use mmff::Mmff;
pub use periodic::Periodic;
pub use uff::Uff;
