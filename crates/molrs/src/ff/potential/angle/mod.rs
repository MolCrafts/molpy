//! Angle forms: each built-in angle style's energy as a [`ScalarForm`] of
//! the angle `θ` (priced by [`ScalarBonded`]), or — a style that also reads
//! a distance (`charmm`'s Urey–Bradley, MMFF's stretch-bend) — as a
//! [`CompoundForm`] of the three positions (priced by [`CompoundTerms`]).
//!
//! [`ScalarForm`]: crate::ff::potential::form_kernel::ScalarForm
//! [`CompoundForm`]: crate::ff::potential::form_kernel::CompoundForm
//! [`ScalarBonded`]: crate::ff::potential::form_kernel::ScalarBonded
//! [`CompoundTerms`]: crate::ff::potential::form_kernel::CompoundTerms

pub(crate) mod charmm;
pub(crate) mod class2;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod uff;

pub use charmm::Charmm;
pub use class2::Class2;
pub use harmonic::Harmonic;
pub use mmff::{Mmff, MmffStretchBend};
pub use uff::Uff;
