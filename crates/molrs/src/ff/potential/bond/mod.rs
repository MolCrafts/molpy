//! Bond forms: each built-in bond style's energy as a [`ScalarForm`] of the
//! bond length `r`, priced by [`ScalarBonded`].
//!
//! [`ScalarForm`]: crate::ff::potential::form_kernel::ScalarForm
//! [`ScalarBonded`]: crate::ff::potential::form_kernel::ScalarBonded

pub(crate) mod class2;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod morse;
pub(crate) mod uff;

pub use class2::Class2;
pub use harmonic::Harmonic;
pub use mmff::Mmff;
pub use morse::Morse;
pub use uff::Uff;
