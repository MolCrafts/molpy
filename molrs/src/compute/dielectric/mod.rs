//! Dielectric series and the static permittivity.
//!
//! | Compute | Data | Output |
//! |---|---|---|
//! | [`DipoleMoment`] | positions (or velocities) `T × N × 3`, charges, optional [`Partition`](crate::compute::Partition) | one `M(t)` series per group |
//! | [`StaticDielectric`] | dipole series, volume, temperature, [`EwaldBoundary`] | ε(0), isotropic and per axis |
//!
//! The current `Ṁ` is the [`Derivative`](crate::compute::Derivative) of the
//! dipole series; divided by the volume it is the current density `J` of the
//! Green–Kubo route. The frequency-dependent permittivity is one of the four
//! routes in `spectroscopy` ([`EinsteinHelfandSpectrum`](crate::compute::EinsteinHelfandSpectrum),
//! …), and the Einstein–Helfand conductivity is [`Msd`](crate::compute::Msd)
//! on the dipole series followed by [`LinearFit`](crate::compute::LinearFit).
//!
//! Every permittivity and conductivity prefactor is formed by one
//! crate-internal type, `DielectricPrefactor`, from the volume and temperature
//! given to `compute` and the [`UnitPreset`](molrs::core::UnitPreset) given to
//! `new`.

mod dipole_moment;
mod ewald_boundary;
mod prefactor;
mod static_dielectric;

pub use dipole_moment::{DipoleMoment, DipoleMomentInput, DipoleMomentResult};
pub use ewald_boundary::EwaldBoundary;
pub(crate) use prefactor::DielectricPrefactor;
pub use static_dielectric::{StaticDielectric, StaticDielectricInput, StaticDielectricResult};
