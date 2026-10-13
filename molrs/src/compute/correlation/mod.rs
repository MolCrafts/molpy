//! Time-correlation functions: the one engine behind every origin-averaged
//! correlation in molrs.
//!
//! A time correlation of an `(n_frames, n_entities, n_components)` series is
//! `C(t) = norm · Σ_i Σ_c w_c Σ_τ a_ic(τ) · b_ic(τ+t)` over time origins `τ`.
//! Three functions cover every case that arises from a trajectory:
//!
//! | Function | `C(t)` | Used by |
//! |---|---|---|
//! | [`autocorrelate`] | `⟨a(τ) · a(τ+t)⟩` | [`Acf`], `Vacf`, `GreenKuboDiffusion`, `GreenKuboConductivity`, `DebyeRelaxation`, `IrFlux`, `RamanTensor`, `Oacf`, hydrogen-bond lifetimes |
//! | [`cross_correlate`] | `⟨a(τ) · b(τ+t)⟩` | `DipoleRateCross`, `VcdCrossFlux`, `RoaCrossTensor` |
//! | [`displacement_correlate`] | `⟨(a(τ+t) − a(τ)) · (b(τ+t) − b(τ))⟩` | `Msd` (window mode), `EinsteinConductivity`, `OnsagerCorrelation` |
//!
//! All three return a [`TimeCorrelation`]: the curve, the time-origin count
//! per lag, and the [`Normalization`] and [`CorrelationKind`] it was built
//! with. The estimator settings — longest lag, mean subtraction,
//! normalization, origin stride, per-component weights, per-entity rows —
//! live in [`TimeCorrelationSpec`]. The derived quantities — `normalized`,
//! `integral` / `integrated_time`, `exponential_fit`, `linear_fit`,
//! `diffusion_coefficient`, `spectrum` — are methods on the result that
//! delegate to the [`fitting`](crate::compute::fitting) and
//! [`spectroscopy`](crate::compute::spectroscopy) `Fit`s.
//!
//! Every column goes through one zero-padded Wiener–Khinchin kernel
//! ([`molrs::signal::LinearCorrelator`](crate::signal::LinearCorrelator)),
//! so a quantity-specific front end only builds its series and names its
//! settings; the arithmetic is shared and tested once.
//!
//! The front ends above sit in their own folders (`dynamics`, `transport`,
//! `msd`, `spectroscopy`, `hbond`); [`Acf`], the bare engine as a
//! [`Compute`](crate::compute::Compute), lives here.

mod acf;
mod derived;
mod displacement;
mod engine;
pub(crate) mod result;

pub use acf::{Acf, AcfArgs};
pub(crate) use derived::einstein_diffusion_coefficient;
pub use displacement::displacement_correlate;
pub use engine::{TimeCorrelationSpec, autocorrelate, cross_correlate};
pub use result::{CorrelationKind, Normalization, TimeCorrelation, lag_times};
