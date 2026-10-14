//! Dielectric and Raman spectra and their consistency checks — the WASM face
//! of `molrs::compute::spectroscopy`.
//!
//! The four ε(ω) routes share one configuration (`units`, then the optional
//! `window`, `pad`, `epsilonInf`) and one data object: the route's
//! correlation curve and the state it was sampled in.
//!
//! ```js
//! const acf = new DipoleAcf(400).compute({ dipoles, dt: 2.0 });
//! const eps = new DipoleAutocorrelationSpectrum("real").compute({
//!     correlation: acf, volume, temperature: 300, boundary: "tin_foil",
//! });
//! const check = new KramersKronig().compute({ spectrum: eps });
//! console.log(check.toJSON().passed);
//! ```

use molrs::compute::{self as mc, Compute, ComputeError, Described, ParamValue};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::{Given, fixed, unit_preset};
use super::handle::{described, output};
use super::time_series::TimeCorrelation;
use super::wire::Fields;

output! {
    /// ε(ω) on a frequency grid: `frequencies`, `eps_real`, `eps_imag`,
    /// `epsilon_inf`. The input of `KramersKronig` and `RouteAgreement`.
    DielectricSpectrumResult => mc::DielectricSpectrumResult
}

output! {
    /// A Kramers–Kronig check: the `recovered` real part, `mae`, `tolerance`,
    /// `passed`.
    KramersKronigCheck => mc::KramersKronigCheck
}

output! {
    /// A conductivity sum-rule check: `integral`, `expected`,
    /// `relative_error`, `passed`.
    SumRuleCheck => mc::SumRuleCheck
}

output! {
    /// Agreement between ε(ω) routes: `pairwise` RMS, `max_rms`, `passed`.
    RouteAgreementCheck => mc::RouteAgreementCheck
}

output! {
    /// The isotropic and anisotropic parts of a polarizability series.
    PolarizabilityPartsResult => mc::PolarizabilityPartsResult
}

output! {
    /// A Raman spectrum: `frequencies_cm1`, the isotropic / anisotropic (and,
    /// when averaged, parallel / perpendicular) intensities.
    RamanSpectrumResult => mc::RamanSpectrumResult
}

/// The `{ correlation, volume, temperature, boundary }` data of every route.
fn route_data(
    data: &JsValue,
    owner: &'static str,
) -> Result<(TimeCorrelation, F, F, mc::EwaldBoundary), ComputeError> {
    let data = Fields::new(data, owner)?;
    Ok((
        data.handle::<TimeCorrelation>("correlation", "TimeCorrelation")?,
        data.scalar("volume")?,
        data.scalar("temperature")?,
        data.choice("boundary")?,
    ))
}

/// One ε(ω) route class: `units` plus the shared optional configuration, and
/// its own optional parameters through `$extra`.
macro_rules! route {
    (
        $(#[$meta:meta])*
        $name:ident $(, $extra:ident: $extra_ty:ty => |$c:ident, $v:ident| $set:expr)?
    ) => {
        $(#[$meta])*
        #[wasm_bindgen]
        pub struct $name {
            inner: mc::$name,
        }

        #[wasm_bindgen]
        impl $name {
            /// `units` is a registered unit preset name; `window` a
            /// [`Window`](mc::Window) name, `pad` the zero-padding factor.
            #[wasm_bindgen(constructor)]
            pub fn new(
                units: &str,
                window: Option<String>,
                pad: Option<usize>,
                epsilon_inf: Option<F>,
                $($extra: Option<$extra_ty>,)?
            ) -> Result<$name, JsError> {
                let inner = mc::$name::new(unit_preset(units)?)
                    .given(window, |c, v| Ok(c.with_window(v.parse()?)))?
                    .given(pad, |c, v| c.with_pad(v))?
                    .given(epsilon_inf, |c, v| Ok(c.with_epsilon_inf(v)))?
                    $(.given($extra, |$c, $v| $set)?)?;
                Ok(Self { inner })
            }

            /// `{ correlation, volume, temperature, boundary }`.
            pub fn compute(&self, data: &JsValue) -> Result<DielectricSpectrumResult, JsError> {
                let (correlation, volume, temperature, boundary) =
                    route_data(data, stringify!($name))?;
                let input = mc::DielectricSpectrumInput {
                    correlation: &correlation.inner,
                    volume,
                    temperature,
                    boundary,
                };
                Ok(self.inner.compute(input)?.into())
            }
        }
    };
}

route! {
    /// ε(ω) from the dipole mean squared displacement (Einstein–Helfand).
    EinsteinHelfandSpectrum
}

route! {
    /// ε(ω) from the current autocorrelation (Green–Kubo).
    GreenKuboSpectrum
}

route! {
    /// ε(ω) from the dipole–dipole-rate cross correlation.
    DipoleRateCrossSpectrum
}

route! {
    /// ε(ω) from the dipole autocorrelation; `plateau` (`[start, end]`)
    /// subtracts the curve's mean over that window first.
    DipoleAutocorrelationSpectrum,
    plateau: Vec<F> => |c, v| {
        let [start, end] = fixed::<F, 2>(&v, "plateau window [start, end]")?;
        Ok(c.with_plateau(mc::Plateau::new(start, end)?))
    }
}

/// The real number `name` an instance of `T` holds (its core default when the
/// instance is fresh).
fn tunable<T: Described>(compute: &T, name: &str) -> Option<F> {
    compute
        .tunables()
        .into_iter()
        .find_map(|(param, value)| match value {
            ParamValue::Float(v) if param == name => Some(v),
            _ => None,
        })
}

/// Kramers–Kronig consistency of an ε(ω).
#[wasm_bindgen]
pub struct KramersKronig {
    inner: mc::KramersKronig,
}

#[wasm_bindgen]
impl KramersKronig {
    /// The tolerance is set as a pair: a tolerance left out keeps the core's.
    #[wasm_bindgen(constructor)]
    pub fn new(
        relative_tolerance: Option<F>,
        absolute_tolerance: Option<F>,
    ) -> Result<KramersKronig, JsError> {
        let fresh = mc::KramersKronig::new();
        if relative_tolerance.is_none() && absolute_tolerance.is_none() {
            return Ok(Self { inner: fresh });
        }
        let held = |name| {
            tunable(&fresh, name).ok_or(ComputeError::OutOfRange {
                field: "KramersKronig tolerance",
                value: name.to_string(),
            })
        };
        let relative = match relative_tolerance {
            Some(v) => v,
            None => held("relative_tolerance")?,
        };
        let absolute = match absolute_tolerance {
            Some(v) => v,
            None => held("absolute_tolerance")?,
        };
        Ok(Self {
            inner: fresh.with_tolerance(relative, absolute)?,
        })
    }

    /// `{ spectrum }`: a `DielectricSpectrumResult`.
    pub fn compute(&self, data: &JsValue) -> Result<KramersKronigCheck, JsError> {
        let data = Fields::new(data, "KramersKronig")?;
        let spectrum =
            data.handle::<DielectricSpectrumResult>("spectrum", "DielectricSpectrumResult")?;
        let input = mc::KramersKronigInput {
            spectrum: &spectrum.inner,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// The f-sum rule of a conductivity spectrum.
#[wasm_bindgen]
pub struct ConductivitySumRule {
    inner: mc::ConductivitySumRule,
}

#[wasm_bindgen]
impl ConductivitySumRule {
    #[wasm_bindgen(constructor)]
    pub fn new(units: &str, tolerance: Option<F>) -> Result<ConductivitySumRule, JsError> {
        let inner = mc::ConductivitySumRule::new(unit_preset(units)?)
            .given(tolerance, |c, v| c.with_tolerance(v))?;
        Ok(Self { inner })
    }

    /// `{ frequencies, conductivity, current_sq_mean, volume, temperature }`.
    pub fn compute(&self, data: &JsValue) -> Result<SumRuleCheck, JsError> {
        let data = Fields::new(data, "ConductivitySumRule")?;
        let frequencies = data.reals("frequencies")?;
        let conductivity = data.reals("conductivity")?;
        let input = mc::SumRuleInput {
            frequencies: frequencies.view(),
            conductivity: conductivity.view(),
            current_sq_mean: data.scalar("current_sq_mean")?,
            volume: data.scalar("volume")?,
            temperature: data.scalar("temperature")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Agreement of two or more ε(ω) routes on one frequency grid.
#[wasm_bindgen]
pub struct RouteAgreement {
    inner: mc::RouteAgreement,
}

#[wasm_bindgen]
impl RouteAgreement {
    /// `part` is a [`PermittivityPart`](mc::PermittivityPart) name.
    #[wasm_bindgen(constructor)]
    pub fn new(part: Option<String>, tolerance: Option<F>) -> Result<RouteAgreement, JsError> {
        let inner = mc::RouteAgreement::new()
            .given(part, |c, v| Ok(c.with_part(v.parse()?)))?
            .given(tolerance, |c, v| c.with_tolerance(v))?;
        Ok(Self { inner })
    }

    /// `{ spectra }`: an array of `DielectricSpectrumResult`.
    pub fn compute(&self, data: &JsValue) -> Result<RouteAgreementCheck, JsError> {
        let data = Fields::new(data, "RouteAgreement")?;
        let held =
            data.handles::<DielectricSpectrumResult>("spectra", "DielectricSpectrumResult")?;
        let spectra: Vec<&mc::DielectricSpectrumResult> = held.iter().map(|s| &*s.inner).collect();
        let input = mc::RouteAgreementInput { spectra: &spectra };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Isotropic / anisotropic split of a polarizability series.
#[wasm_bindgen]
pub struct PolarizabilityParts {
    inner: mc::PolarizabilityParts,
}

#[wasm_bindgen]
impl PolarizabilityParts {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> PolarizabilityParts {
        Self {
            inner: mc::PolarizabilityParts::new(),
        }
    }

    /// `{ alpha }`: `T × 6`, Voigt order `[xx, yy, zz, xy, xz, yz]`.
    pub fn compute(&self, data: &JsValue) -> Result<PolarizabilityPartsResult, JsError> {
        let data = Fields::new(data, "PolarizabilityParts")?;
        let alpha = data.matrix("alpha", Some(6))?;
        let input = mc::PolarizabilityPartsInput {
            alpha: alpha.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// A Raman spectrum from the isotropic and anisotropic correlations.
#[wasm_bindgen]
pub struct RamanSpectrum {
    inner: mc::RamanSpectrum,
}

#[wasm_bindgen]
impl RamanSpectrum {
    #[wasm_bindgen(constructor)]
    pub fn new(
        window: Option<String>,
        pad: Option<usize>,
        incident_frequency: Option<F>,
        averaged: Option<bool>,
    ) -> Result<RamanSpectrum, JsError> {
        let inner = mc::RamanSpectrum::new()
            .given(window, |c, v| Ok(c.with_window(v.parse()?)))?
            .given(pad, |c, v| c.with_pad(v))?
            .given(incident_frequency, |c, v| c.with_incident_frequency(v))?
            .given(averaged, |c, v| Ok(c.with_averaged(v)))?;
        Ok(Self { inner })
    }

    /// `{ isotropic, anisotropic, temperature? }`: two `TimeCorrelation`s.
    pub fn compute(&self, data: &JsValue) -> Result<RamanSpectrumResult, JsError> {
        let data = Fields::new(data, "RamanSpectrum")?;
        let isotropic = data.handle::<TimeCorrelation>("isotropic", "TimeCorrelation")?;
        let anisotropic = data.handle::<TimeCorrelation>("anisotropic", "TimeCorrelation")?;
        let input = mc::RamanSpectrumInput {
            isotropic: &isotropic.inner,
            anisotropic: &anisotropic.inner,
            temperature: data.opt_scalar("temperature")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    EinsteinHelfandSpectrum => mc::EinsteinHelfandSpectrum,
    GreenKuboSpectrum => mc::GreenKuboSpectrum,
    DipoleRateCrossSpectrum => mc::DipoleRateCrossSpectrum,
    DipoleAutocorrelationSpectrum => mc::DipoleAutocorrelationSpectrum,
    KramersKronig => mc::KramersKronig,
    ConductivitySumRule => mc::ConductivitySumRule,
    RouteAgreement => mc::RouteAgreement,
    PolarizabilityParts => mc::PolarizabilityParts,
    RamanSpectrum => mc::RamanSpectrum,
);
