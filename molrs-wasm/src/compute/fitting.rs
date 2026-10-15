//! Fits and integrals of a correlation curve — the WASM face of
//! `molrs::compute::fitting`. Each reads one channel of a
//! [`TimeCorrelation`] a time-series compute returned.
//!
//! ```js
//! const curve = new Vacf(200).compute({ velocities, dt: 1.0 });
//! const integral = new CumulativeTrapezoid().compute({ correlation: curve, channel: 0 });
//! const plateau = new Plateau(0.5, 1.0).compute({ correlation: integral, channel: 0 });
//! ```

use molrs::compute::{self as mc, Compute};
use wasm_bindgen::prelude::*;

use super::config::Given;
use super::handle::{described, output};
use super::time_series::TimeCorrelation;
use super::wire::Fields;

output! {
    /// A straight-line fit: `slope`, `intercept`, `r2` and the fitted lag range.
    LinearFitResult => mc::LinearFitResult
}

output! {
    /// The mean of a curve over a window: `value`, `std`, `n_samples`.
    PlateauResult => mc::PlateauResult
}

output! {
    /// A single-exponential (Debye) fit: `tau`, `amplitude`, `n_samples`.
    DebyeFitResult => mc::DebyeFitResult
}

/// The `{ correlation, channel }` data every fit takes.
fn curve(data: &Fields<'_>) -> Result<(TimeCorrelation, usize), mc::ComputeError> {
    Ok((
        data.handle::<TimeCorrelation>("correlation", "TimeCorrelation")?,
        data.index("channel")?,
    ))
}

/// The running trapezoid integral of a curve, as a curve.
#[wasm_bindgen]
pub struct CumulativeTrapezoid {
    inner: mc::CumulativeTrapezoid,
}

#[wasm_bindgen]
impl CumulativeTrapezoid {
    #[wasm_bindgen(constructor)]
    pub fn new(n_lags: Option<usize>) -> Result<CumulativeTrapezoid, JsError> {
        let inner = mc::CumulativeTrapezoid::new().given(n_lags, |c, v| Ok(c.with_n_lags(v)))?;
        Ok(Self { inner })
    }

    /// `{ correlation, channel }`.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "CumulativeTrapezoid")?;
        let (correlation, channel) = curve(&data)?;
        let input = mc::CumulativeTrapezoidInput {
            correlation: &correlation.inner,
            channel,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// A Debye (single-exponential) fit of a decaying curve.
#[wasm_bindgen]
pub struct DebyeFit {
    inner: mc::DebyeFit,
}

#[wasm_bindgen]
impl DebyeFit {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> DebyeFit {
        Self {
            inner: mc::DebyeFit,
        }
    }

    /// `{ correlation, channel }`.
    pub fn compute(&self, data: &JsValue) -> Result<DebyeFitResult, JsError> {
        let data = Fields::new(data, "DebyeFit")?;
        let (correlation, channel) = curve(&data)?;
        let input = mc::DebyeFitInput {
            correlation: &correlation.inner,
            channel,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// A least-squares line over the `[start, end)` fraction of a curve.
#[wasm_bindgen]
pub struct LinearFit {
    inner: mc::LinearFit,
}

#[wasm_bindgen]
impl LinearFit {
    #[wasm_bindgen(constructor)]
    pub fn new(start: f64, end: f64) -> Result<LinearFit, JsError> {
        Ok(Self {
            inner: mc::LinearFit::new(start, end)?,
        })
    }

    /// `{ correlation, channel }`.
    pub fn compute(&self, data: &JsValue) -> Result<LinearFitResult, JsError> {
        let data = Fields::new(data, "LinearFit")?;
        let (correlation, channel) = curve(&data)?;
        let input = mc::LinearFitInput {
            correlation: &correlation.inner,
            channel,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// The mean of a curve over the `[start, end)` fraction of its lags.
#[wasm_bindgen]
pub struct Plateau {
    inner: mc::Plateau,
}

#[wasm_bindgen]
impl Plateau {
    #[wasm_bindgen(constructor)]
    pub fn new(start: f64, end: f64) -> Result<Plateau, JsError> {
        Ok(Self {
            inner: mc::Plateau::new(start, end)?,
        })
    }

    /// `{ correlation, channel }`.
    pub fn compute(&self, data: &JsValue) -> Result<PlateauResult, JsError> {
        let data = Fields::new(data, "Plateau")?;
        let (correlation, channel) = curve(&data)?;
        let input = mc::PlateauInput {
            correlation: &correlation.inner,
            channel,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    CumulativeTrapezoid => mc::CumulativeTrapezoid,
    DebyeFit => mc::DebyeFit,
    LinearFit => mc::LinearFit,
    Plateau => mc::Plateau,
);
