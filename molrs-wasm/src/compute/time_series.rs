//! Time series — the WASM face of `molrs::compute::time_series`: the one
//! [`Correlator`], its named surfaces ([`Msd`], [`Vacf`], …), and the series
//! operations ([`Derivative`], [`Unwrap`], [`Spectrum`]).
//!
//! The sample interval `dt` is data: it is a field of every `compute` call.
//! `Msd` and `Vacf` are frame-form, as their core types are accumulator-only:
//! `compute` takes one record per frame and folds them, in order, through
//! the core accumulator (`accumulator → push → finish`).
//!
//! ```js
//! const msd = new Msd(100);                                    // maxLag
//! const curve = msd.compute(frames.map((xyz) => ({ a: xyz, dt: 0.5 })));
//! const fit = new LinearFit(0.2, 0.8).compute({ correlation: curve, channel: 0 });
//! console.log(fit.toJSON().slope / 6);                         // D
//! ```

use molrs::compute::{self as mc, Accumulate, Compute, Stream};
use wasm_bindgen::prelude::*;

use super::config::Given;
use super::handle::{described, output};
use super::wire::{Fields, records};
use crate::core::NDArray;

output! {
    /// A time-correlation curve: `lag_times` and `values`, one column per
    /// channel. The input of the fits, the spectra and `CumulativeTrapezoid`.
    TimeCorrelation => mc::TimeCorrelation
}

output! {
    /// A one-sided transform: `frequencies`, `real`, `imag`, `n_fft`.
    OneSidedSpectrum => mc::OneSidedSpectrum
}

/// The time correlation of a series under any [`LagKernel`](mc::LagKernel):
/// the one estimator every named surface configures.
#[wasm_bindgen]
pub struct Correlator {
    inner: mc::Correlator,
}

#[wasm_bindgen]
impl Correlator {
    /// `kernel` is one of `product`, `squared_displacement`, `legendre`,
    /// `wigner_d`, `persistence` (the core `LagKernelKind`); `l` is the order
    /// `legendre` / `wigner_d` require and the others refuse.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        kernel: &str,
        max_lag: usize,
        l: Option<u32>,
        origins: Option<String>,
        mean: Option<String>,
        normalization: Option<String>,
        reduction: Option<String>,
        stride: Option<usize>,
        dimension: Option<usize>,
    ) -> Result<Correlator, JsError> {
        let inner = mc::Correlator::new(mc::LagKernel::new(kernel.parse()?, l)?, max_lag)
            .given(origins, |c, v| Ok(c.with_origins(v.parse()?)))?
            .given(mean, |c, v| Ok(c.with_mean(v.parse()?)))?
            .given(normalization, |c, v| Ok(c.with_normalization(v.parse()?)))?
            .given(reduction, |c, v| Ok(c.with_reduction(v.parse()?)))?
            .given(stride, |c, v| c.with_stride(v))?
            .given(dimension, |c, v| Ok(c.with_dimension(v)))?;
        Ok(Self { inner })
    }

    /// `{ a, b?, dt, boxes? }`: `a` (and `b`) are `T × N × D`.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "Correlator")?;
        let a = data.cube("a")?;
        let b = data.opt_cube("b")?;
        let boxes = data.opt_boxes("boxes")?;
        let input = mc::CorrelatorInput {
            a: a.view(),
            b: b.as_ref().map(|b| b.view()),
            dt: data.scalar("dt")?,
            boxes: boxes.as_deref(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Mean squared displacement of unwrapped positions.
#[wasm_bindgen]
pub struct Msd {
    inner: mc::Msd,
}

#[wasm_bindgen]
impl Msd {
    #[wasm_bindgen(constructor)]
    pub fn new(
        max_lag: usize,
        origins: Option<String>,
        reduction: Option<String>,
        stride: Option<usize>,
    ) -> Result<Msd, JsError> {
        let inner = mc::Msd::new(max_lag)
            .given(origins, |c, v| Ok(c.with_origins(v.parse()?)))?
            .given(reduction, |c, v| Ok(c.with_reduction(v.parse()?)))?
            .given(stride, |c, v| c.with_stride(v))?;
        Ok(Self { inner })
    }

    /// `[{ a, dt, simbox? }, …]`, one record per frame, in time order: `a`
    /// is that frame's unwrapped positions (`N × 3`), `dt` the sample
    /// interval (the same on every frame), `simbox` the frame's `Box` for the
    /// wrapped-trajectory check.
    pub fn compute(&self, frames: &JsValue) -> Result<TimeCorrelation, JsError> {
        let mut accumulator = self.inner.accumulator();
        for frame in records(frames, "Msd")? {
            let data = Fields::new(&frame, "Msd")?;
            let a = data.matrix("a", Some(3))?;
            let simbox = data.opt_simbox("simbox")?;
            accumulator.push(mc::CorrelatorFrame {
                a: a.view(),
                b: None,
                dt: data.scalar("dt")?,
                simbox: simbox.as_ref(),
            })?;
        }
        Ok(accumulator.finish()?.into())
    }
}

/// Velocity autocorrelation.
#[wasm_bindgen]
pub struct Vacf {
    inner: mc::Vacf,
}

#[wasm_bindgen]
impl Vacf {
    /// `normalization` is a [`Normalization`](mc::Normalization) name;
    /// `stride` takes every `stride`-th frame as an origin; `mean` is a
    /// [`MeanPolicy`](mc::MeanPolicy) name (default `"subtract"`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        max_lag: usize,
        normalization: Option<String>,
        stride: Option<usize>,
        mean: Option<String>,
    ) -> Result<Vacf, JsError> {
        let inner = mc::Vacf::new(max_lag)
            .given(normalization, |c, v| Ok(c.with_normalization(v.parse()?)))?
            .given(stride, |c, v| c.with_stride(v))?
            .given(mean, |c, v| Ok(c.with_mean(v.parse()?)))?;
        Ok(Self { inner })
    }

    /// `[{ a, dt }, …]`, one record per frame, in time order: `a` is that
    /// frame's velocities, flat (`n_dof`, one degree of freedom per entry),
    /// `dt` the sample interval (the same on every frame).
    pub fn compute(&self, frames: &JsValue) -> Result<TimeCorrelation, JsError> {
        let mut accumulator = self.inner.accumulator();
        for frame in records(frames, "Vacf")? {
            let data = Fields::new(&frame, "Vacf")?;
            let a = data.reals("a")?;
            accumulator.push(mc::CorrelatorFrame {
                a: a.view().insert_axis(ndarray::Axis(1)),
                b: None,
                dt: data.scalar("dt")?,
                simbox: None,
            })?;
        }
        Ok(accumulator.finish()?.into())
    }
}

/// Cross displacement correlation of two collective coordinates (Onsager).
#[wasm_bindgen]
pub struct OnsagerCorrelation {
    inner: mc::OnsagerCorrelation,
}

#[wasm_bindgen]
impl OnsagerCorrelation {
    /// `stride` takes every `stride`-th frame as an origin.
    #[wasm_bindgen(constructor)]
    pub fn new(max_lag: usize, stride: Option<usize>) -> Result<OnsagerCorrelation, JsError> {
        let inner = mc::OnsagerCorrelation::new(max_lag).given(stride, |c, v| c.with_stride(v))?;
        Ok(Self { inner })
    }

    /// `{ a, b, dt, boxes? }`: `a`, `b` are `T × 3`, unwrapped.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "OnsagerCorrelation")?;
        let a = data.matrix("a", Some(3))?;
        let b = data.matrix("b", Some(3))?;
        let boxes = data.opt_boxes("boxes")?;
        let input = mc::OnsagerInput {
            a: a.view(),
            b: b.view(),
            dt: data.scalar("dt")?,
            boxes: boxes.as_deref(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Total-dipole autocorrelation.
#[wasm_bindgen]
pub struct DipoleAcf {
    inner: mc::DipoleAcf,
}

#[wasm_bindgen]
impl DipoleAcf {
    /// `normalization` is a [`Normalization`](mc::Normalization) name;
    /// `mean` a [`MeanPolicy`](mc::MeanPolicy) name (default `"subtract"`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        max_lag: usize,
        normalization: Option<String>,
        stride: Option<usize>,
        mean: Option<String>,
    ) -> Result<DipoleAcf, JsError> {
        let inner = mc::DipoleAcf::new(max_lag)
            .given(normalization, |c, v| Ok(c.with_normalization(v.parse()?)))?
            .given(stride, |c, v| c.with_stride(v))?
            .given(mean, |c, v| Ok(c.with_mean(v.parse()?)))?;
        Ok(Self { inner })
    }

    /// `{ dipoles, dt }`: `dipoles` is `T × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "DipoleAcf")?;
        let dipoles = data.matrix("dipoles", Some(3))?;
        let input = mc::DipoleAcfInput {
            dipoles: dipoles.view(),
            dt: data.scalar("dt")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Collective-current autocorrelation.
#[wasm_bindgen]
pub struct CurrentAcf {
    inner: mc::CurrentAcf,
}

#[wasm_bindgen]
impl CurrentAcf {
    /// `normalization` is a [`Normalization`](mc::Normalization) name
    /// (`"none"`: the raw lagged sum of the former IR flux correlation);
    /// `mean` a [`MeanPolicy`](mc::MeanPolicy) name (default `"keep"`).
    #[wasm_bindgen(constructor)]
    pub fn new(
        max_lag: usize,
        normalization: Option<String>,
        stride: Option<usize>,
        mean: Option<String>,
    ) -> Result<CurrentAcf, JsError> {
        let inner = mc::CurrentAcf::new(max_lag)
            .given(normalization, |c, v| Ok(c.with_normalization(v.parse()?)))?
            .given(stride, |c, v| c.with_stride(v))?
            .given(mean, |c, v| Ok(c.with_mean(v.parse()?)))?;
        Ok(Self { inner })
    }

    /// `{ current, dt }`: `current` is `T × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "CurrentAcf")?;
        let current = data.matrix("current", Some(3))?;
        let input = mc::CurrentAcfInput {
            current: current.view(),
            dt: data.scalar("dt")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Legendre reorientation correlation of per-molecule vectors.
#[wasm_bindgen]
pub struct ReorientationAcf {
    inner: mc::ReorientationAcf,
}

#[wasm_bindgen]
impl ReorientationAcf {
    /// `reduction` is `mean` or `per_item`; `stride` takes every
    /// `stride`-th frame as an origin. Orders 1 and 2 run by FFT.
    #[wasm_bindgen(constructor)]
    pub fn new(
        l: u32,
        max_lag: usize,
        reduction: Option<String>,
        stride: Option<usize>,
    ) -> Result<ReorientationAcf, JsError> {
        let inner = mc::ReorientationAcf::new(l, max_lag)
            .given(reduction, |c, v| Ok(c.with_reduction(v.parse()?)))?
            .given(stride, |c, v| c.with_stride(v))?;
        Ok(Self { inner })
    }

    /// `{ vectors, dt }`: `vectors` is `T × N × D`, each normalised.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "ReorientationAcf")?;
        let vectors = data.cube("vectors")?;
        let input = mc::ReorientationAcfInput {
            vectors: vectors.view(),
            dt: data.scalar("dt")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Wigner-D rotational autocorrelation of orientation quaternions.
#[wasm_bindgen]
pub struct RotationalAutocorrelation {
    inner: mc::RotationalAutocorrelation,
}

#[wasm_bindgen]
impl RotationalAutocorrelation {
    #[wasm_bindgen(constructor)]
    pub fn new(
        l: u32,
        max_lag: usize,
        reduction: Option<String>,
    ) -> Result<RotationalAutocorrelation, JsError> {
        let inner = mc::RotationalAutocorrelation::new(l, max_lag)
            .given(reduction, |c, v| Ok(c.with_reduction(v.parse()?)))?;
        Ok(Self { inner })
    }

    /// `{ orientations, dt }`: `orientations` is `T × N × 4`, `(w, x, y, z)`.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "RotationalAutocorrelation")?;
        let orientations = data.cube("orientations")?;
        let input = mc::RotationalAutocorrelationInput {
            orientations: orientations.view(),
            dt: data.scalar("dt")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Survival correlation of a 0/1 presence series.
#[wasm_bindgen]
pub struct Survival {
    inner: mc::Survival,
}

#[wasm_bindgen]
impl Survival {
    /// `kind` is a [`SurvivalKind`](mc::SurvivalKind) name, `normalization`
    /// a [`SurvivalNormalization`](mc::SurvivalNormalization) name, `origin`
    /// a [`SurvivalOrigin`](mc::SurvivalOrigin) name.
    #[wasm_bindgen(constructor)]
    pub fn new(
        kind: &str,
        max_lag: usize,
        normalization: Option<String>,
        origin: Option<String>,
    ) -> Result<Survival, JsError> {
        let inner = mc::Survival::new(kind.parse()?, max_lag)
            .given(normalization, |c, v| Ok(c.with_normalization(v.parse()?)))?
            .given(origin, |c, v| Ok(c.with_origin(v.parse()?)))?;
        Ok(Self { inner })
    }

    /// `{ presence, dt, inner?, n_reference? }`: `presence` is `T × n_bonds`
    /// of 0 / 1 — e.g. the `values` of a `Presence`; `inner` (same shape)
    /// for the `inner` origin rule, `n_reference` for the `per_reference`
    /// normalization.
    pub fn compute(&self, data: &JsValue) -> Result<TimeCorrelation, JsError> {
        let data = Fields::new(data, "Survival")?;
        let presence = data.matrix("presence", None)?;
        let inner = data.opt_matrix("inner", None)?;
        let input = mc::SurvivalInput {
            presence: presence.view(),
            dt: data.scalar("dt")?,
            inner: inner.as_ref().map(|i| i.view()),
            n_reference: data.opt_count("n_reference")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Finite-difference time derivative of every channel of a series.
#[wasm_bindgen]
pub struct Derivative {
    inner: mc::Derivative,
}

#[wasm_bindgen]
impl Derivative {
    /// `stencil` is a [`Stencil`](mc::Stencil) name.
    #[wasm_bindgen(constructor)]
    pub fn new(stencil: &str) -> Result<Derivative, JsError> {
        Ok(Self {
            inner: mc::Derivative::new(stencil.parse()?),
        })
    }

    /// `{ series, dt }`: `series` is `T × C`; returns the `T × C` derivative.
    pub fn compute(&self, data: &JsValue) -> Result<NDArray, JsError> {
        let data = Fields::new(data, "Derivative")?;
        let series = data.matrix("series", None)?;
        let input = mc::DerivativeInput {
            series: series.view(),
            dt: data.scalar("dt")?,
        };
        Ok(NDArray::from_ndarray(&self.inner.compute(input)?))
    }
}

/// Unwrap a wrapped trajectory, from image flags or by continuity.
#[wasm_bindgen]
pub struct Unwrap {
    inner: mc::Unwrap,
}

#[wasm_bindgen]
impl Unwrap {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Unwrap {
        Self {
            inner: mc::Unwrap::new(),
        }
    }

    /// `{ positions, images?, boxes }`: `positions` is `T × N × 3`, `images`
    /// the matching flat integer crossings; returns the unwrapped `T × N × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<NDArray, JsError> {
        let data = Fields::new(data, "Unwrap")?;
        let positions = data.cube("positions")?;
        let images = data.opt_images("images", positions.dim())?;
        let boxes = data.boxes("boxes")?;
        let input = mc::UnwrapInput {
            positions: positions.view(),
            images: images.as_ref().map(|i| i.view()),
            boxes: &boxes,
        };
        Ok(NDArray::from_ndarray(&self.inner.compute(input)?))
    }
}

/// Windowed, zero-padded one-sided Fourier transform of a signal.
#[wasm_bindgen]
pub struct Spectrum {
    inner: mc::Spectrum,
}

#[wasm_bindgen]
impl Spectrum {
    /// `window` is a [`Window`](mc::Window) name; `pad` the zero-padding factor.
    #[wasm_bindgen(constructor)]
    pub fn new(window: &str, pad: usize) -> Result<Spectrum, JsError> {
        Ok(Self {
            inner: mc::Spectrum::new(window.parse()?, pad)?,
        })
    }

    /// `{ signal, dt }`: `signal` is 1-D.
    pub fn compute(&self, data: &JsValue) -> Result<OneSidedSpectrum, JsError> {
        let data = Fields::new(data, "Spectrum")?;
        let signal = data.reals("signal")?;
        let input = mc::SpectrumInput {
            signal: signal.view(),
            dt: data.scalar("dt")?,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    Correlator => mc::Correlator,
    Msd => mc::Msd,
    Vacf => mc::Vacf,
    OnsagerCorrelation => mc::OnsagerCorrelation,
    DipoleAcf => mc::DipoleAcf,
    CurrentAcf => mc::CurrentAcf,
    ReorientationAcf => mc::ReorientationAcf,
    RotationalAutocorrelation => mc::RotationalAutocorrelation,
    Survival => mc::Survival,
    Derivative => mc::Derivative,
    Unwrap => mc::Unwrap,
    Spectrum => mc::Spectrum,
);
