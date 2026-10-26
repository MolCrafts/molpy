//! Order parameters — the WASM face of `molrs::compute::order`.
//!
//! ```js
//! const q = new Steinhardt([4, 6]).compute({ neighbors });
//! const solid = new SolidLiquid().compute({ qlm: q.qlm(6), neighbors });
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::Given;
use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// Steinhardt invariants: the orders `l`, one `QlmTable` per order, and
    /// the q_l (and w_l) feature matrix. A `FeatureMatrix` for `Pca` / `Kmeans`.
    SteinhardtResult => mc::SteinhardtResult
}

#[wasm_bindgen]
impl SteinhardtResult {
    /// The q_ℓm table of order `l` — `SolidLiquid`'s and
    /// `ContinuousCoordination`'s `qlm`; `undefined` if `l` was not computed.
    pub fn qlm(&self, l: u32) -> Option<QlmTable> {
        self.inner.qlm(l).cloned().map(QlmTable::from)
    }
}

output! {
    /// Per-particle q_ℓm (or ψ_k) of one order, as complex `{ re, im }`.
    QlmTable => mc::QlmTable
}

output! {
    /// Per-particle bond-orientational order ψ_k as complex `{ re, im }`.
    HexaticResult => mc::HexaticResult
}

output! {
    /// Solid-like bond counts and the per-particle solid flag.
    SolidLiquidResult => mc::SolidLiquidResult
}

output! {
    /// Continuous coordination number per particle.
    ContinuousCoordinationResult => mc::ContinuousCoordinationResult
}

output! {
    /// Nematic order: `order`, `eigenvalues`, `director`, `q_tensor`.
    NematicResult => mc::NematicResult
}

output! {
    /// Cubatic order: `order` and the `director_basis`.
    CubaticResult => mc::CubaticResult
}

/// Steinhardt bond-orientational order of the given orders `l`.
#[wasm_bindgen]
pub struct Steinhardt {
    inner: mc::Steinhardt,
}

#[wasm_bindgen]
impl Steinhardt {
    #[wasm_bindgen(constructor)]
    pub fn new(
        l: Vec<u32>,
        average: Option<bool>,
        wl: Option<bool>,
        wl_normalize: Option<bool>,
    ) -> Result<Steinhardt, JsError> {
        let inner = mc::Steinhardt::new(&l)?
            .given(average, |c, v| Ok(c.with_average(v)))?
            .given(wl, |c, v| Ok(c.with_wl(v)))?
            .given(wl_normalize, |c, v| Ok(c.with_wl_normalize(v)))?;
        Ok(Self { inner })
    }

    /// `{ neighbors }`.
    pub fn compute(&self, data: &JsValue) -> Result<SteinhardtResult, JsError> {
        let data = Fields::new(data, "Steinhardt")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::SteinhardtInput {
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// k-fold bond-orientational order ψ_k in the plane.
#[wasm_bindgen]
pub struct Hexatic {
    inner: mc::Hexatic,
}

#[wasm_bindgen]
impl Hexatic {
    #[wasm_bindgen(constructor)]
    pub fn new(k: u32) -> Result<Hexatic, JsError> {
        Ok(Self {
            inner: mc::Hexatic::new(k)?,
        })
    }

    /// `{ neighbors }`.
    pub fn compute(&self, data: &JsValue) -> Result<HexaticResult, JsError> {
        let data = Fields::new(data, "Hexatic")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::HexaticInput {
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Solid / liquid classification from q_ℓm bond coherence.
#[wasm_bindgen]
pub struct SolidLiquid {
    inner: mc::SolidLiquid,
}

#[wasm_bindgen]
impl SolidLiquid {
    #[wasm_bindgen(constructor)]
    pub fn new(
        q_threshold: Option<F>,
        n_threshold: Option<u32>,
        normalize_q: Option<bool>,
    ) -> Result<SolidLiquid, JsError> {
        let inner = mc::SolidLiquid::new()
            .given(q_threshold, |c, v| c.with_q_threshold(v))?
            .given(n_threshold, |c, v| Ok(c.with_n_threshold(v)))?
            .given(normalize_q, |c, v| Ok(c.with_normalize_q(v)))?;
        Ok(Self { inner })
    }

    /// `{ qlm, neighbors }`: `qlm` from `SteinhardtResult.qlm(l)`.
    pub fn compute(&self, data: &JsValue) -> Result<SolidLiquidResult, JsError> {
        let data = Fields::new(data, "SolidLiquid")?;
        let qlm = data.handle::<QlmTable>("qlm", "QlmTable")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::SolidLiquidInput {
            qlm: &qlm.inner,
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Continuous coordination number from q_ℓm overlaps.
#[wasm_bindgen]
pub struct ContinuousCoordination {
    inner: mc::ContinuousCoordination,
}

#[wasm_bindgen]
impl ContinuousCoordination {
    #[wasm_bindgen(constructor)]
    pub fn new(power: F) -> Result<ContinuousCoordination, JsError> {
        Ok(Self {
            inner: mc::ContinuousCoordination::new(power)?,
        })
    }

    /// `{ qlm, neighbors }`: `qlm` from `SteinhardtResult.qlm(l)`.
    pub fn compute(&self, data: &JsValue) -> Result<ContinuousCoordinationResult, JsError> {
        let data = Fields::new(data, "ContinuousCoordination")?;
        let qlm = data.handle::<QlmTable>("qlm", "QlmTable")?;
        let neighbors = data.neighbors("neighbors")?;
        let input = mc::ContinuousCoordinationInput {
            qlm: &qlm.inner,
            neighbors: &neighbors,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Nematic order tensor of per-particle directors.
#[wasm_bindgen]
pub struct Nematic {
    inner: mc::Nematic,
}

#[wasm_bindgen]
impl Nematic {
    #[wasm_bindgen(constructor)]
    #[allow(clippy::new_without_default)]
    pub fn new() -> Nematic {
        Self {
            inner: mc::Nematic::new(),
        }
    }

    /// `{ directors }`: `N × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<NematicResult, JsError> {
        let data = Fields::new(data, "Nematic")?;
        let directors = data.matrix("directors", Some(3))?;
        let input = mc::NematicInput {
            directors: directors.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Cubatic order by simulated annealing.
#[wasm_bindgen]
pub struct Cubatic {
    inner: mc::Cubatic,
}

#[wasm_bindgen]
impl Cubatic {
    #[wasm_bindgen(constructor)]
    pub fn new(
        seed: Option<u64>,
        initial_temp: Option<F>,
        cooling_rate: Option<F>,
        min_temp: Option<F>,
        n_steps: Option<usize>,
        n_chains: Option<usize>,
    ) -> Result<Cubatic, JsError> {
        let inner = mc::Cubatic::new()
            .given(seed, |c, v| Ok(c.with_seed(v)))?
            .given(initial_temp, |c, v| c.with_initial_temp(v))?
            .given(cooling_rate, |c, v| c.with_cooling_rate(v))?
            .given(min_temp, |c, v| c.with_min_temp(v))?
            .given(n_steps, |c, v| c.with_n_steps(v))?
            .given(n_chains, |c, v| c.with_n_chains(v))?;
        Ok(Self { inner })
    }

    /// `{ directors }`: `N × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<CubaticResult, JsError> {
        let data = Fields::new(data, "Cubatic")?;
        let directors = data.matrix("directors", Some(3))?;
        let input = mc::CubaticInput {
            directors: directors.view(),
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    Steinhardt => mc::Steinhardt,
    Hexatic => mc::Hexatic,
    SolidLiquid => mc::SolidLiquid,
    ContinuousCoordination => mc::ContinuousCoordination,
    Nematic => mc::Nematic,
    Cubatic => mc::Cubatic,
);
