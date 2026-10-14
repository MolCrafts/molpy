//! Order parameters (Steinhardt, hexatic, nematic, cubatic, solid-liquid,
//! rotational autocorrelation) — WASM face of the `molrs::compute` order family.

use super::{js_value, quats};
use crate::core::frame::Frame;
use crate::core::neighbors::Neighbors;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn vectors3(data: &[F], name: &str) -> Result<Vec<[F; 3]>, JsValue> {
    if !data.len().is_multiple_of(3) {
        return Err(JsValue::from_str(&format!(
            "{name}: expected flat [x,y,z,...] length divisible by 3"
        )));
    }
    Ok(data.as_chunks::<3>().0.to_vec())
}

#[wasm_bindgen(js_name = Steinhardt)]
pub struct Steinhardt {
    inner: molrs::compute::Steinhardt,
}

#[wasm_bindgen(js_class = Steinhardt)]
impl Steinhardt {
    #[wasm_bindgen(constructor)]
    pub fn new(
        l_values: &[u32],
        average: Option<bool>,
        wl: Option<bool>,
        wl_normalize: Option<bool>,
    ) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::Steinhardt::new(l_values)
            .map_err(|e| JsValue::from_str(&format!("Steinhardt: {e}")))?;
        inner = inner
            .with_average(average.unwrap_or(false))
            .with_wl(wl.unwrap_or(false))
            .with_wl_normalize(wl_normalize.unwrap_or(false));
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l: Vec<u32>,
            ql: Vec<Vec<F>>,
            wl: Option<Vec<Vec<F>>>,
            qlm_re_im: Vec<Vec<F>>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("Steinhardt compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("Steinhardt: empty result"))?;
            let qlm_re_im = r
                .qlm
                .iter()
                .map(|band| band.iter().flat_map(|c| [c.re, c.im]).collect())
                .collect();
            js_value(&Out {
                l: r.l,
                ql: r.ql,
                wl: r.wl,
                qlm_re_im,
            })
        })
    }
}

#[wasm_bindgen(js_name = Hexatic)]
pub struct Hexatic {
    inner: molrs::compute::Hexatic,
}

#[wasm_bindgen(js_class = Hexatic)]
impl Hexatic {
    #[wasm_bindgen(constructor)]
    pub fn new(k: u32) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::Hexatic::new(k)
                .map_err(|e| JsValue::from_str(&format!("Hexatic: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            k: u32,
            psi_re_im: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("Hexatic compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("Hexatic: empty result"))?;
            js_value(&Out {
                k: r.k,
                psi_re_im: r.psi.iter().flat_map(|c| [c.re, c.im]).collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = Nematic)]
pub struct Nematic;

impl Default for Nematic {
    fn default() -> Self {
        Self::new()
    }
}

#[wasm_bindgen(js_class = Nematic)]
impl Nematic {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        Self
    }

    pub fn compute(&self, directors: &[F]) -> Result<JsValue, JsValue> {
        let directors = vectors3(directors, "Nematic directors")?;
        let dummy = molrs::core::Frame::new();
        let calc = molrs::compute::Nematic::new();
        let mut out = calc
            .compute(&[&dummy], &directors)
            .map_err(|e| JsValue::from_str(&format!("Nematic: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("Nematic: empty result"))?;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            order: F,
            eigenvalues: [F; 3],
            director: [F; 3],
            q_tensor: [[F; 3]; 3],
        }
        js_value(&Out {
            order: r.order,
            eigenvalues: r.eigenvalues,
            director: r.director,
            q_tensor: r.q_tensor,
        })
    }
}

#[wasm_bindgen(js_name = Cubatic)]
pub struct Cubatic {
    seed: u64,
    initial_temp: F,
    cooling_rate: F,
    n_steps: usize,
    n_chains: usize,
}

#[wasm_bindgen(js_class = Cubatic)]
impl Cubatic {
    #[wasm_bindgen(constructor)]
    pub fn new(
        seed: Option<f64>,
        initial_temp: Option<F>,
        cooling_rate: Option<F>,
        n_steps: Option<usize>,
        n_chains: Option<usize>,
    ) -> Self {
        Self {
            seed: seed.unwrap_or(0.0) as u64,
            initial_temp: initial_temp.unwrap_or(1.0),
            cooling_rate: cooling_rate.unwrap_or(0.95),
            n_steps: n_steps.unwrap_or(500),
            n_chains: n_chains.unwrap_or(4),
        }
    }

    pub fn compute(&self, directors: &[F]) -> Result<JsValue, JsValue> {
        let directors = vectors3(directors, "Cubatic directors")?;
        let dummy = molrs::core::Frame::new();
        let calc = molrs::compute::Cubatic::new()
            .with_seed(self.seed)
            .with_initial_temp(self.initial_temp)
            .with_cooling_rate(self.cooling_rate)
            .with_n_steps(self.n_steps)
            .with_n_chains(self.n_chains);
        let mut out = calc
            .compute(&[&dummy], &directors)
            .map_err(|e| JsValue::from_str(&format!("Cubatic: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("Cubatic: empty result"))?;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            order: F,
            director_basis: [[F; 3]; 3],
        }
        js_value(&Out {
            order: r.order,
            director_basis: r.director_basis,
        })
    }
}

#[wasm_bindgen(js_name = SolidLiquid)]
pub struct SolidLiquid {
    inner: molrs::compute::SolidLiquid,
}

#[wasm_bindgen(js_class = SolidLiquid)]
impl SolidLiquid {
    #[wasm_bindgen(constructor)]
    pub fn new(
        l: u32,
        q_threshold: Option<F>,
        n_threshold: Option<u32>,
        normalize_q: Option<bool>,
    ) -> Self {
        let mut inner = molrs::compute::SolidLiquid::new(l);
        if let Some(t) = q_threshold {
            inner = inner.with_q_threshold(t);
        }
        if let Some(n) = n_threshold {
            inner = inner.with_n_threshold(n);
        }
        if let Some(on) = normalize_q {
            inner = inner.with_normalize_q(on);
        }
        Self { inner }
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l: u32,
            n_solid_bonds: Vec<u32>,
            is_solid: Vec<bool>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("SolidLiquid compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("SolidLiquid: empty result"))?;
            js_value(&Out {
                l: r.l,
                n_solid_bonds: r.n_solid_bonds,
                is_solid: r.is_solid,
            })
        })
    }
}

#[wasm_bindgen(js_name = RotationalAutocorrelation)]
pub struct RotationalAutocorrelation {
    l: u32,
}

#[wasm_bindgen(js_class = RotationalAutocorrelation)]
impl RotationalAutocorrelation {
    #[wasm_bindgen(constructor)]
    pub fn new(l: u32) -> Self {
        Self { l }
    }

    pub fn compute(&self, reference: &[F], orientations: &[F]) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            l: u32,
            psi: Vec<F>,
            mean: F,
        }
        let reference = quats(reference, "RotationalAutocorrelation reference")?;
        let orientations = quats(orientations, "RotationalAutocorrelation orientations")?;
        let dummy = molrs::core::Frame::new();
        let calc = molrs::compute::RotationalAutocorrelation::new(self.l);
        let args = molrs::compute::RotationalAutocorrelationArgs {
            ref_orientations: &reference,
            orientations: &orientations,
        };
        let mut out = calc
            .compute(&[&dummy], args)
            .map_err(|e| JsValue::from_str(&format!("RotationalAutocorrelation: {e}")))?;
        let r = out
            .pop()
            .ok_or_else(|| JsValue::from_str("RotationalAutocorrelation: empty result"))?;
        js_value(&Out {
            l: r.l,
            psi: r.psi,
            mean: r.mean,
        })
    }
}
