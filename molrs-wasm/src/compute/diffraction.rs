//! Structure factor and diffraction pattern — WASM face of the
//! `molrs::compute` diffraction family.

use super::{Grid2Out, js_value};
use crate::core::frame::Frame;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = StaticStructureFactorDebye)]
pub struct StaticStructureFactorDebye {
    inner: molrs::compute::StaticStructureFactorDebye,
}

#[wasm_bindgen(js_class = StaticStructureFactorDebye)]
impl StaticStructureFactorDebye {
    /// Sample `n_k` scattering vectors evenly over `[k_min, k_max]` (A^-1).
    #[wasm_bindgen(constructor)]
    pub fn new(k_min: F, k_max: F, n_k: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::StaticStructureFactorDebye::linspace(k_min, k_max, n_k)
                .map_err(|e| JsValue::from_str(&format!("StaticStructureFactorDebye: {e}")))?,
        })
    }

    /// Sample an explicit, possibly non-uniform, set of scattering vectors.
    #[wasm_bindgen(js_name = fromKValues)]
    pub fn from_k_values(k_values: &[F]) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::StaticStructureFactorDebye::new(k_values)
                .map_err(|e| JsValue::from_str(&format!("StaticStructureFactorDebye: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            k_values: Vec<F>,
            sk: Vec<F>,
            n_particles: usize,
        }
        frame.with_frame(|rs_frame| {
            let mut out = self.inner.compute(&[rs_frame], ()).map_err(|e| {
                JsValue::from_str(&format!("StaticStructureFactorDebye compute: {e}"))
            })?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("StaticStructureFactorDebye: empty result"))?;
            js_value(&Out {
                k_values: r.k_values.to_vec(),
                sk: r.sk.to_vec(),
                n_particles: r.n_particles,
            })
        })
    }
}

#[wasm_bindgen(js_name = DiffractionPattern)]
pub struct DiffractionPattern {
    inner: molrs::compute::DiffractionPattern,
}

#[wasm_bindgen(js_class = DiffractionPattern)]
impl DiffractionPattern {
    #[wasm_bindgen(constructor)]
    pub fn new(n_grid: usize, sigma: F, axis: Option<usize>) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::DiffractionPattern::new(n_grid, sigma)
            .map_err(|e| JsValue::from_str(&format!("DiffractionPattern: {e}")))?;
        if let Some(axis) = axis {
            inner = inner.with_axis(axis);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            diffraction: Grid2Out,
            image: Grid2Out,
        }
        frame.with_frame(|rs_frame| {
            let mut out = self
                .inner
                .compute(&[rs_frame], ())
                .map_err(|e| JsValue::from_str(&format!("DiffractionPattern compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("DiffractionPattern: empty result"))?;
            let shape = [r.diffraction.nrows(), r.diffraction.ncols()];
            js_value(&Out {
                diffraction: Grid2Out {
                    data: r.diffraction.iter().copied().collect(),
                    shape,
                },
                image: Grid2Out {
                    data: r.image.iter().copied().collect(),
                    shape,
                },
            })
        })
    }
}
