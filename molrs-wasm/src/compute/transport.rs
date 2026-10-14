//! Transport coefficients (VACF, Green–Kubo, Einstein, Onsager, Debye) —
//! WASM face of the `molrs::compute` transport family.

use super::{SeriesOut, array1, array2, js_value};
use crate::core::frame::Frame;
use molrs::compute::Fit;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = Vacf)]
pub struct Vacf {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = Vacf)]
impl Vacf {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        velocities: &[F],
        n_frames: usize,
        n_dof: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let v = array2(velocities, n_frames, n_dof, "Vacf velocities")?;
        let frames: [&molrs::core::Frame; 0] = [];
        let calc = molrs::compute::Vacf;
        let r = calc
            .compute(&frames, (&v, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("Vacf: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = GreenKuboDiffusion)]
pub struct GreenKuboDiffusion {
    dt: F,
    resolution: usize,
}

#[wasm_bindgen(js_class = GreenKuboDiffusion)]
impl GreenKuboDiffusion {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, resolution: usize) -> Self {
        Self { dt, resolution }
    }

    pub fn compute(
        &self,
        velocities: &[F],
        n_frames: usize,
        n_dof: usize,
    ) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let resolution = self.resolution;
        let v = array2(velocities, n_frames, n_dof, "GreenKuboDiffusion velocities")?;
        let frames: [&molrs::core::Frame; 0] = [];
        let calc = molrs::compute::GreenKuboDiffusion;
        let r = calc
            .compute(&frames, (&v, dt, resolution))
            .map_err(|e| JsValue::from_str(&format!("GreenKuboDiffusion: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.acf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = GreenKuboConductivity)]
pub struct GreenKuboConductivity {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = GreenKuboConductivity)]
impl GreenKuboConductivity {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(&self, current: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        let c = array2(current, n_frames, 3, "GreenKuboConductivity current")?;
        let frames: [&molrs::core::Frame; 0] = [];
        let calc = molrs::compute::GreenKuboConductivity;
        let r = calc
            .compute(&frames, (&c, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("GreenKuboConductivity: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.jacf.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = EinsteinConductivity)]
pub struct EinsteinConductivity {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = EinsteinConductivity)]
impl EinsteinConductivity {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(&self, translational_dipole: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        let d = array2(
            translational_dipole,
            n_frames,
            3,
            "EinsteinConductivity translationalDipole",
        )?;
        let frames: [&molrs::core::Frame; 0] = [];
        let calc = molrs::compute::EinsteinConductivity;
        let r = calc
            .compute(&frames, (&d, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("EinsteinConductivity: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.msd.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = OnsagerCorrelation)]
pub struct OnsagerCorrelation {
    dt: F,
    max_lag: usize,
}

#[wasm_bindgen(js_class = OnsagerCorrelation)]
impl OnsagerCorrelation {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F, max_lag: usize) -> Self {
        Self { dt, max_lag }
    }

    pub fn compute(&self, pi: &[F], pj: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        let pi = array2(pi, n_frames, 3, "Onsager p_i")?;
        let pj = array2(pj, n_frames, 3, "Onsager p_j")?;
        let frames: [&molrs::core::Frame; 0] = [];
        let calc = molrs::compute::OnsagerCorrelation;
        let r = calc
            .compute(&frames, (&pi, &pj, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("OnsagerCorrelation: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.correlation.to_vec(),
        })
    }
}

#[wasm_bindgen(js_name = EinsteinDiffusion)]
pub struct EinsteinDiffusion {
    dt: F,
    frames: Vec<molrs::core::Frame>,
}

#[wasm_bindgen(js_class = EinsteinDiffusion)]
impl EinsteinDiffusion {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F) -> Self {
        Self {
            dt,
            frames: Vec::new(),
        }
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let refs: Vec<&molrs::core::Frame> = self.frames.iter().collect();
        let calc = molrs::compute::EinsteinDiffusion;
        let r = calc
            .compute(&refs, molrs::compute::EinsteinDiffusionArgs { dt })
            .map_err(|e| JsValue::from_str(&format!("EinsteinDiffusion: {e}")))?;
        js_value(&SeriesOut {
            lag_times: r.lag_times.to_vec(),
            values: r.msd.to_vec(),
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

#[wasm_bindgen(js_name = DebyeRelaxation)]
pub struct DebyeRelaxation {
    dt: F,
    max_lag: usize,
    volume: F,
    temperature: F,
    boundary: String,
}

#[wasm_bindgen(js_class = DebyeRelaxation)]
impl DebyeRelaxation {
    #[wasm_bindgen(constructor)]
    pub fn new(volume: F, temperature: F, boundary: Option<String>, dt: F, max_lag: usize) -> Self {
        Self {
            dt,
            max_lag,
            volume,
            temperature,
            boundary: boundary.unwrap_or_else(|| "tinfoil".to_string()),
        }
    }

    pub fn compute(&self, dipoles: &[F], n_frames: usize) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        let max_lag = self.max_lag;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            lag_times: Vec<F>,
            acf: Vec<F>,
            zero_lag_variance: F,
            volume: F,
            temperature: F,
            boundary: String,
        }

        let dipoles = array2(dipoles, n_frames, 3, "DebyeRelaxation dipoles")?;
        let boundary = molrs::compute::EwaldBoundary::from_name(&self.boundary)
            .map_err(|e| JsValue::from_str(&format!("DebyeRelaxation boundary: {e}")))?;
        let calc = molrs::compute::DebyeRelaxation {
            volume: self.volume,
            temperature: self.temperature,
            boundary,
        };
        let frames: [&molrs::core::Frame; 0] = [];
        let r = calc
            .compute(&frames, (&dipoles, dt, max_lag))
            .map_err(|e| JsValue::from_str(&format!("DebyeRelaxation: {e}")))?;
        js_value(&Out {
            lag_times: r.lag_times.to_vec(),
            acf: r.acf.to_vec(),
            zero_lag_variance: r.zero_lag_variance,
            volume: r.volume,
            temperature: r.temperature,
            boundary: self.boundary.clone(),
        })
    }
}

#[wasm_bindgen(js_name = DebyeFit)]
pub struct DebyeFit {
    dt: F,
}

#[wasm_bindgen(js_class = DebyeFit)]
impl DebyeFit {
    #[wasm_bindgen(constructor)]
    pub fn new(dt: F) -> Self {
        Self { dt }
    }

    pub fn fit(&self, phi: &[F]) -> Result<JsValue, JsValue> {
        let dt = self.dt;
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            tau: F,
            amplitude: F,
            n_samples: usize,
        }
        let phi = array1(phi);
        let r = molrs::compute::DebyeFit
            .fit((&phi, dt))
            .map_err(|e| JsValue::from_str(&format!("DebyeFit: {e}")))?;
        js_value(&Out {
            tau: r.tau,
            amplitude: r.amplitude,
            n_samples: r.n_samples,
        })
    }
}
