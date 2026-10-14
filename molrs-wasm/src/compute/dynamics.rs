//! Van Hove correlation and pair survival — WASM face of the
//! `molrs::compute` dynamics family.

use super::{SeriesOut, array2, js_value};
use crate::core::frame::Frame;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use ndarray::Array3;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn array3(
    data: &[F],
    dim0: usize,
    dim1: usize,
    dim2: usize,
    name: &str,
) -> Result<Array3<F>, JsValue> {
    if data.len() != dim0 * dim1 * dim2 {
        return Err(JsValue::from_str(&format!(
            "{name}: data length {} != dim0 * dim1 * dim2 = {} * {} * {}",
            data.len(),
            dim0,
            dim1,
            dim2
        )));
    }
    Array3::from_shape_vec((dim0, dim1, dim2), data.to_vec())
        .map_err(|e| JsValue::from_str(&format!("{name}: {e}")))
}

#[wasm_bindgen(js_name = VanHove)]
pub struct VanHove {
    frames: Vec<molrs::core::Frame>,
    n_r_bins: usize,
    r_max: F,
    lags: Vec<usize>,
    stride: usize,
}

#[wasm_bindgen(js_class = VanHove)]
impl VanHove {
    #[wasm_bindgen(constructor)]
    pub fn new(n_r_bins: usize, r_max: F, lags: Vec<usize>, stride: Option<usize>) -> Self {
        Self {
            frames: Vec::new(),
            n_r_bins,
            r_max,
            lags,
            stride: stride.unwrap_or(1),
        }
    }

    pub fn feed(&mut self, frame: &Frame) -> Result<(), JsValue> {
        frame.with_frame(|rs_frame| {
            self.frames.push(rs_frame.clone());
            Ok(())
        })
    }

    pub fn compute(&self) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            r_edges: Vec<F>,
            r_centers: Vec<F>,
            lags: Vec<usize>,
            g_self: Vec<F>,
            g_distinct: Vec<F>,
            shape: [usize; 2],
            dr: F,
            has_distinct: bool,
        }
        let refs: Vec<&molrs::core::Frame> = self.frames.iter().collect();
        let calc = molrs::compute::VanHove::new(self.n_r_bins, self.r_max, self.lags.clone())
            .map_err(|e| JsValue::from_str(&format!("VanHove: {e}")))?
            .with_stride(self.stride);
        let r = calc
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("VanHove compute: {e}")))?;
        let shape = [r.g_self.nrows(), r.g_self.ncols()];
        js_value(&Out {
            r_edges: r.r_edges.to_vec(),
            r_centers: r.r_centers.to_vec(),
            lags: r.lags,
            g_self: r.g_self.iter().copied().collect(),
            g_distinct: r.g_distinct.iter().copied().collect(),
            shape,
            dr: r.dr,
            has_distinct: r.has_distinct,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Pair-survival time-correlation function — molrs
/// `compute::pair_survival_tcf`, `{ lagTimes, values }`.
///
/// `coordsI` is flat `(nFrames, nI, 3)`, `coordsJ` flat `(nFrames, nJ, 3)`,
/// `boxLengths` flat `(nFrames, 3)`. A pair is born inside `r0` and breaks
/// past `r1`; `method` is `"continuous"`, `"intermittent"` or `"ssp"`.
#[allow(clippy::too_many_arguments)]
#[wasm_bindgen(js_name = pairSurvivalTcf)]
pub fn pair_survival_tcf(
    coords_i: &[F],
    n_frames: usize,
    n_i: usize,
    coords_j: &[F],
    n_j: usize,
    box_lengths: &[F],
    r0: F,
    r1: F,
    method: &str,
    dt: F,
    max_lag: usize,
    exclude_self: bool,
) -> Result<JsValue, JsValue> {
    let ci = array3(coords_i, n_frames, n_i, 3, "pairSurvivalTcf coordsI")?;
    let cj = array3(coords_j, n_frames, n_j, 3, "pairSurvivalTcf coordsJ")?;
    let bl = array2(box_lengths, n_frames, 3, "pairSurvivalTcf boxLengths")?;
    let method = molrs::compute::SurvivalMethod::parse(method)
        .map_err(|e| JsValue::from_str(&format!("pairSurvivalTcf method: {e}")))?;
    let r =
        molrs::compute::pair_survival_tcf(&ci, &cj, &bl, r0, r1, method, dt, max_lag, exclude_self)
            .map_err(|e| JsValue::from_str(&format!("pairSurvivalTcf: {e}")))?;
    js_value(&SeriesOut {
        lag_times: r.lag_times.to_vec(),
        values: r.correlation.to_vec(),
    })
}
