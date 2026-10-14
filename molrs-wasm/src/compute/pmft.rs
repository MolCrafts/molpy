//! Potentials of mean force and torque (freud.pmft) — WASM face of the
//! `molrs::compute` pmft family.

use super::js_value;
use crate::core::frame::Frame;
use crate::core::neighbors::Neighbors;
use molrs::compute::LegacyCompute;
use molrs::core::keys;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

/// The planar orientation angles `frame` states (molrs
/// `compute::planar_orientation_angles`: quaternion z-rotations or
/// `orientations` head–tail axes), as a JS error when the frame is malformed.
fn angles_from_frame(frame: &molrs::core::Frame) -> Result<Option<Vec<F>>, JsValue> {
    molrs::compute::planar_orientation_angles(frame)
        .map_err(|e| JsValue::from_str(&format!("orientations: {e}")))
}

/// `angles_from_frame` for the analyses whose orientations are mandatory.
fn require_angles(frame: &molrs::core::Frame, what: &str) -> Result<Vec<F>, JsValue> {
    angles_from_frame(frame)?.ok_or_else(|| {
        JsValue::from_str(&format!(
            "{what} needs per-atom orientations: add the {} columns to the atoms block, \
             or an `orientations` block of (head, tail) atom pairs",
            keys::QUAT.join(", ")
        ))
    })
}

// ===========================================================================
// PMFT — potentials of mean force and torque (freud.pmft)
// ===========================================================================

/// Binned free-energy surface, shared by every PMFT variant. `density`,
/// `rawCounts` and `pmf` are row-major over `shape`; `edges[k]` holds the
/// `shape[k] + 1` bin edges of axis `axes[k]`.
#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct PmftOut {
    axes: Vec<&'static str>,
    shape: Vec<usize>,
    edges: Vec<Vec<F>>,
    density: Vec<F>,
    raw_counts: Vec<u64>,
    pmf: Vec<F>,
}

#[wasm_bindgen(js_name = PmftR12)]
pub struct PmftR12 {
    inner: molrs::compute::PmftR12,
}

#[wasm_bindgen(js_class = PmftR12)]
impl PmftR12 {
    /// Radial range `r_max` (A); `n_r × n_t1 × n_t2` bins over `(r, θ₁, θ₂)`.
    #[wasm_bindgen(constructor)]
    pub fn new(r_max: F, n_r: usize, n_t1: usize, n_t2: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PmftR12::new(r_max, n_r, n_t1, n_t2)
                .map_err(|e| JsValue::from_str(&format!("PmftR12: {e}")))?,
        })
    }

    /// Requires per-atom orientation angles on the frame.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = vec![require_angles(rs_frame, "PmftR12")?];
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PmftR12Args {
                        nlists,
                        orientations: &orientations,
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PmftR12 compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PmftR12: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["r", "theta1", "theta2"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.r_edges, r.t1_edges, r.t2_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = PmftXy)]
pub struct PmftXy {
    inner: molrs::compute::PmftXy,
}

#[wasm_bindgen(js_class = PmftXy)]
impl PmftXy {
    /// Body-frame window `±x_max × ±y_max` (A); `n_x × n_y` bins.
    #[wasm_bindgen(constructor)]
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PmftXy::new(x_max, y_max, n_x, n_y)
                .map_err(|e| JsValue::from_str(&format!("PmftXy: {e}")))?,
        })
    }

    /// Orientations are optional: without them every query particle is treated
    /// as unrotated, which is the isotropic reference the freud docs describe.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = angles_from_frame(rs_frame)?.map(|angles| vec![angles]);
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PmftXyArgs {
                        nlists,
                        query_orientations: orientations.as_deref(),
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PmftXy compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PmftXy: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["x", "y"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.x_edges, r.y_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = PmftXyt)]
pub struct PmftXyt {
    inner: molrs::compute::PmftXyt,
}

#[wasm_bindgen(js_class = PmftXyt)]
impl PmftXyt {
    /// Body-frame window `±x_max × ±y_max` (A); `n_x × n_y × n_t` bins over
    /// `(x, y, θ)` where `θ` is the relative orientation.
    #[wasm_bindgen(constructor)]
    pub fn new(x_max: F, y_max: F, n_x: usize, n_y: usize, n_t: usize) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PmftXyt::new(x_max, y_max, n_x, n_y, n_t)
                .map_err(|e| JsValue::from_str(&format!("PmftXyt: {e}")))?,
        })
    }

    /// Requires per-atom orientation angles on the frame.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations = vec![require_angles(rs_frame, "PmftXyt")?];
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PmftXytArgs {
                        nlists,
                        orientations: &orientations,
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PmftXyt compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PmftXyt: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["x", "y", "theta"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.x_edges, r.y_edges, r.t_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}

#[wasm_bindgen(js_name = PmftXyz)]
pub struct PmftXyz {
    inner: molrs::compute::PmftXyz,
}

#[wasm_bindgen(js_class = PmftXyz)]
impl PmftXyz {
    /// Body-frame window `±x_max × ±y_max × ±z_max` (A); `n_x × n_y × n_z` bins.
    #[wasm_bindgen(constructor)]
    pub fn new(
        x_max: F,
        y_max: F,
        z_max: F,
        n_x: usize,
        n_y: usize,
        n_z: usize,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::PmftXyz::new(x_max, y_max, z_max, n_x, n_y, n_z)
                .map_err(|e| JsValue::from_str(&format!("PmftXyz: {e}")))?,
        })
    }

    /// Uses the frame's per-atom quaternions when present; otherwise every
    /// query particle is treated as unrotated.
    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let orientations =
                molrs::compute::orientation_quaternions(rs_frame).map(|quats| vec![quats]);
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(
                    &[rs_frame],
                    molrs::compute::PmftXyzArgs {
                        nlists,
                        query_orientations: orientations.as_deref(),
                    },
                )
                .map_err(|e| JsValue::from_str(&format!("PmftXyz compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("PmftXyz: empty result"))?;
            js_value(&PmftOut {
                axes: vec!["x", "y", "z"],
                shape: r.density.shape().to_vec(),
                edges: vec![r.x_edges, r.y_edges, r.z_edges],
                density: r.density.iter().copied().collect(),
                raw_counts: r.raw_counts.iter().copied().collect(),
                pmf: r.pmf.iter().copied().collect(),
            })
        })
    }
}
