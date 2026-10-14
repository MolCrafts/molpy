//! Distance / angle / dihedral and combined distributions — WASM face of
//! the `molrs::compute` distribution family (`DistributionFunction`,
//! `CombinedDistribution`).

use super::js_value;
use crate::core::frame::Frame;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn distribution_compute<O: molrs::compute::Observable + Sync>(
    frame: &Frame,
    calc: &molrs::compute::DistributionFunction<O>,
    groups: molrs::compute::AtomGroups,
) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Out {
        bin_centers: Vec<F>,
        bin_edges: Vec<F>,
        counts: Vec<F>,
        density: Vec<F>,
        density_sin_corrected: Option<Vec<F>>,
        bin_width: F,
        n_binned: F,
        n_raw_samples: usize,
        n_frames: usize,
        angular: bool,
    }
    frame.with_frame(|rs_frame| {
        let r = calc
            .compute(&[rs_frame], &groups)
            .map_err(|e| JsValue::from_str(&format!("Distribution compute: {e}")))?;
        js_value(&Out {
            bin_centers: r.bin_centers.to_vec(),
            bin_edges: r.bin_edges.to_vec(),
            counts: r.counts.to_vec(),
            density: r.density.to_vec(),
            density_sin_corrected: r.density_sin_corrected.map(|v| v.to_vec()),
            bin_width: r.bin_width,
            n_binned: r.n_binned,
            n_raw_samples: r.n_raw_samples,
            n_frames: r.n_frames,
            angular: r.angular,
        })
    })
}

fn distribution_function<O: molrs::compute::Observable>(
    observable: O,
    n_bins: usize,
    bounds: Option<(F, F)>,
) -> Result<molrs::compute::DistributionFunction<O>, JsValue> {
    match bounds {
        Some((min, max)) => molrs::compute::DistributionFunction::new(observable, n_bins, min, max),
        None => molrs::compute::DistributionFunction::over_natural_range(observable, n_bins),
    }
    .map_err(|e| JsValue::from_str(&format!("DistributionFunction: {e}")))
}

/// The core calculator, one per observable type.
enum DistributionKernel {
    Distance(molrs::compute::DistributionFunction<molrs::compute::DistanceObservable>),
    Angle(molrs::compute::DistributionFunction<molrs::compute::AngleObservable>),
    Dihedral(molrs::compute::DistributionFunction<molrs::compute::DihedralObservable>),
}

/// One-dimensional distribution function of an internal coordinate — molrs
/// `compute::DistributionFunction<O>`, as Python's
/// `molrs.compute.DistributionFunction`.
///
/// `observable` is `"distance"` (atom pairs; `min` / `max` required, in the
/// coordinates' length unit), `"angle"` (triplets; radians, natural range
/// `[0, π]`) or `"dihedral"` (quadruplets; radians, natural range `(-π, π]`,
/// kept signed). Passing exactly one bound throws.
///
/// ```js
/// const adf = new DistributionFunction("angle", 90);
/// const r = adf.compute(frame, new Uint32Array([0, 1, 2, 1, 2, 3]));
/// ```
#[wasm_bindgen]
pub struct DistributionFunction {
    kernel: DistributionKernel,
    arity: usize,
}

#[wasm_bindgen]
impl DistributionFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(
        observable: &str,
        n_bins: usize,
        min: Option<F>,
        max: Option<F>,
    ) -> Result<DistributionFunction, JsValue> {
        use molrs::compute::InternalCoordinate as Ic;
        let (observable, arity) = Ic::from_kind(observable)
            .map_err(|e| JsValue::from_str(&format!("DistributionFunction: {e}")))?;
        let bounds = match (min, max) {
            (Some(min), Some(max)) => Some((min, max)),
            (None, None) => None,
            _ => {
                return Err(JsValue::from_str(
                    "DistributionFunction: pass both min and max, or neither",
                ));
            }
        };
        let kernel = match observable {
            Ic::Distance(o) => {
                DistributionKernel::Distance(distribution_function(o, n_bins, bounds)?)
            }
            Ic::Angle(o) => DistributionKernel::Angle(distribution_function(o, n_bins, bounds)?),
            Ic::Dihedral(o) => {
                DistributionKernel::Dihedral(distribution_function(o, n_bins, bounds)?)
            }
        };
        Ok(DistributionFunction { kernel, arity })
    }

    /// The distribution over `groups`, a flat atom-index array of
    /// `arity × nGroups` (2 for distance, 3 for angle, 4 for dihedral).
    pub fn compute(&self, frame: &Frame, groups: &[u32]) -> Result<JsValue, JsValue> {
        let groups =
            molrs::compute::AtomGroups::new(self.arity, groups.iter().map(|&v| v as u64).collect())
                .map_err(|e| JsValue::from_str(&format!("DistributionFunction groups: {e}")))?;
        match &self.kernel {
            DistributionKernel::Distance(calc) => distribution_compute(frame, calc, groups),
            DistributionKernel::Angle(calc) => distribution_compute(frame, calc, groups),
            DistributionKernel::Dihedral(calc) => distribution_compute(frame, calc, groups),
        }
    }
}

#[wasm_bindgen(js_name = CombinedDistribution)]
pub struct CombinedDistribution {
    kinds: Vec<String>,
    axes: Vec<molrs::compute::AxisSpec>,
}

#[wasm_bindgen(js_class = CombinedDistribution)]
impl CombinedDistribution {
    /// `kinds[i]` is `"distance" | "angle" | "dihedral"`; axis `i` bins
    /// observable `i` into `bins[i]` bins over `[mins[i], maxs[i]]`. A non-zero
    /// `sinWeight[i]` marks axis `i` angular so its marginal carries the
    /// sin θ solid-angle correction.
    #[wasm_bindgen(constructor)]
    pub fn new(
        kinds: Vec<String>,
        bins: &[u32],
        mins: &[F],
        maxs: &[F],
        sin_weight: Option<Vec<u8>>,
    ) -> Result<Self, JsValue> {
        let n = kinds.len();
        if bins.len() != n || mins.len() != n || maxs.len() != n {
            return Err(JsValue::from_str(
                "CombinedDistribution: kinds, bins, mins and maxs must have equal length",
            ));
        }
        let sin = sin_weight.unwrap_or_default();
        let axes = (0..n)
            .map(|i| {
                let spec = molrs::compute::AxisSpec::new(bins[i] as usize, mins[i], maxs[i])
                    .map_err(|e| {
                        JsValue::from_str(&format!("CombinedDistribution axis {i}: {e}"))
                    })?;
                Ok(spec.with_sin_weight(sin.get(i).is_some_and(|&v| v != 0)))
            })
            .collect::<Result<Vec<_>, JsValue>>()?;
        Ok(Self { kinds, axes })
    }

    /// `groups` is `number[][]`: one flat atom-index array per observable, each
    /// of length `arity × nGroups` (arity 2/3/4 for distance/angle/dihedral).
    pub fn compute(&self, frame: &Frame, groups: JsValue) -> Result<JsValue, JsValue> {
        use molrs::compute::{AtomGroups, InternalCoordinate};

        let raw: Vec<Vec<u32>> = serde_wasm_bindgen::from_value(groups)
            .map_err(|e| JsValue::from_str(&format!("CombinedDistribution groups: {e}")))?;
        if raw.len() != self.kinds.len() {
            return Err(JsValue::from_str(
                "CombinedDistribution: one atom-index group array per observable is required",
            ));
        }

        let mut observables = Vec::with_capacity(self.kinds.len());
        let mut atom_groups = Vec::with_capacity(self.kinds.len());
        for (i, kind) in self.kinds.iter().enumerate() {
            let (obs, arity) = InternalCoordinate::from_kind(kind).map_err(|e| {
                JsValue::from_str(&format!("CombinedDistribution observable {i}: {e}"))
            })?;
            observables.push(obs);
            atom_groups.push(
                AtomGroups::new(arity, raw[i].iter().map(|&v| v as u64).collect()).map_err(
                    |e| JsValue::from_str(&format!("CombinedDistribution groups {i}: {e}")),
                )?,
            );
        }
        let calc = molrs::compute::CombinedDistribution::new(observables, self.axes.clone())
            .map_err(|e| JsValue::from_str(&format!("CombinedDistribution: {e}")))?;

        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            shape: Vec<usize>,
            edges: Vec<Vec<F>>,
            centers: Vec<Vec<F>>,
            counts: Vec<F>,
            density: Vec<F>,
            n_binned: F,
            n_raw_samples: usize,
            n_frames: usize,
        }
        frame.with_frame(|rs_frame| {
            let r = calc
                .compute(&[rs_frame], &atom_groups)
                .map_err(|e| JsValue::from_str(&format!("CombinedDistribution compute: {e}")))?;
            js_value(&Out {
                shape: r.centers.iter().map(|c| c.len()).collect(),
                edges: r.edges.iter().map(|e| e.to_vec()).collect(),
                centers: r.centers.iter().map(|c| c.to_vec()).collect(),
                counts: r.counts.to_vec(),
                density: r.density.to_vec(),
                n_binned: r.binned,
                n_raw_samples: r.n_raw_samples,
                n_frames: r.n_frames,
            })
        })
    }
}
