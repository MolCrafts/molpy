//! Density fields and correlation functions — WASM face of the
//! `molrs::compute` density family.

use super::{Grid3Out, array2, js_value, usize_pairs};
use crate::core::frame::Frame;
use crate::core::neighbors::Neighbors;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

fn usize_vec(data: &[u32]) -> Vec<usize> {
    data.iter().map(|&v| v as usize).collect()
}

#[wasm_bindgen(js_name = CorrelationFunction)]
pub struct CorrelationFunction {
    inner: molrs::compute::CorrelationFunction,
}

#[wasm_bindgen(js_class = CorrelationFunction)]
impl CorrelationFunction {
    #[wasm_bindgen(constructor)]
    pub fn new(n_bins: usize, r_max: F, r_min: Option<F>) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::CorrelationFunction::new(n_bins, r_max, r_min.unwrap_or(0.0))
                .map_err(|e| JsValue::from_str(&format!("CorrelationFunction: {e}")))?,
        })
    }

    pub fn compute(
        &self,
        frame: &Frame,
        neighbors: &Neighbors,
        values_a: &[F],
        values_b: &[F],
    ) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            bin_edges: Vec<F>,
            bin_centers: Vec<F>,
            bin_counts: Vec<u64>,
            correlation: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let va = vec![values_a.to_vec()];
            let vb = vec![values_b.to_vec()];
            let args = molrs::compute::CorrelationFunctionArgs {
                nlists,
                values_a: &va,
                values_b: &vb,
            };
            let mut out = self
                .inner
                .compute(&[rs_frame], args)
                .map_err(|e| JsValue::from_str(&format!("CorrelationFunction compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("CorrelationFunction: empty result"))?;
            js_value(&Out {
                bin_edges: r.bin_edges.to_vec(),
                bin_centers: r.bin_centers.to_vec(),
                bin_counts: r.bin_counts.to_vec(),
                correlation: r.correlation.to_vec(),
            })
        })
    }
}

#[wasm_bindgen(js_name = LocalDensity)]
pub struct LocalDensity {
    inner: molrs::compute::LocalDensity,
}

#[wasm_bindgen(js_class = LocalDensity)]
impl LocalDensity {
    #[wasm_bindgen(constructor)]
    pub fn new(r_max: F, diameter: Option<F>) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::LocalDensity::new(r_max)
            .map_err(|e| JsValue::from_str(&format!("LocalDensity: {e}")))?;
        if let Some(d) = diameter {
            inner = inner.with_diameter(d);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame, neighbors: &Neighbors) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            n_neighbors: Vec<F>,
            density: Vec<F>,
        }
        frame.with_frame(|rs_frame| {
            let nlists = std::slice::from_ref(&neighbors.inner);
            let mut out = self
                .inner
                .compute(&[rs_frame], nlists)
                .map_err(|e| JsValue::from_str(&format!("LocalDensity compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("LocalDensity: empty result"))?;
            js_value(&Out {
                n_neighbors: r.n_neighbors,
                density: r.density,
            })
        })
    }
}

#[wasm_bindgen(js_name = GaussianDensity)]
pub struct GaussianDensity {
    inner: molrs::compute::GaussianDensity,
}

#[wasm_bindgen(js_class = GaussianDensity)]
impl GaussianDensity {
    #[wasm_bindgen(constructor)]
    pub fn new(
        nx: usize,
        ny: usize,
        nz: usize,
        sigma: F,
        r_max: Option<F>,
    ) -> Result<Self, JsValue> {
        let mut inner = molrs::compute::GaussianDensity::new(nx, ny, nz, sigma)
            .map_err(|e| JsValue::from_str(&format!("GaussianDensity: {e}")))?;
        if let Some(r) = r_max {
            inner = inner.with_r_max(r);
        }
        Ok(Self { inner })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        frame.with_frame(|rs_frame| {
            let mut out = self
                .inner
                .compute(&[rs_frame], ())
                .map_err(|e| JsValue::from_str(&format!("GaussianDensity compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("GaussianDensity: empty result"))?;
            let shape = [
                r.density.shape()[0],
                r.density.shape()[1],
                r.density.shape()[2],
            ];
            js_value(&Grid3Out {
                data: r.density.iter().copied().collect::<Vec<F>>(),
                shape,
            })
        })
    }
}

#[wasm_bindgen(js_name = SphereVoxelization)]
pub struct SphereVoxelization {
    inner: molrs::compute::SphereVoxelization,
}

#[wasm_bindgen(js_class = SphereVoxelization)]
impl SphereVoxelization {
    #[wasm_bindgen(constructor)]
    pub fn new(nx: usize, ny: usize, nz: usize, r_max: F) -> Result<Self, JsValue> {
        Ok(Self {
            inner: molrs::compute::SphereVoxelization::new(nx, ny, nz, r_max)
                .map_err(|e| JsValue::from_str(&format!("SphereVoxelization: {e}")))?,
        })
    }

    pub fn compute(&self, frame: &Frame) -> Result<JsValue, JsValue> {
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            voxels: Grid3Out<u8>,
            raw_counts: Grid3Out<u32>,
        }
        frame.with_frame(|rs_frame| {
            let mut out = self
                .inner
                .compute(&[rs_frame], ())
                .map_err(|e| JsValue::from_str(&format!("SphereVoxelization compute: {e}")))?;
            let r = out
                .pop()
                .ok_or_else(|| JsValue::from_str("SphereVoxelization: empty result"))?;
            let shape = [
                r.voxels.shape()[0],
                r.voxels.shape()[1],
                r.voxels.shape()[2],
            ];
            js_value(&Out {
                voxels: Grid3Out {
                    data: r.voxels.iter().copied().collect::<Vec<u8>>(),
                    shape,
                },
                raw_counts: Grid3Out {
                    data: r.raw_counts.iter().copied().collect::<Vec<u32>>(),
                    shape,
                },
            })
        })
    }
}

#[wasm_bindgen(js_name = SpatialDistribution)]
pub struct SpatialDistribution {
    inner: molrs::compute::SpatialDistribution,
    frames: Vec<molrs::core::Frame>,
    bulk_density: Option<F>,
}

#[wasm_bindgen(js_class = SpatialDistribution)]
impl SpatialDistribution {
    // The JS constructor is positional by wasm-bindgen's design; molvis calls
    // it with these ten arguments, so the shape is the public contract.
    #[allow(clippy::too_many_arguments)]
    #[wasm_bindgen(constructor)]
    pub fn new(
        reference: &[u32],
        template: &[F],
        target: &[u32],
        nx: usize,
        ny: usize,
        nz: usize,
        extent_x: F,
        extent_y: F,
        extent_z: F,
        bulk_density: Option<F>,
    ) -> Result<Self, JsValue> {
        let reference_usize = usize_vec(reference);
        let target_usize = usize_vec(target);
        let template = array2(
            template,
            reference_usize.len(),
            3,
            "SpatialDistribution template",
        )?;
        let grid = molrs::compute::GridSpec {
            n: [nx, ny, nz],
            extent: [extent_x, extent_y, extent_z],
        };
        let mut inner =
            molrs::compute::SpatialDistribution::new(reference_usize, template, target_usize, grid)
                .map_err(|e| JsValue::from_str(&format!("SpatialDistribution: {e}")))?;
        if let Some(rho) = bulk_density {
            inner = inner.with_bulk_density(rho);
        }
        Ok(Self {
            inner,
            frames: Vec::new(),
            bulk_density,
        })
    }

    #[wasm_bindgen(js_name = setOrientationPairs)]
    pub fn set_orientation_pairs(&mut self, pairs: &[u32]) -> Result<(), JsValue> {
        self.inner = self
            .inner
            .clone()
            .with_orientation(usize_pairs(pairs, "SpatialDistribution orientationPairs")?);
        Ok(())
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
            counts: Grid3Out<F>,
            density: Grid3Out<F>,
            g_sdf: Option<Grid3Out<F>>,
            orientation: Option<Vec<F>>,
            orientation_shape: Option<[usize; 4]>,
            voxel_volume: F,
            n_frames: usize,
            bulk_density: Option<F>,
        }
        let refs: Vec<&molrs::core::Frame> = self.frames.iter().collect();
        let r = self
            .inner
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("SpatialDistribution compute: {e}")))?;
        let shape = [r.n[0], r.n[1], r.n[2]];
        let g_sdf = r.g_sdf.as_ref().map(|g| Grid3Out {
            data: g.iter().copied().collect::<Vec<F>>(),
            shape,
        });
        let (orientation, orientation_shape) = match r.orientation.as_ref() {
            Some(o) => (
                Some(o.iter().copied().collect::<Vec<F>>()),
                Some([r.n[0], r.n[1], r.n[2], 3]),
            ),
            None => (None, None),
        };
        js_value(&Out {
            counts: Grid3Out {
                data: r.counts.iter().copied().collect::<Vec<F>>(),
                shape,
            },
            density: Grid3Out {
                data: r.density.iter().copied().collect::<Vec<F>>(),
                shape,
            },
            g_sdf,
            orientation,
            orientation_shape,
            voxel_volume: r.voxel_volume,
            n_frames: r.n_frames,
            bulk_density: self.bulk_density,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}
