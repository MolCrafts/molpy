//! Probe-accessible volume and the pore-size distribution — the WASM face of
//! `molrs::compute::porosity`.
//!
//! ```js
//! const free = new FreeVolume(0.0, 20000, 1).compute({ positions, radii, simbox });
//! const pores = new PoreSizeDistribution(0.5, 0.1).compute({ positions, radii, simbox });
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::Given;
use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// Probe-accessible volume. `bondi_fraction` is present only for a point
    /// probe (`probe_radius = 0`).
    FreeVolumeResult => mc::FreeVolumeResult
}

output! {
    /// Pore radii (local thickness, Å), the cell-volume fraction in each bin,
    /// and the cumulative fraction above each bin's lower edge.
    PoreSizeDistributionResult => mc::PoreSizeDistributionResult
}

/// Monte Carlo accessible volume of a hard probe.
#[wasm_bindgen]
pub struct FreeVolume {
    inner: mc::FreeVolume,
}

#[wasm_bindgen]
impl FreeVolume {
    /// `probeRadius` is in Å. `nSamples` and `seed` keep the core default
    /// when omitted.
    #[wasm_bindgen(constructor)]
    pub fn new(
        probe_radius: F,
        n_samples: Option<usize>,
        seed: Option<u64>,
    ) -> Result<FreeVolume, JsError> {
        let inner = mc::FreeVolume::new(probe_radius)?
            .given(n_samples, |c, v| c.with_n_samples(v))?
            .given(seed, |c, v| c.with_seed(v))?;
        Ok(Self { inner })
    }

    /// `{ positions, radii, simbox }`.
    pub fn compute(&self, data: &JsValue) -> Result<FreeVolumeResult, JsError> {
        let data = Fields::new(data, "FreeVolume")?;
        let positions = data.matrix("positions", Some(3))?;
        let radii = data.reals("radii")?;
        let simbox = data.simbox("simbox")?;
        let input = mc::FreeVolumeInput {
            positions: positions.view(),
            radii: radii.view(),
            simbox: &simbox,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Local-thickness pore-size distribution on an orthogonal grid.
#[wasm_bindgen]
pub struct PoreSizeDistribution {
    inner: mc::PoreSizeDistribution,
}

#[wasm_bindgen]
impl PoreSizeDistribution {
    /// `spacing` is the voxel edge in Å. `binWidth` keeps the core default
    /// when omitted. The histogram is in radius; a diameter is twice that.
    #[wasm_bindgen(constructor)]
    pub fn new(spacing: F, bin_width: Option<F>) -> Result<PoreSizeDistribution, JsError> {
        let inner =
            mc::PoreSizeDistribution::new(spacing)?.given(bin_width, |c, v| c.with_bin_width(v))?;
        Ok(Self { inner })
    }

    /// `{ positions, radii, simbox }`. A triclinic box is refused.
    pub fn compute(&self, data: &JsValue) -> Result<PoreSizeDistributionResult, JsError> {
        let data = Fields::new(data, "PoreSizeDistribution")?;
        let positions = data.matrix("positions", Some(3))?;
        let radii = data.reals("radii")?;
        let simbox = data.simbox("simbox")?;
        let input = mc::PoreSizeDistributionInput {
            positions: positions.view(),
            radii: radii.view(),
            simbox: &simbox,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    FreeVolume => mc::FreeVolume,
    PoreSizeDistribution => mc::PoreSizeDistribution,
);
