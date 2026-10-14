//! Pair presence, hydrogen bonds and the van Hove function — the WASM face of
//! `molrs::compute::{dynamics, hbond}`.
//!
//! ```js
//! const bonds = new HBonds().compute({ positions: traj, donors, acceptors, pairs: tables });
//! const presence = bonds.presence();                    // Presence | undefined
//! const tcf = new Survival("continuous", 50).compute({
//!     presence: presence.toJSON().values, dt: 1.0,
//! });
//! ```

use molrs::compute::{self as mc, Compute};
use molrs::core::{Neighbors as RsNeighbors, PairSource};
use molrs::op::F;
use wasm_bindgen::prelude::*;

use super::config::{Axis, Given};
use super::handle::{described, output};
use super::wire::Fields;

output! {
    /// Presence per frame and pair: `pairs`, `values` (`T × n_pairs`, 0 / 1).
    /// Its `values` are `Survival`'s `presence`.
    Presence => mc::Presence
}

output! {
    /// The van Hove function: self and distinct parts per lag and distance.
    VanHoveResult => mc::VanHoveResult
}

output! {
    /// Hydrogen bonds per frame (`per_frame`), their `counts`, `n_atoms`.
    HBondsResult => mc::HBondsResult
}

#[wasm_bindgen]
impl HBondsResult {
    /// The bond-presence series over every bond ever seen; `undefined` when
    /// no bond formed.
    pub fn presence(&self) -> Option<Presence> {
        self.inner.presence().map(Presence::from)
    }

    /// The connected-component label of every atom in `frame`'s bond network.
    pub fn components(&self, frame: usize) -> Result<Vec<u32>, JsError> {
        Ok(self.inner.components(frame)?)
    }
}

/// Which pairs are within a hysteresis distance, frame by frame.
#[wasm_bindgen]
pub struct PairPresence {
    inner: mc::PairPresence,
}

#[wasm_bindgen]
impl PairPresence {
    /// A pair enters at `r0` and leaves beyond `r1`.
    #[wasm_bindgen(constructor)]
    pub fn new(r0: F, exclude_self: bool, r1: Option<F>) -> Result<PairPresence, JsError> {
        let inner = mc::PairPresence::new(r0, exclude_self)?.given(r1, |c, v| c.with_r1(v))?;
        Ok(Self { inner })
    }

    /// `{ frames }`: one `Neighbors` table per frame. `undefined` when no pair
    /// was ever present.
    pub fn compute(&self, data: &JsValue) -> Result<Option<Presence>, JsError> {
        let data = Fields::new(data, "PairPresence")?;
        let tables = data.neighbor_frames("frames")?;
        let frames: Vec<&(dyn PairSource + Sync)> = tables
            .iter()
            .map(|table| &**table as &(dyn PairSource + Sync))
            .collect();
        let input = mc::PairPresenceInput { frames: &frames };
        Ok(self.inner.compute(input)?.map(Presence::from))
    }
}

/// The van Hove correlation function at the given lags.
#[wasm_bindgen]
pub struct VanHove {
    inner: mc::VanHove,
}

#[wasm_bindgen]
impl VanHove {
    /// `origins` is an [`Origins`](mc::Origins) name.
    #[wasm_bindgen(constructor)]
    pub fn new(
        axis: &Axis,
        lags: Vec<usize>,
        origins: Option<String>,
        stride: Option<usize>,
    ) -> Result<VanHove, JsError> {
        let inner = mc::VanHove::new(axis.inner, lags)?
            .given(origins, |c, v| c.with_origins(v.parse()?))?
            .given(stride, |c, v| c.with_stride(v))?;
        Ok(Self { inner })
    }

    /// `{ positions, boxes }`: `positions` is `T × N × 3`.
    pub fn compute(&self, data: &JsValue) -> Result<VanHoveResult, JsError> {
        let data = Fields::new(data, "VanHove")?;
        let positions = data.cube("positions")?;
        let boxes = data.boxes("boxes")?;
        let input = mc::VanHoveInput {
            positions: positions.view(),
            boxes: &boxes,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

/// Geometric hydrogen-bond detection over a trajectory.
#[wasm_bindgen]
pub struct HBonds {
    inner: mc::HBonds,
}

#[wasm_bindgen]
impl HBonds {
    /// The criterion; `distKind` is an
    /// [`HBondDistanceKind`](mc::HBondDistanceKind) name.
    #[wasm_bindgen(constructor)]
    pub fn new(
        dist_cutoff: Option<F>,
        dist_kind: Option<String>,
        angle_cutoff: Option<F>,
        donor_acceptor_margin: Option<F>,
    ) -> Result<HBonds, JsError> {
        let criterion = mc::HBonds::default()
            .criterion()
            .given(dist_cutoff, |c, v| {
                Ok(mc::HBondCriterion {
                    dist_cutoff: v,
                    ..c
                })
            })?
            .given(dist_kind, |c, v| {
                Ok(mc::HBondCriterion {
                    dist_kind: v.parse()?,
                    ..c
                })
            })?
            .given(angle_cutoff, |c, v| {
                Ok(mc::HBondCriterion {
                    angle_cutoff: v,
                    ..c
                })
            })?
            .given(donor_acceptor_margin, |c, v| {
                Ok(c.with_donor_acceptor_margin(v))
            })?;
        Ok(Self {
            inner: mc::HBonds::new(criterion)?,
        })
    }

    /// `{ positions, donors, acceptors, pairs }`: `positions` `T × N × 3`,
    /// `donors` flat `(donor, hydrogen)` pairs, `pairs` one `Neighbors` per
    /// frame.
    pub fn compute(&self, data: &JsValue) -> Result<HBondsResult, JsError> {
        let data = Fields::new(data, "HBonds")?;
        let positions = data.cube("positions")?;
        let donors = data.pairs("donors")?;
        let acceptors = data.naturals("acceptors")?;
        let tables = data.neighbor_frames("pairs")?;
        let pairs: Vec<&RsNeighbors> = tables.iter().map(|table| &**table).collect();
        let input = mc::HBondsInput {
            positions: positions.view(),
            donors: &donors,
            acceptors: &acceptors,
            pairs: &pairs,
        };
        Ok(self.inner.compute(input)?.into())
    }
}

described!(
    PairPresence => mc::PairPresence,
    VanHove => mc::VanHove,
    HBonds => mc::HBonds,
);
