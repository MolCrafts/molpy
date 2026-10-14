//! Hydrogen bonds, their lifetimes and networks — WASM face of the
//! `molrs::compute` hbond family.

use super::{js_value, u32_pairs, usize_pairs};
use crate::core::frame::Frame;
use molrs::compute::LegacyCompute;
use molrs::op::F;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name = HBonds)]
pub struct HBonds {
    donors: Vec<(u32, u32)>,
    acceptors: Vec<u32>,
    dist_cutoff: F,
    dist_kind: String,
    angle_cutoff: F,
    frames: Vec<molrs::core::Frame>,
}

#[wasm_bindgen(js_class = HBonds)]
impl HBonds {
    #[wasm_bindgen(constructor)]
    pub fn new(
        donors: &[u32],
        acceptors: &[u32],
        dist_cutoff: Option<F>,
        dist_kind: Option<String>,
        angle_cutoff: Option<F>,
    ) -> Result<Self, JsValue> {
        Ok(Self {
            donors: u32_pairs(donors, "HBonds donors")?,
            acceptors: acceptors.to_vec(),
            dist_cutoff: dist_cutoff.unwrap_or(3.5),
            dist_kind: dist_kind.unwrap_or_else(|| "donor_acceptor".to_string()),
            angle_cutoff: angle_cutoff.unwrap_or(150.0),
            frames: Vec::new(),
        })
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
        struct BondOut {
            donor: u32,
            hydrogen: u32,
            acceptor: u32,
            distance: F,
            angle: F,
        }
        #[derive(Serialize)]
        #[serde(rename_all = "camelCase")]
        struct Out {
            per_frame: Vec<Vec<BondOut>>,
            counts: Vec<usize>,
        }
        let dist_kind = match self.dist_kind.to_ascii_lowercase().as_str() {
            "donor_acceptor" | "donor-acceptor" | "da" => {
                molrs::compute::HBondDistanceKind::DonorAcceptor
            }
            "hydrogen_acceptor" | "hydrogen-acceptor" | "ha" => {
                molrs::compute::HBondDistanceKind::HydrogenAcceptor
            }
            other => {
                return Err(JsValue::from_str(&format!(
                    "HBonds distKind: unknown {other}"
                )));
            }
        };
        let criterion =
            molrs::compute::HBondCriterion::new(self.dist_cutoff, dist_kind, self.angle_cutoff);
        let calc =
            molrs::compute::HBonds::new(self.donors.clone(), self.acceptors.clone(), criterion);
        let refs: Vec<&molrs::core::Frame> = self.frames.iter().collect();
        let r = calc
            .compute(&refs, ())
            .map_err(|e| JsValue::from_str(&format!("HBonds: {e}")))?;
        let per_frame = r
            .per_frame
            .into_iter()
            .map(|frame| {
                frame
                    .into_iter()
                    .map(|b| BondOut {
                        donor: b.donor,
                        hydrogen: b.hydrogen,
                        acceptor: b.acceptor,
                        distance: b.distance,
                        angle: b.angle,
                    })
                    .collect()
            })
            .collect();
        js_value(&Out {
            per_frame,
            counts: r.counts,
        })
    }

    pub fn reset(&mut self) {
        self.frames.clear();
    }
}

/// Continuous and intermittent hydrogen-bond lifetime correlations —
/// molrs `compute::hbond_lifetimes` (an `HBondLifetimeResult`:
/// `{ lagTimes, continuous, intermittent, tauContinuous, tauIntermittent }`).
///
/// `presence` is the flat `(nBonds, nFrames)` 0/1 matrix, one row per bond.
#[wasm_bindgen(js_name = hbondLifetimes)]
pub fn hbond_lifetimes(
    presence: &[u8],
    n_bonds: usize,
    n_frames: usize,
    dt: F,
    max_lag: usize,
) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Out {
        lag_times: Vec<F>,
        continuous: Vec<F>,
        intermittent: Vec<F>,
        tau_continuous: F,
        tau_intermittent: F,
    }
    if presence.len() != n_bonds * n_frames {
        return Err(JsValue::from_str(
            "hbondLifetimes: presence length is not nBonds * nFrames",
        ));
    }
    let present: Vec<Vec<bool>> = presence
        .chunks_exact(n_frames)
        .map(|row| row.iter().map(|&v| v != 0).collect())
        .collect();
    let r = molrs::compute::hbond_lifetimes(&present, dt, max_lag)
        .map_err(|e| JsValue::from_str(&format!("hbondLifetimes: {e}")))?;
    js_value(&Out {
        lag_times: r.lag_times.to_vec(),
        continuous: r.continuous.to_vec(),
        intermittent: r.intermittent.to_vec(),
        tau_continuous: r.tau_continuous,
        tau_intermittent: r.tau_intermittent,
    })
}

/// Connected components of a hydrogen-bond network — molrs
/// `compute::hbond_components` (an `HBondNetworkResult`:
/// `{ componentSizes, nComponents }`). `edges` is flat `[a0, b0, a1, b1, …]`.
#[wasm_bindgen(js_name = hbondComponents)]
pub fn hbond_components(n_nodes: usize, edges: &[u32]) -> Result<JsValue, JsValue> {
    #[derive(Serialize)]
    #[serde(rename_all = "camelCase")]
    struct Out {
        component_sizes: Vec<usize>,
        n_components: usize,
    }
    let edges = usize_pairs(edges, "hbondComponents edges")?;
    let r = molrs::compute::hbond_components(n_nodes, &edges);
    js_value(&Out {
        component_sizes: r.component_sizes,
        n_components: r.n_components,
    })
}
