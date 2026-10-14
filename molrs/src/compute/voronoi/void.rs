//! Void (cavity / free-volume) analysis over a radical-Voronoi tessellation.

use molrs::op::F;

use super::cell::VoronoiCells;
use crate::compute::{Compute, ComputeError};
use crate::core::Topology;

/// Outcome of a [`VoronoiVoidAnalysis`].
#[derive(Debug, Clone)]
pub struct VoronoiVoidResult {
    /// Cavity volumes (Å³), descending.
    pub cavity_volumes: Vec<F>,
    /// Total unoccupied (probe) volume (Å³).
    pub total_void_volume: F,
    /// Void fraction = total void volume / box volume.
    pub void_fraction: F,
}

/// Aggregate probe cells of a combined atom+probe tessellation into cavities.
///
/// Configuration: none. Data: the tessellation, the probe mask and the box
/// volume.
///
/// Following the reference implementation's domain-style void aggregation (`src/void.cpp`): a set of
/// **probe generators** is tessellated *together with* the atoms, and the cells
/// belonging to probes are the unoccupied regions. Face-adjacent probe cells are
/// merged (connected-components) into cavities; each cavity's volume is the sum
/// of its probe-cell volumes, and the total void fraction is the probe volume
/// over the box volume.
///
/// The caller builds one [`VoronoiCells`] over `atoms ++ probes` and passes a
/// boolean mask marking which generators are probes — keeping this a pure
/// consumer of the tessellation (no second geometry path).
#[derive(Debug, Clone, Copy, Default)]
pub struct VoronoiVoidAnalysis;

/// The data a [`VoronoiVoidAnalysis`] aggregates.
#[derive(Clone, Copy)]
pub struct VoronoiVoidInput<'a> {
    /// The tessellation of atoms and probes together.
    pub cells: &'a VoronoiCells,
    /// `is_void[i]` marks cell `i` as a void probe.
    pub is_void: &'a [bool],
    /// The box volume (Å³) that normalizes the void fraction.
    pub box_volume: F,
}

impl VoronoiVoidAnalysis {
    /// No configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for VoronoiVoidAnalysis {
    type Input<'a> = VoronoiVoidInput<'a>;
    type Output = VoronoiVoidResult;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] unless there is one mask entry per
    /// cell; [`ComputeError::OutOfRange`] for a non-positive or non-finite box
    /// volume.
    fn compute(&self, input: VoronoiVoidInput<'_>) -> Result<VoronoiVoidResult, ComputeError> {
        let VoronoiVoidInput {
            cells,
            is_void,
            box_volume,
        } = input;
        let n = cells.len();
        if is_void.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: is_void.len(),
                what: "void mask length",
            });
        }
        if !box_volume.is_finite() || box_volume <= 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "VoronoiVoidAnalysis::box_volume",
                value: box_volume.to_string(),
            });
        }

        // Build the void-cell adjacency graph and cluster it with the native
        // connected-components (BFS) facility instead of a local union-find:
        // only void cells are linked; non-void cells stay isolated and never
        // contribute.
        let mut topo = Topology::with_atoms(n);
        for (i, &vi) in is_void.iter().enumerate() {
            if !vi {
                continue;
            }
            for j in cells.neighbors(i) {
                let j = j as usize;
                if j < n && j > i && is_void[j] {
                    topo.add_bond(i, j);
                }
            }
        }
        let component_of = topo.connected_components();

        // Cavity ids are contiguous 0-based component labels, so flat `Vec`s
        // keyed by that label need no hashing. Connected components are a graph
        // invariant, so each cavity holds the same cells under any labelling,
        // and volumes are summed in ascending `i` order, so the floats are
        // deterministic; `seen` marks the labels that owned ≥ 1 probe cell.
        let mut vol_of = vec![0.0 as F; n];
        let mut seen = vec![false; n];
        let mut total = 0.0;
        for (i, &vi) in is_void.iter().enumerate() {
            if !vi {
                continue;
            }
            let c = component_of[i] as usize;
            vol_of[c] += cells.volumes[i];
            seen[c] = true;
            total += cells.volumes[i];
        }

        let mut cavity_volumes: Vec<F> = (0..n).filter(|&c| seen[c]).map(|c| vol_of[c]).collect();
        cavity_volumes.sort_unstable_by(|a, b| b.total_cmp(a));

        Ok(VoronoiVoidResult {
            cavity_volumes,
            total_void_volume: total,
            void_fraction: total / box_volume,
        })
    }
}

crate::compute::spec::compute_spec! {
    VoronoiVoidAnalysis as "VoronoiVoidAnalysis";
    params |s| {}
    inputs {
        cells: Upstream("VoronoiCells");
        is_void: Mask;
        box_volume: Scalar;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::VoronoiFace;

    /// A chain 0–1–2–3 with volumes 1, 2, 3, 4.
    fn chain() -> VoronoiCells {
        let face = |neighbor| VoronoiFace {
            area: 1.0,
            neighbor,
        };
        VoronoiCells {
            volumes: vec![1.0, 2.0, 3.0, 4.0],
            faces: vec![
                vec![face(1)],
                vec![face(0), face(2)],
                vec![face(1), face(3)],
                vec![face(2)],
            ],
        }
    }

    #[test]
    fn adjacent_probes_merge_into_one_cavity() {
        let cells = chain();
        let r = VoronoiVoidAnalysis::new()
            .compute(VoronoiVoidInput {
                cells: &cells,
                is_void: &[true, true, false, true],
                box_volume: 20.0,
            })
            .unwrap();
        assert_eq!(r.cavity_volumes, vec![4.0, 3.0]);
        assert_eq!(r.total_void_volume, 7.0);
        assert!((r.void_fraction - 0.35).abs() < 1e-15);
    }

    #[test]
    fn errors() {
        let cells = chain();
        let run = |is_void: &[bool], box_volume| {
            VoronoiVoidAnalysis::new().compute(VoronoiVoidInput {
                cells: &cells,
                is_void,
                box_volume,
            })
        };
        assert!(matches!(
            run(&[true], 1.0),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        assert!(matches!(
            run(&[true; 4], 0.0),
            Err(ComputeError::OutOfRange { .. })
        ));
    }
}
