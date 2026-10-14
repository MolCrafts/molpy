//! Domain (microheterogeneity) analysis over a radical-Voronoi tessellation.

use molrs::op::F;

use super::cell::VoronoiCells;
use crate::compute::{Compute, ComputeError};
use crate::core::Topology;

/// Outcome of a [`VoronoiDomainAnalysis`].
#[derive(Debug, Clone)]
pub struct VoronoiDomainResult {
    /// Domain sizes (atoms per domain), descending.
    pub sizes: Vec<usize>,
    /// Number of domains.
    pub count: usize,
    /// Fraction of labelled atoms in the largest domain.
    pub largest_fraction: F,
    /// Domain id (0-based component label) per cell.
    pub domain_of: Vec<usize>,
}

/// Partition cells into same-label face-adjacent domains.
///
/// Configuration: none. Data: the tessellation and one label per cell.
///
/// Merges face-adjacent cells that share the same user label into connected
/// domains via the native connected-components over the cell-adjacency graph —
/// the aggregation reference implementation performs in `src/domain.cpp` /
/// `src/posdomain.cpp` (e.g. polar vs. apolar domains in ionic liquids).
/// Returns the domain size distribution, count, and largest-domain fraction.
#[derive(Debug, Clone, Copy, Default)]
pub struct VoronoiDomainAnalysis;

/// The data a [`VoronoiDomainAnalysis`] aggregates.
#[derive(Clone, Copy)]
pub struct VoronoiDomainInput<'a> {
    /// The tessellation.
    pub cells: &'a VoronoiCells,
    /// One label per cell; face-adjacent cells with equal labels merge.
    pub labels: &'a [i64],
}

impl VoronoiDomainAnalysis {
    /// No configuration.
    pub fn new() -> Self {
        Self
    }
}

impl Compute for VoronoiDomainAnalysis {
    type Input<'a> = VoronoiDomainInput<'a>;
    type Output = VoronoiDomainResult;

    /// # Errors
    /// [`ComputeError::DimensionMismatch`] unless there is one label per cell;
    /// [`ComputeError::EmptyInput`] for an empty tessellation (the largest
    /// fraction is undefined).
    fn compute(&self, input: VoronoiDomainInput<'_>) -> Result<VoronoiDomainResult, ComputeError> {
        let VoronoiDomainInput { cells, labels } = input;
        let n = cells.len();
        if labels.len() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: labels.len(),
                what: "domain labels length",
            });
        }
        if n == 0 {
            return Err(ComputeError::EmptyInput);
        }
        // Build the same-label cell-adjacency graph and cluster it with the
        // native connected-components (BFS) facility instead of a local
        // union-find: each cell is a node, and a face-adjacency between two
        // same-label cells is an edge.
        let mut topo = Topology::with_atoms(n);
        for i in 0..n {
            for j in cells.neighbors(i) {
                let j = j as usize;
                if j < n && j > i && labels[i] == labels[j] {
                    topo.add_bond(i, j);
                }
            }
        }
        let component_of = topo.connected_components();

        // `connected_components` labels every cell in `0..n` with a contiguous
        // 0-based component id (isolated cells get their own), so a flat `Vec`
        // keyed by that id tallies domain sizes without hashing. Connected
        // components are a graph invariant, so the partition — and thus the size
        // multiset — is the same under any labelling.
        let mut domain_of = vec![0usize; n];
        let mut size_of = vec![0usize; n];
        for (i, d) in domain_of.iter_mut().enumerate() {
            let c = component_of[i] as usize;
            *d = c;
            size_of[c] += 1;
        }

        let mut sizes: Vec<usize> = size_of.into_iter().filter(|&c| c > 0).collect();
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let count = sizes.len();
        // n > 0, so there is at least one domain.
        let largest_fraction = sizes[0] as F / n as F;

        Ok(VoronoiDomainResult {
            sizes,
            count,
            largest_fraction,
            domain_of,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compute::VoronoiFace;

    /// A ring of four cells, each face-adjacent to its two ring neighbours.
    fn ring() -> VoronoiCells {
        let face = |neighbor| VoronoiFace {
            area: 1.0,
            neighbor,
        };
        VoronoiCells {
            volumes: vec![1.0; 4],
            faces: (0..4i64)
                .map(|i| vec![face((i + 1) % 4), face((i + 3) % 4)])
                .collect(),
        }
    }

    #[test]
    fn same_label_neighbours_merge() {
        let cells = ring();
        let r = VoronoiDomainAnalysis::new()
            .compute(VoronoiDomainInput {
                cells: &cells,
                labels: &[0, 0, 1, 0],
            })
            .unwrap();
        // 0–1 and 3–0 merge (3 cells); 2 stands alone.
        assert_eq!(r.sizes, vec![3, 1]);
        assert_eq!(r.count, 2);
        assert!((r.largest_fraction - 0.75).abs() < 1e-15);
        assert_eq!(r.domain_of[0], r.domain_of[3]);
        assert_ne!(r.domain_of[0], r.domain_of[2]);
    }

    #[test]
    fn errors() {
        let cells = ring();
        assert!(matches!(
            VoronoiDomainAnalysis::new().compute(VoronoiDomainInput {
                cells: &cells,
                labels: &[0, 0],
            }),
            Err(ComputeError::DimensionMismatch { .. })
        ));
        let empty = VoronoiCells {
            volumes: vec![],
            faces: vec![],
        };
        assert!(matches!(
            VoronoiDomainAnalysis::new().compute(VoronoiDomainInput {
                cells: &empty,
                labels: &[],
            }),
            Err(ComputeError::EmptyInput)
        ));
    }
}
