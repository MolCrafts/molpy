//! Domain (microheterogeneity) analysis over a radical-Voronoi tessellation.

use molrs::op::F;

use super::cell::VoronoiCells;
use crate::compute::{Compute, ComputeError, Partition};
use crate::core::Topology;

/// Outcome of a [`VoronoiDomainAnalysis`].
///
/// The domains are themselves a [`Partition`] of the cells (key = domain id),
/// so they feed any group reduction directly.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct VoronoiDomainResult {
    /// Domain sizes (cells per domain), descending.
    pub sizes: Vec<usize>,
    /// Number of domains.
    pub count: usize,
    /// Fraction of the grouped cells that lie in the largest domain.
    pub largest_fraction: F,
    /// Domain id per cell — contiguous, 0-based, numbered in order of each
    /// domain's lowest cell — or `None` for a cell the input partition left in
    /// no group.
    pub domain_of: Vec<Option<usize>>,
}

impl Partition for VoronoiDomainResult {
    fn n_items(&self) -> usize {
        self.domain_of.len()
    }

    fn group_of(&self, item: usize) -> Option<u64> {
        self.domain_of[item].map(|d| d as u64)
    }
}

/// Partition cells into same-group face-adjacent domains.
///
/// Configuration: none. Data: the tessellation and a [`Partition`] of its
/// cells — a label array (`&[u32]`, an `Array1<u32>` column read from a frame)
/// or an upstream result such as a [`ClusterResult`](crate::compute::ClusterResult).
///
/// Merges face-adjacent cells of the same group into connected domains via the
/// native connected-components over the cell-adjacency graph — the
/// aggregation reference implementation performs in `src/domain.cpp` /
/// `src/posdomain.cpp` (e.g. polar vs. apolar domains in ionic liquids). A
/// cell in no group joins no domain. Returns the domain size distribution,
/// count, and largest-domain fraction.
///
/// ```
/// use molrs::compute::{
///     Compute, RadicalVoronoi, RadicalVoronoiInput, VoronoiDomainAnalysis, VoronoiDomainInput,
/// };
/// use molrs::core::SimBox;
/// use ndarray::array;
///
/// // Four generators on a line: the two on the left are polar (0), the two
/// // on the right apolar (1).
/// let simbox = SimBox::cube(8.0, array![0.0, 0.0, 0.0], [true; 3]).unwrap();
/// let positions = array![[1.0, 4.0, 4.0], [3.0, 4.0, 4.0], [5.0, 4.0, 4.0], [7.0, 4.0, 4.0]];
/// let cells = RadicalVoronoi::new().compute(RadicalVoronoiInput {
///     positions: positions.view(),
///     simbox: &simbox,
///     radii: array![0.0, 0.0, 0.0, 0.0].view(),
/// })?;
/// let domains = VoronoiDomainAnalysis::new().compute(VoronoiDomainInput {
///     cells: &cells,
///     partition: &array![0u32, 0, 1, 1],
/// })?;
/// assert_eq!(domains.sizes, vec![2, 2]);
/// # Ok::<(), molrs::compute::ComputeError>(())
/// ```
#[derive(Debug, Clone, Copy, Default)]
pub struct VoronoiDomainAnalysis;

/// The data a [`VoronoiDomainAnalysis`] aggregates.
#[derive(Clone, Copy)]
pub struct VoronoiDomainInput<'a> {
    /// The tessellation.
    pub cells: &'a VoronoiCells,
    /// The group of each cell; face-adjacent cells of one group merge.
    pub partition: &'a (dyn Partition + Sync),
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
    /// [`ComputeError::DimensionMismatch`] unless the partition covers one item
    /// per cell; [`ComputeError::EmptyInput`] when no cell is in a group (an
    /// empty tessellation included) — the largest fraction is undefined.
    fn compute(&self, input: VoronoiDomainInput<'_>) -> Result<VoronoiDomainResult, ComputeError> {
        let VoronoiDomainInput { cells, partition } = input;
        let n = cells.len();
        if partition.n_items() != n {
            return Err(ComputeError::DimensionMismatch {
                expected: n,
                got: partition.n_items(),
                what: "domain partition items",
            });
        }
        let group: Vec<Option<u64>> = (0..n).map(|i| partition.group_of(i)).collect();
        let n_grouped = group.iter().filter(|g| g.is_some()).count();
        if n_grouped == 0 {
            return Err(ComputeError::EmptyInput);
        }
        // Build the same-group cell-adjacency graph and cluster it with the
        // native connected-components facility: each cell is a node, and a
        // face-adjacency between two cells of one group is an edge.
        let mut topo = Topology::with_atoms(n);
        for i in 0..n {
            let Some(gi) = group[i] else { continue };
            for j in cells.neighbors(i) {
                let j = j as usize;
                if j < n && j > i && group[j] == Some(gi) {
                    topo.add_bond(i, j);
                }
            }
        }
        let component_of = topo.connected_components();

        // `connected_components` numbers components in order of their lowest
        // cell, ungrouped (isolated) cells included; renumber the grouped ones
        // so domain ids stay contiguous in that same order.
        let mut domain_id = vec![None; n];
        let mut domain_of = vec![None; n];
        let mut size_of: Vec<usize> = Vec::new();
        for i in 0..n {
            if group[i].is_none() {
                continue;
            }
            let c = component_of[i] as usize;
            let d = *domain_id[c].get_or_insert_with(|| {
                size_of.push(0);
                size_of.len() - 1
            });
            domain_of[i] = Some(d);
            size_of[d] += 1;
        }

        let mut sizes = size_of;
        sizes.sort_unstable_by(|a, b| b.cmp(a));
        let count = sizes.len();
        // n_grouped > 0, so there is at least one domain.
        let largest_fraction = sizes[0] as F / n_grouped as F;

        Ok(VoronoiDomainResult {
            sizes,
            count,
            largest_fraction,
            domain_of,
        })
    }
}

crate::compute::spec::compute_spec! {
    VoronoiDomainAnalysis as "VoronoiDomainAnalysis";
    params |s| {}
    inputs {
        cells: Upstream("VoronoiCells");
        partition: Partition;
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
                partition: &[0u32, 0, 1, 0].as_slice(),
            })
            .unwrap();
        // 0–1 and 3–0 merge (3 cells); 2 stands alone.
        assert_eq!(r.sizes, vec![3, 1]);
        assert_eq!(r.count, 2);
        assert!((r.largest_fraction - 0.75).abs() < 1e-15);
        assert_eq!(r.domain_of[0], r.domain_of[3]);
        assert_ne!(r.domain_of[0], r.domain_of[2]);
        assert_eq!(r.domain_of, vec![Some(0), Some(0), Some(1), Some(0)]);
    }

    /// A partition that leaves some cells ungrouped.
    struct PartialLabels(Vec<Option<u32>>);

    impl Partition for PartialLabels {
        fn n_items(&self) -> usize {
            self.0.len()
        }
        fn group_of(&self, i: usize) -> Option<u64> {
            self.0[i].map(u64::from)
        }
    }

    /// An ungrouped cell joins no domain and does not bridge its neighbours;
    /// the largest fraction counts grouped cells only.
    #[test]
    fn an_ungrouped_cell_is_in_no_domain() {
        let cells = ring();
        let r = VoronoiDomainAnalysis::new()
            .compute(VoronoiDomainInput {
                cells: &cells,
                partition: &PartialLabels(vec![Some(0), None, Some(0), Some(0)]),
            })
            .unwrap();
        // 2–3 and 3–0 merge; 1 is in no domain.
        assert_eq!(r.sizes, vec![3]);
        assert_eq!(r.count, 1);
        assert!((r.largest_fraction - 1.0).abs() < 1e-15);
        assert_eq!(r.domain_of, vec![Some(0), None, Some(0), Some(0)]);
        assert_eq!(r.group_of(1), None);

        let none = PartialLabels(vec![None; 4]);
        assert!(matches!(
            VoronoiDomainAnalysis::new().compute(VoronoiDomainInput {
                cells: &cells,
                partition: &none,
            }),
            Err(ComputeError::EmptyInput)
        ));
    }

    /// The domains are a partition of the cells: their sizes are the domain
    /// sizes.
    #[test]
    fn the_domains_are_a_partition() {
        let cells = ring();
        let r = VoronoiDomainAnalysis::new()
            .compute(VoronoiDomainInput {
                cells: &cells,
                partition: &[0u32, 0, 1, 0].as_slice(),
            })
            .unwrap();
        let sizes: Vec<usize> = r.sizes().into_values().collect();
        assert_eq!(sizes, vec![3, 1]);
    }

    #[test]
    fn errors() {
        let cells = ring();
        assert!(matches!(
            VoronoiDomainAnalysis::new().compute(VoronoiDomainInput {
                cells: &cells,
                partition: &[0u32, 0].as_slice(),
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
                partition: &Vec::<u32>::new(),
            }),
            Err(ComputeError::EmptyInput)
        ));
    }
}
