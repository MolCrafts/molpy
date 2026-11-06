//! Percolation of a bonded network under periodic boundaries.
//!
//! A connected component *percolates* when, walking its bonds, one can arrive
//! back at an atom in a different periodic image: the network is joined to its
//! own image and is infinite. Every closed walk carries an integer translation
//! (in units of the lattice vectors); the translations of all closed walks of a
//! component form a lattice, and its rank is the percolation dimension — 0 for a
//! finite cluster, 1 for a strand, 2 for a sheet, 3 for a network that spans the
//! box in every direction.
//!
//! Algorithm: Livraghi, Höllring, Wick, Smith & Smith, "An Exact Algorithm to
//! Detect the Percolation Transition in Molecular Dynamics Simulations of
//! Cross-Linking Polymer Networks", J. Chem. Theory Comput. 17, 6449 (2021),
//! doi:10.1021/acs.jctc.1c00423. A breadth-first walk gives every atom an image
//! offset; a bond closing a cycle contributes `offset_i + t_ij − offset_j`; the
//! contributions are reduced to a lattice basis (Hermite normal form), exactly,
//! in integers.
//!
//! The translation `t_ij` of a bond is the image of atom `j` that atom `i` is
//! bonded to: `t_ij = H⁻¹ (r_i + d_ij − r_j)` with `d_ij` the minimum-image bond
//! vector, as LAMMPS reads every bond. It is only defined for a bond shorter
//! than half the nearest plane distance; a longer bond is refused, never
//! guessed. Per-atom image flags cannot stand in: an atom of a percolating
//! network is bonded into two images at once, so no unwrapping exists.

use std::collections::VecDeque;

use molrs::core::SimBox;
use molrs::op::F;
use ndarray::{Array1, Array2, ArrayView2};

use crate::compute::{Compute, ComputeError};

/// Percolation analysis of a bonded network (Livraghi et al. 2021).
#[derive(Debug, Clone, Copy, Default)]
pub struct Percolation;

impl Percolation {
    /// The analysis has no parameters.
    pub fn new() -> Self {
        Self
    }
}

/// The data a [`Percolation`] analyses: one snapshot.
#[derive(Clone, Copy)]
pub struct PercolationInput<'a> {
    /// Atom positions, `N × 3` (Å), wrapped or not.
    pub positions: ArrayView2<'a, F>,
    /// The periodic cell.
    pub simbox: &'a SimBox,
    /// Bonds as `(i, j)` atom index pairs.
    pub bonds: &'a [[usize; 2]],
}

/// The components of a network and how each one percolates.
#[derive(Debug, Clone, PartialEq)]
pub struct PercolationResult {
    /// Component of every atom, numbered in order of their lowest atom.
    pub component_idx: Array1<i64>,
    /// Number of components.
    pub n_components: usize,
    /// Atoms per component.
    pub component_sizes: Vec<usize>,
    /// Percolation dimension of every component (0–3).
    pub percolation_dims: Vec<usize>,
    /// Basis of every component's translation lattice, in lattice-vector
    /// units, one row per independent direction (Hermite normal form).
    pub translations: Vec<Vec<[i64; 3]>>,
    /// The image each atom was reached in, `N × 3`: adding `H · image` to the
    /// positions makes every non-percolating component whole.
    pub unwrap_images: Array2<i64>,
}

impl PercolationResult {
    /// The largest percolation dimension over all components.
    pub fn max_dim(&self) -> usize {
        self.percolation_dims.iter().copied().max().unwrap_or(0)
    }
}

/// A lattice basis kept in Hermite normal form: at most one row per pivot
/// column, every combination exact.
#[derive(Default)]
struct Lattice {
    rows: Vec<[i128; 3]>,
}

impl Lattice {
    fn insert(&mut self, mut v: [i128; 3]) {
        for col in 0..3 {
            if v[col] == 0 {
                continue;
            }
            match self.rows.iter().position(|row| pivot(row) == Some(col)) {
                None => {
                    if v[col] < 0 {
                        v = v.map(|x| -x);
                    }
                    self.rows.push(v);
                    self.rows.sort_by_key(|row| pivot(row));
                    return;
                }
                Some(k) => {
                    let row = self.rows[k];
                    let (g, a, b) = extended_gcd(row[col], v[col]);
                    let (p, q) = (row[col] / g, v[col] / g);
                    let mut combined = [0_i128; 3];
                    let mut rest = [0_i128; 3];
                    for c in 0..3 {
                        combined[c] = a * row[c] + b * v[c];
                        rest[c] = p * v[c] - q * row[c];
                    }
                    if combined[col] < 0 {
                        combined = combined.map(|x| -x);
                    }
                    self.rows[k] = combined;
                    v = rest;
                }
            }
        }
    }
}

fn pivot(row: &[i128; 3]) -> Option<usize> {
    row.iter().position(|&x| x != 0)
}

/// `(g, a, b)` with `a·x + b·y = g = gcd(x, y) > 0`.
fn extended_gcd(x: i128, y: i128) -> (i128, i128, i128) {
    let (mut old_r, mut r) = (x, y);
    let (mut old_s, mut s) = (1_i128, 0_i128);
    let (mut old_t, mut t) = (0_i128, 1_i128);
    while r != 0 {
        let quotient = old_r / r;
        (old_r, r) = (r, old_r - quotient * r);
        (old_s, s) = (s, old_s - quotient * s);
        (old_t, t) = (t, old_t - quotient * t);
    }
    if old_r < 0 {
        (-old_r, -old_s, -old_t)
    } else {
        (old_r, old_s, old_t)
    }
}

fn mat_vec(m: ArrayView2<'_, F>, v: [F; 3]) -> [F; 3] {
    std::array::from_fn(|r| (0..3).map(|c| m[[r, c]] * v[c]).sum())
}

impl Compute for Percolation {
    type Input<'a> = PercolationInput<'a>;
    type Output = PercolationResult;

    /// # Errors
    /// [`ComputeError::BadShape`] for positions that are not `N × 3`;
    /// [`ComputeError::OutOfRange`] for a bond index past `N` or a bond too long
    /// for its image to be read by minimum image; [`ComputeError::NonFinite`]
    /// for a non-finite position.
    fn compute(&self, input: PercolationInput<'_>) -> Result<PercolationResult, ComputeError> {
        let positions = input.positions;
        let n = positions.nrows();
        if positions.ncols() != 3 {
            return Err(ComputeError::BadShape {
                expected: "positions N × 3".into(),
                got: format!("{:?}", positions.shape()),
            });
        }
        for (i, row) in positions.rows().into_iter().enumerate() {
            if row.iter().any(|x| !x.is_finite()) {
                return Err(ComputeError::NonFinite {
                    where_: "Percolation positions",
                    index: i,
                });
            }
        }
        let simbox = input.simbox;
        let inv = simbox.inv_view();
        let pbc = simbox.pbc();
        let half_plane = {
            let planes = simbox.nearest_plane_distance();
            (0..3)
                .filter(|&a| pbc[a])
                .map(|a| 0.5 * planes[a])
                .fold(F::INFINITY, F::min)
        };
        let point =
            |i: usize| -> [F; 3] { [positions[[i, 0]], positions[[i, 1]], positions[[i, 2]]] };

        // Bond translations, as an adjacency list.
        let mut adjacency: Vec<Vec<(usize, [i64; 3])>> = vec![Vec::new(); n];
        for &[i, j] in input.bonds {
            if i >= n || j >= n {
                return Err(ComputeError::OutOfRange {
                    field: "Percolation bond index",
                    value: format!("({i}, {j}) with {n} atoms"),
                });
            }
            let (ri, rj) = (point(i), point(j));
            let d = simbox
                .mic()
                .apply([rj[0] - ri[0], rj[1] - ri[1], rj[2] - ri[2]]);
            let length = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if length >= half_plane {
                return Err(ComputeError::OutOfRange {
                    field: "Percolation bond length (minimum image needs < half the cell)",
                    value: format!(
                        "bond ({i}, {j}) is {length:.3} Å, half plane {half_plane:.3} Å"
                    ),
                });
            }
            // The image of j that i is bonded to.
            let gap = [
                ri[0] + d[0] - rj[0],
                ri[1] + d[1] - rj[1],
                ri[2] + d[2] - rj[2],
            ];
            let frac = mat_vec(inv, gap);
            let mut t = [0_i64; 3];
            for a in 0..3 {
                let shift = frac[a].round();
                t[a] = shift as i64;
            }
            adjacency[i].push((j, t));
            adjacency[j].push((i, t.map(|x| -x)));
        }

        // Breadth-first walk per component.
        let mut component_idx = Array1::from_elem(n, -1_i64);
        let mut unwrap_images = Array2::<i64>::zeros((n, 3));
        let mut component_sizes = Vec::new();
        let mut percolation_dims = Vec::new();
        let mut translations = Vec::new();
        for start in 0..n {
            if component_idx[start] >= 0 {
                continue;
            }
            let label = component_sizes.len() as i64;
            component_idx[start] = label;
            let mut size = 1;
            let mut lattice = Lattice::default();
            let mut queue = VecDeque::from([start]);
            while let Some(i) = queue.pop_front() {
                let oi = [
                    unwrap_images[[i, 0]],
                    unwrap_images[[i, 1]],
                    unwrap_images[[i, 2]],
                ];
                for &(j, t) in &adjacency[i] {
                    let reached = [oi[0] + t[0], oi[1] + t[1], oi[2] + t[2]];
                    if component_idx[j] < 0 {
                        component_idx[j] = label;
                        for a in 0..3 {
                            unwrap_images[[j, a]] = reached[a];
                        }
                        size += 1;
                        queue.push_back(j);
                    } else {
                        let loop_vector: [i128; 3] =
                            std::array::from_fn(|a| (reached[a] - unwrap_images[[j, a]]) as i128);
                        if loop_vector != [0; 3] {
                            lattice.insert(loop_vector);
                        }
                    }
                }
            }
            component_sizes.push(size);
            percolation_dims.push(lattice.rows.len());
            translations.push(
                lattice
                    .rows
                    .iter()
                    .map(|row| row.map(|x| x as i64))
                    .collect(),
            );
        }

        Ok(PercolationResult {
            component_idx,
            n_components: component_sizes.len(),
            component_sizes,
            percolation_dims,
            translations,
            unwrap_images,
        })
    }
}

crate::compute::spec::compute_spec! {
    Percolation as "Percolation";
    params |_s| {}
    inputs {
        positions: Positions;
        simbox: SimBox;
        bonds: IndexPairs;
    }
}

#[cfg(test)]
mod tests;
