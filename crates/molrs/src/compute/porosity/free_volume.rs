//! Probe-accessible volume and the Bondi fractional free volume.

use molrs::op::F;
use ndarray::{ArrayView1, ArrayView2};
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

use super::geometry::{self, SurfaceIndex};
use crate::compute::{Compute, ComputeError};
use crate::core::SimBox;

/// Bondi's occupied-volume factor: `V_0 = 1.3 V_w`.
///
/// Bondi, *J. Phys. Chem.* **1954**, 68, 441. `V_w` here is the volume of the
/// union of the van der Waals spheres, which is the complement of the
/// probe-free accessible volume.
pub const BONDI_OCCUPIED_FACTOR: F = 1.3;

const DEFAULT_N_SAMPLES: usize = 20_000;
const DEFAULT_SEED: u64 = 0;

/// Monte Carlo accessible volume of a hard probe in a periodic cell.
///
/// Configuration: the probe radius and the sample count. Data: one radius
/// per atom, the positions and the box.
///
/// A sample is accessible when its clearance to every van der Waals sphere
/// is strictly larger than `probe_radius` (a point on a sphere surface is
/// occupied). The accessible fraction is the fraction of such samples. With
/// `probe_radius = 0` the complement is the occupied fraction of the union
/// of the van der Waals spheres, and [`FreeVolumeResult::bondi_fraction`] is
/// `(V − 1.3 V_w) / V`. A positive probe asks the Zeo++ / Poreblazer
/// question — where the centre of a sphere of that radius can sit — and
/// Bondi's factor does not apply, so `bondi_fraction` is absent.
///
/// The uncertainty is the binomial standard error
/// `√(p (1 − p) / n_samples)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FreeVolume {
    probe_radius: F,
    n_samples: usize,
    seed: u64,
}

/// The data a [`FreeVolume`] analyses. Lengths in Å.
#[derive(Clone, Copy)]
pub struct FreeVolumeInput<'a> {
    /// Atom positions, `N × 3`.
    pub positions: ArrayView2<'a, F>,
    /// Van der Waals radius of each atom, length `N`.
    pub radii: ArrayView1<'a, F>,
    /// The periodic cell.
    pub simbox: &'a SimBox,
}

/// Accessible volume of one snapshot.
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
pub struct FreeVolumeResult {
    /// Fraction of the cell where the probe centre fits.
    pub accessible_fraction: F,
    /// `accessible_fraction ×` the cell volume, Å³.
    pub accessible_volume: F,
    /// Binomial standard error of [`accessible_fraction`](Self::accessible_fraction).
    pub uncertainty: F,
    /// Samples drawn.
    pub n_samples: usize,
    /// Samples the probe centre fitted into.
    pub n_accessible: usize,
    /// Cell volume, Å³.
    pub box_volume: F,
    /// Bondi fractional free volume. Present only for a point probe
    /// (`probe_radius = 0`).
    pub bondi_fraction: Option<F>,
}

impl FreeVolume {
    /// A probe of radius `probe_radius` (Å), with 20,000 samples and seed 0.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] for a negative or non-finite radius.
    pub fn new(probe_radius: F) -> Result<Self, ComputeError> {
        if !probe_radius.is_finite() || probe_radius < 0.0 {
            return Err(ComputeError::OutOfRange {
                field: "FreeVolume::probe_radius",
                value: probe_radius.to_string(),
            });
        }
        Ok(Self {
            probe_radius,
            n_samples: DEFAULT_N_SAMPLES,
            seed: DEFAULT_SEED,
        })
    }

    /// Draw `n_samples` points.
    ///
    /// # Errors
    /// [`ComputeError::OutOfRange`] when `n_samples` is 0.
    pub fn with_n_samples(mut self, n_samples: usize) -> Result<Self, ComputeError> {
        if n_samples == 0 {
            return Err(ComputeError::OutOfRange {
                field: "FreeVolume::n_samples",
                value: n_samples.to_string(),
            });
        }
        self.n_samples = n_samples;
        Ok(self)
    }

    /// Seed of the sample generator.
    pub fn with_seed(mut self, seed: u64) -> Result<Self, ComputeError> {
        self.seed = seed;
        Ok(self)
    }
}

impl Compute for FreeVolume {
    type Input<'a> = FreeVolumeInput<'a>;
    type Output = FreeVolumeResult;

    /// # Errors
    /// [`ComputeError::EmptyInput`] with no atoms;
    /// [`ComputeError::DimensionMismatch`] when `radii` is not one per atom;
    /// [`ComputeError::BadShape`] when `positions` is not `N × 3`;
    /// [`ComputeError::NonFinite`] for a non-finite coordinate;
    /// [`ComputeError::OutOfRange`] for a negative radius or a cell with no
    /// positive volume.
    fn compute(&self, input: FreeVolumeInput<'_>) -> Result<FreeVolumeResult, ComputeError> {
        let spheres = geometry::read_spheres(input.positions, input.radii)?;
        let cell = geometry::read_cell(input.simbox)?;
        let index = SurfaceIndex::build(spheres, cell);
        let probe = self.probe_radius;
        let mut rng = StdRng::seed_from_u64(self.seed);
        let mut n_accessible = 0;
        for _ in 0..self.n_samples {
            let frac = [rng.random(), rng.random(), rng.random()];
            let point = geometry::cartesian(&index.cell, frac);
            if index.clearance(point) > probe {
                n_accessible += 1;
            }
        }
        Ok(finish(self, n_accessible, index.cell.volume))
    }
}

fn finish(calc: &FreeVolume, n_accessible: usize, box_volume: F) -> FreeVolumeResult {
    let n_samples = calc.n_samples;
    let accessible_fraction = n_accessible as F / n_samples as F;
    let occupied = 1.0 - accessible_fraction;
    let bondi_fraction =
        (calc.probe_radius == 0.0).then_some(1.0 - BONDI_OCCUPIED_FACTOR * occupied);
    let uncertainty = (accessible_fraction * (1.0 - accessible_fraction) / n_samples as F).sqrt();
    FreeVolumeResult {
        accessible_fraction,
        accessible_volume: accessible_fraction * box_volume,
        uncertainty,
        n_samples,
        n_accessible,
        box_volume,
        bondi_fraction,
    }
}

crate::compute::spec::compute_spec! {
    FreeVolume as "FreeVolume";
    params |s| {
        probe_radius: Float;
        n_samples: Int = DEFAULT_N_SAMPLES => s.n_samples;
        seed: Int = DEFAULT_SEED => s.seed;
    }
    inputs {
        positions: Positions;
        radii: Scalars;
        simbox: SimBox;
    }
}
