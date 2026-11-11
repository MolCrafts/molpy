//! Free volume and the pore-size distribution of a hard-sphere solid.
//!
//! [`FreeVolume`] is the probe-accessible fraction of a cell, and the Bondi
//! fractional free volume when the probe is a point. [`PoreSizeDistribution`]
//! is the volume-weighted distribution of the largest sphere that covers each
//! void point (Gelb and Gubbins). Both take van der Waals radii from the
//! caller; neither reads a frame.

mod free_volume;
mod geometry;
mod pore_size;

pub use free_volume::{FreeVolume, FreeVolumeInput, FreeVolumeResult};
pub use pore_size::{PoreSizeDistribution, PoreSizeDistributionInput, PoreSizeDistributionResult};

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;
    use crate::compute::{Compute, ComputeError};
    use crate::core::{Cell, SimBox};

    fn cube(length: f64) -> SimBox {
        SimBox::new(Cell::Lengths([length; 3]), [0.0; 3], [true; 3]).unwrap()
    }

    #[test]
    fn a_sphere_that_fills_the_cell_has_no_free_volume() {
        let simbox = cube(10.0);
        let result = FreeVolume::new(0.0)
            .unwrap()
            .with_n_samples(1000)
            .unwrap()
            .compute(FreeVolumeInput {
                positions: array![[5.0, 5.0, 5.0]].view(),
                radii: array![20.0].view(),
                simbox: &simbox,
            })
            .unwrap();
        assert_eq!(result.accessible_fraction, 0.0);
        assert_eq!(result.n_accessible, 0);
        assert_eq!(result.bondi_fraction, Some(1.0 - 1.3));
        assert_eq!(result.box_volume, 1000.0);
    }

    #[test]
    fn bondi_uses_the_same_samples_as_the_accessible_fraction() {
        let simbox = cube(10.0);
        let result = FreeVolume::new(0.0)
            .unwrap()
            .with_n_samples(4000)
            .unwrap()
            .with_seed(7)
            .unwrap()
            .compute(FreeVolumeInput {
                positions: array![[5.0, 5.0, 5.0]].view(),
                radii: array![2.0].view(),
                simbox: &simbox,
            })
            .unwrap();
        let bondi = 1.0 - 1.3 * (1.0 - result.accessible_fraction);
        assert!((result.bondi_fraction.unwrap() - bondi).abs() < 1e-12);
        let sphere = 4.0 / 3.0 * std::f64::consts::PI * 8.0 / 1000.0;
        assert!((result.accessible_fraction - (1.0 - sphere)).abs() < 0.02);
    }

    #[test]
    fn a_probe_sphere_does_not_report_a_bondi_fraction() {
        let simbox = cube(10.0);
        let result = FreeVolume::new(1.0)
            .unwrap()
            .with_n_samples(200)
            .unwrap()
            .compute(FreeVolumeInput {
                positions: array![[5.0, 5.0, 5.0]].view(),
                radii: array![1.0].view(),
                simbox: &simbox,
            })
            .unwrap();
        assert!(result.bondi_fraction.is_none());
    }

    #[test]
    fn a_sphere_sitting_on_the_corner_is_counted_through_the_minimum_image() {
        let simbox = cube(10.0);
        let result = FreeVolume::new(0.0)
            .unwrap()
            .with_n_samples(8000)
            .unwrap()
            .with_seed(3)
            .unwrap()
            .compute(FreeVolumeInput {
                positions: array![[0.0, 0.0, 0.0]].view(),
                radii: array![3.0].view(),
                simbox: &simbox,
            })
            .unwrap();
        let sphere = 4.0 / 3.0 * std::f64::consts::PI * 27.0 / 1000.0;
        assert!((result.accessible_fraction - (1.0 - sphere)).abs() < 0.02);
    }

    #[test]
    fn free_volume_refuses_a_bad_configuration_and_bad_data() {
        assert!(matches!(
            FreeVolume::new(-1.0),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert!(matches!(
            FreeVolume::new(0.0).unwrap().with_n_samples(0),
            Err(ComputeError::OutOfRange { .. })
        ));
        let simbox = cube(10.0);
        let run = |radii: ndarray::Array1<f64>| {
            FreeVolume::new(0.0).unwrap().compute(FreeVolumeInput {
                positions: array![[0.0, 0.0, 0.0]].view(),
                radii: radii.view(),
                simbox: &simbox,
            })
        };
        assert!(matches!(
            run(array![-0.1]),
            Err(ComputeError::OutOfRange { .. })
        ));
        assert!(matches!(
            run(array![1.0, 1.0]),
            Err(ComputeError::DimensionMismatch { .. })
        ));
    }

    #[test]
    fn the_largest_cavity_is_the_minimum_image_distance_to_a_corner_atom() {
        let simbox = cube(10.0);
        let result = PoreSizeDistribution::new(1.0)
            .unwrap()
            .with_bin_width(0.5)
            .unwrap()
            .compute(PoreSizeDistributionInput {
                positions: array![[0.0, 0.0, 0.0]].view(),
                radii: array![0.0].view(),
                simbox: &simbox,
            })
            .unwrap();
        let expected = 4.5 * 3.0_f64.sqrt();
        assert!((result.largest_cavity_radius - expected).abs() < 1e-9);
        assert!((result.void_fraction - 1.0).abs() < 1e-12);
        let sum: f64 = result.fraction.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12);
        assert!((result.cumulative[0] - 1.0).abs() < 1e-12);
        assert!((result.voxel_volume - 1.0).abs() < 1e-12);
    }

    #[test]
    fn a_solid_cell_has_an_empty_pore_histogram() {
        let simbox = cube(4.0);
        let result = PoreSizeDistribution::new(1.0)
            .unwrap()
            .compute(PoreSizeDistributionInput {
                positions: array![[2.0, 2.0, 2.0]].view(),
                radii: array![5.0].view(),
                simbox: &simbox,
            })
            .unwrap();
        assert_eq!(result.largest_cavity_radius, 0.0);
        assert_eq!(result.void_fraction, 0.0);
        assert!(result.radius.is_empty());
    }

    #[test]
    fn pore_size_refuses_a_triclinic_cell_and_a_non_positive_spacing() {
        assert!(matches!(
            PoreSizeDistribution::new(0.0),
            Err(ComputeError::OutOfRange { .. })
        ));
        let tilted = SimBox::new(
            Cell::LengthsTilts {
                lengths: [10.0, 10.0, 10.0],
                tilts: [2.0, 0.0, 0.0],
            },
            [0.0; 3],
            [true; 3],
        )
        .unwrap();
        let err = PoreSizeDistribution::new(1.0)
            .unwrap()
            .compute(PoreSizeDistributionInput {
                positions: array![[0.0, 0.0, 0.0]].view(),
                radii: array![1.0].view(),
                simbox: &tilted,
            });
        assert!(matches!(err, Err(ComputeError::OutOfRange { .. })));
        let simbox = cube(10.0);
        for spacing in [0.001, f64::MIN_POSITIVE] {
            let err =
                PoreSizeDistribution::new(spacing)
                    .unwrap()
                    .compute(PoreSizeDistributionInput {
                        positions: array![[0.0, 0.0, 0.0]].view(),
                        radii: array![1.0].view(),
                        simbox: &simbox,
                    });
            assert!(matches!(err, Err(ComputeError::OutOfRange { .. })));
        }
    }
}
