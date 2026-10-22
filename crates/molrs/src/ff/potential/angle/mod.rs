//! Angle potential kernels.

pub(crate) mod charmm;
pub(crate) mod class2;
pub(crate) mod harmonic;
pub(crate) mod mmff;
pub(crate) mod uff;

pub use charmm::{AngleCharmm, AngleCharmmParams, angle_charmm_constructor};
pub use class2::{AngleClass2, angle_class2_constructor};
pub use harmonic::{AngleHarmonic, angle_harmonic_constructor};
pub use mmff::{
    AngleMmff, AngleMmffStretchBend, angle_mmff_constructor, angle_mmff_stretch_bend_constructor,
};
pub use uff::{AngleUff, angle_uff_constructor};

// The angle-geometry chain rule (force = `dE/dθ / sin θ` times the gradient
// of `cos θ`) is independent of the bending potential, so every angle kernel
// routes its `dE/dθ` through this one helper.
pub(crate) use crate::ff::potential::flat_coords::accumulate_angle_forces;
