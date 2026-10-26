//! [`SpecialBonds`]: LAMMPS `special_bonds`, the 1-2 / 1-3 / 1-4 weights of
//! the non-bonded kernels.

/// The style param under which a compile hands a pair kernel the field's
/// van-der-Waals 1-4 weight ([`SpecialBonds::lj_14`]); a projection, never a
/// style's own parameter.
pub const LJ_14_SCALE: &str = "lj14scale";
/// The style param carrying the Coulomb 1-4 weight ([`SpecialBonds::coul_14`]).
pub const COULOMB_14_SCALE: &str = "coulomb14scale";

/// Per-nonbonded-kind 1-2 / 1-3 / 1-4 interaction scale weights — LAMMPS
/// `special_bonds` semantics, owned by the [`ForceField`](crate::ff::forcefield::ForceField).
///
/// A weight of `0.0` fully excludes that neighbour class; `1.0` leaves it at
/// full strength.
///
/// A frame's special pairs (its `pairs` rows) name each pair's class by its
/// `n_bonds`; these weights price the class, the van-der-Waals and Coulomb
/// kernels independently, at every compile door. A row's own `lj_scale` /
/// `coul_scale` cell overrides them for that pair.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpecialBonds {
    /// LJ / van-der-Waals `[1-2, 1-3, 1-4]` scale weights.
    pub lj: [f64; 3],
    /// Coulomb `[1-2, 1-3, 1-4]` scale weights.
    pub coul: [f64; 3],
}

impl SpecialBonds {
    /// Exclude 1-2 and 1-3 neighbours and leave 1-4 unscaled — named, not a
    /// default: a force field states its own weights, and one that states
    /// none has none ([`ForceField::special_bonds`](crate::ff::forcefield::ForceField::special_bonds)).
    pub const EXCLUDE_12_13: SpecialBonds = SpecialBonds {
        lj: [0.0, 0.0, 1.0],
        coul: [0.0, 0.0, 1.0],
    };

    /// The LJ 1-4 scale weight (the `[2]` entry of [`Self::lj`]).
    pub fn lj_14(&self) -> f64 {
        self.lj[2]
    }

    /// The Coulomb 1-4 scale weight (the `[2]` entry of [`Self::coul`]).
    pub fn coul_14(&self) -> f64 {
        self.coul[2]
    }
}
