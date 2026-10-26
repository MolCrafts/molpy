//! [`SpecialClass`]: which of a force field's `special_bonds` weight sets
//! scales a pair style, a declaration of its
//! [`StyleSpec`](crate::ff::ir::StyleSpec).

/// Which of a force field's special-bonds weight sets scales a pair style.
///
/// A force field may scale close van-der-Waals and electrostatic neighbours
/// differently — Amber uses `1/2` and `1/1.2` — and in molrs those are
/// separate kernels, so each has to say which set is its own. Declared at
/// registration rather than guessed from the style's name: a name is a label,
/// and this is a fact about the physics.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpecialClass {
    /// Scaled by the force field's van-der-Waals weights.
    Vdw,
    /// Scaled by its electrostatic weights.
    Coulomb,
}
