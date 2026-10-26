//! Stable handle for a molrs [`ForceField`] — the force-field analogue of
//! [`crate::FrameRef`].
//!
//! A force field is standalone: unlike a [`molrs::core::Frame`] it does
//! not live in the slot-mapped [`crate::FrameArena`], so this handle is a thin `Rc`
//! share rather than a `(handle, arena)` pair. It exists so a force field can
//! cross a language / extension boundary the same way a frame does — the
//! producing binding (molpy) hands out a `PyCapsule` wrapping a clone of
//! this handle, and a consuming Rust binding (e.g. molpack) resolves the capsule
//! back to a `ForceFieldRef` and borrows the real core [`ForceField`] through
//! [`ForceFieldRef::with_forcefield`]. No marshalling, no parallel data type.
//!
//! Gated behind the `ff` feature because [`molrs::ff`] itself is.

use std::rc::Rc;

use molrs::ff::forcefield::ForceField;

/// Shared-ownership handle to a [`ForceField`]. Cheap to clone (one `Rc` bump).
///
/// Mirrors [`crate::FrameRef`]; bindings wrap this and add language-specific
/// attributes (e.g. `#[pyclass]`). The wrapped force field is immutable through
/// the handle — a relaxer / MD consumer only reads parameters to compile
/// potentials — so take an owned copy via [`ForceFieldRef::clone_forcefield`]
/// if mutation is ever needed.
///
/// ## Threading
///
/// Holds an `Rc`, so it is `!Send` like [`crate::FrameRef`]. The capsule that
/// carries it across a boundary is only ever created, read, and destroyed under
/// the Python GIL, which upholds the single-threaded discipline.
#[derive(Clone)]
pub struct ForceFieldRef {
    ff: Rc<ForceField>,
}

impl ForceFieldRef {
    /// Wrap an owned force field in a fresh shared handle.
    pub fn new(ff: ForceField) -> Self {
        Self { ff: Rc::new(ff) }
    }

    /// Run a closure with immutable access to the underlying [`ForceField`].
    ///
    /// This is the consumer entry point that mirrors
    /// [`FrameRef::with`](crate::FrameRef::with): a downstream binding resolves a
    /// capsule to a `ForceFieldRef` and calls, e.g.,
    /// `ffref.with_forcefield(|ff| PotentialCompiler::new(ff).compile(frame))`.
    pub fn with_forcefield<R>(&self, f: impl FnOnce(&ForceField) -> R) -> R {
        f(&self.ff)
    }

    /// The shared force field, for handles that want to keep sharing it.
    pub fn rc(&self) -> Rc<ForceField> {
        Rc::clone(&self.ff)
    }

    /// Deep-clone the force field out of the handle.
    pub fn clone_forcefield(&self) -> ForceField {
        (*self.ff).clone()
    }
}

impl From<Rc<ForceField>> for ForceFieldRef {
    /// Wrap an already-shared force field, sharing its backing storage.
    fn from(ff: Rc<ForceField>) -> Self {
        Self { ff }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn handle_shares_and_clones() {
        // An empty force field is enough to exercise share + borrow semantics.
        let ff = ForceField::new("test");
        let a = ForceFieldRef::new(ff);
        let b = a.clone();
        // Both handles point at the same backing storage (one extra Rc).
        assert!(Rc::ptr_eq(&a.rc(), &b.rc()));
        // Borrow through the closure entry point.
        let n = a.with_forcefield(|ff| ff.styles().len());
        let m = b.with_forcefield(|ff| ff.styles().len());
        assert_eq!(n, m);
        // Deep copy is independent.
        let _owned: ForceField = a.clone_forcefield();
    }
}
