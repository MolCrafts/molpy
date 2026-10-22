//! Stable handle for a molrs [`Region`] — the region analogue of
//! [`crate::FrameRef`].
//!
//! A region is geometry that a consumer *evaluates*, not data it reads: the
//! producing binding (molrs-python) hands out a `PyCapsule` wrapping a clone
//! of this handle, and a consuming Rust binding (e.g. molpack) resolves the
//! capsule back to a `RegionRef`, takes the shared [`Region`] through
//! [`RegionRef::region`], and calls its methods. No marshalling, no parallel
//! data type — and, unlike a frame, no arena: a region is a self-contained
//! `Arc`.
//!
//! Always compiled: `Region` lives in molrs `core`, which every consumer
//! configuration has.
//!
//! ## Threading
//!
//! Holds an `Arc<dyn Region + Send + Sync>`, so the handle is `Send + Sync`
//! itself — a consumer may share the region into a rayon loop. The capsule
//! that carries it across a boundary is still only created, read, and
//! destroyed under the Python GIL.
//!
//! ## Cross-image calls
//!
//! This is the first handle whose *code* runs in the producer's image: the
//! `Region` methods dispatch through a vtable compiled into molrs-python. The
//! contract a consumer may rely on is the `[F; 3]` surface —
//! [`Region::distance`], [`Region::distance_grad`], [`Region::contains_point`]
//! and [`Region::bounds`] — none of which panics on finite input. The batched
//! [`Region::contains`] can panic on a malformed array and is not part of the
//! cross-image contract.

use std::sync::Arc;

use molrs::core::Region;

/// Shared-ownership handle to a [`Region`]. Cheap to clone (one `Arc` bump).
#[derive(Clone)]
pub struct RegionRef {
    region: Arc<dyn Region + Send + Sync>,
}

impl std::fmt::Debug for RegionRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "RegionRef({:?})", self.region)
    }
}

impl RegionRef {
    /// Wrap a shared region in a handle.
    pub fn new(region: Arc<dyn Region + Send + Sync>) -> Self {
        Self { region }
    }

    /// Run a closure with access to the underlying [`Region`].
    pub fn with_region<R>(&self, f: impl FnOnce(&dyn Region) -> R) -> R {
        f(self.region.as_ref())
    }

    /// The shared region, for consumers that keep it (one `Arc` bump).
    pub fn region(&self) -> Arc<dyn Region + Send + Sync> {
        Arc::clone(&self.region)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::Sphere;
    use ndarray::Array1;

    #[test]
    fn handle_shares_and_evaluates() {
        let sphere: Arc<dyn Region + Send + Sync> = Arc::new(Sphere::new(Array1::zeros(3), 2.0));
        let a = RegionRef::new(sphere);
        let b = a.clone();
        assert!(Arc::ptr_eq(&a.region(), &b.region()));
        assert_eq!(a.with_region(|r| r.distance(&[0.0; 3])), -2.0);
        assert!(b.region().contains_point(&[1.0, 0.0, 0.0]));
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RegionRef>();
    }
}
