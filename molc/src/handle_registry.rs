//! The handle registry: the mutex-protected singleton that owns every object a
//! C handle names.
//!
//! All FFI objects (frames, blocks, boxes, force fields, regions) and interned
//! key strings are owned by the one [`HandleRegistry`], [`REGISTRY`], reached
//! through [`lock_registry`].
//!
//! The registry is lazily initialised on first access and lives for the
//! entire process lifetime.  [`molc_shutdown`](crate::molc_shutdown)
//! clears all contents but does not destroy the mutex.

use std::collections::HashMap;
use std::ffi::CString;
use std::sync::{LazyLock, Mutex, MutexGuard};

use molrs::core::SimBox;
use molrs::ff::forcefield::ForceField;
use molrs_ffi::RegionRef;
use slotmap::SlotMap;

use crate::handle::{BoxKey, ForceFieldKey, RegionKey};

/// Owner of every object reachable through a C-API handle.
///
/// The single source of truth for frames, boxes, force fields, regions and
/// interned keys. It is wrapped in a `Mutex` ([`REGISTRY`]) and accessed
/// through [`lock_registry`].
pub(crate) struct HandleRegistry {
    /// Frames and their blocks, owned by `molrs-ffi`'s arena.
    pub frames: molrs_ffi::FrameArena,

    /// Interned key strings, stored null-terminated for direct return
    /// to C callers via [`molc_key_name`](crate::molc_key_name).
    /// The vector index is the `key_id`.
    pub interned_keys: Vec<CString>,

    /// Reverse lookup: Rust `String` to interned `key_id` (`u32`).
    pub key_to_id: HashMap<String, u32>,

    /// Standalone Box instances, keyed by [`BoxKey`].
    pub simboxes: SlotMap<BoxKey, SimBox>,

    /// Standalone ForceField instances, keyed by [`ForceFieldKey`].
    pub forcefields: SlotMap<ForceFieldKey, ForceField>,

    /// Shared regions, keyed by [`RegionKey`]. The value is the same
    /// `molrs_ffi::RegionRef` the Python and WASM binders hold, so the trait
    /// object never crosses the C boundary — only the two-word handle does.
    pub regions: SlotMap<RegionKey, RegionRef>,
}

impl HandleRegistry {
    fn new() -> Self {
        Self {
            frames: molrs_ffi::FrameArena::new(),
            interned_keys: Vec::new(),
            key_to_id: HashMap::new(),
            simboxes: SlotMap::with_key(),
            forcefields: SlotMap::with_key(),
            regions: SlotMap::with_key(),
        }
    }

    /// Intern a key string, returning its id. Idempotent.
    pub fn intern(&mut self, key: &str) -> u32 {
        if let Some(&id) = self.key_to_id.get(key) {
            return id;
        }
        let id = self.interned_keys.len() as u32;
        let cstr = CString::new(key).unwrap_or_default();
        self.interned_keys.push(cstr);
        self.key_to_id.insert(key.to_owned(), id);
        id
    }

    /// Look up the Rust string for an interned key_id.
    pub fn key_str(&self, id: u32) -> Option<&str> {
        self.interned_keys
            .get(id as usize)
            .and_then(|cs| cs.to_str().ok())
    }

    /// Reset all state: every frame, box, force field and region is dropped
    /// and every interned key forgotten.
    pub fn clear(&mut self) {
        // Destructured so a field added later cannot be forgotten here: a
        // field missed by `clear` would survive `molc_shutdown`.
        let Self {
            frames,
            interned_keys,
            key_to_id,
            simboxes,
            forcefields,
            regions,
        } = self;
        *frames = molrs_ffi::FrameArena::new();
        interned_keys.clear();
        key_to_id.clear();
        simboxes.clear();
        forcefields.clear();
        regions.clear();
    }
}

/// The process-wide handle registry.
pub(crate) static REGISTRY: LazyLock<Mutex<HandleRegistry>> =
    LazyLock::new(|| Mutex::new(HandleRegistry::new()));

/// Lock the handle registry, returning a guard.
pub(crate) fn lock_registry() -> MutexGuard<'static, HandleRegistry> {
    REGISTRY.lock().expect("REGISTRY poisoned")
}

#[cfg(test)]
mod tests {
    use super::*;
    use molrs::core::{Cell, Sphere};
    use ndarray::Array1;
    use std::sync::Arc;

    #[test]
    fn clear_drops_every_kind_of_object() {
        // A local registry: clearing the global one would pull handles out
        // from under the tests running beside this one.
        let mut registry = HandleRegistry::new();
        registry.intern("atoms");
        registry.frames.frame_new();
        registry
            .simboxes
            .insert(SimBox::new(Cell::Lengths([1.0; 3]), [0.0; 3], [true; 3]).unwrap());
        registry.forcefields.insert(ForceField::new("ff"));
        registry
            .regions
            .insert(RegionRef::new(Arc::new(Sphere::new(Array1::zeros(3), 1.0))));
        registry.clear();
        assert!(registry.interned_keys.is_empty() && registry.key_to_id.is_empty());
        assert!(registry.simboxes.is_empty());
        assert!(registry.forcefields.is_empty());
        assert!(registry.regions.is_empty(), "regions survived clear()");
    }
}
