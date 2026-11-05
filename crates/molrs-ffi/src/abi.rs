//! The capsule contract of the molrs FFI layer.
//!
//! All cross-extension interop (molpy ↔ molpack, cxxapi consumers)
//! exchanges raw `molrs_ffi` handles, so both sides must embed a
//! layout-identical molrs core. A capsule name is a fixed per-handle prefix
//! plus this build's [`build_fingerprint`], so two builds that may lay a handle
//! out differently refuse each other's capsules instead of dereferencing a
//! drifted layout. This module is the single source of the capsule names —
//! binders and consumers must take them from here, never hard-code them.

use std::ffi::{CStr, CString};
use std::sync::OnceLock;

/// Fixed name prefix of [`crate::FrameRef`] capsules.
const FRAMEREF_CAPSULE_PREFIX: &str = "molrs.FrameRef";
/// Fixed name prefix of `ForceFieldRef` capsules.
const FORCEFIELDREF_CAPSULE_PREFIX: &str = "molrs.ForceFieldRef";
/// Fixed name prefix of [`crate::RegionRef`] capsules.
const REGIONREF_CAPSULE_PREFIX: &str = "molrs.RegionRef";

/// Size and alignment of every type a handle exchange dereferences: the
/// handles themselves and what they point at, down to the arena lock. A change
/// to any of them fails `tests::layout_matches_the_snapshot` (the `ffi` gate)
/// and changes [`build_fingerprint`].
pub fn layout() -> Vec<(&'static str, usize, usize)> {
    use std::mem::{align_of, size_of};
    let out = vec![
        (
            "FrameRef",
            size_of::<crate::FrameRef>(),
            align_of::<crate::FrameRef>(),
        ),
        (
            "BlockRef",
            size_of::<crate::BlockRef>(),
            align_of::<crate::BlockRef>(),
        ),
        (
            "FrameArena",
            size_of::<crate::FrameArena>(),
            align_of::<crate::FrameArena>(),
        ),
        // What a `FrameArenaCell` points at: the lock a consumer takes on
        // every access. Listed so a build whose arena lock differs (the
        // former `RefCell`, another platform's `RwLock`) refuses the capsule.
        (
            "FrameArenaLock",
            size_of::<std::sync::RwLock<crate::FrameArena>>(),
            align_of::<std::sync::RwLock<crate::FrameArena>>(),
        ),
        (
            "RegionRef",
            size_of::<crate::RegionRef>(),
            align_of::<crate::RegionRef>(),
        ),
        (
            "Frame",
            size_of::<molrs::core::Frame>(),
            align_of::<molrs::core::Frame>(),
        ),
        (
            "Block",
            size_of::<molrs::core::Block>(),
            align_of::<molrs::core::Block>(),
        ),
        (
            "SimBox",
            size_of::<molrs::core::SimBox>(),
            align_of::<molrs::core::SimBox>(),
        ),
    ];
    #[cfg(feature = "ff")]
    let out = {
        let mut out = out;
        out.extend([
            (
                "ForceFieldRef",
                size_of::<crate::ForceFieldRef>(),
                align_of::<crate::ForceFieldRef>(),
            ),
            (
                "ForceField",
                size_of::<molrs::ff::forcefield::ForceField>(),
                align_of::<molrs::ff::forcefield::ForceField>(),
            ),
        ]);
        out
    };
    out
}

/// This build's handle-layout fingerprint, 16 hex digits: FNV-1a over the
/// `rustc` that compiled it and [`layout`]. Two builds exchange handles only
/// when their fingerprints agree — same compiler, same layout — because the
/// layout is the compiler's choice as much as the source's.
pub fn build_fingerprint() -> &'static str {
    static FINGERPRINT: OnceLock<String> = OnceLock::new();
    FINGERPRINT.get_or_init(|| {
        let mut text = String::from(env!("MOLRS_FFI_RUSTC"));
        for (name, size, align) in layout() {
            text.push_str(&format!(";{name}:{size}:{align}"));
        }
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for byte in text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        format!("{hash:016x}")
    })
}

fn capsule_name(prefix: &str) -> CString {
    CString::new(format!("{prefix}/{}", build_fingerprint())).expect("capsule name has no NUL")
}

/// PyCapsule name for [`crate::FrameRef`] handles:
/// `molrs.FrameRef/<build_fingerprint>`.
///
/// An exchange between builds that may lay a handle out differently fails the
/// capsule name check (a clean `ValueError`) instead of dereferencing a
/// drifted layout.
pub fn frameref_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| capsule_name(FRAMEREF_CAPSULE_PREFIX))
}

/// PyCapsule name for `ForceFieldRef` handles:
/// `molrs.ForceFieldRef/<build_fingerprint>`.
///
/// Not gated on the `ff` feature — it is only a string, and a core-only build
/// may still need to *recognize* (and reject) a force-field capsule.
pub fn forcefield_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| capsule_name(FORCEFIELDREF_CAPSULE_PREFIX))
}

/// PyCapsule name for [`crate::RegionRef`] handles:
/// `molrs.RegionRef/<build_fingerprint>`.
///
/// A region crosses the boundary as shared geometry the consumer evaluates
/// in place (see [`crate::RegionRef`]); the fingerprinted name keeps an
/// exchange between differing layouts failing at the name check like the
/// other handles.
pub fn regionref_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| capsule_name(REGIONREF_CAPSULE_PREFIX))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout of the handle-crossing types on a 64-bit target. A change
    /// here is a deliberate layout change: update this snapshot with it.
    #[cfg(all(target_pointer_width = "64", feature = "ff"))]
    const SNAPSHOT: &[(&str, usize, usize)] = &[
        ("FrameRef", 16, 8),
        ("BlockRef", 48, 8),
        ("FrameArena", 32, 8),
        ("FrameArenaLock", 48, 8),
        ("RegionRef", 16, 8),
        ("Frame", 1224, 8),
        ("Block", 328, 8),
        ("SimBox", 1080, 8),
        ("ForceFieldRef", 8, 8),
        ("ForceField", 136, 8),
    ];

    #[test]
    #[cfg(all(target_pointer_width = "64", feature = "ff"))]
    fn layout_matches_the_snapshot() {
        assert_eq!(
            layout(),
            SNAPSHOT,
            "an FFI-crossing layout changed: {:?}",
            layout()
        );
    }

    #[test]
    fn capsule_names_carry_the_build_fingerprint_and_differ() {
        let frame = frameref_capsule_name().to_str().expect("utf8");
        let ff = forcefield_capsule_name().to_str().expect("utf8");
        let region = regionref_capsule_name().to_str().expect("utf8");
        let fingerprint = build_fingerprint();
        assert_eq!(frame, format!("molrs.FrameRef/{fingerprint}"));
        assert_eq!(ff, format!("molrs.ForceFieldRef/{fingerprint}"));
        assert_eq!(region, format!("molrs.RegionRef/{fingerprint}"));
        assert_ne!(frame, ff);
        assert_ne!(frame, region);
        assert_ne!(ff, region);
    }
}
