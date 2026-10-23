//! The ABI contract of the molrs FFI layer.
//!
//! All cross-extension interop (molpy ↔ molpack, cxxapi consumers)
//! exchanges raw `molrs_ffi` handles, so both sides must embed a
//! layout-identical molrs core. The project rule is **minor-line = ABI
//! version**: every downstream shares one molrs minor line, and the layout of
//! every FFI-crossing type is frozen within a minor. This module is the
//! single source of the versioned capsule names and the handshake line —
//! binders and consumers must take them from here, never hard-code them.

use std::ffi::{CStr, CString};
use std::sync::OnceLock;

/// The ABI line of this build: `major.minor` of the statically linked
/// `molcrafts-molrs` core (e.g. `"0.17"`). This crate's version is the core's
/// by policy, and its dependency on the core is pinned to that minor line, so
/// its own `CARGO_PKG_VERSION` names the line.
///
/// Two extensions may exchange `molrs_ffi` handles if and only if their ABI
/// lines are equal. Patch versions may differ — layout is frozen within a
/// minor line.
pub fn abi_line() -> &'static str {
    static LINE: OnceLock<String> = OnceLock::new();
    LINE.get_or_init(|| {
        let mut parts = env!("CARGO_PKG_VERSION").split('.');
        let major = parts.next().expect("the version has a major component");
        let minor = parts.next().expect("the version has a minor component");
        format!("{major}.{minor}")
    })
}

/// Size and alignment of every type a handle exchange dereferences: the
/// handles themselves and what they point at. A change to any of them inside
/// a minor line fails `tests::layout_matches_the_snapshot` (the `ffi` gate)
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

/// This build's ABI fingerprint, 16 hex digits: FNV-1a over the `rustc`
/// version that compiled it and [`layout`]. Two builds of one minor line
/// exchange handles only when their fingerprints agree too — same line, same
/// compiler, same layout — because the line alone cannot see a layout the
/// compiler or a dependency's patch release chose differently.
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

fn versioned_name(prefix: &str) -> CString {
    CString::new(format!("{prefix}/{}+{}", abi_line(), build_fingerprint()))
        .expect("capsule name has no NUL")
}

/// PyCapsule name for [`crate::FrameRef`] handles:
/// `molrs.FrameRef/<abi_line>+<build_fingerprint>`.
///
/// The name carries the ABI line and the build fingerprint, so an exchange
/// between builds that may lay a handle out differently fails the capsule
/// name check (a clean `ValueError`) instead of dereferencing a drifted
/// layout.
pub fn frameref_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.FrameRef"))
}

/// PyCapsule name for `ForceFieldRef` handles:
/// `molrs.ForceFieldRef/<abi_line>+<build_fingerprint>`.
///
/// Not gated on the `ff` feature — it is only a string, and a core-only build
/// may still need to *recognize* (and reject) a force-field capsule.
pub fn forcefield_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.ForceFieldRef"))
}

/// PyCapsule name for [`crate::RegionRef`] handles:
/// `molrs.RegionRef/<abi_line>+<build_fingerprint>`.
///
/// A region crosses the boundary as shared geometry the consumer evaluates
/// in place (see [`crate::RegionRef`]); the versioned name keeps a
/// cross-minor exchange failing at the name check like the other handles.
pub fn regionref_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.RegionRef"))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The layout of the handle-crossing types on a 64-bit target. A change
    /// here inside a minor line is an ABI break: bump the minor version (and
    /// this snapshot), never only the snapshot.
    #[cfg(all(target_pointer_width = "64", feature = "ff"))]
    const SNAPSHOT: &[(&str, usize, usize)] = &[
        ("FrameRef", 16, 8),
        ("BlockRef", 48, 8),
        ("FrameArena", 32, 8),
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
    fn abi_line_is_major_minor_of_molrs_version() {
        let expected: Vec<&str> = env!("CARGO_PKG_VERSION").split('.').take(2).collect();
        assert_eq!(abi_line(), expected.join("."));
        assert_eq!(abi_line().split('.').count(), 2);
    }

    #[test]
    fn capsule_names_carry_the_abi_line_and_differ() {
        let frame = frameref_capsule_name().to_str().expect("utf8");
        let ff = forcefield_capsule_name().to_str().expect("utf8");
        let region = regionref_capsule_name().to_str().expect("utf8");
        let suffix = format!("{}+{}", abi_line(), build_fingerprint());
        assert_eq!(frame, format!("molrs.FrameRef/{suffix}"));
        assert_eq!(ff, format!("molrs.ForceFieldRef/{suffix}"));
        assert_eq!(region, format!("molrs.RegionRef/{suffix}"));
        assert_ne!(frame, ff);
        assert_ne!(frame, region);
        assert_ne!(ff, region);
    }
}
