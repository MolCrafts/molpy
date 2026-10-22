//! The ABI contract of the molrs FFI layer.
//!
//! All cross-extension interop (molrs-python ↔ molpack, cxxapi consumers)
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

fn versioned_name(prefix: &str) -> CString {
    CString::new(format!("{prefix}/{}", abi_line())).expect("capsule name has no NUL")
}

/// PyCapsule name for [`crate::FrameRef`] handles: `molrs.FrameRef/<abi_line>`.
///
/// The name carries the ABI line so a cross-minor handle exchange fails the
/// capsule name check (a clean `ValueError`) instead of dereferencing a
/// possibly drifted layout.
pub fn frameref_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.FrameRef"))
}

/// PyCapsule name for `ForceFieldRef` handles: `molrs.ForceFieldRef/<abi_line>`.
///
/// Not gated on the `ff` feature — it is only a string, and a core-only build
/// may still need to *recognize* (and reject) a force-field capsule.
pub fn forcefield_capsule_name() -> &'static CStr {
    static NAME: OnceLock<CString> = OnceLock::new();
    NAME.get_or_init(|| versioned_name("molrs.ForceFieldRef"))
}

/// PyCapsule name for [`crate::RegionRef`] handles: `molrs.RegionRef/<abi_line>`.
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
        assert_eq!(frame, format!("molrs.FrameRef/{}", abi_line()));
        assert_eq!(ff, format!("molrs.ForceFieldRef/{}", abi_line()));
        assert_eq!(region, format!("molrs.RegionRef/{}", abi_line()));
        assert_ne!(frame, ff);
        assert_ne!(frame, region);
        assert_ne!(ff, region);
    }
}
