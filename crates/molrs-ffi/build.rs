//! Records the compiler that builds this crate: part of the ABI fingerprint
//! (`abi::build_fingerprint`), because struct layout is the compiler's choice
//! and two builds of one minor line by different `rustc`s may lay a handle
//! out differently.

use std::process::Command;

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    let rustc = std::env::var("RUSTC").expect("cargo sets RUSTC for build scripts");
    let output = Command::new(&rustc)
        .arg("-V")
        .output()
        .unwrap_or_else(|e| panic!("cannot run {rustc} -V: {e}"));
    assert!(output.status.success(), "{rustc} -V failed");
    let version = String::from_utf8(output.stdout).expect("rustc -V prints UTF-8");
    println!("cargo:rustc-env=MOLRS_FFI_RUSTC={}", version.trim());
}
