//! Records what the handle fingerprint (`abi::build_fingerprint`) is made of
//! beyond the type layouts: the compiler that builds this crate, because struct
//! layout is the compiler's choice, and the molrs source it embeds, because two
//! builds with equal type sizes can still disagree on what sits inside them.
//! Any edit to the molrs or molrs-ffi sources changes the fingerprint, so a
//! consumer linked against an older build (Atomiverse, molpack) refuses molpy's
//! capsules with a clean error instead of reading them wrong: it is rebuilt.

use std::fs;
use std::path::{Path, PathBuf};
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
    println!("cargo:rustc-env=MOLRS_FFI_SOURCE={}", source_digest());
}

/// FNV-1a over every file of the molrs and molrs-ffi sources (relative path,
/// then bytes), in sorted order. A crate built from crates.io has no sibling
/// checkout; its published version names its source exactly.
fn source_digest() -> String {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").expect("cargo sets it"));
    let molrs = manifest.join("../molrs");
    if !molrs.join("src").is_dir() {
        return format!(
            "release-{}",
            std::env::var("CARGO_PKG_VERSION").expect("cargo sets it")
        );
    }
    let roots = [
        manifest.join("src"),
        molrs.join("src"),
        molrs.join("Cargo.toml"),
    ];
    let mut files = Vec::new();
    for root in &roots {
        println!("cargo:rerun-if-changed={}", root.display());
        collect(root, &mut files);
    }
    files.sort();
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: &[u8]| {
        for byte in bytes {
            hash ^= u64::from(*byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    };
    for file in &files {
        let relative = file.strip_prefix(&manifest).unwrap_or(file);
        feed(relative.to_string_lossy().as_bytes());
        feed(&fs::read(file).unwrap_or_else(|e| panic!("cannot read {}: {e}", file.display())));
    }
    format!("{hash:016x}")
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) {
    if path.is_file() {
        out.push(path.to_path_buf());
        return;
    }
    let Ok(entries) = fs::read_dir(path) else {
        return;
    };
    for entry in entries.flatten() {
        collect(&entry.path(), out);
    }
}
