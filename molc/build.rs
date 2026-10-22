fn main() {
    // Declare cbindgen's real inputs so the generated header cannot go stale —
    // and so writing `include/molc.h` (an OUTPUT living inside the package
    // dir) does not re-trigger the script on every build.
    println!("cargo::rerun-if-changed=src");
    println!("cargo::rerun-if-changed=cbindgen.toml");
    println!("cargo::rerun-if-changed=Cargo.toml");
    println!("cargo::rerun-if-changed=build.rs");

    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let header_path = format!("{crate_dir}/include/molc.h");
    let config = cbindgen::Config::from_file("cbindgen.toml").unwrap_or_default();

    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(config)
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&header_path);
        }
        Err(e) => {
            panic!("cbindgen failed to generate {header_path}: {e:?}");
        }
    }
}
