//! molrs's CMAP against LAMMPS `fix cmap`, on the files molrs writes.
//!
//! An eight-atom backbone C-NH1-CT1-C-NH1-CT1-C-NH1, off-ideal, carries three
//! crossterms that [`assign_cmaps`] finds from its dihedrals: two typed with
//! CHARMM36's alanine map and one with that map transposed (a second map, so
//! the crossterm type a data file gives is checked to select the right one).
//!
//! The pinned numbers are LAMMPS's (30 Mar 2026 develop, `run 0`, `f_cmap`
//! and the per-atom forces dumped at `%.17g`) on exactly these inputs:
//! `scripts/lammps_cmap_check.sh` runs this test with `MOLRS_LAMMPS_CMAP_DIR`
//! set, so molrs writes the data file (`CMAP` section), the `fix cmap` file
//! and the include holding the `fix` line, then runs `lmp` on them and prints
//! the values.

use crate::core::Cell;
use crate::io::writer::WrittenText;
use std::path::Path;

use ndarray::{Array1, ArrayD, Axis};

use super::charmm::tests::{alanine, chain, place};
use crate::ff::compile::PotentialCompiler;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::CMAP_GRID;
use crate::ff::ir::Params;
use crate::ff::potential::fixtures::OneShot;
use crate::ff::typifier::cmap::assign_cmaps;
use crate::io::lammps::LammpsForcefieldWriter;
use molrs::core::Block;
use molrs::core::Frame;
use molrs::core::SimBox;
use molrs::core::TypeLabels;
use molrs::io::lammps::data::write_lammps_data;
use molrs::op::{F, Idx};

const TYPES: [&str; 8] = ["C", "NH1", "CT1", "C", "NH1", "CT1", "C", "NH1"];

/// LAMMPS `f_cmap`, kcal/mol (molrs: −1.25779219530854736, 1.1e-15
/// relative off).
const LAMMPS_ENERGY: F = -1.2577921953085487;

/// LAMMPS's per-atom forces (atom ids 1..8), kcal/mol/Å. molrs's are the same
/// bits but for two components of atom 5, 2e-16 relative off.
const LAMMPS_FORCES: [[F; 3]; 8] = [
    [-0.10699526639217385, 0.6012071735672395, -4.170500260493045],
    [
        0.020057257912214285,
        -5.345666324149171,
        -0.1696490151750738,
    ],
    [-3.0856988093900535, 7.336973164262715, 4.67371248099532],
    [9.08123759289216, -9.262396873949445, -2.5082322145813447],
    [-8.08916900558263, 3.483641434982568, 0.5324725848703223],
    [1.3247296316155102, 2.5591302767973274, 1.7177225121157325],
    [4.3105466301425945, 2.9752152013758804, 0.4854087786175037],
    [-3.454708031197621, -2.348104052887116, -0.5609348663494148],
];

fn system() -> (ForceField, Frame, Vec<F>) {
    let mut ff = ForceField::new("charmm");
    let style = ff.def_style("cmap", "charmm", Params::new()).unwrap();
    let mut ala = Params::new();
    ala.set(CMAP_GRID, alanine());
    style
        .def_type("ala", &["C", "NH1", "CT1", "C", "NH1"], ala)
        .unwrap();
    let mut swapped = Params::new();
    swapped.set(
        CMAP_GRID,
        alanine().reversed_axes().as_standard_layout().to_owned(),
    );
    style
        .def_type("swapped", &["NH1", "CT1", "C", "NH1", "CT1"], swapped)
        .unwrap();

    // φ/ψ pairs in three different cells, then every coordinate nudged.
    let mut x = chain(-63.0, -41.0);
    for (bond, angle, tors) in [
        (1.45, 121.0, -150.0),
        (1.52, 110.0, 75.0),
        (1.33, 117.0, 160.0),
    ] {
        let at = |k: usize| [x[3 * k], x[3 * k + 1], x[3 * k + 2]];
        let n = x.len() / 3;
        let next = place(at(n - 3), at(n - 2), at(n - 1), bond, angle, tors);
        x.extend(next);
    }
    for (k, v) in x.iter_mut().enumerate() {
        *v += 0.02 * ((k * 5 % 13) as F - 6.0) / 6.0 + 20.0;
    }

    let mut atoms = Block::new();
    for (k, key) in ["x", "y", "z"].into_iter().enumerate() {
        let col: Vec<F> = x.iter().skip(k).step_by(3).copied().collect();
        atoms
            .insert(key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let types: Vec<String> = TYPES.iter().map(|t| t.to_string()).collect();
    atoms
        .insert("type", Array1::from_vec(types).into_dyn(), None)
        .unwrap();
    let masses: Vec<F> = TYPES
        .iter()
        .map(|t| if t.starts_with('N') { 14.007 } else { 12.011 })
        .collect();
    atoms
        .insert("mass", Array1::from_vec(masses).into_dyn(), None)
        .unwrap();
    atoms
        .insert(
            "charge",
            Array1::from_vec(vec![0.0 as F; 8]).into_dyn(),
            None,
        )
        .unwrap();
    atoms
        .insert(
            "mol_id",
            Array1::from_vec(vec![1 as Idx; 8]).into_dyn(),
            None,
        )
        .unwrap();
    let mut dihedrals = Block::new();
    for (p, key) in ["atomi", "atomj", "atomk", "atoml"].into_iter().enumerate() {
        let col: Vec<Idx> = (0..5).map(|d| (d + p) as Idx).collect();
        dihedrals
            .insert(key, Array1::from_vec(col).into_dyn(), None)
            .unwrap();
    }
    let mut frame = Frame::new();
    frame.insert("atoms", atoms);
    frame.insert("dihedrals", dihedrals);
    frame.simbox = Some(SimBox::new(Cell::Lengths([40.0; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap());
    assert_eq!(assign_cmaps(&mut frame, &ff).unwrap(), 3);
    (ff, frame, x)
}

/// Write the LAMMPS inputs of [`system`] into `dir` (see the module docs).
fn stage_inputs(dir: &Path, ff: &ForceField, frame: &Frame) {
    let mut lammps = frame.clone();
    lammps.remove("dihedrals");
    write_lammps_data(&dir.join("data.lmp"), &lammps).unwrap();
    let labels = TypeLabels::from_frame(&lammps).unwrap();
    let writer = LammpsForcefieldWriter::new(&labels)
        .with_skip_pair_style(true)
        .with_cmap_file("charmm.cmap");
    std::fs::write(
        dir.join("charmm.cmap"),
        crate::io::writer::written_with(|b| writer.write_cmap(b, ff)).unwrap(),
    )
    .unwrap();
    std::fs::write(dir.join("system.ff"), writer.written(ff).unwrap()).unwrap();
}

#[test]
fn energy_and_forces_are_lammps_fix_cmap() {
    let (ff, frame, x) = system();
    let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
    let (energy, forces) = pots.energy_forces_at(&x);
    if let Some(dir) = std::env::var_os("MOLRS_LAMMPS_CMAP_DIR") {
        stage_inputs(Path::new(&dir), &ff, &frame);
        println!("molrs energy {energy:.17e}");
        for f in forces.chunks(3) {
            println!("molrs force {:.17e} {:.17e} {:.17e}", f[0], f[1], f[2]);
        }
        return;
    }
    let scale = LAMMPS_FORCES
        .iter()
        .flatten()
        .fold(0.0 as F, |m, v| m.max(v.abs()));
    assert!(
        (energy - LAMMPS_ENERGY).abs() <= 1e-8 * LAMMPS_ENERGY.abs(),
        "E {energy} vs LAMMPS {LAMMPS_ENERGY}"
    );
    let got = ArrayD::from_shape_vec(vec![8, 3], forces).unwrap();
    for (a, (row, want)) in got.axis_iter(Axis(0)).zip(LAMMPS_FORCES).enumerate() {
        for k in 0..3 {
            assert!(
                (row[k] - want[k]).abs() <= 1e-8 * scale,
                "atom {a} axis {k}: {} vs LAMMPS {}",
                row[k],
                want[k]
            );
        }
    }
}
