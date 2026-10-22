//! Harmonic proper dihedral (LAMMPS `dihedral_style harmonic`).

use molrs::core::Frame;
use molrs::core::schema::block_names::DIHEDRALS;

use crate::ff::ir::Params;
use crate::ff::potential::ForceTerm;
use crate::ff::potential::improper::cvff::signed_cosine_constructor;

/// Construct a harmonic dihedral from per-type params (`k`, `sign`,
/// `periodicity`) and a Frame's `"dihedrals"` block.
///
/// Harmonic proper dihedral (LAMMPS `dihedral_style harmonic`):
///
/// E(φ) = k · [1 + sign · cos(n·φ)]
///
/// `k` is LAMMPS's `K` (energy), `sign` its `d` (±1 — a sign, not a phase) and
/// `periodicity` its `n`. It is the same function of the dihedral as LAMMPS
/// `improper_style cvff`, so it shares that kernel
/// ([`ImproperCvff`](crate::ff::potential::improper::ImproperCvff)),
/// evaluated over the `"dihedrals"` block.
pub fn dihedral_harmonic_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    signed_cosine_constructor(DIHEDRALS, "harmonic", tp, frame)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::forcefield::ForceField;
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::op::{F, Idx};
    use ndarray::Array1;

    /// LAMMPS `dihedral_style harmonic`: E = K[1 + d·cos(nφ)], here
    /// K = 2, d = −1, n = 2 at φ = 30°.
    #[test]
    fn energy_is_the_lammps_formula() {
        let mut ff = ForceField::new("t");
        ff.def_style("dihedral", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "a-b-c-d",
                &["a", "b", "c", "d"],
                Params::from_iter([("k", 2.0), ("sign", -1.0), ("periodicity", 2.0)]),
            )
            .unwrap();
        let mut dihedrals = Block::new();
        for (key, atom) in [("atomi", 0), ("atomj", 1), ("atomk", 2), ("atoml", 3)] {
            dihedrals
                .insert(key, Array1::from_vec(vec![atom as Idx]).into_dyn(), None)
                .unwrap();
        }
        dihedrals
            .insert(
                "type",
                Array1::from_vec(vec!["a-b-c-d".to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("dihedrals", dihedrals);
        let pots = PotentialCompiler::new(&ff).compile(&frame).unwrap();
        let phi: F = 30.0_f64.to_radians();
        let (s, c) = phi.sin_cos();
        let coords = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s];
        let want = 2.0 * (1.0 - (2.0 * phi).cos());
        let got = pots.energy_at(&coords);
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}
