//! CHARMM angle with Urey–Bradley (LAMMPS `angle_style charmm`).

use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::ANGLES;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{compute_angle, sub3, term_table, validate_coords};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use crate::op::vec3::norm;
use molrs::core::Frame;
use molrs::op::F;

/// One `angle charmm` type's numbers as the kernel holds them (`theta0` in
/// radians).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AngleCharmmParams {
    /// Bending constant, energy/rad².
    pub k: F,
    /// Equilibrium angle, **radians**.
    pub theta0: F,
    /// Urey–Bradley constant, energy/length².
    pub k_ub: F,
    /// Urey–Bradley equilibrium 1-3 distance.
    pub r_ub: F,
}

/// CHARMM angle + Urey–Bradley potential with pre-resolved flat arrays.
///
/// LAMMPS `angle_style charmm`:
/// E = k·(θ − θ0)² + k_ub·(r₁₃ − r_ub)².
///
/// `k` is LAMMPS's `K` (energy/rad², no ½), `theta0` is in **degrees**, `k_ub`
/// is LAMMPS's `K_ub` (energy/length², no ½) and `r_ub` a length — the four
/// numbers of an `angle_coeff t K theta0 K_ub r_ub` line, in that order. The
/// kernel converts `theta0` to radians once, at construction.
///
/// r₁₃ is the distance between the angle's end atoms `i` and `k`. The
/// Urey–Bradley term is a 1-3 harmonic spring that this angle owns: it adds
/// no exclusion and changes no pair list (which 1-3 pairs a non-bonded style
/// sees is `special_bonds`'s answer, as for any angle).
pub struct AngleCharmm {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    params: Vec<AngleCharmmParams>,
}

impl AngleCharmm {
    pub fn new(
        atom_i: Vec<usize>,
        atom_j: Vec<usize>,
        atom_k: Vec<usize>,
        params: Vec<AngleCharmmParams>,
    ) -> Self {
        let n = atom_i.len();
        assert_eq!(atom_j.len(), n);
        assert_eq!(atom_k.len(), n);
        assert_eq!(params.len(), n);
        Self {
            atom_i,
            atom_j,
            atom_k,
            params,
        }
    }

    /// The physics, once; the two entry points differ only in which atoms a
    /// term names.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize),
    ) -> F {
        let _n_atoms = validate_coords(coords);
        let mut energy: F = 0.0;

        for idx in 0..n_terms {
            let (i, j, k) = atoms(idx);
            let p = self.params[idx];

            // Bend: dE/dθ = 2k(θ − θ0).
            let dtheta = compute_angle(coords, i, j, k) - p.theta0;
            energy += p.k * dtheta * dtheta;
            super::accumulate_angle_forces(coords, i, j, k, 2.0 * p.k * dtheta, out);

            // Urey–Bradley: a spring along r₁₃ = x_i − x_k.
            let rik = sub3(coords, i, coords, k);
            let r = norm(rik);
            let dr = r - p.r_ub;
            energy += p.k_ub * dr * dr;
            if r > 0.0 {
                // F_i = −dE/dr · r̂, F_k = −F_i.
                let f_over_r = -2.0 * p.k_ub * dr / r;
                for (d, &x) in rik.iter().enumerate() {
                    out[i * 3 + d] += f_over_r * x;
                    out[k * 3 + d] -= f_over_r * x;
                }
            }
        }

        energy
    }
}

impl Potential for AngleCharmm {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| (self.atom_i[t], self.atom_j[t], self.atom_k[t]),
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for AngleCharmm {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k])
    }

    fn evaluate_terms(&self, coords: &[F], terms: ArrayView2<'_, u32>, acc: &mut Accumulator<'_>) {
        debug_assert_eq!(
            terms.nrows(),
            self.atom_i.len(),
            "the row set is the force field's; only the atoms a row names may be rebound"
        );
        let energy = self.fold(coords, acc.forces_without_virial(), terms.nrows(), |t| {
            (
                terms[[t, 0]] as usize,
                terms[[t, 1]] as usize,
                terms[[t, 2]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

/// Construct an [`AngleCharmm`] from style params, type params, and Frame
/// topology. Every type needs all four of `k`, `theta0` (deg), `k_ub`, `r_ub`.
pub fn angle_charmm_constructor(
    _style_params: &Params,
    type_params: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = type_params.iter().copied().collect();

    let block = frame
        .get(ANGLES)
        .ok_or_else(|| "AngleCharmm: frame missing \"angles\" block".to_string())?;
    let column = |key: &str| {
        block
            .get(key)
            .and_then(|c| c.get::<u64>())
            .ok_or_else(|| format!("AngleCharmm: angles block missing \"{key}\" column"))
    };
    let (i_col, j_col, k_col) = (column("atomi")?, column("atomj")?, column("atomk")?);
    let type_col = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or_else(|| "AngleCharmm: angles block missing \"type\" column".to_string())?;

    let n = i_col.len();
    let (mut ai, mut aj, mut ak) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    let mut params = Vec::with_capacity(n);
    for idx in 0..n {
        let label = &type_col[idx];
        let p = type_map
            .get(label.as_str())
            .ok_or_else(|| format!("AngleCharmm: unknown angle type '{label}'"))?;
        let need = |key: &str| param_reads::type_num("charmm", label, p, key);
        ai.push(i_col[idx] as usize);
        aj.push(j_col[idx] as usize);
        ak.push(k_col[idx] as usize);
        params.push(AngleCharmmParams {
            k: need("k")?,
            // A parameter in degrees (LAMMPS); the kernel works in radians.
            theta0: need("theta0")?.to_radians(),
            k_ub: need("k_ub")?,
            r_ub: need("r_ub")?,
        });
    }

    Ok(ForceTerm::indexed(AngleCharmm::new(ai, aj, ak, params)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::SpecialBonds;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::intramolecular_pairs;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

    const K: F = 33.43;
    const THETA0_DEG: F = 110.1;
    const K_UB: F = 22.53;
    const R_UB: F = 2.179;

    fn pot(theta0_deg: F) -> AngleCharmm {
        AngleCharmm::new(
            vec![0],
            vec![1],
            vec![2],
            vec![AngleCharmmParams {
                k: K,
                theta0: theta0_deg.to_radians(),
                k_ub: K_UB,
                r_ub: R_UB,
            }],
        )
    }

    /// Atom 0 at distance `a` along x, vertex at the origin, atom 2 at
    /// distance `b` at angle `theta_deg` in the xy-plane.
    fn geometry(a: F, b: F, theta_deg: F) -> Vec<F> {
        let t = theta_deg.to_radians();
        vec![a, 0.0, 0.0, 0.0, 0.0, 0.0, b * t.cos(), b * t.sin(), 0.0]
    }

    /// The LAMMPS manual's formula, by hand: law of cosines for r₁₃.
    fn hand_energy(a: F, b: F, theta_deg: F) -> F {
        let t = theta_deg.to_radians();
        let r13 = (a * a + b * b - 2.0 * a * b * t.cos()).sqrt();
        K * (t - THETA0_DEG.to_radians()).powi(2) + K_UB * (r13 - R_UB).powi(2)
    }

    #[test]
    fn energy_is_the_lammps_formula_over_several_geometries() {
        let p = pot(THETA0_DEG);
        for &(a, b, theta) in &[
            (1.53, 1.09, 110.1),
            (1.53, 1.09, 95.0),
            (1.40, 1.60, 130.0),
            (1.0, 1.0, 60.0),
            (1.2, 0.9, 175.0),
        ] {
            let got = p.energy_at(&geometry(a, b, theta));
            let want = hand_energy(a, b, theta);
            assert!(
                (got - want).abs() <= 1e-12 * want.abs().max(1.0),
                "a={a} b={b} θ={theta}: {got} vs {want}"
            );
        }
    }

    #[test]
    fn at_both_equilibria_the_energy_and_force_vanish() {
        // θ = θ0 and r₁₃ = r_ub: a = b = r_ub / (2 sin(θ0/2)).
        let a = R_UB / (2.0 * (THETA0_DEG.to_radians() / 2.0).sin());
        let (e, f) = pot(THETA0_DEG).energy_forces_at(&geometry(a, a, THETA0_DEG));
        assert!(e.abs() < 1e-24, "{e}");
        assert!(f.iter().all(|x| x.abs() < 1e-10), "{f:?}");
    }

    #[test]
    fn forces_match_finite_difference_and_sum_to_zero() {
        let p = pot(THETA0_DEG);
        let coords: Vec<F> = vec![1.3, 0.2, -0.1, 0.05, 0.0, 0.1, -0.4, 1.1, 0.3];
        let (_, analytic) = p.energy_forces_at(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let (mut cp, mut cm) = (coords.clone(), coords.clone());
            cp[d] += h;
            cm[d] -= h;
            let numeric = -(p.energy_at(&cp) - p.energy_at(&cm)) / (2.0 * h);
            assert!(
                (analytic[d] - numeric).abs() < 1e-6,
                "dof {d}: {} vs {numeric}",
                analytic[d]
            );
        }
        // Newton's third law: no net force, and no net torque.
        for d in 0..3 {
            let net: F = (0..3).map(|a| analytic[a * 3 + d]).sum();
            assert!(net.abs() < 1e-10, "net force {d}: {net}");
        }
        let mut torque = [0.0; 3];
        for a in 0..3 {
            let (x, f) = (&coords[a * 3..a * 3 + 3], &analytic[a * 3..a * 3 + 3]);
            torque[0] += x[1] * f[2] - x[2] * f[1];
            torque[1] += x[2] * f[0] - x[0] * f[2];
            torque[2] += x[0] * f[1] - x[1] * f[0];
        }
        assert!(torque.iter().all(|t| t.abs() < 1e-10), "{torque:?}");
    }

    #[test]
    fn the_urey_bradley_force_lies_along_r13() {
        // k = 0 isolates the UB term: the force on i is parallel to x_i − x_k
        // and the vertex feels nothing.
        let p = AngleCharmm::new(
            vec![0],
            vec![1],
            vec![2],
            vec![AngleCharmmParams {
                k: 0.0,
                theta0: 0.0,
                k_ub: K_UB,
                r_ub: R_UB,
            }],
        );
        let coords = geometry(1.5, 1.2, 100.0);
        let (_, f) = p.energy_forces_at(&coords);
        let r13 = sub3(&coords, 0, &coords, 2);
        let cross = [
            f[1] * r13[2] - f[2] * r13[1],
            f[2] * r13[0] - f[0] * r13[2],
            f[0] * r13[1] - f[1] * r13[0],
        ];
        assert!(cross.iter().all(|c| c.abs() < 1e-12), "{cross:?}");
        assert!(f[3..6].iter().all(|x| *x == 0.0), "{f:?}");
    }

    fn frame(label: &str, coords: &[F]) -> Frame {
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].into_iter().enumerate() {
            let col: Vec<F> = (0..3).map(|a| coords[a * 3 + d]).collect();
            atoms
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let mut angles = Block::new();
        for (key, a) in [("atomi", 0), ("atomj", 1), ("atomk", 2)] {
            angles
                .insert(key, Array1::from_vec(vec![a as Idx]).into_dyn(), None)
                .unwrap();
        }
        angles
            .insert(
                "type",
                Array1::from_vec(vec![label.to_owned()]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("angles", angles);
        frame
    }

    fn charmm_ff() -> ForceField {
        let mut ff = ForceField::new("t");
        ff.def_style("angle", "charmm", Params::new())
            .unwrap()
            .def_type(
                "HA-CT-HA",
                &["HA", "CT", "HA"],
                Params::from_iter([
                    ("k", K),
                    ("theta0", THETA0_DEG),
                    ("k_ub", K_UB),
                    ("r_ub", R_UB),
                ]),
            )
            .unwrap();
        ff
    }

    /// `theta0` is stored in degrees, as the `angle_coeff` line writes it, and
    /// the compiled energy is the formula with θ0 in radians.
    #[test]
    fn compiled_theta0_is_degrees() {
        let coords = geometry(1.53, 1.09, 104.0);
        let pots = PotentialCompiler::new(&charmm_ff())
            .compile(&frame("HA-CT-HA", &coords))
            .unwrap();
        let got = pots.energy_at(&coords);
        let want = hand_energy(1.53, 1.09, 104.0);
        assert!((got - want).abs() <= 1e-12 * want, "{got} vs {want}");
    }

    #[test]
    fn both_compile_doors_price_it_the_same() {
        let coords: Vec<F> = vec![1.3, 0.2, -0.1, 0.05, 0.0, 0.1, -0.4, 1.1, 0.3];
        let ff = charmm_ff();
        let compiler = PotentialCompiler::new(&ff);
        let f = frame("HA-CT-HA", &coords);
        let (e1, f1) = compiler.compile(&f).unwrap().energy_forces_at(&coords);
        let typed = compiler.compile_typed(&f).unwrap();
        assert_eq!(typed.len(), 1);
        let (member, weights) = &typed[0];
        assert!(
            weights.is_none(),
            "a bonded member takes no special weights"
        );
        let (e2, f2) = member.energy_forces_at(&coords);
        assert_eq!(e1, e2);
        assert_eq!(f1, f2);
        assert_eq!(e1, pot(THETA0_DEG).energy_at(&coords));
    }

    #[test]
    fn a_type_missing_a_urey_bradley_param_is_refused() {
        let mut ff = ForceField::new("t");
        ff.def_style("angle", "charmm", Params::new())
            .unwrap()
            .def_type(
                "A",
                &["A", "A", "A"],
                Params::from_iter([("k", K), ("theta0", THETA0_DEG), ("k_ub", K_UB)]),
            )
            .unwrap();
        let coords = geometry(1.0, 1.0, 100.0);
        let err = PotentialCompiler::new(&ff)
            .compile(&frame("A", &coords))
            .map(|_| ())
            .unwrap_err();
        assert!(
            matches!(
                err.ir(),
                Some(crate::ff::ir::IrError::MissingParam { style, type_, param })
                    if style == "charmm" && type_ == "A" && param == "r_ub"
            ),
            "{err}"
        );
    }

    /// Urey–Bradley creates no exclusion: whether the 1-3 pair of a charmm
    /// angle is in the pair list is `special_bonds`'s answer alone, exactly as
    /// for any other angle style.
    #[test]
    fn urey_bradley_changes_no_pair_list() {
        let f = frame("HA-CT-HA", &geometry(1.0, 1.0, 100.0));
        let has_13 = |special: &SpecialBonds| {
            let p = intramolecular_pairs(&f, special, None).unwrap();
            let i = p.get("atomi").unwrap().get::<u64>().unwrap();
            let j = p.get("atomj").unwrap().get::<u64>().unwrap();
            i.iter().zip(j.iter()).any(|(&a, &b)| (a, b) == (0, 2))
        };
        let keep_13 = SpecialBonds {
            lj: [0.0, 1.0, 1.0],
            coul: [0.0, 1.0, 1.0],
        };
        assert!(
            has_13(&keep_13),
            "special_bonds keeps 1-3; UB must not drop it"
        );
        assert!(!has_13(&SpecialBonds::default()));
    }

    /// A frame of `xyz` with one `angles` row per `(i, j, k, type)`.
    fn molecule(xyz: &[[F; 3]], angles: &[(Idx, Idx, Idx, &str)]) -> Frame {
        let mut atoms = Block::new();
        for (d, key) in ["x", "y", "z"].into_iter().enumerate() {
            let col: Vec<F> = xyz.iter().map(|p| p[d]).collect();
            atoms
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let mut block = Block::new();
        for (slot, key) in ["atomi", "atomj", "atomk"].into_iter().enumerate() {
            let col: Vec<Idx> = angles.iter().map(|a| [a.0, a.1, a.2][slot]).collect();
            block
                .insert(key, Array1::from_vec(col).into_dyn(), None)
                .unwrap();
        }
        let types: Vec<String> = angles.iter().map(|a| a.3.to_owned()).collect();
        block
            .insert("type", Array1::from_vec(types).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("angles", block);
        frame
    }

    /// Read `include` with the LAMMPS reader, compile it on `frame`, and hold
    /// the energy to LAMMPS's `pe` (= `eangle`) at 1e-10 relative and the
    /// forces to its `fx fy fz` at 1e-10 of the largest component.
    fn agrees_with_lammps(include: &str, frame: &Frame, pe: F, forces: &[[F; 3]]) {
        use crate::io::{lammps::LammpsForcefieldReader, reader::ForceFieldReader};
        let ff = LammpsForcefieldReader::new()
            .read(include.as_bytes())
            .unwrap();
        let coords: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let (e, f) = PotentialCompiler::new(&ff)
            .compile(frame)
            .unwrap()
            .energy_forces_at(&coords);
        assert!(
            (e - pe).abs() <= 1e-10 * pe.abs(),
            "molrs {e} vs LAMMPS {pe}"
        );
        let scale = forces
            .iter()
            .flatten()
            .fold(0.0 as F, |m, x| m.max(x.abs()));
        for (a, want) in forces.iter().enumerate() {
            for d in 0..3 {
                let got = f[a * 3 + d];
                assert!(
                    (got - want[d]).abs() <= 1e-10 * scale,
                    "atom {a} dim {d}: molrs {got} vs LAMMPS {}",
                    want[d]
                );
            }
        }
    }

    /// One `angle_style charmm` angle against LAMMPS (30 Mar 2026, `units
    /// real`, `atom_style molecular`, `run 0`, the include below after
    /// `read_data`): `pe` = `eangle` = 0.024871552479721934 kcal/mol, and the
    /// `write_dump … fx fy fz` forces.
    #[test]
    fn three_atom_urey_bradley_agrees_with_lammps_run_0() {
        let include = "\
special_bonds lj 0.0 0.0 0.0 coul 0.0 0.0 0.0
angle_style charmm
angle_coeff HA-CT-HA 35.5 108.4 5.4 1.802
";
        let frame = molecule(
            &[
                [1.0915, 0.0213, -0.0317],
                [0.0, 0.0, 0.0],
                [-0.3420, 1.0290, 0.0520],
            ],
            &[(0, 1, 2, "HA-CT-HA")],
        );
        agrees_with_lammps(
            include,
            &frame,
            0.024871552479721934,
            &[
                [
                    0.4437994938277033,
                    -1.5177102557296727,
                    -0.07488784735459168,
                ],
                [1.1445217256207856, 1.6102574444284228, 0.03125901116125017],
                [
                    -1.5883212194484888,
                    -0.09254718869875017,
                    0.043628836193341514,
                ],
            ],
        );
    }

    /// `angle_style hybrid harmonic charmm` — the LAMMPS writer's own output
    /// for a mixed field — against LAMMPS `run 0`: `pe` = `eangle` =
    /// 0.3573651687304709 kcal/mol.
    #[test]
    fn hybrid_harmonic_charmm_agrees_with_lammps_run_0() {
        let include = "\
special_bonds lj 0.000000 0.000000 0.000000 coul 0.000000 0.000000 0.000000

angle_style hybrid harmonic charmm
angle_coeff CT-CT-CT harmonic 58.350000 113.600000
angle_coeff HA-CT-CT charmm 33.430000 110.100000 22.530000 2.179000
angle_coeff HA-CT-HA charmm 35.500000 108.400000 5.400000 1.802000
";
        let frame = molecule(
            &[
                [0.0, 0.0, 0.0],
                [1.53, 0.0, 0.0],
                [2.05, 1.44, 0.10],
                [-0.36, 1.02, 0.05],
                [-0.38, -0.50, 0.89],
            ],
            &[
                (0, 1, 2, "CT-CT-CT"),
                (3, 0, 1, "HA-CT-CT"),
                (3, 0, 4, "HA-CT-HA"),
            ],
        );
        agrees_with_lammps(
            include,
            &frame,
            0.3573651687304709,
            &[
                [-1.603595194504185, -2.64881509442167, 2.6016199494198533],
                [-3.5132089052852495, 5.555283389958266, 0.40982273406863357],
                [4.732030500316453, -1.7005876553258585, -0.11809636495318464],
                [-0.7362677577264647, 1.54982962383793, -2.5143182980783343],
                [1.121041357199446, -2.755710264048667, -0.379028020456968],
            ],
        );
    }

    /// Beside another angle style, each style prices its own rows (the
    /// compiler cuts the `angles` block per style), both doors agree, and a
    /// row whose type neither style defines is still refused.
    #[test]
    fn a_mixed_angle_field_prices_each_row_under_its_own_style() {
        let mut ff = charmm_ff();
        ff.def_style("angle", "harmonic", Params::new())
            .unwrap()
            .def_type(
                "CT-CT-CT",
                &["CT", "CT", "CT"],
                Params::from_iter([("k", 58.35), ("theta0", 113.6)]),
            )
            .unwrap();
        let xyz = [
            [0.0, 0.0, 0.0],
            [1.53, 0.0, 0.0],
            [2.05, 1.44, 0.10],
            [-0.36, 1.02, 0.05],
        ];
        let coords: Vec<F> = xyz.iter().flatten().copied().collect();
        let both = molecule(&xyz, &[(0, 1, 2, "CT-CT-CT"), (3, 0, 1, "HA-CT-HA")]);
        let compiler = PotentialCompiler::new(&ff);
        let e = compiler.compile(&both).unwrap().energy_at(&coords);
        let typed: F = compiler
            .compile_typed(&both)
            .unwrap()
            .iter()
            .map(|(m, _)| m.energy_at(&coords))
            .sum();
        assert_eq!(e, typed);
        let harmonic = 58.35 * (compute_angle(&coords, 0, 1, 2) - 113.6_f64.to_radians()).powi(2);
        let ub = pot(THETA0_DEG).energy_at(&[
            xyz[3][0], xyz[3][1], xyz[3][2], 0.0, 0.0, 0.0, 1.53, 0.0, 0.0,
        ]);
        assert!(
            (e - (harmonic + ub)).abs() <= 1e-12 * e,
            "{e} vs {}",
            harmonic + ub
        );

        let stray = molecule(&xyz, &[(0, 1, 2, "CT-CT-CT"), (3, 0, 1, "XX-XX-XX")]);
        let err = compiler.compile(&stray).map(|_| ()).unwrap_err();
        assert!(
            err.to_string()
                .contains("'XX-XX-XX' is defined by no angle style"),
            "{err}"
        );
    }
}
