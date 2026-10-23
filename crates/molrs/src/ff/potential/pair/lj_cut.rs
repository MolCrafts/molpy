//! Lennard-Jones / Mie pair style (`lj/cut`): its form, and the one-type
//! kernel a neighbour loop feeds ([`PairLjCut`]).

use std::sync::Arc;

use crate::ff::ir::Params;
use crate::ff::potential::form_kernel::{
    ParamColumns, ScalarForm, ScalarPair, StyleRefusal, eval_by_term,
};
use crate::ff::potential::pair::PairPotential;
use crate::ff::potential::{Accumulator, PairDriven, Potential};
use molrs::core::Neighbors;
use molrs::op::F;

/// `C = n/(n − m) (n/m)^(m/(n − m))`, the Mie prefactor (4 at 12-6).
fn mie_c(n: F, m: F) -> F {
    (n / (n - m)) * (n / m).powf(m / (n - m))
}

/// LAMMPS `pair_style lj/cut` (and its Mie generalisation `lj/mie/cut`):
///
/// ```text
/// E = C ε [(σ/r)ⁿ − (σ/r)ᵐ] − E_shift,     C = n/(n − m) (n/m)^(m/(n − m))
/// ```
///
/// with the style's exponents `n > m > 0` (integers, 12-6 by default) and,
/// when `shift` is set (`pair_modify shift yes`), `E_shift` the energy at the
/// style's `cutoff`.
#[derive(Clone, Copy, Debug, Default)]
pub struct LjCut;

impl ScalarForm for LjCut {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let law = self.pair_law(p)?;
        Some(move |r: F, x: &[F]| {
            let (e, fpair) = law(r * r, x);
            (e, -fpair * r)
        })
    }

    /// The law in `r²`: the 12-6 needs no `r`, as LAMMPS's `lj/cut` takes
    /// none.
    fn pair_law(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let (n, m) = (p.style_input("n"), p.style_input("m"));
        let shifted = p.style_input("shift") != 0.0;
        let cutoff = p.style_input("cutoff");
        let c = mie_c(n, m);
        let (ni, mi) = (n as i32, m as i32);
        let twelve_six = ni == 12 && mi == 6;
        Some(move |r2: F, x: &[F]| {
            let (eps, sigma) = (x[0], x[1]);
            let ceps = c * eps;
            // The energy at the cutoff a shifted style subtracts (LAMMPS's
            // `offset`).
            let offset = if shifted {
                let sr_c = sigma / cutoff;
                ceps * (sr_c.powi(ni) - sr_c.powi(mi))
            } else {
                0.0
            };
            if twelve_six {
                let inv_r2 = 1.0 / r2;
                let sr2 = sigma * sigma * inv_r2;
                let sr6 = sr2 * sr2 * sr2;
                let sr12 = sr6 * sr6;
                (
                    ceps * (sr12 - sr6) - offset,
                    ceps * (12.0 * sr12 - 6.0 * sr6) * inv_r2,
                )
            } else {
                let sr = sigma / r2.sqrt();
                let sr_n = sr.powi(ni);
                let sr_m = sr.powi(mi);
                (
                    ceps * (sr_n - sr_m) - offset,
                    ceps * (n * sr_n - m * sr_m) / r2,
                )
            }
        })
    }

    fn inputs(&self) -> Vec<String> {
        ["epsilon", "sigma", "n", "m", "shift", "cutoff"]
            .map(String::from)
            .to_vec()
    }

    /// The exponents are integers with `n > m > 0`.
    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        let (n, m) = (p.style_input("n"), p.style_input("m"));
        for (key, v) in [("n", n), ("m", m)] {
            if v.fract() != 0.0 {
                return Err(StyleRefusal::new(key, format!("= {v} is not an integer")));
            }
        }
        if m <= 0.0 || n <= m {
            return Err(StyleRefusal::new(
                "n",
                format!("= {n} with m = {m}: the exponents satisfy n > m > 0"),
            ));
        }
        Ok(())
    }
}

/// A form shifted to zero force and energy at the style's `cutoff`
/// (`E(r) − E(r_c) − (r − r_c) E′(r_c)`): the "smeared", shifted-force
/// truncation of a neighbour-loop kernel.
struct ForceShifted<T>(T);

impl<T: ScalarForm> ScalarForm for ForceShifted<T> {
    fn eval(&self, r: &[F], p: &ParamColumns<'_>, e: &mut [F], de: &mut [F]) {
        eval_by_term(self, r, p, e, de);
    }

    fn pointwise(&self, p: &ParamColumns<'_>) -> Option<impl Fn(F, &[F]) -> (F, F) + '_> {
        let law = self.0.pointwise(p)?;
        let cutoff = p.style_input("cutoff");
        Some(move |r: F, x: &[F]| {
            let (e, de) = law(r, x);
            let (e_c, de_c) = law(cutoff, x);
            (e - (e_c + (r - cutoff) * de_c), de - de_c)
        })
    }

    fn inputs(&self) -> Vec<String> {
        self.0.inputs()
    }

    fn check_style(&self, p: &ParamColumns<'_>) -> Result<(), StyleRefusal> {
        self.0.check_style(p)
    }
}

/// The kernel of a [`PairLjCut`], monomorphized on its form.
#[derive(Clone)]
enum UniformLj {
    Shifted(ScalarPair<LjCut>),
    Smeared(ScalarPair<ForceShifted<LjCut>>),
}

/// LAMMPS `pair_style lj/cut` of one atom type: one `(ε, σ)` priced for
/// every pair a neighbour table hands it — the MD integrators' nonbond
/// kernel, and molpy's `PairLjCut`.
///
/// It is the `lj/cut` form ([`LjCut`]) over a one-type pair table, so it
/// prices a pair exactly as a compiled force field's `lj/cut` does: only at
/// `r < cutoff`, shifted to zero energy there when `shifted`, and shifted to
/// zero force too when `smeared` (which implies `shifted`). A force field's
/// `lj/cut` with per-type rows is compiled through
/// [`PotentialCompiler`](crate::ff::compile::PotentialCompiler).
#[derive(Clone)]
pub struct PairLjCut {
    kernel: UniformLj,
    epsilon: F,
    sigma: F,
    cutoff: F,
    n: i32,
    m: i32,
    shifted: bool,
    smeared: bool,
}

impl PairLjCut {
    /// `ε`, `σ` and `cutoff` positive, `n > m > 0`.
    ///
    /// # Errors
    ///
    /// A parameter outside its domain.
    pub fn new(
        epsilon: F,
        sigma: F,
        cutoff: F,
        n: i32,
        m: i32,
        shifted: bool,
        smeared: bool,
    ) -> Result<Self, String> {
        if epsilon <= 0.0 || sigma <= 0.0 || cutoff <= 0.0 {
            return Err("PairLjCut requires epsilon, sigma, cutoff > 0".into());
        }
        if m <= 0 || n <= m {
            return Err(format!(
                "PairLjCut exponents must satisfy n > m > 0, got n={n}, m={m}"
            ));
        }
        let shifted = shifted || smeared;
        let spec = crate::ff::style_registry::with_global_registry(|r| {
            r.style("pair", "lj/cut").map(|(spec, _)| spec.clone())
        })
        .ok_or("pair lj/cut is a built-in style")?;
        let style = Params::from_iter([
            ("cutoff", cutoff),
            ("n", F::from(n)),
            ("m", F::from(m)),
            ("shift", if shifted { 1.0 } else { 0.0 }),
        ]);
        let row = Params::from_iter([("epsilon", epsilon), ("sigma", sigma)]);
        let kernel = if smeared {
            UniformLj::Smeared(
                ScalarPair::uniform(Arc::new(ForceShifted(LjCut)), &spec, &style, &row)
                    .map_err(|e| e.to_string())?
                    .fused(),
            )
        } else {
            UniformLj::Shifted(
                ScalarPair::uniform(Arc::new(LjCut), &spec, &style, &row)
                    .map_err(|e| e.to_string())?
                    .fused(),
            )
        };
        Ok(Self {
            kernel,
            epsilon,
            sigma,
            cutoff,
            n,
            m,
            shifted,
            smeared,
        })
    }

    pub fn epsilon(&self) -> F {
        self.epsilon
    }
    pub fn sigma(&self) -> F {
        self.sigma
    }
    pub fn cutoff(&self) -> F {
        self.cutoff
    }
    pub fn n(&self) -> i32 {
        self.n
    }
    pub fn m(&self) -> i32 {
        self.m
    }
    pub fn shifted(&self) -> bool {
        self.shifted
    }
    pub fn smeared(&self) -> bool {
        self.smeared
    }
}

impl PairPotential for PairLjCut {
    fn pair_energy_force(&self, r2: F, disp: [F; 3]) -> Option<(F, [F; 3])> {
        match &self.kernel {
            UniformLj::Shifted(k) => k.pair_energy_force(r2, disp),
            UniformLj::Smeared(k) => k.pair_energy_force(r2, disp),
        }
    }
}

impl Potential for PairLjCut {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        // A pair table nobody handed over: nothing to add.
        match &self.kernel {
            UniformLj::Shifted(k) => k.evaluate(coords, acc),
            UniformLj::Smeared(k) => k.evaluate(coords, acc),
        }
    }
}

impl PairDriven for PairLjCut {
    fn evaluate_pairs(
        &self,
        coords: &[F],
        pairs: &Neighbors,
        factor: &[F],
        acc: &mut Accumulator<'_>,
    ) {
        match &self.kernel {
            UniformLj::Shifted(k) => k.evaluate_pairs(coords, pairs, factor, acc),
            UniformLj::Smeared(k) => k.evaluate_pairs(coords, pairs, factor, acc),
        }
    }

    fn binds_a_fixed_pair_list(&self) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::Cell;
    use crate::ff::compile::{ExplicitTerms, PotentialCompiler};
    use crate::ff::forcefield::ForceField;
    use crate::ff::ir::SpecialBonds;
    use crate::ff::potential::ForceTerm;
    use crate::ff::potential::fixtures::OneShot;
    use crate::ff::potential::pair::fixtures::{assert_same, table_over};
    use molrs::core::{Block, Frame};
    use molrs::op::Idx;
    use ndarray::Array1;

    /// The fold gives the same number, bit for bit, however many threads ran.
    ///
    /// Above a pair-count threshold the fold is split across threads, and a
    /// sum of floats is not associative — so the split has to be by a fixed
    /// arithmetic boundary and the partials merged in chunk order. rayon's own
    /// adaptive split would reorder the additions and move the last bits of
    /// every energy, force and virial with them. This asserts the property
    /// rather than the intent: the table here is deliberately larger than the
    /// threshold, so the parallel path is the one being measured.
    #[cfg(feature = "rayon")]
    #[test]
    fn the_fold_does_not_depend_on_the_thread_count() {
        use molrs::core::SimBox;
        use molrs::core::{NeighborList, NeighborPolicy, VerletSkin};

        // 10³ atoms at 3 Å with a 6 Å cutoff clears 8192 pairs comfortably.
        let side = 10_usize;
        let n = side * side * side;
        let l = side as F * 3.0;
        let bx = SimBox::new(Cell::Lengths([l; 3]), [0.0, 0.0, 0.0], [true; 3]).unwrap();
        let pos = ndarray::Array2::from_shape_fn((n, 3), |(a, k)| {
            let (i, j, m) = (a % side, (a / side) % side, a / (side * side));
            let base = [i, j, m][k] as F * 3.0;
            // A deterministic jitter, so no two separations coincide and a
            // reordering cannot be masked by equal terms.
            base + ((a * 7 + k * 13) as F * 0.618).fract() * 0.4
        });
        let mut skin = VerletSkin::new(
            NeighborList::new(6.0).unwrap(),
            6.0,
            NeighborPolicy {
                skin: 0.0,
                ..NeighborPolicy::default()
            },
            pos.view(),
            bx,
        )
        .unwrap();
        let table = skin.pairs_at(pos.view()).unwrap().clone();
        assert!(
            table.query_point_indices().len() > 8_192,
            "the fixture must cross the parallel threshold"
        );

        // Two types, alternating, mixed arithmetically.
        let mut ff = ForceField::new("t");
        ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
        let mut style = Params::from_iter([("cutoff", 6.0)]);
        style.set("mixing", "arithmetic");
        ff.def_style("pair", "lj/cut", style)
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([("epsilon", 0.3), ("sigma", 3.4)]),
            )
            .unwrap()
            .def_type(
                "B",
                &["B"],
                Params::from_iter([("epsilon", 0.5), ("sigma", 3.0)]),
            )
            .unwrap();
        let mut atoms = Block::new();
        let types: Vec<String> = (0..n)
            .map(|i| if i % 2 == 0 { "A" } else { "B" }.to_owned())
            .collect();
        atoms
            .insert("type", Array1::from_vec(types).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        let members = PotentialCompiler::new(&ff).compile_typed(&frame).unwrap();
        let ForceTerm::Pair(kernel) = &members[0].0 else {
            panic!("a pair member");
        };
        let flat: Vec<F> = pos.iter().copied().collect();

        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let (e, out, w) = kernel.over_pairs_virial(&flat, &table);
                (e, out, w.unwrap())
            })
        };

        let (e1, f1, w1) = run(1);
        assert!(e1.abs() > 1.0, "the fixture must interact");
        for threads in [2, 4, 8] {
            let (e, f, w) = run(threads);
            assert_eq!(e1.to_bits(), e.to_bits(), "{threads} threads: energy");
            for (k, (a, b)) in f1.iter().zip(&f).enumerate() {
                assert_eq!(a.to_bits(), b.to_bits(), "{threads} threads: force {k}");
            }
            for c in 0..6 {
                assert_eq!(
                    w1.components[c].to_bits(),
                    w.components[c].to_bits(),
                    "{threads} threads: virial {c}"
                );
            }
        }
    }

    #[test]
    fn mie_c_is_four_for_12_6() {
        assert!((mie_c(12.0, 6.0) - 4.0).abs() < 1e-12);
    }

    #[test]
    fn loop_without_pairs_is_explicit_zero() {
        let lj = PairLjCut::new(1.0, 1.0, 2.5, 12, 6, true, false).unwrap();
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.1, 0.0, 0.0];
        let (e, f) = lj.energy_forces_at(&coords);
        assert_eq!(e, 0.0);
        assert!(f.iter().all(|&x| x == 0.0));
    }

    /// One `lj/cut` pair over atoms 0-1, its row the pair's.
    fn compiled(eps: F, sigma: F) -> crate::ff::potential::Potentials {
        ExplicitTerms::new("pair", "lj/cut")
            .term(
                &[0, 1],
                Params::from_iter([("epsilon", eps), ("sigma", sigma)]),
            )
            .compile()
            .unwrap()
    }

    #[test]
    fn compiled_newton_third_law() {
        let pot = compiled(0.5, 1.0);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.5, 0.3, 0.1];
        let (e, forces) = pot.energy_forces_at(&coords);
        assert!(e.is_finite());
        for dim in 0..3 {
            assert!((forces[dim] + forces[3 + dim]).abs() < 1e-12);
        }
    }

    #[test]
    fn compiled_ignores_loop_pairs() {
        let pot = compiled(1.0, 1.0);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 2.0, 0.0, 0.0];
        let (e0, _) = pot.energy_forces_at(&coords);
        let extra = Neighbors::from_pairs(
            [molrs::core::NeighborPair {
                i: 0,
                j: 1,
                dist_sq: 1.0,
                disp: [1.0, 0.0, 0.0],
            }],
            molrs::core::NeighborColumns::FULL,
            molrs::core::QueryMode::SelfQuery { n_points: 2 },
        );
        let (e1, _) = pot.over_pairs(&coords, &extra);
        assert_eq!(e0, e1);
    }

    /// A field of one `lj/cut` style, types `A` and `B`, stated as `rows`
    /// (`(label, endpoints, ε, σ)`), and a frame of atoms `types` with the
    /// `pairs` list `links`.
    fn field(
        style: Params,
        rows: &[(&str, &[&str], F, F)],
        types: &[&str],
        links: &[(usize, usize)],
    ) -> (ForceField, Frame) {
        let mut ff = ForceField::new("t");
        ff.set_special_bonds(SpecialBonds::EXCLUDE_12_13);
        let s = ff.def_style("pair", "lj/cut", style).unwrap();
        for (label, ends, eps, sigma) in rows {
            s.def_type(
                label,
                ends,
                Params::from_iter([("epsilon", *eps), ("sigma", *sigma)]),
            )
            .unwrap();
        }
        let mut atoms = Block::new();
        atoms
            .insert(
                "type",
                Array1::from_vec(types.iter().map(|t| (*t).to_owned()).collect::<Vec<_>>())
                    .into_dyn(),
                None,
            )
            .unwrap();
        let mut pairs = Block::new();
        let (ai, aj): (Vec<Idx>, Vec<Idx>) =
            links.iter().map(|&(i, j)| (i as Idx, j as Idx)).unzip();
        pairs
            .insert("atomi", Array1::from_vec(ai).into_dyn(), None)
            .unwrap();
        pairs
            .insert("atomj", Array1::from_vec(aj).into_dyn(), None)
            .unwrap();
        let mut frame = Frame::new();
        frame.insert("atoms", atoms);
        frame.insert("pairs", pairs);
        (ff, frame)
    }

    /// `(compile, compile_typed)` of `ff` over `frame` at `coords`, the typed
    /// door over a table of `links`.
    fn both_doors(
        ff: &ForceField,
        frame: &Frame,
        coords: &[F],
        links: &[(usize, usize)],
    ) -> ((F, Vec<F>), (F, Vec<F>)) {
        let compiler = PotentialCompiler::new(ff);
        let compiled = compiler.compile(frame).unwrap().energy_forces_at(coords);
        let typed = compiler.compile_typed(frame).unwrap();
        let ForceTerm::Pair(kernel) = &typed[0].0 else {
            panic!("a pair member");
        };
        (
            compiled,
            kernel.over_pairs(coords, &table_over(coords, links)),
        )
    }

    /// The compiled door prices the style's Mie exponents, as the typed door
    /// and the spec's expression do.
    #[test]
    fn the_compiled_door_takes_the_style_exponents() {
        let coords = [0.0, 0.0, 0.0, 3.7, 0.0, 0.0];
        for (n, m) in [(12.0, 6.0), (9.0, 6.0), (10.0, 4.0)] {
            let (ff, frame) = field(
                Params::from_iter([("cutoff", 100.0), ("n", n), ("m", m)]),
                &[("A", &["A"], 0.2, 3.0)],
                &["A", "A"],
                &[(0, 1)],
            );
            let c = (n / (n - m)) * (n / m).powf(m / (n - m));
            let want = c * 0.2 * ((3.0_f64 / 3.7).powf(n) - (3.0_f64 / 3.7).powf(m));
            let ((e_c, _), (e_t, _)) = both_doors(&ff, &frame, &coords, &[(0, 1)]);
            assert!((e_c - want).abs() < 1e-12, "{n}-{m}: {e_c} vs {want}");
            assert!((e_t - want).abs() < 1e-12, "{n}-{m} typed: {e_t} vs {want}");
        }
    }

    /// Exponents that are no integers, or not `n > m > 0`, are refused by
    /// name at both doors.
    #[test]
    fn exponents_outside_their_domain_are_refused() {
        for (n, m, param) in [(12.5, 6.0, "n"), (12.0, 6.5, "m"), (6.0, 12.0, "n")] {
            let (ff, frame) = field(
                Params::from_iter([("cutoff", 10.0), ("n", n), ("m", m)]),
                &[("A", &["A"], 0.2, 3.0)],
                &["A", "A"],
                &[(0, 1)],
            );
            let compiler = PotentialCompiler::new(&ff);
            for err in [
                compiler.compile(&frame).map(|_| ()).unwrap_err(),
                compiler.compile_typed(&frame).map(|_| ()).unwrap_err(),
            ] {
                assert!(
                    matches!(err.ir(), Some(crate::ff::ir::IrError::BadValue { param: p, .. }) if p == param),
                    "{n}-{m}: {err}"
                );
            }
        }
    }

    /// The uniform kernel stops at its cutoff (`r < cutoff`, as LAMMPS):
    /// every pair inside it interacts and nothing outside it does.
    #[test]
    fn a_typed_kernel_stops_at_its_cutoff() {
        let typed = PairLjCut::new(1.0, 1.0, 2.5, 12, 6, false, false).unwrap();
        assert!(typed.pair_energy_force(4.0, [2.0, 0.0, 0.0]).is_some());
        assert!(
            typed.pair_energy_force(9.0, [3.0, 0.0, 0.0]).is_none(),
            "3 Å is past the 2.5 Å cutoff"
        );
        assert!(
            typed.pair_energy_force(6.25, [2.5, 0.0, 0.0]).is_none(),
            "LAMMPS prices `rsq < cutsq`: nothing at the cutoff itself"
        );
    }

    /// The compiled door truncated and shifted prices a pair as the typed
    /// one does, bit for bit, on each side of the cutoff.
    #[test]
    fn a_truncated_compiled_kernel_is_the_typed_one() {
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, 1.1, 0.2, 0.0, 2.9, -0.4, 0.3];
        let links = [(0_usize, 1_usize), (0, 2), (1, 2)];
        for shift in [0.0, 1.0] {
            let (ff, frame) = field(
                Params::from_iter([("cutoff", 2.5), ("shift", shift)]),
                &[("A", &["A"], 1.0, 1.0)],
                &["A", "A", "A"],
                &links,
            );
            let (compiled, typed) = both_doors(&ff, &frame, &coords, &links);
            assert_same(&format!("shift {shift}"), compiled, typed);
        }
    }

    #[test]
    fn unshifted_energy_at_sigma_is_zero() {
        let lj = PairLjCut::new(1.5, 2.0, 5.0, 12, 6, false, false).unwrap();
        let (e, f) = lj.pair_energy_force(4.0, [2.0, 0.0, 0.0]).unwrap();
        assert!(e.abs() < 1e-12);
        assert!(f[0] > 0.0);
    }

    /// Shifted, the energy is zero at the cutoff; smeared, the force is too,
    /// and below it the energy is `E(r) − E(r_c) − (r − r_c) E′(r_c)`.
    #[test]
    fn a_smeared_kernel_shifts_energy_and_force_to_zero_at_the_cutoff() {
        let (eps, sigma, rc) = (0.7, 1.1, 2.5);
        let lj = |r: F| 4.0 * eps * ((sigma / r).powi(12) - (sigma / r).powi(6));
        let dlj = |r: F| -4.0 * eps * (12.0 * (sigma / r).powi(12) - 6.0 * (sigma / r).powi(6)) / r;
        let shifted = PairLjCut::new(eps, sigma, rc, 12, 6, true, false).unwrap();
        let smeared = PairLjCut::new(eps, sigma, rc, 12, 6, false, true).unwrap();
        assert!(smeared.shifted() && smeared.smeared());
        let near = rc - 1e-9;
        let at = |k: &PairLjCut| k.pair_energy_force(near * near, [near, 0.0, 0.0]).unwrap();
        assert!(at(&shifted).0.abs() < 1e-9);
        assert!(at(&smeared).0.abs() < 1e-9 && at(&smeared).1[0].abs() < 1e-9);
        for r in [1.0, 1.3, 2.0] {
            let want = lj(r) - lj(rc) - (r - rc) * dlj(rc);
            let (e, f) = smeared.pair_energy_force(r * r, [r, 0.0, 0.0]).unwrap();
            assert!((e - want).abs() < 1e-12, "r = {r}: {e} vs {want}");
            let want_f = -(dlj(r) - dlj(rc));
            assert!(
                (f[0] - want_f).abs() < 1e-10,
                "r = {r}: {} vs {want_f}",
                f[0]
            );
        }
    }

    // -- explicit cross rows (NBFIX) -------------------------------------------

    const R_AB: F = 2.5;

    fn lj(eps: F, sigma: F, r: F) -> F {
        let s6 = (sigma / r).powi(6);
        4.0 * eps * (s6 * s6 - s6)
    }

    /// Self rows A (0.1, 3.0) and B (0.4, 3.6), plus `cross` rows, mixed
    /// geometrically; the energy of the A–B pair at `R_AB` at both doors.
    fn ab_energies(cross: &[(&str, &[&str], F, F)]) -> (F, F) {
        let mut style = Params::from_iter([("cutoff", 10.0)]);
        style.set("mixing", "geometric");
        let mut rows: Vec<(&str, &[&str], F, F)> =
            vec![("A", &["A"], 0.1, 3.0), ("B", &["B"], 0.4, 3.6)];
        rows.extend_from_slice(cross);
        let (ff, frame) = field(style, &rows, &["A", "B"], &[(0, 1)]);
        let coords: Vec<F> = vec![0.0, 0.0, 0.0, R_AB, 0.0, 0.0];
        let ((e_c, _), (e_t, _)) = both_doors(&ff, &frame, &coords, &[(0, 1)]);
        (e_c, e_t)
    }

    /// A pair with an explicit cross row is priced with that row, not with the
    /// mixing rule.
    #[test]
    fn an_explicit_cross_row_overrides_the_mixing_rule() {
        let want = lj(0.9, 2.0, R_AB);
        let (c, t) = ab_energies(&[("nbfix", &["A", "B"], 0.9, 2.0)]);
        assert!((c - want).abs() < 1e-12 && (t - want).abs() < 1e-12);
    }

    /// A cross row defined as `B-A` answers for the pair `(A, B)` too: the
    /// force field keys it by its endpoints in byte order.
    #[test]
    fn a_cross_row_is_found_in_either_order() {
        let want = lj(0.9, 2.0, R_AB);
        let (c, t) = ab_energies(&[("nbfix", &["B", "A"], 0.9, 2.0)]);
        assert!((c - want).abs() < 1e-12 && (t - want).abs() < 1e-12);
    }

    /// Without a cross row the pair is mixed, geometrically here.
    #[test]
    fn without_a_cross_row_the_pair_is_mixed() {
        let want = lj((0.1_f64 * 0.4).sqrt(), (3.0_f64 * 3.6).sqrt(), R_AB);
        let (c, t) = ab_energies(&[]);
        assert!((c - want).abs() < 1e-12 && (t - want).abs() < 1e-12);
    }
}
