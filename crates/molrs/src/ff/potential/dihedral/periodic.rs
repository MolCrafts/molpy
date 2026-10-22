//! Periodic / Fourier proper dihedral (AMBER / GAFF).

use crate::ff::ir::IrError;
use crate::ff::potential::param_reads;
use molrs::core::schema::block_names::DIHEDRALS;
use std::collections::HashMap;

use ndarray::{Array2, ArrayView2};

use crate::ff::ir::Params;
use crate::ff::potential::flat_coords::{
    accumulate_dihedral_forces, compute_dihedral, term_table, validate_coords,
};
use crate::ff::potential::{Accumulator, ForceTerm, IndexedTerms, Potential};
use molrs::core::Frame;
use molrs::op::F;

/// One cosine term `k·[1 + cos(n·φ − γ)]` with the phase `γ` in radians.
#[derive(Clone, Copy)]
struct Term {
    k: F,
    n: F,
    d: F,
}

/// Periodic / Fourier proper dihedral with pre-resolved flat arrays.
///
/// Periodic / Fourier proper dihedral (AMBER / GAFF):
///
/// E(φ) = Σ_m k_m · [1 + cos(n_m·φ − γ_m)]
///
/// AMBER-family torsions are a sum of cosine terms per quadruple. The parameter
/// encoding is **per-term indexed keys** `k{m}`, `periodicity{m}`, `phase{m}`
/// (1-indexed, the phase in **degrees**, as LAMMPS `dihedral_style fourier`
/// writes it — the kernel converts it to radians once),
/// scanned upward from `m = 1` until a term is absent. A single unindexed
/// `k`/`periodicity`/`phase` triple is accepted as the one-term case (the common
/// GAFF default), keeping the form identical to one CHARMM term. This is the
/// canonical encoding the molpy → molrs ForceField bridge emits.
pub struct DihedralPeriodic {
    atom_i: Vec<usize>,
    atom_j: Vec<usize>,
    atom_k: Vec<usize>,
    atom_l: Vec<usize>,
    /// Cosine terms per dihedral instance.
    terms: Vec<Vec<Term>>,
}

impl DihedralPeriodic {
    /// The physics, once. Which atoms a term names is the only thing
    /// that differs between the two entry points, so it is the only thing
    /// passed in — a second copy of the loop would be a second place for
    /// the force expression to drift.
    fn fold(
        &self,
        coords: &[F],
        out: &mut [F],
        n_terms: usize,
        atoms: impl Fn(usize) -> (usize, usize, usize, usize),
    ) -> F {
        let _n = validate_coords(coords);
        let mut energy: F = 0.0;
        let forces = out;

        for idx in 0..n_terms {
            let (i, j, k, l) = atoms(idx);
            let phi = compute_dihedral(coords, i, j, k, l);
            let mut de_dphi: F = 0.0;
            for t in &self.terms[idx] {
                let arg = t.n * phi - t.d;
                energy += t.k * (1.0 + arg.cos());
                de_dphi += -t.k * t.n * arg.sin();
            }
            accumulate_dihedral_forces(coords, i, j, k, l, de_dphi, forces);
        }
        energy
    }
}

impl Potential for DihedralPeriodic {
    fn evaluate(&self, coords: &[F], acc: &mut Accumulator<'_>) {
        let energy = self.fold(
            coords,
            acc.forces_without_virial(),
            self.atom_i.len(),
            |t| {
                (
                    self.atom_i[t],
                    self.atom_j[t],
                    self.atom_k[t],
                    self.atom_l[t],
                )
            },
        );
        acc.add_energy(energy);
    }
}

impl IndexedTerms for DihedralPeriodic {
    fn terms(&self) -> Array2<u32> {
        term_table(&[&self.atom_i, &self.atom_j, &self.atom_k, &self.atom_l])
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
                terms[[t, 3]] as usize,
            )
        });
        acc.add_energy(energy);
    }
}

/// Collect the cosine terms from a per-type [`Params`] using the indexed
/// `k{m}`/`periodicity{m}`/`phase{m}` encoding (contiguous from 1), or the
/// single-term `k`/`periodicity`/`phase` spelling.
fn collect_terms(p: &Params, label: &str) -> Result<Vec<Term>, IrError> {
    let num = |key: &str| param_reads::type_num("periodic", label, p, key);
    let term = |m: &str| -> Result<Term, IrError> {
        Ok(Term {
            k: num(&format!("k{m}"))?,
            n: num(&format!("periodicity{m}"))?,
            d: num(&format!("phase{m}"))?.to_radians(), // degrees → radians
        })
    };
    let m = (1..)
        .take_while(|m| p.get(&format!("k{m}")).and_then(|v| v.as_num()).is_some())
        .count();
    let beyond = p
        .iter()
        .filter_map(|(k, v)| v.as_num().map(|x| (k, x)))
        .any(|(key, _)| {
            key.strip_prefix('k')
                .and_then(|i| i.parse::<usize>().ok())
                .is_some_and(|i| i > m)
        });
    if beyond {
        return Err(param_reads::missing(
            "periodic",
            label,
            &format!("k{}", m + 1),
        ));
    }
    match (m, p.get("k").and_then(|v| v.as_num()).is_some()) {
        (0, true) => Ok(vec![term("")?]),
        (0, false) => Err(param_reads::missing("periodic", label, "k1")),
        (_, true) => Err(param_reads::bad(
            "periodic",
            label,
            "k",
            "is given beside `k1`: spell one term `k`, or every term `k<m>`",
        )),
        (m, false) => (1..=m).map(|i| term(&i.to_string())).collect(),
    }
}

/// Construct a [`DihedralPeriodic`] from per-type params and a Frame's
/// `"dihedrals"` block (`atomi/atomj/atomk/atoml/type`).
pub fn dihedral_periodic_constructor(
    _sp: &Params,
    tp: &[(&str, &Params)],
    frame: &Frame,
) -> Result<ForceTerm, crate::ff::potential::CompileError> {
    let type_map: HashMap<&str, &Params> = tp.iter().copied().collect();
    let block = frame
        .get(DIHEDRALS)
        .ok_or("dihedral_periodic: missing \"dihedrals\" block")?;
    let ic = block
        .get("atomi")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomi")?;
    let jc = block
        .get("atomj")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomj")?;
    let kc = block
        .get("atomk")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atomk")?;
    let lc = block
        .get("atoml")
        .and_then(|c| c.get::<u64>())
        .ok_or("missing atoml")?;
    let tc = block
        .get("type")
        .and_then(|c| c.get::<String>())
        .ok_or("missing type")?;

    let n = ic.len();
    let (mut ai, mut aj, mut ak, mut al, mut terms) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );

    for idx in 0..n {
        let p = type_map
            .get(tc[idx].as_str())
            .ok_or_else(|| format!("dihedral_periodic: unknown type '{}'", tc[idx]))?;
        ai.push(ic[idx] as usize);
        aj.push(jc[idx] as usize);
        ak.push(kc[idx] as usize);
        al.push(lc[idx] as usize);
        terms.push(collect_terms(p, tc[idx].as_str())?);
    }
    Ok(ForceTerm::indexed(DihedralPeriodic {
        atom_i: ai,
        atom_j: aj,
        atom_k: ak,
        atom_l: al,
        terms,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::potential::fixtures::OneShot;

    fn quad(phi: F) -> Vec<F> {
        let (s, c) = phi.sin_cos();
        vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, c, s]
    }

    fn single(terms: Vec<Term>) -> DihedralPeriodic {
        DihedralPeriodic {
            atom_i: vec![0],
            atom_j: vec![1],
            atom_k: vec![2],
            atom_l: vec![3],
            terms: vec![terms],
        }
    }

    fn term(k: F, n: F, d_deg: F) -> Term {
        Term {
            k,
            n,
            d: d_deg.to_radians(),
        }
    }

    #[test]
    fn single_term_energy() {
        // K[1+cos(nφ−d)], n=1,d=0: E(0)=2K, E(π)=0.
        let p = single(vec![term(1.5, 1.0, 0.0)]);
        assert!((p.energy_forces_at(&quad(0.0)).0 - 3.0).abs() < 1e-9);
        assert!(p.energy_forces_at(&quad(std::f64::consts::PI)).0.abs() < 1e-9);
    }

    #[test]
    fn multi_term_energy_sums() {
        // Two terms add: at φ=0, E = K1[1+cos(−d1)] + K2[1+cos(−d2)].
        let t = vec![term(1.0, 1.0, 0.0), term(0.5, 2.0, 180.0)];
        let e = single(t).energy_forces_at(&quad(0.0)).0;
        // K1[1+1] + K2[1+cos(180°)] = 2.0 + 0.5*0 = 2.0
        assert!((e - 2.0).abs() < 1e-9, "got {e}");
    }

    #[test]
    fn collect_terms_indexed_and_single() {
        let mut p = Params::new();
        p.set("k1", 1.0);
        p.set("periodicity1", 1.0);
        p.set("phase1", 0.0);
        p.set("k2", 0.5);
        p.set("periodicity2", 2.0);
        p.set("phase2", 180.0);
        let t = collect_terms(&p, "x").unwrap();
        assert_eq!(t.len(), 2);

        let mut q = Params::new();
        q.set("k", 2.0);
        q.set("periodicity", 3.0);
        q.set("phase", 0.0);
        let t2 = collect_terms(&q, "y").unwrap();
        assert_eq!(t2.len(), 1);
        assert_eq!(t2[0].n, 3.0);
    }

    #[test]
    fn numerical_gradient_multiterm() {
        let pot = single(vec![
            term(1.3, 1.0, 0.0),
            term(-0.7, 2.0, 180.0),
            term(0.4, 3.0, 0.0),
        ]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, forces) = pot.energy_forces_at(&coords);
        let h = 1e-6;
        for d in 0..coords.len() {
            let mut cp = coords.clone();
            let mut cm = coords.clone();
            cp[d] += h;
            cm[d] -= h;
            let ep = pot.energy_forces_at(&cp).0;
            let em = pot.energy_forces_at(&cm).0;
            let fd = -(ep - em) / (2.0 * h);
            assert!(
                (forces[d] - fd).abs() < 1e-5,
                "comp {d}: analytic {} vs fd {fd}",
                forces[d]
            );
        }
    }

    #[test]
    fn newtons_third_law() {
        let pot = single(vec![term(1.0, 1.0, 0.0), term(0.5, 2.0, 90.0)]);
        let coords: Vec<F> = vec![0.1, 1.0, 0.2, 0.0, 0.0, 0.0, 1.0, 0.0, -0.1, 1.2, -0.8, 0.5];
        let (_, f) = pot.energy_forces_at(&coords);
        for dim in 0..3 {
            let s: F = (0..4).map(|a| f[a * 3 + dim]).sum();
            assert!(s.abs() < 1e-9, "dim {dim} force sum {s}");
        }
    }
}
