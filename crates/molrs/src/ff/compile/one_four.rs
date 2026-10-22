//! What a force field's 1-4 pairs are, made explicit on a frame.
//!
//! The IR prices a 1-4 pair by one of three mechanisms (the manual,
//! "Force-field IR", 1-4 interactions): `special_bonds` (the pair styles at
//! their own parameters, times the weight), `dihedral charmm` `w`, and
//! per-pair override cells on the frame's `pairs` block. LAMMPS's
//! `lj/charmm` prices a `special_bonds` 1-4 pair at the regular `epsilon` /
//! `sigma`; its `epsilon14` / `sigma14` reach only `w` pairs.
//!
//! Engines that price every bond-graph 1-4 pair once with its own 1-4
//! parameters — OpenMM's `<LennardJonesForce>` (`sigma14`/`epsilon14`),
//! GROMACS `[ pairtypes ]`, CHARMM-via-chamber prmtop — mean the 1-4 pairs at
//! `epsilon14` / `sigma14`. A `lj/charmm` style says so with its string param
//! `one_four`:
//!
//! | `one_four` | A `special_bonds` 1-4 pair is priced at |
//! |---|---|
//! | absent, `"regular"` | the regular `epsilon` / `sigma` (LAMMPS) |
//! | `"epsilon14"` | `epsilon14` / `sigma14` (OpenMM, GROMACS pairtypes) |
//!
//! LAMMPS's pair style has no form for `"epsilon14"`, so the IR holds those
//! pairs as per-pair override rows: [`ForceField::materialize_one_four`]
//! writes them onto a frame, and compiling a `"epsilon14"` field for a frame
//! whose 1-4 pairs lack them is refused (the regular parameters would be a
//! silent loss). A pair a `dihedral charmm` `w > 0` covers is priced by `w`,
//! at `epsilon14` / `sigma14`, whatever `one_four` says.

use std::collections::HashMap;

use ndarray::Array1;

use crate::ff::compile::exceptions::dihedral_weights;
use crate::ff::compile::gathered;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::{CombiningRule, OneFour, Params};
use crate::ff::potential::intramolecular_pairs;
use crate::ff::potential::pair::charmm::{charmm_mixing, charmm_pair_params};
use crate::ff::potential::pair::lj_cut::{lj_pair_params, mixing_of};
use crate::ff::potential::param_reads;
use molrs::core::Frame;
use molrs::core::schema::block_names::{ATOMS, PAIRS};
use molrs::op::F;

/// The van-der-Waals style a 1-4 pair's parameters come from.
enum Vdw {
    None,
    LjCut(HashMap<String, Params>, CombiningRule),
    Charmm(HashMap<String, Params>, CombiningRule, OneFour),
}

fn vdw_style(ff: &ForceField) -> Result<Vdw, String> {
    let mut found = Vdw::None;
    for style in ff.get_styles("pair") {
        let next = match style.name() {
            "lj/charmm" => {
                let (p, rows) = gathered(style).map_err(|e| e.to_string())?;
                Vdw::Charmm(
                    rows.into_iter().collect(),
                    charmm_mixing(&p).map_err(|e| e.to_string())?,
                    OneFour::of(&p)?,
                )
            }
            "lj/cut" => {
                let (p, rows) = gathered(style).map_err(|e| e.to_string())?;
                let num =
                    |k: &str| param_reads::style_num("lj/cut", &p, k).map_err(|e| e.to_string());
                if num("n")? != 12.0 || num("m")? != 6.0 {
                    return Err(
                        "materialize_one_four: pair lj/cut with n/m ≠ 12/6 has no 1-4 row form"
                            .into(),
                    );
                }
                let mixing = mixing_of("lj/cut", &p).map_err(|e| e.to_string())?;
                Vdw::LjCut(rows.into_iter().collect(), mixing)
            }
            _ => continue,
        };
        if !matches!(found, Vdw::None) {
            return Err(
                "materialize_one_four: the force field has two Lennard-Jones styles".into(),
            );
        }
        found = next;
    }
    Ok(found)
}

impl ForceField {
    /// Write this force field's 1-4 pricing of `frame`'s 1-4 pairs as the
    /// IR's per-pair override cells on its `pairs` block, and return how many
    /// rows it filled.
    ///
    /// Every `pairs` row with `is_14` (each 1-4 pair once; the block is built
    /// with [`intramolecular_pairs`] when the frame has none) not covered by a
    /// `dihedral charmm` `w > 0` gets, in each of its null cells:
    ///
    /// - `epsilon`, `sigma`: the pair's 1-4 Lennard-Jones parameters — under
    ///   `lj/charmm` with `one_four = "epsilon14"`, its explicit cross row
    ///   (NBFIX; its `epsilon14` / `sigma14`, absent → its `epsilon` /
    ///   `sigma`) or the two types' `epsilon14` / `sigma14` mixed by the
    ///   style's rule; under any other Lennard-Jones style, the regular pair
    ///   parameters (the energy does not change);
    /// - `lj_scale`, `coul_scale`: the `special_bonds` 1-4 weights.
    ///
    /// A cell already holding a value is kept (an explicit override is final).
    /// Charges are not written: `charge_product` stays null, the atoms'
    /// product. The frame then prices under every compile door exactly as the
    /// field means; LAMMPS's writers refuse the override columns (LAMMPS has no
    /// per-pair 1-4 form).
    ///
    /// # Errors
    ///
    /// A frame without a string `atoms.type` column, a type without a pair
    /// row, a `pairs` block without `atomi` / `atomj` / `is_14`, two
    /// Lennard-Jones styles, an invalid `one_four`, and the errors of
    /// [`intramolecular_pairs`].
    pub fn materialize_one_four(&self, frame: &mut Frame) -> Result<usize, String> {
        if frame.get(PAIRS).is_none() {
            let pairs = intramolecular_pairs(frame, self.special_bonds(), None)?;
            frame.insert(PAIRS, pairs);
        }
        let vdw = vdw_style(self)?;
        let covered = dihedral_weights(self, frame)?;
        let sb = self.special_bonds();
        let (lj14, coul14) = (sb.lj_14(), sb.coul_14());

        let block = frame.get(PAIRS).expect("inserted above");
        let n = block.n_rows().unwrap_or(0);
        if n == 0 {
            return Ok(0);
        }
        let index = |key: &str| -> Result<Vec<usize>, String> {
            Ok(block
                .get(key)
                .and_then(|c| c.get::<u64>())
                .ok_or_else(|| format!("materialize_one_four: pairs block missing \"{key}\""))?
                .iter()
                .map(|&v| v as usize)
                .collect())
        };
        let (ai, aj) = (index("atomi")?, index("atomj")?);
        let is_14: Vec<bool> = block
            .get("is_14")
            .and_then(|c| c.get::<bool>())
            .ok_or("materialize_one_four: pairs block missing a bool \"is_14\" column")?
            .iter()
            .copied()
            .collect();
        // The present cells, which are kept.
        let present = |key: &str| -> Result<(Vec<F>, Vec<bool>), String> {
            match block.get(key) {
                None => Ok((vec![0.0; n], vec![false; n])),
                Some(col) => {
                    let values = col.get::<f64>().ok_or_else(|| {
                        format!("pairs: the per-pair override column '{key}' must be float")
                    })?;
                    let mask = block
                        .validity(key)
                        .map(<[bool]>::to_vec)
                        .unwrap_or_else(|| vec![true; n]);
                    Ok((values.iter().copied().collect(), mask))
                }
            }
        };
        let mut cols: Vec<(&str, Vec<F>, Vec<bool>)> = Vec::new();
        for key in ["epsilon", "sigma", "lj_scale", "coul_scale"] {
            let (v, m) = present(key)?;
            cols.push((key, v, m));
        }
        let types = frame
            .get(ATOMS)
            .and_then(|b| b.get("type"))
            .and_then(|c| c.get::<String>());

        let mut filled = 0;
        for r in 0..n {
            if !is_14[r] {
                continue;
            }
            let key = (ai[r].min(aj[r]), ai[r].max(aj[r]));
            if covered.get(&key).is_some_and(|&w| w > 0.0) {
                continue;
            }
            let lj = match &vdw {
                Vdw::None => None,
                Vdw::LjCut(rows, mixing) | Vdw::Charmm(rows, mixing, _) => {
                    let t = types.ok_or(
                        "materialize_one_four: atoms block missing a string \"type\" column",
                    )?;
                    let (a, b) = (t[ai[r]].as_str(), t[aj[r]].as_str());
                    let refs: HashMap<&str, &Params> =
                        rows.iter().map(|(k, v)| (k.as_str(), v)).collect();
                    Some(match &vdw {
                        Vdw::Charmm(_, _, mode) => {
                            let (regular, one_four) = charmm_pair_params(&refs, *mixing, a, b)
                                .map_err(|e| e.to_string())?;
                            match mode {
                                OneFour::Epsilon14 => one_four,
                                OneFour::Regular => regular,
                            }
                        }
                        _ => lj_pair_params("lj/cut", &refs, *mixing, a, b)
                            .map_err(|e| e.to_string())?,
                    })
                }
            };
            let values = [lj.map(|v| v.0), lj.map(|v| v.1), Some(lj14), Some(coul14)];
            let mut any = false;
            for ((_, v, m), value) in cols.iter_mut().zip(values) {
                if let Some(value) = value
                    && !m[r]
                {
                    v[r] = value;
                    m[r] = true;
                    any = true;
                }
            }
            filled += usize::from(any);
        }
        let block = frame.get_mut(PAIRS).expect("inserted above");
        for (key, values, mask) in cols {
            if !mask.iter().any(|&m| m) {
                continue;
            }
            block
                .insert(key, Array1::from_vec(values).into_dyn(), Some(mask))
                .map_err(|e| e.to_string())?;
        }
        Ok(filled)
    }
}

/// Refuse compiling a `one_four = "epsilon14"` field for a frame with a 1-4
/// pair whose 1-4 parameters differ from its regular ones and which neither
/// an override (`epsilon` and `sigma` cells) nor a `w > 0` dihedral covers:
/// the pair styles would price it at the regular parameters.
///
/// `pairs_14` are the frame's 1-4 pairs `(lo, hi)`, `covered` those an
/// override or `w` prices.
pub(crate) fn check_materialized(
    ff: &ForceField,
    frame: &Frame,
    pairs_14: &[(usize, usize)],
    covered: &dyn Fn(usize, usize) -> bool,
) -> Result<(), String> {
    if ff.special_bonds().lj_14() == 0.0 {
        return Ok(());
    }
    let Some(style) = ff.get_style("pair", "lj/charmm") else {
        return Ok(());
    };
    let (p, rows) = gathered(style).map_err(|e| e.to_string())?;
    if OneFour::of(&p)? != OneFour::Epsilon14 {
        return Ok(());
    }
    let refs: HashMap<&str, &Params> = rows.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let mixing = charmm_mixing(&p).map_err(|e| e.to_string())?;
    let Some(types) = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>())
    else {
        return Ok(());
    };
    for &(i, j) in pairs_14 {
        if covered(i, j) {
            continue;
        }
        let (a, b) = (types[i].as_str(), types[j].as_str());
        let (regular, one_four) =
            charmm_pair_params(&refs, mixing, a, b).map_err(|e| e.to_string())?;
        if regular != one_four {
            return Err(format!(
                "pair lj/charmm declares one_four = \"epsilon14\": the 1-4 pair of atoms {i} \
                 ({a}) and {j} ({b}) is meant at epsilon14/sigma14, which the pair style \
                 prices at the regular epsilon/sigma. Write the frame's 1-4 pairs first: \
                 ForceField::materialize_one_four (Python: ForceField.materialize_one_four)"
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::ir::{ONE_FOUR, ONE_FOUR_EPSILON14, ONE_FOUR_REGULAR, SpecialBonds};
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::op::Idx;

    fn lj_charmm(one_four: Option<&str>) -> ForceField {
        let mut ff = ForceField::new("x");
        let mut sp = Params::from_iter([("inner", 8.0), ("cutoff", 10.0)]);
        if let Some(v) = one_four {
            sp.set(ONE_FOUR, v);
        }
        ff.def_style("pair", "lj/charmm", sp)
            .unwrap()
            .def_type(
                "A",
                &["A"],
                Params::from_iter([
                    ("epsilon", 0.1),
                    ("sigma", 3.0),
                    ("epsilon14", 0.04),
                    ("sigma14", 2.0),
                ]),
            )
            .unwrap()
            .def_type(
                "B",
                &["B"],
                Params::from_iter([("epsilon", 0.4), ("sigma", 4.0)]),
            )
            .unwrap();
        ff.set_special_bonds(SpecialBonds {
            lj: [0.0, 0.0, 0.5],
            coul: [0.0, 0.0, 0.75],
        });
        ff
    }

    /// A-B-B-A chain: (0, 3) is the one 1-4 pair.
    fn chain() -> Frame {
        let mut atoms = Block::new();
        for (k, key) in ["x", "y", "z"].iter().enumerate() {
            let v: Vec<F> = (0..4)
                .map(|i| {
                    if k == 0 {
                        1.5 * i as F
                    } else {
                        0.1 * (i * k) as F
                    }
                })
                .collect();
            atoms
                .insert(*key, Array1::from_vec(v).into_dyn(), None)
                .unwrap();
        }
        let types: Vec<String> = ["A", "B", "B", "A"].map(String::from).to_vec();
        atoms
            .insert("type", Array1::from_vec(types).into_dyn(), None)
            .unwrap();
        atoms
            .insert(
                "charge",
                Array1::from_vec(vec![0.3 as F, -0.3, 0.3, -0.3]).into_dyn(),
                None,
            )
            .unwrap();
        let mut frame = Frame::new();
        frame.insert(ATOMS, atoms);
        let rel = |cols: &[&str], rows: &[&[Idx]]| {
            let mut b = Block::new();
            for (k, c) in cols.iter().enumerate() {
                let v: Vec<Idx> = rows.iter().map(|r| r[k]).collect();
                b.insert(*c, Array1::from_vec(v).into_dyn(), None).unwrap();
            }
            b
        };
        frame.insert(
            "bonds",
            rel(&["atomi", "atomj"], &[&[0, 1], &[1, 2], &[2, 3]]),
        );
        frame.insert(
            "angles",
            rel(&["atomi", "atomj", "atomk"], &[&[0, 1, 2], &[1, 2, 3]]),
        );
        frame.insert(
            "dihedrals",
            rel(&["atomi", "atomj", "atomk", "atoml"], &[&[0, 1, 2, 3]]),
        );
        frame
    }

    fn cells(frame: &Frame, key: &str) -> Vec<Option<F>> {
        let p = frame.get(PAIRS).unwrap();
        let v = p.get(key).unwrap().get::<f64>().unwrap();
        let mask = p.validity(key);
        (0..v.len())
            .map(|r| mask.is_none_or(|m| m[r]).then_some(v[[r]]))
            .collect()
    }

    #[test]
    fn one_four_reads_its_two_values_and_refuses_others() {
        assert_eq!(OneFour::of(&Params::new()), Ok(OneFour::Regular));
        let mut p = Params::new();
        p.set(ONE_FOUR, "regular");
        assert_eq!(OneFour::of(&p), Ok(OneFour::Regular));
        p.set(ONE_FOUR, "epsilon14");
        assert_eq!(OneFour::of(&p), Ok(OneFour::Epsilon14));
        p.set(ONE_FOUR, "sigma14");
        assert!(OneFour::of(&p).unwrap_err().contains("sigma14"));
        let p = Params::from_iter([(ONE_FOUR, 1.0)]);
        assert!(OneFour::of(&p).is_err());
    }

    /// `"epsilon14"`: the 1-4 pair (0, 3) of two A atoms gets A's ε₁₄/σ₁₄ and
    /// the 1-4 weights; the other pairs stay null. The block is built when
    /// the frame has none.
    #[test]
    fn epsilon14_materializes_the_one_four_parameters() {
        let ff = lj_charmm(Some(ONE_FOUR_EPSILON14));
        let mut frame = chain();
        assert_eq!(ff.materialize_one_four(&mut frame).unwrap(), 1);
        let p = frame.get(PAIRS).unwrap();
        let (ai, aj) = (
            p.get("atomi").unwrap().get::<u64>().unwrap(),
            p.get("atomj").unwrap().get::<u64>().unwrap(),
        );
        let r = (0..ai.len()).find(|&r| (ai[r], aj[r]) == (0, 3)).unwrap();
        for (key, want) in [
            ("epsilon", 0.04),
            ("sigma", 2.0),
            ("lj_scale", 0.5),
            ("coul_scale", 0.75),
        ] {
            let c = cells(&frame, key);
            assert_eq!(c[r], Some(want), "{key}");
            assert_eq!(c.iter().filter(|v| v.is_some()).count(), 1, "{key}");
        }
    }

    /// `"regular"` materializes the regular parameters: the energy is the
    /// pair style's, with or without the rows.
    #[test]
    fn regular_materializes_the_regular_parameters_and_keeps_the_energy() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::potential::intramolecular_pairs;
        let ff = lj_charmm(None);
        let mut frame = chain();
        frame.insert(
            PAIRS,
            intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap(),
        );
        let x: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let before = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_at(&x);
        assert_eq!(ff.materialize_one_four(&mut frame).unwrap(), 1);
        assert!(cells(&frame, "epsilon").contains(&Some(0.1)));
        let after = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_at(&x);
        assert!(
            (after - before).abs() <= 1e-12 * before.abs(),
            "{after} vs {before}"
        );
    }

    /// Compiling `"epsilon14"` without the rows is refused, naming the
    /// operation; with them it prices A's ε₁₄/σ₁₄ at the 1-4 weight.
    #[test]
    fn epsilon14_without_rows_is_refused_and_with_them_priced() {
        use crate::ff::compile::PotentialCompiler;
        use crate::ff::potential::intramolecular_pairs;
        let ff = lj_charmm(Some(ONE_FOUR_EPSILON14));
        let mut frame = chain();
        frame.insert(
            PAIRS,
            intramolecular_pairs(&frame, ff.special_bonds(), None).unwrap(),
        );
        let err = PotentialCompiler::new(&ff).compile(&frame).err().unwrap();
        assert!(err.to_string().contains("materialize_one_four"), "{err}");
        let x: Vec<F> = frame.coords().unwrap().into_iter().collect();
        ff.materialize_one_four(&mut frame).unwrap();
        let with = PotentialCompiler::new(&ff)
            .compile(&frame)
            .unwrap()
            .energy_at(&x);
        let mut regular = ff.clone();
        regular
            .get_style_mut("pair", "lj/charmm")
            .unwrap()
            .set_param(ONE_FOUR, ONE_FOUR_REGULAR);
        let mut plain = chain();
        plain.insert(
            PAIRS,
            intramolecular_pairs(&plain, ff.special_bonds(), None).unwrap(),
        );
        let without = PotentialCompiler::new(&regular)
            .compile(&plain)
            .unwrap()
            .energy_at(&x);
        // The difference is the 1-4 LJ at ε₁₄/σ₁₄ minus at ε/σ, times ½.
        let lj = |e: F, s: F, r: F| 4.0 * e * ((s / r).powi(12) - (s / r).powi(6));
        let d: Vec<F> = (0..3).map(|k| x[9 + k] - x[k]).collect();
        let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
        let want = 0.5 * (lj(0.04, 2.0, r) - lj(0.1, 3.0, r));
        assert!(
            ((with - without) - want).abs() <= 1e-12 * want.abs().max(1e-12),
            "{} vs {want}",
            with - without
        );
    }
}
