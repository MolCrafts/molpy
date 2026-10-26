//! The 1-4 exceptions of a force field on a frame: which non-bonded pairs
//! the [`PairExceptions`] kernel prices instead of the pair styles, and at
//! what parameters (the `dihedral charmm` `w`, a `pairs` row's parameter
//! cells, `lj/charmm`'s `epsilon14` / `sigma14`, the row's weights).
//!
//! LAMMPS's `lj/charmm` prices a `special_bonds` 1-4 pair at the regular
//! `epsilon` / `sigma`; its `epsilon14` / `sigma14` reach only `w` pairs.
//! Engines that price every bond-graph 1-4 pair once with its own 1-4
//! parameters — OpenMM's `<LennardJonesForce>` (`sigma14`/`epsilon14`),
//! GROMACS `[ pairtypes ]`, CHARMM-via-chamber prmtop — mean the 1-4 pairs at
//! `epsilon14` / `sigma14`, and a `lj/charmm` style says so with its string
//! param `one_four = "epsilon14"`; this kernel then prices those pairs.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ff::compile::gathered;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::{CombiningRule, OneFour, Params, SpecialBonds, SpecialClass};
use crate::ff::potential::end_pairs;
use crate::ff::potential::pair::PairExceptions;
use crate::ff::potential::pair::lj_rows::{charmm_pair_params, lj_pair_params, mixing_of};
use crate::ff::potential::param_reads;
use crate::ff::special_pairs::{COUL_SCALE, EPS, LJ_SCALE, QQ, SIGMA, SpecialPairs};
use molrs::core::Frame;
use molrs::core::keys::{ATOMI, ATOML};
use molrs::core::schema::block_names::{ATOMS, DIHEDRALS};
use molrs::op::F;

/// What the compiler routes through the exceptions kernel for one frame.
pub(crate) struct Exceptions {
    /// The kernel, when any pair is an exception.
    pub(crate) kernel: Option<PairExceptions>,
    /// The pairs `(lo, hi)` the kernel prices: the pair styles must not.
    pub(crate) replaced: HashSet<(usize, usize)>,
}

/// The van-der-Waals style a 1-4 pair's Lennard-Jones parameters come from.
pub(crate) enum Vdw<'f> {
    None,
    LjCut(HashMap<String, Params>, CombiningRule),
    Charmm(HashMap<String, Params>, CombiningRule),
    /// A van-der-Waals style with no LJ 12-6 form (`buck`, `morse`, Mie
    /// `lj/cut`, …).
    Other(&'f str),
}

/// One exception before it is priced.
struct Entry {
    /// The summed `dihedral charmm` `w` of the dihedrals ending at the pair.
    w: F,
    /// The pair's `pairs` row, when it has one.
    row: Option<usize>,
    /// Whether the pair takes `lj/charmm`'s `epsilon14` / `sigma14`.
    one_four: bool,
}

/// How the compiler prices the frame's exceptions under `ff`.
///
/// A pair is an exception when a `dihedral charmm` with `w > 0` ends at it,
/// when its `pairs` row states one of its parameters (`epsilon`, `sigma`,
/// `charge_product`), or, under `lj/charmm` with `one_four = "epsilon14"`,
/// when it is a 1-4 pair (`n_bonds` 3) its weights still price.
///
/// # Errors
///
/// A `dihedral charmm` `w` outside `[0, 1]`; `w > 0` beside `special_bonds`
/// 1-4 weights other than 0, or without a `lj/charmm` pair style and a
/// Coulomb style (all three LAMMPS's own errors); and, when any exception
/// exists, a pair style the exception cannot stand in for.
pub(crate) fn plan(
    ff: &ForceField,
    frame: &Frame,
    pairs: &SpecialPairs,
    special: &SpecialBonds,
) -> Result<Exceptions, String> {
    let wsum = dihedral_weights(ff, frame)?;
    let rows = pairs.rows();
    let row_of: HashMap<(usize, usize), usize> = rows
        .iter()
        .enumerate()
        .map(|(r, p)| ((p.i, p.j), r))
        .collect();
    let mut entries: BTreeMap<(usize, usize), Entry> = BTreeMap::new();
    for (&key, &w) in &wsum {
        entries.insert(
            key,
            Entry {
                w,
                row: row_of.get(&key).copied(),
                one_four: false,
            },
        );
    }
    for (r, p) in rows.iter().enumerate() {
        if p.has_parameters() {
            entries
                .entry((p.i, p.j))
                .or_insert(Entry {
                    w: 0.0,
                    row: None,
                    one_four: false,
                })
                .row = Some(r);
        }
    }
    if epsilon14(ff, special)? {
        for (r, p) in rows.iter().enumerate() {
            if p.n_bonds != 3 || p.weight(SpecialClass::Vdw, special) == 0.0 {
                continue;
            }
            let e = entries.entry((p.i, p.j)).or_insert(Entry {
                w: 0.0,
                row: Some(r),
                one_four: false,
            });
            e.one_four = e.w == 0.0;
        }
    }
    if entries.is_empty() {
        return Ok(Exceptions {
            kernel: None,
            replaced: HashSet::new(),
        });
    }

    let (vdw, coul) = pair_styles(ff)?;
    if let Vdw::Other(name) = vdw {
        return Err(format!(
            "pair style '{name}': a per-pair 1-4 exception is priced as Lennard-Jones 12-6 \
             plus Coulomb, and '{name}' has no such form to stand in for"
        ));
    }
    let types = frame
        .get(ATOMS)
        .and_then(|b| b.get("type"))
        .and_then(|c| c.get::<String>());
    let charges = frame
        .get(ATOMS)
        .and_then(|b| b.get("charge"))
        .and_then(|c| c.get::<f64>());
    let lj_rows = match &vdw {
        Vdw::Charmm(rows, ..) | Vdw::LjCut(rows, _) => as_refs(rows),
        _ => HashMap::new(),
    };

    let n = entries.len();
    let (mut ai, mut aj) = (Vec::with_capacity(n), Vec::with_capacity(n));
    let (mut es, mut lj_w, mut qq, mut coul_w) = (
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
        Vec::with_capacity(n),
    );
    for (&(i, j), entry) in &entries {
        let w = entry.w;
        let row = entry.row.map(|r| &rows[r]);
        let cell = row.map(|p| p.cells).unwrap_or([None; 5]);
        // The weight a pair the dihedral does not price takes: its row's.
        let weight = |class: SpecialClass| row.map_or(1.0, |p| p.weight(class, special));

        // Van der Waals: the cells, else the 1-4 parameters (a `w` pair, an
        // `epsilon14` 1-4 pair), else the style's own. A field with no
        // Lennard-Jones style prices none of the LJ cells: they belong to a
        // style it does not have.
        let base = |mixing: CombiningRule, charmm: bool| -> Result<(F, F), String> {
            let (a, b) = atom_type_pair(types, i, j)?;
            if charmm {
                let (regular, one_four) =
                    charmm_pair_params(&lj_rows, mixing, a, b).map_err(|e| e.to_string())?;
                Ok(if w > 0.0 || entry.one_four {
                    one_four
                } else {
                    regular
                })
            } else {
                lj_pair_params("lj/cut", &lj_rows, mixing, a, b).map_err(|e| e.to_string())
            }
        };
        let (eps, sigma, w_lj) = match &vdw {
            Vdw::Charmm(_, mixing) | Vdw::LjCut(_, mixing) => {
                let (e, s) = match (cell[EPS], cell[SIGMA]) {
                    (Some(e), Some(s)) => (e, s),
                    (e, s) => {
                        let (be, bs) = base(*mixing, matches!(vdw, Vdw::Charmm(..)))?;
                        (e.unwrap_or(be), s.unwrap_or(bs))
                    }
                };
                let w_lj = cell[LJ_SCALE].unwrap_or_else(|| {
                    if w > 0.0 {
                        w
                    } else {
                        weight(SpecialClass::Vdw)
                    }
                });
                (e, s, w_lj)
            }
            _ => (0.0, 0.0, 0.0),
        };

        // Coulomb, likewise only under a Coulomb style.
        let (k, w_c, qq_ij) = match coul {
            None => (0.0, 0.0, 0.0),
            Some(k) => {
                let w_c = cell[COUL_SCALE].unwrap_or_else(|| {
                    if w > 0.0 {
                        w
                    } else {
                        weight(SpecialClass::Coulomb)
                    }
                });
                let qq_ij = match cell[QQ] {
                    Some(v) => v,
                    None if w_c == 0.0 => 0.0,
                    None => {
                        let q = charges.ok_or_else(|| {
                            "1-4 exceptions: atoms block missing \"charge\" column".to_string()
                        })?;
                        q[i] * q[j]
                    }
                };
                (k, w_c, qq_ij)
            }
        };
        ai.push(i);
        aj.push(j);
        es.push((eps, sigma));
        lj_w.push(w_lj);
        qq.push(k * qq_ij);
        coul_w.push(w_c);
    }
    Ok(Exceptions {
        kernel: Some(PairExceptions::new(ai, aj, &es, lj_w, qq, coul_w)),
        replaced: entries.into_keys().collect(),
    })
}

/// Whether the field prices its 1-4 pairs at `lj/charmm`'s `epsilon14` /
/// `sigma14`: `one_four = "epsilon14"` with a non-zero 1-4 weight.
fn epsilon14(ff: &ForceField, special: &SpecialBonds) -> Result<bool, String> {
    let Some(style) = ff.get_style("pair", "lj/charmm") else {
        return Ok(false);
    };
    let (params, _) = gathered(ff, style).map_err(|e| e.to_string())?;
    Ok(OneFour::of(&params)? == OneFour::Epsilon14 && special.lj_14() != 0.0)
}

fn as_refs(rows: &HashMap<String, Params>) -> HashMap<&str, &Params> {
    rows.iter().map(|(k, v)| (k.as_str(), v)).collect()
}

fn atom_type_pair(
    types: Option<&ndarray::ArrayD<String>>,
    i: usize,
    j: usize,
) -> Result<(&str, &str), String> {
    let t = types.ok_or("1-4 exceptions: atoms block missing \"type\" column")?;
    Ok((t[i].as_str(), t[j].as_str()))
}

/// The sum of `w` over the `dihedral charmm` rows ending at each atom pair,
/// after LAMMPS's checks of the weights.
pub(crate) fn dihedral_weights(
    ff: &ForceField,
    frame: &Frame,
) -> Result<BTreeMap<(usize, usize), F>, String> {
    let mut wsum = BTreeMap::new();
    for style in ff.get_styles("dihedral") {
        if style.name() != "charmm" {
            continue;
        }
        let mut by_type: HashMap<String, F> = HashMap::new();
        let (_, rows) = gathered(ff, style).map_err(|e| e.to_string())?;
        for (name, p) in rows {
            let w = param_reads::type_num("charmm", &name, &p, "w").map_err(|e| e.to_string())?;
            // LAMMPS: "Incorrect weight arg for dihedral coefficients".
            if !(0.0..=1.0).contains(&w) {
                return Err(format!(
                    "dihedral charmm type '{name}': the 1-4 weight w = {w} is outside [0, 1]"
                ));
            }
            by_type.insert(name, w);
        }
        if !by_type.values().any(|&w| w > 0.0) {
            continue;
        }
        check_weightflag(ff)?;
        let Some(block) = frame.get(DIHEDRALS) else {
            continue;
        };
        let Some(labels) = block.get("type").and_then(|c| c.get::<String>()) else {
            continue;
        };
        for (r, (i, l)) in end_pairs(frame, DIHEDRALS, ATOMI, ATOML)
            .into_iter()
            .enumerate()
        {
            if let Some(&w) = by_type.get(labels[r].as_str())
                && w > 0.0
            {
                *wsum.entry((i, l)).or_insert(0.0) += w;
            }
        }
    }
    Ok(wsum)
}

/// LAMMPS's `DihedralCharmm::init_style` checks for a field with `w > 0`.
fn check_weightflag(ff: &ForceField) -> Result<(), String> {
    let sb = ff.required_special_bonds()?;
    if sb.lj[2] != 0.0 || sb.coul[2] != 0.0 {
        return Err(format!(
            "dihedral charmm has a 1-4 weight w > 0 beside special_bonds 1-4 weights \
             lj {} coul {}: the pair would be priced twice. LAMMPS refuses it too \
             (\"Must use 'special_bonds charmm' with dihedral style charmm for use with \
             CHARMM pair styles\"): set the 1-4 weights to 0, or w to 0",
            sb.lj[2], sb.coul[2]
        ));
    }
    let names: Vec<&str> = ff.get_styles("pair").iter().map(|s| s.name()).collect();
    if !names.contains(&"lj/charmm") {
        return Err(
            "dihedral charmm has a 1-4 weight w > 0, which prices the 1-4 pair with the \
             epsilon14/sigma14 of a lj/charmm pair style, and the force field has none \
             (LAMMPS: \"Dihedral charmm is incompatible with Pair style\")"
                .into(),
        );
    }
    if !names
        .iter()
        .any(|n| matches!(*n, "coul/charmm" | "coul/cut"))
    {
        return Err(
            "dihedral charmm has a 1-4 weight w > 0, which prices the 1-4 Coulomb pair with \
             the Coulomb pair style's constant, and the force field has none (coul/charmm)"
                .into(),
        );
    }
    Ok(())
}

/// The force field's van-der-Waals style and Coulomb prefactor
/// (`coulomb / dielectric`), as far as a 1-4 pair needs them.
///
/// Each pair style is classed by the [`SpecialClass`] its registered spec
/// declares (in the force field's own registry), not by its name: a Coulomb
/// style prices `q_i q_j / r` in its declared constant, every other one is
/// van der Waals.
///
/// # Errors
///
/// Two van-der-Waals styles or two Coulomb styles (a 1-4 pair stands in for
/// one of each); a buffered Coulomb (`delta ≠ 0`); a Coulomb style that
/// declares no `coulomb` / `dielectric`.
pub(crate) fn pair_styles(ff: &ForceField) -> Result<(Vdw<'_>, Option<F>), String> {
    let (mut vdw, mut coul) = (Vdw::None, None);
    let mut seen_vdw = None::<&str>;
    for style in ff.get_styles("pair") {
        let name = style.name();
        let class = ff
            .registry()
            .with(|r| r.style("pair", name).map(|(spec, _)| spec.special_class()))
            .unwrap_or(SpecialClass::Vdw);
        let (p, rows) = gathered(ff, style).map_err(|e| e.to_string())?;
        let num = |k: &str| param_reads::style_num(name, &p, k).map_err(|e| e.to_string());
        if class == SpecialClass::Coulomb {
            if p.get("delta")
                .and_then(|v| v.as_num())
                .is_some_and(|d| d != 0.0)
            {
                return Err(format!(
                    "pair style '{name}': a buffered Coulomb (delta ≠ 0) has no 1-4 \
                     exception form"
                ));
            }
            if coul.is_some() {
                return Err("1-4 exceptions: the force field has two Coulomb styles".into());
            }
            coul = Some(num("coulomb")? / num("dielectric")?);
            continue;
        }
        let found = match name {
            "lj/cut" if num("n")? == 12.0 && num("m")? == 6.0 => Vdw::LjCut(
                rows.into_iter().collect(),
                mixing_of(name, &p).map_err(|e| e.to_string())?,
            ),
            "lj/charmm" => {
                // Refused here, not only where it is read: a field whose
                // `one_four` is neither value has no 1-4 pricing at all.
                OneFour::of(&p)?;
                Vdw::Charmm(
                    rows.into_iter().collect(),
                    mixing_of(name, &p).map_err(|e| e.to_string())?,
                )
            }
            _ => Vdw::Other(name),
        };
        if let Some(prev) = seen_vdw {
            return Err(format!(
                "1-4 exceptions: pair styles '{prev}' and '{name}' are both van der Waals; \
                 a 1-4 pair stands in for one Lennard-Jones style"
            ));
        }
        seen_vdw = Some(name);
        vdw = found;
    }
    Ok((vdw, coul))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ff::compile::PotentialCompiler;
    use crate::ff::ir::{ONE_FOUR, ONE_FOUR_EPSILON14};
    use crate::ff::potential::fixtures::OneShot;
    use molrs::core::Block;
    use molrs::op::Idx;
    use ndarray::Array1;

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

    /// The A-B-B-A chain along x with its special pairs; (0, 3) is the one
    /// 1-4 pair.
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
            "pairs",
            rel(
                &["atomi", "atomj", "n_bonds"],
                &[
                    &[0, 1, 1],
                    &[1, 2, 1],
                    &[2, 3, 1],
                    &[0, 2, 2],
                    &[1, 3, 2],
                    &[0, 3, 3],
                ],
            ),
        );
        frame
    }

    /// `"epsilon14"` prices the 1-4 pair (0, 3) of two A atoms at A's
    /// ε₁₄/σ₁₄ and the 1-4 weight, with nothing written onto the frame:
    /// the energy differs from `"regular"` by exactly that pair's change.
    #[test]
    fn epsilon14_prices_the_one_four_pairs_at_their_own_parameters() {
        let frame = chain();
        let x: Vec<F> = frame.coords().unwrap().into_iter().collect();
        let energy = |ff: &ForceField| {
            PotentialCompiler::new(ff)
                .compile(&frame)
                .unwrap()
                .energy_at(&x)
        };
        let lj = |eps: F, sigma: F| {
            let r2: F = (0..3).map(|k| (x[9 + k] - x[k]).powi(2)).sum();
            let s6 = (sigma * sigma / r2).powi(3);
            4.0 * eps * (s6 * s6 - s6)
        };
        let shift = energy(&lj_charmm(Some(ONE_FOUR_EPSILON14))) - energy(&lj_charmm(None));
        let want = 0.5 * (lj(0.04, 2.0) - lj(0.1, 3.0));
        assert!((shift - want).abs() < 1e-12, "{shift} vs {want}");
    }
}
