//! The 1-4 exceptions of a force field on a frame: which non-bonded pairs
//! the [`PairExceptions`] kernel prices instead of the pair styles, and at
//! what parameters (the `dihedral charmm` `w`, the per-pair override cells,
//! `special_bonds`).

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ff::compile::gathered;
use crate::ff::compile::one_four::check_materialized;
use crate::ff::forcefield::ForceField;
use crate::ff::ir::{CombiningRule, OneFour, Params, SpecialClass};
use crate::ff::potential::pair::PairExceptions;
use crate::ff::potential::pair::lj_rows::{charmm_pair_params, lj_pair_params, mixing_of};
use crate::ff::potential::param_reads;
use crate::ff::potential::{bond_classes, end_pairs};
use molrs::core::Frame;
use molrs::core::keys::{ATOMI, ATOMJ, ATOML};
use molrs::core::schema::PAIR_OVERRIDE_COLUMNS;
use molrs::core::schema::block_names::{ATOMS, BONDS, DIHEDRALS, PAIR_OVERRIDES, PAIRS};
use molrs::op::F;

/// What the compiler routes through the exceptions kernel for one frame.
pub(crate) struct Exceptions {
    /// The kernel, when any pair is an exception.
    pub(crate) kernel: Option<PairExceptions>,
    /// Rows of the frame's `pairs` block that carry an override cell: the
    /// regular pair kernels must not price them.
    pub(crate) override_rows: Vec<usize>,
    /// Those rows' atom pairs, `(lo, hi)`.
    pub(crate) replaced: Vec<(usize, usize)>,
}

/// One `pairs` row's override cells, in [`PAIR_OVERRIDE_COLUMNS`] order.
type Cells = [Option<F>; 5];
const EPS: usize = 0;
const SIGMA: usize = 1;
const QQ: usize = 2;
const LJ_SCALE: usize = 3;
const COUL_SCALE: usize = 4;

/// The van-der-Waals style a 1-4 pair's Lennard-Jones parameters come from:
/// the one answer [`pair_styles`] gives the exceptions kernel and
/// [`ForceField::materialize_one_four`] alike.
pub(crate) enum Vdw<'f> {
    None,
    LjCut(HashMap<String, Params>, CombiningRule),
    Charmm(HashMap<String, Params>, CombiningRule, OneFour),
    /// A van-der-Waals style with no LJ 12-6 form (`buck`, `morse`, Mie
    /// `lj/cut`, …).
    Other(&'f str),
}

/// How the compiler prices the frame's 1-4 exceptions under `ff`.
///
/// `overlay` reads `pair_overrides` when the frame has no `pairs` block —
/// the neighbour-driven door. [`PotentialCompiler::compile`](crate::ff::compile::PotentialCompiler::compile)
/// passes `false`: a bonded field must not grow an exceptions kernel from
/// an overlay that is not an evaluation list.
///
/// # Errors
///
/// A `dihedral charmm` `w` outside `[0, 1]`; `w > 0` beside `special_bonds`
/// 1-4 weights other than 0, or without a `lj/charmm` pair style and a
/// Coulomb style (all three LAMMPS's own errors); an override column that is
/// not float, holds a non-finite value, or names one pair twice; and, when
/// any exception exists, a pair style the exception cannot stand in for.
pub(crate) fn plan(ff: &ForceField, frame: &Frame, overlay: bool) -> Result<Exceptions, String> {
    let wsum = dihedral_weights(ff, frame)?;
    let source = exception_source(frame, overlay);
    let (cells, override_rows) = override_cells(source)?;
    let rows_ij = pair_ends(frame, source);
    check_one_four(ff, frame, &wsum, &cells, &rows_ij)?;
    let block = source.map(|(name, _)| name).unwrap_or(PAIRS);

    let mut entries: BTreeMap<(usize, usize), (F, Option<usize>)> = BTreeMap::new();
    for (&key, &w) in &wsum {
        entries.insert(key, (w, None));
    }
    let mut replaced = Vec::with_capacity(override_rows.len());
    for &r in &override_rows {
        let key = rows_ij[r];
        let e = entries.entry(key).or_insert((0.0, None));
        if e.1.is_some() {
            return Err(format!(
                "{block}: atoms {} and {} carry per-pair overrides on two rows; a pair \
                 has one set of exception parameters",
                key.0, key.1
            ));
        }
        e.1 = Some(r);
        replaced.push(key);
    }
    if entries.is_empty() {
        return Ok(Exceptions {
            kernel: None,
            override_rows,
            replaced,
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
    let is_14 = frame
        .get(PAIRS)
        .and_then(|b| b.get("is_14"))
        .and_then(|c| c.get::<bool>());
    let classes = bond_distances(frame)?;
    let sb = ff.required_special_bonds()?;
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
    for (&(i, j), &(w, row)) in &entries {
        // An overlay row on a 1-2 or 1-3 that `special_bonds` excludes is the
        // row the list builder refuses. Pricing it here would be the other answer.
        if block == PAIR_OVERRIDES && row.is_some() {
            if let Some(class) = classes
                .as_ref()
                .and_then(|c| c.get(&(i.min(j), i.max(j))).copied())
            {
                let slot = match class {
                    1 => Some(0usize),
                    2 => Some(1usize),
                    _ => None,
                };
                if let Some(k) = slot {
                    if sb.lj[k] == 0.0 && sb.coul[k] == 0.0 {
                        return Err(format!(
                            "pair_overrides: atoms {i} and {j} carry a per-pair override, and \
                             the force field's special_bonds exclude that pair (1-2 or 1-3), \
                             so the override would be dropped"
                        ));
                    }
                }
            }
        }
        let cell = row.map(|r| cells[r]).unwrap_or([None; 5]);
        // The class weight is needed only for a pair the dihedral does not
        // price, and only where an override leaves a scale null.
        let class = || match &classes {
            Some(c) => c.get(&(i.min(j), i.max(j))).copied().unwrap_or(0),
            None => {
                let flagged = row.is_some_and(|r| is_14.is_some_and(|f| f[r]));
                if flagged { 3 } else { 0 }
            }
        };
        let weight = |table: [f64; 3]| match class() {
            d @ 1..=3 => table[d - 1],
            _ => 1.0,
        };

        // Van der Waals: the cells, else the dihedral's 1-4 parameters, else
        // the style's own. A field with no Lennard-Jones style prices none of
        // the LJ cells: they belong to a style it does not have.
        let base = |mixing: CombiningRule, charmm: bool| -> Result<(F, F), String> {
            let (a, b) = atom_type_pair(types, i, j)?;
            if charmm {
                let (regular, one_four) =
                    charmm_pair_params(&lj_rows, mixing, a, b).map_err(|e| e.to_string())?;
                Ok(if w > 0.0 { one_four } else { regular })
            } else {
                lj_pair_params("lj/cut", &lj_rows, mixing, a, b).map_err(|e| e.to_string())
            }
        };
        let (eps, sigma, w_lj) = match &vdw {
            Vdw::Charmm(_, mixing, _) | Vdw::LjCut(_, mixing) => {
                let (e, s) = match (cell[EPS], cell[SIGMA]) {
                    (Some(e), Some(s)) => (e, s),
                    (e, s) => {
                        let (be, bs) = base(*mixing, matches!(vdw, Vdw::Charmm(..)))?;
                        (e.unwrap_or(be), s.unwrap_or(bs))
                    }
                };
                let w_lj =
                    cell[LJ_SCALE].unwrap_or_else(|| if w > 0.0 { w } else { weight(sb.lj) });
                (e, s, w_lj)
            }
            _ => (0.0, 0.0, 0.0),
        };

        // Coulomb, likewise only under a Coulomb style.
        let (k, w_c, qq_ij) = match coul {
            None => (0.0, 0.0, 0.0),
            Some(k) => {
                let w_c =
                    cell[COUL_SCALE].unwrap_or_else(|| if w > 0.0 { w } else { weight(sb.coul) });
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
        override_rows,
        replaced,
    })
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

/// A `lj/charmm` field with `one_four = "epsilon14"` must have its 1-4 pairs
/// priced by an override (`epsilon` and `sigma` cells) or a `w` dihedral:
/// [`check_materialized`]. The 1-4 pairs are the `pairs` rows flagged `is_14`,
/// or, for a frame without a `pairs` block (the neighbour-driven door), the
/// bond-graph 1-4 pairs.
fn check_one_four(
    ff: &ForceField,
    frame: &Frame,
    wsum: &BTreeMap<(usize, usize), F>,
    cells: &[Cells],
    rows_ij: &[(usize, usize)],
) -> Result<(), String> {
    let mut overridden: HashSet<(usize, usize)> = HashSet::new();
    for (r, c) in cells.iter().enumerate() {
        if c[EPS].is_some() && c[SIGMA].is_some() {
            overridden.insert(rows_ij[r]);
        }
    }
    let pairs_14: Vec<(usize, usize)> = match frame
        .get(PAIRS)
        .and_then(|b| b.get("is_14"))
        .and_then(|c| c.get::<bool>())
    {
        Some(flags) => rows_ij
            .iter()
            .zip(flags.iter())
            .filter(|(_, f)| **f)
            .map(|(k, _)| *k)
            .collect(),
        None => {
            let mut at_three: Vec<(usize, usize)> = bond_distances(frame)?
                .unwrap_or_default()
                .into_iter()
                .filter(|&(_, d)| d == 3)
                .map(|(pair, _)| pair)
                .collect();
            at_three.sort_unstable();
            at_three
        }
    };
    let covered = |i: usize, j: usize| {
        overridden.contains(&(i, j)) || wsum.get(&(i, j)).is_some_and(|&w| w > 0.0)
    };
    check_materialized(ff, frame, &pairs_14, &covered)
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

/// `pairs` when present, otherwise `pair_overrides` when `overlay`.
fn exception_source(frame: &Frame, overlay: bool) -> Option<(&'static str, &molrs::core::Block)> {
    if let Some(block) = frame.get(PAIRS) {
        return Some((PAIRS, block));
    }
    if overlay {
        return frame
            .get(PAIR_OVERRIDES)
            .map(|block| (PAIR_OVERRIDES, block));
    }
    None
}

/// Each row's override cells of the block [`exception_source`] chose, and
/// the rows that carry any.
fn override_cells(
    source: Option<(&str, &molrs::core::Block)>,
) -> Result<(Vec<Cells>, Vec<usize>), String> {
    let Some((name, block)) = source else {
        return Ok((Vec::new(), Vec::new()));
    };
    let n = block.n_rows().unwrap_or(0);
    let mut cells = vec![[None; 5]; n];
    for (c, &key) in PAIR_OVERRIDE_COLUMNS.iter().enumerate() {
        let Some(col) = block.get(key) else {
            continue;
        };
        let values = col
            .get::<f64>()
            .ok_or_else(|| format!("{name}: the per-pair override column '{key}' must be float"))?;
        let valid = block.validity(key);
        for r in 0..n {
            if valid.is_some_and(|m| !m[r]) {
                continue;
            }
            let v = values[r];
            if !v.is_finite() {
                return Err(format!(
                    "{name} row {r}: '{key}' = {v}; a null cell is a validity mask, not a \
                     non-finite value"
                ));
            }
            cells[r][c] = Some(v);
        }
    }
    let rows = (0..n)
        .filter(|&r| cells[r].iter().any(Option::is_some))
        .collect();
    Ok((cells, rows))
}

/// `(lo, hi)` of each row of the block [`exception_source`] chose.
fn pair_ends(frame: &Frame, source: Option<(&str, &molrs::core::Block)>) -> Vec<(usize, usize)> {
    match source {
        Some((name, _)) => end_pairs(frame, name, ATOMI, ATOMJ),
        None => Vec::new(),
    }
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
            "lj/charmm" => Vdw::Charmm(
                rows.into_iter().collect(),
                mixing_of(name, &p).map_err(|e| e.to_string())?,
                OneFour::of(&p)?,
            ),
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

/// Bond-graph distance of each atom pair, up to 3.
type BondDistances = HashMap<(usize, usize), usize>;

/// Bond-graph distances up to 3 ([`bond_classes`]) when the frame carries
/// `bonds`; `None` without them, where a `pairs` row's `is_14` flag is all
/// the classification there is.
fn bond_distances(frame: &Frame) -> Result<Option<BondDistances>, String> {
    if frame.get(BONDS).is_none() {
        return Ok(None);
    }
    let n_atoms = frame.get(ATOMS).and_then(|b| b.n_rows()).unwrap_or(0);
    bond_classes(frame, n_atoms).map(Some)
}
